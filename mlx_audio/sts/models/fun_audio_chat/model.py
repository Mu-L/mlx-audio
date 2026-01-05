# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""Main FunAudioChat model for speech-to-speech and speech-to-text."""

import glob
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_map


from .audio_encoder import FunAudioChatAudioEncoder
from .config import FunAudioChatConfig
from .discrete_encoder import FunAudioChatDiscreteEncoder
from .language_model import LanguageModel
from .speech_decoder import FunAudioChatDecoder



@dataclass
class FunAudioChatOutput:

    text: str = ""
    audio_tokens: Optional[mx.array] = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0


class FunAudioChatForConditionalGeneration(nn.Module):
    

    def __init__(self, config: FunAudioChatConfig):
        super().__init__()
        self.config = config
        self.audio_tower = FunAudioChatAudioEncoder(config.audio_config)
        self.discrete_audio_tower = FunAudioChatDiscreteEncoder(config.audio_config)
        self.language_model = LanguageModel(config.text_config)

        # Speech decoder 
        if config.audio_config.enable_audio_invert_tower:
            self.audio_invert_tower = FunAudioChatDecoder(config.audio_config)
        else:
            self.audio_invert_tower = None

        self.audio_token_index = config.audio_token_index

        self._tokenizer = None

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the input embeddings from the language model."""
        return self.language_model.embed_tokens

    def encode_audio(
        self,
        input_features: mx.array,
        attention_mask: Optional[mx.array] = None,
        target_length: Optional[int] = None,
        apply_projection: bool = True,
    ) -> mx.array:
        """Encode audio features using the continuous audio tower.

        Args:
            input_features: Mel-spectrogram features
            attention_mask: Optional attention mask
            target_length: Optional target output length (for alignment with discrete tokens)
            apply_projection: If True, apply continual_output_matching through discrete encoder

        Returns:
            Encoded audio features
        """
        # Get continuous features from audio tower
        continuous_features = self.audio_tower(
            input_features, attention_mask=attention_mask, target_length=target_length
        )

        if apply_projection:
            # Apply continual_output_matching through the discrete encoder
            # This is required for S2T mode to properly align features with LM embedding space
            processed_features, _ = self.discrete_audio_tower(
                input_ids=None,
                continuous_features=continuous_features,
                attention_mask=attention_mask,
            )
            return processed_features

        return continuous_features

    def encode_discrete_audio(
        self,
        input_ids: mx.array,
        continuous_features: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Encode discrete audio tokens.

        Args:
            input_ids: Discrete audio token IDs
            continuous_features: Optional continuous features to fuse
            attention_mask: Optional attention mask

        Returns:
            Tuple of (encoded features, updated attention mask)
        """
        return self.discrete_audio_tower(
            input_ids,
            continuous_features=continuous_features,
            attention_mask=attention_mask,
        )

    def _merge_audio_text_embeddings(
        self,
        input_ids: mx.array,
        audio_embeds: Optional[mx.array] = None,
        audio_positions: Optional[List[Tuple[int, int]]] = None,
    ) -> mx.array:
        """Merge audio embeddings into text embeddings at specified positions.

        Args:
            input_ids: Text token IDs
            audio_embeds: Audio embeddings to insert
            audio_positions: List of (start, end) positions for each audio segment

        Returns:
            Merged embeddings
        """
        text_embeds = self.get_input_embeddings()(input_ids)

        if audio_embeds is None or audio_positions is None:
            return text_embeds

        batch_size = text_embeds.shape[0]

        for b in range(batch_size):
            if b < len(audio_positions):
                for i, (start, end) in enumerate(audio_positions[b] if isinstance(audio_positions[b], list) else [audio_positions[b]]):
                    if i < audio_embeds.shape[0]:
                        audio_len = min(end - start, audio_embeds.shape[1])
                        text_embeds = mx.concatenate([
                            text_embeds[:, :start, :],
                            audio_embeds[i:i+1, :audio_len, :],
                            text_embeds[:, end:, :]
                        ], axis=1) if b == 0 else text_embeds
         

        return text_embeds

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        audio_embeds: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            input_ids: Text token IDs
            input_features: Mel-spectrogram features for audio encoding
            audio_embeds: Pre-computed audio embeddings
            inputs_embeds: Pre-computed input embeddings
            attention_mask: Attention mask
            cache: KV cache for generation

        Returns:
            Logits for next token prediction
        """

        if input_features is not None and audio_embeds is None:
            audio_embeds = self.encode_audio(input_features)
            mx.eval(audio_embeds)

        # Use provided embeddings or compute from input_ids        if inputs_embeds is None:
            if input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            else:
                raise ValueError("Either input_ids or inputs_embeds must be provided")


        logits = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache=cache,
        )

        return logits

    def decode_speech(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        """Decode hidden states to speech token logits.

        Args:
            hidden_states: Language model hidden states
            attention_mask: Attention mask
            cache: KV cache

        Returns:
            Speech token logits
        """
        if self.audio_invert_tower is None:
            raise ValueError("Audio invert tower is not enabled")

        return self.audio_invert_tower(hidden_states, attention_mask=attention_mask, cache=cache)

    def model_quant_predicate(self, p: str, m: nn.Module) -> bool:

        return p.startswith("language_model.")

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized = {}

        for key, value in weights.items():
            new_key = key

            # Map continuous_audio_tower to audio_tower (audio encoder)
            if key.startswith("continuous_audio_tower."):
                new_key = key.replace("continuous_audio_tower.", "audio_tower.")
                # Map ln_post to layer_norm
                new_key = new_key.replace(".ln_post.", ".layer_norm.")

            # Map audio_tower.* (discrete encoder weights) to discrete_audio_tower.*
            elif key.startswith("audio_tower."):
                # embed_tokens, output_matching, continual_output_matching all go to discrete_audio_tower
                new_key = key.replace("audio_tower.", "discrete_audio_tower.")

            # Map language_model weights (already has language_model prefix)
            elif key.startswith("language_model."):
                pass  # Keep as-is

            # Map audio_invert_tower weights
            elif key.startswith("audio_invert_tower."):
                pass  # Keep as-is

            sanitized[new_key] = value

        return sanitized

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: mx.Dtype = mx.bfloat16,
        verbose: bool = False,
        **kwargs,
    ) -> "Model":
        """Load model from pretrained weights.

        Args:
            model_path: HuggingFace model ID or local path
            dtype: Data type for weights
            verbose: Print weight loading diagnostics
            **kwargs: Additional arguments

        Returns:
            Loaded model
        """
        from transformers import AutoTokenizer, AutoProcessor

        # Download or locate model
        if Path(model_path).exists():
            model_path_resolved = Path(model_path)
        else:
            try:
                model_path_resolved = Path(
                    snapshot_download(
                        repo_id=model_path,
                        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken"],
                    )
                )
            except Exception as e:
                raise ValueError(f"Could not download model from {model_path}: {e}")

        # Load config
        config_path = model_path_resolved / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        config = FunAudioChatConfig.from_dict(config_dict)

        # Create model
        model = cls(config)

        # Load tokenizer
        try:
            model._tokenizer = AutoTokenizer.from_pretrained(
                str(model_path_resolved), trust_remote_code=True
            )
        except Exception as e:
            warnings.warn(f"Could not load tokenizer: {e}")
            model._tokenizer = None

        # Load processor
        try:
            model._processor = AutoProcessor.from_pretrained(
                str(model_path_resolved), trust_remote_code=True
            )
        except Exception as e:
            warnings.warn(f"Could not load processor: {e}")
            model._processor = None

        # Load weights
        weights = {}
        weight_files = glob.glob(str(model_path_resolved / "model*.safetensors"))
        if not weight_files:
            weight_files = glob.glob(str(model_path_resolved / "*.safetensors"))

        for file in weight_files:
            weights.update(mx.load(file))

        # Sanitize weights
        weights = cls.sanitize(weights)

        # Load weights with transposition for linear layers
        weights = _transpose_weights(model, weights)

        # Handle quantized weights
        quantization = config_dict.get("quantization", None)
        if quantization is not None:

            def class_predicate(p, m):
                if p in quantization:
                    return quantization[p]
                if not hasattr(m, "to_quantized"):
                    return False
                if hasattr(m, "weight") and m.weight.size % 64 != 0:
                    return False
                return f"{p}.scales" in weights

            nn.quantize(
                model,
                group_size=quantization["group_size"],
                bits=quantization["bits"],
                class_predicate=class_predicate,
            )

        # Debug: show weight loading info
        if verbose:
            mlx_params = set(dict(tree_flatten(model.parameters())).keys())
            loaded_keys = set(weights.keys())
            missing = mlx_params - loaded_keys
            extra = loaded_keys - mlx_params
            matched = mlx_params & loaded_keys

            print(f"[Weight Loading] Model params: {len(mlx_params)}")
            print(f"[Weight Loading] Loaded weights: {len(loaded_keys)}")
            print(f"[Weight Loading] Matched: {len(matched)}")
            print(f"[Weight Loading] Missing from weights: {len(missing)}")
            print(f"[Weight Loading] Extra in weights: {len(extra)}")

            if missing and len(missing) < 30:
                print(f"[Weight Loading] Missing keys sample: {sorted(missing)[:10]}")
            if extra and len(extra) < 30:
                print(f"[Weight Loading] Extra keys sample: {sorted(extra)[:10]}")

        model.load_weights(list(weights.items()), strict=False)

        # Cast to specified dtype
        if dtype != mx.float32:

            def cast_to_dtype(x):
                if isinstance(x, mx.array) and x.dtype in [mx.float32, mx.float16, mx.bfloat16]:
                    return x.astype(dtype)
                return x

            model.apply_to_modules(
                lambda k, m: m.update(tree_map(cast_to_dtype, m.parameters()))
            )

        mx.eval(model.parameters())
        return model

    def _preprocess_audio(self, audio, sample_rate: int = 16000) -> Tuple[mx.array, float]:
        """Preprocess audio to mel spectrogram using WhisperFeatureExtractor.

        Args:
            audio: Audio path (str), waveform (np.ndarray/mx.array)
            sample_rate: Target sample rate

        Returns:
            Tuple of (mel spectrogram, audio duration in seconds)
        """
        import numpy as np
        from mlx_audio.stt.utils import load_audio
        from transformers import WhisperFeatureExtractor

        # Load audio if path
        if isinstance(audio, str):
            audio = load_audio(audio, sr=sample_rate)

        # Convert to numpy if needed
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # If already 3D, assume it's mel spectrogram (can't determine duration)
        if audio.ndim == 3:
            return mx.array(audio), 0.0

        # Calculate audio duration
        audio_duration = len(audio) / sample_rate

        # Use WhisperFeatureExtractor for consistent preprocessing
        fe = WhisperFeatureExtractor.from_pretrained(
            "FunAudioLLM/Fun-Audio-Chat-8B",
            trust_remote_code=True
        )

        # Don't pad to 30 seconds - use actual audio length
        features = fe(
            audio,
            sampling_rate=sample_rate,
            return_tensors="np",
            padding="do_not_pad",
        )
        mel = features.input_features  # (1, 128, time)

        # Convert to MLX
        mel = mx.array(mel)

        # Truncate if too long (max_source_positions is 1500 for conv output)
        # Conv2 has stride=2, so input can be 2x max_source_positions = 3000
        max_input_frames = self.config.audio_config.max_source_positions * 2
        if mel.shape[2] > max_input_frames:
            mel = mel[:, :, :max_input_frames]
            # Adjust duration for truncation
            audio_duration = min(audio_duration, max_input_frames / 100.0)

        return mel, audio_duration

    def _build_prompt_with_processor(
        self,
        audio_len: int,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Tuple[mx.array, int]:
        """Build prompt using AutoProcessor's chat template.

        Args:
            audio_len: Number of audio embedding tokens
            prompt: Optional instruction/prompt text
            system_prompt: Optional system prompt override

        Returns:
            Tuple of (input_ids, audio_start_position)
        """
        # Constants from HuggingFace reference implementation
        DEFAULT_S2T_PROMPT = "You are asked to generate text tokens."
        AUDIO_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>"

        system = system_prompt or DEFAULT_S2T_PROMPT

        # Build conversation in HuggingFace format
        if prompt is None:
            conversation = [
                {"role": "system", "content": system},
                {"role": "user", "content": AUDIO_TEMPLATE},
            ]
        else:
            conversation = [
                {"role": "system", "content": system},
                {"role": "user", "content": AUDIO_TEMPLATE + "\n" + prompt},
            ]

        # Use processor if available, otherwise fall back to tokenizer
        if self._processor is not None:
            text = self._processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            # Tokenize - processor may have tokenizer attribute or be a tokenizer itself
            tokenizer = getattr(self._processor, 'tokenizer', self._processor)
            tokens = tokenizer.encode(text, add_special_tokens=False)
        else:
            # Fallback: manually build template
            text = f"<|im_start|>system\n{system}<|im_end|>\n"
            if prompt is None:
                text += f"<|im_start|>user\n{AUDIO_TEMPLATE}<|im_end|>\n<|im_start|>assistant\n"
            else:
                text += f"<|im_start|>user\n{AUDIO_TEMPLATE}\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            tokens = self._tokenizer.encode(text, add_special_tokens=False)

        # Find the <|AUDIO|> placeholder position and replace with audio tokens
        # The <|AUDIO|> token ID is self.audio_token_index
        audio_start = None
        new_tokens = []
        for i, tok in enumerate(tokens):
            if tok == self.audio_token_index:
                audio_start = len(new_tokens)
                # Replace single <|AUDIO|> token with audio_len placeholder tokens
                new_tokens.extend([self.audio_token_index] * audio_len)
            else:
                new_tokens.append(tok)

        # If no audio token found, insert after audio_bos
        if audio_start is None:
            # Find <|audio_bos|> and insert after it
            audio_bos_token = self._tokenizer.encode("<|audio_bos|>", add_special_tokens=False)
            if audio_bos_token:
                audio_bos_id = audio_bos_token[0]
                for i, tok in enumerate(new_tokens):
                    if tok == audio_bos_id:
                        audio_start = i + 1
                        # Insert audio placeholder tokens
                        new_tokens = new_tokens[:audio_start] + [self.audio_token_index] * audio_len + new_tokens[audio_start:]
                        break

        if audio_start is None:
            audio_start = 0  # Fallback

        return mx.array([new_tokens]), audio_start

    def generate(
        self,
        audio: Optional[Union[str, mx.array]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.2,
        repetition_context_size: int = 20,
        verbose: bool = False,
        **kwargs,
    ) -> FunAudioChatOutput:
        """Generate text from audio input (S2T mode).

        Args:
            audio: Audio path or waveform
            prompt: Optional text prompt/instruction
            system_prompt: Optional system prompt override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling (0 to disable)
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            repetition_context_size: Number of recent tokens to consider for repetition penalty
            verbose: Print tokens during generation
            **kwargs: Additional arguments

        Returns:
            FunAudioChatOutput with generated text
        """
        start_time = time.time()

        if self._tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call from_pretrained first.")

        # Preprocess audio
        if audio is not None:
            mel, _ = self._preprocess_audio(audio)
            # Encode audio with continuous audio tower
            audio_embeds = self.encode_audio(mel)
            mx.eval(audio_embeds)
            audio_len = audio_embeds.shape[1]
        else:
            audio_embeds = None
            audio_len = 0

        # Build prompt using AutoProcessor template
        input_ids, audio_start = self._build_prompt_with_processor(
            audio_len=audio_len,
            prompt=prompt,
            system_prompt=system_prompt,
        )

        # Get text embeddings
        text_embeds = self.get_input_embeddings()(input_ids)

        # Replace placeholder positions with audio embeddings
        if audio_embeds is not None and audio_len > 0:
            text_embeds = mx.concatenate([
                text_embeds[:, :audio_start, :],
                audio_embeds,
                text_embeds[:, audio_start + audio_len:, :]
            ], axis=1)

        mx.eval(text_embeds)
        prompt_tokens = text_embeds.shape[1]

        # Create sampler and logits processors from mlx_lm
        sampler = make_sampler(temperature, top_p, top_k)
        logits_processors = make_logits_processors(
            logit_bias=kwargs.get("logit_bias", None),
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )

        # Generate tokens using KV cache
        generated_tokens = []

        # Initialize KV cache for each layer
        num_layers = len(self.language_model.layers)
        cache = [KVCache() for _ in range(num_layers)]

        # Process prompt (prefill)
        logits = self.language_model(inputs_embeds=text_embeds, cache=cache)
        mx.eval(logits)

        # EOS token IDs - include both <|endoftext|> and <|im_end|>
        eos_ids = self.config.text_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        # Add <|im_end|> (151645) which is the chat format EOS
        im_end_id = self._tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if im_end_id:
            eos_ids = list(set(eos_ids + im_end_id))

        for i in range(max_tokens):
            # Get next token logits
            next_logits = logits[:, -1, :]

            # Apply logits processors (including repetition penalty)
            if logits_processors and generated_tokens:
                # Build token context for repetition penalty (must be int32)
                token_context = mx.array([generated_tokens[-repetition_context_size:]], dtype=mx.int32)
                for processor in logits_processors:
                    next_logits = processor(token_context, next_logits)

            # Sample next token
            next_token = sampler(next_logits)
            next_token_id = next_token.item()

            # Check for EOS
            if next_token_id in eos_ids:
                break

            generated_tokens.append(next_token_id)

            if verbose:
                print(self._tokenizer.decode([next_token_id]), end="", flush=True)

            # Clear memory periodically
            if i % 50 == 0:
                mx.clear_cache()

            # Prepare next input
            next_embed = self.get_input_embeddings()(next_token[None, :])

            # Generate next logits using cache
            logits = self.language_model(inputs_embeds=next_embed, cache=cache)
            mx.eval(logits)

        if verbose:
            print()

        end_time = time.time()
        total_time = end_time - start_time

        # Decode generated tokens
        text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return FunAudioChatOutput(
            text=text.strip(),
            prompt_tokens=prompt_tokens,
            generation_tokens=len(generated_tokens),
            total_tokens=prompt_tokens + len(generated_tokens),
            total_time=total_time,
            prompt_tps=prompt_tokens / total_time if total_time > 0 else 0,
            generation_tps=len(generated_tokens) / total_time if total_time > 0 else 0,
        )

    def stream_generate(
        self,
        audio: Optional[Union[str, mx.array]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.2,
        repetition_context_size: int = 20,
        **kwargs,
    ) -> Generator[str, None, FunAudioChatOutput]:
        """Stream generate text from audio input (S2T mode).

        Yields tokens as they are generated, then returns final output.

        Args:
            audio: Audio path or waveform
            prompt: Optional text prompt/instruction
            system_prompt: Optional system prompt override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling (0 to disable)
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            repetition_context_size: Number of recent tokens to consider for repetition penalty
            **kwargs: Additional arguments

        Yields:
            Generated token strings

        Returns:
            FunAudioChatOutput with generated text
        """
        start_time = time.time()

        if self._tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call from_pretrained first.")

        # Preprocess audio
        if audio is not None:
            mel, _ = self._preprocess_audio(audio)
            audio_embeds = self.encode_audio(mel)
            mx.eval(audio_embeds)
            audio_len = audio_embeds.shape[1]
        else:
            audio_embeds = None
            audio_len = 0

        # Build prompt using AutoProcessor template
        input_ids, audio_start = self._build_prompt_with_processor(
            audio_len=audio_len,
            prompt=prompt,
            system_prompt=system_prompt,
        )

        # Get text embeddings
        text_embeds = self.get_input_embeddings()(input_ids)

        # Replace placeholder positions with audio embeddings
        if audio_embeds is not None and audio_len > 0:
            text_embeds = mx.concatenate([
                text_embeds[:, :audio_start, :],
                audio_embeds,
                text_embeds[:, audio_start + audio_len:, :]
            ], axis=1)

        mx.eval(text_embeds)
        prompt_tokens = text_embeds.shape[1]

        # Create sampler and logits processors
        sampler = make_sampler(temperature, top_p, top_k)
        logits_processors = make_logits_processors(
            logit_bias=kwargs.get("logit_bias", None),
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )

        # Initialize KV cache
        num_layers = len(self.language_model.layers)
        cache = [KVCache() for _ in range(num_layers)]

        # Process prompt (prefill)
        logits = self.language_model(inputs_embeds=text_embeds, cache=cache)
        mx.eval(logits)

        # EOS token IDs - include both <|endoftext|> and <|im_end|>
        eos_ids = self.config.text_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        # Add <|im_end|> (151645) which is the chat format EOS
        im_end_id = self._tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if im_end_id:
            eos_ids = list(set(eos_ids + im_end_id))

        generated_tokens = []

        for i in range(max_tokens):
            next_logits = logits[:, -1, :]

            # Apply logits processors
            if logits_processors and generated_tokens:
                token_context = mx.array([generated_tokens[-repetition_context_size:]], dtype=mx.int32)
                for processor in logits_processors:
                    next_logits = processor(token_context, next_logits)

            # Sample next token
            next_token = sampler(next_logits)
            next_token_id = next_token.item()

            # Check for EOS
            if next_token_id in eos_ids:
                break

            generated_tokens.append(next_token_id)

            # Yield the token
            token_str = self._tokenizer.decode([next_token_id])
            yield token_str

            # Clear memory periodically
            if i % 50 == 0:
                mx.clear_cache()

            # Prepare next input
            next_embed = self.get_input_embeddings()(next_token[None, :])

            # Generate next logits using cache
            logits = self.language_model(inputs_embeds=next_embed, cache=cache)
            mx.eval(logits)

        end_time = time.time()
        total_time = end_time - start_time

        # Decode generated tokens
        text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return FunAudioChatOutput(
            text=text.strip(),
            prompt_tokens=prompt_tokens,
            generation_tokens=len(generated_tokens),
            total_tokens=prompt_tokens + len(generated_tokens),
            total_time=total_time,
            prompt_tps=prompt_tokens / total_time if total_time > 0 else 0,
            generation_tps=len(generated_tokens) / total_time if total_time > 0 else 0,
        )


def _transpose_weights(model: nn.Module, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Transpose weights from PyTorch to MLX format where needed.

    Note: Linear layer weights do NOT need transposition - both PyTorch and MLX
    use (out_features, in_features) format. Only Conv layers may need transposition.

    Args:
        model: Target model
        weights: Weights to transpose

    Returns:
        Transposed weights
    """
    mlx_params = dict(tree_flatten(model.parameters()))
    new_weights = {}

    for key, value in weights.items():
        if key not in mlx_params:
            new_weights[key] = value
            continue

        target_shape = mlx_params[key].shape

        # 2D weights (Linear layers) - NO transposition needed
        # Both PyTorch and MLX use (out_features, in_features) format
        if len(value.shape) == 2 and len(target_shape) == 2:
            # Only transpose if shapes are different AND reversed
            # This prevents incorrect transposition of square matrices
            if value.shape != target_shape and value.shape == target_shape[::-1]:
                value = mx.transpose(value)

        # 3D weights (Conv layers)
        elif len(value.shape) == 3 and len(target_shape) == 3:
            v, t = value.shape, target_shape
            # Conv1d: PyTorch (out, in, k) -> MLX (out, k, in) per MLX docs
            if v[0] == t[0] and v[1] == t[2] and v[2] == t[1]:
                # (out, in, k) -> (out, k, in)
                value = mx.transpose(value, (0, 2, 1))
            # Alternative: PyTorch (out, in, k) -> MLX (in, k, out)
            elif v[0] == t[2] and v[1] == t[0] and v[2] == t[1]:
                value = mx.transpose(value, (1, 2, 0))

        new_weights[key] = value

    return new_weights


# Alias for backward compatibility
Model = FunAudioChatForConditionalGeneration
