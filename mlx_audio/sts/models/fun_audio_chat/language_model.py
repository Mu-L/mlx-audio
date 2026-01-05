# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.qwen3 import Qwen3Model

try:
    from .config import Qwen3Config
except ImportError:
    from mlx_audio.sts.models.fun_audio_chat.config import Qwen3Config


class LanguageModel(nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
        **kwargs,
    ) -> mx.array:
        hidden_states = self.model(
            inputs=input_ids,
            input_embeddings=inputs_embeds,
            cache=cache,
        )

        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden_states)
        else:
            logits = self.lm_head(hidden_states)

        return logits

    @property
    def layers(self):
        return self.model.layers

    @property
    def embed_tokens(self):
        return self.model.embed_tokens
