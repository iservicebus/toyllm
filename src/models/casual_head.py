# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.
# Copyright (c) 2024, Jingnan Zhou.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0


import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import torch
import torch.nn as nn
from torch.nn import  CrossEntropyLoss

from typing import Optional, Tuple

from core.model_config import ModelConfig

from core.transformer import Transformer

class CasualHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> dict:
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {
            "loss": loss,
            "logits": lm_logits
        }

