# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.
# Copyright (c) 2024, Jingnan Zhou.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0



import torch
import torch.nn as nn
from torch.nn import  CrossEntropyLoss

from typing import Optional, Tuple

from core.model_config import ModelConfig

from core.transformer import Transformer


class BaseModel(nn.Module):
    def __init__(self, config: ModelConfig, lm_head: nn.Module, is_cross_attention: bool = False):
        super().__init__()
        self.transformer = Transformer(config, is_cross_attention)
        self.lm_head = lm_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> dict:
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = transformer_outputs["last_hidden_state"]


        lm_output = self.lm_head(hidden_states, labels)

        return {
            "loss": lm_output["loss"],
            "logits": lm_output["logits"],
        }
