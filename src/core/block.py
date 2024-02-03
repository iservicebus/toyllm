# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.
# Copyright (c) 2024, Jingnan Zhou.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0



import torch
import torch.nn as nn
from torch.nn import  CrossEntropyLoss

from typing import Optional, Tuple, Union
from transformers.activations import ACT2FN

from .model_config import ModelConfig
from .attention import ClassicAttention
from .mlp import ClassicMLP


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class ClassicBlock(nn.Module):

    def __init__(self, config: ModelConfig, is_cross_attention: bool = False):
        super().__init__()
        hidden_size = config.n_embd
        self.is_cross_attention = is_cross_attention

        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.layernorm_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = ClassicAttention(config, is_cross_attention)
        self.layernorm_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if is_cross_attention:
            self.crossattention = ClassicAttention(config, is_cross_attention=True)
            self.layernorm_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = ClassicMLP(config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        ##  1: layer norm
        hidden_states = self.layernorm_1(hidden_states)
        ## 2: self attention
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        attn_output = attn_outputs[0]  # outputs(output_attn, att_weight)
        att_weight = attn_outputs[1]

        # 3: residual connection
        hidden_states = attn_output + residual
    
        # 4: cross attention 
        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.layernorm_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights
        # 5: layer norm again
        residual = hidden_states
        hidden_states = self.layernorm_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)

        # 6: residual connection again
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states, att_weight  # hidden_states, att_weight
