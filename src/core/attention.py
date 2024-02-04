# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.
# Copyright (c) 2024, Jingnan Zhou.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0



import torch
import torch.nn as nn

from typing import Optional, Tuple, Union
from transformers.activations import ACT2FN

from .model_config import ModelConfig


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



class ClassicAttention (nn.Module):

    """
    Basic functions of classic attention
        define query, value and query structures based on paper "Attention Is All You Need"
        include cross-attention
        include attention from the previous layer
        include attention and head masking
        add attention dropout and residual dropout
    """


    def __init__(self, config: ModelConfig, is_cross_attention: bool = False):
        super().__init__()
        self.config =config
        self.is_cross_attention = is_cross_attention

        max_positions = config.n_positions

        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )

        # attention structure 
        if self.is_cross_attention:

            self.q_attn = Conv1D(self.config.n_embd, self.config.n_embd)
            self.c_attn = Conv1D(2*self.config.n_embd, self.config.n_embd)
        else:
            self.c_attn = Conv1D(3*self.config.n_embd, self.config.n_embd)

        self.c_proj = Conv1D(self.config.n_embd, self.config.n_embd)    

        # attention regularization
        self.attn_dropout = nn.Dropout(self.config.attn_pdrop)
        self.resid_dropout = nn.Dropout(self.config.resid_pdrop)
        self.pruned_heads = set()


    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.config.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )


        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)



    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
 
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `is_cross_attention=True`."
                )
            q_state = self.q_attn(hidden_states)
            k_state, v_state = self.c_attn(encoder_hidden_states).split(self.config.n_embd, dim=2)
            attention_mask = encoder_attention_mask
        else:
            q_state, k_state, v_state = self.c_attn(hidden_states).split(self.config.n_embd, dim=2)

        #split attention heads

        head_dim = self.config.n_embd // self.config.n_head
        if head_dim * self.config.n_head != self.config.n_embd:
            raise ValueError(
                f"`n_embd` must be divisible by n_head (got `n_embd`: {self.config.n_embd} and `n_head`:"
                f" {self.config.n_head})."
            )

        # tensor structure (batch, head, seq_length, head_features)
        q_state = self._split_heads(q_state, self.config.n_head, head_dim)
        k_state = self._split_heads(k_state, self.config.n_head, head_dim)
        v_state = self._split_heads(v_state, self.config.n_head, head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            k_state = torch.cat((past_key, k_state), dim=-2)
            v_state = torch.cat((past_value, v_state), dim=-2)
        
        attn_output, attn_weights = self._attn(q_state, k_state, v_state, attention_mask, head_mask)

        # tensor structure (batch, seq_length, n_embd)
        attn_output = self._merge_heads(attn_output, self.config.n_head, head_dim)


        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights  


