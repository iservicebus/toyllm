import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from labml_helpers.module import Module
from torch import nn
from transformer.modules import Encoder, MultiHeadAttention, TransformerLayer, EmbeddingsWithPositionalEncoding
from transformer.feed_forward import FeedForward
from .modelconfigs import ModelConfigs

from utils import subsequent_mask


class AutoregressiveModel(Module):
    """
    ## Auto regressive model
    """

    def __init__(self, configs: ModelConfigs):
        super().__init__()

        # Number of different characters

        # FFN with ReLU activation
        # $$FFN_{ReLU}(x)(x, W_1, W_2, b_1, b_2) = \text{ReLU}_1(x W_1 + b_1) W_2 + b_2$$
        ffn = FeedForward(configs.d_model, configs.d_ff, configs.dropout, nn.ReLU())

        # Initialize [Multi-Head Attention module](../mha.html)
        mha = MultiHeadAttention(configs.n_heads, configs.d_model, configs.dropout)

        # Initialize the [Transformer Block](../models.html#TransformerLayer)
        transformer_layer = TransformerLayer(d_model=configs.d_model, self_attn=mha, src_attn=None,
                                             feed_forward=ffn, dropout_prob=configs.dropout)

        self.src_embed = EmbeddingsWithPositionalEncoding(configs.d_model, configs.n_tokens)
        self.encoder = Encoder(transformer_layer, configs.n_layers)
        self.generator = nn.Linear(configs.d_model, configs.n_tokens)
        # This will be initialized on the first call
        self.src_mask = None

    def forward(self, src: torch.Tensor):
        # Create subsequent mask, so that the transformer can only pay attention to past tokens.
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            self.src_mask = subsequent_mask(len(src)).to(src.device)
        # Embed the tokens (`src`) and run it through the the transformer
        res = self.encoder(self.src_embed(src), self.src_mask)
        # Generate logits of the next token
        return self.generator(res)
