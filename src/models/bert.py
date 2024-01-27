import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from labml_helpers.module import Module
from torch import nn
from transformer.modules import Encoder, MultiHeadAttention, TransformerLayer, EmbeddingsWithPositionalEncoding
from transformer.feed_forward import FeedForward

from .modelconfigs import ModelConfigs

from utils import subsequent_mask

class BERT(Module):
    """
    ## Bidirectional Encoder Representations from Transformers (BERT) model
    """

    def __init__(self, mconfigs: ModelConfigs):
        super().__init__()

        # Number of different characters

        # FFN with ReLU activation
        # $$FFN_{ReLU}(x)(x, W_1, W_2, b_1, b_2) = \text{ReLU}_1(x W_1 + b_1) W_2 + b_2$$
        ffn = FeedForward(mconfigs.d_model, mconfigs.d_ff, mconfigs.dropout, nn.ReLU())

        # Initialize [Multi-Head Attention module](../mha.html)
        mha = MultiHeadAttention(mconfigs.n_heads, mconfigs.d_model, mconfigs.dropout)

        # Initialize the [Transformer Block](../models.html#TransformerLayer)
        transformer_layer = TransformerLayer(d_model=mconfigs.d_model, self_attn=mha, src_attn=None,
                                             feed_forward=ffn, dropout_prob=mconfigs.dropout)

        self.src_embed = EmbeddingsWithPositionalEncoding(mconfigs.d_model, mconfigs.n_tokens)
        self.encoder = Encoder(transformer_layer, mconfigs.n_layers)
        self.generator = nn.Linear(mconfigs.d_model, mconfigs.n_tokens)

    def forward(self, x: torch.Tensor):
        # Get the token embeddings with positional encodings
        x = self.src_embed(x)
        # Transformer encoder
        x = self.encoder(x, None)
        # Logits for the output
        y = self.generator(x)

        # Return results
        # (second value is for state, since our trainer is used with RNNs also)
        return y, None

