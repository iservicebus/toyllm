from labml.configs import BaseConfigs
"""
# Model  Configuration
"""

class ModelConfigs(BaseConfigs):

    """
    ## model parameters
    """

    # Number of token in vocabulary
    n_tokens: int = 1

    # Batch size
    batch_size: int = 16
    # Length of the sequence, or context size
    seq_len: int = 512
    # Number of features in the embedding
    d_model: int = 64
#    d_model: int = 128
#    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 2
#    n_heads: int = 4
#    n_heads: int = 8

#    epochs: int = 5
    epochs: int = 1

    # Number of features in in the hidden layer
    d_ff: int = 2048

    grad_norm_clip: float = 0.5

    # Predefined GLU variants
    glu_variant: str = 'none'

    # Dropout probability
    dropout: float = 0.1

    def init(self):
        pass

    def __init__(self):
        super().__init__()


 
