

import dataclasses
import torch
from labml_helpers.module import Module
from torch import nn
from torch.utils.data import Dataset, DataLoader

from labml import experiment, lab, tracker, monit, logger
from labml.logger import Text
from labml.utils.download import download_file
from labml_nn.experiments.nlp_autoregression import transpose_batch
from labml_nn.optimizers.noam import Noam
from model.models import Encoder, MultiHeadAttention, TransformerLayer, EmbeddingsWithPositionalEncoding
from model.feed_forward import FeedForward
from utils import subsequent_mask



@dataclasses.dataclass
class Configs:
    """
    ### Configurations
    """
    d_model: int = 64
#    d_model: int = 128
#    d_model: int = 512

    seq_len: int = 128
    batch_size: int = 32

    n_layers: int = 6
    n_heads: int = 2
#    n_heads: int = 4
#    n_heads: int = 8

    dropout: float = 0.1
    d_ff: int = 2048
    glu_variant: str = 'GLU'
#    epochs: int = 5
    epochs: int = 1

    grad_norm_clip: float = 0.5



class TinyShakespeareDataset(Dataset):
    """
    ### Tiny Shakespeare Dataset
    """

    def __init__(self, seq_len: int):
        # Location of the text file
    #    path = '../data/tiny_shakespeare.txt'
        path = '../data/tiny_data.txt'
        # Read the downloaded file
        with open(str(path), 'r') as f:
            text = f.read()

        # Extract the characters
        chars = list(set(text))
        # Character to id (integer) map
        self.stoi = {c: i for i, c in enumerate(chars)}
        # Id to character map
        self.itos = {i: c for i, c in enumerate(chars)}
        # Length of a training sample
        self.seq_len = seq_len
        # Data in the form of a tensor of ids
        self.data = self.text_to_i(text)

    def text_to_i(self, text: str):
        """
        Transform the text into a tensor of ids
        """
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self):
        """
        Number of samples in the dataset.

        *This will read the dataset `seq_len` times in a single epoch.*
        """
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        """
        Return a sample
        """
        return self.data[idx:idx + self.seq_len], self.data[idx + 1:idx + self.seq_len + 1]


class AutoregressiveModel(Module):
    """
    ## Auto regressive model
    """

    def __init__(self):
        super().__init__()

        configs = Configs()
        dataset = TinyShakespeareDataset(configs.seq_len)
        # Number of different characters
        n_chars = len(dataset.stoi)

        # FFN with ReLU activation
        # $$FFN_{ReLU}(x)(x, W_1, W_2, b_1, b_2) = \text{ReLU}_1(x W_1 + b_1) W_2 + b_2$$
        ffn = FeedForward(configs.d_model, configs.d_ff, configs.dropout, nn.ReLU())

        # Initialize [Multi-Head Attention module](../mha.html)
        mha = MultiHeadAttention(configs.n_heads, configs.d_model, configs.dropout)

        # Initialize the [Transformer Block](../models.html#TransformerLayer)
        transformer_layer = TransformerLayer(d_model=configs.d_model, self_attn=mha, src_attn=None,
                                             feed_forward=ffn, dropout_prob=configs.dropout)

        self.src_embed = EmbeddingsWithPositionalEncoding(configs.d_model, n_chars)
        self.encoder = Encoder(transformer_layer, configs.n_layers)
        self.generator = nn.Linear(configs.d_model, n_chars)
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




class Inference:
    """
    ## Inference
    """

    def __init__(self, configs: Configs):
        # Get the device
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        # Initialize the dataset
        self.dataset = TinyShakespeareDataset(configs.seq_len)
        # Number of different characters
        n_chars = len(self.dataset.stoi)


        checkpoint = torch.load("../build/checkpoint.pth")
       # self.model = checkpoint['model']
        self.model = AutoregressiveModel()
        self.model.load_state_dict(checkpoint['state_dict'])
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.eval()

    def sample(self):
        """
        ### Sampling function to generate samples periodically while training
        """


        # Starting prompt
        prompt = 'It is'
        # Collect output for printing
        log = [(prompt, Text.subtle)]
        # Sample 25 tokens
        for i in monit.iterate('Sample', 25):
            # Tokenize the prompt
            data = self.dataset.text_to_i(prompt).unsqueeze(-1)
            data = data.to(self.device)
            # Get the model output
            output = self.model(data)
            # Get the model prediction (greedy)
            output = output.argmax(dim=-1).squeeze()
            # Add the prediction to prompt
            prompt += self.dataset.itos[output[-1].item()]
            # Add the prediction for logging
            log += [(self.dataset.itos[output[-1].item()], Text.value)]

        # Print the sampled output
        logger.log(log)


def main():
    # Create experiment
    experiment.create(name="glu_variants")
    # Create configs
    configs = Configs()
    # Load configurations
    experiment.configs(dataclasses.asdict(configs))

    # Create trainer
    inference = Inference(configs)
    # Set models for training and loading
    experiment.add_pytorch_models({'model': inference.model})

    # Start the experiment
    with experiment.start():
        # model inference
        with torch.no_grad():
            inference.sample()

if __name__ == '__main__':
    main()