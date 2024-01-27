
import dataclasses
import torch
from torch.utils.data import Dataset
from labml import experiment,  monit, logger
from labml.logger import Text
from models.modelconfigs import ModelConfigs
from models.autoregressive import AutoregressiveModel

class TinyShakespeareDataset(Dataset):
    """
    ### Tiny Shakespeare Dataset
    """

    def __init__(self, seq_len: int):
        # Location of the text file
        path = '../data/tiny_shakespeare.txt'
        #path = '../data/tiny_data.txt'
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


class Inference:
    """
    ## Inference
    """

    def __init__(self, configs: ModelConfigs):
        # Get the device
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        # Initialize the dataset
        self.dataset = TinyShakespeareDataset(configs.seq_len)
        # Number of different characters
        n_chars = len(self.dataset.stoi)
        
        #self.model = torch.load('../build/model.pth')
        self.model = AutoregressiveModel(configs)
        self.model.load_state_dict(torch.load('../build/model.pth'))
        #checkpoint = torch.load("../build/checkpoint.pth")
       # self.model = checkpoint['model']
        #self.model = AutoregressiveModel()
        #self.model.load_state_dict(checkpoint['state_dict'])
        #for parameter in self.model.parameters():
        #    parameter.requires_grad = False

        self.model.eval()

    def sample(self):
        """
        ### Sampling function to generate samples periodically while training
        """


        # Starting prompt
        prompt = 'it is'
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
            print("\n output is: ", (self.dataset.itos[output[-1].item()], Text.value))
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
    configs = ModelConfigs()
    dataset = TinyShakespeareDataset(configs.seq_len)
    # Number of different characters
    n_chars = len(dataset.stoi)
    configs.n_tokens= n_chars


    # Override configurations
    experiment.configs(configs, {
        # Batch size
        'batch_size': 2,
        # Sequence length of $32$. We use a short sequence length to train faster.
        # Otherwise it takes forever to train.
        #'seq_len': 32,

        # Train for 1024 epochs.
        #'epochs': 1024,

        # Transformer configurations (same as defaults)
        #'d_model': 128,
    })


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