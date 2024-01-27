import torch
from labml_helpers.module import Module
from torch import nn
from torch.utils.data import Dataset, DataLoader

from labml import experiment, lab, tracker, monit, logger
from labml.logger import Text
from labml.utils.download import download_file
from labml_nn.experiments.nlp_autoregression import transpose_batch
from labml_nn.optimizers.noam import Noam
from models.autoregressive import AutoregressiveModel
from models.modelconfigs import ModelConfigs


class TinyShakespeareDataset(Dataset):
    """
    ### Tiny Shakespeare Dataset
    """

    def __init__(self, seq_len: int):
        # Location of the text file
        path = '../data/tiny_shakespeare.txt'
    #    path = '../data/tiny_data.txt'
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

class Trainer:
    """
    ## Trainer
    """

    def __init__(self, configs: ModelConfigs, dataset: TinyShakespeareDataset):
        # Get the device
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        # Initialize the dataset
        self.dataset = dataset
        # Initialize the dataloader
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=configs.batch_size,
                                     collate_fn=transpose_batch,
                                     shuffle=True)

        # Initialize the model with an
        # [embedding layer](../models.html#EmbeddingsWithPositionalEncoding)
        # (with fixed positional encoding)
        # [transformer encoder](../models.html#Encoder) and
        # a linear layer to generate logits.
        self.model = AutoregressiveModel(configs)
        self.model.load_state_dict(torch.load('../build/model.pth'))
        # Move the model to the current device
        self.model.to(self.device)

        # Initialize [Noam optimizer](../../optimizers/noam.html)
        self.optimizer = Noam(self.model.parameters(), lr=1.0, warmup=2_000, d_model=configs.d_model)

        # Cross-entropy loss
        self.loss_func = nn.CrossEntropyLoss()
        # Number of training epochs;
        # *note that our dataset definition repeats the data `seq_len` times in a single epoch*
        self.epochs = configs.epochs
        # Gradient clipping norm
        self.grad_norm_clip = configs.grad_norm_clip

        # Set tracker configurations
        tracker.set_scalar("loss.*", True)

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

    def train(self):
        """
        ### Train the model
        """

        # Loop for the given number of epochs
        for epoch in monit.loop(self.epochs):
            # Iterate over the minibatches
            for i, batch in monit.enum('Train', self.dataloader):
                print(' epoch = %d, sequence= %d' %(epoch, i))
                # Move data to the device
                data, target = batch[0].to(self.device), batch[1].to(self.device)

                # Set tracker step, as the number of characters trained on
                tracker.add_global_step(data.shape[0] * data.shape[1])

                # Set model state to training
                self.model.train()
                # Evaluate the model
                output = self.model(data)

                # Calculate loss
                loss = self.loss_func(output.view(-1, output.shape[-1]), target.view(-1))
                # Log the loss
                tracker.add("loss.train", loss)

                # Calculate gradients
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
                # Take optimizer step
                self.optimizer.step()
                # Log the model parameters and gradients
                if (i + 1) % 100 == 0:
                    tracker.add('model', self.model)
                # Clear the gradients
                self.optimizer.zero_grad()

                # Generate a sample
                if (i + 1) % 100 == 0:
                    torch.save(self.model.state_dict(), '../build/model.pth')
                    self.model.eval()
                    with torch.no_grad():
                        self.sample()

                # Save the tracked metrics
                if (i + 1) % 10 == 0:
                    tracker.save()

            # Save the model
            experiment.save_checkpoint()
       # checkpoint = {'model': AutoregressiveModel(), 'state_dict': self.model.state_dict(), 'optimizer' : self.optimizer.state_dict()}
        #torch.save(checkpoint, '../build/checkpoint.pth')
        #torch.save(self.model, '../build/model.pth')

        torch.save(self.model.state_dict(), '../build/model.pth')
        #torch.save(self.model, '../build/model.pth')

        #experiment.add_artifact(file_or_directory='../build/model.pth')        

def main():
    # Create experiment
    experiment.create(name="glu_variants")

    # Create configs
    configs = ModelConfigs()

    dataset = TinyShakespeareDataset(configs.seq_len)
   
     # Number of different characters
    n_chars = len(dataset.stoi)
    configs.n_tokens= n_chars
   
    # Load configurations

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
    trainer = Trainer(configs, dataset)
    # Set models for training and loading
    experiment.add_pytorch_models({'model': trainer.model})

    # Start the experiment
    with experiment.start():
        # Train the model
        trainer.train()

if __name__ == '__main__':
    main()