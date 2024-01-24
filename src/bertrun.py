import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from labml import experiment, tracker
from torch.utils.data import DataLoader, RandomSampler

from labml_helpers.module import Module
from torch import nn
from labml_helpers.datasets.text import TextDataset,  SequentialUnBatchedDataset, TextFileDataset
from labml_helpers.train_valid import TrainValidConfigs,  BatchIndex
from labml_helpers.device import DeviceConfigs
from labml.configs import option
from labml_helpers.metrics.accuracy import Accuracy

from optimizers.configs import OptimizerConfigs
from models.modelconfigs import ModelConfigs
from models.bert import BERT

from labml_helpers.train_valid import TrainValidConfigs

from typing import List

import torch

class CrossEntropyLoss(Module):
    """
    ### Cross entropy loss
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.loss(outputs.view(-1, outputs.shape[-1]), targets.view(-1))


def transpose_batch(batch):
    """
    ### Transpose batch

    `DataLoader` collects the batches on the first dimension.
    We need to transpose it to be sequence first.
    """

    transposed_data = list(zip(*batch))
    # Stack the batch along the second dimension `dim=1`
    src = torch.stack(transposed_data[0], dim=1)
    tgt = torch.stack(transposed_data[1], dim=1)

    return src, tgt


class MaskedLan:
    """
    ## Masked Language

    This class implements the masking procedure for a given batch of token sequences.
    """

    def __init__(self, *,
                 padding_token: int, mask_token: int, no_mask_tokens: List[int], n_tokens: int,
                 masking_prob: float = 0.15, randomize_prob: float = 0.1, no_change_prob: float = 0.1,
                 ):
        """
        * `padding_token` is the padding token `[PAD]`.
          We will use this to mark the labels that shouldn't be used for loss calculation.
        * `mask_token` is the masking token `[MASK]`.
        * `no_mask_tokens` is a list of tokens that should not be masked.
        This is useful if we are training the MLM with another task like classification at the same time,
        and we have tokens such as `[CLS]` that shouldn't be masked.
        * `n_tokens` total number of tokens (used for generating random tokens)
        * `masking_prob` is the masking probability
        * `randomize_prob` is the probability of replacing with a random token
        * `no_change_prob` is the probability of replacing with original token
        """
        self.n_tokens = n_tokens
        self.no_change_prob = no_change_prob
        self.randomize_prob = randomize_prob
        self.masking_prob = masking_prob
        self.no_mask_tokens = no_mask_tokens + [padding_token, mask_token]
        self.padding_token = padding_token
        self.mask_token = mask_token

    def __call__(self, x: torch.Tensor):
        """
        * `x` is the batch of input token sequences.
         It's a tensor of type `long` with shape `[seq_len, batch_size]`.
        """

        # Mask `masking_prob` of tokens
        full_mask = torch.rand(x.shape, device=x.device) < self.masking_prob
        # Unmask `no_mask_tokens`
        for t in self.no_mask_tokens:
            full_mask &= x != t

        # A mask for tokens to be replaced with original tokens
        unchanged = full_mask & (torch.rand(x.shape, device=x.device) < self.no_change_prob)
        # A mask for tokens to be replaced with a random token
        random_token_mask = full_mask & (torch.rand(x.shape, device=x.device) < self.randomize_prob)
        # Indexes of tokens to be replaced with random tokens
        random_token_idx = torch.nonzero(random_token_mask, as_tuple=True)
        # Random tokens for each of the locations
        random_tokens = torch.randint(0, self.n_tokens, (len(random_token_idx[0]),), device=x.device)
        # The final set of tokens that are going to be replaced by `[MASK]`
        mask = full_mask & ~random_token_mask & ~unchanged

        # Make a clone of the input for the labels
        y = x.clone()

        # Replace with `[MASK]` tokens;
        # note that this doesn't include the tokens that will have the original token unchanged and
        # those that get replace with a random token.
        x.masked_fill_(mask, self.mask_token)
        # Assign random tokens
        x[random_token_idx] = random_tokens

        # Assign token `[PAD]` to all the other locations in the labels.
        # The labels equal to `[PAD]` will not be used in the loss.
        y.masked_fill_(~full_mask, self.padding_token)

        # Return the masked input and the labels
        return x, y


class BERTRun( TrainValidConfigs):

    # Number of features in in the hidden layer
    d_ff: int = 2048

    """
    ## model configuration
    """
    grad_norm_clip: float = 0.5

    # Predefined GLU variants
    glu_variant: str = 'none'

    # Dropout probability
    dropout: float = 0.1

    # Data loaders shuffle with replacement
    dataloader_shuffle_with_replacement: bool = False

    # Training data loader
    train_loader: DataLoader = 'shuffled_train_loader'
    # Validation data loader
    valid_loader: DataLoader = 'shuffled_valid_loader'
    # Training device
    device: torch.device = DeviceConfigs()

    # Optimizer
    optimizer: torch.optim.Adam
    # Loss function
    loss_func = CrossEntropyLoss()
    # Accuracy function
    accuracy = Accuracy()
    mconfigs: ModelConfigs
    # MLM model
    model: BERT

  # [Masked Language Model (MLM) class](index.html) to generate the mask
    mlm: MaskedLan

   # Text dataset
    text: TextDataset

    # Tokens that shouldn't be masked
    no_mask_tokens: List[int] = []
    # Probability of masking a token
    masking_prob: float = 0.15
    # Probability of replacing the mask with a random token
    randomize_prob: float = 0.1
    # Probability of replacing the mask with original token
    no_change_prob: float = 0.1
    # [Masked Language Model (MLM) class](index.html) to generate the mask

    # `[MASK]` token
    mask_token: int
    # `[PADDING]` token
    padding_token: int

    # Prompt to sample
    prompt: str = [
        "We are accounted poor citizens, the patricians good.",
        "What authority surfeits on would relieve us: if they",
        "would yield us but the superfluity, while it were",
        "wholesome, we might guess they relieved us humanely;",
        "but they think we are too dear: the leanness that",
        "afflicts us, the object of our misery, is as an",
        "inventory to particularise their abundance; our",
        "sufferance is a gain to them Let us revenge this with",
        "our pikes, ere we become rakes: for the gods know I",
        "speak this in hunger for bread, not in thirst for revenge.",
    ]


    def __init__(self, configs: ModelConfigs, text: TextDataset):
        super().__init__()
        self.mconfigs = configs
        self.text = text


    def init(self):

        """
        ### Initialization
        """

        # `[MASK]` token
        self.mask_token = self.text.n_tokens - 1
        # `[PAD]` token
        self.padding_token = self.text.n_tokens - 2

        # [Masked Language Model (MLM) class](index.html) to generate the mask
        self.mlm = MaskedLan(padding_token=self.padding_token,
                       mask_token=self.mask_token,
                       no_mask_tokens=self.no_mask_tokens,
                       n_tokens=self.text.n_tokens,
                       masking_prob=self.masking_prob,
                       randomize_prob=self.randomize_prob,
                       no_change_prob=self.no_change_prob)

        # Accuracy metric (ignore the labels equal to `[PAD]`)
        self.accuracy = Accuracy(ignore_index=self.padding_token)
        # Cross entropy loss (ignore the labels equal to `[PAD]`)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.padding_token)
        #
        self.model = BERT(self.mconfigs)

        """
        ### Initialization
        """
        # Set tracker configurations
        tracker.set_scalar("accuracy.*", True)
        tracker.set_scalar("loss.*", True)
        tracker.set_text("sampled", False)

        # Add accuracy as a state module.
        # The name is probably confusing, since it's meant to store
        # states between training and validation for RNNs.
        # This will keep the accuracy metric stats separate for training and validation.
        self.state_modules = [self.accuracy]



    def step(self, batch: any, batch_idx: BatchIndex):

        """
        ### Training or validation step
        """
        # Move the input to the device
        data = batch[0].to(self.device)
        # Update global step (number of tokens processed) when in training mode
        if self.mode.is_train:
            tracker.add_global_step(data.shape[0] * data.shape[1])

        # Get the masked input and labels
        with torch.no_grad():
            data, labels = self.mlm(data)

        # Whether to capture model outputs
        with self.mode.update(is_log_activations=batch_idx.is_last):
            # Get model outputs.
            # It's returning a tuple for states when using RNNs.
            # This is not implemented yet.
            output, *_ = self.model(data)
        # Calculate and log the loss
        loss = self.loss_func(output.view(-1, output.shape[-1]), labels.view(-1))
        tracker.add("loss.", loss)

        # Calculate and log accuracy
        self.accuracy(output, labels)
        self.accuracy.track()

        # Train the model
        if self.mode.is_train:
            # Calculate gradients
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
            # Take optimizer step
            self.optimizer.step()
            # Log the model parameters and gradients on last batch of every epoch
            if batch_idx.is_last:
                tracker.add('model', self.model)
            # Clear the gradients
            self.optimizer.zero_grad()

        # Save the tracked metrics
        tracker.save()


@option(BERTRun.optimizer)
def _optimizer(c: BERTRun):
    """
    ### Default [optimizer configurations](../optimizers/configs.html)
    """

    optimizer = OptimizerConfigs()
    optimizer.parameters = c.model.parameters()
    optimizer.optimizer = 'Adam'
    optimizer.d_model = c.mconfigs.d_model

    return optimizer

@option(BERTRun.train_loader)
def shuffled_train_loader(c: BERTRun):
    """
    ### Shuffled training data loader
    """
    dataset = SequentialUnBatchedDataset(text=c.text.train,
                                         dataset=c.text,
                                         seq_len=c.mconfigs.seq_len)
    sampler = RandomSampler(dataset, replacement=c.dataloader_shuffle_with_replacement)

    return DataLoader(dataset,
                      batch_size=c.mconfigs.batch_size,
                      collate_fn=transpose_batch,
                      sampler=sampler)


@option(BERTRun.valid_loader)
def shuffled_valid_loader(c: BERTRun):
    """
    ### Shuffled validation data loader
    """
    dataset = SequentialUnBatchedDataset(text=c.text.valid,
                                         dataset=c.text,
                                         seq_len=c.mconfigs.seq_len)
    sampler = RandomSampler(dataset, replacement=c.dataloader_shuffle_with_replacement)

    return DataLoader(dataset,
                      batch_size=c.mconfigs.batch_size,
                      collate_fn=transpose_batch,
                      sampler=sampler)



def basic_english():
    """
    ### Basic  english tokenizer

    We use character level tokenizer in this experiment.
    You can switch by setting,

    ```
    'tokenizer': 'basic_english',
    ```

    in the configurations dictionary when starting the experiment.
    """

    from torchtext.data import get_tokenizer
    return get_tokenizer('basic_english')


def character_tokenizer(x: str):
    """
    ### Character level tokenizer
    """
    return list(x)

def main():
    # Create experiment
    experiment.create(name="mlm")


    configs = ModelConfigs()



    text = TextFileDataset(
        os.path.join(os.path.dirname(__file__), '../data/tiny_shakespeare.txt'),
        character_tokenizer
    )

    # Override configurations
    experiment.configs(configs, {
        # Batch size
        'n_tokens': text.n_tokens,
    })


    # Create configs
    conf = BERTRun(configs, text)

    # Set models for saving and loading
#    experiment.add_pytorch_models({'model': conf.model})
    # Start the experiment
    with experiment.start():
        # Run training
        conf.run()


#
if __name__ == '__main__':
    main()