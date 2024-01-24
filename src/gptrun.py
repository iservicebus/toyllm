import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from labml import experiment, monit, logger, tracker
from torch.utils.data import DataLoader, RandomSampler

from labml_helpers.module import Module
from torch import nn
from labml_helpers.datasets.text import TextDataset, SequentialDataLoader, SequentialUnBatchedDataset, TextFileDataset
from labml_helpers.train_valid import TrainValidConfigs, hook_model_outputs, BatchIndex
from labml_helpers.device import DeviceConfigs

from labml.configs import option
from labml.logger import Text

from labml_helpers.metrics.accuracy import Accuracy

from optimizers.configs import OptimizerConfigs
from models.modelconfigs import ModelConfigs
from models.gpt import GPT

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

def _init_weights(module):
    """
    ### Initialize weights

    Weights of linear layers and embedding layers are initialized
    to $\mathcal{N}(0, 0.02)$
    instead of the default Xavier initialzation.
    """

    if not isinstance(module, (nn.Linear, nn.Embedding)):
        return

    module.weight.data.normal_(mean=0.0, std=0.02)

    # Initialize biases to $0$
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()



class GPTRun( TrainValidConfigs):

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

   # Weight decay
    weight_decay: float = 0.1
    # Number of tokens for wamup
    warmup_steps: int = 128 * 128 * 20

    # Custom optimizer
    #optimizer = 'transformer_optimizer'
    optimizer: torch.optim.Adam
    # Loss function
    loss_func = CrossEntropyLoss()
    # Accuracy function
    accuracy = Accuracy()
    mconfigs: ModelConfigs
    # GPT model
    model: GPT

   # Text dataset
    text: TextDataset

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
        #
        self.model = GPT(self.mconfigs).to(self.device)

        # Apply custom weight initialization
        self.model.apply(_init_weights)

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

        # Set training/eval mode
        self.model.train(self.mode.is_train)

        # Move data to the device
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # Update global step (number of tokens processed) when in training mode
        if self.mode.is_train:
            tracker.add_global_step(data.shape[0] * data.shape[1])

        # Whether to capture model outputs
        with self.mode.update(is_log_activations=batch_idx.is_last and self.is_log_model_activations):
            # Get model outputs.
            # It's returning a tuple for states when using RNNs.
            # This is not implemented yet. 😜
            output, *_ = self.model(data)

        # Calculate and log loss
        loss = self.loss_func(output, target)
        tracker.add("loss.", loss)

        # Calculate and log accuracy
        self.accuracy(output, target)
        self.accuracy.track()

        #self.other_metrics(output, target)

        # Train the model
        if self.mode.is_train:
            # Calculate gradients
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
            # Take optimizer step
            self.optimizer.step()
            # Log the model parameters and gradients on last batch of every epoch
            if batch_idx.is_last and self.is_log_model_params_grads:
                tracker.add('model', self.model)
            # Clear the gradients
            self.optimizer.zero_grad()

        # Save the tracked metrics
        tracker.save()


@option(GPTRun.optimizer)
def _optimizer(c: GPTRun):
    """
    ### Default [optimizer configurations](../optimizers/configs.html)
    """

    optimizer = OptimizerConfigs()
    optimizer.parameters = c.model.parameters()
    optimizer.optimizer = 'Adam'
    optimizer.d_model = c.mconfigs.d_model

    return optimizer


def transformer_optimizer(c: GPTRun):

    """
    ### Create custom optimizer with weight decay

    This code is taken from [minGPT](https://github.com/karpathy/minGPT).
    This applies weight decay only to weights of linear layers.
    """
    # Collect names of parameters to apply weight decay
    decay = set()
    for mn, m in c.model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn  # full param name

            if fpn.endswith('weight') and isinstance(m, nn.Linear):
                decay.add(fpn)

    # Get all the parameters
    param_dict = {pn: p for pn, p in c.model.named_parameters()}
    # Parameters that are not decayed
    no_decay = set(param_dict.keys()) - decay

    # create the pytorch optimizer object
    opt_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": c.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    # Create a [configurable optimizer](../optimizers/configs.html#OptimizerConfigs),
    # so that we can change these simply by passing
    # a config dictionary.
    optimizer = OptimizerConfigs()

    # Set parameter groups for optimization.
    optimizer.parameters = opt_groups
    # Use [cosine decay optimizer](../optimizers/adam_warmup_cosine_decay.html).
    # This is what GPT uses.
    optimizer.optimizer = 'AdamWarmupCosineDecay'
    # Set model embedding size, required if we use [Noam optimizer](../optimizers/noam.html)
    # which has an exponential decay.
    optimizer.d_model = c.mconfigs.d_model
    # Set default weight decay.
    # This is not required since we set the weight decay in the parameter groups.
    optimizer.weight_decay = c.weight_decay
    # GPT uses a maximum learning rate of $6 \times 10^{-4}$.
    optimizer.learning_rate = 6e-4
    # $\beta_1 = 0.9, \beta_2 = 0.95$
    optimizer.betas = (0.9, 0.95)
    # $\epsilon = 10^{-8}$
    optimizer.eps = 1e-8
    # Weight decay is decoupled from gradients
    optimizer.weight_decouple = True
    # Total number of optimization steps for learning rate cosine decay
    optimizer.total_steps = c.mconfigs.epochs * len(c.text.train) // (c.mconfigs.batch_size * c.mconfigs.seq_len)
    # Number of warmup optimization steps
    optimizer.warmup = c.warmup_steps // (c.mconfigs.batch_size * c.mconfigs.seq_len)

    return optimizer


@option(GPTRun.train_loader)
def shuffled_train_loader(c: GPTRun):
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


@option(GPTRun.valid_loader)
def shuffled_valid_loader(c: GPTRun):
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
    conf = GPTRun(configs, text)

    # Set models for saving and loading
#    experiment.add_pytorch_models({'model': conf.model})
    # Start the experiment
    with experiment.start():
        # Run training
        conf.run()


#
if __name__ == '__main__':
    main()