"""
---
title: Configurable optimizer module
summary: This implements a configurable module for optimizers.
---

# Configurable Optimizer
"""

from typing import Tuple

import torch

from labml.configs import BaseConfigs, option, meta_config
from . import WeightDecay


class OptimizerConfigs(BaseConfigs):
    """
    <a id="OptimizerConfigs"></a>

    ## Optimizer Configurations
    """

    # Optimizer
    optimizer: torch.optim.Adam

    # Weight decay
    weight_decay_obj: WeightDecay
    # Whether weight decay is decoupled;
    # i.e. weight decay is not added to gradients
    weight_decouple: bool = True
    # Weight decay
    weight_decay: float = 0.0
    # Whether weight decay is absolute or should be multiplied by learning rate
    weight_decay_absolute: bool = False

    # Whether the adam update is optimized (different epsilon)
    optimized_adam_update: bool = True

    # Parameters to be optimized
    parameters: any

    # Learning rate $\alpha$
    learning_rate: float = 0.01
    # Beta values $(\beta_1, \beta_2)$ for Adam
    betas: Tuple[float, float] = (0.9, 0.999)
    # Epsilon $\epsilon$ for adam
    eps: float = 1e-08

    # Momentum for SGD
    momentum: float = 0.5
    # Whether to use AMSGrad
    amsgrad: bool = False

    # Number of warmup optimizer steps
    warmup: int = 2_000
    # Total number of optimizer steps (for cosine decay)
    total_steps: int = int(1e10)

    # Whether to degenerate to SGD in AdaBelief
    degenerate_to_sgd: bool = True

    # Whether to use Rectified Adam in AdaBelief
    rectify: bool = True

    # Model embedding size for Noam optimizer
    d_model: int

    rho: float

    def __init__(self):
        super().__init__(_primary='optimizer')


meta_config(OptimizerConfigs.parameters)


@option(OptimizerConfigs.weight_decay_obj, 'L2')
def _weight_decay(c: OptimizerConfigs):
    return WeightDecay(c.weight_decay, c.weight_decouple, c.weight_decay_absolute)



@option(OptimizerConfigs.optimizer, 'Adam')
def _adam_optimizer(c: OptimizerConfigs):
    if c.amsgrad:
        from .amsgrad import AMSGrad
        return AMSGrad(c.parameters,
                       lr=c.learning_rate, betas=c.betas, eps=c.eps,
                       optimized_update=c.optimized_adam_update,
                       weight_decay=c.weight_decay_obj, amsgrad=c.amsgrad)
    else:
        from .adam import Adam
        return Adam(c.parameters,
                    lr=c.learning_rate, betas=c.betas, eps=c.eps,
                    optimized_update=c.optimized_adam_update,
                    weight_decay=c.weight_decay_obj)


@option(OptimizerConfigs.optimizer, 'Noam')
def _noam_optimizer(c: OptimizerConfigs):
    from .noam import Noam
    return Noam(c.parameters,
                lr=c.learning_rate, betas=c.betas, eps=c.eps,
                weight_decay=c.weight_decay_obj, amsgrad=c.amsgrad, warmup=c.warmup,
                d_model=c.d_model)
