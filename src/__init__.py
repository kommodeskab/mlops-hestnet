import os
from functools import partial
from typing import Dict, Optional, TypeAlias, TypedDict, Union

import numpy as np
from matplotlib.figure import Figure
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

TensorDict = dict[str, Tensor]
OptimizerType = Optional[partial[Optimizer]]
LRSchedulerType = Optional[dict[str, partial[LRScheduler] | str]]
ImageType = list[Tensor | Figure | np.ndarray]
PathLike = Union[str, bytes, os.PathLike]

class Batch(TypedDict):
    input: Tensor
    target: Tensor


class ModelOutput(TypedDict):
    output: Tensor


class LossOutput(TypedDict):
    loss: Tensor


class StepOutput(TypedDict):
    loss: Tensor
    model_output: ModelOutput | None = None
    loss_output: LossOutput | None = None

# Hestnet
class HestNetBatch(TypedDict):
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor

type HestNetOutput = CausalLMOutputWithCrossAttentions
# class HestnetOutput(TypedDict):
#     output: CausalLMOutputWithCrossAttentions

class HestNetStepOutput(TypedDict):
    loss: Tensor
    model_output: ModelOutput | None = None
