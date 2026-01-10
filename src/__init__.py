from typing import TypedDict, Dict, Optional
from functools import partial
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from matplotlib.figure import Figure
import numpy as np
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import TypeAlias

TensorDict = Dict[str, Tensor]
OptimizerType = Optional[partial[Optimizer]]
LRSchedulerType = Optional[dict[str, partial[LRScheduler] | str]]
ImageType = list[Tensor | Figure | np.ndarray]


class Batch(TypedDict):
    input: Tensor
    target: Tensor


class ModelOutput(TypedDict):
    output: Tensor


class LossOutput(TypedDict):
    loss: Tensor


class StepOutput(TypedDict):
    loss: Tensor
    model_output: Optional[ModelOutput] = None
    loss_output: Optional[LossOutput] = None

# Hestnet
class HestNetBatch(TypedDict):
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor

HestNetOutput: TypeAlias = CausalLMOutputWithCrossAttentions
# class HestnetOutput(TypedDict):
#     output: CausalLMOutputWithCrossAttentions

class HestNetStepOutput(TypedDict):
    loss: Tensor
    model_output: Optional[ModelOutput] = None
