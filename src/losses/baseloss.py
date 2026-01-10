import torch.nn as nn

from src import Batch, LossOutput, ModelOutput


class BaseLossFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        msg = "Loss function not implemented"
        raise NotImplementedError(msg)

    def __call__(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        return self.forward(model_output, batch)
