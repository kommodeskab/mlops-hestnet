from src.lightning_modules import BaseLightningModule
from src.losses import BaseLossFunction
import torch.nn as nn
from src import OptimizerType, LRSchedulerType, TextBatch, ModelOutput, StepOutput
from src.networks import CausalTransformer

class CausalLLM(BaseLightningModule):
    """A dummy LightningModule for testing purposes."""

    def __init__(
        self,
        network: CausalTransformer,
        loss_fn: BaseLossFunction,
        optimizer: OptimizerType = None,
        lr_scheduler: LRSchedulerType = None,
    ):
        # optimizer and lr_scheduler can be None, for example during testing
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.network = network
        self.loss_fn = loss_fn

    def forward(self, batch: TextBatch) -> ModelOutput:
        output = self.network.forward(batch["input_ids"])
        return ModelOutput(output=output)

    def common_step(self, batch: TextBatch, batch_idx: int) -> ModelOutput:
        output = self.forward(batch)
        loss = self.loss_fn(output, batch)
        return StepOutput(
            loss=loss["loss"],
            loss_output=loss,
            model_output=output,
        )