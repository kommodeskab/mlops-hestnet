import torch.nn as nn

from src import HestNetBatch, HestNetOutput, HestNetStepOutput, LRSchedulerType, OptimizerType, StepOutput
from src.lightning_modules import BaseLightningModule


class HestnetModule(BaseLightningModule):
    """Hestnet LightningModule."""

    def __init__(
        self,
        network: nn.Module,
        # loss_fn: BaseLossFunction,
        optimizer: OptimizerType = None,
        lr_scheduler: LRSchedulerType = None,
    ):
        # optimizer and lr_scheduler can be None, for example during testing
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.network = network

    def forward(self, batch: HestNetBatch) -> HestNetOutput:
        return self.network(batch)
        # return HestnetOutput()
        # return ModelOutput(output=output)

    def common_step(self, batch: HestNetBatch, batch_idx: int) -> HestNetStepOutput:
        output = self.forward(batch) # Already takes attention mask as input and computes loss
        return HestNetStepOutput(
            loss=output.loss,
            model_output=output,
        )

    def training_step(self, batch: HestNetBatch, batch_idx: int):
        step_output = self.common_step(batch, batch_idx)
        self.log("train_loss", step_output.get("loss"), on_step=True, on_epoch=True, prog_bar=True)
        return step_output.get("loss")
    
    def validation_step(self, batch: HestNetBatch, batch_idx: int):
        step_output = self.common_step(batch, batch_idx)
        self.log("val_loss", step_output.get("loss"), on_step=False, on_epoch=True, prog_bar=True)
        return step_output.get("loss")