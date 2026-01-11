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

    def common_step(self, batch: HestNetBatch, batch_idx: int) -> StepOutput:
        output = self.forward(batch) # Already takes attention mask as input and computes loss
        return StepOutput(
            loss=output.loss,
            loss_output=output, # This is only included for the log_loss_callback
            model_output=output,
        )