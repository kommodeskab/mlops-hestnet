from src.lightning_modules import BaseLightningModule
import torch.nn as nn
from src import OptimizerType, LRSchedulerType, HestNetBatch, HestNetOutput, HestNetStepOutput

class HestnetModule(BaseLightningModule):
    """Hestnet LightningModule"""

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
        output = self.network(batch)
        return output
        # return HestnetOutput()
        # return ModelOutput(output=output)

    def common_step(self, batch: HestNetBatch, batch_idx: int) -> HestNetStepOutput:
        output = self.forward(batch) # Already takes attention mask as input and computes loss
        return HestNetStepOutput(
            loss=output.loss,
            model_output=output,
        )
