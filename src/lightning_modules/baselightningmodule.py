import pytorch_lightning as pl
import torch
import torch.nn.init as init
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import Dataset

from src import Batch, ImageType, LRSchedulerType, ModelOutput, OptimizerType
from src.data_modules import BaseDM
from src.utils import temporary_seed


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        optimizer: OptimizerType = None,
        lr_scheduler: LRSchedulerType = None,
    ):
        super().__init__()
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler

    @property
    def datamodule(self) -> BaseDM:
        return self.trainer.datamodule

    @property
    def trainset(self) -> Dataset:
        return self.datamodule.trainset

    @property
    def valset(self) -> Dataset | None:
        return self.datamodule.valset

    @property
    def testset(self) -> Dataset | None:
        return self.datamodule.testset

    @property
    def logger(self) -> WandbLogger:
        return self.trainer.logger

    def log_images(self, key: str, images: list[ImageType], **kwargs) -> None:
        logger = self.logger
        logger.log_image(key=key, images=images, step=self.global_step)

    @staticmethod
    def init_weights(model: nn.Module) -> None:
        """Initializes the weights of the forward and backward models
        using the Kaiming Normal initialization.
        """

        @torch.no_grad()
        def initialize(m):
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, nn.ConvTranspose2d)):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        # Apply initialization to networks
        model.apply(initialize)

    def common_step(self, batch: Batch, batch_idx: int) -> ModelOutput:
        r"""The common step contains the logic for both training, validation, and test steps. \n
        If train/val/test steps differ, override them in the subclass.
        """
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)

    def training_step(self, batch: Batch, batch_idx: int) -> ModelOutput:
        return self.common_step(batch, batch_idx)

    def validation_step(self, batch: Batch, batch_idx: int) -> ModelOutput:
        with temporary_seed(0):
            return self.common_step(batch, batch_idx)

    def test_step(self, batch: Batch, batch_idx: int) -> ModelOutput:
        with temporary_seed(0):
            return self.common_step(batch, batch_idx)

    def configure_optimizers(self):
        assert self.partial_optimizer is not None, "Optimizer must be provided during training."
        assert self.partial_lr_scheduler is not None, "Learning rate scheduler must be provided during training."

        optim = self.partial_optimizer(self.parameters())
        scheduler = self.partial_lr_scheduler.pop("scheduler")(optim)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": scheduler, **self.partial_lr_scheduler},
        }
