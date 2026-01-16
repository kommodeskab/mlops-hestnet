from pytorch_lightning import Callback
import pytorch_lightning as pl
import time
from typing import Literal


class BatchesPerSecondCallback(Callback):
    """
    A callback for logging the number of batches processed per second.
    This can be useful for monitoring training performance.
    Logs both during training, validation, and testing phases.
    """

    def __init__(self):
        # initialize time trackers
        self.info = {
            "train": {"steps": 0, "summed_time": 0.0, "last_time": None},
            "val": {"steps": 0, "summed_time": 0.0, "last_time": None},
            "test": {"steps": 0, "summed_time": 0.0, "last_time": None},
        }

    def _track_bps(self, phase: Literal["train", "val", "test"], pl_module: pl.LightningModule):
        info = self.info[phase]
        before = info["last_time"]
        now = time.time()

        if before is None:
            # if this is the first batch, just set the time
            info["last_time"] = now
            return

        elapsed = now - before
        info["summed_time"] += elapsed
        info["steps"] += 1

        bps = info["steps"] / info["summed_time"]

        pl_module.log(f"{phase}_batches_per_second", bps)
        info["last_time"] = now

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # this measures time between the start of batches, i.e. a full iteration
        # could use almost any hook
        self._track_bps("train", pl_module)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._track_bps("val", pl_module)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._track_bps("test", pl_module)
