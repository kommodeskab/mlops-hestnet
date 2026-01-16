from pytorch_lightning import Callback
import pytorch_lightning as pl
import time
from typing import Literal


class BatchesPerSecondCallback(Callback):
    """
    A callback for logging the number of batches processed per second.
    This can be useful for monitoring training performance.
    Logs both during training and validation phases.
    """

    def __init__(self):
        # initialize time trackers
        self.train_info = {"steps": 0, "avg": 0.0, "time": None, "log_key": "train_batches_per_second"}
        self.val_info = {"steps": 0, "avg": 0.0, "time": None, "log_key": "val_batches_per_second"}

    def _track_bps(self, phase: Literal["train", "val"], pl_module: pl.LightningModule):
        info = self.train_info if phase == "train" else self.val_info

        before = info["time"]
        steps = info["steps"]
        avg = info["avg"]

        now = time.time()

        if before is not None:
            elapsed = now - before
            curr_bps = 1.0 / elapsed
            updated_avg = (avg * steps + curr_bps) / (steps + 1)
            pl_module.log(info["log_key"], curr_bps)
            info["avg"] = updated_avg
            info["steps"] += 1

        info["time"] = now

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._track_bps("train", pl_module)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._track_bps("val", pl_module)
