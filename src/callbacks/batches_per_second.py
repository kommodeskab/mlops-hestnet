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

    def __init__(self, smoothing_factor: float = 0.999):
        # initialize time trackers
        self.train_time = None
        self.val_time = None
        self.train_bps_avg = None
        self.val_bps_avg = None
        self.smoothing_factor = smoothing_factor

    def _calculate_bps_avg(self, previous_avg: float | None, new_bps: float) -> float:
        if previous_avg is None:
            return new_bps

        return (1 - self.smoothing_factor) * new_bps + self.smoothing_factor * previous_avg

    def _track_bps(self, phase: Literal["train", "val"], pl_module: pl.LightningModule):
        if phase == "train":
            before = self.train_time
            bps_avg = self.train_bps_avg
            log_key = "train_batches_per_second"
        else:
            before = self.val_time
            bps_avg = self.val_bps_avg
            log_key = "val_batches_per_second"

        now = time.time()

        if before is not None:
            elapsed = now - before
            bps = 1.0 / elapsed
            bps_avg = self._calculate_bps_avg(bps_avg, bps)
            pl_module.log(log_key, bps_avg)

        if phase == "train":
            self.train_time = now
            self.val_time = None
            self.train_bps_avg = bps_avg
        else:
            self.val_time = now
            self.train_time = None
            self.val_bps_avg = bps_avg

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx == 0:
            return  # skip first batch to avoid skewed timing

        self._track_bps("train", pl_module)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._track_bps("val", pl_module)
