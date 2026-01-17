from pytorch_lightning import Callback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from pathlib import Path
from torch import Tensor


class LogModelCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        ckpt_callback: ModelCheckpoint = trainer.checkpoint_callback
        assert ckpt_callback is not None, "ModelCheckpoint callback is required for LogModelCallback."

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        ckpt_callback: ModelCheckpoint = trainer.checkpoint_callback

        checkpoints = {}
        if hasattr(ckpt_callback, "last_model_path") and hasattr(ckpt_callback, "current_score"):
            checkpoints[ckpt_callback.last_model_path] = (ckpt_callback.current_score, "latest")

        if hasattr(ckpt_callback, "best_model_path") and hasattr(ckpt_callback, "best_model_score"):
            checkpoints[ckpt_callback.best_model_path] = (ckpt_callback.best_model_score, "best")

        if hasattr(ckpt_callback, "best_k_models"):
            for key, value in ckpt_callback.best_k_models.items():
                checkpoints[key] = (value, "best_k")

        for t, p, s, tag in checkpoints:
            metadata = {
                "score": s.item() if isinstance(s, Tensor) else s,
                "original_filename": Path(p).name,
                ckpt_callback.__class__.__name__: {
                    k: getattr(ckpt_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(ckpt_callback, k)
                },
            }
            if not self._checkpoint_name:
                self._checkpoint_name = f"model-{self.experiment.id}"
            artifact = wandb.Artifact(name=self._checkpoint_name, type="model", metadata=metadata)
            artifact.add_file(p, name="model.ckpt", policy=self.add_file_policy)
            aliases = ["latest", "best"] if p == ckpt_callback.best_model_path else ["latest"]
            self.experiment.log_artifact(artifact, aliases=aliases)
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t
