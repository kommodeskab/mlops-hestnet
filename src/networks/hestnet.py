import torch.nn as nn
from torch import Tensor
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from pathlib import Path
import os
from transformers.tokenization_utils_base import BatchEncoding
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
import logging

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from src.utils import (
    instantiate_callbacks,
    get_ckpt_path,
    model_config_from_id,
    get_current_time,
)
import pytorch_lightning as pl
import os
import hydra
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Callback
import wandb
import yaml
import logging


os.environ["HYDRA_FULL_ERROR"] = "1"

load_dotenv()
MODEL_DIR = Path(os.getenv("MODEL_PATH")) / "pretrained" 

class HestNet(nn.Module):
    """HestNet."""

    def __init__(
        self,
        checkpoint: str,
    ):
        super().__init__()
        # Can move the tokenizer out of the dataset if that is desirable.
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir = MODEL_DIR / checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir = MODEL_DIR / checkpoint)

    def forward(self, inputs: BatchEncoding, **kwargs) -> Tensor:
        outputs = self.model(**inputs, labels=inputs["input_ids"], **kwargs)
        return outputs


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.model.network.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir = MODEL_DIR / checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir = MODEL_DIR / checkpoint)
    inputs = tokenizer("Jeg bor i et kommodeskab", return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    logits = outputs.logits
    print(outputs)

if __name__ == "__main__":
    # Run with python ./src/networks/hestnet.py +experiment=hestnet
    main()