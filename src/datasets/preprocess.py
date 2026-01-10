import gc
import logging
import os
from pathlib import Path

import hydra
import yaml
from dotenv import load_dotenv
from memory_profiler import profile
from omegaconf import DictConfig, OmegaConf

from datasets import Dataset, load_dataset
from src import PathLike
from src.datasets.utils import get_tokenize_function

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = Path(os.getenv("DATA_PATH"))

os.environ["HYDRA_FULL_ERROR"] = "1"

@profile
def preprocess(
        name: str,
        checkpoint: str,
        save_path: PathLike,
        size: int | None = None,
        batch_size: int = 1000,
        num_proc: int = 4,
        writer_batch_size: int = 1000,
        **kwargs):

    dataset: Dataset = load_dataset(
        path=name,
        split="train",  # The dataset only has the trian split.
        cache_dir=str(CACHE_DIR),
        token=HF_TOKEN,
        **kwargs,
    )
    if size:
        dataset = dataset.shuffle(seed=42).select(range(size))
    tokenize_function = get_tokenize_function(checkpoint)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batch_size=batch_size,
        writer_batch_size=writer_batch_size,
        num_proc=num_proc,
        keep_in_memory=False,
        remove_columns=dataset.column_names,
    )

    gc.collect()  # Force garbage collection

    tokenized_dataset.save_to_disk(save_path)
    return tokenized_dataset


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocess")
def my_app(cfg: DictConfig) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"{Path.cwd()=}")

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.info("Config:\n%s", yaml.dump(config, default_flow_style=False, sort_keys=False))
    save_path = Path(cfg.preprocess_path)
    logger.info(f"Saving processed dataset to {save_path}")
    # Load and preprocess the dataset

    ds = preprocess(
        name=cfg.name,
        checkpoint=cfg.checkpoint,
        save_path=save_path,
        size=cfg.size,
        batch_size=cfg.checkpoint,
        writer_batch_size=cfg.checkpoint,
        num_proc=cfg.checkpoint,
    )
    logger.info(ds.features)
    logger.info("Dataset successfully preprocessed and written to disk")


if __name__ == "__main__":
    my_app()
