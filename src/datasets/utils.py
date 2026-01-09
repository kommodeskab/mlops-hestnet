from pathlib import Path
import os
import logging
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
CACHE_DIR = Path("data") / "raw" / "danish_gigaword"

logger = logging.getLogger(__name__)

def load_dataset(name, cache_dir = None, split = None):
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Huggingface api handles downloading the dataset and loading it if it is downloaded.
    ds = load_dataset(
        name,
        split=split,
        cache_dir=str(cache_dir),
        token = HF_TOKEN,
    )
    logger.info(f"Loaded splits: {list(ds.keys())}")
    return ds

def load_danish_gigaword(cache_dir = None, split = None):
    name = "danish-foundation-models/danish-gigaword"
    load_dataset(name, cache_dir=cache_dir, split=split)

if __name__ == "__main__":
    load_danish_gigaword(cache_dir=CACHE_DIR)
