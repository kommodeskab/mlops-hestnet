import logging
import os
from pathlib import Path

import psutil
from dotenv import load_dotenv
from transformers import AutoTokenizer

from datasets import get_dataset_config_names, load_dataset

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = Path(os.getenv("DATA_PATH"))  # Works on different operating systems

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory usage: {mem_info.rss / 1024**3:.2f} GB")


def get_tokenize_function(checkpoint: str, **tokenizer_kwargs):
    """Create a tokenize function for the given checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(elem):
        return tokenizer(elem["text"], padding=True, truncation=True, return_tensors="pt", **tokenizer_kwargs)

    return tokenize_function


def get_group_texts_function(block_size=128):
    def group_texts(elem):
        concatenated_elems = {k: sum(elem[k], []) for k in elem}
        total_length = len(concatenated_elems[list(elem.keys())[0]])
        # We drop the small remainder
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_elems.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return group_texts


if __name__ == "__main__":
    name = "danish-foundation-models/danish-gigaword"
    configs = get_dataset_config_names(name)
    print(configs)
    ds = load_dataset(
        path=name,
        split="train",  # The dataset only has the trian split.
        cache_dir=str(CACHE_DIR),
        token=HF_TOKEN,
    )
    print(ds.features)
    sample = ds[1]
