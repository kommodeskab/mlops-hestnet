from pathlib import Path
import os
import logging
from datasets import load_dataset, get_dataset_config_names
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = Path(os.getenv("DATA_PATH"))  # Works on different operating systems
PROCESSED_DATA_PATH = Path("data") / "processed" / "danish_gigaword"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess(checkpoint: str, save_path=PROCESSED_DATA_PATH, **kwargs):
    name = "danish-foundation-models/danish-gigaword"
    dataset = load_dataset(
        path=name,
        split="train",  # The dataset only has the trian split.
        cache_dir=str(CACHE_DIR),
        token=HF_TOKEN,
        **kwargs,
    )
    tokenize_function = get_tokenize_function(checkpoint)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    save_path.mkdir(parents=True, exist_ok=True)
    tokenized_dataset.save_to_disk(save_path)
    return tokenized_dataset


def get_tokenize_function(checkpoint: str, **tokenizer_kwargs):
    """Create a tokenize function for the given checkpoint."""

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(elem):
        return tokenizer(elem, truncation=True, return_tensors="pt", **tokenizer_kwargs)

    return tokenize_function


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
    # print(sample)
