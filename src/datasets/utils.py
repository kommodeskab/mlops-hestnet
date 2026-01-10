from typing import Optional
from pathlib import Path
import os
import logging
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = Path(os.getenv("DATA_PATH"))  # Works on different operating systems
PROCESSED_DATA_PATH = Path("data") / "processed" / "danish_gigaword"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

block_size = 128

def preprocess(
        name: str,
        checkpoint: str, 
        save_path=PROCESSED_DATA_PATH,
        size: Optional[int] = None,
        num_proc: int = 4,
        **kwargs):
        
    dataset = load_dataset(
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
        batched=True,
        num_proc=num_proc,
    remove_columns=dataset.column_names,
    )
    # lm_dataset = tokenized_dataset.map(lambda elem: group_texts(elem, block_size), batched=True, num_proc=4)
    # save_path.mkdir(parents=True, exist_ok=True)
    # tokenized_dataset.save_to_disk(save_path)
    return tokenized_dataset


def get_tokenize_function(checkpoint: str, **tokenizer_kwargs):
    """Create a tokenize function for the given checkpoint."""

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(elem):
        return tokenizer(elem["text"], padding=True, truncation=True, return_tensors="pt", **tokenizer_kwargs)

    return tokenize_function

def group_texts(elem, block_size = 128):
    concatenated_elems = {k: sum(elem[k], []) for k in elem.keys()}
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


if __name__ == "__main__":
    # name = "danish-foundation-models/danish-gigaword"
    # configs = get_dataset_config_names(name)
    # print(configs)
    # ds = load_dataset(
    #     path=name,
    #     split="train",  # The dataset only has the trian split.
    #     cache_dir=str(CACHE_DIR),
    #     token=HF_TOKEN,
    # )
    # print(ds.features)
    # sample = ds[1]
    # del ds
    
    # Load and preprocess the dataset
    ds = preprocess("danish-foundation-models/danish-gigaword", "distilbert/distilgpt2", size=1000)
    print(ds.features)
