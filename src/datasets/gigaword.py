import torch
from src.datasets import BaseDataset
from src import Batch
from pathlib import Path
import os
import logging
from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv

from transformers import AutoTokenizer


load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
CACHE_DIR = Path(os.getenv('DATA_PATH')) # Works on different operating systems

logger = logging.getLogger(__name__)

checkpoint = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(elem):
    return tokenizer(elem['text'], truncation=True, return_tensors="pt")

class DGigawordDataset(BaseDataset):
    """Danish Gigaword dataset"""

    def __init__(self, **kwargs):
        super().__init__()
        self.name="danish-foundation-models/danish-gigaword"

        self.ds = load_dataset(
            path=self.name,
            split="train",  # The dataset only has the trian split.
            cache_dir=str(CACHE_DIR),
            token = HF_TOKEN,
            **kwargs
        )
        self.size = len(self.ds)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        encoded_input = tokenize_function(self.ds[index])
        return encoded_input # contains "input_ids" and "attention_mask"
        # return Batch(input=input, target=target)


if __name__ == "__main__":
    dataset = DGigawordDataset()
    sample = dataset[0]
    print(sample)
