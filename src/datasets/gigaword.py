from src.datasets import BaseDataset
from pathlib import Path
import os
import logging
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = Path(os.getenv("DATA_PATH"))  # Works on different operating systems

logger = logging.getLogger(__name__)

checkpoint = "distilgpt2"  # Abstract out later
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(elem):
    return tokenizer(elem["text"], truncation=True, return_tensors="pt")


class DGigawordDataset(BaseDataset):
    """Danish Gigaword dataset"""

    def __init__(self, **kwargs):
        super().__init__()
        self.name = "danish-foundation-models/danish-gigaword"

        self.ds = load_dataset(
            path=self.name,
            split="train",  # The dataset only has the trian split.
            cache_dir=str(CACHE_DIR),
            token=HF_TOKEN,
            **kwargs,
        )
        self.size = len(self.ds)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> BatchEncoding:
        encoded_input = tokenize_function(self.ds[index])
        return encoded_input  # contains "input_ids" and "attention_mask"
        #
        # return Batch(input=input, target=target)


if __name__ == "__main__":
    dataset = DGigawordDataset()
    print(len(dataset))
    sample = dataset[0]
    print(sample)
    for i in range(20):
        sample = dataset[i]
        print(sample['input_ids'].shape)
        print(sample['attention_mask'].shape)
