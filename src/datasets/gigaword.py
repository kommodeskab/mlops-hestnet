from src.datasets import BaseDataset
from pathlib import Path
import os
import logging
from datasets import load_dataset
from dotenv import load_dotenv
from src.datasets.utils import get_tokenize_function


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") # Huggingface Token
CACHE_DIR = Path(os.getenv("DATA_PATH"))  # Works on different operating systems

logger = logging.getLogger(__name__)


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

    def __getitem__(self, index: int) -> dict:
        return self.ds[index]['text']


if __name__ == "__main__":
    dataset = DGigawordDataset()
    print(len(dataset))

    checkpoint = "distilbert/distilgpt2"
    tokenize_function = get_tokenize_function(checkpoint)
    sample = tokenize_function(dataset[0])
    print(sample)
    for i in range(20):
        sample = tokenize_function(dataset[i])
        print(sample["input_ids"].shape)
        print(sample["attention_mask"].shape)
