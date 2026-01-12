import logging
from datasets import load_dataset
from src import TextSample
from src.datasets.tokenization import Tokenizer
from src.datasets.textdataset import BaseTextDataset


logger = logging.getLogger(__name__)


class DGigawordDataset(BaseTextDataset):
    """Danish Gigaword dataset."""

    def __init__(
        self,
    ):
        super().__init__()
        self.name = "danish-foundation-models/danish-gigaword"
        self.ds = load_dataset(
            path=self.name,
            split="train",  # The dataset only has the trian split.
            cache_dir=self.data_path,
        )

    @property
    def features(self):
        return self.ds.features

    @property
    def column_names(self):
        return self.ds.column_names

    @property
    def config_name(self):
        return self.ds.config_name

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> TextSample:
        return TextSample(text=self.ds[index]["text"])


if __name__ == "__main__":
    tokenizer = Tokenizer("distilbert/distilgpt2")
    dataset = DGigawordDataset()
    sample = dataset[0]
    print(sample["text"])
