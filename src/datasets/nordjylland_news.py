from src.datasets import BaseTextDataset
from datasets import load_dataset
from src import TextSample
from typing import Literal


class NordjyllandNewsDataset(BaseTextDataset):
    """
    A Scandinavian Reddit dataset for text summarization.
    Can be found at: https://huggingface.co/datasets/alexandrainst/nordjylland-news-summarization

    Args:
        split (Literal["train", "val", "test"]): The dataset split to use. Defaults to "train".
    """

    def __init__(self, split: Literal["train", "val", "test"] = "train"):
        super().__init__()
        self.name = "alexandrainst/nordjylland-news-summarization"
        self.ds = load_dataset(path=self.name, cache_dir=self.data_path, split=split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> TextSample:
        sample = self.ds[index]
        summary, body = sample["summary"], sample["text"]
        text = summary + " " + body
        return TextSample(text=text)


if __name__ == "__main__":
    dataset = NordjyllandNewsDataset()
    sample = dataset[0]
    print(sample)
    print(len(dataset))
