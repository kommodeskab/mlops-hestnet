from src.datasets import BaseTextDataset
from datasets import load_dataset
from src import TextSample


class RedditDaDataset(BaseTextDataset):
    """
    A Danish Reddit dataset.
    Can be found at https://huggingface.co/datasets/DDSC/reddit-da
    """

    def __init__(self):
        super().__init__()
        self.name = "DDSC/reddit-da"
        self.ds = load_dataset(
            path=self.name,
            split="train",
            cache_dir=self.data_path,
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> TextSample:
        return TextSample(text=self.ds[index]["text"])


class ScandiRedditDataset(BaseTextDataset):
    """
    A danish Reddit dataset.
    Can be found at https://huggingface.co/datasets/alexandrainst/scandi-reddit
    """

    def __init__(self):
        super().__init__()
        self.name = "alexandrainst/scandi-reddit"
        self.ds = load_dataset(
            path=self.name,
            name="da",
            cache_dir=self.data_path,
            split="train",
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> TextSample:
        return TextSample(text=self.ds[index]["doc"])


if __name__ == "__main__":
    dataset = RedditDaDataset()
    sample = dataset[0]
    print(sample)
    print(len(dataset))

    dataset_da = ScandiRedditDataset()
    sample_da = dataset_da[0]
    print(sample_da)
    print(len(dataset_da))
