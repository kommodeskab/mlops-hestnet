import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import DataCollatorForLanguageModeling

from src.datasets.gigaword import TDGigawordDataset

logger = logging.getLogger(__name__)


def split_dataset(
    train_dataset: Dataset,
    val_dataset: Dataset | None,
    train_val_split: float | None = None,
) -> tuple[Dataset, Dataset]:
    if train_val_split is not None:
        train_dataset, val_dataset = random_split(train_dataset, [train_val_split, 1 - train_val_split])

    return train_dataset, val_dataset


class TDGigawordDM(pl.LightningDataModule):
    def __init__(
        self,
        trainset: TDGigawordDataset,
        valset: TDGigawordDataset | None = None,
        testset: TDGigawordDataset | None = None,
        train_val_split: float | None = None,
        **kwargs,
    ):
        """A base data module for datasets.
        It takes a dataset and splits into train and validation (if valset is None).
        """
        super().__init__()
        self.checkpoint = trainset.checkpoint
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=trainset.tokenizer, mlm=False)
        self.trainset, self.valset = split_dataset(trainset, valset, train_val_split)
        self.testset = testset
        self.kwargs = kwargs

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            collate_fn=self.data_collator,
            **self.kwargs,
        )

    def val_dataloader(self):
        if self.valset is None:
            return None

        kwargs = self.kwargs.copy()
        # remove the shuffle and drop_last for validation dataloader
        for key in ["shuffle", "drop_last"]:
            if key in kwargs:
                kwargs.pop(key)

        return DataLoader(
            dataset=self.valset,
            collate_fn=self.data_collator,
            shuffle=False,
            drop_last=True,
            **kwargs,
        )

    def test_dataloader(self):
        if self.testset is None:
            return None

        kwargs = self.kwargs.copy()
        # remove the shuffle and drop_last for test dataloader
        for key in ["shuffle", "drop_last"]:
            if key in kwargs:
                kwargs.pop(key)

        return DataLoader(
            dataset=self.testset,
            collate_fn=self.data_collator,
            shuffle=False,
            drop_last=True,
            **kwargs,
        )


if __name__ == "__main__":
    datamodule = TDGigawordDM(TDGigawordDataset("distilbert/distilgpt2"), train_val_split=0.9)
    train_loader = datamodule.train_dataloader()
    print(len(train_loader))
    val_loader = datamodule.val_dataloader()
    print(len(val_loader))

    for batch in train_loader:
        print(batch)
        break

    for batch in val_loader:
        print(batch)
        break
