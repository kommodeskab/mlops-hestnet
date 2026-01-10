import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


def split_dataset(
    train_dataset: Dataset, val_dataset: Dataset | None, train_val_split: float | None = None,
) -> tuple[Dataset, Dataset]:
    if train_val_split is not None:
        train_dataset, val_dataset = random_split(train_dataset, [train_val_split, 1 - train_val_split])

    return train_dataset, val_dataset


class BaseDM(pl.LightningDataModule):
    def __init__(
        self,
        trainset: Dataset,
        valset: Dataset | None = None,
        testset: Dataset | None = None,
        train_val_split: float | None = None,
        **kwargs,
    ):
        """A base data module for datasets.
        It takes a dataset and splits into train and validation (if valset is None).
        """
        super().__init__()
        self.trainset, self.valset = split_dataset(trainset, valset, train_val_split)
        self.testset = testset
        self.kwargs = kwargs

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
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
            shuffle=False,
            drop_last=True,
            **kwargs,
        )
