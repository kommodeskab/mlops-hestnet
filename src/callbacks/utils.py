from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def get_batch_from_dataset(dataset: Dataset, batch_size: int, shuffle: bool = False) -> Tensor:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return next(iter(dataloader))
