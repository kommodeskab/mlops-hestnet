from src.datasets.basedataset import BaseDataset
from src import TextSample


class BaseTextDataset(BaseDataset):
    """
    A base dataset for datasets that contain just text.
    """

    def __init__(self):
        super().__init__()

    def __getitem__(self, index: int) -> TextSample: ...
