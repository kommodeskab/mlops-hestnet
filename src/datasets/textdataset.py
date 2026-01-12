from src.datasets.basedataset import BaseDataset
from src import TextSample

class BaseTextDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        
    def __getitem__(self, index: int) -> TextSample: ...