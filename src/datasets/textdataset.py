from src.datasets.basedataset import BaseDataset
from src import TextSample

class BaseTextDataset(BaseDataset):
    name: str
    
    def __init__(self):
        super().__init__()
        
    def __getitem__(self, index: int) -> TextSample: ...
    
class DummyTextDataset(BaseTextDataset):    
    def __init__(self):
        super().__init__()
        self.name = "DummyTextDataset"
        
    def __getitem__(self, index: int) -> TextSample:
        return 1000
    
    def __len__(self) -> int:
        return TextSample(text="This is a dummy text sample.")