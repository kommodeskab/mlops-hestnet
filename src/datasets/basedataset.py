import hashlib
import os

from dotenv import load_dotenv
from torch.utils.data import Dataset

from src import Batch

load_dotenv()


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    @property
    def unique_identifier(self):
        attr_str = str(vars(self))
        return hashlib.md5(attr_str.encode()).hexdigest()

    @property
    def data_path(self):
        return os.getenv("DATA_PATH")

    def __len__(self) -> int:
        msg = "Length method not implemented"
        raise NotImplementedError(msg)

    def __getitem__(self, index: int) -> Batch:
        msg = "Get item method not implemented"
        raise NotImplementedError(msg)
