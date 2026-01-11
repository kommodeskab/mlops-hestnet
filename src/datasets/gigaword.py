import logging
from dotenv import load_dotenv
from datasets import load_dataset
from src.datasets import BaseDataset
from src.datasets.utils import get_tokenize_function
from src import TextSample

load_dotenv()

logger = logging.getLogger(__name__)


class DGigawordDataset(BaseDataset):
    """Danish Gigaword dataset."""

    def __init__(
        self,
        checkpoint: str,
        ):
        super().__init__()
        self.name = "danish-foundation-models/danish-gigaword"
        self.ds = load_dataset(
            path=self.name,
            split="train",  # The dataset only has the trian split.
            cache_dir=self.data_path
        )
        self.tokenizer = get_tokenize_function(checkpoint)

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
        text = self.ds[index]['text']
        tokens = self.tokenizer(text)
        return TextSample(
            text = text,
            input_ids = tokens['input_ids'].squeeze(),
            attention_mask = tokens['attention_mask'].squeeze(),
        )
    
if __name__ == "__main__":
    dataset = DGigawordDataset()
    sample = dataset[0]
    print(sample['attention_mask'].shape)
    print(sample['input_ids'].shape)