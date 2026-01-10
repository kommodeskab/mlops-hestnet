from typing import Optional
from src.datasets import BaseDataset
from pathlib import Path
import os
import logging
from datasets import load_dataset, load_from_disk, Dataset
from dotenv import load_dotenv
from src.datasets.utils import get_tokenize_function
from transformers import AutoTokenizer
from src import TensorDict, PathLike

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") # Huggingface Token
CACHE_DIR = Path(os.getenv("DATA_PATH"))  # Works on different operating systems

logger = logging.getLogger(__name__)


class DGigawordDataset(BaseDataset):
    """Danish Gigaword dataset"""

    def __init__(
        self, 
        size: Optional[int] = None, 
        skip_load: bool = False,
        **kwargs
    ):
        super().__init__()
        self.name = "danish-foundation-models/danish-gigaword"
        self.kwargs = kwargs
        self.dataset_loaded = False
        # self.ds = None
        if not skip_load:
            self.load(size, **kwargs)
        else:
            self.ds = None
            self.size = 0

    def load(self, size, **kwargs):
        if self.dataset_loaded:
            logger.warning("Dataset already preprocessed. Skipping.")
            return
        self.ds = load_dataset(
            path=self.name,
            split="train",  # The dataset only has the trian split.
            cache_dir=str(CACHE_DIR),
            token=HF_TOKEN,
            **kwargs,
        )    
        if size:
            self.ds = self.ds.shuffle(seed=42).select(range(size))
        self.size = len(self.ds)
        self.dataset_loaded = True

    @property
    def features(self):
        """Dynamically return current dataset features."""
        return self.ds.features

    @property
    def column_names(self):
        return self.ds.column_names

    @property
    def config_name(self):
        return self.ds.config_name

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        return self.ds[index]

    def _preprocess(self, checkpoint: str, num_proc: int = 4) -> Dataset:
        tokenize_function = get_tokenize_function(checkpoint)

        processed_ds = self.ds.map(
            tokenize_function, 
            batched=True,
            num_proc=num_proc,
        remove_columns=self.ds.column_names,
        )
        logger.info(f"Preprocessed dataset with {len(self.ds)} samples. Processed dataset has {len(processed_ds)} samples")
        return processed_ds
    


class TDGigawordDataset(DGigawordDataset):
    """Tokenized Danish Gigaword dataset"""

    def __init__(
        self, 
        checkpoint: Optional[str],
        size: Optional[int] = None, 
        preprocess: bool = False,
        preprocessed_path: PathLike = None,
        num_proc: int = 4,
        **kwargs
    ):
        if preprocessed_path:
            super().__init__(size=None, skip_load=preprocessed_path, **kwargs)
            self.ds = load_from_disk(preprocessed_path, keep_in_memory=False)
            self.ds.set_format("torch")
            self.size = len(self.ds)
            self.preprocessed = True 
        else:
            super().__init__(size, **kwargs)
            self.preprocessed = False

        self.checkpoint = checkpoint if checkpoint else "distilbert/distilgpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if preprocess:
            if not preprocessed_path:
                if not checkpoint:
                    raise ValueError("checkpoint must be provided when preprocess=True")
                self.preprocess(num_proc)
            else:
                logger.warning("Preprocessed dataset loaded. Skipping.")

    def tokenize_function(self, elem):
        # Ensure elements are tokenized to the same length for batched and parrelilized operations.
        return self.tokenizer(elem["text"], padding='max_length', truncation=True)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> TensorDict:
        if self.preprocessed:
            return self.ds[index]
        else:
            # Make sure to return tensors when tokenizing on the fly
            tokenized = self.tokenizer(self.ds[index]["text"], padding='max_length', truncation=True, return_tensors="pt")
            return self._labels_map(tokenized)

    def _preprocess(self, num_proc: int = 4) -> Dataset:
        assert not self.preprocessed, "Called _preprocess but dataset is already preprocessed!"

        processed_ds = self.ds.map(
            self.tokenize_function, 
            batched=True,
            num_proc=num_proc,
        remove_columns=self.ds.column_names,
        ).map(self._labels_map)
        logger.info(f"Preprocessed dataset with {len(self.ds)} samples. Processed dataset has {len(processed_ds)} samples")
        return processed_ds

    def preprocess(self, num_proc: int = 4):
        """Public method to preprocess after initialization."""
        if self.preprocessed:
            logger.warning("Dataset already preprocessed. Skipping.")
            return
        
        self.ds = self._preprocess(num_proc)
        self.ds.set_format("torch") # function changes dataset in-place
        self.size = len(self.ds)
        self.preprocessed = True
    
    @staticmethod
    def _labels_map(elem):
        # Set labels to input_ids
        return elem | {"labels": elem["input_ids"]}


if __name__ == "__main__":
    # Standard DGigawordDataset returns dictionaries with text
    dataset = DGigawordDataset(1000)
    print(len(dataset))
    print(dataset.features)
    for i in range(10):
        sample = dataset[i]
        print(sample)

    # # TDGigawordDataset returns dictionaries with tokens and attention maps.
    checkpoint = "distilbert/distilgpt2"
    dataset = TDGigawordDataset(checkpoint, 1000)
    print(dataset.features)
    for i in range(10):
        sample = dataset[i]
        print(sample["input_ids"].shape)
        print(sample["attention_mask"].shape)

    dataset = TDGigawordDataset(checkpoint, 1000, preprocess=True, num_proc=8)
    print(dataset.features)
    for i in range(10):
        sample = dataset[i]
        print(sample["input_ids"].shape)
        print(sample["attention_mask"].shape)

    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=dataset.tokenizer, mlm=False)

    dl = DataLoader(dataset=dataset, collate_fn=data_collator)

    for i, batch in enumerate(dl):
        print(batch.keys())  # dict_keys(['input_ids', 'attention_mask', 'labels'])
        print(batch['input_ids'].shape)  # [batch_size, seq_len]
        print(batch['labels'].shape)     # [batch_size, seq_len]
        if i == 10:
            break
