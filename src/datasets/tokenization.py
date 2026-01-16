import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Optional
from src import TensorDict, TokenizedBatch
from src.datasets.basedataset import BaseDataset
from src.datasets.textdataset import BaseTextDataset
from torch.utils.data import ConcatDataset
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets
import os
from tqdm import tqdm


class Tokenizer:
    def __init__(
        self,
        checkpoint: str,
        max_length: Optional[int] = None,
    ):
        self.name = checkpoint
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(checkpoint)
        self.max_length = max_length or self.tokenizer.model_max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, string: str | list[str]) -> TensorDict:
        return self.tokenizer(
            string,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding_side="left",
        )

    def batch_decode(self, token_ids: list[list[int]]) -> list[str]:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def preprocess_dataset(
    dataset: BaseTextDataset,
    tokenizer: Tokenizer,
    batch_size: int = 1000,
    num_proc: int = 4,
) -> HFDataset:
    def gen():
        for i in range(len(dataset)):
            yield dataset[i]

    hf_dataset = HFDataset.from_generator(gen)

    tokenized_dataset = hf_dataset.map(
        lambda x: tokenizer(x["text"]),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset...",
        batch_size=batch_size,
        writer_batch_size=batch_size,
        num_proc=num_proc,
    )

    return tokenized_dataset


class TokenizedDataset(BaseDataset):
    def __init__(
        self, datasets: list[BaseTextDataset], tokenizer: Tokenizer, preprocess: bool = False, force: bool = False
    ):
        super().__init__()
        self.preprocess = preprocess
        self.datasets: list[BaseTextDataset | HFDataset] = []
        self.tokenizer = tokenizer

        if not preprocess:
            self.dataset = ConcatDataset(datasets)

        else:
            # if we preprocess..
            for ds in tqdm(datasets, desc="Preprocessing datasets.."):
                # loop over all datasets
                # define where the preprocessed dataset should be stored
                tokenname = tokenizer.name.replace("/", "_")
                datasetname = ds.name.replace("/", "_")
                preprocessed_path = f"{self.data_path}/preprocessed/{tokenname}/{datasetname}"

                # if the preprocessed dataset exists, load it
                if os.path.exists(preprocessed_path):
                    tokenized_ds = HFDataset.load_from_disk(preprocessed_path)
                else:
                    # else preprocess and save it as a huggingface dataset
                    tokenized_ds = preprocess_dataset(
                        dataset=ds,
                        tokenizer=tokenizer,
                        batch_size=1000,
                        num_proc=4,
                    )
                    os.makedirs(preprocessed_path, exist_ok=True)
                    tokenized_ds.save_to_disk(preprocessed_path)

                self.datasets.append(tokenized_ds)

            self.dataset = concatenate_datasets(self.datasets)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> TokenizedBatch:
        item = self.dataset[index]

        if self.preprocess:
            # if we preprocessed, the dataset items are already tokenized
            return TokenizedBatch(
                input_ids=torch.tensor(item["input_ids"]),
                attention_mask=torch.tensor(item["attention_mask"]),
            )

        # otherwise, we need to tokenize on the fly
        text = item["text"]
        tokens = self.tokenizer(text)
        return TokenizedBatch(
            input_ids=tokens["input_ids"].squeeze(),
            attention_mask=tokens["attention_mask"].squeeze(),
        )


if __name__ == "__main__":
    from src.datasets import DGigawordDataset

    class DummyDataset(BaseTextDataset):
        def __init__(self):
            super().__init__()
            self.name = "dummy"
            self.data = ["This is a test." for _ in range(10_000)]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index: int) -> dict:
            return {"text": self.data[index]}

    # textdatasets = [DummyDataset() for _ in range(5)]
    textdatasets = [DGigawordDataset() for _ in range(2)]
    tokenizer = Tokenizer("distilbert/distilgpt2")
    tokenized_dataset = TokenizedDataset(textdatasets, tokenizer, preprocess=True)
    sample = tokenized_dataset[0]
    print(sample)
