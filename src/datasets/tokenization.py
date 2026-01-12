from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Optional
from src import TensorDict, TokenizedSample
from src.datasets.basedataset import BaseDataset
from src.datasets.textdataset import TextDataset


class Tokenizer:
    def __init__(
        self,
        checkpoint: str,
        max_length: Optional[int] = None,
    ):
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


class TokenizedDataset(BaseDataset):
    def __init__(
        self,
        dataset: TextDataset,
        tokenizer: Tokenizer,
    ):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> TokenizedSample:
        text = self.dataset[index]["text"]
        tokens = self.tokenizer(text)
        return TokenizedSample(
            text=text,
            input_ids=tokens["input_ids"].squeeze(),
            attention_mask=tokens["attention_mask"].squeeze(),
        )
