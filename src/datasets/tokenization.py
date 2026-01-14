from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Optional
from src import TensorDict, TokenizedBatch
from src.datasets.basedataset import BaseDataset
from src.datasets.textdataset import BaseTextDataset
from torch.utils.data import ConcatDataset


class Tokenizer:
    """
    Wrapper around a HuggingFace tokenizer for encoding text into tokens.

    Handles tokenization of strings or lists of strings with configurable
    padding, truncation, and left-side padding for causal language models.

    Args:
        checkpoint (str): The HuggingFace checkpoint name or path for the tokenizer.
        max_length (Optional[int]): Maximum sequence length for tokenization.
    """

    def __init__(
        self,
        checkpoint: str,
        max_length: Optional[int] = None,
    ):
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(checkpoint)
        self.max_length = max_length or self.tokenizer.model_max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, string: str | list[str]) -> TensorDict:
        """
        Tokenize input string(s) into token tensors.

        Args:
            string (str | list[str]): A single string or list of strings to tokenize.

        Returns:
            TensorDict: Dictionary containing 'input_ids' and 'attention_mask' as PyTorch tensors.
        """
        return self.tokenizer(
            string,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding_side="left",
        )

    def batch_decode(self, token_ids: list[list[int]]) -> list[str]:
        """
        Decode token IDs back into text strings.

        Args:
            token_ids (list[list[int]]): Batch of token ID sequences.

        Returns:
            list[str]: Decoded text strings with special tokens removed.
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


class TokenizedDataset(BaseDataset):
    """
    Dataset that tokenizes text on-the-fly using a provided tokenizer.

    Combines multiple text datasets and applies tokenization to each sample,
    returning tokenized batches with input IDs and attention masks.

    Args:
        datasets (list[BaseTextDataset]): List of text datasets to combine.
        tokenizer (Tokenizer): Tokenizer to use for tokenizing text samples.
    """

    def __init__(
        self,
        datasets: list[BaseTextDataset],
        tokenizer: Tokenizer,
    ):
        super().__init__()
        self.dataset = ConcatDataset(datasets)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> TokenizedBatch:
        text = self.dataset[index]["text"]
        tokens = self.tokenizer(text)
        return TokenizedBatch(
            input_ids=tokens["input_ids"].squeeze(),
            attention_mask=tokens["attention_mask"].squeeze(),
        )
