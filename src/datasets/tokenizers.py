from transformers import AutoTokenizer
from typing import Optional
from src import TensorDict

class Tokenizer:
    def __init__(
        self,
        checkpoint: str,
        max_length: Optional[int] = None,
    ):
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.max_length = max_length or self.tokenizer.model_max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def __call__(self, string: str) -> TensorDict:
        return self.tokenizer(
            string,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )