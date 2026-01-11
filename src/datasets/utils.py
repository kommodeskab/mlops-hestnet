from transformers import AutoTokenizer

def get_tokenize_function(checkpoint: str, **tokenizer_kwargs):
    """Create a tokenize function for the given checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(string: str) -> dict:
        return tokenizer(string, padding=True, truncation=True, return_tensors="pt", **tokenizer_kwargs)

    return tokenize_function