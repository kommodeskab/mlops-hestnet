import pytest
import torch

from src import TextSample
from src.datasets import DummyDataset, DGigawordDataset, Tokenizer


# here is an example test function
def test_dummy_dataset():
    size = 10
    dataset = DummyDataset(size=size)
    assert len(dataset) == size, f"Dataset length should be {size}"
    for i in range(size):
        data = dataset[i]
        assert "input" in data
        assert "target" in data
        assert data["input"].shape == (10,), "Input shape should be (10,)"
        assert data["target"].shape == (1,), "Target shape should be (1,)"


def _get_random_indices(size, n_samples=10):
    """Helper to get random test indices including edge cases."""
    n_samples = min(n_samples, size)
    idx = torch.randperm(size)[:n_samples].tolist()
    return [0] + idx + [size - 1]


def _validate_raw_sample(sample: TextSample):
    """Helper to validate raw text samples."""
    assert "text" in sample, "Raw sample should contain 'text' key"
    assert isinstance(sample["text"], str), "Text should be a string"


def _validate_tokenizer_output(data, tokenizer, expected_batch_size=1):
    """Helper to validate tokenizer output structure."""
    assert "input_ids" in data, "Data dictionary should contain 'input_ids' key"
    assert "attention_mask" in data, "Data dictionary should contain 'attention_mask' key"
    assert data["input_ids"].ndim == 2, "input_ids shape should be (1, x)"
    assert (
        data["input_ids"].shape[0] == expected_batch_size and data["input_ids"].shape[1] <= tokenizer.max_length
    ), f"input_ids shape should be ({expected_batch_size}, <={tokenizer.max_length}), got {data['input_ids'].shape}"
    assert (
        data["input_ids"].shape == data["attention_mask"].shape
    ), f"input_ids and attention_mask shapes must match, got {data['input_ids'].shape} vs {data['attention_mask'].shape}"


def _validate_tokenized_sample(data):
    """Helper to validate tokenized samples."""
    assert "input_ids" in data, "Tokenized sample should contain 'input_ids' key"
    assert "attention_mask" in data, "Tokenized sample should contain 'attention_mask' key"
    # assert "labels" in data, "Tokenized sample should contain 'labels' key"

    # Preprocessed data should be tensors
    assert isinstance(
        data["input_ids"], torch.Tensor
    ), f"Preprocessed input_ids should be torch.Tensor, got {type(data['input_ids'])}"
    assert isinstance(
        data["attention_mask"], torch.Tensor
    ), f"Preprocessed attention_mask should be torch.Tensor, got {type(data['attention_mask'])}"
    if "labels" in data:
        assert isinstance(
            data["labels"], torch.Tensor
        ), f"Preprocessed labels should be torch.Tensor, got {type(data['labels'])}"


@pytest.mark.parametrize("checkpoint", ["distilbert/distilgpt2"])
def test_dgigaword_dataset(checkpoint, N_TRAIN):
    """Test raw DGigawordDataset returns text samples."""
    dataset = DGigawordDataset()
    assert len(dataset) == N_TRAIN, f"Dataset length should be {N_TRAIN}, got {len(dataset)}"
    assert "text" in dataset.column_names, f"Dataset should have 'text' column, got {dataset.column_names}"

    for i in _get_random_indices(len(dataset), n_samples=5):
        _validate_raw_sample(dataset[i])


@pytest.mark.parametrize("checkpoint", ["distilbert/distilgpt2"])
@pytest.mark.parametrize("sample", [TextSample(text="Jeg bor i et kommodeskab")])
def test_tokenizer(checkpoint: str, sample: TextSample):
    """Test tokenizer processes text correctly."""
    tokenizer = Tokenizer(checkpoint)
    data = tokenizer(sample["text"])
    _validate_tokenizer_output(data, tokenizer)
    _validate_tokenized_sample(data)


@pytest.mark.parametrize("checkpoint", ["distilbert/distilgpt2"])
def test_dgigaword_dataset_and_tokenizer(checkpoint):
    """Test raw DGigawordDataset combined with tokenizer."""
    dataset = DGigawordDataset()
    tokenizer = Tokenizer(checkpoint)

    for i in _get_random_indices(len(dataset), n_samples=5):
        sample = dataset[i]["text"]
        data = tokenizer(sample)
        _validate_tokenizer_output(data, tokenizer)
        _validate_tokenized_sample(data)


if __name__ == "__main__":
    checkpoint = "distilbert/distilgpt2"
    sample = TextSample(text="Jeg bor i et kommodeskab")
    test_tokenizer(checkpoint, sample)
    test_dgigaword_dataset_and_tokenizer(checkpoint)
