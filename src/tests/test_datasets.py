import torch
import pytest
from src.datasets import DummyDataset
from src.datasets import DGigawordDataset, TDGigawordDataset
from src.datasets.utils import get_tokenize_function
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

SEED = 42
torch.manual_seed(SEED)
N_TRAIN = 471550 

# here is an example test function
def test_dummy_dataset():
    """Test dummy dataset returns correct shapes and keys."""
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


def _validate_raw_sample(sample, checkpoint):
    """Helper to validate raw text samples."""
    assert "text" in sample, "Raw sample should contain 'text' key"
    assert isinstance(sample["text"], str), "Text should be a string"
    
    # Tokenize and validate
    tokenize_function = get_tokenize_function(checkpoint)
    data = tokenize_function(sample)
    assert "input_ids" in data, "Data dictionary should contain 'input_ids' key"
    assert "attention_mask" in data, "Data dictionary should contain 'attention_mask' key"
    assert data["input_ids"].ndim == 2, "input_ids shape should be (1, x)"
    assert data["input_ids"].shape[0] == 1 and data["input_ids"].shape[1] <= 1024, (
        f"input_ids shape should be (1, <=1024), got {data['input_ids'].shape}"
    )
    assert data["input_ids"].shape == data["attention_mask"].shape, (
        f"input_ids and attention_mask shapes must match, got {data['input_ids'].shape} vs {data['attention_mask'].shape}"
    )


def _validate_tokenized_sample(sample):
    """Helper to validate tokenized samples."""
    assert "input_ids" in sample, "Tokenized sample should contain 'input_ids' key"
    assert "attention_mask" in sample, "Tokenized sample should contain 'attention_mask' key"
    assert "labels" in sample, "Tokenized sample should contain 'labels' key"

    # Preprocessed data should be tensors
    assert isinstance(sample["input_ids"], torch.Tensor), (
        f"Preprocessed input_ids should be torch.Tensor, got {type(sample['input_ids'])}"
    )
    assert isinstance(sample["attention_mask"], torch.Tensor), (
        f"Preprocessed attention_mask should be torch.Tensor, got {type(sample['attention_mask'])}"
    )
    if "labels" in sample:
        assert isinstance(sample["labels"], torch.Tensor), (
            f"Preprocessed labels should be torch.Tensor, got {type(sample['labels'])}"
        )


@pytest.mark.parametrize("size", [None, 1, 69, 100])
def test_dgigaword_dataset(size):
    """Test raw DGigawordDataset returns text samples."""
    dataset = DGigawordDataset(size)
    if size is None: size = N_TRAIN
    assert len(dataset) == size, f"Dataset length should be {size}, got {len(dataset)}"
    assert "text" in dataset.column_names, f"Dataset should have 'text' column, got {dataset.column_names}"
    
    for i in _get_random_indices(size, n_samples=5):
        _validate_raw_sample(dataset[i], "distilbert/distilgpt2")


@pytest.mark.parametrize("checkpoint", ["distilbert/distilgpt2"])
@pytest.mark.parametrize("size", [None, 1,  69, 1000]) 
@pytest.mark.parametrize("preprocess", [False, True])
@pytest.mark.parametrize("num_proc", [1, 4])
def test_tdgigaword_dataset(checkpoint, size, preprocess, num_proc):
    """Test TDGigawordDataset with various configurations."""
    if size is None and preprocess:
        pytest.skip("Skipping size=None with preprocess=True (OOM issues)")
    dataset = TDGigawordDataset(
        checkpoint=checkpoint,
        size=size,
        preprocess=preprocess,
        num_proc=num_proc
    )
    if size is None: size = N_TRAIN
    assert len(dataset) == size, f"Dataset length should be {size}, got {len(dataset)}"
    assert dataset.preprocessed == preprocess, (
        f"Dataset preprocessed state should be {preprocess}, got {dataset.preprocessed}"
    )
    
    # Test samples
    for i in _get_random_indices(size, n_samples=5):
        sample = dataset[i]
        _validate_tokenized_sample(sample)

@pytest.mark.parametrize("checkpoint", ["distilbert/distilgpt2"])
@pytest.mark.parametrize("size", [1, 69, 1000])
@pytest.mark.parametrize("preprocess", [False, True])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_tdgigaword_dataloader(checkpoint, size, preprocess, batch_size):
    """Test TDGigawordDataset works with DataLoader and DataCollator."""

    dataset = TDGigawordDataset(checkpoint=checkpoint, size=size, preprocess=preprocess)
    data_collator = DataCollatorForLanguageModeling(tokenizer=dataset.tokenizer, mlm=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator, num_workers = 0)
    
    batch = next(iter(dataloader))
    assert "input_ids" in batch, f"Batch should contain 'input_ids', got keys: {batch.keys()}"
    assert "attention_mask" in batch, f"Batch should contain 'attention_mask', got keys: {batch.keys()}"
    assert "labels" in batch, f"Batch should contain 'labels', got keys: {batch.keys()}"
    assert batch["input_ids"].shape[0] == min(batch_size, size), (
        f"Batch size should be 4, got {batch['input_ids'].shape[0]}"
    )
    assert batch["input_ids"].shape == batch["labels"].shape, (
        f"input_ids and labels shapes must match, got {batch['input_ids'].shape} vs {batch['labels'].shape}"
    )


if __name__ == "__main__":
    test_dummy_dataset()
    test_dgigaword_dataset("distilbert/distilgpt2", None)
    test_tdgigaword_dataloader("distilbert/distilgpt2", 100, 32)
