import itertools
import pytest
import torch

from src import TextSample
from src.datasets import DummyDataset, DGigawordDataset, Tokenizer, TokenizedDataset
from src.data_modules import BaseDM

pytestmark = pytest.mark.unit


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


def _validate_tokenizer_output_batched(data, tokenizer, expected_batch_size=1):
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


def _validate_tokenizer_output_squeezed(data, Tdataset):
    """Helper to validate tokenized dataset output structure."""
    assert "input_ids" in data, "Data dictionary should contain 'input_ids' key"
    assert "attention_mask" in data, "Data dictionary should contain 'attention_mask' key"
    assert data["input_ids"].ndim == 1, "input_ids shape should be (x)"
    assert (
        data["input_ids"].shape[0] <= Tdataset.tokenizer.max_length
    ), f"input_ids shape should be (x <={Tdataset.tokenizer.max_length}), got {data['input_ids'].shape}"
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

    for sample in itertools.islice(dataset, 5):
        _validate_raw_sample(sample)


@pytest.mark.parametrize("checkpoint", ["distilbert/distilgpt2"])
@pytest.mark.parametrize("sample", ["Jeg bor i et kommodeskab"])
def test_tokenizer(checkpoint: str, sample: str):
    """Test tokenizer processes text correctly."""
    tokenizer = Tokenizer(checkpoint)
    assert (
        tokenizer.max_length <= tokenizer.tokenizer.model_max_length
    ), f"Tokenizer max length {tokenizer.max_length} should be less than or equal to model max length {tokenizer.tokenizer.model_max_length}"
    assert (
        tokenizer.tokenizer.pad_token
    ), f"Tokenizer should contain a pad_token but found {tokenizer.tokenizer.pad_token}"
    data = tokenizer(sample)
    _validate_tokenizer_output_batched(data, tokenizer)
    _validate_tokenized_sample(data)
    decoded_data = tokenizer.batch_decode(data["input_ids"])
    assert isinstance(decoded_data[0], str), f"Decoded data should be list[str] but got {type(decoded_data)}"
    assert decoded_data[0] == sample, f"Decoded data should recover the sample but got {decoded_data[0]=} != {sample=}"


@pytest.mark.parametrize("checkpoint", ["distilbert/distilgpt2"])
def test_dgigaword_dataset_and_tokenizer(checkpoint):
    """Test raw DGigawordDataset combined with tokenizer."""
    dataset = DGigawordDataset()
    tokenizer = Tokenizer(checkpoint)

    for i in _get_random_indices(len(dataset), n_samples=5):
        sample = dataset[i]["text"]
        data = tokenizer(sample)
        _validate_tokenizer_output_batched(data, tokenizer)
        _validate_tokenized_sample(data)


@pytest.mark.parametrize("checkpoint", ["distilbert/distilgpt2"])
def test_tokenized_dataset(checkpoint, dataset_counts, N_TRAIN):
    """Test tokenized dataset"""
    datasets = dataset_counts
    tokenizer = Tokenizer(checkpoint)
    Tdataset = TokenizedDataset(datasets, tokenizer)
    assert len(Tdataset) == N_TRAIN * len(
        datasets
    ), f"Dataset length should be {N_TRAIN * len(datasets)}, got {len(Tdataset)}"

    for i in _get_random_indices(len(Tdataset), n_samples=5):
        data = Tdataset[i]
        _validate_tokenizer_output_squeezed(data, Tdataset)
        _validate_tokenized_sample(data)


@pytest.mark.parametrize("checkpoint", ["distilbert/distilgpt2"])
@pytest.mark.parametrize("train_val_split", [None, 0.95, 0.1])
def test_datamodule(checkpoint, dataset_counts, train_val_split):
    """Test datamodule compatibility"""
    datasets = dataset_counts
    tokenizer = Tokenizer(checkpoint)
    # It would be nice if the datamodule stored the tokenizer as well.
    # At the moment it is overwritten when the dataset is split into subsets.
    datamodule = BaseDM(
        TokenizedDataset(datasets, tokenizer),
        train_val_split=train_val_split,
    )

    for data in itertools.islice(datamodule.train_dataloader(), 5):
        _validate_tokenizer_output_batched(data, tokenizer)
        _validate_tokenized_sample(data)

    if datamodule.valset:
        for data in itertools.islice(datamodule.val_dataloader(), 5):
            _validate_tokenizer_output_batched(data, tokenizer)
            _validate_tokenized_sample(data)

    if datamodule.testset:
        for data in itertools.islice(datamodule.test_dataloader(), 5):
            _validate_tokenizer_output_batched(data, tokenizer)
            _validate_tokenized_sample(data)


if __name__ == "__main__":
    checkpoint = "distilbert/distilgpt2"
    sample = "Jeg bor i et kommodeskab"
    # test_tokenizer(checkpoint, sample)
    # test_dgigaword_dataset_and_tokenizer(checkpoint)
    test_datamodule(checkpoint, [DGigawordDataset()], 1)
