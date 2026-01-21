import pytest
import torch

from src.datasets.tokenization import TokenizedDataset
from src.datasets.tokenization import Tokenizer
from src.callbacks.utils import get_batch_from_dataset
from src.networks import CausalTransformer

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("checkpoint", ["distilbert/distilgpt2"])
@pytest.mark.parametrize("batch_size", [1, 2])  # 2 is max
def test_causal_transformer(checkpoint, dataset_counts, batch_size):
    datasets = dataset_counts
    tokenizer = Tokenizer(checkpoint)
    Tdataset = TokenizedDataset(datasets, tokenizer)

    batch = get_batch_from_dataset(Tdataset, batch_size=batch_size)
    model = CausalTransformer(checkpoint)
    output = model.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

    # Test that outputs contain required attributes
    assert hasattr(output, "loss"), "Model outputs should contain 'loss' attribute"
    assert hasattr(output, "logits"), "Model outputs should contain 'logits' attribute"

    # Test loss properties
    assert isinstance(output.loss, torch.Tensor), "Loss should be a torch.Tensor"
    assert output.loss.ndim == 0 or (
        output.loss.ndim == 1 and output.loss.shape[0] == 1
    ), f"Loss should be a scalar, got shape {output.loss.shape}"
    assert output.loss.item() >= 0, f"Loss should be non-negative, got {output.loss.item():.4f}"
    assert not torch.isnan(output.loss), "Loss should not be NaN"
    assert not torch.isinf(output.loss), "Loss should not be infinite"

    # Test logits properties
    assert isinstance(output.logits, torch.Tensor), "Logits should be a torch.Tensor"
    assert output.logits.ndim == 3, f"Logits should be 3D (batch, seq_len, vocab_size), got {output.logits.ndim}D"
    assert output.logits.shape[0] == batch_size, f"Batch size should be {batch_size}, got {output.logits.shape[0]}"
    assert output.logits.shape[1] > 0, f"Sequence length should be positive, got {output.logits.shape[1]}"
    assert output.logits.shape[2] > 0, f"Vocab size should be positive, got {output.logits.shape[2]}"
