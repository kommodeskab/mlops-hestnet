import pytest
import torch
from src.networks import HestNet


@pytest.mark.parametrize("checkpoint", ["distilbert/distilgpt2"])
def test_HestNet(checkpoint):
    """Test HestNet model initialization and forward pass."""
    model = HestNet(checkpoint)
    text = "Jeg bor i et kommodeskab"
    outputs = model(text)

    # Test that outputs contain required attributes
    assert hasattr(outputs, "loss"), "Model outputs should contain 'loss' attribute"
    assert hasattr(outputs, "logits"), "Model outputs should contain 'logits' attribute"

    # Test loss properties
    assert isinstance(outputs.loss, torch.Tensor), "Loss should be a torch.Tensor"
    assert outputs.loss.ndim == 0 or (outputs.loss.ndim == 1 and outputs.loss.shape[0] == 1), (
        f"Loss should be a scalar, got shape {outputs.loss.shape}"
    )
    assert outputs.loss.item() >= 0, f"Loss should be non-negative, got {outputs.loss.item():.4f}"
    assert not torch.isnan(outputs.loss), "Loss should not be NaN"
    assert not torch.isinf(outputs.loss), "Loss should not be infinite"

    # Test logits properties
    assert isinstance(outputs.logits, torch.Tensor), "Logits should be a torch.Tensor"
    assert outputs.logits.ndim == 3, f"Logits should be 3D (batch, seq_len, vocab_size), got {outputs.logits.ndim}D"
    assert outputs.logits.shape[0] == 1, f"Batch size should be 1, got {outputs.logits.shape[0]}"
    assert outputs.logits.shape[1] > 0, f"Sequence length should be positive, got {outputs.logits.shape[1]}"
    assert outputs.logits.shape[2] > 0, f"Vocab size should be positive, got {outputs.logits.shape[2]}"

    # Test with different input
    text2 = "Dette er en test"
    outputs2 = model(text2)
    assert hasattr(outputs2, "loss"), "Model should work with different inputs"
    assert not torch.isnan(outputs2.loss), "Loss should be valid for different inputs"
