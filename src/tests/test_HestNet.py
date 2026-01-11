import pytest
import torch

from src.data_modules import TDGigawordDM
from src.datasets import TDGigawordDataset
from src.lightning_modules import HestnetModule
from src.networks import HestNet


@pytest.mark.parametrize("checkpoint", ["distilbert/distilgpt2"])
def test_HestNet(checkpoint):
    """Test HestNet model initialization and forward pass."""
    model = HestNet(checkpoint)
    text = "Jeg bor i et kommodeskab"
    inputs = model.tokenizer(text, return_tensors="pt")
    outputs = model(inputs, labels=inputs["input_ids"])

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
    inputs = model.tokenizer(text2, return_tensors="pt")
    outputs2 = model(inputs, labels=inputs["input_ids"])
    assert hasattr(outputs2, "loss"), "Model should work with different inputs"
    assert not torch.isnan(outputs2.loss), "Loss should be valid for different inputs"





@pytest.mark.parametrize("checkpoint", ["distilbert/distilgpt2"])
def test_HestnetModule(checkpoint):
    """Test HestnetModule Lightning module initialization and forward pass."""
    network = HestNet(checkpoint=checkpoint)
    
    module = HestnetModule(
        network=network,
        optimizer=torch.optim.AdamW,
        lr_scheduler={
            "scheduler": torch.optim.lr_scheduler.ConstantLR,
            "monitor": "val_loss",
            "interval": "step",
            "frequency": 1,
        }
    )
    
    assert module.network is not None, "Network should be initialized"
    assert module.partial_optimizer is not None, "Optimizer should be set"
    assert module.partial_lr_scheduler is not None, "LR scheduler should be set"
    
    # Test forward pass
    text = "Jeg bor i et kommodeskab"
    inputs = network.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    batch = {**inputs, "labels": inputs["input_ids"]}
    
    output = module.forward(batch)
    assert hasattr(output, "loss"), "Forward output should contain 'loss' attribute"
    assert hasattr(output, "logits"), "Forward output should contain 'logits' attribute"
    
    # Test common_step
    step_output = module.common_step(batch, batch_idx=0)
    assert "loss" in step_output, "Step output should contain 'loss' attribute"
    assert isinstance(step_output.get("loss"), torch.Tensor), f"Loss should be a Tensor, got {type(step_output.loss)}"
    assert step_output.get("loss").item() >= 0, f"Loss should be non-negative, got {step_output.loss.item():.4f}"
    assert not torch.isnan(step_output.get("loss")), "Step output loss should not be NaN"


@pytest.mark.parametrize("checkpoint,size", [("distilbert/distilgpt2", 20)])
def test_HestnetModule_with_datamodule(checkpoint, size):
    """Test HestnetModule integration with TDGigawordDM."""
    # Create dataset and datamodule
    dataset = TDGigawordDataset(checkpoint=checkpoint, size=size, preprocess=True)
    dm = TDGigawordDM(
        trainset=dataset,
        train_val_split=0.8,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    
    # Create model
    network = HestNet(checkpoint=checkpoint)
    module = HestnetModule(
        network=network,
        optimizer=torch.optim.AdamW,
        lr_scheduler={
            "scheduler": torch.optim.lr_scheduler.ConstantLR,
            "monitor": "val_loss",
            "interval": "step",
            "frequency": 1,
        }
    )
    
    # Test with train batch
    train_loader = dm.train_dataloader()
    train_batch = next(iter(train_loader))
    
    train_output = module.common_step(train_batch, batch_idx=0)
    assert train_output["loss"] is not None, "Train step should return a loss"
    assert not torch.isnan(train_output["loss"]), "Train loss should not be NaN"
    
    # Test with val batch
    val_loader = dm.val_dataloader()
    val_batch = next(iter(val_loader))
    
    val_output = module.common_step(val_batch, batch_idx=0)
    assert val_output["loss"] is not None, "Val step should return a loss"
    assert not torch.isnan(val_output["loss"]), "Val loss should not be NaN"


if __name__ == "__main__":
    test_HestNet("distilbert/distilgpt2")
    test_HestnetModule("distilbert/distilgpt2")
    test_HestnetModule_with_datamodule("distilbert/distilgpt2", 20)
