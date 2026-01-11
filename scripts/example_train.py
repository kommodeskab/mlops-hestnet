"""Minimal example of training HestNet on TDGigawordDataset with PyTorch Lightning.

This script demonstrates:
- Loading a pretrained HuggingFace model (DistilGPT2)
- Fine-tuning on the Danish Gigaword dataset
- Using PyTorch Lightning for the training loop
- Transfer learning with causal language modeling
"""

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.data_modules import TDGigawordDM
from src.datasets import TDGigawordDataset
from src.lightning_modules import HestnetModule
from src.networks import HestNet


def monitor_gpu():
    # Monitor GPU memory
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")


def main():
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    pl.seed_everything(42)
    monitor_gpu()

    # Configuration
    checkpoint = "distilbert/distilgpt2"  # Pretrained model to fine-tune
    dataset_size = 1000  # Small subset for quick testing
    batch_size = 2
    max_epochs = 3
    learning_rate = 5e-5

    # Create dataset
    print(f"Loading dataset ({dataset_size} samples)...")
    trainset = TDGigawordDataset(
        checkpoint=checkpoint,
        size=dataset_size,
        preprocess=True,  # Tokenize on-the-fly
        num_proc=4,
    )

    # Create data module with train/val split
    print("Creating data module...")
    datamodule = TDGigawordDM(
        trainset=trainset,
        train_val_split=0.9,  # 90% train, 10% val
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # Set to 0 for debugging, increase for speed
    )

    # Create model
    print(f"Loading pretrained model: {checkpoint}")
    network = HestNet(checkpoint=checkpoint)

    # Create Lightning module with optimizer
    print("Creating Lightning module...")
    model = HestnetModule(
        network=network,
        optimizer=torch.optim.AdamW,
        lr_scheduler={
            "scheduler": torch.optim.lr_scheduler.ConstantLR,
            "monitor": "val_loss",
            "interval": "step",
            "frequency": 1,
        },
    )

    # Manually configure optimizer with learning rate
    # (normally done via Hydra config, but here we do it explicitly)
    model.partial_optimizer = lambda params: torch.optim.AdamW(params, lr=learning_rate)
    model.partial_lr_scheduler = {
        "scheduler": lambda optimizer: torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1),
        "monitor": "val_loss",
        "interval": "step",
        "frequency": 1,
    }

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    # Setup logger (optional - set offline=True to disable wandb)
    logger = WandbLogger(
        project="hestnet-example",
        name="minimal-training",
        offline=True,  # Set to False to log to wandb
    )

    # Create trainer
    print("Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",  # Automatically use GPU if available
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
    )

    # Train the model
    print("Starting training...")
    monitor_gpu()
    trainer.fit(model, datamodule)

    # Test generation after training
    print("\nTesting text generation...")
    model.eval()
    network.eval()

    prompt = "Jeg bor i et kommodeskab"
    inputs = network.tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        generated = network.generate(
            inputs,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )

    generated_text = network.decode(generated)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text[0]}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
