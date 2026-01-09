import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from pathlib import Path
import os
import hydra
from omegaconf import DictConfig


load_dotenv()
MODEL_DIR = Path(os.getenv("MODEL_PATH")) / "pretrained"
os.environ["HYDRA_FULL_ERROR"] = "1"


class HestNet(nn.Module):
    """HestNet."""

    def __init__(
        self,
        checkpoint: str,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=MODEL_DIR / checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=MODEL_DIR / checkpoint)

    def tokenize_function(self, elem):
        print(elem)
        return self.tokenizer(elem["text"], truncation=True, return_tensors="pt")

    def get_tokinizer(checkpoint):
        return AutoTokenizer.from_pretrained(checkpoint)

    def forward(self, inputs: dict, **kwargs) -> Tensor:
        encoded_input = self.tokenize_function(inputs)  # contains "input_ids" and "attention_mask"
        outputs = self.model(**encoded_input, labels=encoded_input["input_ids"], **kwargs)
        return outputs


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Example
    checkpoint = cfg.model.network.checkpoint
    model = HestNet(checkpoint)
    text = "Jeg bor i et kommodeskab"
    outputs = model({"text": text})
    print(f"Model: {checkpoint}")
    print(f"Input: {text}")
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}")

    # Equivalent example
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=MODEL_DIR / checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=MODEL_DIR / checkpoint)
    inputs = tokenizer("Jeg bor i et kommodeskab", return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])
    print(f"Logits shape: {outputs.logits.shape}")


if __name__ == "__main__":
    # Run with python ./src/networks/hestnet.py +experiment=hestnet
    main()
