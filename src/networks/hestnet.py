import os
from pathlib import Path

import hydra
import torch.nn as nn
from dotenv import load_dotenv
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from src import TensorDict

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
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=MODEL_DIR / checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=MODEL_DIR / checkpoint)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, inputs, **generation_kwargs):
        """Generate text embeddings from input.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **generation_kwargs: Arguments for generate (max_new_tokens, do_sample, etc.)

        """
        generation_params = {
            "max_new_tokens": 100,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        generation_params.update(generation_kwargs)

        return self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **generation_params,
        )

    def decode(self, outputs):
        """Decode text embeddings."""
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def forward(self, inputs: TensorDict, raw_text=False, **kwargs) -> CausalLMOutputWithCrossAttentions:
        if raw_text:  # Nice functionality to have, but in the long run I do not want to mix input types.
            inputs = self.tokenizer(inputs, return_tensors="pt")
            inputs = inputs | {"labels": inputs["input_ids"]}
        if "labels" not in inputs and "labels" not in kwargs:
            msg = "inputs must contain 'labels' dict or labels parameter must be passed explicitly"
            raise ValueError(msg)
        return self.model(**inputs, **kwargs)  # inputs contains "input_ids", "attention_mask" and "labels"


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Example
    prompt = "Jeg bor i et kommodeskab "
    checkpoint = cfg.model.network.checkpoint
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=MODEL_DIR / checkpoint)
    model = HestNet(checkpoint)
    inputs = model.tokenizer(prompt, return_tensors="pt")
    print(inputs)
    outputs = model(inputs, labels=inputs["input_ids"])
    print(f"Model: {checkpoint}")
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}")
    print(type(outputs))
    print(type(outputs.loss))
    print(type(outputs.loss.item()))

    outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    response = model.decode(outputs)
    print(response)

    # Temporary functionality allowing for passing prompts directly. Enjoy while it lasts.
    model(prompt, raw_text=True)
    response = model.decode(outputs)
    print(response)

    # Equivalent example
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=MODEL_DIR / checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=MODEL_DIR / checkpoint)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])
    print(f"Logits shape: {outputs.logits.shape}")


if __name__ == "__main__":
    # Run with python ./src/networks/hestnet.py +experiment=hestnet
    main()
