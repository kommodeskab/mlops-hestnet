import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from pathlib import Path
import os
import hydra
from omegaconf import DictConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


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
    



    # @property
    # def model(self):
    #     return self.model

    # def tokenize_function(self, elem):
    #     print(elem)
    #     return self.tokenizer(elem, truncation=True, return_tensors="pt")

    # def _get_tokinizer(self):
    #     return self.tokenizer

    def forward(self, inputs, **kwargs) -> CausalLMOutputWithCrossAttentions:
        # encoded_input = self.tokenize_function(inputs)  # contains "input_ids", "attention_mask" and "labels"
        outputs = self.model(**inputs, **kwargs)
        return outputs
    #: dict[str: Tensor]


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Example
    prompt = "Jeg bor i et kommodeskab "
    checkpoint = cfg.model.network.checkpoint
    #tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=MODEL_DIR / checkpoint)
    model = HestNet(checkpoint)
    inputs = model.tokenizer(prompt, return_tensors="pt")
    print(inputs)
    outputs = model(inputs, labels=inputs["input_ids"])  
    print(f"Model: {checkpoint}")
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}")
    print(type(outputs))

    outputs = model.model.generate(inputs.input_ids, attention_mask= inputs.attention_mask, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    response = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(response)
    # answer_start_index = outputs.start_logits.argmax()
    # answer_end_index = outputs.end_logits.argmax()
    # predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    # tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

    # Equivalent example
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=MODEL_DIR / checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=MODEL_DIR / checkpoint)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])
    print(f"Logits shape: {outputs.logits.shape}")


if __name__ == "__main__":
    # Run with python ./src/networks/hestnet.py +experiment=hestnet
    main()
