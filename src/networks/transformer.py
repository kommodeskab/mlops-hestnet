import torch.nn as nn
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, PreTrainedModel
from torch import Tensor

load_dotenv()


class CausalTransformer(nn.Module):
    def __init__(
        self,
        checkpoint: str,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(checkpoint)
        self.model.train()

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output["logits"]


if __name__ == "__main__":
    from src.datasets.gigaword import DGigawordDataset
    from src.callbacks.utils import get_batch_from_dataset

    dataset = DGigawordDataset("distilbert/distilgpt2")
    batch = get_batch_from_dataset(dataset, batch_size=1)
    model = CausalTransformer(checkpoint="distilbert/distilgpt2")
    output = model.forward(batch)
    print(output["output"].shape)
