import torch.nn as nn
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
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

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> CausalLMOutputWithCrossAttentions:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,  # HuggingFace automatically computes loss when labels are provided. HuggingFace also automatically shifts the inputs for causal language modeling.
        )
        return output


if __name__ == "__main__":
    from src.datasets.gigaword import DGigawordDataset
    from src.datasets.tokenization import TokenizedDataset
    from src.datasets.tokenization import Tokenizer
    from src.callbacks.utils import get_batch_from_dataset

    tokenizer = Tokenizer(checkpoint="distilbert/distilgpt2")
    text_dataset = DGigawordDataset()
    dataset = TokenizedDataset(
        datasets=[text_dataset],
        tokenizer=tokenizer,
    )

    batch = get_batch_from_dataset(dataset, batch_size=1)
    model = CausalTransformer(checkpoint="distilbert/distilgpt2")
    output = model.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    print(output)
    print(output.keys())
    print(output.loss)
