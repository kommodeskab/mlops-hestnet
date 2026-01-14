import torch.nn as nn
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch import Tensor

load_dotenv()


class CausalTransformer(nn.Module):
    """
    A causal language model wrapper based on a pretrained transformer from HuggingFace.

    Args:
        checkpoint (str): The HuggingFace checkpoint name or path.
    """

    def __init__(
        self,
        checkpoint: str,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            checkpoint
        )  # Note that this is cached in the global hugging face cache
        self.model.train()  # eval mode by default, but we want train mode

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> CausalLMOutputWithCrossAttentions:
        """
        The forward pass of the causal transformer.

        Args:
            input_ids (Tensor): Input token IDs for the causal language model.
            attention_mask (Tensor): Attention mask indicating which tokens should be attended to (1) and which should be ignored (0).

        Returns:
            CausalLMOutputWithCrossAttentions: The output from the network's forward pass.
        """
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
