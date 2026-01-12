from src.lightning_modules import BaseLightningModule
from src import OptimizerType, LRSchedulerType, TokenizedBatch, StepOutput
from src.networks import CausalTransformer
from torch import Tensor
from src.datasets import Tokenizer
import torch
from src import LossOutput


class CausalLLM(BaseLightningModule):
    """A dummy LightningModule for testing purposes."""

    def __init__(
        self,
        network: CausalTransformer,
        tokenizer: Tokenizer,
        optimizer: OptimizerType = None,
        lr_scheduler: LRSchedulerType = None,
    ):
        # optimizer and lr_scheduler can be None, for example during testing
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.tokenizer = tokenizer
        self.network = network

    def forward(self, batch: TokenizedBatch):
        return self.network.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

    def common_step(self, batch: TokenizedBatch, batch_idx: int) -> StepOutput:
        output = self.forward(batch)
        loss = output.loss
        return StepOutput(
            loss=loss,  # we take gradients on this
            loss_output=LossOutput(
                loss=loss
            ),  # we log this. the loss_output can also contain other things, like metrics
            model_output=output,
        )

    @torch.no_grad()
    def generate(self, text: list[str], **kwargs) -> list[str]:
        generation_params = {
            "max_new_tokens": 100,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
        }
        generation_params.update(kwargs)

        inputs = self.tokenizer(text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs: Tensor = self.network.model.generate(**inputs, **generation_params)
        decoded = self.tokenizer.batch_decode(outputs.tolist())
        return decoded


if __name__ == "__main__":
    from src.networks import CausalTransformer
    from src.datasets import Tokenizer

    tokenizer = Tokenizer("distilbert/distilgpt2")
    network = CausalTransformer("distilbert/distilgpt2")

    llm = CausalLLM(
        network=network,
        tokenizer=tokenizer,
    )

    text = ["Dette er en test sætning.", "Dette er en anden sætning."]
    generations = llm.generate(text)
    print(generations)
