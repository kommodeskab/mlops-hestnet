from src.lightning_modules import BaseLightningModule
from src import OptimizerType, LRSchedulerType, TokenizedBatch, StepOutput
from src.networks import CausalTransformer
from torch import Tensor
from src.datasets import Tokenizer
import torch
from src import LossOutput
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class CausalLLM(BaseLightningModule):
    """
    A dummy LightningModule for testing purposes.

    Args:
        network (CausalTransformer): The causal transformer network.
        tokenizer (Tokenizer): The tokenizer for encoding and decoding text.
        optimizer (OptimizerType, optional): The optimizer to use. Defaults to None.
        lr_scheduler (LRSchedulerType, optional): The learning rate scheduler to use. Defaults to None.

    """

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

    def forward(self, batch: TokenizedBatch) -> CausalLMOutputWithCrossAttentions:
        """
        A simple forward pass. This forward uses the forward function of the `network` passed to the module.
        The returned output of type `CausalLMOutputWithCrossAttentions` contains the loss and logits.
        The loss is automatically computed using Huggingface logic.

        Args:
            batch (TokenizedBatch): The batch containing tokens and attention mask.

        Returns:
            CausalLMOutputWithCrossAttentions: The output from the network's forward pass.
        """
        return self.network.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

    def common_step(self, batch: TokenizedBatch, batch_idx: int) -> StepOutput:
        """
        The training/validation/test step. Given a batch, the model does a forward pass and returns a `StepOutput`
        which contains the loss and model_output.

        Args:
            batch (TokenizedBatch): The batch containing tokens and attention mask.
            batch_idx (int): The index of the batch. Not relevant for this task.

        Returns:
            StepOutput: The output containing loss and model output.
        """
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
        """
        Method for generating new text. The method does:
        1) Tokenization of input text.
        2) Recursive token generation using next token prediction scheme. (uses build-in Huggingface logic)
        3) Decoding of generated tokens back to text.

        Example usage:
        >>> generations = llm.generate(["The cat", "Once upon a time"], max_new_tokens=50)
        >>> print(generations)
        >>> # ['The cat sat on the mat.', 'Once upon a time in Hollywood.'] example output

        Args:
            text (list[str]): List of input text strings to generate continuations for. Also called 'prompts'.
            **kwargs: Additional generation parameters to override defaults, e.g., max_new_tokens, temperature, top_k, top_p, etc.

        Returns:
            list[str]: The generated text continuations for each input prompt.
        """
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

    def batch_generate(self, text: list[str], batch_size: int, **kwargs) -> list[str]:
        """
        Method for generating new text in batches. This is useful when generating text for a large number of prompts.

        Args:
            text (list[str]): List of input text strings to generate continuations for.
            batch_size (int): The size of each batch for generation.
            **kwargs: Additional generation parameters to override defaults.
        Returns:
            list[str]: The generated text continuations for each input prompt.
        """
        all_generations = []
        for i in range(0, len(text), batch_size):
            batch_text = text[i : i + batch_size]
            generations = self.generate(batch_text, **kwargs)
            all_generations.extend(generations)
        return all_generations


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
    generations = llm.batch_generate(text, batch_size=2)
    print(len(generations))
