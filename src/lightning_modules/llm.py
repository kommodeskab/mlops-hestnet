from src.lightning_modules import BaseLightningModule
from src.losses import BaseLossFunction
from src import OptimizerType, LRSchedulerType, TokenizedBatch, ModelOutput, StepOutput
from src.networks import CausalTransformer
from src.datasets import TokenizedDataset
from torch import Tensor
from src.datasets import Tokenizer

class CausalLLM(BaseLightningModule):
    """A dummy LightningModule for testing purposes."""

    def __init__(
        self,
        network: CausalTransformer,
        tokenizer: Tokenizer,
        loss_fn: BaseLossFunction,
        optimizer: OptimizerType = None,
        lr_scheduler: LRSchedulerType = None,
    ):
        # optimizer and lr_scheduler can be None, for example during testing
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.tokenizer = tokenizer
        self.network = network
        self.loss_fn = loss_fn

    def forward(self, batch: TokenizedBatch) -> ModelOutput:
        output = self.network.forward(batch["input_ids"])
        return ModelOutput(output=output)

    def common_step(self, batch: TokenizedBatch, batch_idx: int) -> StepOutput:
        output = self.forward(batch)
        loss = self.loss_fn(output, batch)
        return StepOutput(
            loss=loss["loss"],
            loss_output=loss,
            model_output=output,
        )
    
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
    from src.datasets import Tokenizer, TokenizedDataset
    
    tokenier = Tokenizer("distilbert/distilgpt2")
    
    network = CausalTransformer("distilbert/distilgpt2")
    loss_fn = ...
    
    llm = CausalLLM(
        network=network,
        loss_fn=loss_fn,
    )
    
    llm._tokenizer = tokenier
    
    text = ["Dette er en test sætning.", "Dette er en anden sætning."]
    generations = llm.generate(text)
    print(generations)