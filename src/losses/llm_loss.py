import torch.nn as nn
from src.losses.baseloss import BaseLossFunction
from src import ModelOutput, LossOutput, TokenizedBatch

class CausalLMLoss(BaseLossFunction):
    def __init__(self, ignore_index=-100):
        super().__init__()
        # Standard default for HF models is CrossEntropyLoss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, model_output: ModelOutput, batch: TokenizedBatch) -> LossOutput:
        logits = model_output["output"]
        labels = batch["input_ids"]

        # Use reshape instead of contiguous().view()
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        loss = self.loss_fn(
            shift_logits.reshape(-1, shift_logits.size(-1)), 
            shift_labels.reshape(-1)
        )
        
        return LossOutput(loss=loss)