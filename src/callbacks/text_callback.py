from pytorch_lightning import Callback
import pytorch_lightning as pl
from src.lightning_modules import CausalLLM
from src.utils import temporary_seed

class TextCallback(Callback):
    def __init__(self, additional_text: list[str] = []):
        super().__init__()
        self.text = [
            "Hovedstaden i Danmark er",
            "Damen bar en trøje af",
            "Katten sad på en",
        ]
        self.text.extend(additional_text)
        
    def on_validation_end(self, trainer: pl.Trainer, pl_module: CausalLLM):
        logger = pl_module.logger
        
        with temporary_seed(42):
            generations = pl_module.generate(self.text)
                    
        logger.log_text(
            "Generations",
            columns=["Prompt", "Generation"],
            data=list(zip(self.text, generations)),
            step = pl_module.global_step,
        )
        