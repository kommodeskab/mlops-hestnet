from pytorch_lightning import Callback
import pytorch_lightning as pl
from src.lightning_modules import CausalLLM
from src.utils import temporary_seed


class TextCallback(Callback):
    """
    A callback for generating text samples at the end of validation.
    This is useful for monitoring/sanity checking the text generation quality during training.

    Args:
        text (list[str]): A list of text prompts to use for generation.
    """

    def __init__(self, text: list[str]):
        super().__init__()
        # hydra instantiation results in omegaconf.listconfig.ListConfig
        # convert to normal list
        self.text = list(text)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: CausalLLM):
        logger = pl_module.logger

        # use the same seed for generation everytime
        # this is useful for comparing the output at different training steps
        with temporary_seed(42):
            generations = pl_module.generate(self.text)

        # this is the default way to log tables in pytorch lightning
        # TODO: make this into a method in the BaseLightningModule
        logger.log_text(
            "Generations",
            columns=["Prompt", "Generation"],
            data=list(zip(self.text, generations)),
            step=pl_module.global_step,
        )
