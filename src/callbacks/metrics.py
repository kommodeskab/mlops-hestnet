from sentence_transformers import SentenceTransformer
import torch
from src.datasets import BaseTextDataset
from torch import Tensor
from pytorch_lightning import Callback
from src.utils import temporary_seed
from src.lightning_modules import CausalLLM
import pytorch_lightning as pl
import logging

logger = logging.getLogger(__name__)


class FrechetDistanceMetric(Callback):
    def __init__(
        self,
        reference_dataset: BaseTextDataset,
    ):
        super().__init__()
        self.references = [reference_dataset[i]["text"] for i in range(len(reference_dataset))]
        # make prompts. they should consist of the first few words of each reference
        # the model then test if the generated text from these prompts are similar to the references
        self.prompts = [" ".join(ref.split()[:4]) for ref in self.references]

        model_name = "Qwen/Qwen3-Embedding-0.6B"
        self.model = SentenceTransformer(model_name)
        self.model.eval()
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Loaded {model_name} model with {num_params} parameters for Frechet Distance Metric...")
        self.ref_stats = None

    @torch.no_grad()
    def _get_stats(self, text: list[str]) -> tuple[Tensor, Tensor]:
        # limit the number of tokens to 100
        # why?
        # 1) efficiency
        # 2) to avoid sentence length bias
        embeds = self.model.encode_document(text)
        embeds = torch.from_numpy(embeds)
        mu = torch.mean(embeds, dim=0).double()
        sigma = torch.cov(embeds.T).double()
        return mu, sigma

    def _register_ref_stats(self):
        self.ref_stats = self._get_stats(self.references)

    @staticmethod
    def frechet_distance(
        mu_x: Tensor,
        sigma_x: Tensor,
        mu_y: Tensor,
        sigma_y: Tensor,
    ) -> float:
        a = (mu_x - mu_y).square().sum()
        b = sigma_x.float().trace() + sigma_y.float().trace()
        c = torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum()
        return (a + b - 2 * c).clamp(min=0.0).item()

    def on_train_start(self, trainer: pl.Trainer, pl_module: CausalLLM):
        self.model.to(pl_module.device)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: CausalLLM):
        logger = pl_module.logger

        if self.ref_stats is None:
            # compute and store reference stats only once since they are constant
            self._register_ref_stats()

        mu_ref, sigma_ref = self.ref_stats

        with temporary_seed(42):
            generations = pl_module.generate(self.prompts)

        mu_gen, sigma_gen = self._get_stats(generations)
        fcd = self.frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)
        logger.log_metrics({"FrechetDistance": fcd}, step=pl_module.global_step)


if __name__ == "__main__":
    generated = ["This is a generated sentence.", "Another generated sentence."]

    class DummyDataset(BaseTextDataset):
        def __init__(self):
            super().__init__()

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {"text": "This is a reference sentence."}

    ref_dataset = DummyDataset()
    print(len(ref_dataset))
    metric = FrechetDistanceMetric(reference_dataset=ref_dataset)
    metric._register_ref_stats()
    mu_gen, sigma_gen = metric._get_stats(generated)
    mu_ref, sigma_ref = metric.ref_stats
    fcd = metric.frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)
    print(f"Frechet Distance: {fcd}")
