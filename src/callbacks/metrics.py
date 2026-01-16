import torch
from FlagEmbedding import BGEM3FlagModel
from src.datasets import BaseTextDataset
from torch import Tensor
from pytorch_lightning import Callback
from src.utils import temporary_seed
from src.lightning_modules import CausalLLM
import pytorch_lightning as pl


class FrechetDistanceMetric(Callback):
    def __init__(
        self,
        reference_dataset: BaseTextDataset,
    ):
        super().__init__()
        self.references = [reference_dataset[i]["text"] for i in range(len(reference_dataset))]
        self.prompts = [""] * len(reference_dataset)  # empty prompts for unconditional generation
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        self.ref_stats = None

    def _get_stats(self, text: list[str]) -> tuple[Tensor, Tensor]:
        # limit the number of tokens to 100
        # why?
        # 1) efficiency
        # 2) to avoid sentence length bias
        embeds = self.model.encode(text, batch_size=32, max_length=100)["dense_vecs"]
        embeds = torch.from_numpy(embeds)
        mu = embeds.mean(dim=0)
        sigma = torch.cov(embeds.T)
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
        b = sigma_x.trace() + sigma_y.trace()
        c = torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum()
        return (a + b - 2 * c).clamp(min=0.0).item()

    def on_validation_end(self, trainer: pl.Trainer, pl_module: CausalLLM):
        if self.ref_stats is None:
            # compute and store reference stats only once since they are constant
            self._register_ref_stats()

        mu_ref, sigma_ref = self.ref_stats

        with temporary_seed(42):
            generations = pl_module.generate(self.prompts)

        mu_gen, sigma_gen = self._get_stats(generations)
        fcd = self.frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)
        pl_module.log("FrechetDistance", fcd)


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
