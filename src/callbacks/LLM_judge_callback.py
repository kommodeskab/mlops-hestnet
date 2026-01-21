from pytorch_lightning import Callback
import pytorch_lightning as pl
from src.lightning_modules import CausalLLM
from src.utils import temporary_seed
from dotenv import load_dotenv
from google import genai
import logging

logger = logging.getLogger(__name__)


class LLMJudgeCallback(Callback):
    """
    Callback that uses an LLM (Gemini) to evaluate model generations during validation.

    Args:
        text: List of prompt texts to generate from.
        judge_prompt: List of strings forming the judge instruction prompt.
        model_name: Gemini model to use for evaluation. Defaults to "gemini-2.0-flash-exp".
        use_vertexai: Whether to use Vertex AI. Defaults to True.
        seed: Random seed for generation. Defaults to 42.
    """

    def __init__(
        self,
        text: list[str],
        judge_prompt: list[str],
        model_name: str = "gemini-2.5-flash",
        use_vertexai: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        # Convert from OmegaConf ListConfig to regular lists
        self.text = list(text)
        self.judge_prompt = list(judge_prompt)
        self.model_name = model_name
        self.seed = seed

        # Load environment variables and initialize client
        load_dotenv()
        try:
            self.client = genai.Client(vertexai=use_vertexai)
            logger.info(f"LLMJudgeCallback initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.client = None

    def on_validation_end(self, trainer: pl.Trainer, pl_module: CausalLLM) -> None:
        """Generate samples and evaluate them using LLM judge."""
        if self.client is None:
            logger.warning("Gemini client not initialized. Skipping LLM judge evaluation.")
            return
        try:
            with temporary_seed(self.seed):
                generations = pl_module.generate(self.text)

            prompt = self._format_prompt(generations)
            response = self.client.models.generate_content(model=self.model_name, contents=prompt)

            score, evaluation = self._parse_response(response.text)
            self._log_results(pl_module, generations, score, evaluation)

        except Exception as e:
            logger.error(f"LLMJudgeCallback error: {e}", exc_info=True)

    def _format_prompt(self, generations: list[str]) -> str:
        samples = "\n\n".join(
            [
                f"Prompt {i+1}: {prompt}\nResponse {i+1}: {gen}"
                for i, (prompt, gen) in enumerate(zip(self.text, generations))
            ]
        )
        return f"{self.judge_prompt}\n\n{samples}"

    def _parse_response(self, text: str) -> tuple[str, str]:
        try:
            score, evaluation = text.split("|", 1)
            return score.strip(), evaluation.strip()
        except ValueError:
            logger.warning(f"Invalid LLM judge format: {text}")
            return "N/A", text

    def _log_results(self, pl_module: CausalLLM, generations: list[str], score: str, evaluation: str) -> None:
        pl_module.logger.log_text(
            "LLM_Judge_Evaluations",
            columns=["Prompt", "Generation", "Score", "Evaluation"],
            data=[[p, g, score, evaluation] for p, g in zip(self.text, generations)],
            step=pl_module.global_step,
        )
        pl_module.logger.log_metrics(
            metrics={"score": score},
            step=pl_module.global_step,
            sync_dist=True,  # ensure logging works in multi-GPU setups
        )
