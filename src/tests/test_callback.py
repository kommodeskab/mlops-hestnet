from pathlib import Path
import yaml
import pytest
from unittest.mock import patch
from src.utils import get_root
from src.callbacks.LLM_judge_callback import LLMJudgeCallback


class TestLLMJudgeCallback:
    """Tests for LLMJudgeCallback"""

    @pytest.fixture
    def config(self):
        """Load config from project root"""
        config_path = Path(get_root()) / "configs" / "callbacks" / "LLMjudgecallback.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)["llmjudge_callback"]

    @pytest.fixture
    def callback_params(self, config):
        """Use config data in test parameters"""
        return {
            "text": config["text"],
            "judge_prompt": config["judge_prompt"],
            "model_name": "gemini-2.5-flash",
            "project": "test-project",
            "use_vertexai": False,
            "seed": 42,
        }

    def test_init_no_client(self, callback_params):
        """Test successful initialization"""
        with (
            patch("src.callbacks.LLM_judge_callback.genai.Client") as mock_client,
            patch("src.callbacks.LLM_judge_callback.load_dotenv") as mock_dotenv,
        ):
            callback = LLMJudgeCallback(**callback_params)

            assert callback.text == callback_params["text"]
            assert callback.judge_prompt == callback_params["judge_prompt"]
            assert callback.model_name == "gemini-2.5-flash"
            assert callback.seed == 42
            mock_dotenv.assert_called_once()
            mock_client.assert_called_once()

    # TODO GITHUB BOT WITH GCLOUD API KEY
    # def test_instantiation(callback_params):
    #     LLMJudgeCallback(**callback_params)
