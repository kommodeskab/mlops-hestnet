from pathlib import Path
import yaml
import pytest
from unittest.mock import Mock, patch
from src.utils import get_root
from src.callbacks.LLM_judge_callback import LLMJudgeCallback


class TestLLMJudgeCallback:
    """Tests for LLMJudgeCallback"""

    pytestmark = pytest.mark.unit

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

    def test_format_prompt(self, callback_params):
        """Test successful prompt formatting"""
        with (
            patch("src.callbacks.LLM_judge_callback.genai.Client"),
            patch("src.callbacks.LLM_judge_callback.load_dotenv"),
        ):
            callback = LLMJudgeCallback(**callback_params)
            generations = ["Jeg bor i et kommodeskab", "København"]

            prompt = callback._format_prompt(generations)

            assert f"Prompt 1: {callback_params["text"][0]}" in prompt
            assert f"Response 1: {generations[0]}" in prompt
            assert f"Prompt 2: {callback_params["text"][1]}" in prompt
            assert f"Response 2: {generations[1]}" in prompt

    def test_parse_response_valid_no_client(self, callback_params):
        """Test parsing valid LLM response"""
        with (
            patch("src.callbacks.LLM_judge_callback.genai.Client"),
            patch("src.callbacks.LLM_judge_callback.load_dotenv"),
        ):
            callback = LLMJudgeCallback(**callback_params)

            score, evaluation = callback._parse_response("85 | Good response, clear and concise")

            assert score == "85"
            assert evaluation == "Good response, clear and concise"

    def test_init_client_failure(self, callback_params):
        """Test initialization when client fails"""
        with (
            patch("src.callbacks.LLM_judge_callback.genai.Client") as mock_client,
            patch("src.callbacks.LLM_judge_callback.load_dotenv"),
        ):
            mock_client.side_effect = Exception("API error")

            callback = LLMJudgeCallback(**callback_params)
            assert callback.client is None

    def test_on_validation_end_no_client(self, callback_params):
        """Test validation end when client is not initialized"""
        with (
            patch("src.callbacks.LLM_judge_callback.genai.Client") as mock_client,
            patch("src.callbacks.LLM_judge_callback.load_dotenv"),
        ):
            mock_client.side_effect = Exception("API error")
            callback = LLMJudgeCallback(**callback_params)

            mock_trainer = Mock()
            mock_module = Mock()

            # Should not raise exception
            callback.on_validation_end(mock_trainer, mock_module)
            mock_module.generate.assert_not_called()

    def test_on_validation_end_generation_error(self, callback_params):
        """Test validation end when generation fails"""
        with (
            patch("src.callbacks.LLM_judge_callback.genai.Client") as mock_client,
            patch("src.callbacks.LLM_judge_callback.load_dotenv"),
        ):
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            callback = LLMJudgeCallback(**callback_params)

            mock_trainer = Mock()
            mock_module = Mock()
            mock_module.generate.side_effect = Exception("Generation failed")

            # Should handle error without throwing an exception
            callback.on_validation_end(mock_trainer, mock_module)

    # Real client with real API calls
    # TODO GITHUB BOT WITH GCLOUD API KEY
    @pytest.mark.integration
    @pytest.mark.unit(False)
    def test_init(self, callback_params):
        callback = LLMJudgeCallback(**callback_params)
        assert callback.text == callback_params["text"]
        assert callback.judge_prompt == callback_params["judge_prompt"]
        assert callback.model_name == "gemini-2.5-flash"
        assert callback.seed == 42
        assert callback.client is not None

    @pytest.mark.integration
    @pytest.mark.unit(False)
    def test_callback(self, callback_params):
        callback = LLMJudgeCallback(**callback_params)
        assert callback.client is not None

        mock_trainer = Mock()
        mock_module = Mock()
        mock_module.generate.side_effect = ["Jeg bor i et kommodeskab"] * len(callback_params["text"])
        mock_logger = Mock()
        mock_module.logger = mock_logger

        callback.on_validation_end(mock_trainer, mock_module)

        # Check log_metrics and log_text were called and logged valid data
        mock_logger.log_metrics.assert_called_once()
        logged_metrics = mock_logger.log_metrics.call_args[1]["metrics"]
        assert 1 <= int(logged_metrics["score"]) <= 100

        mock_logger.log_text.assert_called_once()
        text_call = mock_logger.log_text.call_args
        logged_data = text_call[1]["data"]
        assert 1 <= int(logged_data[0][2]) <= 100
        assert isinstance(logged_data[0][3], str)  # evaluation column
