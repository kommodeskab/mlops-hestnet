import pytest
from datasets import Dataset
from unittest.mock import patch

from src.datasets import DGigawordDataset


@pytest.fixture(scope="session", autouse=True)
def setup_test_env(tmp_path_factory):
    """Create a minimal .env file for testing."""
    tmp_dir = tmp_path_factory.mktemp("test_data")

    env_content = f"""
DATA_PATH={tmp_dir}/data
HF_TOKEN=test_token_placeholder
"""

    # Write to a temporary .env file
    env_file = tmp_dir / ".env"
    env_file.write_text(env_content.strip())

    # Load the environment variables
    from dotenv import load_dotenv

    load_dotenv(env_file, override=True)

    (tmp_dir / "data").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "models").mkdir(parents=True, exist_ok=True)

    yield tmp_dir


# Setup mock dataset for automated tests
@pytest.fixture(scope="session")
def N_TRAIN():
    """Number of samples in the mock dataset."""
    return 1000


@pytest.fixture(scope="session")
def mock_gigaword_dataset(N_TRAIN):
    """Create a tiny synthetic dataset for testing."""
    fake_data = {
        "text": [
            "Jeg bor i et kommodeskab",
        ]
        * N_TRAIN
    }
    return Dataset.from_dict(fake_data)


# Patch mock_load in instead of src.datasets.gigaword.load_dataset for all tests
@pytest.fixture(scope="session", autouse=True)
def mock_dataset_loading(mock_gigaword_dataset):
    """Mock HuggingFace dataset loading to avoid downloads in CI."""
    with patch("src.datasets.gigaword.load_dataset", return_value=mock_gigaword_dataset) as mock_load:
        yield mock_load


# Delay construction of lists of datasets for tests such that they use mock_load
@pytest.fixture(params=[[1], [2]])
def dataset_counts(request):
    """Fixture that creates datasets based on count."""
    return [DGigawordDataset() for _ in range(request.param[0])]
