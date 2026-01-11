import pytest
from pathlib import Path
import os

@pytest.fixture(scope="session", autouse=True)
def setup_test_env(tmp_path_factory):
    """Create a minimal .env file for testing."""
    tmp_dir = tmp_path_factory.mktemp("test_data")
    
    env_content = f"""
DATA_PATH={tmp_dir}/data
MODEL_PATH={tmp_dir}/models
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
    