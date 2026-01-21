import os
os.chdir(os.path.join("src", "app"))

from fastapi.testclient import TestClient
from src.app.api import app
client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200

def test_prompt():
    with TestClient(app) as client:
        prompt = "What is the "
        response = client.post("/submit", 
                            data={"prompt": prompt, 
                                  "use_finetuned": "false"})

        assert response.status_code == 200
        assert len(response.text) > len(prompt)

def test_prompt_fine_tuned():
    with TestClient(app) as client:
        prompt = "What is the "
        response = client.post("/submit", 
                            data={"prompt": prompt, 
                                  "use_finetuned": "true"})

        assert response.status_code == 200
        assert len(response.text) > len(prompt)