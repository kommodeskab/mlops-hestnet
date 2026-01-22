from pathlib import Path
import logging
import os
import torch
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer


# Run from project root
"""
To launch:
python -m uvicorn --reload --port 8000 src.app.api:app
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

templates = Jinja2Templates(directory=Path("src/app/templates"))


# Converts state_dicts from pytorch lightning to huggingface format
def convert_lightning_to_hf(state_dict):
    hf_state_dict = {}
    target_prefix = "network.model."
    for k, v in state_dict.items():
        if k.startswith(target_prefix):
            # Remove the specific prefix
            new_key = k.replace(target_prefix, "")
            hf_state_dict[new_key] = v

    return hf_state_dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, model_ft, tokenizer
    print("Loading models")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model_ft = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model_path = Path("weights/fine-tuned.ckpt")
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location="cpu")
        state_dict = convert_lightning_to_hf(ckpt["state_dict"])
        missing, unexpected = model_ft.load_state_dict(state_dict)
        assert len(missing) == 0, "Missing keys in state_dict"
        assert len(unexpected) == 0, "Unexpected keys in state_dict"
    else:
        print("No fine-tuned model")
    model.eval()
    model_ft.eval()

    yield

    print("Cleaning up")
    del model, model_ft, tokenizer


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/submit", response_class=HTMLResponse)
async def submit(request: Request, prompt: str = Form(...), use_finetuned: bool = Form(False)):
    if use_finetuned:
        model_used = model_ft
    else:
        model_used = model
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        outputs = model_used.generate(**inputs, max_new_tokens=50, do_sample=True, top_p=0.95)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    logger.info("PROMPT RECEIVED")

    if len(response) <= len(prompt):
        logger.error("ERROR: empty output")

    return templates.TemplateResponse(
        {
            "request": request,
            "result": response,
            "prompt": prompt,
        },
        "index.html",
    )
