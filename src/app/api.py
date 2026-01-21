from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
import logging

'''
To launch:
python -m uvicorn --reload --port 8000 api:app
'''

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

templates = Jinja2Templates(directory="templates")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, model_ft, tokenizer
    print("Loading models")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    ckpt = torch.load("weights/fine-tuned-bad.ckpt", map_location="cpu")
    state_dict = ckpt['state_dict']
    model_ft = copy.copy(model)
    model_ft.load_state_dict(state_dict, strict=False)
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
async def submit(request: Request, prompt: str = Form(...),
                 use_finetuned: bool = Form(False)):
    if use_finetuned:
        model_used = model_ft
    else:
        model_used = model
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model_used.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    logger.info('PROMPT RECEIVED') 

    if len(response) <= len(prompt):
        logger.error('ERROR: empty output')
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": response,
            "prompt": prompt,
        }
    )