from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import warnings
from pydantic import BaseModel
from src.api.model_loader import load_artifacts
from src.api.predict import predict_category

warnings.filterwarnings("ignore")

app = FastAPI(title="Text Classification API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class TextRequest(BaseModel):
    text: str


@app.on_event("startup")
def startup_event():
    global model, tokenizer, label_classes
    model, tokenizer, label_classes = load_artifacts()


@app.post("/predict")
def predict_text(request: TextRequest):
    try:
        result = predict_category(request.text, model, tokenizer, label_classes)
        return {
            "predicted_class": result["label"],
            "confidence": f"{result['confidence']:.2%}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
def classify_text(request: Request, text: str = Form(...)):
    try:
        result = predict_category(text, model, tokenizer, label_classes)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "text": text,
            "result": f"{result['label']} (Confidence: {result['confidence']:.2%})"
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "text": text,
            "result": f"Error: {str(e)}"
        })
