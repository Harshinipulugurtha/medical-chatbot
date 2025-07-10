# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from model_utils import ask_gemini, analyze_image, highlight_medical_entities
from pdf_utils import extract_text_from_pdf
from pathlib import Path
import os
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("/app/data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

@app.post("/ask")
async def ask_question(question: str = Form(...), context: str = Form(""), tone: str = Form("formal"), simplify: bool = Form(False)):
    answer = ask_gemini(question, context=context, tone=tone, simple=simplify)
    highlighted = highlight_medical_entities(answer)
    return {"answer": highlighted}

@app.post("/analyze_image")
async def analyze_image_route(image: UploadFile = File(...)):
    image_path = DATA_DIR / image.filename
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)
    result = analyze_image(str(image_path))
    return {"analysis": result}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_path = DATA_DIR / file.filename
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    text = extract_text_from_pdf(str(pdf_path))
    return {"content": text}
