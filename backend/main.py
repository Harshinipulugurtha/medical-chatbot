# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from model_utils import ask_gemini, analyze_image, highlight_medical_entities, summarize_content
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
    try:
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        # Check image format and size before analysis
        from PIL import Image as PILImage
        img = PILImage.open(image_path)
        if img.size[0] < 100 or img.size[1] < 100:
            return {"analysis": "❌ Error: Image too small for analysis. Please upload an image at least 100x100 pixels."}
        if img.format not in ["JPEG", "PNG", "JPG"]:
            return {"analysis": f"❌ Error: Unsupported image format: {img.format}. Please upload a PNG or JPEG image."}
        result = analyze_image(str(image_path))
        summary = summarize_content(result)
        return {"analysis": summary}
    except Exception as e:
        return {"analysis": f"❌ Error: Could not process image. Reason: {str(e)}"}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_path = DATA_DIR / file.filename
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    text = extract_text_from_pdf(str(pdf_path))
    summary = summarize_content(text)
    return {"content": summary}
