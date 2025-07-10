# backend/pdf_utils.py
from PyPDF2 import PdfReader

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text.strip()
