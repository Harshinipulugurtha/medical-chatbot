# backend/model_utils.py
import google.generativeai as genai
from transformers import pipeline
import os
from PIL import Image

genai.configure(api_key=os.getenv("API_KEY"))
NER = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

def ask_gemini(question, context="", tone="formal", simple=False):
    tone_map = {
        "formal": "Give a detailed medical explanation.",
        "friendly": "Respond warmly and clearly.",
        "child": "Explain like to a 10-year-old."
    }
    if simple:
        prompt = f"""You are a helpful multilingual medical assistant.\nContext: {context}\nQuestion: {question}\nPlease answer in 2-4 short sentences, using very simple words, as if explaining to a young child or someone with no medical background. Do not include technical details, long explanations, or any greetings.\n\nAnswer:"""
    else:
        prompt = f"""You are a helpful multilingual medical assistant.\nContext: {context}\nQuestion: {question}\nTone: {tone_map.get(tone, '')}\n\nAnswer:"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

def analyze_image(image_path):
    prompt = "You are a radiologist. Analyze the medical image for abnormalities."
    model = genai.GenerativeModel("gemini-2.5-flash")
    img = Image.open(image_path)
    response = model.generate_content([prompt, img])
    return response.text.strip()

def highlight_medical_entities(text):
    entities = NER(text)
    for ent in entities:
        label = ent['entity_group']
        word = ent['word']
        emoji = {"DISEASE": "🦠", "SYMPTOM": "🤒", "MEDICATION": "💊"}.get(label.upper(), "🔍")
        text = text.replace(word, f"**{emoji} {word}**")
    return text
