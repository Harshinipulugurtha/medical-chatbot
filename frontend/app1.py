#frontend/app.py
import streamlit as st
import requests
from tts_utils import generate_audio_html
import os
from transformers import pipeline
import string
from io import BytesIO
from PIL import Image, UnidentifiedImageError

# -------------------------
# Utility Functions
# -------------------------
def translate_question(text, lang_code):
    if lang_code == "en":
        return text
    model_map = {
        "fr": "Helsinki-NLP/opus-mt-fr-en",
        "es": "Helsinki-NLP/opus-mt-es-en",
        "de": "Helsinki-NLP/opus-mt-de-en",
        "hi": "Helsinki-NLP/opus-mt-hi-en",
        "zh": "Helsinki-NLP/opus-mt-zh-en"
    }
    model_name = model_map.get(lang_code)
    if not model_name:
        return text
    translator = pipeline("translation", model=model_name)
    return translator(text, max_length=512)[0]['translation_text']

def translate_answer(text, lang_code):
    if lang_code == "en":
        return text
    model_map = {
        "fr": "Helsinki-NLP/opus-mt-en-fr",
        "es": "Helsinki-NLP/opus-mt-en-es",
        "de": "Helsinki-NLP/opus-mt-en-de",
        "hi": "Helsinki-NLP/opus-mt-en-hi",
        "zh": "Helsinki-NLP/opus-mt-en-zh"
    }
    model_name = model_map.get(lang_code)
    if not model_name:
        return text
    translator = pipeline("translation", model=model_name)
    return translator(text, max_length=512)[0]['translation_text']

def is_greeting(text):
    greetings = {"hi", "hello", "hey", "good morning", "good evening", 
                 "bonjour", "salut", "hola", "hallo", "namaste", "你好"}
    text_clean = text.lower().translate(str.maketrans('', '', string.punctuation)).strip()
    return text_clean in greetings

# -------------------------
# Page Configuration & CSS
# -------------------------
st.set_page_config(page_title="🦠 Medical Assistant", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
        .main-title {
            font-size: 2.5em;
            font-weight: 800;
            color: #003366;
            margin-bottom: 0.5em;
        }
        .section-heading {
            font-size: 1.6em;
            font-weight: 700;
            color: #1a1a1a;
            margin-top: 2em;
        }
        .upload-box {
            border: 2px dashed #ccc;
            padding: 1em;
            border-radius: 12px;
            background-color: #f9f9f9;
        }
        .stDownloadButton>button {
            background-color: #2E8B57 !important;
            color: white !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("🩺 Medical Assistant")
page = st.sidebar.radio("Navigate", ["🏞️ Home", "🖼️ Image Analysis", "📄 PDF Report Analysis"])

# -------------------------
# Session State Initialization
# -------------------------
for key in ["image_messages", "pdf_messages"]:
    if key not in st.session_state:
        st.session_state[key] = []
for key in ["image_analysis", "pdf_analysis"]:
    if key not in st.session_state:
        st.session_state[key] = ""
for key in ["image_analysis_spoken", "pdf_analysis_spoken"]:
    if key not in st.session_state:
        st.session_state[key] = False
for key in ["image_analysis_displayed", "pdf_analysis_displayed"]:
    if key not in st.session_state:
        st.session_state[key] = False

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# -------------------------
# Global Inputs (Language & Role)
# -------------------------
language_map = {
    "en": "English", "fr": "French", "es": "Spanish",
    "de": "German", "hi": "Hindi", "zh": "Chinese"
}

output_lang = st.sidebar.selectbox("🌐 Response language:", list(language_map.keys()), format_func=lambda x: language_map[x])
simple_explanation = st.sidebar.checkbox("📖 Simple explanation mode (for kids / non-experts)")
tone = st.sidebar.selectbox("🧘 Tone:", ["formal", "friendly", "child"])

role_map = {
    "radiologist": "Radiologist",
    "general_physician": "General Physician",
    "orthopedist": "Orthopedist",
    "cardiologist": "Cardiologist",
    "neurologist": "Neurologist",
    "dermatologist": "Dermatologist",
    "pediatrician": "Pediatrician",
    "dentist": "Dentist"
}
role = st.sidebar.selectbox("👨‍⚕️ Medical Expert Role:", list(role_map.keys()), format_func=lambda x: role_map[x])

# -------------------------
# Home Page
# -------------------------
if page == "🏞️ Home":
    st.title("🩺 Welcome to Medical Assistant")
    st.markdown("""
    Upload a medical image or PDF report to begin analysis.
    You can ask questions in natural language, choose your preferred language, and set the tone of the assistant.
    """)

# -------------------------
# Image Analysis with Chat
# -------------------------
if page == "🖼️ Image Analysis":
    st.subheader("🖼️ Upload and Analyze Medical Image")
    image_file = st.file_uploader("Upload medical image", type=["png", "jpg", "jpeg"])

    if image_file:
        try:
            image_bytes = image_file.read()
            img = Image.open(BytesIO(image_bytes))
            st.image(img, caption="Uploaded Image Preview")
            image_file.seek(0)
            with st.spinner("Analyzing image..."):
                res = requests.post(f"{BACKEND_URL}/analyze_image", files={"image": image_file}, timeout=15)
                if res.status_code != 200:
                    st.error(f"❌ Backend error: {res.status_code} - {res.text}")
                try:
                    result = res.json().get("analysis", "❌ Failed to get analysis")
                except:
                    result = "❌ Invalid backend response."
                translated_result = translate_answer(result, output_lang)
                st.session_state.image_analysis = translated_result
                st.session_state.image_analysis_displayed = False
                st.session_state.image_messages = []
        except UnidentifiedImageError:
            st.error("❌ Error: Unsupported image format.")

    if st.session_state.image_analysis:
        st.divider()
        st.subheader("💬 Ask Questions About the Image")

        if not st.session_state.image_analysis_displayed:
            with st.chat_message("assistant"):
                st.markdown(st.session_state.image_analysis)
                st.components.v1.html(generate_audio_html(st.session_state.image_analysis, lang=output_lang, key="image_analysis"), height=100)
            st.session_state.image_analysis_displayed = True

        user_input = st.chat_input("Ask a question about the uploaded image...", key="image_chat")

        if user_input:
            st.session_state.image_messages.append(("user", user_input))
            with st.spinner("Thinking..."):
                history = [f"Image Analysis: {st.session_state.image_analysis}"]
                for role_msg, msg in st.session_state.image_messages:
                    history.append(f"{role_msg.capitalize()}: {msg}")
                payload = {
                    "question": user_input,
                    "context": "\n".join(history),
                    "tone": tone,
                    "role": role,
                    "simplify": simple_explanation
                }
                res = requests.post(f"{BACKEND_URL}/ask", data=payload, timeout=15)
                try:
                    answer = res.json().get("answer", "❌ No response")
                except:
                    answer = "❌ Invalid response from backend."
                translated_answer = translate_answer(answer, output_lang)
                st.session_state.image_messages.append(("assistant", translated_answer))

        for idx, (role, msg) in enumerate(st.session_state.image_messages):
            with st.chat_message(role):
                st.markdown(msg)
                if role == "assistant":
                    st.components.v1.html(generate_audio_html(msg, lang=output_lang, key=f"assistant_image_{idx}"), height=100)

# -------------------------
# PDF Report Analysis with Chat
# -------------------------
if page == "📄 PDF Report Analysis":
    st.subheader("📄 Upload and Analyze Medical PDF Report")
    pdf_file = st.file_uploader("Upload medical report (PDF)", type=["pdf"])

    if pdf_file:
        with st.spinner("Analyzing PDF report..."):
            pdf_file.seek(0)
            try:
                res = requests.post(f"{BACKEND_URL}/upload_pdf", files={"file": pdf_file}, timeout=15)
                if res.status_code != 200:
                    st.error(f"❌ Backend error: {res.status_code} - {res.text}")
                    result = None
                else:
                    res_json = res.json()
                    result = res_json.get("content", "❌ Failed to analyze PDF")
            except Exception as e:
                st.error(f"❌ Could not process PDF: {e}")
                result = None

            if result:
                translated_result = translate_answer(result, output_lang)
                st.session_state.pdf_analysis = translated_result
                st.session_state.pdf_analysis_displayed = False
                st.session_state.pdf_messages = []


    if st.session_state.pdf_analysis:
        st.divider()
        st.subheader("📁 Ask Questions About the PDF")

        if not st.session_state.pdf_analysis_displayed:
            with st.chat_message("assistant"):
                st.markdown(st.session_state.pdf_analysis)
                st.components.v1.html(generate_audio_html(st.session_state.pdf_analysis, lang=output_lang, key="pdf_analysis"), height=100)
            st.session_state.pdf_analysis_displayed = True

        user_input = st.chat_input("Ask a question about the PDF report...", key="pdf_chat")

        if user_input:
            st.session_state.pdf_messages.append(("user", user_input))
            with st.spinner("Thinking..."):
                history = [f"PDF Analysis: {st.session_state.pdf_analysis}"]
                for role_msg, msg in st.session_state.pdf_messages:
                    history.append(f"{role_msg.capitalize()}: {msg}")
                payload = {
                    "question": user_input,
                    "context": "\n".join(history),
                    "tone": tone,
                    "role": role,
                    "simplify": simple_explanation
                }
                res = requests.post(f"{BACKEND_URL}/ask", data=payload, timeout=15)
                try:
                    answer = res.json().get("answer", "❌ No response")
                except:
                    answer = "❌ Invalid response from backend."
                translated_answer = translate_answer(answer, output_lang)
                st.session_state.pdf_messages.append(("assistant", translated_answer))

        for idx, (role, msg) in enumerate(st.session_state.pdf_messages):
            with st.chat_message(role):
                st.markdown(msg)
                if role == "assistant":
                    st.components.v1.html(generate_audio_html(msg, lang=output_lang, key=f"assistant_pdf_{idx}"), height=100)
