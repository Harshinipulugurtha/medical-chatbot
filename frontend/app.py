# frontend/app.py
import streamlit as st
import requests
from mic_utils import record_and_transcribe
from tts_utils import speak_text
from ner_display import display_ner_highlighted
import os
from transformers import pipeline

st.set_page_config(page_title="🧠 Medical Assistant", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🩺 Medical Multimodal Assistant")

# Language and tone selection
language_map = {
    "en": "English", "fr": "French", "es": "Spanish",
    "de": "German", "hi": "Hindi", "zh": "Chinese"
}

output_lang = st.selectbox("🌐 Response language:", list(language_map.keys()), format_func=lambda x: language_map[x])
input_lang = st.selectbox("🎙️ Voice input language:", list(language_map.keys()), format_func=lambda x: language_map[x])
simple_explanation = st.checkbox("📖 Simple explanation mode (for kids / non-experts)")
tone = st.selectbox("🧘 Tone:", ["formal", "friendly", "child"])

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("🖼️ Upload Image")
    image_file = st.file_uploader("Analyze medical image", type=["png", "jpg", "jpeg"])
    if image_file:
        with st.spinner("Analyzing image..."):
            res = requests.post(f"{BACKEND_URL}/analyze_image", files={"image": image_file})
            result = res.json().get("analysis", "")
            st.session_state.messages.append(("assistant", result))
            speak_text(result)
    
    st.subheader("📄 Upload Report (PDF)")
    pdf_file = st.file_uploader("Analyze PDF", type=["pdf"])
    if pdf_file:
        with st.spinner("Extracting PDF..."):
            res = requests.post(f"{BACKEND_URL}/upload_pdf", files={"file": pdf_file})
            pdf_text = res.json().get("content", "")
            st.session_state.messages.append(("assistant", f"📘 Extracted Report:\n\n{pdf_text[:800]}..."))
            speak_text(pdf_text[:300])

    st.subheader("🎙️ Voice Question")
    if st.button("🎤 Speak"):
        transcript = record_and_transcribe()
        if transcript:
            st.session_state.messages.append(("user", transcript))

with col1:
    for role, msg in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(msg)

    user_input = st.chat_input("Type a medical question...")
    if user_input:
        st.session_state.messages.append(("user", user_input))

    # Send to backend and display answer
    if st.session_state.messages and st.session_state.messages[-1][0] == "user":
        with st.spinner("Thinking..."):
            history = [
                f"User: {q}\nAssistant: {a}"
                for i in range(len(st.session_state.messages) - 2)
                if (st.session_state.messages[i][0] == "user" and st.session_state.messages[i+1][0] == "assistant")
                for q, a in [(st.session_state.messages[i][1], st.session_state.messages[i+1][1])]
            ]
            chat_history = "\n\n".join(history[-3:])  # Last 3 Q&A pairs
            query = st.session_state.messages[-1][1]
            payload = {
                "question": query,
                "context": chat_history,
                "tone": tone,  # or use st.selectbox
                "simplify": simple_explanation
            }
            res = requests.post(f"{BACKEND_URL}/ask", data=payload)
            answer = res.json().get("answer", "❌ No response")
            st.session_state.messages.append(("assistant", answer))
            speak_text(answer)

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
