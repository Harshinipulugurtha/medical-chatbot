# frontend/app.py
import streamlit as st
import requests
from mic_utils import record_and_transcribe
from tts_utils import speak_text
from ner_display import display_ner_highlighted
import os
from transformers import pipeline
import string
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from io import BytesIO
from pydub import AudioSegment
import tempfile

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

st.set_page_config(page_title="🧠 Medical Assistant", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
def is_greeting(text):
    greetings = {"hi", "hello", "hey", "good morning", "good evening", 
                 "bonjour", "salut", "hola", "hallo", "namaste", "你好"}
    text_clean = text.lower().translate(str.maketrans('', '', string.punctuation)).strip()
    # Only return True if the entire input matches a greeting exactly
    return text_clean in greetings

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
            with st.chat_message("assistant"):
                st.markdown(result)
            speak_text(result)
    
    st.subheader("📄 Upload Report (PDF)")
    pdf_file = st.file_uploader("Analyze PDF", type=["pdf"])
    if pdf_file:
        with st.spinner("Extracting PDF..."):
            res = requests.post(f"{BACKEND_URL}/upload_pdf", files={"file": pdf_file})
            pdf_text = res.json().get("content", "")
            st.session_state.messages.append(("assistant", f"📘 Extracted Report:\n\n{pdf_text[:800]}..."))
            with st.chat_message("assistant"):
                st.markdown(f"📘 Extracted Report:\n\n{pdf_text[:800]}...")
            speak_text(pdf_text[:300])

    st.subheader("🎙️ Voice Question")
    audio = mic_recorder(start_prompt="🎤 Speak", stop_prompt="⏹️ Stop", just_once=True, key="voice")
    if audio:
        def transcribe_audio(audio_bytes, language_code="en-US"):
            recognizer = sr.Recognizer()
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
                    audio = AudioSegment.from_file(BytesIO(audio_bytes))
                    audio.export(wav_file.name, format="wav")
                with sr.AudioFile(wav_file.name) as source:
                    audio_data = recognizer.record(source)
                    return recognizer.recognize_google(audio_data, language=language_code)
            except Exception as e:
                return f"❌ Audio processing error: {e}"
        spoken_text = transcribe_audio(audio["bytes"], {
            "en": "en-US", "fr": "fr-FR", "es": "es-ES", "de": "de-DE", "hi": "hi-IN", "zh": "zh-CN"
        }[input_lang])
        st.session_state.messages.append(("user", spoken_text))

with col1:
    for role, msg in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(msg)

    user_input = st.chat_input("Type a medical question...")
    if user_input:
        st.session_state.messages.append(("user", user_input))

    # Send to backend and display answer
    if st.session_state.messages and st.session_state.messages[-1][0] == "user":
        user_msg = st.session_state.messages[-1][1]
        if is_greeting(user_msg):
            answer = {
                "en": "Hello! How can I assist you today?",
                "fr": "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
                "es": "¡Hola! ¿Cómo puedo ayudarte hoy?",
                "de": "Hallo! Wie kann ich Ihnen heute helfen?",
                "hi": "नमस्ते! मैं आज आपकी कैसे मदद कर सकता हूं?",
                "zh": "你好！我今天能帮您什么？"
            }.get(output_lang, "Hello! How can I assist you today?")
            st.session_state.messages.append(("assistant", answer))
            with st.chat_message("assistant"):
                st.markdown(answer)
            speak_text(answer)
        else:
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
                    "tone": tone,
                    "simplify": True
                }
                res = requests.post(f"{BACKEND_URL}/ask", data=payload)
                answer = res.json().get("answer", "❌ No response")
                translated_answer = translate_answer(answer, output_lang)
                st.session_state.messages.append(("assistant", translated_answer))
                with st.chat_message("assistant"):
                    st.markdown(translated_answer)
                speak_text(translated_answer)

    # After user input handling, add this block to process the last message if it was from voice
    if st.session_state.messages and st.session_state.messages[-1][0] == "user":
        user_msg = st.session_state.messages[-1][1]
        if is_greeting(user_msg):
            answer = {
                "en": "Hello! How can I assist you today?",
                "fr": "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
                "es": "¡Hola! ¿Cómo puedo ayudarte hoy?",
                "de": "Hallo! Wie kann ich Ihnen heute helfen?",
                "hi": "नमस्ते! मैं आज आपकी कैसे मदद कर सकता हूं?",
                "zh": "你好！我今天能帮您什么？"
            }.get(output_lang, "Hello! How can I assist you today?")
            st.session_state.messages.append(("assistant", answer))
            with st.chat_message("assistant"):
                st.markdown(answer)
            speak_text(answer)
        else:
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
                    "tone": tone,
                    "simplify": True
                }
                res = requests.post(f"{BACKEND_URL}/ask", data=payload)
                answer = res.json().get("answer", "❌ No response")
                translated_answer = translate_answer(answer, output_lang)
                st.session_state.messages.append(("assistant", translated_answer))
                with st.chat_message("assistant"):
                    st.markdown(translated_answer)
                speak_text(translated_answer)
