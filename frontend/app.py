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
# Initialize analysis storage
if "image_analysis" not in st.session_state:
    st.session_state.image_analysis = ""
if "pdf_analysis" not in st.session_state:
    st.session_state.pdf_analysis = ""

st.title("AI medical assistant")
st.write("A friendly AI that helps you understand medical images and reports.")

# Language and tone selection
language_map = {
    "en": "English", "fr": "French", "es": "Spanish",
    "de": "German", "hi": "Hindi", "zh": "Chinese"
}

output_lang = st.selectbox("🌐 Response language:", list(language_map.keys()), format_func=lambda x: language_map[x])
input_lang = st.selectbox("🎙️ Voice input language:", list(language_map.keys()), format_func=lambda x: language_map[x])
simple_explanation = st.checkbox("📖 Simple explanation mode (for kids / non-experts)")
tone = st.selectbox("🧘 Tone:", ["formal", "friendly", "child"])

# Medical Expert Role selection
role_map = {
    "radiologist": "Radiologist (a doctor who looks at medical images like X-rays)",
    "general_physician": "General Physician (a doctor for general health problems)",
    "orthopedist": "Orthopedist (a doctor who treats bone and joint issues)",
    "cardiologist": "Cardiologist (a doctor who specializes in heart and blood vessels)",
    "neurologist": "Neurologist (a doctor who treats brain and nervous system disorders)",
    "dermatologist": "Dermatologist (a doctor who treats skin conditions)",
    "pediatrician": "Pediatrician (a doctor who treats children)",
    "dentist": "Dentist (a doctor who treats teeth and oral health)"
}
role = st.selectbox("👨‍⚕️ Medical Expert Role:", list(role_map.keys()), format_func=lambda x: role_map[x])

st.subheader("🖼️ Upload Image")
image_file = st.file_uploader("Analyze medical image", type=["png", "jpg", "jpeg"])
if image_file:
    from PIL import Image, UnidentifiedImageError
    from io import BytesIO
    try:
        image_bytes = image_file.read()
        img = Image.open(BytesIO(image_bytes))
        st.image(img, caption="Preview of uploaded image")
        valid_image = True
    except UnidentifiedImageError:
        st.error("❌ Error: Could not process image. The file is not a valid or supported image format.")
        valid_image = False
    except Exception as e:
        st.error(f"❌ Error: Could not process image. Reason: {e}")
        valid_image = False
    if valid_image:
        # Rewind file for backend upload
        image_file.seek(0)
        with st.spinner("Analyzing image..."):
            res = requests.post(f"{BACKEND_URL}/analyze_image", files={"image": image_file})
            if res.status_code == 200:
                try:
                    result = res.json().get("analysis", "")
                except Exception:
                    result = "❌ Error: Could not decode server response."
            else:
                result = f"❌ Error: Server returned status code {res.status_code}"
            st.session_state.image_analysis = result
            st.session_state.messages.append(("assistant", result))
            with st.chat_message("assistant"):
                st.markdown(result)
                speak_text(result)  # Always speak after analysis

st.subheader("📄 Upload Report (PDF)")
pdf_file = st.file_uploader("Analyze PDF", type=["pdf"])
if pdf_file:
    with st.spinner("Extracting PDF..."):
        res = requests.post(f"{BACKEND_URL}/upload_pdf", files={"file": pdf_file})
        pdf_text = res.json().get("content", "")
        st.session_state.pdf_analysis = pdf_text
        st.session_state.messages.append(("assistant", f"📘 Extracted Report:\n\n{pdf_text}"))
        with st.chat_message("assistant"):
            st.markdown(f"📘 Extracted Report:\n\n{pdf_text}")
            speak_text(f"📘 Extracted Report:\n\n{pdf_text}")  # Always speak after analysis

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

user_input = st.chat_input("Type a medical question...")
if user_input:
    st.session_state.messages.append(("user", user_input))
    user_msg = user_input
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
    else:
        with st.spinner("Thinking..."):
            # Build full chat history for context
            history = []
            for role_msg, msg in st.session_state.messages:
                if role_msg == "user":
                    history.append(f"User: {msg}")
                elif role_msg == "assistant":
                    history.append(f"Assistant: {msg}")
            # Add image and PDF analysis to context if available
            if st.session_state.image_analysis:
                history.append(f"Image Analysis: {st.session_state.image_analysis}")
            if st.session_state.pdf_analysis:
                history.append(f"PDF Analysis: {st.session_state.pdf_analysis}")
            chat_history = "\n".join(history)
            payload = {
                "question": user_msg,
                "context": chat_history,
                "tone": tone,
                "role": role,
                "simplify": True
            }
            res = requests.post(f"{BACKEND_URL}/ask", data=payload)
            answer = res.json().get("answer", "❌ No response")
            translated_answer = translate_answer(answer, output_lang)
            st.session_state.messages.append(("assistant", translated_answer))

# Display all messages in order (full chat history)
for idx, (role, msg) in enumerate(st.session_state.messages):
    with st.chat_message(role):
        st.markdown(msg)
        if role == "assistant":
            # Only speak if this is not the image or PDF analysis message and it's the last assistant message
            is_image_analysis = (msg == st.session_state.image_analysis)
            is_pdf_analysis = (msg == f"📘 Extracted Report:\n\n{st.session_state.pdf_analysis}")
            is_last_assistant = idx == max(i for i, (r, _) in enumerate(st.session_state.messages) if r == "assistant")
            if not (is_image_analysis or is_pdf_analysis) and is_last_assistant:
                speak_text(msg, key=f"assistant_{idx}")
