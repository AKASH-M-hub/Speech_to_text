import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
from fpdf import FPDF
from docx import Document
import matplotlib.pyplot as plt
from collections import Counter
import tempfile, os, time, base64, re

# Mic recorder (works in browser)
from audio_recorder_streamlit import audio_recorder

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="EchoScribe AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Minimal stopwords (to avoid extra deps)
# ---------------------------
STOPWORDS = set("""
a an and are as at be by for from has he in is it its of on that the to was were will with your you i me my we our theirs ours
""".split())

# ---------------------------
# Styles
# ---------------------------
import streamlit as st

# Custom CSS for interactive theme
import streamlit as st

# Inject Custom Fonts
st.markdown(
    """
    <style>
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #1e3c72, #2a5298, #1e3c72);
        background-size: 400% 400%;
        animation: gradientMove 15s ease infinite;
        color: white !important;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Title */
    h1, h2, h3, h4, h5, h6, p {
        color: #f8f9fa !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        color: white;
        border: none;
        padding: 0.6em 1.2em;
        border-radius: 10px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(255, 65, 108, 0.7);
    }

    /* File uploader */
    .uploadedFile {
        background: rgba(255, 255, 255, 0.1);
        border: 2px dashed #ff416c;
        padding: 10px;
        border-radius: 10px;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# ---------------------------
# Session State
# ---------------------------
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {source, lang, text}
if "file_details" not in st.session_state:
    st.session_state.file_details = None

# ---------------------------
# Helpers
# ---------------------------
def clean_quotes(path_or_name: str) -> str:
    return path_or_name.strip().strip('"').strip("'")

def save_bytes_to_temp_wav(raw_bytes: bytes) -> str:
    """Assumes raw_bytes are WAV; writes to temp wav and returns path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(raw_bytes)
        return tmp.name

def convert_mp3_bytes_to_wav_path(mp3_bytes: bytes) -> str:
    """Converts MP3 bytes to temp WAV path using pydub (requires ffmpeg)."""
    audio = AudioSegment.from_file(BytesIO(mp3_bytes), format="mp3")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio.export(tmp.name, format="wav")
        return tmp.name

def transcribe_wav_path(wav_path: str, language: str) -> str:
    r = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            r.adjust_for_ambient_noise(source, duration=0.4)
            audio_data = r.record(source)
        return r.recognize_google(audio_data, language=language)
    except sr.UnknownValueError:
        return "Error: Could not understand the audio (maybe too noisy or unclear)."
    except sr.RequestError as e:
        return f"API Error: Could not request results; {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def make_txt_bytes(text: str) -> bytes:
    return text.encode("utf-8")

from fpdf import FPDF

def make_pdf_bytes(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)   # <-- Change "Arial" to "Helvetica"
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest="S").encode("latin-1")

def make_docx_bytes(text: str) -> bytes:
    doc = Document()
    for para in (text.split("\n") if text else [""]):
        doc.add_paragraph(para)
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()

def quick_summarize(text: str, max_sentences: int = 3) -> str:
    """
    Lightweight extractive summary: pick the most frequent-content sentences.
    No heavy NLP deps. Good enough for a glance.
    """
    if not text or len(text.split()) < 20:
        return text
    # Split sentences (naive)
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sents) <= max_sentences:
        return text
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    words = [w for w in words if w not in STOPWORDS]
    freq = Counter(words)
    scores = []
    for i, s in enumerate(sents):
        sw = [w.lower() for w in re.findall(r"[A-Za-z']+", s)]
        score = sum(freq.get(w, 0) for w in sw)
        scores.append((score, i, s))
    top = [s for _, _, s in sorted(scores, reverse=True)[:max_sentences]]
    # Preserve original order among chosen
    chosen = sorted([(i, s) for _, i, s in scores if s in top], key=lambda x: x[0])
    return " ".join(s for _, s in chosen)

def top_words(text: str, n: int = 10):
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    words = [w for w in words if w not in STOPWORDS]
    return Counter(words).most_common(n)

# ---------------------------
# Header
# ---------------------------
st.markdown("""
<div class="title-container">
  <h1>EchoScribe AI</h1>
  <h3>Upload or record audio • Choose language • Transcribe • Download as TXT / PDF / DOCX</h3>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar (Settings)
# ---------------------------
st.sidebar.header("Settings ⚙️")
language = st.sidebar.selectbox(
    "Transcription Language",
    [
        "en-US", "en-GB", "hi-IN", "ta-IN", "te-IN", "ml-IN",
        "fr-FR", "de-DE", "es-ES", "it-IT", "ja-JP", "ko-KR"
    ],
    index=0,
    help="Google Speech language code"
)
st.sidebar.caption("Tip: For MP3 support, ensure FFmpeg is installed on your system.")

# ---------------------------
# Layout
# ---------------------------
left, right = st.columns([1.2, 1.8], gap="large")

# ============ LEFT: INPUTS (Upload or Record) ============
with left:
    st.subheader("1) Add Audio", anchor=False)
    upload_box = st.container()
    rec_box = st.container()

    with upload_box:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        uploaded = st.file_uploader("📂 Upload audio (WAV or MP3)", type=["wav", "mp3"])
        do_transcribe = st.button("✨ Transcribe Uploaded Audio", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with rec_box:
        st.markdown('<div class="glass" style="margin-top:16px;">', unsafe_allow_html=True)
        st.write("🎤 Or record from microphone (works in browser)")
        audio_bytes = audio_recorder(pause_threshold=3.0, text="Start / Stop Recording", icon_size="3x")
        rec_transcribe = st.button("✨ Transcribe Recorded Audio", use_container_width=True)
        st.caption("Note: Recording happens in your browser. Audio is processed only to generate text.")
        st.markdown("</div>", unsafe_allow_html=True)

# ============ RIGHT: OUTPUTS ============
with right:
    st.subheader("2) Transcription & Exports", anchor=False)
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    # Handle uploaded file
    if uploaded and do_transcribe:
        with st.spinner("Processing audio…"):
            try:
                if uploaded.name.lower().endswith(".mp3"):
                    wav_path = convert_mp3_bytes_to_wav_path(uploaded.getvalue())
                else:
                    wav_path = save_bytes_to_temp_wav(uploaded.getvalue())

                ph = st.empty()
                for pct, msg in [(25, "Preparing audio…"), (55, "Recognizing speech…"), (85, "Finalizing…")]:
                    time.sleep(0.3)
                    ph.progress(pct, text=msg)

                text = transcribe_wav_path(wav_path, language)
                ph.empty()
                os.remove(wav_path)

                st.session_state.transcription = text
                st.session_state.history.append({
                    "source": f"Upload: {uploaded.name}",
                    "lang": language,
                    "text": text
                })
            except Exception as e:
                st.error(f"Error: {e}")

    # Handle recorded audio
    if audio_bytes and rec_transcribe:
        with st.spinner("Processing recorded audio…"):
            try:
                wav_path = save_bytes_to_temp_wav(audio_bytes)
                ph = st.empty()
                for pct, msg in [(25, "Preparing audio…"), (55, "Recognizing speech…"), (85, "Finalizing…")]:
                    time.sleep(0.3)
                    ph.progress(pct, text=msg)

                text = transcribe_wav_path(wav_path, language)
                ph.empty()
                os.remove(wav_path)

                st.session_state.transcription = text
                st.session_state.history.append({
                    "source": "Microphone",
                    "lang": language,
                    "text": text
                })
            except Exception as e:
                st.error(f"Error: {e}")

    # Show transcription (with simple typing effect)
    if st.session_state.transcription:
        txt = st.session_state.transcription
        holder = st.empty()
        shown = ""
        for ch in txt[:1000]:  # limit effect for long texts
            shown += ch
            holder.markdown(f"<div class='result-text'>{shown}▌</div>", unsafe_allow_html=True)
            time.sleep(0.005)
        holder.markdown(f"<div class='result-text'>{txt}</div>", unsafe_allow_html=True)

        # Summary
        st.markdown("##### ✨ Quick Summary")
        st.info(quick_summarize(txt, max_sentences=3) or "—")

        # Downloads
        col_a, col_b, col_c, col_d = st.columns([1,1,1,2])
        with col_a:
            st.download_button("⬇️ TXT", data=make_txt_bytes(txt), file_name="transcription.txt", mime="text/plain", use_container_width=True)
        with col_b:
            st.download_button("⬇️ PDF", data=make_pdf_bytes(txt), file_name="transcription.pdf", mime="application/pdf", use_container_width=True)
        with col_c:
            st.download_button("⬇️ DOCX", data=make_docx_bytes(txt), file_name="transcription.docx",
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               use_container_width=True)
        with col_d:
            if st.button("🧹 Clear", use_container_width=True):
                st.session_state.transcription = ""
                st.experimental_rerun()

        # Top words chart
        st.markdown("##### 🔎 Top Words")
        top = top_words(txt, n=10)
        if top:
            labels, vals = zip(*top)
            fig, ax = plt.subplots()
            ax.bar(labels, vals)
            ax.set_ylabel("Count")
            ax.set_xticklabels(labels, rotation=30, ha="right")
            st.pyplot(fig)
        else:
            st.caption("No significant words to visualize.")
    else:
        st.info("Your transcription will appear here after processing.")

    st.markdown("</div>", unsafe_allow_html=True)

# ============ HISTORY ============
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("3) History")
st.markdown('<div class="glass">', unsafe_allow_html=True)
if st.session_state.history:
    for item in st.session_state.history[::-1][:5]:
        st.write(f"**Source:** {item['source']}  •  **Lang:** {item['lang']}")
        st.code((item['text'] or "")[:800] + ("…" if len(item['text']) > 800 else ""), language="text")
else:
    st.caption("No transcriptions yet.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.caption("EchoScribe AI © 2025 • Built with Streamlit + Google Speech Recognition")
