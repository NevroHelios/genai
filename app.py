import kagglehub
import streamlit as st
import torchaudio
import librosa
import requests
import torch
import io
import pandas as pd

from src.models.effnet import GenreClassifier
from src.utils import predict
from src.config import CFG

st.set_page_config(page_title="Messy Audio", page_icon="🎵", layout="centered")

MODEL_PATH = 'best.ckpt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'wb') as f:
        f.write(requests.get(CFG.MODEL_DOWN_URL).content)
    model = GenreClassifier.load_from_checkpoint(MODEL_PATH, map_location=DEVICE)
    model.eval()
    return model

model = load_model()

st.title("🎵 Audio Inference App")
st.caption("Upload an audio file and run inference with a cached model.")

uploaded = st.file_uploader(
    "Drop an audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    accept_multiple_files=True,
)

results = {}
if uploaded:
    with st.expander(label="Audio Files"):
        for upload in uploaded:
            st.write(upload.name)
            st.audio(upload, format=upload.type)
    if st.button(label="Find Genre", type='primary'):
        with st.spinner(text="cooking"):
            for upload in uploaded:
                y, sr = librosa.load(io.BytesIO(upload.read()), sr=CFG.SR, mono=True)
                res = predict(y=y, model=model, device=DEVICE)
                results[upload.name] = res #  ('genre', probs)

audio = st.audio_input(label="record", sample_rate=CFG.SR)
if audio:
    y, sr = librosa.load(io.BytesIO(audio.read()), sr=CFG.SR, mono=True)
    duration = len(y) / sr
    if duration < 5.0:
        st.warning(f"Recording is {duration:.1f}s — need at least 5s.")
    else:
        with st.spinner(text="cooking"):
            res = predict(y=y, model=model, device=DEVICE)
            results["recorded_audio"] = res

with st.sidebar:
    for name, res in results.items():
        st.write(name, " -> ", res[0])
        with st.expander("Probabilities"):
            genre, probs = res
            pairs = sorted(
                zip(CFG.GENRE_MAP.keys(), probs),
                key=lambda x: x[1], reverse=True
            )
            for g, p in pairs:
                col1, col2 = st.columns([3, 1])
                col1.progress(float(p), text=g)
                col2.write(f"{p:.1%}")
