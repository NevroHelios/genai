import kagglehub
import streamlit as st
import torchaudio
import librosa
import requests
import torch
import io
import pandas as pd
from typing import Dict
import uuid
import tempfile

from src.models.effnet import GenreClassifier
from src.utils import predict
from src.config import CFG

st.set_page_config(page_title="Messy Audio", page_icon="🎵", layout="centered", initial_sidebar_state='expanded')

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

def show_results(results: Dict):
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

def download_audio(url, vid_id):
    yt = YouTube(url, 'WEB')  # auto-generates POT via bundled node
    stream = yt.streams.get_audio_only()
    with tempfile.TemporaryDirectory() as tmp:
        path = stream.download(output_path=tmp, filename=f"{vid_id}.m4a")
        return path



tab_upload, tab_record, tab_sample = st.tabs([
    "📁 Upload Files", "🎙️ Record", "🎧 Try Samples"
])

col1, col2 = st.columns(2)

results = {}
with tab_upload:
    results.clear()
    st.caption("Upload an audio file and run inference with a cached model.")
    uploaded = st.file_uploader(
        "Drop an audio file",
        type=["wav", "mp3", "ogg", "flac", "m4a"],
        accept_multiple_files=True,
    )
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
            show_results(results=results)

with tab_record:
    results.clear()
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
            show_results(results)

SAMPLES = {
    "Classical - Victor Herbert _Venetian_Love_Song": "samples/classical.ogg",
}

with tab_sample:
    results.clear()
    st.caption("Pick a sample clip and see what the model predicts.")
    chosen = st.radio("Choose a sample", list(SAMPLES.keys()), horizontal=True)
    sample_path = SAMPLES[chosen]
    st.audio(sample_path)
    if st.button("Classify Sample", type="primary"):
        with st.spinner(text="cooking"):
            y, sr = librosa.load(sample_path, sr=CFG.SR, mono=True)
            res = predict(y=y, model=model, device=DEVICE)
            results[chosen] = res
        show_results(results)

with st.sidebar:
    st.markdown("### 🎵 Music Genre Classification — Messy Mashup")
    st.markdown("""
    Built on top of my submission for the **Messy Mashup** Kaggle competition —
    classifying audio into 10 genres: Blues, Classical, Country, Disco, Hip-Hop,
    Jazz, Metal, Pop, Reggae, and Rock.
    """)

    st.divider()

    st.markdown("#### 🏗️ What I built")
    st.markdown("""
    Training audio came as **separated stems** (vocals, bass, drums, other) while
    test audio was fully mixed mashups with additive noise — a deliberate distribution
    shift the model had to generalise across.

    I built a **stem-mixing augmentation pipeline** that dynamically recombines stems
    across songs at runtime with randomised per-stem gain, time-stretching (±10%), and
    ESC-50 noise injection at randomised SNR (10–30 dB). Every training sample is
    synthesised on-the-fly, giving effectively unbounded variety from just 800 source songs.

    Six architectures were trained: a **custom CNN from scratch**, a **Conformer**, and
    four **EfficientNet** variants (V2-S ×2, V5, V7). The final model is
    **EfficientNet-V7** (65M params), trained for 41 epochs with differential learning
    rates and 7-crop TTA at inference.
    """)

    st.divider()

    st.markdown("#### ⚙️ Technical stack")
    st.markdown("""
    - **Framework:** PyTorch Lightning + W&B
    - **Backbone:** EfficientNet-V7 (ImageNet pretrained, differential LR fine-tuning)
    - **Input:** Log-mel spectrogram (128 mel bins, 10s crops, 3-channel replication)
    - **Augmentation:** Cross-song stem mixing · Time stretch · ESC-50 noise · SpecAugment · Mixup (α=0.25)
    - **Head:** Dropout(0.4) → Linear(→512) → SiLU → Dropout(0.25) → Linear(→10)
    - **Scheduler:** OneCycleLR with cosine annealing
    - **TTA:** 7-crop temporal averaging at inference
    """)

    st.divider()

    st.markdown("#### 📊 Results")
    st.table({
        "Metric": [
            "Val Macro F1 (best epoch)",
            "Kaggle Public LB",
            "Kaggle Private LB",
            "Hardest classes",
            "Easiest classes",
        ],
        "Score": [
            "0.9268",
            "0.93911",
            "0.93030",
            "Country (0.78), Rock (0.77)",
            "Classical (0.98), Metal (0.97)",
        ]
    })

    st.divider()

    st.markdown("#### 👤 Author")
    st.markdown("[nevrohelios](https://github.com/nevrohelios) · IIT Madras BS · Data Science")