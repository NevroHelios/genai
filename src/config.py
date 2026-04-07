"""config file for genre classification"""

from pathlib import Path
import numpy as np


class CFG:
    """configs for genre classification"""
    ROOT_DIR = "."
    TRAIN_STEMS = Path(ROOT_DIR) / "messy_mashup" / "genres_stems"
    NOISE_DIR = Path(ROOT_DIR) / "messy_mashup" / "ESC-50-master" / "audio"
    TEST_DIR = Path(ROOT_DIR) / "messy_mashup" / "mashups"
    TEST_CSV = Path(ROOT_DIR) / "messy_mashup" / "test.csv"
    SUBMISSION = Path(ROOT_DIR) / "submission.csv"

    SR = 22050
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    DURATION_SEC = 10
    CROP_FRAMES = int(np.ceil(DURATION_SEC * SR / HOP_LENGTH)) + 1

    BATCH_SIZE = 32
    NUM_WORKERS = 40
    MAX_EPOCHS = 15
    LR = 3e-4
    WEIGHT_DECAY = 1e-3
    LABEL_SMOOTH = 0.1
    MIXUP_ALPHA = 0.25
    SEED = 42

    SAMPLES_PER_EPOCH_TRAIN = 4000
    SAMPLES_PER_EPOCH_VAL = 600
    SAMPLES_PER_EPOCH_TEST = 600

    WANDB_PROJECT = "genre-classifier"
    WANDB_RUN_NAME = "efficientnet-v2s-v3"
    KAGGLE_MODEL_HANDLE = "nevrohelios/genre-classifier/pyTorch/best-checkpoint"

    ## hf details
    HF_USERNAME = "NevroHelios"
    REPO_NAME = "genre-classifier"
    MODEL_PATH = "best-epoch30-val_f10.9093.ckpt"  # effnet b2-s

    filename = Path(MODEL_PATH).name
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"
    MODEL_DOWN_URL = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


    GENRE_MAP = {
        "blues": 0,
        "classical": 1,
        "country": 2,
        "disco": 3,
        "hiphop": 4,
        "jazz": 5,
        "metal": 6,
        "pop": 7,
        "reggae": 8,
        "rock": 9,
    }
    IDX2GENRE = {v: k for k, v in GENRE_MAP.items()}
