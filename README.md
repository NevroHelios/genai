# dlgenai — Audio Genre Classifier

EfficientNet-V2-S trained on mel-spectrograms to classify 10 music genres, using PyTorch Lightning and wandb.

## Setup

**Requirements:** Python 3.13, [uv](https://docs.astral.sh/uv/)

```bash
# Create venv and install dependencies
uv sync
```

> Without uv, use standard venv:
> ```bash
> python3.13 -m venv .venv
> source .venv/bin/activate
> pip install -e .
> ```

## Data

Place the competition data under `messy_mashup/`:
```
messy_mashup/
├── genres_stems/     # per-genre stem audio files (training)
├── mashups/          # mixed audio files (test)
├── test.csv
└── ESC-50-master/audio/  # background noise augmentation
```

## Train

```bash
# Log in to wandb (first time only)
wandb login

python train.py
# or 
uv run train.py
```

Checkpoints are saved locally by Lightning. The best checkpoint is also uploaded to Kaggle Hub (`nevrohelios/genre-classifier/pyTorch/best-checkpoint`).

## Config

All hyperparameters are in `config.py` (`CFG` class): batch size, learning rate, mel-spectrogram settings, epochs, etc.