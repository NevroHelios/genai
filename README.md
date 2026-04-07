
# dlgenai — Audio Genre Classifier

EfficientNet-V7 trained on log-mel spectrograms to classify 10 music genres under distribution shift (clean stems → noisy mashups). Built with PyTorch Lightning and W&B.

**Kaggle private LB: 0.93030**

## Setup

**Requirements:** Python 3.13, [uv](https://docs.astral.sh/uv/)

```bash
uv sync
```

> Without uv:
> ```bash
> python3.13 -m venv .venv
> source .venv/bin/activate
> pip install -e .
> ```

## Data

Place the competition data under `messy_mashup/`:

```
messy_mashup/
├── genres_stems/         # per-genre stem audio files (training)
├── mashups/              # mixed audio files (test)
├── test.csv
└── ESC-50-master/audio/  # background noise for augmentation
```

## Train

```bash
wandb login   # first time only
uv run train.py
```

Checkpoints are saved locally by Lightning. The best checkpoint is also uploaded to Kaggle Hub (`nevrohelios/genre-classifier/pyTorch/best-checkpoint`).

## Config

All hyperparameters live in `config.py` (`CFG` class). See `report.pdf` for architecture details, augmentation design, and full results.

