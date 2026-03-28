"""dataset code for genre classification"""

import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import CFG, GENRE_MAP


def seed_everything(seed=42):
    """set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_audio(path, sr=CFG.SR):
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        return y
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.zeros(sr * 5, dtype=np.float32)


def to_log_mel(y, sr=CFG.SR):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=CFG.N_FFT, hop_length=CFG.HOP_LENGTH, n_mels=CFG.N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max)


def split_songs(stem_root: Path, seed: int = 42):
    rng = random.Random(seed)
    splits = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }

    for genre_dir in sorted(stem_root.glob("*")):
        genre = genre_dir.name
        if genre not in GENRE_MAP:
            continue
        songs = sorted(list(genre_dir.iterdir()))
        rng.shuffle(songs)
        n = len(songs)
        n_val = max(1, int(0.1 * n))
        n_test = max(1, int(0.1 * n))
        splits["train"][genre] = songs[: n - n_val - n_test]
        splits["val"][genre] = songs[n - n_val - n_test : n - n_test]
        splits["test"][genre] = songs[n - n_test :]

    for name, split in splits.items():
        print(f"  {name:5s}: {sum(len(v) for v in split.values())} songs")
    return splits


class MashupDataset(Dataset):
    """loads the wav files and augments them on the fly"""
    STEM_NAMES = ("vocals.wav", "bass.wav", "drums.wav", "other.wav")
    STEM_GAINS = {
        "vocals.wav": (0.7, 1.3),
        "bass.wav": (0.5, 1.1),
        "drums.wav": (0.6, 1.2),
        "other.wav": (0.4, 1.0),
    }

    def __init__(
        self,
        genre_to_songs,
        noise_dir,
        samples_per_epoch=4000,
        is_train=True,
        snr_db_range=(10, 30),
        n_noise_clips_range=(1, 3),
    ):
        self.genre_to_songs = genre_to_songs
        self.is_train = is_train
        self.snr_db_range = snr_db_range
        self.n_noise_clips_range = n_noise_clips_range
        self.noise_files = sorted(noise_dir.glob("*.wav"))

        genres = sorted(genre_to_songs.keys())
        per_genre = samples_per_epoch // len(genres)
        remainder = samples_per_epoch % len(genres)
        self.epoch_samples = []
        for i, g in enumerate(genres):
            self.epoch_samples.extend([g] * (per_genre + (1 if i < remainder else 0)))
        random.shuffle(self.epoch_samples)

    def __len__(self):
        return len(self.epoch_samples)

    def _mix_stems(self, genre):
        songs = self.genre_to_songs[genre]
        chosen = (
            random.sample(songs, 4) if len(songs) >= 4 else random.choices(songs, k=4)
        )

        stems = []
        for stem_name, song_dir in zip(self.STEM_NAMES, chosen):
            path = song_dir / stem_name
            if not path.exists():
                available = list(song_dir.glob("*.wav"))
                path = random.choice(available) if available else None

            y = (
                load_audio(path)
                if path
                else np.random.randn(CFG.SR * 10).astype(np.float32) * 0.01
            )

            if self.is_train and random.random() < 0.3:
                try:
                    y = librosa.effects.time_stretch(y, rate=random.uniform(0.9, 1.1))
                except Exception:
                    pass

            stems.append((y, stem_name))

        maxlen = max(len(y) for y, _ in stems)
        mix = np.zeros(maxlen, dtype=np.float32)
        for y, name in stems:
            lo, hi = self.STEM_GAINS.get(name, (0.5, 1.0))
            padded = np.pad(y, (0, maxlen - len(y)))[:maxlen]
            mix += random.uniform(lo, hi) * padded

        peak = np.abs(mix).max()
        return mix / peak if peak > 1e-6 else mix

    def _add_noise(self, mix):
        sig_rms = np.sqrt(np.mean(mix**2))
        result = mix.copy()
        for _ in range(random.randint(*self.n_noise_clips_range)):
            try:
                noise = load_audio(random.choice(self.noise_files))
                n_pad = np.zeros_like(mix)
                if len(noise) < len(mix):
                    start = random.randint(0, len(mix) - len(noise))
                    n_pad[start : start + len(noise)] = noise
                else:
                    n_pad[:] = noise[: len(mix)]
                snr = random.uniform(*self.snr_db_range)
                n_rms = max(np.sqrt(np.mean(n_pad**2)), 1e-8)
                result += n_pad * (sig_rms / (10 ** (snr / 20)) / n_rms)
            except Exception:
                continue
        peak = np.abs(result).max()
        return result / peak if peak > 1e-6 else result

    def __getitem__(self, idx):
        genre = self.epoch_samples[idx]
        label = GENRE_MAP[genre]
        try:
            mix = self._add_noise(self._mix_stems(genre))
            mel = to_log_mel(mix)
            T = mel.shape[-1]

            if T >= CFG.CROP_FRAMES:
                start = (
                    random.randint(0, T - CFG.CROP_FRAMES)
                    if self.is_train
                    else (T - CFG.CROP_FRAMES) // 2
                )
                mel = mel[:, start : start + CFG.CROP_FRAMES]
            else:
                mel = np.pad(mel, ((0, 0), (0, CFG.CROP_FRAMES - T)))

            mel = (mel - mel.mean()) / max(mel.std(), 1e-6)
            mel_t = torch.from_numpy(mel).float().unsqueeze(0).expand(3, -1, -1)
        except Exception:
            mel_t = torch.randn(3, CFG.N_MELS, CFG.CROP_FRAMES)

        return mel_t, torch.tensor(label, dtype=torch.long)
