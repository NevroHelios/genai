from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
import kagglehub
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score
from tqdm.auto import tqdm

from config import CFG, IDX2GENRE
from dataset import load_audio, to_log_mel
from model import GenreClassifier


def run_sanity_check(train_loader, val_loader, device):
    print("\nSANITY CHECK")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(
            f"  [GPU]  {torch.cuda.get_device_name(0)}  |  VRAM: {props.total_memory / 1024**3:.1f} GB"
        )
    else:
        print("  [GPU]  No CUDA — running on CPU")

    for name, loader in [("train", train_loader), ("val", val_loader)]:
        mel, lbl = next(iter(loader))
        assert (
            mel.ndim == 4
            and mel.shape[1:3] == (3, CFG.N_MELS)
            and mel.shape[3] == CFG.CROP_FRAMES
        )
        assert not (torch.isnan(mel).any() or torch.isinf(mel).any())
        assert 0 <= lbl.min() and lbl.max() < 10
        print(
            f"  [{name:5s}]  shape={tuple(mel.shape)}  mean={mel.mean():.3f}  "
            f"std={mel.std():.3f}  labels={dict(sorted(Counter(lbl.tolist()).items()))}"
        )

    probe = GenreClassifier(num_classes=10).to(device).eval()
    dummy = torch.randn(4, 3, CFG.N_MELS, CFG.CROP_FRAMES).to(device)
    with torch.no_grad():
        out = probe(dummy)
    assert out.shape == (4, 10) and not torch.isnan(out).any()

    loss = nn.CrossEntropyLoss()(out, torch.randint(0, 10, (4,)).to(device))
    assert torch.isfinite(loss) and loss.item() < 10.0
    print(
        f"  [MODEL] OK | output={tuple(out.shape)}  [LOSS] CE={loss.item():.4f} (expected ~{np.log(10):.2f})"
    )

    del probe, dummy, out
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("  ✓ All checks passed\n")


def evaluate_local(model, dataset, device):
    model.eval()
    all_preds, all_labels = [], []
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    with torch.no_grad():
        for mel, label in tqdm(loader, desc="Eval"):
            all_preds.append(model(mel.to(device)).argmax(1).cpu())
            all_labels.append(label)

    preds, labels = torch.cat(all_preds), torch.cat(all_labels)
    macro_f1 = MulticlassF1Score(num_classes=10, average="macro")(preds, labels).item()

    per_class_scores = MulticlassF1Score(num_classes=10, average="none")(preds, labels)
    print("\n  Per-class F1:")
    per_class_dict = {}
    for i, s in enumerate(per_class_scores):
        print(f"    {IDX2GENRE[i]:12s}: {s:.4f} {'█' * int(s * 30)}")
        per_class_dict[f"test_f1_{IDX2GENRE[i]}"] = s.item()

    wandb.log({"test_f1_macro": macro_f1, **per_class_dict})
    return macro_f1


def predict_test(model, device, n_crops=7):
    def infer(path):
        y = load_audio(path)
        mel = to_log_mel(y)
        T = mel.shape[-1]
        starts = np.linspace(0, max(T - CFG.CROP_FRAMES, 0), n_crops).astype(int)
        crops = np.stack(
            [
                mel[:, s : s + CFG.CROP_FRAMES]
                if T >= CFG.CROP_FRAMES
                else np.pad(mel, ((0, 0), (0, CFG.CROP_FRAMES - T)))
                for s in starts
            ]
        )
        for i, _ in enumerate(crops):
            crops[i] = (crops[i] - crops[i].mean()) / max(crops[i].std(), 1e-6)

        t = (
            torch.from_numpy(crops)
            .float()
            .unsqueeze(1)
            .expand(-1, 3, -1, -1)
            .to(device)
        )
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ),
        ):
            logits = model(t)
        return IDX2GENRE[logits.float().mean(0).argmax().item()]

    test_df = pd.read_csv(CFG.TEST_CSV)
    model.eval()
    preds = [
        {
            "id": row["id"],
            "genre": infer(Path(CFG.ROOT_DIR) / "messy_mashup" / row["filename"]),
        }
        for _, row in tqdm(
            test_df.iterrows(), total=len(test_df), desc="Test inference"
        )
    ]
    submission = pd.DataFrame(preds)
    submission.to_csv(CFG.SUBMISSION, index=False)
    print(f"\nSaved {len(submission)} predictions → {CFG.SUBMISSION}")

    vc = submission["genre"].value_counts()
    print(vc)
    wandb.log(
        {
            "prediction_distribution": wandb.Table(
                columns=["genre", "count"], data=[[g, int(c)] for g, c in vc.items()]
            )
        }
    )
    return submission


def upload_to_kagglehub(ckpt_path: Path, best_score: float):
    print("\nKAGGLEHUB UPLOAD")
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"[WARN] Checkpoint not found at {ckpt_path} — skipping.")
        return
    try:
        kagglehub.model_upload(
            handle=CFG.KAGGLE_MODEL_HANDLE,
            local_model_dir=str(ckpt_path.parent),
            license_name="Apache 2.0",
            version_notes=(
                f"EfficientNet-V2-S | val_f1={best_score:.4f} | "
                f"epochs={CFG.MAX_EPOCHS} | lr={CFG.LR} | "
                f"mixup={CFG.MIXUP_ALPHA} | ls={CFG.LABEL_SMOOTH}"
            ),
        )
        print("Uploaded to KaggleHub")
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        raise
