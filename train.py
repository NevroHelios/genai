import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from config import CFG
from dataset import MashupDataset, seed_everything, split_songs
from models.effnet import GenreClassifier
from utils import evaluate_local, upload_to_kagglehub

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")


def build_loaders(splits):
    nw = CFG.NUM_WORKERS
    kwargs = dict(
        num_workers=nw,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
        pin_memory=True,
    )

    train_ds = MashupDataset(
        splits["train"],
        CFG.NOISE_DIR,
        samples_per_epoch=CFG.SAMPLES_PER_EPOCH_TRAIN,
        is_train=True,
    )
    val_ds = MashupDataset(
        splits["val"],
        CFG.NOISE_DIR,
        samples_per_epoch=CFG.SAMPLES_PER_EPOCH_VAL,
        is_train=False,
    )
    test_ds = MashupDataset(
        splits["test"],
        CFG.NOISE_DIR,
        samples_per_epoch=CFG.SAMPLES_PER_EPOCH_TEST,
        is_train=False,
    )

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, **kwargs
    )
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, **kwargs)
    return train_loader, val_loader, test_ds


def main():
    seed_everything(CFG.SEED)

    wandb.init(
        project=CFG.WANDB_PROJECT,
        name=CFG.WANDB_RUN_NAME,
        config={
            "architecture": "EfficientNet-V2-S",
            "sr": CFG.SR,
            "n_mels": CFG.N_MELS,
            "crop_frames": CFG.CROP_FRAMES,
            "batch_size": CFG.BATCH_SIZE,
            "max_epochs": CFG.MAX_EPOCHS,
            "lr": CFG.LR,
            "weight_decay": CFG.WEIGHT_DECAY,
            "label_smoothing": CFG.LABEL_SMOOTH,
            "mixup_alpha": CFG.MIXUP_ALPHA,
            "samples_train": CFG.SAMPLES_PER_EPOCH_TRAIN,
            "samples_val": CFG.SAMPLES_PER_EPOCH_VAL,
            "scheduler": "OneCycleLR",
        },
    )
    logger = WandbLogger(
        project=CFG.WANDB_PROJECT, name=CFG.WANDB_RUN_NAME, log_model=False
    )

    splits = split_songs(CFG.TRAIN_STEMS, seed=CFG.SEED)

    train_loader, val_loader, test_ds = build_loaders(splits)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GenreClassifier(num_classes=10, lr=CFG.LR)
    total = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  Params: {total:,} total | {train_p:,} trainable | {total * 4 / 1024**2:.1f} MB"
    )
    wandb.run.summary.update(
        {"model_params_total": total, "model_params_trainable": train_p}
    )

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_f1:.4f}",
        verbose=True,
    )
    trainer = pl.Trainer(
        max_epochs=CFG.MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_f1", patience=8, mode="max", min_delta=0.005
            ),
            ckpt_cb,
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_loader, val_loader)

    print("\nEvaluation on test split")
    best_path = ckpt_cb.best_model_path
    best_score = float(ckpt_cb.best_model_score)
    print(f"  Best ckpt: {best_path}  |  val_f1: {best_score:.4f}")
    wandb.run.summary["best_val_f1"] = best_score

    best_model = GenreClassifier.load_from_checkpoint(best_path).to(device).eval()
    local_f1 = evaluate_local(best_model, test_ds, device)
    print(f"  Local test F1: {local_f1:.4f}")
    wandb.run.summary["local_test_f1"] = local_f1

    upload_to_kagglehub(Path(best_path), best_score)
    wandb.finish()


if __name__ == "__main__":
    main()
