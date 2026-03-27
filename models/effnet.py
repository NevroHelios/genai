import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchaudio
from torchmetrics.classification import MulticlassF1Score
from torchvision import models

from config import CFG


class GenreClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes=10,
        lr=CFG.LR,
        label_smoothing=CFG.LABEL_SMOOTH,
        mixup_alpha=CFG.MIXUP_ALPHA,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=60)

        backbone = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT
        )
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x):
        if self.training:
            x = self.time_mask(self.freq_mask(x[:, 0:1, :, :])).expand_as(x)
        return self.head(self.backbone(x))

    def _mixup(self, x, y):
        lam = float(np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha))
        lam = max(0.55, min(0.95, lam))
        idx = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y_a, y_b, lam = self._mixup(x, y)
        logits = self(x)
        loss = lam * self.loss_fn(logits, y_a) + (1 - lam) * self.loss_fn(logits, y_b)
        self.train_f1.update(logits.argmax(1), y_a if lam >= 0.5 else y_b)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.val_f1.update(logits.argmax(1), y)
        self.log("val_loss", self.loss_fn(logits, y), on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.val_f1.reset()

    def configure_optimizers(self):
        bp = list(self.backbone.parameters())
        mid = len(bp) // 2
        groups = [
            {"params": bp[:mid], "lr": self.hparams.lr * 0.1},
            {"params": bp[mid:], "lr": self.hparams.lr * 0.5},
            {"params": list(self.head.parameters()), "lr": self.hparams.lr},
        ]
        optimizer = torch.optim.AdamW(groups, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.hparams.lr * 0.1, self.hparams.lr * 0.5, self.hparams.lr],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=100,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
