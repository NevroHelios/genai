import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchaudio
from torchmetrics.classification import MulticlassF1Score
from torchvision import models
from typing import Any

from src.config import CFG



class GenreClassifierConformer(pl.LightningModule):
    def __init__(self, lr:float, label_smoothing:float=CFG.LABEL_SMOOTH,
                 num_classes:int = 10,
                 mixup_alpha=CFG.MIXUP_ALPHA,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.lr = lr
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=60)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)

        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.backbone = torchaudio.models.Conformer(
            input_dim=CFG.N_MELS,
            num_heads=8,
            ffn_dim=4 * CFG.N_MELS,
            num_layers=8,
            depthwise_conv_kernel_size=31
        )

        self.clf = nn.Sequential(
            nn.Linear(in_features=CFG.N_MELS, out_features=256),
            nn.SiLU(),
            nn.Linear(in_features=256, out_features=10)
        )

    def get_block(self, in_feat:int, out_feat:int):
        """retrieve basic blocks (can be upgraded)"""
        return [
            nn.Conv2d(in_channels=in_feat, out_channels=out_feat, kernel_size=3),
            nn.BatchNorm2d(num_features=out_feat),
            nn.ReLU()
        ]

    def forward(self, x):
        if self.training:
            x = self.time_mask(self.freq_mask(x[:, 0, :, :])) #[b, c, f, t]
        else:
            x = x[:, 0, :, :]
        x = x.permute(0, 2, 1) # [b, t, f]
        lengths = torch.full((x.size(0),), x.size(1), device=x.device)
        x, _ = self.backbone(x, lengths)
        x = x.mean(dim=1)
        return self.clf(x)

    def _mixup(self, x, y):
        lam = float(np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha))
        lam = max(0.55, min(0.95, lam))
        idx = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam

    def training_step(self, batch, batch_idx):
        x, y = batch # [b, c, f, t]
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
        #x = torch.squeeze(x, dim=1).permute(0, 2, 1)
        logits = self(x)
        self.val_f1.update(logits.argmax(1), y)
        self.log("val_loss", self.loss_fn(logits, y), on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters())
        schedular = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=100,
        )
        return {
            "optimizer" : optimizer,
            "lr_scheduler": {
                "scheduler": schedular,
                "interval": "step"
            }
        }
