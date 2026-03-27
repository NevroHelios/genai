import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchaudio
from torchmetrics.classification import MulticlassF1Score
from torchvision import models
from typing import Any

from config import CFG



class GenreClassifierCNN(pl.LightningModule):
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

        blocks = []
        in_feat = 3
        out_feat = 64
        for _ in range(5):
            blocks.extend(self.get_block(in_feat, out_feat))
            in_feat = out_feat
            out_feat *= 2

        self.model = nn.Sequential(
            *blocks,
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(),
            nn.Linear(in_features=in_feat, out_features=self.num_classes)
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
            x = self.time_mask(self.freq_mask(x[:, 0:1, :, :])).expand_as(x)
        return self.model(x)

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
        return torch.optim.Adam(self.parameters())

