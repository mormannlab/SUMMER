"""
Small feedforward model for next-bin spike prediction.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl


class LinearNextBin(pl.LightningModule):
    """
    Two-layer linear network: Linear -> ReLU -> Linear.
    Predicts next-bin spike counts from current-bin counts.
    """

    def __init__(self, n_units, hidden_size=64, lr=1e-3):
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr

        self.model = nn.Sequential(
            nn.Linear(n_units, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_units),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_fn(out, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_fn(out, y)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_fn(out, y)
        self.log("test_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LinearLabel(pl.LightningModule):
    """
    Two-layer feedforward classifier: Linear -> ReLU -> Linear -> n_classes.
    Predicts frame-level annotation label from binned spike counts (one bin = one frame).
    """

    def __init__(self, n_units, n_classes, hidden_size=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = nn.Sequential(
            nn.Linear(n_units, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
