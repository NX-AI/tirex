# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from dataclasses import dataclass
from typing import TypedDict

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from ..util import EarlyStopping, set_seed, train_val_split


class TrainingMetrics(TypedDict):
    train_loss: float
    val_loss: float


@dataclass
class TrainConfig:
    # Training loop parameters
    max_epochs: int
    log_every_n_steps: int
    device: str

    # Optimizer parameters
    lr: float
    weight_decay: float

    # Loss parameters
    class_weights: torch.Tensor | None
    task_type: str

    # Earlystopping parameters
    patience: int
    delta: float

    # Reproducability
    seed: int | None

    # Data loading parameters
    batch_size: int
    val_split_ratio: float
    stratify: bool = False

    def __post_init__(self) -> None:
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")

        if self.log_every_n_steps <= 0:
            raise ValueError(f"log_every_n_steps must be positive, got {self.log_every_n_steps}")

        if self.lr <= 0:
            raise ValueError(f"lr (learning rate) must be positive, got {self.lr}")

        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if not (0 < self.val_split_ratio < 1):
            raise ValueError(f"val_split_ratio must be in (0, 1), got {self.val_split_ratio}")

        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")

        if self.delta < 0:
            raise ValueError(f"delta must be non-negative, got {self.delta}")

        if self.task_type not in ["classification", "regression"]:
            raise ValueError(f"task_type must be 'classification' or 'regression', got {self.task_type}")

        if self.stratify and self.task_type == "regression":
            raise ValueError("stratify=True is not valid for regression tasks")


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_config: TrainConfig,
    ) -> None:
        self.device = train_config.device
        self.train_config = train_config

        self.model = model.to(self.device)
        class_weights = (
            self.train_config.class_weights.to(self.device) if self.train_config.class_weights is not None else None
        )
        if self.train_config.task_type == "classification":
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(self.device)
        elif self.train_config.task_type == "regression":
            self.loss_fn = torch.nn.MSELoss().to(self.device)
        else:
            raise ValueError(f"Unsupported task_type: {self.train_config.task_type}")

        self.optimizer: Optimizer | None = None
        self.early_stopper = EarlyStopping(patience=self.train_config.patience, delta=self.train_config.delta)

    def fit(
        self, train_data: tuple[torch.Tensor, torch.Tensor], val_data: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> TrainingMetrics:
        if self.train_config.seed is not None:
            set_seed(self.train_config.seed)

        self._freeze_embedding()

        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.train_config.lr, weight_decay=self.train_config.weight_decay
            )

        train_loader, val_loader = self._create_data_loaders(train_data, val_data)

        for epoch in range(self.train_config.max_epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)

            self._log_epoch_metrics(epoch, train_loss, val_loss)

            stop_training = self.early_stopper(epoch=epoch + 1, val_loss=val_loss)
            if stop_training:
                break

        return {"train_loss": train_loss, "val_loss": val_loss}

    def _freeze_embedding(self) -> None:
        if hasattr(self.model, "emb_model"):
            for param in self.model.emb_model.parameters():
                param.requires_grad = False

    def _train_epoch(self, train_loader: DataLoader) -> float:
        train_loss = []
        self.model.train()
        for batch in train_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            y_hat = self.model.head(x)  # Only classification head is involved, embeddings are precomputed
            loss = self.loss_fn(y_hat, y)

            loss.backward()
            self.optimizer.step()

            train_loss.append(loss.detach())

        return torch.stack(train_loss).mean().item()

    @torch.inference_mode()
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        self.model.eval()
        val_loss = []
        for batch in val_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            y_hat = self.model.head(x)  # Only classification head is involved, embeddings are precomputed
            loss = self.loss_fn(y_hat, y)

            val_loss.append(loss.detach())
        return torch.stack(val_loss).mean().item()

    def _create_data_loaders(
        self, train_data: tuple[torch.Tensor, torch.Tensor], val_data: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[DataLoader, DataLoader]:
        if val_data is None:
            train_data, val_data = train_val_split(
                train_data, self.train_config.val_split_ratio, self.train_config.stratify, self.train_config.seed
            )

        train_embeddings = self.model.emb_model(train_data[0])
        val_embeddings = self.model.emb_model(val_data[0])

        train_loader = DataLoader(
            TensorDataset(train_embeddings, train_data[1]),
            batch_size=self.train_config.batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            TensorDataset(val_embeddings, val_data[1]),
            batch_size=self.train_config.batch_size,
            shuffle=False,
        )
        return train_loader, val_loader

    def _log_epoch_metrics(self, epoch: int, train_loss: float, val_loss: float) -> None:
        if epoch % self.train_config.log_every_n_steps == 0:
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
