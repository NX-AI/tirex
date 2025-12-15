# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from abc import ABC, abstractmethod

import numpy as np
import torch

from tirex.util import train_val_split

from ..embedding import TiRexEmbedding


class BaseTirexEmbeddingModel(ABC):
    """Abstract base class for TiRex models.

    This base class provides common functionality for all TiRex classifier and regression models,
    including embedding model initialization and a consistent interface.

    """

    def __init__(
        self, data_augmentation: bool = False, device: str | None = None, compile: bool = False, batch_size: int = 512
    ) -> None:
        """Initializes a base TiRex model.

        This base class initializes the embedding model and common configuration
        used by both classification and regression models.

        Args:
            data_augmentation : bool
                Whether to use data_augmentation for embeddings (sample statistics and first-order differences of the original data). Default: False
            device : str | None
                Device to run the embedding model on. If None, uses CUDA if available, else CPU. Default: None
            compile: bool
                Whether to compile the frozen embedding model. Default: False
            batch_size : int
                Batch size for embedding calculations. Default: 512
        """

        # Set device
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._compile = compile

        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.emb_model = TiRexEmbedding(
            device=self.device,
            data_augmentation=self.data_augmentation,
            batch_size=self.batch_size,
            compile=self._compile,
        )

    @abstractmethod
    def fit(self, train_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Abstract method for model training"""
        pass

    def _compute_embeddings(self, x: torch.Tensor) -> np.ndarray:
        """Compute embeddings for input time series data.

        Args:
            x: Input time series data as torch.Tensor with shape
                (batch_size, num_variates, seq_len).

        Returns:
            np.ndarray: Embeddings with shape (batch_size, embedding_dim).
        """
        self.emb_model.eval()
        x = x.to(self.device)
        return self.emb_model(x).cpu().numpy()

    def _create_train_val_datasets(
        self,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor] | None = None,
        val_split_ratio: float = 0.2,
        stratify: bool = False,
        seed: int | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        if val_data is None:
            train_data, val_data = train_val_split(
                train_data=train_data, val_split_ratio=val_split_ratio, stratify=stratify, seed=seed
            )
        return train_data, val_data

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save model to file.

        Args:
            path: File path where the model should be saved.
        """
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, path: str):
        """Load model from file.

        Args:
            path: File path to the saved model checkpoint.

        Returns:
            Instance of the model class with loaded weights and configuration.
        """
        pass
