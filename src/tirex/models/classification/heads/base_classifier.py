# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from abc import ABC, abstractmethod

import torch

from ..embedding import TiRexEmbedding


class BaseTirexClassifier(ABC):
    """Abstract base class for TiRex classification models.

    This base class provides common functionality for all TiRex classifiers,
    including embedding model initialization and a consistent interface.

    """

    def __init__(
        self, data_augmentation: bool = False, device: str | None = None, compile: bool = False, batch_size: int = 512
    ) -> None:
        """Initializes a TiRex classification model.

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
        pass

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels for input time series data.

        Args:
            x: Input time series data as torch.Tensor with shape
                (batch_size, num_variates, seq_len).
        Returns:
            torch.Tensor: Predicted class labels with shape (batch_size,).
        """
        self.emb_model.eval()
        x = x.to(self.device)
        embeddings = self.emb_model(x).cpu().numpy()
        return torch.from_numpy(self.head.predict(embeddings)).long()

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities for input time series data.

        Args:
            x: Input time series data as torch.Tensor with shape
                (batch_size, num_variates, seq_len).
        Returns:
            torch.Tensor: Class probabilities with shape (batch_size, num_classes).
        """
        self.emb_model.eval()
        x = x.to(self.device)
        embeddings = self.emb_model(x).cpu().numpy()
        return torch.from_numpy(self.head.predict_proba(embeddings))

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Saving model abstract method"""
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, path: str):
        """Loading model abstract method"""
        pass
