# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch

from .base_tirex import BaseTirexEmbeddingModel


class BaseTirexClassifier(BaseTirexEmbeddingModel):
    """Abstract base class for TiRex classification models."""

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels for input time series data.

        Args:
            x: Input time series data as torch.Tensor with shape
                (batch_size, num_variates, seq_len).
        Returns:
            torch.Tensor: Predicted class labels with shape (batch_size,).
        """
        emb = self._compute_embeddings(x)
        return torch.from_numpy(self.head.predict(emb)).long()

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities for input time series data.

        Args:
            x: Input time series data as torch.Tensor with shape
                (batch_size, num_variates, seq_len).
        Returns:
            torch.Tensor: Class probabilities with shape (batch_size, num_classes).
        """
        emb = self._compute_embeddings(x)
        return torch.from_numpy(self.head.predict_proba(emb))
