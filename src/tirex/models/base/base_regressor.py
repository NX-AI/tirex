# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch

from .base_tirex import BaseTirexEmbeddingModel


class BaseTirexRegressor(BaseTirexEmbeddingModel):
    """Abstract base class for TiRex regression models."""

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict values for input time series data.

        Args:
            x: Input time series data as torch.Tensor with shape
                (batch_size, num_variates, seq_len).
        Returns:
            torch.Tensor: Predicted values with shape (batch_size,).
        """
        emb = self._compute_embeddings(x)
        return torch.from_numpy(self.head.predict(emb)).float()
