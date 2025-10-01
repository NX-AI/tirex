# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
import torch.nn as nn
import torch.nn.functional as F

from tirex.models.slstm.layer import sLSTMBlockConfig, sLSTMLayer
from tirex.util import round_up_to_next_multiple_of


class sLSTMBlock(nn.Module):
    def __init__(self, config: sLSTMBlockConfig, backend: str):
        super().__init__()
        self.config = config
        self.norm_slstm = RMSNorm(config.embedding_dim)
        self.slstm_layer = sLSTMLayer(config, backend)
        self.norm_ffn = RMSNorm(config.embedding_dim)

        up_proj_dim = round_up_to_next_multiple_of(config.embedding_dim * config.ffn_proj_factor, 64)
        self.ffn = FeedForward(config.embedding_dim, up_proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.slstm_layer(self.norm_slstm(x), slstm_state=None)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, up_proj_dim: int):
        super().__init__()
        self.proj_up_gate = nn.Linear(embedding_dim, up_proj_dim, bias=False)
        self.proj_up = nn.Linear(embedding_dim, up_proj_dim, bias=False)
        self.proj_down = nn.Linear(up_proj_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.proj_up_gate(x)) * self.proj_up(x)
        x = self.proj_down(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._rms_normalize(x.float()).to(x.dtype)
        x = x * self.weight
        return x

    def _rms_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
