# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cell import sLSTMBlockConfig, sLSTMCell


class sLSTMLayer(nn.Module):
    def __init__(self, config: sLSTMBlockConfig, backend: str):
        super().__init__()
        self.config = config

        in_features, num_heads = self.config.embedding_dim, self.config.num_heads
        self.fgate = LinearHeadwiseExpand(in_features, num_heads)
        self.igate = LinearHeadwiseExpand(in_features, num_heads)
        self.zgate = LinearHeadwiseExpand(in_features, num_heads)
        self.ogate = LinearHeadwiseExpand(in_features, num_heads)

        self.slstm_cell = sLSTMCell(self.config, backend)
        self.group_norm = MultiHeadLayerNorm(ndim=in_features, num_heads=num_heads)

    def forward(self, x: torch.Tensor, slstm_state: torch.Tensor | None = None) -> torch.Tensor:
        x_g = torch.cat((self.fgate(x), self.igate(x), self.zgate(x), self.ogate(x)), dim=-1)

        y, slstm_state = self.slstm_cell(x_g, state=slstm_state)

        return self.group_norm(y).transpose(1, 2).view(x.shape[0], x.shape[1], -1)


class LinearHeadwiseExpand(nn.Module):
    def __init__(self, in_features, num_heads, expand_factor_up: float = 1):
        super().__init__()
        assert num_heads <= in_features, "num_heads must be <= in_features"
        assert in_features % num_heads == 0, "in_features must be a multiple of num_heads"
        self.num_heads = num_heads

        out_features = round(expand_factor_up * in_features)
        out_features_per_head = out_features // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, out_features_per_head, in_features // num_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(*shape[:-1], self.num_heads, -1)
        x = torch.einsum("...hd,hod->...ho", x, self.weight)
        x = x.reshape(*shape[:-1], -1)
        return x


class MultiHeadLayerNorm(nn.Module):
    def __init__(self, ndim: int, num_heads: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim))
        self.num_heads = num_heads

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = input.shape

        assert NH == self.num_heads
        gn_in_1 = input.transpose(1, 2)  # (B, S, NH, DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)  # (B * S, NH * DH)
        residual_weight = 1.0 + self.weight
        out = F.group_norm(gn_in_2, num_groups=self.num_heads, weight=residual_weight)
        # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out
