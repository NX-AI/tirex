# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from dataclasses import dataclass

import torch


class StandardScaler:
    def __init__(self, eps: float = 1e-5, nan_loc: float = 0.0):
        self.eps = eps
        self.nan_loc = nan_loc

    def scale(
        self,
        x: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=self.nan_loc)
            scale = torch.nan_to_num(torch.nanmean((x - loc).square(), dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale == 0, torch.abs(loc) + self.eps, scale)
        else:
            loc, scale = loc_scale

        return ((x - loc) / scale), (loc, scale)

    def re_scale(self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        loc, scale = loc_scale
        return x * scale + loc


class Patcher:
    def __init__(self, patch_size: int, patch_stride: int, left_pad: bool):
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.left_pad = left_pad
        assert self.patch_size % self.patch_stride == 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        length = x.shape[-1]

        if length < self.patch_size or (length % self.patch_stride != 0):
            if length < self.patch_size:
                padding_size = (
                    *x.shape[:-1],
                    self.patch_size - (length % self.patch_size),
                )
            else:
                padding_size = (
                    *x.shape[:-1],
                    self.patch_stride - (length % self.patch_stride),
                )
            padding = torch.full(size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device)
            if self.left_pad:
                x = torch.concat((padding, x), dim=-1)
            else:
                x = torch.concat((x, padding), dim=-1)

        return x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)


@dataclass
class PatchedUniTokenizerState:
    scale_state: float


class PatchedUniTokenizer:
    def __init__(self, patch_size: int, patch_stride: int | None = None, scaler: StandardScaler | None = None):
        self.patch_size = patch_size
        self.patch_stride = patch_size if patch_stride is None else patch_stride
        self.scaler = StandardScaler() if scaler is None else scaler
        self.patcher = Patcher(self.patch_size, self.patch_stride, left_pad=True)

    def context_input_transform(self, data: torch.Tensor):
        assert data.ndim == 2
        data, scale_state = self.scaler.scale(data)
        return self.patcher(data), PatchedUniTokenizerState(scale_state)

    def output_transform(self, data: torch.Tensor, tokenizer_state: PatchedUniTokenizerState):
        data_shape = data.shape
        data = self.scaler.re_scale(data.reshape(data_shape[0], -1), tokenizer_state.scale_state).view(*data_shape)
        return data
