# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from typing import NamedTuple

import torch


class StandardScalerState(NamedTuple):
    loc: torch.Tensor
    scale: torch.Tensor


class StandardScaler:
    def scale(self, x: torch.Tensor) -> tuple[torch.Tensor, StandardScalerState]:
        state = self.get_loc_scale(x)
        return ((x - state.loc) / state.scale), state

    def re_scale(self, x: torch.Tensor, state: StandardScalerState) -> torch.Tensor:
        return x * state.scale + state.loc

    def get_loc_scale(self, x: torch.Tensor, eps=1e-5):
        loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
        scale = torch.nan_to_num(torch.nanmean((x - loc).square(), dim=-1, keepdim=True).sqrt(), nan=1.0)
        scale = torch.where(scale == 0, torch.abs(loc) + eps, scale)
        return StandardScalerState(loc=loc, scale=scale)


class PatchedTokenizer:
    def __init__(self, patch_size: int):
        self.patch_size = patch_size
        self.scaler = StandardScaler()

    def input_transform(self, data: torch.Tensor) -> tuple[torch.Tensor, StandardScalerState]:
        assert data.ndim == 2
        assert data.shape[1] % self.patch_size == 0, "Length of data has to be a multiple of patch_size!"

        scaled_data, scale_state = self.scaler.scale(data)
        patched_data = scaled_data.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        return patched_data, scale_state

    def output_transform(self, data: torch.Tensor, scaler_state: StandardScalerState) -> torch.Tensor:
        assert data.shape[-1] == self.patch_size

        rescaled_data = self.scaler.re_scale(data.reshape(data.shape[0], -1), scaler_state)
        unpatched_data = rescaled_data.view(*data.shape[:-2], data.shape[-2] * self.patch_size)
        return unpatched_data
