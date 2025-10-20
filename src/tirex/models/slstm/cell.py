# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import warnings
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from tirex.util import dataclass_from_dict


@dataclass
class sLSTMBlockConfig:
    embedding_dim: int
    num_heads: int
    ffn_proj_factor: float = 2.6667
    num_states: int = 4
    num_gates: int = 4

    @property
    def head_dim(self):
        return self.embedding_dim // self.num_heads


class sLSTMCell(nn.Module):
    def __init__(self, config: sLSTMBlockConfig, backend: Literal["torch", "cuda"]):
        super().__init__()
        assert backend in ["torch", "cuda"], f"Backend can either be torch or cuda, not {backend}!"
        self.config = config
        self.backend = backend

        self._recurrent_kernel_ = nn.Parameter(
            torch.empty((config.num_heads, config.head_dim, config.num_gates * config.head_dim), dtype=None)
        )

        self._bias_ = nn.Parameter(torch.empty((config.num_heads * config.num_gates * config.head_dim), dtype=None))

        self._impl_forward_torch = sLSTMCellTorch.slstm_forward

    def forward(self, input: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input = self._get_input(input)
        state = self._get_state(input, state)

        if self.backend == "torch":
            output, state = self._impl_torch(input, state)
        elif self.backend == "cuda":
            output, state = self._impl_cuda(input, state)

        return self._permute_output(output).to(input.dtype), state.to(input.dtype)

    def _impl_torch(self, input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        input = input.to(dtype=torch.bfloat16)
        state = state.to(dtype=torch.bfloat16)
        recurrent_kernel = self._recurrent_kernel_.to(dtype=torch.bfloat16)
        bias = self._bias_.to(dtype=torch.float32)

        input = input.view(input.shape[0], input.shape[1], -1)
        bias = (
            bias.reshape(self.config.num_heads, self.config.num_gates, self.config.head_dim)
            .permute(1, 0, 2)
            .reshape(-1)
        )

        return self._impl_forward_torch(input, state, recurrent_kernel, bias)

    def _impl_cuda(self, input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if input.device.type != "cuda":
            warnings.warn(
                f"You use TiRex with sLSTM CUDA kernels BUT DO NOT LOAD THE DEVICE ON A CUDA DEVICE (device type is {input.device.type})!"
                "This is not supported and calls to the model will likely lead to an error if you dont move your model to a CUDA device!"
                "If you want to run TiRex on CPU you need to disable sLSTM CUDA kernels but be aware of the downsides (see FAQ)"
            )

        if not hasattr(self, "func"):
            try:
                from xlstm.blocks.slstm.cell import sLSTMCellConfig as sLSTMCellConfigCuda, sLSTMCellFuncGenerator
            except ModuleNotFoundError:
                raise ValueError(
                    'xlstm package not found! To use the custom cuda backend, install the additional dependencies with: pip install -e ".[cuda]"'
                )
            cuda_config = dataclass_from_dict(
                sLSTMCellConfigCuda, {**asdict(self.config), "hidden_size": self.config.embedding_dim}
            )
            self.func = sLSTMCellFuncGenerator(False, cuda_config)

        input = input.permute(0, 1, 3, 2, 4).reshape(input.shape[0], input.shape[1], -1)

        all_states = self.func.apply(
            False,
            input.contiguous(),
            state.contiguous(),
            self._recurrent_kernel_.contiguous(),
            self._bias_.contiguous(),
        )

        state = all_states[:, -1]
        output = all_states[0][1:]
        return output, state

    def _get_input(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.config.embedding_dim * self.config.num_gates, (
            f"Input size mismatch: Expected input size {self.config.embedding_dim * self.config.num_gates}, but got {x.size(-1)}."
        )
        return x.view(x.shape[0], x.shape[1], self.config.num_gates, self.config.num_heads, -1).permute(1, 0, 2, 3, 4)

    def _get_state(self, input: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        B = input.shape[1]
        if state is None:
            state = torch.zeros(
                (self.config.num_states, B, self.config.embedding_dim),
                dtype=input.dtype,
                device=input.device,
            )

        assert state.shape == (self.config.num_states, B, self.config.embedding_dim)
        return state

    def _permute_output(self, output: torch.Tensor) -> torch.Tensor:
        output = output.view(output.shape[0], output.shape[1], self.config.num_heads, self.config.head_dim)
        return output.permute(1, 2, 0, 3)


class sLSTMCellTorch:
    @staticmethod
    def slstm_forward(
        x: torch.Tensor,  # [S, B, G*I]
        states: torch.Tensor,  # [4, B, H] only the first is used for recurrence!
        R: torch.Tensor,  # [K, R*H, H] - K num_heads
        b: torch.Tensor,  # [T*H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_gates = 4
        num_heads = R.shape[0]
        S, B, _ = x.shape
        H = R.shape[1] * num_heads
        assert states.shape == (num_gates, B, H)

        states = states.to(R.dtype).unbind(dim=0)
        output = []
        for i in range(S):
            Ry = (
                states[0]
                .reshape(B, num_heads, 1, -1)
                .matmul(R.unsqueeze(0))
                .reshape(B, num_heads, num_gates, -1)
                .transpose(1, 2)
                .reshape(B, -1)
            )
            states = sLSTMCellTorch.slstm_forward_pointwise(
                x[i].float(), Ry.float(), b.float(), [s.float() for s in states]
            )
            states = [s.to(dtype=R.dtype) for s in states]
            output.append(states[0])

        return torch.stack(output), torch.stack(states)  # (S, B, H), 4 x (B, H)

    @staticmethod
    def slstm_forward_pointwise(
        Wx: torch.Tensor,  # dim [B, 4*H]
        Ry: torch.Tensor,  # dim [B, 4*H]
        b: torch.Tensor,  # dim [1, 4*H]
        states: torch.Tensor,  # dim 4 x [B, H]
    ) -> list[torch.Tensor]:
        y, c, n, m = states

        raw = Wx + Ry + b
        iraw, fraw, zraw, oraw = torch.unbind(raw.view(raw.shape[0], 4, -1), dim=1)

        # Equations reference the xlstm paper on page 4: https://arxiv.org/pdf/2405.04517
        logfplusm = m + F.logsigmoid(torch.clamp(fraw, max=15))  # eq 15 # Clamp to avoid subnomals
        mnew = torch.where(torch.all(n == 0.0), iraw, torch.max(iraw, logfplusm))  # eq 15
        ogate = torch.sigmoid(oraw)  # eq 14
        igate = torch.minimum(torch.exp(iraw - mnew), torch.ones_like(iraw))  # eq 16
        fgate = torch.minimum(torch.exp(logfplusm - mnew), torch.ones_like(iraw))  # eq 17
        zgate = torch.tanh(zraw)  # eq 11
        cnew = fgate * c + igate * zgate  # eq 8
        nnew = fgate * n + igate  # eq 9
        hnew = ogate * cnew / nnew  # eq 10

        return [hnew, cnew, nnew, mnew]  # 4 x (B, H)
