# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import warnings
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tirex.util import dataclass_from_dict


@dataclass
class sLSTMBlockConfig:
    embedding_dim: int
    num_heads: int
    num_blocks: int
    ffn_proj_factor: float = 2.6667

    num_states: int = 4  # this is for the sLSTM, a standard LSTM  has 2
    num_gates: int = 4

    @property
    def head_dim(self):
        return self.embedding_dim // self.num_heads


class sLSTMCell(nn.Module):
    def __init__(self, config: sLSTMBlockConfig, backend: str):
        super().__init__()
        self.config = config
        self.backend = backend

        self._recurrent_kernel_ = nn.Parameter(
            torch.empty((config.num_heads, config.head_dim, config.num_gates * config.head_dim), dtype=None)
        )

        self._bias_ = nn.Parameter(torch.empty((config.num_heads * config.num_gates * config.head_dim), dtype=None))

    def forward(self, input: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input = self._get_input(input)
        state = self._get_state(input, state)

        if self.backend == "torch":
            all_states = self._impl_torch(input, state)
        elif self.backend == "cuda":
            all_states = self._impl_cuda(input, state)

        state = all_states[:, -1]
        output = self._permute_output(all_states[0][1:])
        return output.to(input.dtype), state.to(input.dtype)

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

        return slstm_forward(input, state, recurrent_kernel, bias)[0]

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

        return self.func.apply(
            False,
            input.contiguous(),
            state.contiguous(),
            self._recurrent_kernel_.contiguous(),
            self._bias_.contiguous(),
        )

    def _get_input(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.config.embedding_dim * self.config.num_gates, (
            f"Input size mismatch: Expected input size {self.config.embedding_dim * self.config.num_gates}, but got {input.size(-1)}."
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


def slstm_forward(
    x: torch.Tensor,  # [S, B, G*I]
    states: torch.Tensor,  # [4, B, H] only the first is used for recurrence!
    R: torch.Tensor,  # [K, R*H, H] - K num_heads
    b: torch.Tensor,  # [T*H]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_states = states.shape[0]
    sequence_dim = x.shape[0]
    # this only works for a fully-connected RNN, for a hin change this
    num_gates_r = R.shape[2] // R.shape[1]
    hidden_dim = R.shape[1] * R.shape[0]
    batch_dim = x.shape[1]
    num_heads = R.shape[0]

    assert batch_dim == states.shape[1]
    assert hidden_dim == states.shape[2]

    states_all = torch.zeros(
        [num_states, sequence_dim + 1, batch_dim, hidden_dim],
        device=x.device,
        dtype=x.dtype,
    )
    states_all[:, 0] = states
    for i, Wx_t in enumerate(x.unbind(dim=0)):
        Ry = (
            states[0]
            .reshape(batch_dim, num_heads, 1, -1)
            .matmul(R.unsqueeze(0))
            .reshape(batch_dim, num_heads, num_gates_r, -1)
            .transpose(1, 2)
            .reshape(batch_dim, -1)
        )
        sdtype = states.dtype
        Wx_t, Ry, b, states = Wx_t.float(), Ry.float(), b.float(), states.float()
        states, gates = slstm_forward_pointwise(Wx_t, Ry, b, states)
        states = states.to(dtype=sdtype)
        states_all[:, i + 1] = states

    # shapes ([S, B, H], ([B,H], [B,H], [B,H])
    return states_all, states


def slstm_forward_pointwise(
    Wx: torch.Tensor,  # dim [B, 4*H]
    Ry: torch.Tensor,  # dim [B, 4*H]
    b: torch.Tensor,  # dim [1, 4*H]
    states: torch.Tensor,  # dim [4, B, H]
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = Wx + Ry + b
    y, c, n, m = torch.unbind(states.view(4, states.shape[1], -1), dim=0)

    iraw, fraw, zraw, oraw = torch.unbind(raw.view(raw.shape[0], 4, -1), dim=1)
    # with torch.no_grad():  # THE difference to maxg aka max_gradient (here max / max_static)
    logfplusm = m + F.logsigmoid(fraw)
    if torch.all(n == 0.0):
        mnew = iraw
    else:
        mnew = torch.max(iraw, logfplusm)
    ogate = torch.sigmoid(oraw)
    igate = torch.minimum(torch.exp(iraw - mnew), torch.ones_like(iraw))
    fgate = torch.minimum(torch.exp(logfplusm - mnew), torch.ones_like(iraw))
    cnew = fgate * c + igate * torch.tanh(zraw)
    nnew = fgate * n + igate
    ynew = ogate * cnew / nnew

    # shapes ([B,H], [B,H], [B,H]), ([B,H],[B,H],[B,H],[B,H])
    return torch.stack((ynew, cnew, nnew, mnew), dim=0), torch.stack((igate, fgate, zraw, ogate), dim=0)
