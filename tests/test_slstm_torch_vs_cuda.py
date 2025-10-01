# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import copy

import pytest
import torch

from tirex.models.slstm.cell import sLSTMBlockConfig, sLSTMCell

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="This test needs CUDA.")


@pytest.mark.parametrize("with_in_state", [True, False])
def test_with_in_state(with_in_state):
    run_slstm_torch_vs_cuda(with_in_state=with_in_state)


@pytest.mark.parametrize("sequence_length", [1, 2, 4])
def test_sequence_length(sequence_length):
    run_slstm_torch_vs_cuda(sequence_length=sequence_length)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batch_size(batch_size):
    run_slstm_torch_vs_cuda(batch_size=batch_size)


@pytest.mark.parametrize("num_heads", [4, 1])
def test_num_heads(num_heads):
    run_slstm_torch_vs_cuda(num_heads=num_heads, with_in_state=True, atol=5e-5)


@pytest.mark.parametrize("hidden_size", [64, 8])
def test_hidden_size(hidden_size):
    run_slstm_torch_vs_cuda(hidden_size=hidden_size, with_in_state=True)


def test_complex():
    run_slstm_torch_vs_cuda(
        hidden_size=128, batch_size=2, sequence_length=8, num_heads=4, with_in_state=True, atol=1e-5
    )


def test_long_sequence():
    run_slstm_torch_vs_cuda(sequence_length=128, atol=1e-5)


def set_seed(seed):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_slstm_torch_vs_cuda(
    batch_size=1, sequence_length=1, with_in_state=False, num_heads=4, hidden_size=64, rtol=1.3e-6, atol=1e-6
):
    device_cuda = "cuda"
    config = sLSTMBlockConfig(embedding_dim=hidden_size, num_heads=num_heads)

    set_seed(42)
    recurrent_kernel_weight = torch.randn(
        (config.num_heads, config.head_dim, config.num_gates * config.head_dim), dtype=torch.bfloat16
    )
    bias_weight = torch.randn((config.num_heads * config.num_gates * config.head_dim), dtype=torch.bfloat16)

    cell_torch = sLSTMCell(copy.deepcopy(config), backend="torch")
    cell_torch._recurrent_kernel_.data = recurrent_kernel_weight
    cell_torch._bias_.data = bias_weight

    cell_cuda = sLSTMCell(copy.deepcopy(config), backend="cuda").to(device_cuda)
    cell_cuda._recurrent_kernel_.data = recurrent_kernel_weight.to(device_cuda)
    cell_cuda._bias_.data = bias_weight.to(device_cuda)

    set_seed(42)
    current_input = torch.randn((batch_size, sequence_length, 4 * config.embedding_dim))
    state = torch.randn((4, batch_size, hidden_size)) if with_in_state else None

    output_torch, state_torch = cell_torch.forward(current_input, state)
    output_cuda, state_cuda = cell_cuda.forward(
        current_input.to(device_cuda), state.to(device_cuda) if state is not None else state
    )

    torch.testing.assert_close(output_torch, output_cuda.cpu(), rtol=rtol, atol=atol)
    torch.testing.assert_close(state_torch, state_cuda.cpu(), rtol=rtol, atol=atol)
