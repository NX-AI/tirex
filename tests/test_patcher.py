# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch

from tirex.models.patcher import PatchedTokenizer


def rms(x: torch.Tensor):
    return torch.nanmean(x.square(), dim=-1, keepdim=True).sqrt()


def test_patcher_decode_encode():
    patcher = PatchedTokenizer(patch_size=32)

    input = torch.randn((2, 256))

    patched_context, state = patcher.input_transform(input)
    output = patcher.output_transform(patched_context, state)

    assert patched_context.shape == (2, 8, 32)
    assert input.shape == output.shape
    torch.testing.assert_close(input, output)

    context_rms = rms(patched_context.view(2, -1) - state.loc)
    context_mean = torch.nanmean(patched_context.view(2, -1), dim=-1, keepdim=True)
    torch.testing.assert_close(context_rms, torch.ones((2, 1)), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(context_mean, torch.zeros((2, 1)))


def test_patcher_nan():
    patcher = PatchedTokenizer(patch_size=32)

    input = torch.randn((2, 256))
    input[0, 0:64] = torch.nan

    patched_context, state = patcher.input_transform(input)
    output = patcher.output_transform(patched_context, state)

    torch.testing.assert_close(input, output, equal_nan=True)
