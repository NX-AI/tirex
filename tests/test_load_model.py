# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import warnings

import pytest
import torch

from tirex import load_model


def test_load_model_with_default_parameters():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = load_model("NX-AI/TiRex")

        assert model is not None
        assert model.blocks[0].slstm_layer.slstm_cell.backend == "torch"

        context = torch.randn(1, 64)
        _, _ = model.forecast(context, prediction_length=32)
        assert len(w) == 0  # no warnings check


def test_load_model_with_cpu_device_no_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = load_model("NX-AI/TiRex", device="cpu")

        assert model is not None
        assert model.blocks[0].slstm_layer.slstm_cell.backend == "torch"
        context = torch.randn(1, 64)
        _, _ = model.forecast(context, prediction_length=32)
        assert len(w) == 0  # no warnings check


def test_load_model_cuda_device_without_cuda_raises_error():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available, skipping no-CUDA test")

    with pytest.raises(ValueError) as exc_info:
        model = load_model("NX-AI/TiRex", device="cuda:0")

    error_msg = str(exc_info.value)
    assert "CUDA is not available" in error_msg
    assert "No GPU is present" in error_msg
    assert "PyTorch is installed without CUDA support (CPU-only version)" in error_msg
    assert "To resolve: use device='cpu'" in error_msg


def test_load_model_with_mps_device():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available, skipping MPS test")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = load_model("NX-AI/TiRex", device="mps")

        assert model is not None
        assert model.blocks[0].slstm_layer.slstm_cell.backend == "torch"

        device = next(model.parameters()).device
        assert device.type == "mps"

        context = torch.randn(1, 64).to("mps")
        _, _ = model.forecast(context, prediction_length=32)
        assert len(w) == 0  # no warnings check
