# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from pathlib import Path

import numpy as np
import pytest
import torch

from tirex import ForecastModel, load_model


def load_tensor_from_txt_file(path):
    base_path = Path(__file__).parent.resolve() / "data"
    return torch.from_numpy(np.genfromtxt(base_path / path, dtype=np.float32))


def load_tensor_from_pt_file(path):
    base_path = Path(__file__).parent.resolve() / "data"
    return torch.load(base_path / path)


@pytest.fixture
def tirex_model() -> ForecastModel:
    return load_model("NX-AI/TiRex")


@pytest.mark.parametrize("resample_strategy", ([None, "frequency"]))
def test_forecast_air_traffic(tirex_model, resample_strategy):
    context = load_tensor_from_txt_file("air_passengers.csv")[:-12]

    quantiles, mean = tirex_model.forecast(context, prediction_length=24, resample_strategy=resample_strategy)

    ref_mean = load_tensor_from_txt_file("air_passengers_forecast_ref.csv").unsqueeze(0)
    ref_quantiles = load_tensor_from_pt_file("air_passengers_quantiles_ref.pt")

    # default rtol & atol for bfloat16
    torch.testing.assert_close(mean, ref_mean, rtol=1.6e-2, atol=1e-5)
    torch.testing.assert_close(quantiles, ref_quantiles, rtol=1.6e-2, atol=1e-5)


@pytest.mark.parametrize(
    "resample_strategy, ref_mean_path, ref_quantiles_path",
    (
        [None, "loop_seattle_5T_forecast_ref.csv", "loop_seattle_5T_quantiles_ref.pt"],
        ["frequency", "loop_seattle_5T_forecast_resampled_ref.csv", "loop_seattle_5T_quantiles_resampled_ref.pt"],
    ),
)
def test_forecast_seattle_5T(tirex_model, resample_strategy, ref_mean_path, ref_quantiles_path):
    context = load_tensor_from_txt_file("loop_seattle_5T.csv")[:-512]

    quantiles, mean = tirex_model.forecast(context, prediction_length=768, resample_strategy=resample_strategy)

    ref_mean = load_tensor_from_txt_file(ref_mean_path).unsqueeze(0)
    ref_quantiles = load_tensor_from_pt_file(ref_quantiles_path)

    # default rtol & atol for bfloat16
    torch.testing.assert_close(mean, ref_mean, rtol=1.6e-2, atol=1e-5)
    torch.testing.assert_close(quantiles, ref_quantiles, rtol=1.6e-2, atol=1e-5)


def test_full_rollout_is_a_no_op_within_one_patch(tirex_model):
    context = load_tensor_from_txt_file("air_passengers.csv")[:-12]
    patch_size = tirex_model.tokenizer.patch_size

    quantiles, mean = tirex_model.forecast(context, prediction_length=patch_size)
    quantiles_full, mean_full = tirex_model.forecast(context, prediction_length=patch_size, full_rollout=True)

    # a single patch already covers the horizon, so both settings run the same single forward pass
    assert torch.equal(quantiles_full, quantiles)
    assert torch.equal(mean_full, mean)


def test_full_rollout_over_multiple_patches(tirex_model):
    context = load_tensor_from_txt_file("air_passengers.csv")[:-12]
    prediction_length = 3 * tirex_model.tokenizer.patch_size

    quantiles, mean = tirex_model.forecast(context, prediction_length=prediction_length)
    quantiles_full, mean_full = tirex_model.forecast(context, prediction_length=prediction_length, full_rollout=True)

    assert quantiles_full.shape == quantiles.shape
    assert mean_full.shape == mean.shape
    assert torch.isfinite(quantiles_full).all()
    # the horizon is predicted in one pass instead of three, which changes the forecast
    assert not torch.equal(quantiles_full, quantiles)


def test_dynamic_padding_shortens_the_padded_context(tirex_model):
    context = load_tensor_from_txt_file("air_passengers.csv")[:-12]

    quantiles, mean = tirex_model.forecast(context, prediction_length=24)
    quantiles_dyn, mean_dyn = tirex_model.forecast(context, prediction_length=24, dynamic_padding=True)

    assert quantiles_dyn.shape == quantiles.shape
    assert mean_dyn.shape == mean.shape
    assert torch.isfinite(quantiles_dyn).all()
    # the context is padded to the next multiple of the patch size instead of the training context
    # length, so the model sees fewer masked tokens and the forecast changes
    assert not torch.equal(quantiles_dyn, quantiles)


def test_dynamic_padding_is_a_no_op_for_a_full_context(tirex_model):
    context = load_tensor_from_txt_file("loop_seattle_5T.csv")[: tirex_model.config.train_ctx_len]

    quantiles, mean = tirex_model.forecast(context, prediction_length=24)
    quantiles_dyn, mean_dyn = tirex_model.forecast(context, prediction_length=24, dynamic_padding=True)

    # the context already fills the training context length, so there is nothing to pad
    assert torch.equal(quantiles_dyn, quantiles)
    assert torch.equal(mean_dyn, mean)
