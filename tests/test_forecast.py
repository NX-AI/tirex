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


def test_forecast_air_traffic(tirex_model):
    context = load_tensor_from_txt_file("air_passengers.csv")[:-12]

    quantiles, mean = tirex_model.forecast(context, prediction_length=24)

    ref_mean = load_tensor_from_txt_file("air_passengers_forecast_ref.csv").unsqueeze(0)
    ref_quantiles = load_tensor_from_pt_file("air_passengers_quantiles_ref.pt")

    # default rtol & atol for bfloat16
    torch.testing.assert_close(mean, ref_mean, rtol=1.6e-2, atol=1e-5)
    torch.testing.assert_close(quantiles, ref_quantiles, rtol=1.6e-2, atol=1e-5)


def test_forecast_seattle_5T(tirex_model):
    context = load_tensor_from_txt_file("loop_seattle_5T.csv")[:-512]

    quantiles, mean = tirex_model.forecast(context, prediction_length=768)

    ref_mean = load_tensor_from_txt_file("loop_seattle_5T_forecast_ref.csv").unsqueeze(0)
    ref_quantiles = load_tensor_from_pt_file("loop_seattle_5T_quantiles_ref.pt")

    # default rtol & atol for bfloat16
    torch.testing.assert_close(mean, ref_mean, rtol=1.6e-2, atol=1e-5)
    torch.testing.assert_close(quantiles, ref_quantiles, rtol=1.6e-2, atol=1e-5)
