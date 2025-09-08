from pathlib import Path

import numpy as np
import pytest
import torch

from tirex import ForecastModel, load_model


def load_tensor_from_file(path):
    base_path = Path(__file__).parent.resolve() / "data"
    return torch.from_numpy(np.genfromtxt(base_path / path, dtype=np.float32))


@pytest.fixture
def tirex_model() -> ForecastModel:
    return load_model("NX-AI/TiRex")


def test_forecast_air_traffic(tirex_model):
    context = load_tensor_from_file("air_passengers.csv")[:-12]

    quantiles, mean = tirex_model.forecast(context, prediction_length=24)

    ref_data = load_tensor_from_file("air_passengers_forecast_ref.csv").unsqueeze(0)

    torch.testing.assert_close(mean, ref_data, rtol=1.6e-2, atol=1e-5)  # default rtol & atol for bfloat16


def test_forecast_seattle_5T(tirex_model):
    context = load_tensor_from_file("loop_seattle_5T.csv")[:-512]

    quantiles, mean = tirex_model.forecast(context, prediction_length=768)

    ref_data = load_tensor_from_file("loop_seattle_5T_forecast_ref.csv").unsqueeze(0)

    torch.testing.assert_close(mean, ref_data, rtol=1.6e-2, atol=1e-5)  # default rtol & atol for bfloat16
