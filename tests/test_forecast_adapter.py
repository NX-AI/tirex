# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import inspect

import numpy as np
import pytest
import torch

from tirex.api_adapter.forecast import ForecastModel, _format_output, _gen_forecast, get_batches


def fc_random_from_tensor(batch, prediction_length, **kwargs):
    assert isinstance(batch, torch.Tensor)
    B, L = batch.shape
    return_val = torch.rand((B, prediction_length, 9))
    return return_val, return_val[:, :, 4]


@pytest.fixture
def dummy_fc_func():
    return fc_random_from_tensor


class DummyForecaster(ForecastModel):
    def _forecast_quantiles(self, batch: torch.Tensor, **kwargs):
        return fc_random_from_tensor(batch, **kwargs)


# ----- Tests: Output formatting -----
def test_format_output_shapes(dummy_fc_func):
    B, L = 2, 5
    PL = 10
    q, m = dummy_fc_func(torch.rand(B, L), prediction_length=PL)
    out_q, out_m = _format_output(q, m, [{}] * B, "torch")
    assert isinstance(out_q, torch.Tensor)
    assert isinstance(out_m, torch.Tensor)
    assert out_q.shape == (B, PL, 9)
    assert out_m.shape == (B, PL)


def test_format_output_shapes(dummy_fc_func):
    B, L = 2, 5
    PL = 10
    q, m = dummy_fc_func(torch.rand(B, L), prediction_length=PL)
    out_q, out_m = _format_output(q, m, [{}] * B, "numpy")
    assert isinstance(out_q, np.ndarray)
    assert isinstance(out_m, np.ndarray)
    assert out_q.shape == (B, PL, 9)
    assert out_m.shape == (B, PL)


# ----- Tests: Forecast generation -----
def test_gen_forecast_single_batch(dummy_fc_func):
    context = torch.rand((5, 20))
    batches = get_batches(context, batch_size=2)
    q, m = _gen_forecast(dummy_fc_func, batches, "torch", prediction_length=10, yield_per_batch=False)
    assert q.shape == (5, 10, 9)
    assert m.shape == (5, 10)


def test_gen_forecast_iterator(dummy_fc_func):
    context = torch.rand((5, 20))
    batches = get_batches(context, batch_size=2)
    iterator = _gen_forecast(dummy_fc_func, batches, "torch", prediction_length=10, yield_per_batch=True)
    outputs = list(iterator)
    assert len(outputs) == 3
    for i, (q, m) in enumerate(outputs):
        if i < 2:
            assert q.shape == (2, 10, 9)
            assert m.shape == (2, 10)
        else:
            assert q.shape == (1, 10, 9)
            assert m.shape == (1, 10)


# ----- Tests: ForecastModel integration -----
def test_forecast_with_variable_lengths():
    model = DummyForecaster()
    context = [torch.arange(4), torch.arange(7), torch.arange(5), torch.arange(3)]
    out_q, out_m = model.forecast(context, output_type="torch", prediction_length=10, batch_size=3)
    assert out_q.shape == (4, 10, 9)
    assert out_m.shape == (4, 10)


def test_forecast_iterator_mode():
    model = DummyForecaster()
    context = torch.rand((5, 20))
    iterator = model.forecast(context, yield_per_batch=True, output_type="torch", prediction_length=10, batch_size=2)
    results = list(iterator)
    assert len(results) == 3
    for i, (q, m) in enumerate(results):
        if i < 2:
            assert q.shape == (2, 10, 9)
            assert m.shape == (2, 10)
        else:
            assert q.shape == (1, 10, 9)
            assert m.shape == (1, 10)


# ----- Tests: forecast options -----
class OptionRecordingForecaster(ForecastModel):
    def __init__(self):
        self.seen_kwargs = []

    def _forecast_quantiles(self, batch: torch.Tensor, **kwargs):
        self.seen_kwargs.append(kwargs)
        return fc_random_from_tensor(batch, **kwargs)


def test_forecast_option_defaults_are_forwarded():
    model = OptionRecordingForecaster()
    model.forecast(torch.rand((2, 20)), prediction_length=10)
    assert model.seen_kwargs == [{"prediction_length": 10, "full_rollout": False, "dynamic_padding": False}]


def test_forecast_options_are_forwarded():
    model = OptionRecordingForecaster()
    model.forecast(torch.rand((2, 20)), prediction_length=10, full_rollout=True, dynamic_padding=True)
    assert model.seen_kwargs == [{"prediction_length": 10, "full_rollout": True, "dynamic_padding": True}]


@pytest.mark.parametrize("fc_method", ["forecast", "forecast_gluon", "forecast_hfdata"])
@pytest.mark.parametrize("option", ["full_rollout", "dynamic_padding"])
def test_forecast_options_are_public_parameters(fc_method, option):
    parameter = inspect.signature(getattr(ForecastModel, fc_method)).parameters.get(option)
    assert parameter is not None, f"{fc_method} does not expose {option}"
    assert parameter.default is False
