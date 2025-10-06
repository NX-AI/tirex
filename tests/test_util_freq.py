# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import math

import numpy as np
import torch

from tirex.util import frequency_resample, run_fft_analysis


def _sine_series(length: int, period: int, amplitude: float = 1.0) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32)
    return amplitude * torch.sin(2 * math.pi * t / period)


def test_frequency_resample_downsamples_long_forecasts():
    series = _sine_series(length=4096, period=128)

    resampled, factor, upsample_fn = frequency_resample(
        series,
        prediction_length=200,
        patch_size=64,
        max_period=256,
    )

    assert math.isclose(factor, 0.5, rel_tol=1e-5)
    assert resampled.shape[0] == math.ceil(series.shape[0] * factor)

    restored = upsample_fn(resampled)
    assert restored.shape[0] == series.shape[0]


def test_frequency_resample_keeps_short_forecasts():
    series = _sine_series(length=4096, period=128)

    resampled, factor, upsample_fn = frequency_resample(
        series,
        prediction_length=50,
        patch_size=64,
        max_period=256,
    )

    assert factor == 1.0
    assert resampled.shape[0] == series.shape[0]
    restored = upsample_fn(resampled)
    assert restored.shape[0] == series.shape[0]


def test_frequency_resample_nifr_rounds_to_integer_fraction():
    period = 190
    series = _sine_series(length=4096, period=period)

    resampled, factor, _ = frequency_resample(
        series,
        prediction_length=200,
        patch_size=64,
        max_period=256,
        nifr_enabled=True,
        nifr_start_integer=2,
        nifr_end_integer=12,
        nifr_clamp_large_factors=False,
    )

    assert math.isclose(factor, 1 / 3, rel_tol=1e-6)
    assert resampled.shape[0] == math.ceil(series.shape[0] * factor)


def test_frequency_resample_nifr_clamps_large_factors_to_identity():
    period = 90
    series = _sine_series(length=4096, period=period)

    resampled, factor, _ = frequency_resample(
        series,
        prediction_length=200,
        patch_size=64,
        max_period=256,
        nifr_enabled=True,
        nifr_start_integer=2,
        nifr_end_integer=12,
        nifr_clamp_large_factors=True,
    )

    assert factor == 1.0
    assert resampled.shape[0] == series.shape[0]


def test_run_fft_analysis_short_series_returns_empty():
    freqs, spec, peaks = run_fft_analysis(np.array([1.0]))
    assert freqs.size == 0
    assert spec.size == 0
    assert peaks.size == 0


def test_run_fft_analysis_detects_primary_frequency_with_nans():
    length = 2048
    period = 64
    series = _sine_series(length=length, period=period).numpy()
    series[10:20] = np.nan
    series[100] = np.nan

    freqs, spec, peaks = run_fft_analysis(
        series,
        min_period=16,
        max_period=256,
        peak_prominence=0.05,
    )

    assert peaks.size > 0
    dominant_freq = freqs[peaks[0]]
    assert math.isclose(dominant_freq, 1 / period, rel_tol=1e-2)
    assert math.isclose(float(spec.max()), 1.0, rel_tol=1e-5)
