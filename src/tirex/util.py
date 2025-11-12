# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from collections.abc import Callable, Sequence
from dataclasses import fields
from functools import partial
from math import ceil, isfinite
from typing import Literal, Optional

import numpy as np
import torch

COLOR_CONTEXT = "#4a90d0"
COLOR_FORECAST = "#d94e4e"
COLOR_GROUND_TRUTH = "#4a90d0"
COLOR_CUTOFF_LINE = "#000000"
COLOR_QUANTILES = "#d94e4e"
ALPHA_QUANTILES = 0.1


def frequency_resample(
    ts: torch.Tensor,
    prediction_length: int,
    patch_size: int = 64,
    peak_prominence: float = 0.1,
    selection_method: Literal["low_harmonic", "high_harmonic", "highest_amplitude"] = "low_harmonic",
    min_period: int | None = None,
    max_period: int = 1000,
    bandpass_filter: bool = True,
    nifr_enabled: bool = True,
    nifr_start_integer: int = 2,
    nifr_end_integer: int = 12,
    nifr_clamp_large_factors: bool = False,
) -> tuple[torch.Tensor, float, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Downsample a time series according to a frequency-based strategy and return a helper to upsample
    forecasts back to the original resolution.

    Parameters
    ----------
    ts : torch.Tensor
        1D time series of shape [T].
    prediction_length : int
        Requested forecast horizon; short horizons (<100) skip resampling.
    patch_size : int, default 64
        Nominal patch size used to align one dominant period to one patch.
    peak_prominence : float, default 0.1
        Threshold for FFT peak detection on the normalized spectrum.
    selection_method : {"low_harmonic", "high_harmonic", "highest_amplitude"}, default "low_harmonic"
        How to resolve two dominant peaks in ~2x harmonic relation.
    min_period : int or None, optional
        Minimum period to consider; if None, defaults to `patch_size`.
    max_period : int, default 1000
        Maximum period to consider for FFT peak search.
    bandpass_filter : bool, default True
        If True, suppresses very low frequencies before peak search.
    nifr_enabled : bool, default True
        Enable nearest-integer-fraction rounding of the factor.
    nifr_start_integer : int, default 2
        Smallest integer k used for 1/k grid when NIFR is enabled.
    nifr_end_integer : int, default 12
        Largest integer k used for 1/k grid when NIFR is enabled.
    nifr_clamp_large_factors : bool, default False
        If True, clamps large factors in [1, 1/nifr_start_integer] to 1.0.

    Returns
    -------
    resampled_ts : torch.Tensor
        The resampled input series.
    sample_factor : float
        Applied sampling factor (<= 1 means downsampling; 1.0 = identity).
    fc_resample_fn : Callable[[torch.Tensor], torch.Tensor]
        Function that upsamples a forecast back to the original resolution using the inverse factor.

    Notes
    -----
    - For short horizons (prediction_length < 100), resampling is disabled and the factor is set to 1.0.
    - The factor is clamped to at most 1.0 to avoid upsampling the context.
    """

    sample_factor = frequency_factor(
        ts,
        max_period=max_period,
        min_period=min_period,
        bandpass_filter=bandpass_filter,
        selection_method=selection_method,
        peak_prominence=peak_prominence,
        patch_size=patch_size,
        nifr_enabled=nifr_enabled,
        nifr_start_integer=nifr_start_integer,
        nifr_end_integer=nifr_end_integer,
        nifr_clamp_large_factors=nifr_clamp_large_factors,
    )

    sample_factor = min(1, sample_factor)

    if prediction_length < 100:
        # do not resample for short forecasts
        sample_factor = 1.0

    fc_resample_factor = 1 / sample_factor
    fc_resample_fn = partial(resample, sample_rate=fc_resample_factor)
    resampled_ts = resample(ts, sample_rate=sample_factor)

    return resampled_ts, sample_factor, fc_resample_fn


def frequency_factor(
    ts: torch.Tensor,
    patch_size: int = 64,  # This doesn't have to match model patch size, but rather the 'target frequency'
    peak_prominence: float = 0.1,
    selection_method: Literal["low_harmonic", "high_harmonic", "highest_amplitude"] = "low_harmonic",
    min_period: int | None = None,
    max_period: int = 1000,
    bandpass_filter: bool = True,
    nifr_enabled: bool = False,
    nifr_start_integer: int = 2,
    nifr_end_integer: int = 12,
    nifr_clamp_large_factors: bool = False,
) -> float:
    """
    Estimate a sampling factor from the dominant frequency of a 1D series so that one period
    approximately fits into one patch of length `patch_size`.

    The factor is computed as `patch_size / period`, where `period = 1 / f*` and `f*`
    is the selected dominant frequency from the one-sided FFT of the series (NaNs are
    linearly interpolated for analysis). If two prominent peaks are detected whose
    frequencies are in roughly a 2x harmonic relation (ratio in [1.5, 2.5]),
    `selection_method` determines whether to select the lower or higher harmonic. A set of
    guards returns 1.0 (identity) for short series, invalid/non-finite results, or when no
    prominent peak is found. Optional nearest-integer-fraction rounding (NIFR) can snap the
    factor to the closest value in {1} ∪ {1/k | k ∈ [nifr_start_integer, nifr_end_integer]}.

    Parameters
    ----------
    ts : torch.Tensor
        Input 1D series (last dim is time). NaNs are linearly interpolated for FFT analysis only;
        the original series is not modified.
    patch_size : int, default 64
        Target number of samples per period.
    peak_prominence : float, default 0.1
        Minimum normalized spectrum height to treat a bin as a peak.
    selection_method : {"low_harmonic", "high_harmonic", "highest_amplitude"}, default "low_harmonic"
        Rule for picking between two ~2x related peaks.
    min_period : int or None, optional
        Minimum period to consider. If None, defaults to `patch_size`.
    max_period : int, default 1000
        Series shorter than `2 * max_period` return 1.0.
    bandpass_filter : bool, default True
        If True, very low frequencies below 1 / max_period are suppressed.
    nifr_enabled : bool, default False
        Enable nearest-integer-fraction rounding of the factor.
    nifr_start_integer : int, default 2
        Smallest integer k used for the 1/k grid when NIFR is enabled.
    nifr_end_integer : int, default 12
        Largest integer k used for the 1/k grid when NIFR is enabled.
    nifr_clamp_large_factors : bool, default False
        If True, clamps factors in [1, 1/nifr_start_integer] to 1.0.

    Returns
    -------
    float
        The sampling factor. Values <= 0 or non-finite are mapped to 1.0. If no valid
        dominant frequency is found or the series is too short, returns 1.0.

    Notes
    -----
    - The factor is computed as `patch_size / period`, where `period = 1 / f*` and `f*` is the selected
      dominant FFT frequency.
    - If two prominent peaks are detected ~2x apart, `selection_method` determines whether to select the lower or higher harmonic.
    - Optional nearest-integer-fraction rounding (NIFR) can snap the factor to the closest value in {1} ∪ {1/k | k ∈ [nifr_start_integer, nifr_end_integer]}.
    """
    if min_period is None:
        # NOTE: be careful when min_period is not matching patch_size, it can create unexpected scaling factors!
        min_period = patch_size

    if isinstance(ts, torch.Tensor):
        ts_tensor = ts.to(torch.float32)
    else:
        ts_tensor = torch.as_tensor(ts, dtype=torch.float32)

    # NOTE: If the series is shorter than max_period *2, FFT may not be accurate, to avoid detecting these peaks, we don't scale
    if ts_tensor.numel() < max_period * 2:
        return 1.0

    freqs, specs, peak_idc = run_fft_analysis(
        ts_tensor,
        scaling="amplitude",
        peak_prominence=peak_prominence,
        min_period=min_period,
        max_period=max_period,
        bandpass_filter=bandpass_filter,
    )

    # No detectable peaks -> keep original sampling
    if peak_idc.numel() == 0:
        return 1.0

    # Choose initial candidate as the highest-amplitude peak
    chosen_idx = int(peak_idc[0].item())

    # If two peaks exist, check for ~2x harmonic relation and prefer the higher/lower one
    if peak_idc.numel() >= 2:
        idx_a = int(peak_idc[0].item())  # highest amplitude
        idx_b = int(peak_idc[1].item())  # second highest amplitude
        f_a = float(freqs[idx_a].item())
        f_b = float(freqs[idx_b].item())

        # Determine lower/higher frequency
        low_f = min(f_a, f_b)
        high_f = max(f_a, f_b)

        if low_f > 0:
            ratio = high_f / low_f
            # Roughly half relation
            if 1.5 <= ratio <= 2.5:
                if selection_method == "low_harmonic":
                    chosen_idx = idx_a if f_a < f_b else idx_b
                elif selection_method == "high_harmonic":
                    chosen_idx = idx_a if f_a > f_b else idx_b

    chosen_freq = float(freqs[chosen_idx].item())

    # Guard against zero or non-finite frequency
    if not isfinite(chosen_freq) or chosen_freq <= 0:
        return 1.0

    # Convert to period and compute scaling factor so one period fits one patch
    period = 1.0 / chosen_freq
    factor = resampling_factor(period, patch_size)
    factor = round(factor, 4)

    # Guard against factor being negative
    if not isfinite(factor) or factor <= 0:
        return 1.0

    # nearest interger fraction rounding (nifr)
    if nifr_enabled:
        device = ts_tensor.device
        dtype = torch.float32
        base = torch.ones(1, device=device, dtype=dtype)
        if nifr_end_integer >= nifr_start_integer:
            denominators = torch.arange(nifr_start_integer, nifr_end_integer + 1, device=device, dtype=dtype)
            candidate_factors = torch.cat([base, 1.0 / denominators])
        else:
            candidate_factors = base

        factor_tensor = torch.tensor(factor, device=device, dtype=dtype)
        diff = torch.abs(factor_tensor - candidate_factors)
        min_idx = int(torch.argmin(diff).item())
        factor_tensor = candidate_factors[min_idx]

        if nifr_clamp_large_factors:
            # Clamp everything between 1 and 1/nifr_start_integer to 1, that is no scaling
            if candidate_factors.numel() > 1:
                clamp_threshold = candidate_factors[1]
                one = torch.tensor(1.0, device=device, dtype=dtype)
                factor_tensor = torch.where(factor_tensor < clamp_threshold, factor_tensor, one)

        factor = float(factor_tensor.item())

    return float(factor)


def resample(ts: torch.Tensor, sample_rate: float, window_position: str = "center") -> torch.Tensor:
    """
    Resample the time series using NaN-tolerant window averaging with size 1/sample_rate.

    - If sample_rate > 1 the series is upsampled; windows may collapse to a single index.
    - If sample_rate < 1 the series is downsampled; windows span multiple indices.
    - If sample_rate == 1 the series is returned unchanged (cast to float for NaN support).

    Window alignment controlled by `window_position`:
    - "center": average over [c - L/2, c + L/2]
    - "left"  : average over [c - L,   c]
    - "right" : average over [c,       c + L]

    The window is truncated at the boundaries. NaNs are ignored via nan-mean semantics;
    if a window contains only NaNs, the output is NaN.

    Arguments:
    ----------
    ts: torch.Tensor of shape [..., T]
        The time series to be rescaled (last dim is time).
    sample_rate: float
        The factor determining the final number of timesteps in the series, i.e., T' = ceil(T * sample_rate).
    window_position: {"center", "left", "right"}
        Placement of the averaging window relative to each target coordinate.

    Returns:
    --------
    torch.Tensor of shape [..., ceil(T * sample_rate)] with dtype float.
    """
    # Validate inputs
    if sample_rate <= 0 or sample_rate == 1:
        # Invalid or no scaling; return original as float
        return ts.to(torch.float)

    src_num_timesteps = ts.shape[-1]
    tgt_num_timesteps = ceil(src_num_timesteps * sample_rate)

    # Do not change coordinate creation logic
    src_coords = torch.arange(src_num_timesteps, device=ts.device)
    tgt_coords = torch.linspace(0, src_num_timesteps - 1, tgt_num_timesteps, device=ts.device)

    if sample_rate == 1:
        return ts.to(torch.float)

    # Branch: upsampling -> linear interpolation between nearest neighbors (NaN-aware)
    if sample_rate > 1:
        # Neighbour indices for each target coordinate along the last dimension
        tgt_in_src_idx_lo = tgt_coords.floor().to(torch.long)
        tgt_in_src_idx_hi = tgt_coords.ceil().to(torch.long)

        # Distances in index space and offsets from lower index
        dist = src_coords[tgt_in_src_idx_hi] - src_coords[tgt_in_src_idx_lo]

        # Work in float for NaN support; gather neighbour values
        src_lo_vals = ts[..., tgt_in_src_idx_lo].to(torch.float)
        src_hi_vals = ts[..., tgt_in_src_idx_hi].to(torch.float)
        diff = src_hi_vals - src_lo_vals
        offset = tgt_coords - src_coords[tgt_in_src_idx_lo]

        # Allocate output
        tgt_values = torch.empty(*ts.shape[:-1], tgt_num_timesteps, dtype=torch.float, device=ts.device)

        # Masks
        exact_mask = dist == 0
        interp_mask = ~exact_mask

        # Exact source index -> take the source value
        if exact_mask.any():
            tgt_values[..., exact_mask] = src_lo_vals[..., exact_mask]

        # Linear interpolate where indices differ
        if interp_mask.any():
            tgt_values[..., interp_mask] = (
                diff[..., interp_mask] / dist[interp_mask].to(torch.float) * offset[interp_mask]
                + src_lo_vals[..., interp_mask]
            )

        # Propagate NaNs from either neighbour
        nan_mask = torch.isnan(src_lo_vals) | torch.isnan(src_hi_vals)
        if nan_mask.any():
            tgt_values[..., nan_mask] = torch.nan

        return tgt_values

    # Window length in source-index units
    L = 1.0 / sample_rate
    half_L = 0.5 * L

    if window_position == "center":
        left_f = tgt_coords - half_L
        right_f = tgt_coords + half_L
    elif window_position == "left":
        left_f = tgt_coords - L
        right_f = tgt_coords
    elif window_position == "right":
        left_f = tgt_coords
        right_f = tgt_coords + L
    else:
        raise ValueError("window_position must be one of {'center','left','right'}")

    # Convert to integer indices, inclusive bounds
    left_idx = torch.ceil(left_f).to(torch.long)
    right_idx = torch.floor(right_f).to(torch.long)

    # Clip to valid range and ensure non-empty windows (at least one index)
    left_idx = torch.clamp(left_idx, 0, src_num_timesteps - 1)
    right_idx = torch.clamp(right_idx, 0, src_num_timesteps - 1)
    right_idx = torch.maximum(right_idx, left_idx)

    # Prepare cumulative sums for fast [l, r] segment nan-mean along the last dim
    ts_float = ts.to(torch.float)
    valid_mask = ~torch.isnan(ts_float)

    values_filled = torch.where(valid_mask, ts_float, torch.zeros_like(ts_float))
    counts = valid_mask.to(torch.float)

    cumsum_vals = values_filled.cumsum(dim=-1)
    cumsum_cnts = counts.cumsum(dim=-1)

    # Pad a leading zero to make inclusive range sums easy: sum[l:r] = cs[r] - cs[l-1]
    pad_shape = (*ts.shape[:-1], 1)
    zeros_vals = torch.zeros(pad_shape, dtype=cumsum_vals.dtype, device=ts.device)
    zeros_cnts = torch.zeros(pad_shape, dtype=cumsum_cnts.dtype, device=ts.device)
    cumsum_vals = torch.cat([zeros_vals, cumsum_vals], dim=-1)
    cumsum_cnts = torch.cat([zeros_cnts, cumsum_cnts], dim=-1)

    # Build broadcastable indices for gather along the last dim
    prefix_shape = ts.shape[:-1]
    target_len = tgt_num_timesteps

    def _expand_index(idx: torch.Tensor) -> torch.Tensor:
        # idx shape [target_len] -> [..., target_len]
        view_shape = (1,) * len(prefix_shape) + (target_len,)
        return idx.view(view_shape).expand(*prefix_shape, target_len)

    # For inclusive [l, r], use cumsum at (r+1) and (l)
    r_plus1 = torch.clamp(right_idx + 1, 0, src_num_timesteps)
    l_idx = left_idx

    r_plus1_exp = _expand_index(r_plus1)
    l_exp = _expand_index(l_idx)

    seg_sums = cumsum_vals.gather(dim=-1, index=r_plus1_exp) - cumsum_vals.gather(dim=-1, index=l_exp)
    seg_cnts = cumsum_cnts.gather(dim=-1, index=r_plus1_exp) - cumsum_cnts.gather(dim=-1, index=l_exp)

    # Compute nan-mean: where count==0 -> NaN
    with torch.no_grad():
        safe_cnts = torch.where(seg_cnts > 0, seg_cnts, torch.ones_like(seg_cnts))
    averages = seg_sums / safe_cnts
    averages = torch.where(seg_cnts > 0, averages, torch.full_like(averages, float("nan")))

    return averages


def run_fft_analysis(
    y,
    dt: float = 1.0,
    window: str = "hann",
    detrend: bool = True,
    scaling: str = "amplitude",
    peak_prominence: float = 0.1,
    min_period: int = 64,
    max_period: int = 1000,
    bandpass_filter: bool = True,
):
    """
    Compute one-sided FFT frequencies and spectrum magnitude for a real 1D signal.

    Parameters
    ----------
    y : array_like
        1D time series (regularly sampled). NaNs will be linearly interpolated.
    dt : float
        Sampling period (time between samples). Frequencies are cycles per unit of dt.
    window : {'hann', None}
        Optional taper to reduce leakage.
    detrend : bool
        If True, remove the mean before FFT.
    scaling : {'amplitude', 'power', 'raw'}
        - 'amplitude': one-sided amplitude spectrum with window-power compensation.
        - 'power'    : one-sided power (not density) with window-power compensation.
        - 'raw'      : rfft(yw) (no normalization, mostly for debugging).
    peak_prominence : float
        Absolute threshold on the normalized spectrum for peak detection.

    Returns
    -------
    f : ndarray
        Frequencies (non-negative), length N//2 + 1, in cycles per unit (1/dt).
    spec : ndarray
        Spectrum corresponding to `f` under the chosen `scaling`.
    peaks_idx : ndarray
        Indices into f of detected peaks.
    """
    if isinstance(y, torch.Tensor):
        y_tensor = y.to(torch.float32)
    else:
        y_tensor = torch.as_tensor(y, dtype=torch.float32)

    if y_tensor.ndim != 1:
        y_tensor = y_tensor.reshape(-1)

    n = y_tensor.numel()
    device = y_tensor.device

    if n < 2:
        empty = torch.empty(0, dtype=y_tensor.dtype, device=device)
        return empty, empty, empty

    # Fill NaNs linearly (handles edge NaNs as well)
    y_tensor = _nan_linear_interpolate(y_tensor)

    if detrend:
        y_tensor = y_tensor - torch.mean(y_tensor)

    # Windowing
    if window == "hann":
        w = torch.hann_window(n, device=device, dtype=y_tensor.dtype)
        yw = y_tensor * w
        # average window power (for proper amplitude/power normalization)
        w_power = torch.sum(w.square()) / n
    elif window is None:
        yw = y_tensor
        w_power = torch.tensor(1.0, device=device, dtype=y_tensor.dtype)
    else:
        raise ValueError("window must be either 'hann' or None")

    # FFT (one-sided)
    Y = torch.fft.rfft(yw)
    f = torch.fft.rfftfreq(n, d=dt, device=device, dtype=y_tensor.dtype)  # cycles per unit time

    if scaling == "raw":
        spec = torch.abs(Y)
    elif scaling == "amplitude":
        # One-sided amplitude with window power compensation
        spec = torch.abs(Y) / (n * torch.sqrt(w_power))
        if spec.numel() > 1:
            if n % 2 == 0 and spec.numel() > 2:
                spec[1:-1] *= 2.0
            else:
                spec[1:] *= 2.0
    elif scaling == "power":
        # One-sided power (not PSD)
        spec = (torch.abs(Y) ** 2) / (n**2 * w_power)
        if spec.numel() > 1:
            if n % 2 == 0 and spec.numel() > 2:
                spec[1:-1] *= 2.0
            else:
                spec[1:] *= 2.0
    else:
        raise ValueError("scaling must be 'amplitude', 'power', or 'raw'")

    # Normalize the spectrum by its maximum value
    if spec.numel() > 0:
        max_val = torch.max(spec)
        if max_val > 0:
            spec = spec / max_val

    # Find peaks in the spectrum
    peaks_idx = custom_find_peaks(
        f,
        spec,
        max_peaks=2,
        prominence_threshold=peak_prominence,
        min_period=min_period,
        max_period=max_period,
        bandpass_filter=bandpass_filter,
    )

    return f, spec, peaks_idx


def _nan_linear_interpolate(y: torch.Tensor) -> torch.Tensor:
    """
    Linearly interpolate NaN values in a 1D torch tensor.
    """
    y = y.to(torch.float32)
    if y.ndim != 1:
        y = y.reshape(-1)
    n = y.numel()
    mask = torch.isfinite(y)
    if mask.all():
        return y
    if (~mask).all():
        return torch.zeros(n, dtype=y.dtype, device=y.device)

    idx = torch.arange(n, device=y.device)
    valid_idx = idx[mask]
    valid_vals = y[mask]

    insert_pos = torch.searchsorted(valid_idx, idx)
    prev_pos = torch.clamp(insert_pos - 1, min=0)
    next_pos = torch.clamp(insert_pos, max=valid_idx.numel() - 1)

    prev_idx = valid_idx[prev_pos]
    next_idx = valid_idx[next_pos]

    prev_vals = valid_vals[prev_pos]
    next_vals = valid_vals[next_pos]

    has_prev = insert_pos > 0
    has_next = insert_pos < valid_idx.numel()

    result = y.clone()
    missing = ~mask
    if missing.any():
        idx_missing = idx[missing]
        prev_idx_missing = prev_idx[missing]
        next_idx_missing = next_idx[missing]
        prev_vals_missing = prev_vals[missing]
        next_vals_missing = next_vals[missing]
        has_prev_missing = has_prev[missing]
        has_next_missing = has_next[missing]

        interp_vals = torch.empty_like(idx_missing, dtype=y.dtype)

        both_mask = has_prev_missing & has_next_missing
        if both_mask.any():
            denom = (next_idx_missing[both_mask] - prev_idx_missing[both_mask]).to(y.dtype)
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            t = (idx_missing[both_mask].to(y.dtype) - prev_idx_missing[both_mask].to(y.dtype)) / denom
            interp_vals[both_mask] = (
                prev_vals_missing[both_mask] + (next_vals_missing[both_mask] - prev_vals_missing[both_mask]) * t
            )

        left_only = has_prev_missing & ~has_next_missing
        if left_only.any():
            interp_vals[left_only] = prev_vals_missing[left_only]

        right_only = ~has_prev_missing & has_next_missing
        if right_only.any():
            interp_vals[right_only] = next_vals_missing[right_only]

        # Handle corner case where neither prev nor next exists (shouldn't happen due to earlier checks)
        neither = ~(both_mask | left_only | right_only)
        if neither.any():
            interp_vals[neither] = 0.0

        result[missing] = interp_vals

    return result


def resampling_factor(inverted_freq, path_size):
    """
    Compute the resampling factor based on the inverted frequency and path size.
    """
    if inverted_freq <= 0:
        return 1.0
    factor = path_size / inverted_freq
    return factor


def custom_find_peaks(
    f: torch.Tensor,
    spec: torch.Tensor,
    *,
    max_peaks: int = 5,
    prominence_threshold: float = 0.1,
    min_period: int = 64,
    max_period: int = 1000,
    bandpass_filter: bool = True,
):
    """
    Finds prominent peaks in a spectrum using a simple custom logic.

    A peak is a local maximum. A peak is considered prominent if its height
    (on a normalized spectrum) is greater than a given threshold.

    Parameters
    ----------
    f : np.ndarray
        Frequency array (currently unused but kept for API consistency).
    spec : np.ndarray
        The normalized spectrum.
    max_peaks : int
        The maximum number of peaks to return.
    prominence_threshold : float
        The minimum height for a peak to be considered prominent.
    min_period : int
        Minimum period to consider for peaks.
    max_period : int
        Maximum period to consider for peaks.
    bandpass_filter : bool
        If True, suppress very low frequencies below 1 / max_period before peak search.

    Returns
    -------
    torch.Tensor
        Long tensor of indices of detected peaks in descending order of prominence.
    """
    if spec.numel() < 5:  # Need at least 5 points to exclude last two bins
        return spec.new_empty(0, dtype=torch.long)

    if bandpass_filter:  # only truly filter low frequencies, high frequencies are dealt with later
        min_freq = 1 / max_period
        freq_mask = (f >= min_freq).to(spec.dtype)
        spec = spec * freq_mask

    # Find all local maxima, excluding the last two bins
    candidates = torch.arange(1, spec.size(0) - 2, device=spec.device, dtype=torch.long)
    if candidates.numel() == 0:
        return spec.new_empty(0, dtype=torch.long)

    center = spec[candidates]
    left = spec[candidates - 1]
    right = spec[candidates + 1]
    local_mask = (center > left) & (center > right)

    if not local_mask.any():
        return spec.new_empty(0, dtype=torch.long)

    local_maxima_indices = candidates[local_mask]

    # Filter by prominence (height)
    heights = spec[local_maxima_indices]
    prominence_mask = heights > prominence_threshold
    if not prominence_mask.any():
        return spec.new_empty(0, dtype=torch.long)

    prominent_indices = local_maxima_indices[prominence_mask]

    prominent_heights = spec[prominent_indices]

    # Check for clear peaks below min_period (do lowpass filter)
    for idx in prominent_indices.tolist():
        freq_val = float(f[idx].item())
        if freq_val <= 0:
            continue
        period = 1.0 / freq_val
        if period < min_period:
            return spec.new_empty(0, dtype=torch.long)

    # Filter by period
    period_filtered_peaks = []
    for idx, prominence in zip(prominent_indices.tolist(), prominent_heights.tolist()):
        freq_val = float(f[idx].item())
        if freq_val <= 0:
            continue
        period = 1.0 / freq_val

        if min_period <= period <= max_period:
            period_filtered_peaks.append((idx, prominence))

    if not period_filtered_peaks:
        return spec.new_empty(0, dtype=torch.long)

    # Sort by height and return the top `max_peaks`
    period_filtered_peaks.sort(key=lambda x: x[1], reverse=True)
    top_indices = [p[0] for p in period_filtered_peaks[:max_peaks]]
    peak_indices = torch.tensor(top_indices, dtype=torch.long, device=spec.device)

    return peak_indices


def round_up_to_next_multiple_of(x: int, multiple_of: int) -> int:
    return int(((x + multiple_of - 1) // multiple_of) * multiple_of)


def dataclass_from_dict(cls, dict: dict):
    class_fields = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in dict.items() if k in class_fields})


def select_quantile_subset(quantiles: torch.Tensor, quantile_levels: list[float]):
    """
    Select specified quantile levels from the quantiles.
    """
    trained_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert set(quantile_levels).issubset(trained_quantiles), (
        f"Only the following quantile_levels are supported: {quantile_levels}"
    )
    quantile_levels_idx = [trained_quantiles.index(q) for q in quantile_levels]
    quantiles_idx = torch.tensor(quantile_levels_idx, dtype=torch.long, device=quantiles.device)
    return torch.index_select(quantiles, dim=-1, index=quantiles_idx).squeeze(-1)


def plot_forecast(
    context: torch.Tensor | np.ndarray,
    forecasts: torch.Tensor | np.ndarray | None = None,
    ground_truth: torch.Tensor | np.ndarray | None = None,
    ax=None,
    x: Sequence | None = None,
    quantiles: tuple[int, int] = (0, 8),
):
    """
    Plots the historical context, optional ground-truth future, and forecast.

    Parameters
    ----------
    context : torch.Tensor or np.ndarray
        The historical time series data to be plotted.
    forecasts : torch.Tensor or np.ndarray, optional
        The forecasts data including quantiles, of shape [Q, N],
        where Q=9 quantiles are required, and N is the number of forecast timesteps.
    ground_truth : torch.Tensor or np.ndarray, optional
        The actual future data to compare the forecast against.
    ax : matplotlib.Axes, optional
        The matplotlib axes object to plot on.
    x : Sequence, optional
        X-axis values (e.g., timestamps or indices) for the data. The sequence must be slicable.
    quantiles : tuple[int], optional
        A tuple indicating the indices of the quantiles to use to plot as shaded areas
        around the median forecast. Set to None to deactivate. Default is (0, 8).

    Returns
    -------
    matplotlib.Axes
        The Axes object with the plotted forecast.
    """
    try:
        from matplotlib import pyplot as plt
    except ImportError as e:
        raise ImportError(
            "'plot_forecast' requires matplotlib to be installed. "
            "Please install TiRex package with plotting support via "
            "\"pip install 'tirex-ts[plotting]'\"."
        ) from e

    if quantiles is not None and len(quantiles) != 2:
        raise ValueError(
            "quantiles must either be a collection of two values for min- and max quantile, respectively, or None."
        )

    if ax is None:
        # default to current axis
        ax = plt.gca()

    # determine all lenghts for clarity
    context_size = len(context)
    forecast_size = len(forecasts) if forecasts is not None else 0
    ground_truth_size = len(ground_truth) if ground_truth is not None else 0
    full_size = context_size + max(forecast_size, ground_truth_size)

    if x is None:
        x = range(full_size)
    elif len(x) < full_size:
        raise ValueError(
            "Not enough 'x' values provided to have one for every timestep in context, forecast, and ground truth window."
        )

    # plot context
    ax.plot(x[:context_size], context, label="Ground Truth Context", color=COLOR_CONTEXT)

    # plot ground truth if supplied
    if ground_truth is not None:
        ax.plot(
            x[context_size : context_size + ground_truth_size],
            ground_truth,
            label="Ground Truth Future",
            color=COLOR_GROUND_TRUTH,
            linestyle="--",
        )

    # plot forecasts if supplied
    # forecasts are a 2D array with quantiles as rows, and data for each timestep as columns
    if forecasts is not None:
        median_forecast = forecasts[:, 4]
        forecast_x = x[context_size : context_size + forecast_size]
        ax.plot(forecast_x, median_forecast, label="Forecast (Median)", color=COLOR_FORECAST)

        if quantiles is not None:
            min_quantile, max_quantile = quantiles
            lower_bound = forecasts[:, min_quantile]
            upper_bound = forecasts[:, max_quantile]

            ax.fill_between(
                forecast_x,
                lower_bound,
                upper_bound,
                color=COLOR_QUANTILES,
                alpha=ALPHA_QUANTILES,
                label=f"Forecast {(min_quantile + 1) * 10}% - {(max_quantile + 1) * 10}% Quantiles",
            )

    if ground_truth is not None or forecasts is not None:
        ax.axvline(x[context_size], color=COLOR_CUTOFF_LINE, linestyle=":")

    ax.set_xlim(left=x[0])
    ax.legend()
    ax.grid()

    return ax
