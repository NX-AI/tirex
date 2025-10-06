# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import numpy as np
import pytest
import torch

from tirex.api_adapter.forecast import _pad_time_series_batch
from tirex.api_adapter.standard_adapter import (
    _batch_iterable,
    _batched,
    _batched_slice,
    get_batches,
)


def _assert_batch_structure(batch):
    series, meta = batch
    assert isinstance(series, list)
    assert isinstance(meta, list)
    assert len(series) == len(meta)
    for ts in series:
        assert isinstance(ts, torch.Tensor)
        assert ts.ndim == 1
    for m in meta:
        assert isinstance(m, dict)


@pytest.mark.parametrize(
    "context",
    [
        np.array([1.0, 2.0, 3.0]),
        torch.tensor([1.0, 2.0, 3.0]),
        [np.array([1.0, 2.0]), np.array([1.0, 5.0, 4.0])],
        [torch.tensor([1.0, 2.0]), torch.tensor([1.0, 4.0, 4.0, 4.0])],
    ],
)
def test_get_batches_various_types(context):
    batches = list(get_batches(context, batch_size=2))
    assert len(batches) >= 1
    for batch in batches:
        _assert_batch_structure(batch)


def test_single_sample_tensor():
    context = torch.arange(10)
    batches = list(get_batches(context, batch_size=4))
    assert len(batches) == 1
    series, meta = batches[0]
    assert len(series) == 1
    assert torch.equal(series[0], context)
    assert meta == [{}]


def test_tensor_batch_preserves_lengths():
    context = torch.arange(12).reshape(4, 3)
    batches = list(get_batches(context, batch_size=2))
    assert len(batches) == 2
    first_series, _ = batches[0]
    second_series, _ = batches[1]
    assert all(len(ts) == 3 for ts in first_series + second_series)


def test_list_of_variable_length_arrays():
    context = [torch.arange(5), torch.arange(8), torch.arange(6)]
    batches = list(get_batches(context, batch_size=3))
    assert len(batches) == 1
    series, meta = batches[0]
    lengths = [len(ts) for ts in series]
    assert lengths == [5, 8, 6]
    assert meta == [{}, {}, {}]


def test_list_of_variable_length_arrays_more_than_batchsize():
    context = [torch.arange(i + 1) for i in range(5)]
    batches = list(get_batches(context, batch_size=3))
    assert len(batches) == 2
    first_lengths = [len(ts) for ts in batches[0][0]]
    second_lengths = [len(ts) for ts in batches[1][0]]
    assert first_lengths == [1, 2, 3]
    assert second_lengths == [4, 5]


def test_numpy_variable_length_arrays():
    context = [np.arange(5), np.arange(8), np.arange(6)]
    batches = list(get_batches(context, batch_size=2))
    assert len(batches) == 2
    first_lengths = [len(ts) for ts in batches[0][0]]
    second_lengths = [len(ts) for ts in batches[1][0]]
    assert first_lengths == [5, 8]
    assert second_lengths == [6]


# ----- Tests for _batched_slice -----
def test_batched_slice_basic():
    data = torch.arange(10).reshape(5, 2)
    meta = [{"id": i} for i in range(5)]
    batches = list(_batched_slice(data, meta, 2))
    assert len(batches) == 3
    for series, meta_batch in batches:
        assert all(isinstance(ts, torch.Tensor) for ts in series)
        assert len(series) == len(meta_batch)
    assert [len(ts) for ts in batches[-1][0]] == [2]
    assert batches[-1][1] == [{"id": 4}]


def test_batched_slice_no_meta():
    data = torch.arange(6).reshape(3, 2)
    batches = list(_batched_slice(data, None, 2))
    assert len(batches) == 2
    assert all(meta == [{}] * len(series) for series, meta in batches)


# ----- Tests for _batched -----
def test_batched_even_split():
    data = list(range(8))
    batches = list(_batched(data, 2))
    assert len(batches) == 4
    assert all(len(batch) == 2 for batch in batches)


def test_batched_uneven_split():
    data = list(range(5))
    batches = list(_batched(data, 2))
    assert all(len(batch) == 2 for batch in batches[0:-1])
    assert batches[-1] == (4,)


# ----- Tests for _batch_iterable -----
def test_batch_iterable_groups_without_padding():
    data = [
        (torch.arange(3), {"id": 0}),
        (torch.arange(5), {"id": 1}),
        (torch.arange(2), {"id": 2}),
    ]
    batches = list(_batch_iterable(data, batch_size=2))
    assert len(batches) == 2
    first_series, first_meta = batches[0]
    second_series, second_meta = batches[1]
    assert [len(ts) for ts in first_series] == [3, 5]
    assert [len(ts) for ts in second_series] == [2]
    assert first_meta == [{"id": 0}, {"id": 1}]
    assert second_meta == [{"id": 2}]


# ----- Tests for _pad_time_series_batch -----
def test_pad_time_series_batch_left_pad_with_nan():
    series_list = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])]
    padded = _pad_time_series_batch(series_list, max_length=3)
    assert padded.shape == (2, 3)
    assert torch.isnan(padded[0, 0])
    torch.testing.assert_close(padded[0, 1:], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(padded[1], torch.tensor([3.0, 4.0, 5.0]))


def test_pad_time_series_batch_preserves_dtype_and_device():
    series_list = [torch.tensor([1, 2, 3], dtype=torch.int64, device=torch.device("cpu"))]
    padded = _pad_time_series_batch(series_list, max_length=4)
    assert padded.dtype == torch.float32  # cast to float for NaN support
    assert padded.device == torch.device("cpu")
    torch.testing.assert_close(padded[0, -3:], torch.tensor([1.0, 2.0, 3.0]))


def test_pad_time_series_batch_empty_input():
    padded = _pad_time_series_batch([], max_length=5)
    assert padded.shape == (0, 5)
