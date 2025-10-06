# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import itertools
from collections.abc import Iterable, Iterator, Sequence
from typing import Union

import numpy as np
import torch

ContextType = Union[
    torch.Tensor,
    np.ndarray,
    list[torch.Tensor],
    list[np.ndarray],
]


def _ensure_1d_tensor(sample) -> torch.Tensor:
    if isinstance(sample, torch.Tensor):
        tensor = sample
    else:
        tensor = torch.as_tensor(sample)

    if tensor.ndim > 1:
        tensor = tensor.squeeze()

    assert tensor.ndim == 1, "Each sample must be one-dimensional"
    return tensor


def _batched_slice(
    full_batch,
    full_meta: list[dict] | None,
    batch_size: int,
) -> Iterator[tuple[list[torch.Tensor], list[dict]]]:
    total = len(full_batch)
    for start in range(0, total, batch_size):
        batch = full_batch[start : start + batch_size]
        meta = full_meta[start : start + batch_size] if full_meta is not None else [{} for _ in range(len(batch))]

        batch_series = []
        for idx in range(len(batch)):
            sample = batch[idx]
            tensor = _ensure_1d_tensor(sample)
            batch_series.append(tensor)

        yield batch_series, meta


def _batched(iterable: Iterable, n: int):
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def _batch_iterable(
    iterable: Iterable[tuple[torch.Tensor, dict | None]],
    batch_size: int,
) -> Iterator[tuple[list[torch.Tensor], list[dict]]]:
    for batch in _batched(iterable, batch_size):
        series_list: list[torch.Tensor] = []
        meta_list: list[dict] = []

        for sample, meta in batch:
            tensor = _ensure_1d_tensor(sample)
            assert len(tensor) > 0, "Each sample needs to have a length > 0"
            series_list.append(tensor)
            meta_list.append(meta if meta is not None else {})

        yield series_list, meta_list


def get_batches(context: ContextType, batch_size: int):
    batches = None
    if isinstance(context, torch.Tensor):
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2
        batches = _batched_slice(context, None, batch_size)
    elif isinstance(context, np.ndarray):
        if context.ndim == 1:
            context = np.expand_dims(context, axis=0)
        assert context.ndim == 2
        batches = _batched_slice(context, None, batch_size)
    elif isinstance(context, (list, Iterable)):
        batches = _batch_iterable(map(lambda x: (torch.Tensor(x), None), context), batch_size)
    if batches is None:
        raise ValueError(f"Context type {type(context)} not supported! Supported Types: {ContextType}")
    return batches
