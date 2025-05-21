

from abc import ABC, abstractmethod
import itertools
from typing import List, Union, Literal

import numpy as np
import pandas as pd
import torch


try:
    from gluonts.dataset.common import Dataset as GluonDataset
    from gluonts.model.forecast import QuantileForecast
    _GLUONTS_AVAILABLE = True
except ImportError:
    _GLUONTS_AVAILABLE = False

try:
    from datasets import Dataset as HFDataset
    _HF_DATASETS_AVAILABLE = True
except ImportError:
    _HF_DATASETS_AVAILABLE = False


ContextType = Union[
    torch.Tensor,
    np.ndarray,
    List[torch.Tensor], 
    List[np.ndarray], 
    'gluonts.dataset.common.Dataset' if _GLUONTS_AVAILABLE else object,
    'datasets.Dataset' if _HF_DATASETS_AVAILABLE else object,
]


def _batched_slice(full_batch, batch_size):
    if len(full_batch) <= batch_size:
        yield full_batch 
    else:
        for i in range(0, len(full_batch), batch_size):
            yield full_batch[i : i + batch_size]

def _batched(iterable, n):
    it = iter(iterable)
    while (batch := tuple(itertools.islice(it, n))):
        yield batch


def _batch_pad_iterable(iterable, batch_size):
    for batch in _batched(iterable, batch_size):
        max_len = max(len(c) for c in batch)
        padded_batch = []
        for sample in batch:
            assert isinstance(sample, torch.Tensor)
            assert sample.ndim == 1
            padded_sample = torch.full(
                size=(max_len - len(sample),), fill_value=torch.nan, device=sample.device
            )
            padded_batch.append(padded_sample)
        yield torch.stack(padded_batch)


def _get_gluon_ts_map(**predict_kwargs):
    return lambda x: torch.Tensor(x["target"])


def _get_hf_map(**predict_kwargs):
    raise NotImplemented("Not implemented!")


def _get_batches(context, batch_size, **predict_kwargs):
    batches = None
    if isinstance(context, List):
        batches = _batch_pad_iterable(map(lambda x: torch.Tensor(x), context), batch_size)
    elif isinstance(context, torch.Tensor):
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2
        batches = _batched_slice(context, batch_size)
    elif isinstance(context, np.ndarray):
        if context.ndim == 1:
            context = np.expand_dims(context, axis=0)
        assert context.ndim == 2
        batches = map(lambda x: torch.Tensor(x), _batched_slice(context, batch_size))
    if batches is None and _GLUONTS_AVAILABLE:
        if isinstance(context, GluonDataset):
            batches = _batch_pad_iterable(map(_get_gluon_ts_map(**predict_kwargs), context), batch_size)
    if batches is None and _HF_DATASETS_AVAILABLE:
        if isinstance(context, HFDataset):
            batches = _batch_pad_iterable(map(_get_hf_map(**predict_kwargs), context), batch_size)
    if batches is None:
        raise ValueError(f"Context type {type(context)} not supported! Supported Types: {ContextType}")
    return batches


def _format_output(
    quantiles: torch.Tensor,
    means: torch.Tensor,
    sample_meta: List[dict],
    quantile_levels: List[float],
    output_type: Literal["torch", "numpy", "gluonts"],
):
    if output_type == "torch":
        return quantiles
    elif output_type == "numpy":
        return quantiles.cpu().numpy()
    elif output_type == "gluonts":
        if not _GLUONTS_AVAILABLE:
            raise ValueError(
                "output_type glutonts needs GluonTs but GluonTS is not available (not installed)!"
            )
        #if freq is None or start_dates is None:
        #    raise ValueError(
        #        "For 'gluonts' output type, 'freq' and 'start_dates' must be provided."
        #    )
        forecasts = []
        for i in range(quantiles.shape[0]):
            freq = sample_meta[i].get("freq", "h") # h is default frequency
            start_date = sample_meta[i].get("start_date", pd.Period("01-01-2000", freq=freq))
            forecasts.append(QuantileForecast(
                forecast_arrays=torch.cat((quantiles[i], means[i].unsqueeze(1)), dim=1).T.cpu().numpy(),
                start_date=start_date,
                item_id=None, # TODO: wire trough input meta
                forecast_keys=list(map(str, quantile_levels)) + ["mean"]
            ))
        return forecasts
    else:
        raise ValueError(f"Invalid output type: {output_type}")



def _as_generator(batches, predict_batch, to_output):
    for batch in batches:
        quantiles, mean = predict_batch(batch)
        yield to_output(quantiles, mean)



class ForecastModel(ABC):
    
    @abstractmethod
    def _forecast_quantiles(self, batch, **predict_kwargs):
        pass


    def forecast(
        self,
        context: ContextType,
        output_type: Literal["torch", "numpy", "gluonts"] = "torch",
        batch_size: int = 512,
        quantile_levels: List[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        yield_per_batch: bool = False,
        **predict_kwargs
    ):
        assert batch_size >= 1, "Batch size must be >= 1"

        batches = _get_batches(context, batch_size, **predict_kwargs)

        prediction_q = []
        prediction_m = []
        sample_meta = []
        for batch in batches:
            quantiles, mean = self._predict_quantiles(batch, **predict_kwargs)
            prediction_q.append(quantiles)
            prediction_m.append(mean)
            sample_meta.extend([{}] * len(batch))

        prediction_q = torch.cat(prediction_q, dim=0)
        prediction_m = torch.cat(prediction_m, dim=0)

        return _format_output(
            quantiles=prediction_q,
            means=prediction_m,
            sample_meta=sample_meta,
            quantile_levels=quantile_levels,
            output_type=output_type,
        )
