# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from abc import ABC, abstractmethod
from functools import partial
from math import ceil
from typing import Literal, Optional

import torch

from tirex.util import frequency_resample

from .standard_adapter import ContextType, get_batches

DEF_TARGET_COLUMN = "target"
DEF_META_COLUMNS = ("start", "item_id")

# Allowed resampling strategies (extend as new strategies are implemented)
RESAMPLE_STRATEGIES: list[str] = ["frequency"]


def _format_output(
    quantiles: torch.Tensor,
    means: torch.Tensor,
    sample_meta: list[dict],
    output_type: Literal["torch", "numpy", "gluonts"],
):
    if output_type == "torch":
        return quantiles.cpu(), means.cpu()
    elif output_type == "numpy":
        return quantiles.cpu().numpy(), means.cpu().numpy()
    elif output_type == "gluonts":
        try:
            from .gluon import format_gluonts_output
        except ImportError:
            raise ValueError("output_type glutonts needs GluonTs but GluonTS is not available (not installed)!")
        return format_gluonts_output(quantiles, means, sample_meta)
    else:
        raise ValueError(f"Invalid output type: {output_type}")


def _pad_time_series_batch(
    batch_series: list[torch.Tensor],
    max_length: int,
) -> torch.Tensor:
    if not batch_series:
        return torch.empty((0, max_length))

    first = batch_series[0]
    dtype = first.dtype if first.is_floating_point() else torch.float32
    device = first.device

    padded = torch.full((len(batch_series), max_length), float("nan"), dtype=dtype, device=device)

    for idx, series in enumerate(batch_series):
        series = series.to(padded.dtype)
        series_len = series.shape[0]
        padded[idx, max_length - series_len :] = series

    return padded


def _as_generator(batches, fc_func, output_type, **predict_kwargs):
    for batch_ctx, batch_meta in batches:
        quantiles, mean = fc_func(batch_ctx, **predict_kwargs)
        yield _format_output(
            quantiles=quantiles,
            means=mean,
            sample_meta=batch_meta,
            output_type=output_type,
        )


def _call_fc_with_padding(fc_func, batch_series: list[torch.Tensor], **predict_kwargs):
    if not batch_series:
        raise ValueError("Received empty batch for forecasting")

    max_len = max(series.shape[0] for series in batch_series)
    padded_ts = _pad_time_series_batch(batch_series, max_len)

    return fc_func(padded_ts, **predict_kwargs)


def _resample_fc_func_wrapper(
    fc_func,
    batch,
    resample_strategy: str,
    max_context: int = 2016,
    **predict_kwargs,
):
    # downsample the time series based on the dominant frequencies, if enabled
    max_period = (max_context // 1000) * 500
    prediction_length = predict_kwargs.get("prediction_length", 100)
    batch_resampled_ts: list[torch.Tensor] = []
    fc_resample_fns = []
    scaling_factors = []

    # select the function doing the resampling
    ctx_resample_fn = lambda x: (x, 1.0, (lambda y: y))
    match resample_strategy:
        case "frequency":
            ctx_resample_fn = frequency_resample
        case _:
            raise RuntimeError("This shouldn't happen.")

    for series in batch:
        resampled_ts, _sample_factor, fc_resample_fn = ctx_resample_fn(
            series,
            prediction_length=prediction_length,
            max_period=max_period,
        )

        batch_resampled_ts.append(resampled_ts)
        fc_resample_fns.append(fc_resample_fn)
        scaling_factors.append(_sample_factor)

    # Compute per-item required horizons (in downsampled domain)
    per_item_pred_lens = [int(ceil(prediction_length * sf)) for sf in scaling_factors]
    max_pred_len = max(per_item_pred_lens) if per_item_pred_lens else int(prediction_length)
    predict_kwargs.update(prediction_length=max_pred_len)

    max_ts_length = max(ts.shape[0] for ts in batch_resampled_ts)
    padded_ts = _pad_time_series_batch(batch_resampled_ts, max_ts_length)
    print(f"Average sample batch factor: {sum(scaling_factors) / len(scaling_factors)}")

    # generate prediction
    fc_quantiles, fc_mean = fc_func(padded_ts, **predict_kwargs)

    batch_prediction_q = []
    batch_prediction_m = []
    for el_q, el_m, fc_resample_fn, item_pred_len in zip(fc_quantiles, fc_mean, fc_resample_fns, per_item_pred_lens):
        # truncate the forecasts to their individual sample factor adjusted prediction lengths
        el_q = el_q[:item_pred_len, ...]  # [T, Q]
        el_m = el_m[:item_pred_len]  # [T]

        # upsample prediction
        quantiles = fc_resample_fn(el_q.squeeze(0).transpose(0, 1)).transpose(0, 1)  # [T, Q]
        mean = fc_resample_fn(el_m.squeeze(0))

        quantiles = quantiles[:prediction_length, ...]
        mean = mean[:prediction_length]

        batch_prediction_q.append(quantiles)
        batch_prediction_m.append(mean)

    return torch.stack(batch_prediction_q, dim=0), torch.stack(batch_prediction_m, dim=0)


def _gen_forecast(
    fc_func,
    batches,
    output_type,
    yield_per_batch,
    resample_strategy: str | None = None,
    max_context: int = 2016,
    **predict_kwargs,
):
    base_fc_func = fc_func

    if resample_strategy is not None:
        if resample_strategy not in RESAMPLE_STRATEGIES:
            raise ValueError(f"Invalid resample strategy: {resample_strategy}. Allowed: {RESAMPLE_STRATEGIES}")
        fc_func = partial(
            _resample_fc_func_wrapper,
            base_fc_func,
            resample_strategy=resample_strategy,
            max_context=max_context,
        )
    else:
        fc_func = partial(_call_fc_with_padding, base_fc_func)

    if yield_per_batch:
        return _as_generator(batches, fc_func, output_type, **predict_kwargs)

    prediction_q = []
    prediction_m = []
    sample_meta = []
    for batch_ctx, batch_meta in batches:
        quantiles, mean = fc_func(batch_ctx, **predict_kwargs)
        prediction_q.append(quantiles)
        prediction_m.append(mean)
        sample_meta.extend(batch_meta)

    prediction_q = torch.cat(prediction_q, dim=0)
    prediction_m = torch.cat(prediction_m, dim=0)

    return _format_output(
        quantiles=prediction_q,
        means=prediction_m,
        sample_meta=sample_meta,
        output_type=output_type,
    )


class ForecastModel(ABC):
    @abstractmethod
    def _forecast_quantiles(self, batch, **predict_kwargs):
        pass

    @property
    def max_context_length(self) -> int:
        # retrieve the max_context attribute of the model configuration if present
        return getattr(getattr(self, "config", None), "max_context", 2016)

    def forecast(
        self,
        context: ContextType,
        output_type: Literal["torch", "numpy", "gluonts"] = "torch",
        batch_size: int = 512,
        yield_per_batch: bool = False,
        resample_strategy: Literal["frequency"] | None = None,
        **predict_kwargs,
    ):
        """
        This method takes historical context data as input and outputs probabilistic forecasts.

        Args:
            output_type (Literal["torch", "numpy", "gluonts"], optional):
                Specifies the desired format of the returned forecasts:
                - "torch": Returns forecasts as `torch.Tensor` objects [batch_dim, forecast_len, quantile_count]
                - "numpy": Returns forecasts as `numpy.ndarray` objects [batch_dim, forecast_len, quantile_count]
                - "gluonts": Returns forecasts as a list of GluonTS `Forecast` objects.
                Defaults to "torch".

            batch_size (int, optional): The number of time series instances to process concurrently by the model.
                                        Defaults to 512. Must be $>= 1$.

            yield_per_batch (bool, optional): If `True`, the method will act as a generator, yielding
                                              forecasts batch by batch as they are computed.
                                              Defaults to `False`.

            resample_strategy (Optional[str], optional): Choose a resampling strategy. Allowed values: "frequency".
                                                If `None`, no resampling is applied. Currently only "frequency" is supported.

            **predict_kwargs: Additional keyword arguments that are passed directly to the underlying
                              prediction mechanism of the pre-trained model. Refer to the model's
                              internal prediction method documentation for available options.

        Returns:
            The return type depends on `output_type` and `yield_per_batch`:
                - If `yield_per_batch` is `True`: An iterator that yields forecasts. Each yielded item
                  will correspond to a batch of forecasts in the format specified by `output_type`.
                - If `yield_per_batch` is `False`: A single object containing all forecasts.
                  - If `output_type="torch"`: `Tuple[torch.Tensor, torch.Tensor]` (quantiles, mean).
                  - If `output_type="numpy"`: `Tuple[numpy.ndarray, numpy.ndarray]` (quantiles, mean).
                  - If `output_type="gluonts"`: A `List[gluonts.model.forecast.Forecast]` of all forecasts.

        Args:
            context (ContextType): The historical "context" data of the time series:
                - `torch.Tensor`: 1D `[context_length]` or 2D `[batch_dim, context_length]` tensor
                - `np.ndarray`: 1D `[context_length]` or 2D `[batch_dim, context_length]` array
                - `List[torch.Tensor]`: List of 1D tensors (samples with different lengths get padded per batch)
                - `List[np.ndarray]`: List of 1D arrays (samples with different lengths get padded per batch)
        """
        assert batch_size >= 1, "Batch size must be >= 1"
        batches = get_batches(context, batch_size)
        return _gen_forecast(
            self._forecast_quantiles,
            batches,
            output_type,
            yield_per_batch,
            resample_strategy=resample_strategy,
            max_context=self.max_context_length,
            **predict_kwargs,
        )

    def forecast_gluon(
        self,
        gluonDataset,
        output_type: Literal["torch", "numpy", "gluonts"] = "torch",
        batch_size: int = 512,
        yield_per_batch: bool = False,
        resample_strategy: Literal["frequency"] | None = None,
        data_kwargs: dict = {},
        **predict_kwargs,
    ):
        """
        This method takes historical context data as input and outputs probabilistic forecasts.

        Args:
            output_type (Literal["torch", "numpy", "gluonts"], optional):
                Specifies the desired format of the returned forecasts:
                - "torch": Returns forecasts as `torch.Tensor` objects [batch_dim, forecast_len, quantile_count]
                - "numpy": Returns forecasts as `numpy.ndarray` objects [batch_dim, forecast_len, quantile_count]
                - "gluonts": Returns forecasts as a list of GluonTS `Forecast` objects.
                Defaults to "torch".

            batch_size (int, optional): The number of time series instances to process concurrently by the model.
                                        Defaults to 512. Must be $>= 1$.

            yield_per_batch (bool, optional): If `True`, the method will act as a generator, yielding
                                              forecasts batch by batch as they are computed.
                                              Defaults to `False`.

            resample_strategy (Optional[str], optional): Choose a resampling strategy. Allowed values: "frequency".
                                                If `None`, no resampling is applied. Currently only "frequency" is supported.

            **predict_kwargs: Additional keyword arguments that are passed directly to the underlying
                              prediction mechanism of the pre-trained model. Refer to the model's
                              internal prediction method documentation for available options.

        Returns:
            The return type depends on `output_type` and `yield_per_batch`:
                - If `yield_per_batch` is `True`: An iterator that yields forecasts. Each yielded item
                  will correspond to a batch of forecasts in the format specified by `output_type`.
                - If `yield_per_batch` is `False`: A single object containing all forecasts.
                  - If `output_type="torch"`: `Tuple[torch.Tensor, torch.Tensor]` (quantiles, mean).
                  - If `output_type="numpy"`: `Tuple[numpy.ndarray, numpy.ndarray]` (quantiles, mean).
                  - If `output_type="gluonts"`: A `List[gluonts.model.forecast.Forecast]` of all forecasts.

        Args:
            gluonDataset (gluon_ts.dataset.common.Dataset): A GluonTS dataset object containing the
                                                            historical time series data.

            data_kwargs (dict, optional): Additional keyword arguments passed to the
                                          autogluon data processing function.
        """
        assert batch_size >= 1, "Batch size must be >= 1"
        try:
            from .gluon import get_gluon_batches
        except ImportError:
            raise ValueError("forecast_gluon glutonts needs GluonTs but GluonTS is not available (not installed)!")

        batches = get_gluon_batches(gluonDataset, batch_size, **data_kwargs)
        return _gen_forecast(
            self._forecast_quantiles,
            batches,
            output_type,
            yield_per_batch,
            resample_strategy=resample_strategy,
            max_context=self.max_context_length,
            **predict_kwargs,
        )

    def forecast_hfdata(
        self,
        hf_dataset,
        output_type: Literal["torch", "numpy", "gluonts"] = "torch",
        batch_size: int = 512,
        yield_per_batch: bool = False,
        resample_strategy: Literal["frequency"] | None = None,
        data_kwargs: dict = {},
        **predict_kwargs,
    ):
        """
        This method takes historical context data as input and outputs probabilistic forecasts.

        Args:
            output_type (Literal["torch", "numpy", "gluonts"], optional):
                Specifies the desired format of the returned forecasts:
                - "torch": Returns forecasts as `torch.Tensor` objects [batch_dim, forecast_len, quantile_count]
                - "numpy": Returns forecasts as `numpy.ndarray` objects [batch_dim, forecast_len, quantile_count]
                - "gluonts": Returns forecasts as a list of GluonTS `Forecast` objects.
                Defaults to "torch".

            batch_size (int, optional): The number of time series instances to process concurrently by the model.
                                        Defaults to 512. Must be $>= 1$.

            yield_per_batch (bool, optional): If `True`, the method will act as a generator, yielding
                                              forecasts batch by batch as they are computed.
                                              Defaults to `False`.

            resample_strategy (Optional[str], optional): Choose a resampling strategy. Allowed values: "frequency".
                                                If `None`, no resampling is applied. Currently only "frequency" is supported.

            **predict_kwargs: Additional keyword arguments that are passed directly to the underlying
                              prediction mechanism of the pre-trained model. Refer to the model's
                              internal prediction method documentation for available options.

        Returns:
            The return type depends on `output_type` and `yield_per_batch`:
                - If `yield_per_batch` is `True`: An iterator that yields forecasts. Each yielded item
                  will correspond to a batch of forecasts in the format specified by `output_type`.
                - If `yield_per_batch` is `False`: A single object containing all forecasts.
                  - If `output_type="torch"`: `Tuple[torch.Tensor, torch.Tensor]` (quantiles, mean).
                  - If `output_type="numpy"`: `Tuple[numpy.ndarray, numpy.ndarray]` (quantiles, mean).
                  - If `output_type="gluonts"`: A `List[gluonts.model.forecast.Forecast]` of all forecasts.

        Args:
            hf_dataset (datasets.Dataset): A Hugging Face `Dataset` object containing the
                                           historical time series data.

            data_kwargs (dict, optional): Additional keyword arguments passed to the
                                          datasets data processing function.
        """
        assert batch_size >= 1, "Batch size must be >= 1"
        try:
            from .hf_data import get_hfdata_batches
        except ImportError:
            raise ValueError(
                "forecast_hfdata glutonts needs HuggingFace datasets but datasets is not available (not installed)!"
            )

        batches = get_hfdata_batches(hf_dataset, batch_size, **data_kwargs)
        return _gen_forecast(
            self._forecast_quantiles,
            batches,
            output_type,
            yield_per_batch,
            resample_strategy=resample_strategy,
            max_context=self.max_context_length,
            **predict_kwargs,
        )
