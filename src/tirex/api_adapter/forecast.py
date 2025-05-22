from abc import ABC, abstractmethod
from typing import List, Literal

import torch


from .standard_adapter import get_batches, ContextType

try:
    from .gluon import get_gluon_batches, format_gluonts_output
    _GLUONTS_AVAILABLE = True
except ImportError:
    _GLUONTS_AVAILABLE = False

try:
    from .hf_data import get_hfdata_batches
    _HF_DATASETS_AVAILABLE = True
except ImportError:
    _HF_DATASETS_AVAILABLE = False


DEF_TARGET_COLUMN  = "target"
DEF_META_COLUMNS = ("start", "item_id")


def _format_output(
    quantiles: torch.Tensor,
    means: torch.Tensor,
    sample_meta: List[dict],
    quantile_levels: List[float],
    output_type: Literal["torch", "numpy", "gluonts"],
):
    if output_type == "torch":
        return quantiles.cpu(), means.cpu()
    elif output_type == "numpy":
        return quantiles.cpu().numpy(), means.cpu().numpy()
    elif output_type == "gluonts":
        if not _GLUONTS_AVAILABLE:
            raise ValueError(
                "output_type glutonts needs GluonTs but GluonTS is not available (not installed)!"
            )
        return format_gluonts_output(quantiles, means, sample_meta, quantile_levels)
    else:
        raise ValueError(f"Invalid output type: {output_type}")



def _as_generator(batches, fc_func, quantile_levels, output_type, **predict_kwargs):
    for batch_ctx, batch_meta in batches:
        quantiles, mean = fc_func(batch_ctx, **predict_kwargs)
        yield _format_output(
            quantiles=quantiles,
            means=mean,
            sample_meta=batch_meta,
            quantile_levels=quantile_levels,
            output_type=output_type,
        )


def _gen_forecast(
    fc_func,
    batches,
    output_type,
    quantile_levels,
    yield_per_batch,
    **predict_kwargs
):
    
    if yield_per_batch:
        return _as_generator(batches, fc_func, quantile_levels, output_type, **predict_kwargs)

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
        quantile_levels=quantile_levels,
        output_type=output_type,
    )

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
        """
        Generates forecasts.
        This method takes historical "context" data as input and outputs probabilistic forecasts.

        Args:
            context (ContextType): The historical context" data of the time series:
                - torch.Tensor. 1D [context_length] or 2D [batch_dim, context_length] tensor
                - np.ndarray:  1D [context_length] or 2D [batch_dim, context_length] array
                - List[torch.Tensor]: List of 1D tensor (Samples with differnt lengths get padded per batch)
                - List[np.ndarray]. List of 1D array  (Samples with differnt lengths get padded per batch)

            output_type (Literal["torch", "numpy", "gluonts"], optional)
                Specifies the desired format of the returned forecasts:
                - "torch": Returns forecasts as `torch.Tensor` objects [batch_dim, forecast_len, |quantile_levels|]
                - "numpy": Returns forecasts as `numpy.ndarray` objects [batch_dim, forecast_len, |quantile_levels|]
                - "gluonts": Returns forecasts as list of GluonTS `Forecast` objects
                Defaults to "torch"

            batch_size (int, optional): The number of time series instances to process concurrently by the model.
                                        Defaults to 512.
            quantile_levels (List[float], optional): Quantile levels for which predictions should be generated.
                                                     Defaults to (0.1, 0.2, ..., 0.9).
            yield_per_batch (bool, optional): If `True`, the method will act as a generator, yielding
                                              forecasts batch by batch as they are computed.
            **predict_kwargs: Additional keyword arguments that are passed directly to the underlying
                              prediction mechanism of the pre-trained model. Refer to the model's
                              internal prediction method documentation for available options.
        Returns:
            Union[torch.Tensor, numpy.ndarray, List[gluonts.model.forecast.Forecast], Iterator]:
                The return type depends on `output_type` and `yield_per_batch`:
                - If `yield_per_batch` is `True`: An iterator that yields forecasts. Each yielded item
                  will correspond to a batch of forecasts in the format specified by `output_type`.
                - If `yield_per_batch` is `False`: A single object containing all forecasts.
                  - If `output_type="torch"`: A `torch.Tensor` (or tuple/dict of Tensors).
                  - If `output_type="numpy"`: A `numpy.ndarray` (or tuple/dict of arrays).
                  - If `output_type="gluonts"`: A list of `gluonts.model.forecast.Forecast` objects.
        """
        assert batch_size >= 1, "Batch size must be >= 1"
        batches = get_batches(context, batch_size)
        return _gen_forecast(
            self._forecast_quantiles, batches, output_type, quantile_levels,
            yield_per_batch, **predict_kwargs
        )


    def forecast_gluon(
        self,
        gluonDataset,
        output_type: Literal["torch", "numpy", "gluonts"] = "torch",
        batch_size: int = 512,
        quantile_levels: List[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        yield_per_batch: bool = False,
        data_kwargs: dict = {},
        **predict_kwargs
    ):
        assert batch_size >= 1, "Batch size must be >= 1"
        if not _GLUONTS_AVAILABLE:
            raise ValueError(
                "forecast_gluon glutonts needs GluonTs but GluonTS is not available (not installed)!"
            )
        batches = get_gluon_batches(gluonDataset, batch_size, **data_kwargs)
        return _gen_forecast(
            self._forecast_quantiles, batches, output_type, quantile_levels,
            yield_per_batch, **predict_kwargs
        )

    def forecast_hfdata(
        self,
        hf_dataset,
        output_type: Literal["torch", "numpy", "gluonts"] = "torch",
        batch_size: int = 512,
        quantile_levels: List[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        yield_per_batch: bool = False,
        data_kwargs: dict = {},
        **predict_kwargs      
    ):
        assert batch_size >= 1, "Batch size must be >= 1"
        if not _HF_DATASETS_AVAILABLE:
            raise ValueError(
                "forecast_hfdata glutonts needs hugginface datasets but datasets is not available (not installed)!"
            )
        batches = get_hfdata_batches(hf_dataset, batch_size, **data_kwargs)
        return _gen_forecast(
            self._forecast_quantiles, batches, output_type, quantile_levels,
            yield_per_batch, **predict_kwargs
        )


