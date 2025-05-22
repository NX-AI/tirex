import logging
from typing import List, Optional, Tuple
import torch

from ..api_adapter.forecast import ForecastModel


class TensorQuantileUniPredictMixin(ForecastModel):
    
    def _forecast_tensor(
        self,
        context: torch.Tensor,
        prediction_length: Optional[int] = None,
        **predict_kwargs,
    ) -> torch.Tensor:
        raise NotImplemented("Not implemente!")
        

    def _forecast_quantiles(
        self,
        context: torch.Tensor,
        prediction_length: Optional[int] = None,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **predict_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:            
        predictions = (
            self._forecast_tensor(context, prediction_length=prediction_length, **predict_kwargs)
            .detach()
            .swapaxes(1, 2)
        )

        training_quantile_levels = list(self.quantiles)

        if set(quantile_levels).issubset(set(training_quantile_levels)):
            quantiles = predictions[
                ..., [training_quantile_levels.index(q) for q in quantile_levels]
            ]
        else:
            if min(quantile_levels) < min(training_quantile_levels) or max(quantile_levels) > max(training_quantile_levels):
                logging.warning(
                    f"\tQuantiles to be predicted ({quantile_levels}) are not within the range of "
                    f"quantiles that the model was trained on ({training_quantile_levels}). "
                    "Quantile predictions will be set to the minimum/maximum levels at which the model"
                    "was trained on. This may significantly affect the quality of the predictions."
                )
            # Interpolate quantiles
            augmented_predictions = torch.cat(
                [predictions[..., [0]], predictions, predictions[..., [-1]]],
                dim=-1,
            )
            quantiles = torch.quantile(
                augmented_predictions,
                q=torch.tensor(quantile_levels, dtype=augmented_predictions.dtype),
                dim=-1,
            ).permute(1, 2, 0)
        # median as mean
        mean = predictions[:, :, training_quantile_levels.index(0.5)]
        return quantiles, mean
