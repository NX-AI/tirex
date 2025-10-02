# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..api_adapter.forecast import ForecastModel
from ..base import PretrainedModel
from ..util import dataclass_from_dict
from .patcher import PatchedUniTokenizer
from .slstm.block import RMSNorm, sLSTMBlock, sLSTMBlockConfig

LOGGER = logging.getLogger()


@dataclass
class TiRexZeroConfig:
    input_patch_size: int
    output_patch_size: int
    quantiles: list[float]
    block_kwargs: dict
    input_ff_dim: int
    train_ctx_len: int
    nan_mask_value: int = 0


class TiRexZero(nn.Module, PretrainedModel, ForecastModel):
    def __init__(self, backend, model_config: TiRexZeroConfig, train_ctx_len=None):
        super().__init__()
        self.config = TiRexZeroConfig(**model_config, train_ctx_len=train_ctx_len, nan_mask_value=0)
        assert self.config.input_patch_size == self.config.output_patch_size

        self.tokenizer = PatchedUniTokenizer(patch_size=self.config.input_patch_size)

        num_blocks = self.config.block_kwargs["num_blocks"]
        block_config = dataclass_from_dict(sLSTMBlockConfig, self.config.block_kwargs)
        self.input_patch_embedding = ResidualBlock(
            in_dim=self.config.input_patch_size * 2,
            h_dim=self.config.input_ff_dim,
            out_dim=block_config.embedding_dim,
        )

        self.blocks = nn.ModuleList([sLSTMBlock(block_config, backend) for i in range(num_blocks)])

        self.out_norm = RMSNorm(block_config.embedding_dim)

        self.output_patch_embedding = ResidualBlock(
            in_dim=block_config.embedding_dim,
            h_dim=self.config.input_ff_dim,
            out_dim=len(self.config.quantiles) * self.config.output_patch_size,
        )

    @classmethod
    def register_name(cls):
        return "TiRex"

    def _forecast_quantiles(
        self,
        context: torch.Tensor,
        prediction_length: int | None = None,
        quantile_levels: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        output_device: str = "cpu",
        auto_cast: bool = False,
        **predict_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.input_patch_embedding.hidden_layer.weight.device
        context = context.to(device)

        with torch.autocast(device_type=device.type, enabled=auto_cast):
            predictions = self._forecast_tensor(
                context=context, prediction_length=prediction_length, **predict_kwargs
            ).detach()
        predictions = predictions.to(torch.device(output_device)).swapaxes(1, 2)

        training_quantile_levels = self.config.quantiles

        if set(quantile_levels).issubset(set(training_quantile_levels)):
            quantiles = predictions[..., [training_quantile_levels.index(q) for q in quantile_levels]]
        else:
            quantiles = self._interpolate_quantiles(predictions, quantile_levels)

        # median as mean
        mean = predictions[:, :, training_quantile_levels.index(0.5)]
        return quantiles, mean

    @torch.inference_mode()
    def _forecast_tensor(
        self,
        context: torch.Tensor,
        prediction_length: int | None = None,
        max_context: int | None = None,
        max_accelerated_rollout_steps: int = 1,
    ) -> torch.Tensor:
        predictions = []
        if prediction_length is None:
            prediction_length = self.tokenizer.patch_size
        remaining = -(prediction_length // -self.tokenizer.patch_size)
        if max_context is None:
            max_context = self.config.train_ctx_len
        min_context = max(self.config.train_ctx_len, max_context)

        context = context.to(dtype=torch.float32)
        while remaining > 0:
            if context.shape[-1] > max_context:
                context = context[..., -max_context:]
            if context.shape[-1] < min_context:
                pad = torch.full(
                    (context.shape[0], min_context - context.shape[-1]),
                    fill_value=torch.nan,
                    device=context.device,
                    dtype=context.dtype,
                )
                context = torch.concat((pad, context), dim=1)
            tokenized_tensor, tokenizer_state = self.tokenizer.context_input_transform(context)
            fut_rollouts = min(remaining, max_accelerated_rollout_steps)
            with torch.no_grad():
                prediction, _ = self._forward_model_tokenized(input_token=tokenized_tensor, rollouts=fut_rollouts)
                prediction = prediction[:, :, -fut_rollouts:, :].to(tokenized_tensor)  # predicted token
                # [bs, num_quantiles, num_predicted_token, output_patch_size]
            prediction = self.tokenizer.output_transform(prediction, tokenizer_state)
            prediction = prediction.flatten(start_dim=2)

            predictions.append(prediction)
            remaining -= fut_rollouts

            if remaining <= 0:
                break

            context = torch.cat([context, torch.full_like(prediction[:, 0, :], fill_value=torch.nan)], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length].to(dtype=torch.float32)

    def _forward_model_tokenized(
        self,
        input_token: torch.Tensor,
        input_mask=None,
        rollouts=1,
    ):
        input_mask = (
            input_mask.to(input_token.dtype)
            if input_mask is not None
            else torch.isnan(input_token).logical_not().to(input_token.dtype)
        )
        assert rollouts >= 1
        bs, numb_ctx_token, token_dim = input_token.shape
        if rollouts > 1:
            input_token_rollout_pad = torch.full(
                (bs, rollouts - 1, token_dim),
                fill_value=torch.nan,
                device=input_token.device,
                dtype=input_token.dtype,
            )
            input_token = torch.cat((input_token, input_token_rollout_pad), dim=1)
            input_mask_rollout_pad = torch.full(
                (bs, rollouts - 1, token_dim),
                fill_value=False,
                device=input_mask.device,
                dtype=input_mask.dtype,
            )
            input_mask = torch.cat((input_mask, input_mask_rollout_pad), dim=1)

        input_token = torch.nan_to_num(input_token, nan=self.config.nan_mask_value)

        hidden_states = self.input_patch_embedding(torch.cat((input_token, input_mask), dim=2))

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.out_norm(hidden_states)

        quantile_preds = self.output_patch_embedding(hidden_states)
        quantile_preds = torch.unflatten(
            quantile_preds, -1, (len(self.config.quantiles), self.config.output_patch_size)
        )
        quantile_preds = torch.transpose(quantile_preds, 1, 2)  # switch quantile and num_token_dimension
        # quantile_preds: [batch_size, num_quantiles, num_token, output_patch_size]

        quantile_preds = self._forward_model(torch.cat((input_token, input_mask), dim=2))

        quantile_preds = torch.unflatten(
            quantile_preds, -1, (len(self.config.quantiles), self.config.output_patch_size)
        )
        quantile_preds = torch.transpose(quantile_preds, 1, 2)  # switch quantile and num_token_dimension
        # quantile_preds: [batch_size, num_quantiles, num_token, output_patch_size]
        return quantile_preds, hidden_states

    def _forward_model(self, input: torch.Tensor):
        hidden_states = self.input_patch_embedding(input)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.out_norm(hidden_states)

        return self.output_patch_embedding(hidden_states)

    def _interpolate_quantiles(self, predictions: torch.Tensor, quantile_levels: list[float]):
        training_quantile_levels = self.config.quantiles
        if min(quantile_levels) < min(training_quantile_levels) or max(quantile_levels) > max(training_quantile_levels):
            logging.warning(
                f"Requested quantile levels ({quantile_levels}) fall outside the range of "
                f"quantiles the model was trained on ({training_quantile_levels}). "
                "Predictions for out-of-range quantiles will be clamped to the nearest "
                "boundary of the trained quantiles (i.e., minimum or maximum trained level). "
                "This can significantly impact prediction accuracy, especially for extreme quantiles. "
            )

        augmented_predictions = torch.cat(
            [predictions[..., [0]], predictions, predictions[..., [-1]]],
            dim=-1,
        )
        quantiles = torch.quantile(
            augmented_predictions,
            q=torch.tensor(quantile_levels, dtype=augmented_predictions.dtype),
            dim=-1,
        ).permute(1, 2, 0)
        return quantiles

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # rename keys of state_dict, because the block_stack was moved directly into the tirex model
        checkpoint["state_dict"] = {k.replace("block_stack.", ""): v for k, v in checkpoint["state_dict"].items()}


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor):
        hid = F.relu(self.hidden_layer(x))
        out = self.output_layer(hid)
        res = self.residual_layer(x)
        out = out + res
        return out
