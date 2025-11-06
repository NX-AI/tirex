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
from .patcher import PatchedTokenizer
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

        self.tokenizer = PatchedTokenizer(patch_size=self.config.input_patch_size)

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

    @torch.inference_mode()
    def _forecast_quantiles(
        self,
        context: torch.Tensor,
        prediction_length: int | None = None,
        output_device: str = "cpu",
        max_accelerated_rollout_steps: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.input_patch_embedding.hidden_layer.weight.device
        context = context.to(device)

        quantiles = self._forecast_tensor(context, prediction_length, new_patch_count=max_accelerated_rollout_steps)
        quantiles = quantiles.to(torch.device(output_device)).swapaxes(1, 2)

        mean = quantiles[:, :, self.config.quantiles.index(0.5)].squeeze(-1)  # median as mean
        return quantiles, mean

    def _forecast_tensor(
        self,
        context: torch.Tensor,
        prediction_length: int | None = None,
        new_patch_count: int = 1,
    ) -> torch.Tensor:
        predictions = []
        if prediction_length is None:
            prediction_length = self.tokenizer.patch_size
        if prediction_length <= 0:
            raise ValueError("prediction_length needs to be > 0")

        remaining = -(prediction_length // -self.tokenizer.patch_size)

        context = context.to(dtype=torch.float32)
        while remaining > 0:
            new_patch_count = min(remaining, new_patch_count)
            prediction = self._forecast_single_step(context, new_patch_count)

            predictions.append(prediction)
            remaining -= new_patch_count

            if remaining <= 0:
                break

            context = torch.cat([context, torch.full_like(prediction[:, 0, :], fill_value=torch.nan)], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length].to(dtype=torch.float32)

    def _forecast_single_step(self, context: torch.Tensor, new_patch_count: int = 1) -> torch.Tensor:
        max_context, min_context = self.config.train_ctx_len, self.config.train_ctx_len

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

        input_token, tokenizer_state = self.tokenizer.input_transform(context)
        prediction = self._forward_model_tokenized(input_token=input_token, new_patch_count=new_patch_count)
        predicted_token = prediction[:, :, -new_patch_count:, :].to(input_token)  # predicted token
        # Shape: [bs, num_quantiles, num_predicted_token, output_patch_size]
        predicted_token = self.tokenizer.output_transform(predicted_token, tokenizer_state)

        return predicted_token

    def _forward_model_tokenized(self, input_token: torch.Tensor, input_mask=None, new_patch_count=1):
        input_mask = (
            input_mask.to(input_token.dtype)
            if input_mask is not None
            else torch.isnan(input_token).logical_not().to(input_token.dtype)
        )
        assert new_patch_count >= 1
        bs, numb_ctx_token, token_dim = input_token.shape
        if new_patch_count > 1:
            input_token_rollout_pad = torch.full(
                (bs, new_patch_count - 1, token_dim),
                fill_value=torch.nan,
                device=input_token.device,
                dtype=input_token.dtype,
            )
            input_token = torch.cat((input_token, input_token_rollout_pad), dim=1)
            input_mask_rollout_pad = torch.full(
                (bs, new_patch_count - 1, token_dim),
                fill_value=False,
                device=input_mask.device,
                dtype=input_mask.dtype,
            )
            input_mask = torch.cat((input_mask, input_mask_rollout_pad), dim=1)

        input_token = torch.nan_to_num(input_token, nan=self.config.nan_mask_value)

        quantile_preds = self._forward_model(torch.cat((input_token, input_mask), dim=2))

        quantile_preds = torch.unflatten(
            quantile_preds, -1, (len(self.config.quantiles), self.config.output_patch_size)
        )
        quantile_preds = torch.transpose(quantile_preds, 1, 2)  # switch quantile and num_token_dimension
        # quantile_preds: [batch_size, num_quantiles, num_token, output_patch_size]
        return quantile_preds

    def _forward_model(self, input: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_patch_embedding(input)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.out_norm(hidden_states)

        return self.output_patch_embedding(hidden_states)

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
