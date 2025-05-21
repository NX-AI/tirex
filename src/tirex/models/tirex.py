from abc import ABC
from dataclasses import dataclass
from collections import deque
import logging
from typing import Dict, List, Optional, Tuple
from dacite import from_dict, Config
import lightning as L
import torch

#from models.txlstm.model.covert import LoadConvertedCheckpointMixin

from base import PretrainedModel

from models.mixed_stack import xLSTMMixedLargeBlockStack, xLSTMMixedLargeConfig
from models.components import ResidualBlock, PatchedUniTokenizer
from models.predict_utils import TensorQuantileUniPredictMixin


LOGGER = logging.getLogger()


@dataclass
class TiRexZeroConfig:
    input_patch_size: int
    output_patch_size: int
    quantiles: List[float]
    block_kwargs: Dict
    input_ff_dim: int


class TiRexZero(L.LightningModule, PretrainedModel, TensorQuantileUniPredictMixin): #LoadConvertedCheckpointMixin
    def __init__(self, model_config: dict, train_ctx_len = None):
        super().__init__()
        self.model_config: TiRexZeroConfig = from_dict(TiRexZeroConfig, model_config, config=Config(strict=True))
        assert self.model_config.input_patch_size == self.model_config.output_patch_size
        self.train_ctx_len = train_ctx_len

        # Block Stack
        self.nan_mask_value = 0
        self.block_stack, resolved_config = self.init_block(self.model_config.block_kwargs)
        self.model_config.block_kwargs = resolved_config

        # Input Layer
        self.input_patch_embedding = ResidualBlock(
            in_dim=self.model_config.input_patch_size * 2,
            h_dim=self.model_config.input_ff_dim,
            out_dim=self.model_config.block_kwargs.embedding_dim,
        )
        self.tokenizer = PatchedUniTokenizer(
            patch_size=self.model_config.input_patch_size,
        )

        # Output Layer
        self.num_quantiles = len(self.model_config.quantiles)
        quantiles = torch.tensor(self.model_config.quantiles)
        self.register_buffer("quantiles", quantiles, persistent=False)

        self.output_patch_embedding = ResidualBlock(
                in_dim=self.model_config.block_kwargs.embedding_dim,
                h_dim=self.model_config.input_ff_dim,
                out_dim=self.num_quantiles * self.model_config.output_patch_size,
            )

        self.save_hyperparameters()
    
    @classmethod
    def register_name(cls):
        return "tirex"

    def init_block(self, block_kwargs):
        config = from_dict(xLSTMMixedLargeConfig, block_kwargs)
        return xLSTMMixedLargeBlockStack(config), config


    @property
    def quantiles(self):
        return self.model.quantiles


    def _forward_model_tokenized(
        self,
        input_token,
        input_mask,
        rollouts=1,
    ):
        input_mask = (
            input_mask.to(input_token.dtype)
            if input_mask is not None
            else torch.isnan(input_token).logical_not().to(input_token.dtype)
        )
        assert rollouts >= 1
        batch_size, numb_token, token_dim = input_token.shape
        input_token = torch.nan_to_num(input_token, nan=self.nan_mask_value)
        input_embeds = self.input_patch_embedding(torch.cat((input_token, input_mask), dim=2))

        #hidden_states = []
        #for rollout in range(rollout):
        x = self.block_stack(input_embeds)
        if isinstance(x, tuple):
            hidden_states = x[0]
        else:
            hidden_states = x

        quantile_preds = self.output_patch_embedding(hidden_states)
        quantile_preds = torch.unflatten(quantile_preds, -1, (self.num_quantiles, self.model_config.output_patch_size))
        quantile_preds = torch.transpose(quantile_preds, 1, 2) # switch quantile and num_token_dimension
        #quantile_preds: [batch_size, num_quantiles, num_token, output_patch_size]

        return quantile_preds, hidden_states


    def _predict_tensor(  # type: ignore[override]
        self,
        context: torch.Tensor,
        prediction_length: Optional[int] = None,
        max_context: Optional[int] = None,
        output_device: str = "cpu"
    ) -> torch.Tensor:
        
        predictions = []
        if prediction_length is None:
            prediction_length = self.tokenizer.patch_size
        remaining = -(prediction_length // -self.tokenizer.patch_size)
        if max_context is None:
            max_context = self.train_ctx_len
        min_context = max(self.train_ctx_len, max_context)

        context = context.to(
            device=self.device,
            dtype=torch.float32,
        )
        while remaining > 0:
            if context.shape[-1] > max_context:
                context = context[..., -max_context:]
            if context.shape[-1] < min_context:
                pad = torch.full((context.shape[0], min_context - context.shape[-1]),
                                  fill_value=torch.nan, device=context.device, dtype=context.dtype)
                context = torch.concat((pad, context), dim=1)
            tokenized_tensor, tokenizer_state = self.tokenizer.context_input_transform(context)
            with torch.no_grad():
                prediction, _ = self._forward_model_tokenized(
                    input_token=tokenized_tensor,
                    input_mask=torch.isnan(tokenized_tensor),
                )
                prediction = prediction[:, :, -1, :].to(tokenized_tensor) # predicted token
            prediction = self.tokenizer.output_transform(prediction, tokenizer_state)

            predictions.append(prediction)
            remaining -= 1

            if remaining <= 0:
                break

            context = torch.cat([context, torch.full_like(prediction[:, 0, :], fill_value=torch.nan)], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length].to(
            dtype=torch.float32, device=output_device
        )
