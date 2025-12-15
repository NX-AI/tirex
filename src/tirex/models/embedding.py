# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
import torch.nn as nn
import torch.nn.functional as F

from tirex import load_model
from tirex.util import nanmax, nanmin, nanstd


class TiRexEmbedding(nn.Module):
    def __init__(
        self, device: str | None = None, data_augmentation: bool = False, batch_size: int = 512, compile: bool = False
    ) -> None:
        super().__init__()
        self.data_augmentation = data_augmentation
        self.number_of_patches = 8
        self.batch_size = batch_size

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._compile = compile

        self.model = load_model(path="NX-AI/TiRex", device=self.device, compile=self._compile)

    def _gen_emb_batched(self, data: torch.Tensor) -> torch.Tensor:
        batches = list(torch.split(data, self.batch_size))
        embedding_list = []
        for batch in batches:
            embedding = self.model._embed_context(batch)
            embedding_list.append(embedding.cpu())
        return torch.cat(embedding_list, dim=0)

    def _calculate_n_patches(self, data: torch.Tensor) -> int:
        _, _, n_steps = data.shape
        n_patches = -(-n_steps // self.model.config.input_patch_size)
        return n_patches

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        n_patches = self._calculate_n_patches(data)

        embedding = torch.stack(
            [self._gen_emb_batched(var_slice) for var_slice in torch.unbind(data, dim=1)], dim=1
        )  # Stack in case of multivar
        embedding = self.process_embedding(embedding, n_patches)

        if self.data_augmentation:
            # Difference Embedding
            diff_data = torch.diff(data, dim=-1, prepend=data[..., :0])
            n_patches = self._calculate_n_patches(diff_data)

            diff_embedding = torch.stack(
                [self._gen_emb_batched(var_slice) for var_slice in torch.unbind(diff_data, dim=1)], dim=1
            )
            diff_embedding = self.process_embedding(diff_embedding, n_patches)
            embedding = torch.cat((diff_embedding, embedding), dim=-1)

            # Stats Embedding
            stat_features = self._generate_stats_features(data)
            normalized_stats = self._normalize_stats(stat_features)
            normalized_stats = normalized_stats.to(embedding.device)

            # Concat all together
            embedding = torch.cat((embedding, normalized_stats), dim=-1)

        return embedding

    def process_embedding(self, embedding: torch.Tensor, n_patches: int) -> torch.Tensor:
        # embedding shape: (bs, var_dim, n_patches, n_layer, emb_dim)
        embedding = embedding[:, :, -n_patches:, :, :]
        embedding = torch.mean(embedding, dim=2)  # sequence
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        embedding = torch.transpose(embedding, 1, -2).flatten(start_dim=-2)  # var
        embedding = torch.transpose(embedding, 1, -2).flatten(start_dim=-2)  # layer
        embedding = F.layer_norm(embedding, (embedding.shape[-1],))
        return embedding

    def _normalize_stats(self, stat_features: torch.Tensor) -> torch.Tensor:
        dataset_mean = torch.nanmean(stat_features, dim=0, keepdim=True)
        dataset_std = nanstd(stat_features, dim=0, keepdim=True)
        stat_features = (stat_features - dataset_mean) / (dataset_std + 1e-8)
        stat_features = torch.nan_to_num(stat_features, nan=0.0)

        stat_features = (stat_features - stat_features.nanmean(dim=-1, keepdim=True)) / (
            stat_features.std(dim=-1, keepdim=True) + 1e-8
        )
        return stat_features

    def _generate_stats_features(self, data: torch.Tensor) -> torch.Tensor:
        bs, variates, n_steps = data.shape

        patch_size = max(1, n_steps // self.number_of_patches)
        n_full_patches = n_steps // patch_size
        n_remain = n_steps % patch_size

        # [batch, variates, n_patches, patch_size]
        patches = data[..., : n_full_patches * patch_size].unfold(-1, patch_size, patch_size)

        # Stats for full patches
        patch_means = torch.nanmean(patches, dim=-1)
        patch_stds = nanstd(patches, dim=-1)
        patch_maxes = nanmax(patches, dim=-1)
        patch_mins = nanmin(patches, dim=-1)

        stats = [patch_means, patch_stds, patch_maxes, patch_mins]

        # Handle last smaller patch if needed
        if n_remain > 0:
            self._handle_remaining_patch(data, stats, n_full_patches * patch_size)

        stats = torch.stack(stats, dim=-1)  # [batch, variates, n_patches(+1), 4]
        return stats.flatten(start_dim=1)  # [batch, variates * n_patches * 4]

    def _handle_remaining_patch(self, data: torch.Tensor, stats: list[torch.Tensor], full_patch_length: int) -> None:
        last_patch = data[..., full_patch_length:]

        mean_last = last_patch.mean(dim=-1, keepdim=True)
        std_last = last_patch.std(dim=-1, keepdim=True)
        max_last = last_patch.max(dim=-1, keepdim=True)
        min_last = last_patch.min(dim=-1, keepdim=True)

        stats[0] = torch.cat([stats[0], mean_last], dim=-1)
        stats[1] = torch.cat([stats[1], std_last], dim=-1)
        stats[2] = torch.cat([stats[2], max_last], dim=-1)
        stats[3] = torch.cat([stats[3], min_last], dim=-1)
