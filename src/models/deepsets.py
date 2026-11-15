"""Mean- and max-pooling DeepSets classifiers for peptide bags."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DeepSetsMIL(nn.Module):
    """Permutation-invariant peptide-bag classifier with mean or max pooling."""

    def __init__(
        self,
        peptide_vocab_size: int,
        peptide_embedding_dim: int,
        numeric_feature_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if min(peptide_vocab_size, peptide_embedding_dim, numeric_feature_dim, hidden_dim) < 1:
            raise ValueError("all model dimensions must be positive")
        if pooling not in {"mean", "max"}:
            raise ValueError("pooling must be 'mean' or 'max'")
        self.pooling = pooling
        self.peptide_embedding = nn.Embedding(peptide_vocab_size, peptide_embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(peptide_embedding_dim + numeric_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, peptide_ids: Tensor, numeric_features: Tensor, bag_index: Tensor) -> Tensor:
        if peptide_ids.ndim != 1 or bag_index.ndim != 1:
            raise ValueError("peptide_ids and bag_index must be one-dimensional")
        if numeric_features.ndim != 2 or numeric_features.shape[0] != peptide_ids.shape[0]:
            raise ValueError("numeric_features must have one row per instance")
        if bag_index.shape[0] != peptide_ids.shape[0] or bag_index.numel() == 0:
            raise ValueError("bag_index must contain one non-empty assignment per instance")
        if bag_index.min().item() < 0:
            raise ValueError("bag_index must be non-negative")
        encoded = self.encoder(torch.cat((self.peptide_embedding(peptide_ids), numeric_features), dim=1))
        bag_count = int(bag_index.max().item()) + 1
        pooled = []
        for bag_id in range(bag_count):
            selected = encoded[bag_index == bag_id]
            if selected.shape[0] == 0:
                raise ValueError("bag_index must be contiguous with no empty bags")
            pooled.append(selected.mean(dim=0) if self.pooling == "mean" else selected.max(dim=0).values)
        return self.classifier(torch.stack(pooled)).squeeze(1)
