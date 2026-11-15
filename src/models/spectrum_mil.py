"""Gated-attention MIL for channel-specific raw-spectrum cell instances."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpectrumAttentionMIL(nn.Module):
    """Encode spectrum, reporter/precursor, and optional PSM features per instance."""

    def __init__(
        self,
        spectrum_feature_dim: int,
        numeric_feature_dim: int,
        hidden_dim: int,
        spectrum_projection_dim: int = 32,
        peptide_vocab_size: int = 0,
        peptide_embedding_dim: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if numeric_feature_dim < 1 or hidden_dim < 1:
            raise ValueError("numeric_feature_dim and hidden_dim must be positive")
        if spectrum_feature_dim < 0 or peptide_vocab_size < 0:
            raise ValueError("optional feature dimensions cannot be negative")
        self.uses_spectrum = spectrum_feature_dim > 0
        self.uses_peptide = peptide_vocab_size > 0
        if self.uses_spectrum:
            if spectrum_projection_dim < 1:
                raise ValueError("spectrum projection dimension must be positive")
            self.spectrum_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(spectrum_feature_dim, spectrum_projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.spectrum_encoder = None
            spectrum_projection_dim = 0
        if self.uses_peptide:
            if peptide_embedding_dim < 1:
                raise ValueError("peptide embedding dimension must be positive")
            self.peptide_embedding: nn.Module | None = nn.Embedding(
                peptide_vocab_size, peptide_embedding_dim, padding_idx=0
            )
        else:
            self.peptide_embedding = None
            peptide_embedding_dim = 0
        combined_dim = numeric_feature_dim + spectrum_projection_dim + peptide_embedding_dim
        self.instance_encoder = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_v = nn.Linear(hidden_dim, hidden_dim)
        self.attention_u = nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        spectrum_features: Tensor,
        numeric_features: Tensor,
        peptide_ids: Tensor,
        bag_index: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return one logit per bag and optionally normalized instance weights."""
        instance_count = numeric_features.shape[0]
        if numeric_features.ndim != 2 or instance_count == 0:
            raise ValueError("numeric features must contain at least one instance")
        if bag_index.ndim != 1 or len(bag_index) != instance_count:
            raise ValueError("bag_index must have one value per instance")
        if peptide_ids.ndim != 1 or len(peptide_ids) != instance_count:
            raise ValueError("peptide_ids must have one value per instance")
        if spectrum_features.ndim != 2 or len(spectrum_features) != instance_count:
            raise ValueError("spectrum features must have one row per instance")
        if bag_index.min().item() < 0:
            raise ValueError("bag_index must be non-negative")
        features = [numeric_features]
        if self.spectrum_encoder is not None:
            features.append(self.spectrum_encoder(spectrum_features))
        elif spectrum_features.shape[1] != 0:
            raise ValueError("model was configured without spectrum features")
        if self.peptide_embedding is not None:
            features.append(self.peptide_embedding(peptide_ids))
        encoded = self.instance_encoder(torch.cat(features, dim=1))
        raw_attention = self.attention_w(
            torch.tanh(self.attention_v(encoded)) * torch.sigmoid(self.attention_u(encoded))
        ).squeeze(1)
        bag_count = int(bag_index.max().item()) + 1
        pooled = []
        attention = torch.empty_like(raw_attention) if return_attention else None
        for bag_id in range(bag_count):
            mask = bag_index == bag_id
            if not torch.any(mask):
                raise ValueError("bag_index must be contiguous with no empty bags")
            weights = torch.softmax(raw_attention[mask], dim=0)
            if attention is not None:
                attention[mask] = weights
            pooled.append(torch.sum(weights[:, None] * encoded[mask], dim=0))
        logits = self.classifier(torch.stack(pooled)).squeeze(1)
        return (logits, attention) if attention is not None else logits
