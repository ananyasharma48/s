"""Protein-plus-spectrum multiple-instance learning models."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpectrumBagEncoder(nn.Module):
    """Encode unordered spectrum instances and pool them with gated attention."""

    def __init__(
        self,
        spectrum_feature_dim: int,
        numeric_feature_dim: int,
        hidden_dim: int,
        spectrum_projection_dim: int,
        peptide_vocab_size: int,
        peptide_embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spectrum_encoder = nn.Sequential(
            nn.Linear(spectrum_feature_dim, spectrum_projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.peptide_embedding = nn.Embedding(
            peptide_vocab_size, peptide_embedding_dim, padding_idx=0
        )
        combined = spectrum_projection_dim + numeric_feature_dim + peptide_embedding_dim
        self.instance_encoder = nn.Sequential(
            nn.Linear(combined, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.attention_v = nn.Linear(hidden_dim, hidden_dim)
        self.attention_u = nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)
        self.output_dim = hidden_dim

    def forward(
        self,
        spectrum_features: Tensor,
        numeric_features: Tensor,
        peptide_ids: Tensor,
        bag_index: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if len(numeric_features) == 0 or len(spectrum_features) != len(numeric_features):
            raise ValueError("spectrum bags must contain aligned nonempty instances")
        if len(peptide_ids) != len(numeric_features) or len(bag_index) != len(numeric_features):
            raise ValueError("instance arrays and bag indices must align")
        encoded = self.instance_encoder(torch.cat((
            self.spectrum_encoder(spectrum_features),
            numeric_features,
            self.peptide_embedding(peptide_ids),
        ), dim=1))
        raw_attention = self.attention_w(
            torch.tanh(self.attention_v(encoded)) * torch.sigmoid(self.attention_u(encoded))
        ).squeeze(1)
        bag_count = int(bag_index.max().item()) + 1
        pooled = []
        attention = torch.empty_like(raw_attention) if return_attention else None
        for bag_id in range(bag_count):
            selected = bag_index == bag_id
            if not torch.any(selected):
                raise ValueError("bag indices must be contiguous with no empty bag")
            weights = torch.softmax(raw_attention[selected], dim=0)
            if attention is not None:
                attention[selected] = weights
            pooled.append(torch.sum(weights[:, None] * encoded[selected], dim=0))
        result = torch.stack(pooled)
        return (result, attention) if attention is not None else result


class JointMultimodalMIL(nn.Module):
    """Concatenate learned protein and spectrum-bag embeddings."""

    def __init__(
        self, protein_dim: int, protein_hidden_dim: int,
        spectrum_encoder: SpectrumBagEncoder, dropout: float,
    ) -> None:
        super().__init__()
        self.spectrum_encoder = spectrum_encoder
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, protein_hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        combined = protein_hidden_dim + spectrum_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined, protein_hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(protein_hidden_dim, 1),
        )

    def forward(
        self, protein_features: Tensor, spectrum_features: Tensor,
        numeric_features: Tensor, peptide_ids: Tensor, bag_index: Tensor,
    ) -> Tensor:
        spectrum = self.spectrum_encoder(
            spectrum_features, numeric_features, peptide_ids, bag_index
        )
        protein = self.protein_encoder(protein_features)
        return self.classifier(torch.cat((protein, spectrum), dim=1)).squeeze(1)


class ResidualMultimodalMIL(nn.Module):
    """Add a gated spectrum residual to a frozen protein-logistic prediction."""

    def __init__(
        self, protein_dim: int, spectrum_encoder: SpectrumBagEncoder,
        protein_coefficient: Tensor, protein_intercept: Tensor, dropout: float,
    ) -> None:
        super().__init__()
        self.spectrum_encoder = spectrum_encoder
        self.protein_logit = nn.Linear(protein_dim, 1)
        with torch.no_grad():
            self.protein_logit.weight.copy_(protein_coefficient.reshape(1, -1))
            self.protein_logit.bias.copy_(protein_intercept.reshape(1))
        for parameter in self.protein_logit.parameters():
            parameter.requires_grad = False
        self.spectrum_delta = nn.Sequential(
            nn.Linear(spectrum_encoder.output_dim, spectrum_encoder.output_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(spectrum_encoder.output_dim, 1),
        )
        nn.init.zeros_(self.spectrum_delta[-1].weight)
        nn.init.zeros_(self.spectrum_delta[-1].bias)
        self.gate_logit = nn.Parameter(torch.tensor(-3.0))

    @property
    def gate_value(self) -> Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(
        self, protein_features: Tensor, spectrum_features: Tensor,
        numeric_features: Tensor, peptide_ids: Tensor, bag_index: Tensor,
    ) -> Tensor:
        spectrum = self.spectrum_encoder(
            spectrum_features, numeric_features, peptide_ids, bag_index
        )
        baseline = self.protein_logit(protein_features).squeeze(1)
        residual = self.spectrum_delta(spectrum).squeeze(1)
        return baseline + self.gate_value * residual
