"""A compact gated-attention MIL classifier for variable-size peptide bags."""

import torch
from torch import Tensor, nn


class AttentionMIL(nn.Module):
    """Permutation-invariant binary bag classifier with gated attention pooling."""

    def __init__(
        self,
        peptide_vocab_size: int,
        peptide_embedding_dim: int,
        numeric_feature_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if min(peptide_vocab_size, peptide_embedding_dim, numeric_feature_dim, hidden_dim) < 1:
            raise ValueError("all model dimensions must be positive")
        self.peptide_embedding = nn.Embedding(peptide_vocab_size, peptide_embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(peptide_embedding_dim + numeric_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_v = nn.Linear(hidden_dim, hidden_dim)
        self.attention_u = nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        peptide_ids: Tensor,
        numeric_features: Tensor,
        bag_index: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return one logit per bag and optionally one normalized weight per instance."""
        if peptide_ids.ndim != 1 or bag_index.ndim != 1:
            raise ValueError("peptide_ids and bag_index must be one-dimensional")
        if numeric_features.ndim != 2 or numeric_features.shape[0] != peptide_ids.shape[0]:
            raise ValueError("numeric_features must have one row per instance")
        if bag_index.shape[0] != peptide_ids.shape[0] or bag_index.numel() == 0:
            raise ValueError("bag_index must contain one non-empty assignment per instance")
        if bag_index.min().item() < 0:
            raise ValueError("bag_index must be non-negative")

        encoded = self.encoder(torch.cat((self.peptide_embedding(peptide_ids), numeric_features), dim=1))
        raw_attention = self.attention_w(torch.tanh(self.attention_v(encoded)) * torch.sigmoid(self.attention_u(encoded))).squeeze(1)
        bag_count = int(bag_index.max().item()) + 1
        bag_embeddings = []
        attention = torch.empty_like(raw_attention) if return_attention else None
        for bag_id in range(bag_count):
            mask = bag_index == bag_id
            if not torch.any(mask):
                raise ValueError("bag_index must be contiguous with no empty bags")
            weights = torch.softmax(raw_attention[mask], dim=0)
            if attention is not None:
                attention[mask] = weights
            bag_embeddings.append(torch.sum(weights.unsqueeze(1) * encoded[mask], dim=0))
        logits = self.classifier(torch.stack(bag_embeddings)).squeeze(1)
        return (logits, attention) if attention is not None else logits
