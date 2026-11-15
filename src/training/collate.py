"""Collation for batches containing variable-length peptide bags."""

from __future__ import annotations

import torch


def collate_peptide_bags(batch: list[dict[str, object]]) -> dict[str, object]:
    """Concatenate instances while preserving their bag assignments."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    bag_indices = []
    for bag_id, item in enumerate(batch):
        count = int(item["peptide_ids"].shape[0])
        if count < 1:
            raise ValueError("every collated bag must contain at least one instance")
        bag_indices.append(torch.full((count,), bag_id, dtype=torch.long))
    return {
        "sample_ids": [str(item["sample_id"]) for item in batch],
        "instance_ids": torch.cat([item["instance_ids"] for item in batch]),
        "peptide_ids": torch.cat([item["peptide_ids"] for item in batch]),
        "numeric_features": torch.cat([item["numeric_features"] for item in batch]),
        "bag_index": torch.cat(bag_indices),
        "labels": torch.stack([item["label"] for item in batch]),
    }
