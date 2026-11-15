"""Datasets and training-only intensity normalization for peptide MIL."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class IntensityStandardizer:
    """Mean and standard deviation fitted only on selected training bags."""

    mean: float
    standard_deviation: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.standard_deviation

    def as_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "standard_deviation": self.standard_deviation}


def validate_mil_tables(
    instances: pd.DataFrame,
    bags: pd.DataFrame,
    vocabulary: pd.DataFrame,
) -> None:
    """Validate the storage invariants required by the MIL loader."""
    instance_columns = {"instance_id", "peptide_id", "log2_intensity"}
    bag_columns = {
        "bag_id", "sample_id", "label_raw", "label_binary", "start_offset",
        "end_offset", "bag_size", "raw_file_stem",
    }
    vocabulary_columns = {"peptide_id", "peptide"}
    if missing := instance_columns - set(instances.columns):
        raise ValueError(f"instance table is missing columns: {sorted(missing)}")
    if missing := bag_columns - set(bags.columns):
        raise ValueError(f"bag index is missing columns: {sorted(missing)}")
    if missing := vocabulary_columns - set(vocabulary.columns):
        raise ValueError(f"vocabulary is missing columns: {sorted(missing)}")
    if bags.empty or instances.empty or vocabulary.empty:
        raise ValueError("MIL tables must be nonempty")
    if bags["sample_id"].duplicated().any():
        raise ValueError("bag sample IDs must be unique")
    if not np.array_equal(bags["bag_id"].to_numpy(), np.arange(len(bags))):
        raise ValueError("bag IDs must be contiguous and zero-based")
    if int(bags.iloc[0]["start_offset"]) != 0 or int(bags.iloc[-1]["end_offset"]) != len(instances):
        raise ValueError("bag offsets do not cover the complete instance table")
    if not np.array_equal(
        bags["start_offset"].iloc[1:].to_numpy(),
        bags["end_offset"].iloc[:-1].to_numpy(),
    ):
        raise ValueError("bag offsets are not contiguous")
    if not np.array_equal(
        (bags["end_offset"] - bags["start_offset"]).to_numpy(),
        bags["bag_size"].to_numpy(),
    ) or (bags["bag_size"] < 1).any():
        raise ValueError("bag sizes do not match offsets or include an empty bag")
    if not np.array_equal(instances["instance_id"].to_numpy(), np.arange(len(instances))):
        raise ValueError("instance IDs must be contiguous and zero-based")
    valid_ids = set(vocabulary["peptide_id"].astype(int))
    if not set(instances["peptide_id"].astype(int)).issubset(valid_ids):
        raise ValueError("an instance references an unknown peptide ID")
    log_intensity = instances["log2_intensity"].to_numpy(dtype=np.float64)
    if not np.isfinite(log_intensity).all():
        raise ValueError("log2 intensities must be finite")


def fit_intensity_standardizer(
    instances: pd.DataFrame,
    bags: pd.DataFrame,
    sample_ids: list[str],
) -> IntensityStandardizer:
    """Fit log-intensity normalization using instances from specified bags only."""
    if not sample_ids:
        raise ValueError("at least one training sample is required")
    indexed = bags.set_index("sample_id")
    missing = sorted(set(sample_ids) - set(indexed.index))
    if missing:
        raise ValueError(f"training samples are absent from bag index: {missing[:5]}")
    chunks = []
    values = instances["log2_intensity"].to_numpy(dtype=np.float64)
    for sample_id in sample_ids:
        bag = indexed.loc[sample_id]
        chunks.append(values[int(bag["start_offset"]):int(bag["end_offset"])])
    selected = np.concatenate(chunks)
    mean = float(selected.mean())
    standard_deviation = float(selected.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("training intensity standard deviation must be positive and finite")
    return IntensityStandardizer(mean, standard_deviation)


class PeptideBagDataset(Dataset):
    """Read contiguous peptide bags and optionally subsample training instances."""

    def __init__(
        self,
        instances: pd.DataFrame,
        bags: pd.DataFrame,
        sample_ids: list[str],
        standardizer: IntensityStandardizer,
        maximum_instances_per_bag: int | None = None,
        seed: int = 0,
    ) -> None:
        if maximum_instances_per_bag is not None and maximum_instances_per_bag < 1:
            raise ValueError("maximum_instances_per_bag must be positive or None")
        indexed = bags.set_index("sample_id", drop=False)
        missing = sorted(set(sample_ids) - set(indexed.index))
        if missing:
            raise ValueError(f"samples are absent from bag index: {missing[:5]}")
        self.records = indexed.loc[sample_ids].reset_index(drop=True)
        self.instance_ids = instances["instance_id"].to_numpy(dtype=np.int64)
        self.peptide_ids = instances["peptide_id"].to_numpy(dtype=np.int64)
        self.log2_intensity = instances["log2_intensity"].to_numpy(dtype=np.float32)
        self.standardizer = standardizer
        self.maximum_instances_per_bag = maximum_instances_per_bag
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        bag = self.records.iloc[index]
        start, end = int(bag["start_offset"]), int(bag["end_offset"])
        positions = np.arange(start, end, dtype=np.int64)
        limit = self.maximum_instances_per_bag
        if limit is not None and len(positions) > limit:
            bag_seed = self.seed + self.epoch * 1_000_003 + int(bag["bag_id"]) * 10_007
            positions = np.random.default_rng(bag_seed).choice(positions, size=limit, replace=False)
            positions.sort()
        numeric = self.standardizer.transform(
            self.log2_intensity[positions].astype(np.float64)
        ).astype(np.float32)
        return {
            "sample_id": str(bag["sample_id"]),
            "instance_ids": torch.from_numpy(self.instance_ids[positions].copy()),
            "peptide_ids": torch.from_numpy(self.peptide_ids[positions].copy()),
            "numeric_features": torch.from_numpy(numeric[:, None]),
            "label": torch.tensor(float(bag["label_binary"]), dtype=torch.float32),
        }
