"""Lazy HDF5 dataset and training-only normalization for spectrum MIL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


BASE_NUMERIC_DIM = 4
PSM_NUMERIC_DIM = 4


@dataclass(frozen=True)
class SpectrumArrayStore:
    """One process-wide memory cache derived from the aligned Stage 11 HDF5 rows."""

    spectrum_features: np.ndarray
    precursor_features: np.ndarray
    reporter_intensities: np.ndarray

    @classmethod
    def from_hdf5(
        cls,
        spectra_path: str | Path,
        reporter_path: str | Path,
        downsample_factor: int,
        chunk_size: int = 4096,
    ) -> "SpectrumArrayStore":
        if downsample_factor < 1:
            raise ValueError("downsample factor must be positive")
        with h5py.File(spectra_path, "r") as spectra:
            source = spectra["spectrum_embedding"]
            if source.shape[1] % downsample_factor != 0:
                raise ValueError("spectrum dimension is not divisible by downsample factor")
            output_dim = source.shape[1] // downsample_factor
            compressed = np.empty((source.shape[0], output_dim), dtype=np.float32)
            for start in range(0, source.shape[0], chunk_size):
                values = np.asarray(source[start:start + chunk_size], dtype=np.float32)
                if downsample_factor > 1:
                    values = values.reshape(
                        len(values), output_dim, downsample_factor
                    ).sum(axis=2)
                norms = np.linalg.norm(values, axis=1, keepdims=True)
                compressed[start:start + len(values)] = values / np.maximum(norms, 1e-12)
            precursor = np.asarray(spectra["precursor_features"][:, :3], dtype=np.float32)
        with h5py.File(reporter_path, "r") as reporters:
            reporter_values = np.asarray(
                reporters["reporter_intensities"][:], dtype=np.float32
            )
        return cls(compressed, precursor, reporter_values)


@dataclass(frozen=True)
class VectorStandardizer:
    """Per-column training-partition standardization."""

    mean: np.ndarray
    standard_deviation: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return (values - self.mean) / self.standard_deviation

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "standard_deviation": self.standard_deviation.astype(float).tolist(),
        }


def make_psm_vocabulary(metadata: pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame]:
    """Assign deterministic PSM peptide IDs, reserving zero for unidentified spectra."""
    peptides = sorted(metadata["psm_sequence"].dropna().astype(str).unique())
    vocabulary = pd.DataFrame({
        "peptide_id": np.arange(1, len(peptides) + 1, dtype=np.int64),
        "peptide": peptides,
    })
    return dict(zip(vocabulary["peptide"], vocabulary["peptide_id"])), vocabulary


def psm_numeric_features(metadata: pd.DataFrame) -> np.ndarray:
    """Encode confidence while retaining an explicit identification indicator."""
    q_value = pd.to_numeric(metadata["psm_q_value"], errors="coerce").to_numpy(float)
    pep = pd.to_numeric(metadata["psm_pep"], errors="coerce").to_numpy(float)
    pif = pd.to_numeric(metadata["psm_precursor_ion_fraction"], errors="coerce").to_numpy(float)
    has_psm = metadata["has_psm"].fillna(False).to_numpy(dtype=float)
    q_feature = np.zeros(len(metadata), dtype=np.float32)
    pep_feature = np.zeros(len(metadata), dtype=np.float32)
    q_mask = np.isfinite(q_value)
    pep_mask = np.isfinite(pep)
    q_feature[q_mask] = np.clip(-np.log10(np.clip(q_value[q_mask], 1e-12, 1.0)), 0, 12)
    pep_feature[pep_mask] = np.clip(-np.log10(np.clip(pep[pep_mask], 1e-12, 1.0)), 0, 12)
    return np.column_stack((
        q_feature,
        pep_feature,
        np.nan_to_num(pif, nan=0.0, posinf=0.0, neginf=0.0),
        has_psm,
    )).astype(np.float32)


def validate_spectrum_bags(
    bags: pd.DataFrame,
    metadata: pd.DataFrame,
    spectra_path: str | Path,
    reporter_path: str | Path,
) -> None:
    """Validate Stage 11 row alignment and cell-bag references before training."""
    required_bags = {
        "sample_id", "label_raw", "label_binary", "raw_file_stem",
        "reporter_channel", "spectrum_count", "spectrum_ids",
    }
    if missing := required_bags - set(bags.columns):
        raise ValueError(f"spectrum bag index is missing columns: {sorted(missing)}")
    if bags.empty or metadata.empty or bags["sample_id"].duplicated().any():
        raise ValueError("spectrum bags and metadata must be nonempty and unique")
    if not np.array_equal(metadata["spectrum_id"], np.arange(len(metadata))):
        raise ValueError("spectrum metadata IDs must be contiguous and row-aligned")
    with h5py.File(spectra_path, "r") as spectra, h5py.File(reporter_path, "r") as reporters:
        if spectra["spectrum_embedding"].shape[0] != len(metadata):
            raise ValueError("spectrum HDF5 rows differ from metadata")
        if reporters["reporter_intensities"].shape != (len(metadata), 16):
            raise ValueError("reporter HDF5 shape differs from Stage 11 contract")
    for bag in bags.itertuples(index=False):
        ids = np.asarray(bag.spectrum_ids, dtype=np.int64)
        if len(ids) != int(bag.spectrum_count) or len(ids) == 0:
            raise ValueError(f"invalid spectrum count for {bag.sample_id}")
        if np.any(np.diff(ids) <= 0) or ids.min() < 0 or ids.max() >= len(metadata):
            raise ValueError(f"invalid spectrum references for {bag.sample_id}")
        run_names = metadata.iloc[ids]["raw_file_stem"].unique()
        if len(run_names) != 1 or run_names[0] != bag.raw_file_stem:
            raise ValueError(f"bag crosses raw files: {bag.sample_id}")


class SpectrumBagDataset(Dataset):
    """Read channel-specific spectrum bags lazily from aligned Stage 11 stores."""

    def __init__(
        self,
        bags: pd.DataFrame,
        metadata: pd.DataFrame,
        sample_ids: list[str],
        spectra_path: str | Path,
        reporter_path: str | Path,
        standardizer: VectorStandardizer,
        peptide_to_id: dict[str, int],
        use_spectrum: bool,
        use_psm: bool,
        spectrum_downsample_factor: int,
        maximum_instances_per_bag: int | None,
        seed: int,
        array_store: SpectrumArrayStore | None = None,
    ) -> None:
        if spectrum_downsample_factor < 1:
            raise ValueError("spectrum downsample factor must be positive")
        if maximum_instances_per_bag is not None and maximum_instances_per_bag < 1:
            raise ValueError("maximum instances must be positive or None")
        indexed = bags.set_index("sample_id", drop=False)
        missing = sorted(set(sample_ids) - set(indexed.index))
        if missing:
            raise ValueError(f"samples are absent from spectrum bags: {missing[:5]}")
        self.records = indexed.loc[sample_ids].reset_index(drop=True)
        self.metadata = metadata
        self.spectra_path = str(spectra_path)
        self.reporter_path = str(reporter_path)
        self.standardizer = standardizer
        self.use_spectrum = bool(use_spectrum)
        self.use_psm = bool(use_psm)
        self.downsample = int(spectrum_downsample_factor)
        self.maximum_instances_per_bag = maximum_instances_per_bag
        self.seed = int(seed)
        self.epoch = 0
        self.array_store = array_store
        self._spectra: h5py.File | None = None
        self._reporters: h5py.File | None = None
        peptide_values = metadata["psm_sequence"].fillna("").astype(str)
        self.peptide_ids = peptide_values.map(peptide_to_id).fillna(0).to_numpy(dtype=np.int64)
        self.psm_numeric = psm_numeric_features(metadata)

    def _open(self) -> None:
        if self.array_store is None and self._spectra is None:
            self._spectra = h5py.File(self.spectra_path, "r")
            self._reporters = h5py.File(self.reporter_path, "r")

    def close(self) -> None:
        if self._spectra is not None:
            self._spectra.close()
            self._reporters.close()
            self._spectra = None
            self._reporters = None

    def __del__(self) -> None:
        self.close()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        self._open()
        bag = self.records.iloc[index]
        spectrum_ids = np.asarray(bag["spectrum_ids"], dtype=np.int64)
        limit = self.maximum_instances_per_bag
        if limit is not None and len(spectrum_ids) > limit:
            bag_seed = self.seed + self.epoch * 1_000_003 + int(bag["bag_id"]) * 10_007
            spectrum_ids = np.random.default_rng(bag_seed).choice(
                spectrum_ids, size=limit, replace=False
            )
            spectrum_ids.sort()
        channel_index = int(str(bag["reporter_channel"]).removeprefix("RI")) - 1
        if self.array_store is not None:
            reporter_intensity = self.array_store.reporter_intensities[
                spectrum_ids, channel_index
            ]
            precursor = self.array_store.precursor_features[spectrum_ids]
        else:
            reporter_intensity = np.asarray(
                self._reporters["reporter_intensities"][spectrum_ids, channel_index],
                dtype=np.float32,
            )
            precursor = np.asarray(
                self._spectra["precursor_features"][spectrum_ids, :3], dtype=np.float32
            )
        numeric = np.column_stack((np.log1p(reporter_intensity), precursor))
        if self.use_psm:
            numeric = np.column_stack((numeric, self.psm_numeric[spectrum_ids]))
        numeric = self.standardizer.transform(numeric).astype(np.float32)
        if self.use_spectrum:
            if self.array_store is not None:
                spectrum = self.array_store.spectrum_features[spectrum_ids]
            else:
                spectrum = np.asarray(
                    self._spectra["spectrum_embedding"][spectrum_ids], dtype=np.float32
                )
                if spectrum.shape[1] % self.downsample != 0:
                    raise ValueError("spectrum dimension is not divisible by downsample factor")
                if self.downsample > 1:
                    spectrum = spectrum.reshape(
                        len(spectrum), spectrum.shape[1] // self.downsample, self.downsample
                    ).sum(axis=2)
                    norm = np.linalg.norm(spectrum, axis=1, keepdims=True)
                    spectrum = spectrum / np.maximum(norm, 1e-12)
        else:
            spectrum = np.empty((len(spectrum_ids), 0), dtype=np.float32)
        peptide_ids = self.peptide_ids[spectrum_ids] if self.use_psm else np.zeros(
            len(spectrum_ids), dtype=np.int64
        )
        label_value = bag["label_binary"]
        if isinstance(label_value, str):
            label_value = int(label_value == "sc_u")
        return {
            "sample_id": str(bag["sample_id"]),
            "spectrum_ids": torch.from_numpy(spectrum_ids.copy()),
            "spectrum_features": torch.from_numpy(spectrum.copy()),
            "numeric_features": torch.from_numpy(numeric.copy()),
            "peptide_ids": torch.from_numpy(peptide_ids.copy()),
            "reporter_intensity": torch.from_numpy(reporter_intensity.copy()),
            "label": torch.tensor(float(label_value), dtype=torch.float32),
        }


def fit_numeric_standardizer(
    bags: pd.DataFrame,
    metadata: pd.DataFrame,
    sample_ids: list[str],
    spectra_path: str | Path,
    reporter_path: str | Path,
    use_psm: bool,
    array_store: SpectrumArrayStore | None = None,
) -> VectorStandardizer:
    """Fit instance numeric normalization on selected training cells only."""
    indexed = bags.set_index("sample_id")
    missing = sorted(set(sample_ids) - set(indexed.index))
    if missing or not sample_ids:
        raise ValueError(f"invalid standardizer sample IDs: {missing[:5]}")
    psm_features = psm_numeric_features(metadata)
    sums = np.zeros(BASE_NUMERIC_DIM + (PSM_NUMERIC_DIM if use_psm else 0), dtype=np.float64)
    squared_sums = np.zeros_like(sums)
    count = 0
    spectra_handle = None
    reporter_handle = None
    if array_store is None:
        spectra_handle = h5py.File(spectra_path, "r")
        reporter_handle = h5py.File(reporter_path, "r")
    try:
        for sample_id in sample_ids:
            bag = indexed.loc[sample_id]
            ids = np.asarray(bag["spectrum_ids"], dtype=np.int64)
            channel = int(str(bag["reporter_channel"]).removeprefix("RI")) - 1
            if array_store is not None:
                reporter = array_store.reporter_intensities[ids, channel]
                precursor = array_store.precursor_features[ids]
            else:
                reporter = np.asarray(
                    reporter_handle["reporter_intensities"][ids, channel]
                )
                precursor = np.asarray(spectra_handle["precursor_features"][ids, :3])
            values = np.column_stack((np.log1p(reporter), precursor))
            if use_psm:
                values = np.column_stack((values, psm_features[ids]))
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            sums += values.sum(axis=0)
            squared_sums += np.square(values).sum(axis=0)
            count += len(values)
    finally:
        if spectra_handle is not None:
            spectra_handle.close()
            reporter_handle.close()
    mean = sums / count
    variance = np.maximum(squared_sums / count - np.square(mean), 0.0)
    standard_deviation = np.sqrt(variance)
    standard_deviation[standard_deviation < 1e-8] = 1.0
    return VectorStandardizer(mean.astype(np.float32), standard_deviation.astype(np.float32))


def collate_spectrum_bags(batch: list[dict[str, object]]) -> dict[str, object]:
    """Concatenate variable-length spectrum bags and retain bag assignments."""
    if not batch:
        raise ValueError("cannot collate an empty spectrum batch")
    bag_indices = []
    for bag_id, item in enumerate(batch):
        count = int(item["spectrum_ids"].shape[0])
        if count < 1:
            raise ValueError("every spectrum bag must be nonempty")
        bag_indices.append(torch.full((count,), bag_id, dtype=torch.long))
    return {
        "sample_ids": [str(item["sample_id"]) for item in batch],
        "spectrum_ids": torch.cat([item["spectrum_ids"] for item in batch]),
        "spectrum_features": torch.cat([item["spectrum_features"] for item in batch]),
        "numeric_features": torch.cat([item["numeric_features"] for item in batch]),
        "peptide_ids": torch.cat([item["peptide_ids"] for item in batch]),
        "reporter_intensity": torch.cat([item["reporter_intensity"] for item in batch]),
        "bag_index": torch.cat(bag_indices),
        "labels": torch.stack([item["label"] for item in batch]),
    }
