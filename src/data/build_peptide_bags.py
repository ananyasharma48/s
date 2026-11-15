"""Build the Stage 7 variable-length peptide MIL dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import load_yaml_config
from src.utils.logging import configure_logging


MANIFEST_COLUMNS = [
    "sample_id",
    "label_raw",
    "raw_file_stem",
    "tmt_set",
    "reporter_channel",
    "batch_digest",
    "batch_sort",
    "batch_chromatography",
]


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_inputs(
    peptide_matrix: pd.DataFrame,
    manifest: pd.DataFrame,
    positive_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and align the peptide matrix to valid manifest samples."""
    required = {"is_valid_target", *MANIFEST_COLUMNS}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    valid = manifest.loc[manifest["is_valid_target"], MANIFEST_COLUMNS].copy()
    if valid.empty:
        raise ValueError("manifest has no valid target samples")
    if valid["sample_id"].duplicated().any():
        duplicates = valid.loc[valid["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"valid manifest sample IDs are duplicated: {duplicates[:5]}")
    labels = set(valid["label_raw"].dropna().astype(str))
    if positive_label not in labels or len(labels) != 2:
        raise ValueError(f"expected two labels including {positive_label}, found {sorted(labels)}")
    if peptide_matrix.empty:
        raise ValueError("peptide matrix is empty")
    if peptide_matrix.index.has_duplicates:
        raise ValueError("peptide matrix contains duplicate peptide identifiers")
    if peptide_matrix.columns.has_duplicates:
        raise ValueError("peptide matrix contains duplicate sample identifiers")
    valid_ids = valid["sample_id"].astype(str).tolist()
    matrix_ids = peptide_matrix.columns.astype(str).tolist()
    missing_samples = sorted(set(valid_ids) - set(matrix_ids))
    extra_samples = sorted(set(matrix_ids) - set(valid_ids))
    if missing_samples or extra_samples:
        raise ValueError(
            "peptide matrix and valid manifest samples differ: "
            f"missing={missing_samples[:5]}, extra={extra_samples[:5]}"
        )
    matrix = peptide_matrix.copy()
    matrix.columns = matrix.columns.astype(str)
    matrix.index = matrix.index.astype(str)
    converted = matrix.apply(pd.to_numeric, errors="coerce")
    failed_conversion = matrix.notna() & converted.isna()
    if failed_conversion.to_numpy().any():
        raise ValueError("peptide matrix contains nonnumeric observed values")
    observed = converted.to_numpy(dtype=float)
    observed = observed[~np.isnan(observed)]
    if observed.size == 0:
        raise ValueError("peptide matrix has no observed measurements")
    if not np.isfinite(observed).all():
        raise ValueError("peptide matrix contains non-finite observed measurements")
    if (observed <= 0).any():
        raise ValueError("observed peptide intensities must be positive; missing values must be NA")
    matrix = converted.reindex(columns=valid_ids)
    valid["sample_id"] = valid["sample_id"].astype(str)
    valid["label_raw"] = valid["label_raw"].astype(str)
    valid["label_binary"] = valid["label_raw"].eq(positive_label).astype("int8")
    return matrix, valid.reset_index(drop=True)


def make_peptide_vocabulary(peptide_matrix: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic IDs to peptides, reserving zero for padding/unknown."""
    peptides = sorted(peptide_matrix.index.astype(str).tolist())
    return pd.DataFrame(
        {
            "peptide_id": np.arange(1, len(peptides) + 1, dtype=np.int32),
            "peptide": peptides,
        }
    )


def encode_peptide_bags(
    peptide_matrix: pd.DataFrame,
    valid_manifest: pd.DataFrame,
    vocabulary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode observed peptide-cell measurements as contiguous variable-length bags."""
    peptide_to_id = vocabulary.set_index("peptide")["peptide_id"]
    instance_frames: list[pd.DataFrame] = []
    bag_records: list[dict[str, object]] = []
    offset = 0
    for bag_id, sample in enumerate(valid_manifest.itertuples(index=False)):
        observed = peptide_matrix[sample.sample_id].dropna()
        if observed.empty:
            raise ValueError(f"sample has zero detected peptides: {sample.sample_id}")
        peptide_ids = peptide_to_id.loc[observed.index].to_numpy(dtype=np.int32)
        intensities = observed.to_numpy(dtype=np.float64)
        order = np.argsort(peptide_ids)
        peptide_ids = peptide_ids[order]
        intensities = intensities[order]
        log2_intensities = np.log2(intensities + 1.0).astype(np.float32)
        bag_size = len(peptide_ids)
        instance_frames.append(
            pd.DataFrame(
                {
                    "instance_id": np.arange(offset, offset + bag_size, dtype=np.int64),
                    "peptide_id": peptide_ids,
                    "intensity": intensities,
                    "log2_intensity": log2_intensities,
                }
            )
        )
        record = {
            "bag_id": bag_id,
            "sample_id": sample.sample_id,
            "label_raw": sample.label_raw,
            "label_binary": sample.label_binary,
            "start_offset": offset,
            "end_offset": offset + bag_size,
            "bag_size": bag_size,
            "total_intensity": float(intensities.sum()),
            "median_intensity": float(np.median(intensities)),
            "mean_log2_intensity": float(log2_intensities.mean()),
        }
        for column in MANIFEST_COLUMNS[2:]:
            record[column] = getattr(sample, column)
        bag_records.append(record)
        offset += bag_size
    instances = pd.concat(instance_frames, ignore_index=True)
    bags = pd.DataFrame(bag_records)
    integer_columns = ["bag_id", "label_binary", "start_offset", "end_offset", "bag_size"]
    bags[integer_columns] = bags[integer_columns].astype(np.int64)
    return instances, bags


def reconstruct_bag(
    instances: pd.DataFrame,
    bag: pd.Series,
    vocabulary: pd.DataFrame,
) -> pd.Series:
    """Reconstruct one sparse peptide-intensity vector from stored offsets."""
    start, end = int(bag["start_offset"]), int(bag["end_offset"])
    selected = instances.iloc[start:end]
    peptide_by_id = vocabulary.set_index("peptide_id")["peptide"]
    peptides = peptide_by_id.loc[selected["peptide_id"]].to_numpy()
    return pd.Series(
        selected["intensity"].to_numpy(dtype=np.float64),
        index=pd.Index(peptides, name="peptide"),
        name=str(bag["sample_id"]),
    )


def validate_bag_dataset(
    peptide_matrix: pd.DataFrame,
    instances: pd.DataFrame,
    bags: pd.DataFrame,
    vocabulary: pd.DataFrame,
) -> None:
    """Verify offsets, values, identities, and exact source-matrix reconstruction."""
    if not np.array_equal(bags["bag_id"].to_numpy(), np.arange(len(bags))):
        raise AssertionError("bag IDs must be contiguous and zero-based")
    if int(bags.iloc[0]["start_offset"]) != 0:
        raise AssertionError("first bag must start at offset zero")
    if not np.array_equal(
        bags["start_offset"].iloc[1:].to_numpy(),
        bags["end_offset"].iloc[:-1].to_numpy(),
    ):
        raise AssertionError("bag offsets are not contiguous")
    if int(bags.iloc[-1]["end_offset"]) != len(instances):
        raise AssertionError("final bag offset does not equal the number of instances")
    if not np.array_equal(
        (bags["end_offset"] - bags["start_offset"]).to_numpy(),
        bags["bag_size"].to_numpy(),
    ):
        raise AssertionError("stored bag sizes do not match offsets")
    if (bags["bag_size"] < 1).any():
        raise AssertionError("every bag must contain at least one instance")
    if not np.array_equal(instances["instance_id"].to_numpy(), np.arange(len(instances))):
        raise AssertionError("instance IDs must be contiguous and zero-based")
    valid_peptide_ids = set(vocabulary["peptide_id"].astype(int))
    if not set(instances["peptide_id"].astype(int)).issubset(valid_peptide_ids):
        raise AssertionError("an instance references an unknown peptide ID")
    intensity = instances["intensity"].to_numpy(dtype=np.float64)
    if not np.isfinite(intensity).all() or (intensity <= 0).any():
        raise AssertionError("all stored intensities must be finite and positive")
    expected_log = np.log2(intensity + 1.0)
    if not np.allclose(instances["log2_intensity"], expected_log, rtol=1e-6, atol=1e-6):
        raise AssertionError("stored log2 intensities do not match raw intensities")
    for _, bag in bags.iterrows():
        start, end = int(bag["start_offset"]), int(bag["end_offset"])
        ids = instances.iloc[start:end]["peptide_id"].to_numpy()
        if len(np.unique(ids)) != len(ids) or np.any(np.diff(ids) <= 0):
            raise AssertionError(f"bag peptide IDs are duplicated or unsorted: {bag['sample_id']}")
        reconstructed = reconstruct_bag(instances, bag, vocabulary).sort_index()
        original = peptide_matrix[str(bag["sample_id"])].dropna().sort_index()
        if not reconstructed.index.equals(original.index):
            raise AssertionError(f"reconstructed peptide identities differ: {bag['sample_id']}")
        if not np.array_equal(
            reconstructed.to_numpy(dtype=np.float64),
            original.to_numpy(dtype=np.float64),
        ):
            raise AssertionError(f"reconstructed peptide intensities differ: {bag['sample_id']}")


def bag_summary_table(bags: pd.DataFrame) -> pd.DataFrame:
    """Summarize bag sizes and intensity distributions by label and batch."""
    grouped = bags.groupby(["label_raw", "batch_chromatography"], dropna=False)
    return grouped.agg(
        sample_count=("sample_id", "count"),
        bag_size_mean=("bag_size", "mean"),
        bag_size_median=("bag_size", "median"),
        bag_size_minimum=("bag_size", "min"),
        bag_size_maximum=("bag_size", "max"),
        total_intensity_mean=("total_intensity", "mean"),
        total_intensity_median=("total_intensity", "median"),
        cell_median_intensity_mean=("median_intensity", "mean"),
        cell_mean_log2_intensity_mean=("mean_log2_intensity", "mean"),
    ).reset_index()


def plot_bag_sizes(bags: pd.DataFrame, output_path: Path) -> None:
    """Plot bag-size distributions by label and chromatography batch."""
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for axis, column, title in (
        (axes[0], "label_raw", "Bag size by biological label"),
        (axes[1], "batch_chromatography", "Bag size by chromatography batch"),
    ):
        categories = sorted(bags[column].dropna().astype(str).unique())
        values = [bags.loc[bags[column].astype(str).eq(category), "bag_size"] for category in categories]
        axis.boxplot(values, tick_labels=categories, showfliers=True)
        axis.set_xlabel(column.replace("_", " ").title())
        axis.set_ylabel("Observed peptide instances per cell")
        axis.set_title(title)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_stage_seven(
    config_path: str | Path = "configs/mil.yaml",
    dry_run: bool = False,
) -> dict[str, object]:
    """Build, validate, and optionally write all Stage 7 artifacts."""
    config = load_yaml_config(config_path)
    data_config = config.get("data", {})
    matrix_path = Path(data_config.get(
        "peptide_matrix",
        "data/processed/fixed_features/peptide_by_sample.parquet",
    ))
    manifest_path = Path(data_config.get(
        "manifest",
        "data/interim/manifests/master_manifest.parquet",
    ))
    output_directory = Path(data_config.get("output_directory", "data/processed/mil_bags"))
    positive_label = str(config.get("positive_label", "sc_u"))
    peptide_matrix = pd.read_parquet(matrix_path)
    manifest = pd.read_parquet(manifest_path)
    peptide_matrix, valid_manifest = prepare_inputs(peptide_matrix, manifest, positive_label)
    vocabulary = make_peptide_vocabulary(peptide_matrix)
    instances, bags = encode_peptide_bags(peptide_matrix, valid_manifest, vocabulary)
    validate_bag_dataset(peptide_matrix, instances, bags, vocabulary)
    summary = {
        "samples": int(len(bags)),
        "peptides": int(len(vocabulary)),
        "instances": int(len(instances)),
        "minimum_bag_size": int(bags["bag_size"].min()),
        "median_bag_size": float(bags["bag_size"].median()),
        "mean_bag_size": float(bags["bag_size"].mean()),
        "maximum_bag_size": int(bags["bag_size"].max()),
        "positive_label": positive_label,
        "positive_samples": int(bags["label_binary"].sum()),
        "negative_samples": int((1 - bags["label_binary"]).sum()),
        "reserved_padding_unknown_id": 0,
        "intensity_transform": "log2(intensity + 1)",
    }
    if dry_run:
        return {**summary, "dry_run": True}
    table_directory = Path("outputs/tables")
    report_directory = Path("outputs/reports")
    figure_directory = Path("outputs/figures")
    for directory in (output_directory, table_directory, report_directory, figure_directory):
        directory.mkdir(parents=True, exist_ok=True)
    vocabulary.to_parquet(output_directory / "peptide_vocabulary.parquet", index=False)
    instances.to_parquet(output_directory / "peptide_instances.parquet", index=False)
    bags.to_parquet(output_directory / "bag_index.parquet", index=False)
    bags.to_csv(table_directory / "stage_07_bag_statistics.csv", index=False)
    grouped_summary = bag_summary_table(bags)
    grouped_summary.to_csv(table_directory / "stage_07_bag_summary_by_label_batch.csv", index=False)
    plot_bag_sizes(bags, figure_directory / "stage_07_bag_size_distribution.png")
    run_metadata = {
        **summary,
        "peptide_matrix": str(matrix_path),
        "peptide_matrix_sha256": file_sha256(matrix_path),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "vocabulary_path": str(output_directory / "peptide_vocabulary.parquet"),
        "instances_path": str(output_directory / "peptide_instances.parquet"),
        "bag_index_path": str(output_directory / "bag_index.parquet"),
    }
    (table_directory / "stage_07_run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )
    label_counts = bags["label_raw"].value_counts().sort_index()
    batch_counts = bags["batch_chromatography"].value_counts().sort_index()
    batch_median_bag_sizes = (
        bags.groupby("batch_chromatography")["bag_size"].median().sort_index()
    )
    (report_directory / "stage_07_peptide_mil_dataset.md").write_text(
        "\n".join(
            [
                "# Stage 7 — Peptide-level MIL dataset",
                "",
                "## Representation",
                "",
                "- One bag represents one valid single cell.",
                "- One instance represents one peptide detected in that cell.",
                "- Repeated evidence rows for the same peptide and cell were already aggregated in Stage 3; Stage 7 does not treat them as separate instances.",
                "- Each instance stores its peptide ID, original aggregated reporter intensity, and `log2(intensity + 1)`.",
                "- Peptide ID `0` is reserved for future padding or unknown peptides; observed peptide IDs begin at `1`.",
                "",
                "## Corpus",
                "",
                f"- Cells/bags: {summary['samples']}",
                f"- Unique peptide identities: {summary['peptides']}",
                f"- Observed peptide instances: {summary['instances']}",
                f"- Labels: {', '.join(f'`{label}` = {count}' for label, count in label_counts.items())}",
                f"- Chromatography batches: {', '.join(f'`{batch}` = {count}' for batch, count in batch_counts.items())}",
                "",
                "## Bag sizes",
                "",
                f"- Minimum: {summary['minimum_bag_size']}",
                f"- Median: {summary['median_bag_size']:.0f}",
                f"- Mean: {summary['mean_bag_size']:.1f}",
                f"- Maximum: {summary['maximum_bag_size']}",
                "- Per-cell and label-by-batch intensity summaries are saved with the bag statistics.",
                f"- Median bag size by chromatography batch: {', '.join(f'`{batch}` = {value:.0f}' for batch, value in batch_median_bag_sizes.items())}.",
                "- The batch differences reinforce the need to keep the saved raw-file-grouped evaluation design.",
                "",
                "## Leakage and integrity",
                "",
                "- No label-based filtering, imputation, scaling, or global prevalence filtering was applied.",
                "- The full vocabulary is an identity lookup only; fold-specific modeling transforms remain deferred to Stage 8.",
                "- Every bag is nonempty, has unique peptide IDs, and reconstructs its source matrix column exactly.",
                "- Saved Stage 5 raw-file-grouped fold assignments remain unchanged.",
                "",
                "## Stage gate",
                "",
                "Stage 7 is ready when bag reconstruction tests and the complete project test suite pass.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mil.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logger = configure_logging(Path("outputs/logs/stage_07_peptide_bags.log"))
    logger.info("Stage 7 completed: %s", run_stage_seven(args.config, args.dry_run))


if __name__ == "__main__":
    main()
