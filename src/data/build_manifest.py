"""Build an auditable raw-file/TMT-channel master sample manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.config import load_yaml_config
from src.utils.logging import configure_logging


MANIFEST_COLUMNS = [
    "sample_id", "source_cell_id", "label_raw", "label_binary", "is_valid_target",
    "exclusion_reason", "raw_file", "raw_file_stem", "tmt_set", "reporter_channel",
    "reporter_index", "reporter_mz", "batch_digest", "batch_sort",
    "batch_chromatography", "sample_category", "is_single_cell", "is_carrier",
    "is_reference", "is_normalization", "is_empty", "is_unused",
    "processed_matrix_column", "has_processed_features", "has_raw_file",
    "has_channel_annotation", "evidence_row_count", "reporter_signal_row_count", "reporter_mapping_offset",
]


def normalize_set(value: str) -> str:
    """Normalize raw-file/set IDs for matching while preserving the original ID."""
    return Path(str(value).strip()).name.removesuffix(".raw")


def category_and_exclusion(label: str, targets: set[str]) -> tuple[str, str | None]:
    """Classify a literal channel label without expanding biological meanings."""
    normalized = label.strip()
    lower = normalized.lower()
    if normalized in targets:
        return "target", None
    if lower.startswith("carrier"):
        return "carrier", "carrier_channel"
    if lower == "reference":
        return "reference", "reference_channel"
    if lower in {"norm", "normalization"}:
        return "normalization", "normalization_channel"
    if lower in {"unused", "empty"}:
        return lower, f"{lower}_channel"
    return "non_target", f"non_target_label:{normalized}"


def reporter_columns(annotation: pd.DataFrame) -> list[str]:
    """Return RI columns in numeric order and reject malformed annotation tables."""
    columns = [column for column in annotation.columns if column.startswith("RI") and column[2:].isdigit()]
    if not columns:
        raise ValueError("annotation file has no RI<n> reporter-channel columns")
    return sorted(columns, key=lambda name: int(name[2:]))


def evidence_coverage(evidence_csv: Path, raw_stems: set[str]) -> dict[str, dict[str, int]]:
    """Count evidence and non-missing reporter values for each local raw file."""
    reporter_fields = [f"Reporter.intensity.corrected.{index}" for index in range(17)]
    usecols = ["Raw.file", *reporter_fields]
    counts = {stem: {"evidence_row_count": 0, **{field: 0 for field in reporter_fields}, **{f"{field}_positive": 0 for field in reporter_fields}} for stem in raw_stems}
    for chunk in pd.read_csv(evidence_csv, usecols=usecols, chunksize=100_000, low_memory=False):
        chunk["_raw_stem"] = chunk["Raw.file"].astype(str).map(normalize_set)
        chunk = chunk[chunk["_raw_stem"].isin(raw_stems)]
        for raw_stem, group in chunk.groupby("_raw_stem"):
            counts[raw_stem]["evidence_row_count"] += len(group)
            for field in reporter_fields:
                numeric = pd.to_numeric(group[field], errors="coerce")
                counts[raw_stem][field] += int(numeric.notna().sum())
                counts[raw_stem][f"{field}_positive"] += int(numeric.gt(0).sum())
    return counts


def infer_reporter_offsets(annotation: pd.DataFrame, raw_stems: set[str], coverage: dict[str, dict[str, int]]) -> dict[str, int]:
    """Infer whether RI1 maps to corrected reporter index 0 or 1 for each run."""
    by_set = annotation.assign(_set_key=annotation["Set"].map(normalize_set)).set_index("_set_key")
    channels = reporter_columns(annotation)
    offsets = {}
    for raw_stem in raw_stems:
        candidates = {}
        for offset in (0, 1):
            candidates[offset] = sum(
                int(coverage[raw_stem].get(f"Reporter.intensity.corrected.{int(channel[2:]) - offset}", 0) > 0)
                for channel in channels
                if str(by_set.loc[raw_stem, channel]).strip().lower() != "unused" and int(channel[2:]) >= offset
            )
        if candidates[0] == candidates[1]:
            raise ValueError(f"ambiguous reporter-index mapping for {raw_stem}: {candidates}")
        offsets[raw_stem] = max(candidates, key=candidates.get)
    return offsets


def build_manifest(
    annotation: pd.DataFrame,
    batch: pd.DataFrame,
    raw_stems: set[str],
    coverage: dict[str, dict[str, int]],
    target_labels: set[str],
    reporter_offsets: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one candidate-sample row per local set and reporter channel."""
    if "Set" not in annotation or "set" not in batch:
        raise ValueError("annotation requires Set and batch metadata requires set")
    annotation = annotation.copy()
    batch = batch.copy()
    annotation["_set_key"] = annotation["Set"].map(normalize_set)
    batch["_set_key"] = batch["set"].map(normalize_set)
    if annotation["_set_key"].duplicated().any() or batch["_set_key"].duplicated().any():
        raise ValueError("duplicate normalized set IDs prevent an unambiguous manifest")
    ri_columns = reporter_columns(annotation)
    annotation_by_set = annotation.set_index("_set_key")
    batch_by_set = batch.set_index("_set_key")
    rows: list[dict[str, object]] = []
    unmatched: list[dict[str, str]] = []
    for raw_stem in sorted(raw_stems):
        if raw_stem not in annotation_by_set.index:
            unmatched.append({"record_type": "local_raw", "record_id": raw_stem, "reason": "missing_channel_annotation"})
            continue
        if raw_stem not in batch_by_set.index:
            unmatched.append({"record_type": "local_raw", "record_id": raw_stem, "reason": "missing_batch_metadata"})
            continue
        annotation_row, batch_row = annotation_by_set.loc[raw_stem], batch_by_set.loc[raw_stem]
        for channel in ri_columns:
            label = str(annotation_row[channel]).strip()
            reporter_offset = reporter_offsets[raw_stem]
            reporter_index = int(channel[2:]) - reporter_offset
            reporter_field = f"Reporter.intensity.corrected.{reporter_index}"
            category, exclusion_reason = category_and_exclusion(label, target_labels)
            is_biological_target = exclusion_reason is None
            evidence = coverage.get(raw_stem, {})
            signal_rows = int(evidence.get(f"{reporter_field}_positive", 0))
            if is_biological_target and signal_rows == 0:
                category, exclusion_reason = "target_without_signal", "no_positive_reporter_signal"
            is_target = exclusion_reason is None
            rows.append({
                "sample_id": f"{raw_stem}::{channel}", "source_cell_id": pd.NA,
                "label_raw": label, "label_binary": label if is_biological_target else pd.NA,
                "is_valid_target": is_target, "exclusion_reason": exclusion_reason,
                "raw_file": f"{raw_stem}.raw", "raw_file_stem": raw_stem,
                "tmt_set": annotation_row["Set"], "reporter_channel": channel,
                "reporter_index": reporter_index, "reporter_mz": pd.NA,
                "batch_digest": batch_row["digest"], "batch_sort": batch_row["sortday"],
                "batch_chromatography": batch_row["lcbatch"], "sample_category": category,
                "is_single_cell": label.lower().startswith("sc_"),
                "is_carrier": category == "carrier", "is_reference": category == "reference",
                "is_normalization": category == "normalization", "is_empty": category == "empty",
                "is_unused": category == "unused", "processed_matrix_column": pd.NA,
                "has_processed_features": bool(is_target and signal_rows > 0),
                "has_raw_file": True, "has_channel_annotation": True,
                "evidence_row_count": int(evidence.get("evidence_row_count", 0)),
                "reporter_signal_row_count": signal_rows,
                "reporter_mapping_offset": reporter_offset,
            })
    for set_key in sorted(set(annotation_by_set.index) - set(batch_by_set.index)):
        unmatched.append({"record_type": "annotation_set", "record_id": set_key, "reason": "missing_batch_metadata"})
    for set_key in sorted(set(batch_by_set.index) - set(annotation_by_set.index)):
        unmatched.append({"record_type": "batch_set", "record_id": set_key, "reason": "missing_channel_annotation"})
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS), pd.DataFrame(unmatched, columns=["record_type", "record_id", "reason"])


def report_manifest(manifest: pd.DataFrame, unmatched: pd.DataFrame) -> str:
    """Generate the required concise Stage-2 report."""
    valid = manifest[manifest["is_valid_target"]]
    label_counts = valid["label_raw"].value_counts().to_dict()
    exclusion_counts = manifest.loc[~manifest["is_valid_target"], "exclusion_reason"].value_counts().to_dict()
    return "\n".join([
        "# Stage 2 — Master sample manifest", "", "## Scope", "",
        "Local LCA9/LCA10/LCB7 raw-file subset only. Source cell IDs and processed-matrix columns are unavailable in the supplied files.", "",
        "## Reporter-channel convention", "",
        "Reporter-index offsets are inferred per run from non-unused annotation channels and observed corrected reporter-signal coverage: LCA9/LCA10 use RI1 → corrected `.0`; LCB7 uses RI1 → corrected `.1`. Physical reporter m/z values remain unverified and are left blank.", "",
        "## Coverage", "",
        f"- Candidate channels: {len(manifest)}",
        f"- Valid targets: {len(valid)} ({label_counts})",
        f"- Valid targets with evidence: {int(valid['has_processed_features'].sum())}",
        f"- Local raw files represented: {manifest['raw_file_stem'].nunique()}",
        f"- Unmatched source records: {len(unmatched)}", "",
        "## Exclusions", "",
        *[f"- {reason}: {count}" for reason, count in sorted(exclusion_counts.items())], "",
        "## Gate", "",
        "Raw-file/channel/label/batch mappings are available for this local subset. Do not treat `sample_id` as an original cell ID, and do not use reporter m/z values until the TMT reagent/mass convention is verified.", "",
    ])


def run_stage_two(config_path: str | Path, output_dir: str | Path = "outputs/tables") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load configured sources, build the manifest, and write all Stage-2 artifacts."""
    config = load_yaml_config(config_path)
    paths, options = config["paths"], config["manifest"]
    annotation = pd.read_csv(paths["annotation_csv"])
    batch = pd.read_csv(paths["batch_csv"])
    raw_roots = paths.get("raw_roots", [paths["inventory_root"]])
    raw_stems = {path.stem for root in raw_roots for path in Path(root).glob("*.raw")}
    coverage = evidence_coverage(Path(paths["evidence_csv"]), raw_stems)
    reporter_offsets = infer_reporter_offsets(annotation, raw_stems, coverage)
    manifest, unmatched = build_manifest(annotation, batch, raw_stems, coverage, set(options["target_labels"]), reporter_offsets)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    parquet_path = Path("data/interim/manifests/master_manifest.parquet")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(parquet_path, index=False)
    manifest.to_csv(output_path / "master_manifest.csv", index=False)
    manifest.groupby(["label_raw", "is_valid_target"], dropna=False).size().rename("count").reset_index().to_csv(output_path / "manifest_summary.csv", index=False)
    unmatched.to_csv(output_path / "unmatched_records.csv", index=False)
    manifest.loc[~manifest["is_valid_target"]].to_csv(output_path / "excluded_samples.csv", index=False)
    report_path = output_path.parent / "reports"
    report_path.mkdir(parents=True, exist_ok=True)
    (report_path / "manifest_report.md").write_text(report_manifest(manifest, unmatched), encoding="utf-8")
    return manifest, unmatched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output-dir", default="outputs/tables")
    parser.add_argument("--dry-run", action="store_true", help="validate configuration and source paths without writing outputs")
    args = parser.parse_args()
    logger = configure_logging(Path("outputs/logs/stage_02_manifest.log"))
    if args.dry_run:
        config = load_yaml_config(args.config)
        for name in ("annotation_csv", "batch_csv", "evidence_csv"):
            if not Path(config["paths"][name]).exists():
                raise FileNotFoundError(f"Missing configured source: {name}")
        for raw_root in config["paths"].get("raw_roots", [config["paths"]["inventory_root"]]):
            if not Path(raw_root).exists():
                raise FileNotFoundError(f"Missing configured raw root: {raw_root}")
        logger.info("Stage 2 dry run passed")
        return
    manifest, unmatched = run_stage_two(args.config, args.output_dir)
    logger.info("Manifest completed: %d candidate channels, %d unmatched records", len(manifest), len(unmatched))


if __name__ == "__main__":
    main()
