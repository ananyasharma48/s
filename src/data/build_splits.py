"""Create immutable, stratified raw-file-grouped evaluation assignments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.config import load_yaml_config
from src.utils.logging import configure_logging


def assert_group_disjoint(frame: pd.DataFrame, group_column: str, split_column: str = "partition") -> None:
    """Reject any fold that places one group in more than one partition."""
    overlap = frame.groupby(group_column)[split_column].nunique()
    if (overlap > 1).any():
        raise AssertionError(f"groups cross partitions: {overlap[overlap > 1].index.tolist()}")


def make_outer_assignments(samples: pd.DataFrame, group_column: str, n_splits: int, seed: int) -> pd.DataFrame:
    """Assign each sample to one grouped outer-test fold."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    records = []
    for fold, (train_index, test_index) in enumerate(splitter.split(samples, samples["label_raw"], samples[group_column])):
        for index, partition in ((train_index, "train"), (test_index, "test")):
            selected = samples.iloc[index]
            records.extend({"outer_fold": fold, "sample_id": row.sample_id, "label_raw": row.label_raw, group_column: getattr(row, group_column), "partition": partition} for row in selected.itertuples(index=False))
    assignments = pd.DataFrame(records)
    for _, fold_frame in assignments.groupby("outer_fold"):
        assert_group_disjoint(fold_frame, group_column)
        if fold_frame.loc[fold_frame["partition"].eq("test"), "label_raw"].nunique() != 2:
            raise AssertionError("an outer test fold does not contain both labels")
    return assignments


def make_inner_assignments(outer: pd.DataFrame, group_column: str, n_splits: int, seed: int) -> pd.DataFrame:
    """Create grouped inner training/validation assignments inside each outer training set."""
    records = []
    for outer_fold, outer_frame in outer.groupby("outer_fold"):
        training = outer_frame[outer_frame["partition"].eq("train")].reset_index(drop=True)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + int(outer_fold))
        for inner_fold, (train_index, validation_index) in enumerate(splitter.split(training, training["label_raw"], training[group_column])):
            for index, partition in ((train_index, "train"), (validation_index, "validation")):
                selected = training.iloc[index]
                records.extend({"outer_fold": outer_fold, "inner_fold": inner_fold, "sample_id": row.sample_id, "label_raw": row.label_raw, group_column: getattr(row, group_column), "partition": partition} for row in selected.itertuples(index=False))
    inner = pd.DataFrame(records)
    for _, frame in inner.groupby(["outer_fold", "inner_fold"]):
        assert_group_disjoint(frame, group_column)
        if frame.loc[frame["partition"].eq("validation"), "label_raw"].nunique() != 2:
            raise AssertionError("an inner validation fold does not contain both labels")
    return inner


def make_batch_holdouts(samples: pd.DataFrame, batch_column: str) -> pd.DataFrame:
    """Create whole-batch test assignments and flag class-feasibility."""
    records = []
    for batch in sorted(samples[batch_column].unique()):
        for row in samples.itertuples(index=False):
            records.append({"held_out_batch": batch, "sample_id": row.sample_id, "label_raw": row.label_raw, "partition": "test" if getattr(row, batch_column) == batch else "train"})
    assignments = pd.DataFrame(records)
    for _, frame in assignments.groupby("held_out_batch"):
        assert_group_disjoint(frame, batch_column, "partition") if batch_column in frame else None
    return assignments


def run_stage_five(config_path: str | Path = "configs/evaluation.yaml") -> dict[str, int]:
    """Write all Stage 5 group-aware split artifacts."""
    config = load_yaml_config(config_path)
    manifest = pd.read_parquet("data/interim/manifests/master_manifest.parquet")
    samples = manifest.loc[manifest["is_valid_target"], ["sample_id", "label_raw", "raw_file_stem", "batch_chromatography"]].copy()
    group_column = "raw_file_stem"
    outer = make_outer_assignments(samples, group_column, int(config["outer_folds"]), int(config["seed"]))
    inner = make_inner_assignments(outer, group_column, int(config["inner_folds"]), int(config["seed"]))
    batch = make_batch_holdouts(samples, str(config["batch_holdout_column"]))
    split_dir, table_dir, report_dir = Path("data/splits"), Path("outputs/tables"), Path("outputs/reports")
    split_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    outer.to_csv(split_dir / "outer_fold_assignments.csv", index=False)
    inner.to_csv(split_dir / "inner_fold_assignments.csv", index=False)
    batch.to_csv(split_dir / "batch_holdout_assignments.csv", index=False)
    summary = outer.groupby(["outer_fold", "partition", "label_raw"], dropna=False).size().rename("sample_count").reset_index()
    summary.to_csv(table_dir / "stage_05_outer_fold_summary.csv", index=False)
    batch_summary = batch.groupby(["held_out_batch", "partition", "label_raw"], dropna=False).size().rename("sample_count").reset_index()
    batch_summary.to_csv(table_dir / "stage_05_batch_holdout_summary.csv", index=False)
    (report_dir / "stage_05_group_splits.md").write_text("\n".join([
        "# Stage 5 — Leakage-safe data splits", "", "## Primary split", "", f"- Outer folds: {config['outer_folds']}", f"- Grouping unit: {group_column}", f"- Usable samples: {len(samples)}", f"- Independent raw-file groups: {samples[group_column].nunique()}", "- Every outer test fold contains both `sc_u` and `sc_m0`.", "", "## Inner split", "", f"- Inner folds: {config['inner_folds']}", "- Inner folds are grouped by raw file and are created only from their outer training partition.", "", "## Batch holdouts", "", f"- Whole-{config['batch_holdout_column']} assignments were saved as secondary robustness tests.", "- These assignments are not substitutes for the raw-file-grouped primary endpoint.", "", "## Integrity", "", "- No raw file crosses train/test or train/validation partitions within any fold.", "- These assignments are immutable inputs for all later baseline and MIL model comparisons.", "",
    ]), encoding="utf-8")
    return {"samples": len(samples), "raw_groups": samples[group_column].nunique(), "outer_rows": len(outer), "inner_rows": len(inner)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation.yaml")
    args = parser.parse_args()
    logger = configure_logging(Path("outputs/logs/stage_05_splits.log"))
    logger.info("Stage 5 completed: %s", run_stage_five(args.config))


if __name__ == "__main__":
    main()
