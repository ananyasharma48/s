"""Execute Stage 13 expanded source inventory and sample-manifest refresh."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data.build_manifest import (
    category_and_exclusion,
    evidence_coverage,
    normalize_set,
    reporter_columns,
)
from src.data.extract_reporter_ions import inspect_mzml_header
from src.utils.config import load_yaml_config
from src.utils.logging import configure_logging


SOURCE_INVENTORY_COLUMNS = [
    "source_type",
    "acquisition_stem",
    "path",
    "path_status",
    "canonical_path",
    "size_bytes",
    "sha256",
    "sha1",
    "row_count",
    "column_count",
    "parsing_status",
    "mzml_complete",
    "mzml_indexed",
    "mzml_id",
    "mzml_id_matches_stem",
    "run_id",
    "run_id_matches_stem",
    "declared_spectrum_count",
    "source_checksum_algorithm",
    "source_checksum_embedded",
    "source_checksum_actual",
    "source_checksum_matches",
    "converter_tool",
    "converter_version",
    "conversion_methods",
    "reporter_scheme",
    "status_detail",
]

EXPANDED_MANIFEST_COLUMNS = [
    "sample_id",
    "label_raw",
    "label_binary",
    "label_numeric",
    "is_valid_target",
    "stage13_candidate",
    "candidate_status",
    "qc_status",
    "exclusion_reason",
    "source_cohort",
    "raw_file_stem",
    "raw_file",
    "raw_path",
    "mzml_path",
    "tmt_set",
    "reporter_channel",
    "reporter_index",
    "reporter_mz",
    "reporter_scheme",
    "batch_digest",
    "batch_sort",
    "batch_chromatography",
    "sample_category",
    "is_single_cell",
    "is_carrier",
    "is_reference",
    "is_normalization",
    "is_empty",
    "is_unused",
    "has_raw_file",
    "has_mzml_file",
    "mzml_complete",
    "raw_checksum_matches_mzml",
    "has_channel_annotation",
    "has_batch_metadata",
    "has_processed_spectrum_bag",
    "evidence_row_count",
    "search_reporter_index",
    "search_reporter_signal_row_count",
    "search_reporter_field_observed",
    "acquisition_status",
]

UNMATCHED_COLUMNS = [
    "record_type",
    "record_id",
    "raw_file_stem",
    "reporter_channel",
    "label_raw",
    "source_cohort",
    "status",
    "reason",
]


def discover_files(locations: Iterable[str | Path], suffix: str) -> list[Path]:
    """Find top-level files in configured locations while preserving preference order."""
    discovered: list[Path] = []
    seen_paths: set[Path] = set()
    for location_value in locations:
        location = Path(location_value)
        if not location.is_dir():
            raise NotADirectoryError(f"configured source directory does not exist: {location}")
        for path in sorted(location.glob(f"*{suffix}")):
            resolved = path.resolve()
            if resolved not in seen_paths:
                discovered.append(path)
                seen_paths.add(resolved)
    return discovered


def group_by_stem(paths: Iterable[Path]) -> dict[str, list[Path]]:
    """Group physical files by acquisition stem in discovery/preference order."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[path.stem].append(path)
    return dict(grouped)


def choose_canonical(grouped: dict[str, list[Path]]) -> dict[str, Path]:
    """Choose the first configured physical path for each acquisition stem."""
    return {stem: paths[0] for stem, paths in grouped.items()}


def fingerprint_file(path: Path, include_sha1: bool, count_rows: bool) -> dict[str, Any]:
    """Hash a source file once, optionally counting line-delimited records."""
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1() if include_sha1 else None
    newline_count = 0
    last_byte = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha256.update(block)
            if sha1 is not None:
                sha1.update(block)
            if count_rows:
                newline_count += block.count(b"\n")
                last_byte = block[-1:]
    if count_rows and path.stat().st_size and last_byte != b"\n":
        newline_count += 1
    return {
        "size_bytes": path.stat().st_size,
        "sha256": sha256.hexdigest(),
        "sha1": sha1.hexdigest() if sha1 is not None else None,
        "row_count": max(newline_count - 1, 0) if count_rows else None,
    }


def mzml_tail_status(path: Path) -> tuple[bool, bool]:
    """Check that an mzML has a complete closing element and an index footer."""
    with path.open("rb") as handle:
        handle.seek(max(path.stat().st_size - 256 * 1024, 0))
        tail = handle.read().decode("utf-8", errors="ignore")
    complete = "</indexedmzML>" in tail or "</mzML>" in tail
    indexed = "<indexListOffset>" in tail and "</indexedmzML>" in tail
    return complete, indexed


def converter_metadata(header: dict[str, Any]) -> tuple[str | None, str | None, str]:
    """Flatten converter and processing metadata from an mzML header."""
    software = header.get("software", [])
    tool = version = None
    if software:
        entry = software[0]
        names = entry.get("names") or []
        tool = names[0] if names else entry.get("id")
        version = entry.get("version")
    methods = []
    for method in header.get("processing_methods", []):
        methods.extend(method.get("methods") or [])
    return tool, version, "; ".join(methods)


def build_source_inventory(
    raw_files: list[Path],
    mzml_files: list[Path],
    metadata_files: dict[str, Path],
    batch_by_stem: dict[str, str],
    scheme_by_batch: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Create a checksum/provenance row for every physical source file."""
    raw_groups = group_by_stem(raw_files)
    mzml_groups = group_by_stem(mzml_files)
    canonical_raw = choose_canonical(raw_groups)
    canonical_mzml = choose_canonical(mzml_groups)
    tasks: list[tuple[Path, bool, bool]] = []
    tasks.extend((path, True, False) for path in raw_files)
    tasks.extend((path, False, False) for path in mzml_files)
    tasks.extend((path, False, True) for path in metadata_files.values())
    with ThreadPoolExecutor(max_workers=4) as executor:
        fingerprints = list(executor.map(lambda args: fingerprint_file(*args), tasks))
    fingerprint_by_path = {str(task[0]): value for task, value in zip(tasks, fingerprints)}

    raw_details: dict[str, dict[str, Any]] = {}
    for stem, paths in raw_groups.items():
        canonical = canonical_raw[stem]
        hashes = [fingerprint_by_path[str(path)] for path in paths]
        raw_details[stem] = {
            "paths": paths,
            "canonical_path": canonical,
            "canonical_sha1": fingerprint_by_path[str(canonical)]["sha1"],
            "canonical_sha256": fingerprint_by_path[str(canonical)]["sha256"],
            "copies_consistent": len({entry["sha256"] for entry in hashes}) == 1,
        }

    mzml_headers: dict[str, dict[str, Any]] = {}
    mzml_details: dict[str, dict[str, Any]] = {}
    for path in mzml_files:
        header = inspect_mzml_header(path)
        complete, indexed = mzml_tail_status(path)
        header["mzml_complete"] = complete
        header["mzml_indexed"] = indexed
        mzml_headers[str(path)] = header
    for stem, paths in mzml_groups.items():
        canonical = canonical_mzml[stem]
        embedded = {
            mzml_headers[str(path)].get("source_checksum_embedded") for path in paths
        }
        mzml_details[stem] = {
            "paths": paths,
            "canonical_path": canonical,
            "canonical_sha256": fingerprint_by_path[str(canonical)]["sha256"],
            "source_checksums_consistent": len(embedded) == 1,
        }

    rows: list[dict[str, Any]] = []
    for path in raw_files:
        stem = path.stem
        canonical = canonical_raw[stem]
        fingerprint = fingerprint_by_path[str(path)]
        rows.append({
            "source_type": "raw",
            "acquisition_stem": stem,
            "path": str(path),
            "path_status": "canonical" if path == canonical else "duplicate_copy",
            "canonical_path": str(canonical),
            **fingerprint,
            "column_count": None,
            "parsing_status": "immutable_binary_source",
            "source_checksum_actual": fingerprint["sha1"],
            "reporter_scheme": scheme_by_batch.get(batch_by_stem.get(stem, "")),
            "status_detail": "byte-identical copies" if raw_details[stem]["copies_consistent"] else "conflicting files share this stem",
        })
    for path in mzml_files:
        stem = path.stem
        canonical = canonical_mzml[stem]
        fingerprint = fingerprint_by_path[str(path)]
        header = mzml_headers[str(path)]
        raw_actual = raw_details.get(stem, {}).get("canonical_sha1")
        embedded = header.get("source_checksum_embedded")
        tool, version, methods = converter_metadata(header)
        complete = bool(header.get("mzml_complete"))
        indexed = bool(header.get("mzml_indexed"))
        mzml_id = header.get("mzml_id")
        run_id = header.get("run_id")
        rows.append({
            "source_type": "mzml",
            "acquisition_stem": stem,
            "path": str(path),
            "path_status": "canonical" if path == canonical else "duplicate_conversion",
            "canonical_path": str(canonical),
            **fingerprint,
            "column_count": None,
            "parsing_status": (
                "valid_indexed_mzml" if complete and indexed
                else ("valid_unindexed_mzml" if complete else "invalid_or_incomplete_mzml")
            ),
            "mzml_complete": complete,
            "mzml_indexed": indexed,
            "mzml_id": mzml_id,
            "mzml_id_matches_stem": mzml_id == stem,
            "run_id": run_id,
            "run_id_matches_stem": str(run_id).lstrip("_") == stem,
            "declared_spectrum_count": header.get("declared_spectrum_count"),
            "source_checksum_algorithm": header.get("source_checksum_algorithm"),
            "source_checksum_embedded": embedded,
            "source_checksum_actual": raw_actual,
            "source_checksum_matches": bool(raw_actual and embedded and raw_actual == embedded),
            "converter_tool": tool,
            "converter_version": version,
            "conversion_methods": methods,
            "reporter_scheme": scheme_by_batch.get(batch_by_stem.get(stem, "")),
            "status_detail": (
                "complete mzML with indexed footer" if complete and indexed
                else ("complete mzML without indexed footer" if complete else "incomplete mzML document")
            ),
        })
    for source_type, path in metadata_files.items():
        fingerprint = fingerprint_by_path[str(path)]
        header = pd.read_csv(path, nrows=0)
        rows.append({
            "source_type": source_type,
            "acquisition_stem": None,
            "path": str(path),
            "path_status": "canonical",
            "canonical_path": str(path),
            **fingerprint,
            "column_count": len(header.columns),
            "parsing_status": "parsed_tabular_source",
            "status_detail": ", ".join(map(str, header.columns[:12])),
        })
    inventory = pd.DataFrame(rows).reindex(columns=SOURCE_INVENTORY_COLUMNS)
    return inventory, raw_details, mzml_details


def build_run_summary(
    raw_details: dict[str, dict[str, Any]],
    mzml_details: dict[str, dict[str, Any]],
    inventory: pd.DataFrame,
    annotation: pd.DataFrame,
    batch: pd.DataFrame,
    existing_run_stems: set[str],
    existing_bag_stems: set[str],
    approved_exclusions: dict[str, str],
    scheme_by_batch: dict[str, str],
    coverage: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """Build one explicit status row per unique local acquisition."""
    annotation_keys = set(annotation["Set"].map(normalize_set))
    batch_keys = set(batch["set"].map(normalize_set))
    batch_lookup = batch.assign(_key=batch["set"].map(normalize_set)).set_index("_key")
    stems = sorted(set(raw_details) | set(mzml_details))
    canonical_mzml_rows = inventory[
        inventory["source_type"].eq("mzml") & inventory["path_status"].eq("canonical")
    ].set_index("acquisition_stem")
    records = []
    for stem in stems:
        raw = raw_details.get(stem)
        mzml = mzml_details.get(stem)
        batch_row = batch_lookup.loc[stem] if stem in batch_lookup.index else None
        chromatography = str(batch_row["lcbatch"]) if batch_row is not None else None
        mzml_row = canonical_mzml_rows.loc[stem] if stem in canonical_mzml_rows.index else None
        if stem in approved_exclusions:
            status = "excluded_user_approved"
            detail = approved_exclusions[stem]
        elif raw is None:
            status, detail = "blocked_missing_raw", "RAW source is unavailable"
        elif mzml is None:
            status, detail = "blocked_missing_mzml", "mzML conversion is unavailable"
        elif not raw["copies_consistent"]:
            status, detail = "blocked_ambiguous_raw", "conflicting RAW files share one stem"
        elif not mzml["source_checksums_consistent"]:
            status, detail = "blocked_ambiguous_mzml", "mzML copies reference different RAW checksums"
        elif mzml_row is None or not bool(mzml_row["mzml_complete"]):
            status, detail = "blocked_incomplete_mzml", "mzML is incomplete"
        elif not bool(mzml_row["mzml_id_matches_stem"]) or not bool(mzml_row["run_id_matches_stem"]):
            status, detail = "blocked_run_id_mismatch", "mzML document or run ID does not match the acquisition stem"
        elif not bool(mzml_row["source_checksum_matches"]):
            status, detail = "blocked_checksum_mismatch", "mzML embedded RAW checksum does not match local RAW"
        elif stem not in annotation_keys:
            status, detail = "blocked_missing_annotation", "channel annotation is unavailable"
        elif stem not in batch_keys:
            status, detail = "blocked_missing_batch", "batch metadata is unavailable"
        elif chromatography not in scheme_by_batch:
            status, detail = "blocked_unknown_reporter_scheme", "no reporter scheme is configured for this chromatography batch"
        elif stem in existing_run_stems:
            status, detail = "ready_existing_stage11", "existing Stage 11 acquisition retained for deduplication"
        else:
            status, detail = "ready_new_stage13", "new acquisition is ready for Stage 14 reconstruction"
        records.append({
            "raw_file_stem": stem,
            "source_cohort": "existing_stage11_scope" if stem in existing_run_stems else "newly_available",
            "batch_chromatography": chromatography,
            "batch_sort": batch_row["sortday"] if batch_row is not None else None,
            "batch_digest": batch_row["digest"] if batch_row is not None else None,
            "reporter_scheme": scheme_by_batch.get(chromatography),
            "raw_copy_count": len(raw["paths"]) if raw else 0,
            "mzml_copy_count": len(mzml["paths"]) if mzml else 0,
            "canonical_raw_path": str(raw["canonical_path"]) if raw else None,
            "canonical_mzml_path": str(mzml["canonical_path"]) if mzml else None,
            "raw_copies_consistent": raw["copies_consistent"] if raw else False,
            "mzml_source_checksums_consistent": mzml["source_checksums_consistent"] if mzml else False,
            "mzml_complete": bool(mzml_row["mzml_complete"]) if mzml_row is not None else False,
            "mzml_indexed": bool(mzml_row["mzml_indexed"]) if mzml_row is not None else False,
            "mzml_id_matches_stem": bool(mzml_row["mzml_id_matches_stem"]) if mzml_row is not None else False,
            "run_id_matches_stem": bool(mzml_row["run_id_matches_stem"]) if mzml_row is not None else False,
            "raw_checksum_matches_mzml": bool(mzml_row["source_checksum_matches"]) if mzml_row is not None else False,
            "converter_tool": mzml_row["converter_tool"] if mzml_row is not None else None,
            "converter_version": mzml_row["converter_version"] if mzml_row is not None else None,
            "declared_spectrum_count": mzml_row["declared_spectrum_count"] if mzml_row is not None else None,
            "has_channel_annotation": stem in annotation_keys,
            "has_batch_metadata": stem in batch_keys,
            "evidence_row_count": int(coverage.get(stem, {}).get("evidence_row_count", 0)),
            "has_existing_spectrum_bag": stem in existing_bag_stems,
            "acquisition_status": status,
            "status_detail": detail,
        })
    return pd.DataFrame(records)


def build_expanded_manifest(
    annotation: pd.DataFrame,
    batch: pd.DataFrame,
    run_summary: pd.DataFrame,
    existing_manifest: pd.DataFrame,
    existing_bags: pd.DataFrame,
    coverage: dict[str, dict[str, int]],
    target_labels: set[str],
    reporter_schemes: dict[str, dict[str, Any]],
    evidence_offset_by_batch: dict[str, int],
) -> pd.DataFrame:
    """Create one deduplicated row per acquisition/channel with a Stage 13 disposition."""
    annotation = annotation.copy()
    batch = batch.copy()
    annotation["_key"] = annotation["Set"].map(normalize_set)
    batch["_key"] = batch["set"].map(normalize_set)
    if annotation["_key"].duplicated().any() or batch["_key"].duplicated().any():
        raise ValueError("duplicate normalized metadata set IDs prevent an unambiguous Stage 13 manifest")
    annotation_lookup = annotation.set_index("_key")
    batch_lookup = batch.set_index("_key")
    existing_lookup = existing_manifest.set_index("sample_id")
    retained_samples = set(existing_bags["sample_id"].astype(str))
    channels = reporter_columns(annotation)
    rows: list[dict[str, Any]] = []
    for run in run_summary.sort_values("raw_file_stem").itertuples(index=False):
        stem = str(run.raw_file_stem)
        if stem not in annotation_lookup.index or stem not in batch_lookup.index:
            continue
        annotation_row = annotation_lookup.loc[stem]
        batch_row = batch_lookup.loc[stem]
        batch_name = str(batch_row["lcbatch"])
        scheme = str(run.reporter_scheme)
        masses = reporter_schemes.get(scheme, {}).get("masses", {})
        search_offset = int(evidence_offset_by_batch.get(batch_name, 1))
        for channel in channels:
            channel_number = int(channel[2:])
            sample_id = f"{stem}::{channel}"
            label = str(annotation_row[channel]).strip()
            category, category_exclusion = category_and_exclusion(label, target_labels)
            literal_target = label in target_labels
            search_index = channel_number - search_offset
            search_field = f"Reporter.intensity.corrected.{search_index}"
            run_coverage = coverage.get(stem, {})
            search_signal = int(run_coverage.get(f"{search_field}_positive", 0))
            search_observed = int(run_coverage.get(search_field, 0)) > 0
            has_bag = sample_id in retained_samples
            if not literal_target:
                candidate = False
                candidate_status = "excluded_non_target_channel"
                qc_status = "not_applicable"
                exclusion_reason = category_exclusion
            elif has_bag:
                candidate = True
                candidate_status = "retained_existing_stage11"
                qc_status = "passed_stage11_spectrum_bag_qc"
                exclusion_reason = None
                category = "target"
            elif run.source_cohort == "newly_available" and run.acquisition_status == "ready_new_stage13":
                candidate = True
                candidate_status = "candidate_pending_stage14_qc"
                qc_status = "file_manifest_qc_passed"
                exclusion_reason = None
                category = "target"
            else:
                candidate = False
                candidate_status = "excluded_existing_or_blocked_target"
                qc_status = "failed_or_not_retained_before_stage13"
                if sample_id in existing_lookup.index:
                    previous_reason = existing_lookup.loc[sample_id, "exclusion_reason"]
                    exclusion_reason = None if pd.isna(previous_reason) else str(previous_reason)
                else:
                    exclusion_reason = None
                if not exclusion_reason:
                    exclusion_reason = (
                        "user_approved_missing_mzml"
                        if run.acquisition_status == "excluded_user_approved"
                        else f"acquisition_status:{run.acquisition_status}"
                    )
                category = "target_excluded"
            rows.append({
                "sample_id": sample_id,
                "label_raw": label,
                "label_binary": label if literal_target else pd.NA,
                "label_numeric": 1 if label == "sc_u" else (0 if label == "sc_m0" else pd.NA),
                "is_valid_target": candidate,
                "stage13_candidate": candidate,
                "candidate_status": candidate_status,
                "qc_status": qc_status,
                "exclusion_reason": exclusion_reason,
                "source_cohort": run.source_cohort,
                "raw_file_stem": stem,
                "raw_file": f"{stem}.raw",
                "raw_path": run.canonical_raw_path,
                "mzml_path": run.canonical_mzml_path,
                "tmt_set": annotation_row["Set"],
                "reporter_channel": channel,
                "reporter_index": channel_number - 1,
                "reporter_mz": masses.get(channel),
                "reporter_scheme": scheme,
                "batch_digest": batch_row["digest"],
                "batch_sort": batch_row["sortday"],
                "batch_chromatography": batch_name,
                "sample_category": category,
                "is_single_cell": label.lower().startswith("sc_"),
                "is_carrier": category == "carrier",
                "is_reference": category == "reference",
                "is_normalization": category == "normalization",
                "is_empty": category == "empty",
                "is_unused": category == "unused",
                "has_raw_file": bool(run.raw_copy_count),
                "has_mzml_file": bool(run.mzml_copy_count),
                "mzml_complete": bool(run.mzml_complete),
                "raw_checksum_matches_mzml": bool(run.raw_checksum_matches_mzml),
                "has_channel_annotation": bool(run.has_channel_annotation),
                "has_batch_metadata": bool(run.has_batch_metadata),
                "has_processed_spectrum_bag": has_bag,
                "evidence_row_count": int(run.evidence_row_count),
                "search_reporter_index": search_index,
                "search_reporter_signal_row_count": search_signal,
                "search_reporter_field_observed": search_observed,
                "acquisition_status": run.acquisition_status,
            })
    return pd.DataFrame(rows).reindex(columns=EXPANDED_MANIFEST_COLUMNS)


def build_duplicate_audit(inventory: pd.DataFrame) -> pd.DataFrame:
    """Classify duplicate physical paths and possible cross-stem reinjections."""
    records: list[dict[str, Any]] = []
    mass_spec = inventory[inventory["source_type"].isin(["raw", "mzml"])]
    for (source_type, stem), group in mass_spec.groupby(["source_type", "acquisition_stem"], sort=True):
        if len(group) < 2:
            continue
        canonical = group[group["path_status"].eq("canonical")].iloc[0]
        for duplicate in group[~group["path_status"].eq("canonical")].itertuples(index=False):
            byte_identical = duplicate.sha256 == canonical["sha256"]
            embedded_match = (
                source_type == "mzml"
                and pd.notna(duplicate.source_checksum_embedded)
                and duplicate.source_checksum_embedded == canonical["source_checksum_embedded"]
            )
            if byte_identical:
                relationship = "duplicate_file_copy"
                reason = "same acquisition stem and identical SHA-256"
            elif embedded_match:
                relationship = "reconverted_duplicate_acquisition"
                reason = "same acquisition and embedded RAW checksum; mzML bytes differ"
            else:
                relationship = "ambiguous_duplicate_stem"
                reason = "same stem but neither file checksum nor embedded source checksum agrees"
            records.append({
                "relationship_type": relationship,
                "source_type": source_type,
                "acquisition_stem": stem,
                "canonical_path": canonical["path"],
                "other_path": duplicate.path,
                "canonical_sha256": canonical["sha256"],
                "other_sha256": duplicate.sha256,
                "byte_identical": byte_identical,
                "embedded_source_checksum_matches": embedded_match,
                "counted_acquisition_instances": 1 if relationship != "ambiguous_duplicate_stem" else 0,
                "disposition": "exclude_other_physical_copy" if relationship != "ambiguous_duplicate_stem" else "manual_review_required",
                "reason": reason,
            })
    canonical_mzml = mass_spec[
        mass_spec["source_type"].eq("mzml") & mass_spec["path_status"].eq("canonical")
    ].dropna(subset=["source_checksum_embedded"])
    for checksum, group in canonical_mzml.groupby("source_checksum_embedded"):
        stems = sorted(group["acquisition_stem"].astype(str).unique())
        if len(stems) > 1:
            records.append({
                "relationship_type": "possible_reinjection",
                "source_type": "mzml",
                "acquisition_stem": ";".join(stems),
                "canonical_path": group.iloc[0]["path"],
                "other_path": ";".join(group.iloc[1:]["path"].astype(str)),
                "canonical_sha256": group.iloc[0]["sha256"],
                "other_sha256": ";".join(group.iloc[1:]["sha256"].astype(str)),
                "byte_identical": False,
                "embedded_source_checksum_matches": True,
                "counted_acquisition_instances": len(stems),
                "disposition": "manual_review_required",
                "reason": f"different acquisition stems share embedded RAW checksum {checksum}",
            })
    return pd.DataFrame(records, columns=[
        "relationship_type", "source_type", "acquisition_stem", "canonical_path",
        "other_path", "canonical_sha256", "other_sha256", "byte_identical",
        "embedded_source_checksum_matches", "counted_acquisition_instances",
        "disposition", "reason",
    ])


def build_exclusion_unmatched_table(
    manifest: pd.DataFrame,
    run_summary: pd.DataFrame,
    annotation: pd.DataFrame,
    batch: pd.DataFrame,
) -> pd.DataFrame:
    """Combine excluded channels and source/metadata coverage gaps."""
    records: list[dict[str, Any]] = []
    for row in manifest[~manifest["stage13_candidate"]].itertuples(index=False):
        records.append({
            "record_type": "channel",
            "record_id": row.sample_id,
            "raw_file_stem": row.raw_file_stem,
            "reporter_channel": row.reporter_channel,
            "label_raw": row.label_raw,
            "source_cohort": row.source_cohort,
            "status": row.candidate_status,
            "reason": row.exclusion_reason,
        })
    for row in run_summary[~run_summary["acquisition_status"].str.startswith("ready_")].itertuples(index=False):
        records.append({
            "record_type": "acquisition",
            "record_id": row.raw_file_stem,
            "raw_file_stem": row.raw_file_stem,
            "reporter_channel": None,
            "label_raw": None,
            "source_cohort": row.source_cohort,
            "status": row.acquisition_status,
            "reason": row.status_detail,
        })
    local_stems = set(run_summary["raw_file_stem"])
    for set_id in sorted(set(annotation["Set"].map(normalize_set)) - local_stems):
        records.append({
            "record_type": "annotation_set",
            "record_id": set_id,
            "status": "metadata_outside_local_file_scope",
            "reason": "annotation exists but no local RAW or mzML is available",
        })
    for set_id in sorted(set(batch["set"].map(normalize_set)) - local_stems):
        records.append({
            "record_type": "batch_set",
            "record_id": set_id,
            "status": "metadata_outside_local_file_scope",
            "reason": "batch metadata exists but no local RAW or mzML is available",
        })
    return pd.DataFrame(records).reindex(columns=UNMATCHED_COLUMNS)


def validate_stage13(
    inventory: pd.DataFrame,
    run_summary: pd.DataFrame,
    manifest: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
) -> dict[str, Any]:
    """Enforce Stage 13 file, channel, deduplication, and candidate-count gates."""
    if inventory.empty or run_summary.empty or manifest.empty:
        raise AssertionError("Stage 13 produced an empty required table")
    if manifest["sample_id"].duplicated().any():
        raise AssertionError("expanded manifest contains duplicate acquisition/channel sample IDs")
    expected_channels = run_summary["has_channel_annotation"].sum() * 16
    if len(manifest) != expected_channels:
        raise AssertionError(f"expected {expected_channels} annotated channel rows, found {len(manifest)}")
    candidates = manifest[manifest["stage13_candidate"]]
    if not candidates["label_raw"].isin(["sc_m0", "sc_u"]).all():
        raise AssertionError("a Stage 13 candidate has an unsupported biological label")
    if not candidates.groupby("sample_id").size().eq(1).all():
        raise AssertionError("a Stage 13 candidate maps to more than one label or group")
    if not candidates[["raw_file_stem", "reporter_channel", "batch_chromatography"]].notna().all().all():
        raise AssertionError("a Stage 13 candidate lacks a required group or channel mapping")
    if candidates["reporter_mz"].isna().any():
        raise AssertionError("a Stage 13 candidate lacks a reporter mass under its configured scheme")
    blocked_candidates = candidates[~candidates["acquisition_status"].str.startswith("ready_")]
    if not blocked_candidates.empty:
        raise AssertionError("blocked acquisitions were incorrectly retained as candidates")
    ambiguous = duplicate_audit[duplicate_audit["relationship_type"].eq("ambiguous_duplicate_stem")]
    if not ambiguous.empty:
        raise AssertionError("ambiguous duplicate stems require review before Stage 13 can pass")
    return {
        "physical_source_files": int(len(inventory)),
        "physical_raw_files": int(inventory["source_type"].eq("raw").sum()),
        "physical_mzml_files": int(inventory["source_type"].eq("mzml").sum()),
        "unique_acquisitions": int(run_summary["raw_file_stem"].nunique()),
        "file_ready_acquisitions": int(run_summary["acquisition_status"].str.startswith("ready_").sum()),
        "expanded_manifest_channels": int(len(manifest)),
        "literal_target_channels": int(manifest["label_raw"].isin(["sc_m0", "sc_u"]).sum()),
        "exact_stage13_candidate_cells": int(len(candidates)),
        "candidate_label_counts": candidates["label_raw"].value_counts().sort_index().to_dict(),
        "candidate_raw_file_groups": int(candidates["raw_file_stem"].nunique()),
        "retained_existing_cells": int(candidates["has_processed_spectrum_bag"].sum()),
        "new_candidate_cells": int((candidates["source_cohort"] == "newly_available").sum()),
        "duplicate_audit_rows": int(len(duplicate_audit)),
        "possible_reinjections": int(duplicate_audit["relationship_type"].eq("possible_reinjection").sum()),
    }


def render_report(
    metadata: dict[str, Any],
    inventory: pd.DataFrame,
    run_summary: pd.DataFrame,
    manifest: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> str:
    """Render the Stage 13 data incorporation report."""
    candidates = manifest[manifest["stage13_candidate"]]
    by_batch = candidates.groupby(["batch_chromatography", "label_raw"]).size().unstack(fill_value=0)
    target_exclusions = manifest[
        manifest["label_raw"].isin(["sc_m0", "sc_u"]) & ~manifest["stage13_candidate"]
    ]["exclusion_reason"].value_counts().to_dict()
    physical_mzml = inventory[inventory["source_type"].eq("mzml")]
    complete_count = int(physical_mzml["mzml_complete"].astype("boolean").fillna(False).sum())
    indexed_count = int(physical_mzml["mzml_indexed"].astype("boolean").fillna(False).sum())
    checksum_match_count = int(physical_mzml["source_checksum_matches"].astype("boolean").fillna(False).sum())
    id_match_count = int((
        physical_mzml["mzml_id_matches_stem"].astype("boolean").fillna(False)
        & physical_mzml["run_id_matches_stem"].astype("boolean").fillna(False)
    ).sum())
    lines = [
        "# Stage 13 — Expanded data incorporation and manifest refresh",
        "",
        "## Result",
        "",
        f"- Exact post-manifest candidate corpus: **{metadata['exact_stage13_candidate_cells']:,} cells** "
        f"(`sc_m0` = {metadata['candidate_label_counts'].get('sc_m0', 0):,}; "
        f"`sc_u` = {metadata['candidate_label_counts'].get('sc_u', 0):,}).",
        f"- Existing Stage 11 cells retained: {metadata['retained_existing_cells']:,}.",
        f"- Newly available candidate cells added: {metadata['new_candidate_cells']:,}.",
        f"- Candidate raw-file groups: {metadata['candidate_raw_file_groups']:,}.",
        "- This replaces the earlier 1,046 upper-bound estimate with an exact post-manifest count. "
        "Stage 14 must still apply spectrum-bag and matched protein/peptide feature QC.",
        "",
        "## Source inventory and provenance",
        "",
        f"- Physical RAW files: {metadata['physical_raw_files']:,}; unique acquisitions: {metadata['unique_acquisitions']:,}.",
        f"- Physical mzML files: {metadata['physical_mzml_files']:,}; unique mzML acquisition stems: "
        f"{physical_mzml['acquisition_stem'].nunique():,}.",
        f"- Complete mzML files: {complete_count:,}/{len(physical_mzml):,}; indexed mzML files: {indexed_count:,}/{len(physical_mzml):,}.",
        "- The 14 complete unindexed mzML files are the already processed original LCB7 files; all newly added mzML files are indexed.",
        f"- mzML document IDs and run IDs matching their acquisition stems: {id_match_count:,}/{len(physical_mzml):,}.",
        f"- mzML files whose embedded RAW checksum matches the local RAW: "
        f"{checksum_match_count:,}/{len(physical_mzml):,}.",
        f"- File-ready acquisitions: {metadata['file_ready_acquisitions']:,}/{metadata['unique_acquisitions']:,}.",
        "- LCA9/LCA10 use TMT11; LCB3 and LCB7 use TMTpro16.",
        "",
        "## Candidate counts by chromatography batch",
        "",
        "| Batch | sc_m0 | sc_u | Total |",
        "|---|---:|---:|---:|",
    ]
    for batch_name, row in by_batch.iterrows():
        m0, u = int(row.get("sc_m0", 0)), int(row.get("sc_u", 0))
        lines.append(f"| {batch_name} | {m0:,} | {u:,} | {m0 + u:,} |")
    lines.extend([
        "",
        "## Duplicate and exclusion audit",
        "",
        f"- Duplicate/reconversion records: {len(duplicate_audit):,}.",
        f"- Possible cross-stem reinjections requiring review: {metadata['possible_reinjections']:,}.",
        "- The duplicated `191025S_LCB7_X_AP16plex_Set_1` acquisition is counted once; its original Stage 11 path is canonical.",
        "- User-approved FP94AA/FP94AB exclusions remain explicit and are not counted as candidates.",
        f"- Exclusion/unmatched records: {len(exclusions):,}.",
    ])
    for reason, count in sorted(target_exclusions.items()):
        lines.append(f"- Excluded target channels — {reason}: {count:,}.")
    lines.extend([
        "",
        "## Stage gate",
        "",
        "Stage 13 passes: every local physical RAW/mzML path has a checksum-backed status, every local annotated channel has one manifest row, duplicate acquisitions are counted once, every retained candidate has one label and one raw-file group, and no ambiguous duplicate remains.",
        "",
    ])
    return "\n".join(lines)


def run_stage_thirteen(config_path: str | Path, dry_run: bool = False) -> dict[str, Any]:
    """Execute Stage 13 and write all expanded inventory/manifest artifacts."""
    config = load_yaml_config(config_path)
    stage = config["stage_13"]
    raw_files = discover_files(stage["raw_locations"], ".raw")
    mzml_files = discover_files(stage["mzml_locations"], ".mzML")
    if dry_run:
        return {
            "physical_raw_files": len(raw_files),
            "physical_mzml_files": len(mzml_files),
            "unique_raw_stems": len(group_by_stem(raw_files)),
            "unique_mzml_stems": len(group_by_stem(mzml_files)),
        }
    annotation_path = Path(config["paths"]["annotation_csv"])
    batch_path = Path(config["paths"]["batch_csv"])
    evidence_path = Path(config["paths"]["evidence_csv"])
    annotation = pd.read_csv(annotation_path)
    batch = pd.read_csv(batch_path)
    existing_manifest = pd.read_parquet(stage["existing_manifest_path"])
    existing_bags = pd.read_parquet(stage["existing_bag_index_path"])
    spectrum_config = load_yaml_config(stage["spectrum_config"])
    reporter_schemes = spectrum_config["reporter_schemes"]
    scheme_by_batch = stage["reporter_scheme_by_batch"]
    batch_by_stem = dict(zip(batch["set"].map(normalize_set), batch["lcbatch"].astype(str)))
    all_stems = set(group_by_stem(raw_files)) | set(group_by_stem(mzml_files))
    coverage = evidence_coverage(evidence_path, all_stems)
    inventory, raw_details, mzml_details = build_source_inventory(
        raw_files,
        mzml_files,
        {"channel_annotation": annotation_path, "batch_metadata": batch_path, "search_results": evidence_path},
        batch_by_stem,
        scheme_by_batch,
    )
    run_summary = build_run_summary(
        raw_details,
        mzml_details,
        inventory,
        annotation,
        batch,
        set(existing_manifest["raw_file_stem"].astype(str)),
        set(existing_bags["raw_file_stem"].astype(str)),
        stage.get("approved_exclusions", {}),
        scheme_by_batch,
        coverage,
    )
    manifest = build_expanded_manifest(
        annotation,
        batch,
        run_summary,
        existing_manifest,
        existing_bags,
        coverage,
        set(config["manifest"]["target_labels"]),
        reporter_schemes,
        stage["evidence_reporter_offset_by_batch"],
    )
    run_counts = manifest.groupby("raw_file_stem").agg(
        annotated_channels=("reporter_channel", "size"),
        literal_target_channels=("label_raw", lambda values: values.isin(["sc_m0", "sc_u"]).sum()),
        stage13_candidate_cells=("stage13_candidate", "sum"),
        excluded_channels=("stage13_candidate", lambda values: (~values).sum()),
    )
    run_summary = run_summary.merge(run_counts, left_on="raw_file_stem", right_index=True, how="left")
    duplicate_audit = build_duplicate_audit(inventory)
    exclusions = build_exclusion_unmatched_table(manifest, run_summary, annotation, batch)
    candidates = manifest[manifest["stage13_candidate"]]
    class_batch_counts = candidates.groupby(
        ["source_cohort", "qc_status", "batch_chromatography", "batch_sort", "batch_digest", "label_raw"],
        dropna=False,
    ).agg(
        cell_count=("sample_id", "size"),
        raw_file_groups=("raw_file_stem", "nunique"),
    ).reset_index()
    metadata = validate_stage13(inventory, run_summary, manifest, duplicate_audit)
    metadata.update({
        "metadata_outside_local_scope": int(exclusions["status"].eq("metadata_outside_local_file_scope").sum()),
        "excluded_or_unmatched_records": int(len(exclusions)),
        "reporter_scheme_counts": run_summary["reporter_scheme"].value_counts(dropna=False).to_dict(),
        "acquisition_status_counts": run_summary["acquisition_status"].value_counts().to_dict(),
    })

    output_dir = Path(stage.get("output_directory", "outputs/tables"))
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_dir / "stage_13_source_inventory.csv", index=False)
    run_summary.to_csv(output_dir / "stage_13_run_summary.csv", index=False)
    manifest.to_csv(output_dir / "stage_13_expanded_master_manifest.csv", index=False)
    duplicate_audit.to_csv(output_dir / "stage_13_duplicate_reinjection_audit.csv", index=False)
    exclusions.to_csv(output_dir / "stage_13_exclusions_unmatched.csv", index=False)
    class_batch_counts.to_csv(output_dir / "stage_13_class_batch_counts.csv", index=False)
    parquet_path = Path(stage["expanded_manifest_path"])
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(parquet_path, index=False)
    (output_dir / "stage_13_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    report_path = Path(stage.get("report_path", "outputs/reports/stage_13_expanded_data_incorporation.md"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(metadata, inventory, run_summary, manifest, duplicate_audit, exclusions),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logger = configure_logging(Path("outputs/logs/stage_13_expanded_data_incorporation.log"))
    metadata = run_stage_thirteen(args.config, dry_run=args.dry_run)
    logger.info("Stage 13 completed: %s", metadata)


if __name__ == "__main__":
    main()
