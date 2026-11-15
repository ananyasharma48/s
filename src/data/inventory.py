"""Create a checksum-backed inventory for local proteomics source files."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pymzml

from src.utils.config import load_yaml_config
from src.utils.logging import configure_logging


INVENTORY_COLUMNS = ["relative_path", "filename", "extension", "size_bytes", "sha256", "row_count", "column_count", "column_names_preview", "candidate_role", "parsing_status", "error_message"]
DATA_DICTIONARY_COLUMNS = ["field_name", "source_file", "status", "description"]


def sha256sum(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 checksum without loading a file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_path(path: Path) -> str:
    """Classify a source file conservatively from extension and name."""
    suffix, name = path.suffix.lower(), path.name.lower()
    if suffix == ".raw":
        return "raw_mass_spectrometry"
    if suffix == ".mzml":
        return "mzml"
    if suffix in {".csv", ".tsv", ".txt", ".parquet", ".xlsx"}:
        if any(token in name for token in ("cell", "annotation", "metadata")):
            return "cell_metadata"
        if any(token in name for token in ("channel", "tmt", "reporter")):
            return "channel_annotation"
        if any(token in name for token in ("peptide", "precursor")):
            return "peptide_or_precursor_matrix"
        if "protein" in name:
            return "protein_matrix"
        if any(token in name for token in ("psm", "search", "evidence")):
            return "psm_or_search_result"
    return "unknown"


def inspect_tabular(path: Path) -> tuple[int | None, int | None, str, str, str]:
    """Read only headers and a small sample of a supported tabular file."""
    try:
        if path.suffix.lower() == ".csv":
            sample = pd.read_csv(path, nrows=100)
        elif path.suffix.lower() == ".tsv":
            sample = pd.read_csv(path, sep="\t", nrows=100)
        elif path.suffix.lower() == ".parquet":
            sample = pd.read_parquet(path).head(100)
        else:
            return None, None, "", "not_applicable", ""
        return None, len(sample.columns), ", ".join(map(str, sample.columns[:12])), "parsed_sample", ""
    except Exception as error:
        return None, None, "", "parse_error", f"{type(error).__name__}: {error}"


def inspect_file(path: Path, root: Path) -> dict[str, Any]:
    """Create one inventory row for *path*."""
    row_count, column_count, preview, parsing_status, error_message = inspect_tabular(path)
    if path.suffix.lower() == ".raw":
        parsing_status = "identified_not_fully_parsed"
    elif path.suffix.lower() == ".mzml":
        try:
            next(iter(pymzml.run.Reader(str(path))))
            parsing_status = "parsed_first_spectrum"
        except Exception as error:
            parsing_status = "parse_error"
            error_message = f"{type(error).__name__}: {error}"
    return {"relative_path": str(path.relative_to(root)), "filename": path.name, "extension": path.suffix.lower() or "[no_extension]", "size_bytes": path.stat().st_size, "sha256": sha256sum(path), "row_count": row_count, "column_count": column_count, "column_names_preview": preview, "candidate_role": classify_path(path), "parsing_status": parsing_status, "error_message": error_message}


def build_data_dictionary(inventory: pd.DataFrame) -> pd.DataFrame:
    """Describe expected metadata fields and make unavailable inputs explicit."""
    metadata_present = bool((inventory["candidate_role"] == "cell_metadata").any())
    fields = {"celltype": "literal biological label; retain without expansion until verified", "batch_digest": "digestion batch identifier", "batch_sort": "cell sorting batch identifier", "batch_chromatography": "chromatography batch identifier", "raw.file": "source raw mass-spectrometry file identifier", "tmt_set": "multiplex/TMT experiment identifier", "reporter_channel": "TMT reporter-ion channel assigned to the sample"}
    status = "awaiting_cell_metadata" if not metadata_present else "requires_field_review"
    return pd.DataFrame([{"field_name": field, "source_file": "not available", "status": status, "description": description} for field, description in fields.items()], columns=DATA_DICTIONARY_COLUMNS)


def render_report(inventory: pd.DataFrame, source_root: Path) -> str:
    """Render a concise Markdown inventory report."""
    counts = Counter(inventory["candidate_role"])
    raw_stems = {Path(name).stem for name in inventory.loc[inventory["extension"] == ".raw", "filename"]}
    mzml_stems = {Path(name).stem for name in inventory.loc[inventory["extension"] == ".mzml", "filename"]}
    lines = ["# Stage 1 — Local data inventory", "", "## Scope", "", f"Inventory root: `{source_root}`", "", "This is a scoped inventory of the currently available LCA9 subset. It is not a complete SCoPE2/MassIVE inventory.", "", "## File summary", "", f"- Files inventoried: {len(inventory)}"]
    lines.extend(f"- {role}: {count}" for role, count in sorted(counts.items()))
    lines.extend([f"- Raw files with a matching mzML stem: {len(raw_stems & mzml_stems)}", f"- Raw files without a matching mzML stem: {len(raw_stems - mzml_stems)}", f"- mzML files without a matching raw stem: {len(mzml_stems - raw_stems)}", "", "## Missing inputs", "", "- Cell metadata / `Cells.csv` equivalent: not present.", "- TMT channel-annotation file: not present.", "- Processed protein-by-cell matrix: not present.", "- Processed peptide/precursor-by-cell matrix: not present.", "- PSM/search-result table: not present.", "", "## Consequence", "", "A master sample manifest cannot yet be built, and no raw-file channel can be assigned a biological target label. Do not proceed to Stage 2 until the missing annotations are supplied.", ""])
    return "\n".join(lines)


def run_inventory(source_root: Path, output_dir: Path, dry_run: bool = False) -> pd.DataFrame:
    """Scan every file below *source_root* and emit Stage-1 outputs."""
    if not source_root.is_dir():
        raise NotADirectoryError(f"Inventory root does not exist: {source_root}")
    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No files found below inventory root: {source_root}")
    if dry_run:
        return pd.DataFrame({"relative_path": [str(path.relative_to(source_root)) for path in files]})
    inventory = pd.DataFrame([inspect_file(path, source_root) for path in files], columns=INVENTORY_COLUMNS)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_dir / "data_inventory.csv", index=False)
    build_data_dictionary(inventory).to_csv(output_dir / "data_dictionary.csv", index=False)
    report_dir = output_dir.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "data_inventory.md").write_text(render_report(inventory, source_root), encoding="utf-8")
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--input-root", help="override paths.inventory_root in the YAML config")
    parser.add_argument("--output-dir", default="outputs/tables")
    parser.add_argument("--dry-run", action="store_true", help="list files without calculating checksums or writing outputs")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    source_root = Path(args.input_root or config["paths"]["inventory_root"]).resolve()
    logger = configure_logging(Path("outputs/logs/stage_01_inventory.log"))
    inventory = run_inventory(source_root, Path(args.output_dir), dry_run=args.dry_run)
    logger.info("Inventory %s: %d files", "would scan" if args.dry_run else "completed", len(inventory))


if __name__ == "__main__":
    main()
