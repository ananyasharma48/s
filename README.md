# Single-Cell Proteomics MIL

Reproducible, leakage-aware classification of `sc_u` and `sc_m0` cells from
single-cell mass-spectrometry proteomics data.

## Stage 0 quick start

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
python -m src.training.smoke_test --config configs/mil.yaml
```

The smoke test validates the core runtime, creates two variable-size dummy
peptide bags, executes an attention-MIL forward pass, and completes one
optimization step.

## Raw-file conversion

Raw Thermo files are not converted during Stage 0. For the later raw-spectrum
pilot, use ThermoRawFileParser (cross-platform) or ProteoWizard `msconvert`.
Record the exact conversion command, tool version, source checksum, and output
checksum in the conversion log. Conversion outputs belong in
`data/massive_mzml/`; source `.raw` files remain immutable in
`data/massive_raw/`.

## Layout

- `configs/` — versioned YAML configuration
- `data/` — ignored source and derived data locations
- `src/` — package source code
- `tests/` — unit and smoke tests
- `outputs/` — ignored reports, tables, checkpoints, figures, and logs

Later stages must create a data inventory and master manifest before training
on real measurements.

## Stage 7 peptide MIL dataset

After Stages 1–6 have passed, build the variable-length peptide bags with:

```bash
python -m src.data.build_peptide_bags --config configs/mil.yaml
```

The command writes a peptide vocabulary, one contiguous instance table, and a
bag index to `data/processed/mil_bags/`. Use `--dry-run` to validate and
summarize the inputs without writing artifacts.

## Stage 8 peptide MIL models

After the Stage 7 reconstruction checks pass, train the grouped-fold MIL
models with:

```bash
python -m src.training.train_mil --config configs/mil.yaml
```

The command uses a saved grouped inner fold for early stopping, retrains on
each complete outer-training partition, and evaluates mean pooling, max
pooling, and attention MIL on the untouched outer test partition. Use
`--dry-run` to validate all artifacts and execute one real-data forward pass.

## Stage 9 primary evaluation

After Stage 8, run the group-aware statistical evaluation with:

```bash
python -m src.evaluation.evaluate --config configs/evaluation.yaml
```

This evaluates aligned out-of-fold predictions, resamples complete raw-file
groups for confidence intervals, checks calibration, compares MIL models with
peptide logistic regression, and reports subgroup robustness and cell-level
errors. Use `--dry-run` to validate prediction alignment without running the
bootstrap.

## Stage 10 raw-file pilot

After Stage 9, validate the supplied FP94AJ RAW/mzML pair and extract its TMT
reporter ions with:

```bash
python -m src.data.extract_reporter_ions --config configs/spectrum_mil.yaml
```

The command verifies conversion provenance, parses MS1/MS2 scans, extracts the
configured TMT11 reporter windows, joins channels to the master manifest,
checks the intensities against the MaxQuant search results, and saves spectra,
reporter tables, example plots, and a Stage 10 report. Use `--dry-run` to check
all input paths, mzML header metadata, reporter configuration, and manifest
coverage without parsing the full file.

## Stage 11 full raw-spectrum preprocessing

After the Stage 10 pilot passes, preprocess every configured mzML run with:

```bash
python -m src.data.build_spectrum_bags --config configs/spectrum_mil.yaml
```

The command verifies each RAW/mzML pair, filters MS2 spectra, extracts TMT11 or
TMTpro16 signals per run, stores one normalized binned fragment representation
per shared spectrum, and builds channel-specific spectrum bags for valid cells.
The configured FP94AA/FP94AB exclusions are retained in the processing log.

## Stage 12 spectrum-aware MIL

Train reporter-only, binned-spectrum, and spectrum-plus-PSM attention models
with grouped nested evaluation using:

```bash
python -m src.training.train_spectrum_mil --config configs/spectrum_mil.yaml
```

Training reads only the cached Stage 11 HDF5/Parquet artifacts. It evaluates
128, 256, 512, and 1,024 spectrum bag limits, records runtime and memory,
compares against processed-data models on common cells, and saves spectrum
attention outputs. Use `--dry-run` for real-data forward and split checks.

## Stage 13 expanded data incorporation

Refresh the checksum-backed inventory and build a deduplicated manifest across
the original and newly available RAW/mzML acquisitions with:

```bash
python -m src.data.expand_corpus --config configs/data.yaml
```

The command verifies indexed mzML completeness, embedded RAW checksums,
converter provenance, TMT11/TMTpro16 assignments, duplicate physical copies,
annotation/batch coverage, and every reporter-channel disposition. It preserves
the Stage 11 corpus and writes a separate expanded manifest for Stage 14.

## Stage 14 matched multimodal corpus

Extend the spectrum stores, reconstruct peptide and protein abundance vectors,
match the modalities, and rebuild group-aware splits with:

```bash
python -m src.data.build_multimodal_corpus --config configs/spectrum_mil.yaml
```

## Stage 15 expanded corpus model retraining

Retrain every conventional, peptide-MIL, and spectrum-MIL model on the matched
1,046-cell corpus with shared raw-file-grouped outer folds. Neural models run
with three deterministic seeds, and the pipeline writes resumable checkpoints,
out-of-fold predictions, calibration and subgroup metrics, group-bootstrap
confidence intervals, paired comparisons, figures, and a final report.

```bash
python -m src.training.train_expanded_models --config configs/stage_15.yaml
```

## Stage 16 multimodal protein plus spectrum MIL

Compare the Stage 15 protein and spectrum references with validation-weighted
late fusion, joint embedding fusion, and a protein-initialized gated residual
MIL model. All models use the Stage 14 aligned cells and grouped folds, with
three deterministic neural seeds and paired raw-file bootstrap comparisons.

```bash
python -m src.training.train_multimodal_mil --config configs/stage_16.yaml
```

## Stage 17 group-aware learning curves

Measure data sufficiency at 200, 400, 600, 800, and all 1,046 cells with
deterministic whole-raw-file cohorts, repeated subsampling, and identical nested
grouped folds across all Stage 15 models and the primary Stage 16 fusion models:

```bash
scripts/run_stage_17_learning_curves.sh
```

The outputs separate empirical learning-curve points from descriptive
extrapolations and include runtime scaling, group-bootstrap intervals, and
paired comparisons against peptide logistic regression.

The command copies the immutable Stage 11 spectrum stores, appends only novel
acquisitions, joins search-result PSM identities to mzML-extracted RI1–RI16
signals, creates matched peptide/protein matrices and spectrum bags, records all
cell-level exclusions, and writes new raw-file-grouped outer and inner folds.
