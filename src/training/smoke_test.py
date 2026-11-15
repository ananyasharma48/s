"""Stage-0 executable smoke test for package imports and attention MIL."""

import argparse
import importlib
import tempfile
from pathlib import Path

import h5py
import pandas as pd
import torch
from torch import nn

from src.models.attention_mil import AttentionMIL
from src.utils.config import load_yaml_config
from src.utils.seeds import seed_everything


REQUIRED_MODULES = (
    "pandas", "numpy", "scipy", "pyarrow", "sklearn", "torch", "torchmetrics",
    "matplotlib", "joblib", "yaml", "tqdm", "pyteomics", "pymzml", "h5py",
)


def _verify_tabular_io() -> None:
    """Round-trip a tiny matrix through Stage-0 tabular storage formats."""
    matrix = pd.DataFrame({"cell_a": [1.0, 0.0], "cell_b": [0.5, 2.0]}, index=["pep_1", "pep_2"])
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        csv_path = directory / "matrix.csv"
        parquet_path = directory / "matrix.parquet"
        hdf5_path = directory / "matrix.h5"
        matrix.to_csv(csv_path)
        matrix.to_parquet(parquet_path)
        if pd.read_csv(csv_path, index_col=0).shape != matrix.shape:
            raise RuntimeError("CSV round-trip changed matrix shape")
        if pd.read_parquet(parquet_path).shape != matrix.shape:
            raise RuntimeError("Parquet round-trip changed matrix shape")
        with h5py.File(hdf5_path, "w") as handle:
            handle.create_dataset("matrix", data=matrix.to_numpy())
        with h5py.File(hdf5_path, "r") as handle:
            if handle["matrix"].shape != matrix.shape:
                raise RuntimeError("HDF5 round-trip changed matrix shape")


def run_smoke_test(config_path: str) -> float:
    """Execute a deterministic forward and optimizer step; return the loss."""
    config = load_yaml_config(config_path)
    for module_name in REQUIRED_MODULES:
        importlib.import_module(module_name)
    _verify_tabular_io()
    seed_everything(int(config["seed"]))
    model_config = config["model"]
    model = AttentionMIL(**model_config)
    peptide_ids = torch.tensor([1, 5, 8, 2, 3], dtype=torch.long)
    numeric_dimension = int(model_config["numeric_feature_dim"])
    numeric_features = torch.linspace(0.1, 0.9, steps=5 * numeric_dimension).reshape(5, numeric_dimension)
    bag_index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    labels = torch.tensor([0.0, 1.0])
    logits = model(peptide_ids, numeric_features, bag_index)
    if logits.shape != labels.shape:
        raise RuntimeError(f"unexpected logit shape: {tuple(logits.shape)}")
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    training_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if not torch.isfinite(loss):
        raise RuntimeError("smoke-test loss was not finite")
    return float(loss.detach())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mil.yaml")
    args = parser.parse_args()
    loss = run_smoke_test(args.config)
    print(f"Stage 0 smoke test passed (loss={loss:.6f})")


if __name__ == "__main__":
    main()
