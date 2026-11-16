"""Helpers shared by the analysis scripts: loading checkpoints per run, metadata and subsets."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterator

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.builder import ARCHITECTURES, build_model, canonical_architecture, find_checkpoint, load_weights
from src.config import Config
from src.data import SPLIT_FOLDER_ALIASES, make_loaders


def analysis_config(config: Config, architecture: str) -> Config:
    """Copy of ``config`` with ``model.architecture`` set (accepts Hugging Face names)."""
    config = copy.deepcopy(config)
    config.model.architecture = canonical_architecture(architecture)
    return config


def iter_runs(
    config: Config,
    checkpoint_dir: str | Path,
    runs: list[int],
    device: torch.device,
) -> Iterator[tuple[int, torch.nn.Module]]:
    """Yield ``(run_id, model)`` with the best checkpoint of every available run loaded."""
    model = build_model(config, device)
    found = False
    for run in runs:
        path = find_checkpoint(checkpoint_dir, config.model.architecture, run)
        if path is None:
            print(f"[!] No checkpoint for {config.model.architecture} run {run} in {checkpoint_dir}; skipping.")
            continue
        found = True
        checkpoint = load_weights(model, path, device)
        model.eval()
        print(f"\nRun {run}: {path} (epoch {checkpoint.get('epoch')})")
        yield run, model
    if not found:
        raise FileNotFoundError(f"No checkpoint found for {config.model.architecture} in {checkpoint_dir}")


def loader(dataset: Dataset, config: Config) -> DataLoader:
    """Non-shuffled DataLoader with the batch size and workers of ``config``."""
    return make_loaders({"eval": dataset}, config.training.batch_size, config.training.num_workers)["eval"]


def merged(datasets: list[Dataset]) -> ConcatDataset:
    return ConcatDataset([d for d in datasets if len(d) > 0])


def metadata_path(config: Config, split: str, metadata_folder: str | Path | None = None) -> Path:
    """Metadata JSON written by ``samples_generation.metadata_generator_main`` for ``split``."""
    from configs import settings

    prefix = "timeseries" if ARCHITECTURES[config.model.architecture].temporal == "multi" else "simple"
    folder = Path(metadata_folder or settings.METADATA_FOLDER)
    return folder / f"{prefix}_{SPLIT_FOLDER_ALIASES.get(split, split)}_metadata.json"


def load_split_metadata(config: Config, split: str, dataset, metadata_folder=None) -> pd.DataFrame:
    """Metadata of ``split`` aligned row-by-row with ``dataset.files`` (column ``dataset_index``).

    Adds ``max_fire_size`` (largest fire intersecting the tile, in ha) and ``parsed_date``.
    """
    path = metadata_path(config, split, metadata_folder)
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {path}. Generate it with "
            "`python -m samples_generation.metadata_generator_main` (requires the raw fire cubes)."
        )
    df = pd.read_json(path)
    index_of = {p.name: i for i, p in enumerate(dataset.files)}
    df["dataset_index"] = df["filename"].map(index_of)
    missing = df["dataset_index"].isna().sum()
    if missing:
        print(f"[!] {missing} metadata rows of {split} have no matching sample file; ignored.")
    df = df.dropna(subset=["dataset_index"]).astype({"dataset_index": int})
    df["max_fire_size"] = df["all_fire_sizes"].apply(lambda s: max(s) if isinstance(s, list) and s else 0.0)
    df["parsed_date"] = pd.to_datetime(df["start_date" if "start_date" in df.columns else "date"])
    df["fireday"] = pd.to_numeric(df.get("fireday", 1), errors="coerce").fillna(1)
    return df


def summarize(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    """Mean and standard deviation over runs of ``value_cols`` for each group."""
    stats = df.groupby(group_cols, sort=False)[value_cols].agg(["mean", "std"])
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    return stats.reset_index()
