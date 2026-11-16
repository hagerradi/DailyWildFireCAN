"""Datasets and dataloaders for the released DailyWildfireCAN samples.

A samples folder (``settings.SAMPLE_FOLDER`` for mono-temporal samples,
``settings.TIMESERIES_SAMPLE_FOLDER`` for multi-temporal ones) contains one
sub-folder per split with ``sample_<idx>.npz`` files::

    <samples_folder>/
        train/  val/  test_easy/  test_medium/  test_hard/

The test splits extracted from the Hugging Face tarballs are named ``test_time``
(Test-Easy), ``test_space`` (Test-Medium) and ``test_spacetime`` (Test-Hard); both
naming schemes are accepted.

Each ``.npz`` file stores ``x`` (``(C, H, W)`` or ``(T, C, H, W)``, already
normalised), ``y`` (binary next-day new-burn mask), ``delta_t`` (days between the
Sentinel-2 acquisition and the prediction day), ``coords`` (lon, lat),
``fire_date`` and, for multi-temporal samples, ``positions``.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import torch
from skimage import morphology
from torch.utils.data import DataLoader, Dataset

from src.features import (
    DYNAMIC_FEATURES,
    FEATURE_NAMES,
    RGB_FEATURES,
    STATE_FEATURES,
    STATIC_FEATURES,
)

SPLITS = ("train", "val", "test_easy", "test_medium", "test_hard")
TEST_SPLITS = ("test_easy", "test_medium", "test_hard")
SPLIT_LABELS = {
    "train": "Train",
    "val": "Validation",
    "test_easy": "Test-Easy",
    "test_medium": "Test-Medium",
    "test_hard": "Test-Hard",
}
# Folder names produced by the sample generator and contained in the released tarballs.
SPLIT_FOLDER_ALIASES = {
    "test_easy": "test_time",
    "test_medium": "test_space",
    "test_hard": "test_spacetime",
}

# Input layouts produced by :class:`WildfireDataset` (see ``layout`` argument).
LAYOUTS = ("grids", "rgb_env", "branches")


def resolve_split_dir(samples_folder: str | Path, split: str) -> Path:
    """Return the folder holding ``split`` inside ``samples_folder``.

    Accepts both the paper names (``test_easy``/``test_medium``/``test_hard``) and the
    folder names found in the released tarballs (``test_time``/``test_space``/
    ``test_spacetime``). Returns the paper-named path if neither exists.
    """
    base = Path(samples_folder)
    candidates = [base / split]
    if split in SPLIT_FOLDER_ALIASES:
        candidates.append(base / SPLIT_FOLDER_ALIASES[split])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _natural_key(path: Path) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def list_samples(split_dir: str | Path) -> list[Path]:
    """List the ``.npz`` samples of a split folder in natural order (sample_2 < sample_10)."""
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        return []
    return sorted(split_dir.glob("*.npz"), key=_natural_key)


def fill_label_holes(y: np.ndarray) -> np.ndarray:
    """Binarise a label mask and fill single-pixel holes left by the VIIRS reprojection."""
    return morphology.remove_small_holes(y > 0, area_threshold=1).astype(np.float32)


class WildfireDataset(Dataset):
    """Reads the ``.npz`` samples of one split.

    Args:
        split_dir: Folder containing the ``sample_<idx>.npz`` files.
        layout: How the 50 input channels are returned:

            * ``"grids"``: a single ``input_grids`` tensor (``(C, H, W)`` or ``(T, C, H, W)``).
            * ``"rgb_env"``: ``input_rgb`` (Sentinel-2 B02/B03/B04) and ``input_env``
              (the 47 other channels), used by the dual DINOv3 model.
            * ``"branches"``: ``state``, ``dynamic`` and ``constant`` tensors, used by the
              multi-modal model. Static channels are taken from the first time step.
        extra_keys: Optional ``.npz`` keys to return: ``"delta_t"`` (returned as
            ``satellite_age``), ``"positions"``, ``"coords"``, ``"olmo_emb"``.
        label_variant: Optional alternative target used by the label-variation analysis
            (Section 5.4 of the paper), read from ``<split_dir>_temporal_labels/``
            (written by ``samples_generation.data_generator.generate_raw_temporal_labels``):

            * ``"spatial"``: the next-day target dilated by one pixel (``y_dilated``).
            * ``"temporal"``: union of the burned areas of days T, T+1 and T+2
              (``y_minus1``, ``y``, ``y_plus1``).

    Every item is a dict with the input tensors, ``label`` (``(H, W)`` float) and the
    requested extra keys.
    """

    _EXTRA_KEY_NAMES = {"delta_t": "satellite_age"}

    def __init__(
        self,
        split_dir: str | Path,
        layout: str = "grids",
        extra_keys: tuple[str, ...] = (),
        label_variant: str | None = None,
    ) -> None:
        if layout not in LAYOUTS:
            raise ValueError(f"Unknown layout '{layout}', expected one of {LAYOUTS}")
        self.split_dir = Path(split_dir)
        self.layout = layout
        self.extra_keys = tuple(extra_keys)
        if label_variant not in (None, "spatial", "temporal"):
            raise ValueError(f"Unknown label_variant '{label_variant}'")
        self.label_variant = label_variant
        self.label_dir = self.split_dir.parent / f"{self.split_dir.name}_temporal_labels"
        self.files = list_samples(self.split_dir)

        idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
        self.rgb_idx = [idx[f] for f in RGB_FEATURES]
        self.env_idx = [i for i in range(len(FEATURE_NAMES)) if i not in self.rgb_idx]
        self.state_idx = [idx[f] for f in STATE_FEATURES]
        self.dynamic_idx = [idx[f] for f in DYNAMIC_FEATURES]
        self.static_idx = [idx[f] for f in STATIC_FEATURES]

    def __len__(self) -> int:
        return len(self.files)

    def _split_inputs(self, x: np.ndarray) -> dict[str, torch.Tensor]:
        def t(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a)).float()

        if self.layout == "grids":
            return {"input_grids": t(x)}
        if self.layout == "rgb_env":
            return {"input_rgb": t(x[..., self.rgb_idx, :, :]), "input_env": t(x[..., self.env_idx, :, :])}
        static = x[0, self.static_idx] if x.ndim == 4 else x[self.static_idx]
        return {
            "state": t(x[..., self.state_idx, :, :]),
            "dynamic": t(x[..., self.dynamic_idx, :, :]),
            "constant": t(static),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.files[idx]
        with np.load(path) as data:
            x = data["x"]
            y = data["y"]
            extras = {k: data[k] for k in self.extra_keys if k in data}

        if self.label_variant is not None:
            with np.load(self.label_dir / path.name) as labels:
                if self.label_variant == "spatial":
                    y = labels["y_dilated"]
                else:
                    y = np.maximum.reduce([labels["y_minus1"], y, labels["y_plus1"]])

        sample = self._split_inputs(x)
        sample["label"] = torch.from_numpy(fill_label_holes(y))
        for key, value in extras.items():
            sample[self._EXTRA_KEY_NAMES.get(key, key)] = torch.from_numpy(np.asarray(value)).float()
        return sample


def build_datasets(
    samples_folder: str | Path,
    splits: tuple[str, ...] = SPLITS,
    **dataset_kwargs,
) -> dict[str, WildfireDataset]:
    """Create one :class:`WildfireDataset` per split. Missing splits get an empty dataset."""
    datasets = {}
    for split in splits:
        split_dir = resolve_split_dir(samples_folder, split)
        datasets[split] = WildfireDataset(split_dir, **dataset_kwargs)
        if len(datasets[split]) == 0:
            warnings.warn(f"No samples found for split '{split}' in {split_dir}", stacklevel=2)
    return datasets


def make_loaders(
    datasets: dict[str, Dataset],
    batch_size: int,
    num_workers: int = 4,
) -> dict[str, DataLoader | None]:
    """Wrap datasets in DataLoaders (only ``train`` is shuffled); empty splits map to ``None``."""
    loaders = {}
    for split, dataset in datasets.items():
        if len(dataset) == 0:
            loaders[split] = None
            continue
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
