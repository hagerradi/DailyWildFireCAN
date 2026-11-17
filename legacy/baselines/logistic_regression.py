from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import torch
import yaml
from skimage import morphology
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler


CHANNEL_NAMES: tuple[str, ...] = (
    "tmax",
    "rh",
    "ws",
    "prec",
    "u10",
    "v10",
    "evi",
    "ndvi",
    "s2_b02",
    "s2_b03",
    "s2_b04",
    "s2_b08",
    "s2_b11",
    "s2_b12",
    "s2_scl",
    "dem",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "biomass",
    "closure",
    "prcb",
    "prcc",
    "previous_burn_mask",
    "fireday_grid",
)

FEATURE_SETS: dict[str, tuple[int, ...]] = {
    "all_channels": tuple(range(len(CHANNEL_NAMES))),
    "no_satellite": (0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),
    "environment_only": tuple(range(23)),
}

TEST_SPLITS = ("test_easy", "test_medium", "test_hard")


@dataclass
class DataConfig:
    sample_folder: str | None = None
    output_dir: str | None = None
    cache_dir: str | None = None


@dataclass
class FeatureConfig:
    feature_set: str = "all_channels"
    patch_size: int = 3


@dataclass
class SplitFileLimits:
    train: int | None = None
    val: int | None = None
    test: int | None = None


@dataclass
class TrainSamplingConfig:
    max_positive_pixels: int = 500_000
    negative_to_positive_ratio: float = 1.0


@dataclass
class SamplingConfig:
    reuse_cache: bool = True
    max_files_per_split: SplitFileLimits = field(default_factory=SplitFileLimits)
    train: TrainSamplingConfig = field(default_factory=TrainSamplingConfig)


@dataclass
class TrainingConfig:
    w_values: list[float] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    epochs: int = 1
    batch_size: int = 131_072
    alpha: float = 0.0001
    penalty: str = "l2"
    learning_rate: str = "optimal"
    standardize: bool = True


@dataclass
class EvaluationConfig:
    threshold: float = 0.5


@dataclass
class LogisticRegressionConfig:
    seed: int = 42
    run_name: str = "logistic-regression"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LogisticRegressionConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        sampling_raw = raw.get("sampling") or {}
        return cls(
            seed=raw.get("seed", 42),
            run_name=raw.get("run_name", "logistic-regression"),
            data=DataConfig(**(raw.get("data") or {})),
            features=FeatureConfig(**(raw.get("features") or {})),
            sampling=SamplingConfig(
                reuse_cache=sampling_raw.get("reuse_cache", True),
                max_files_per_split=SplitFileLimits(**(sampling_raw.get("max_files_per_split") or {})),
                train=TrainSamplingConfig(**(sampling_raw.get("train") or {})),
            ),
            training=TrainingConfig(**(raw.get("training") or {})),
            evaluation=EvaluationConfig(**(raw.get("evaluation") or {})),
        )


@dataclass
class MatrixMetadata:
    split: str
    matrix_path: str
    labels_path: str
    rows: int
    feature_dim: int
    positive_rows: int
    negative_rows: int
    total_positive_pixels: int
    total_negative_pixels: int
    sampled_positive_pixels: int
    sampled_negative_pixels: int
    files_signature: str


def run_from_config(config_path: str | Path) -> None:
    configure_output_buffering()
    config = LogisticRegressionConfig.from_yaml(config_path)
    validate_config(config)

    run_id = f"{config.run_name}-{int(time.time())}"
    sample_folder = resolve_sample_folder(config.data.sample_folder)
    output_dir = resolve_output_dir(config.data.output_dir, run_id)
    cache_dir = resolve_cache_dir(config.data.cache_dir, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    channel_indices = FEATURE_SETS[config.features.feature_set]
    feature_manifest = build_feature_manifest(channel_indices, config.features.patch_size)

    print(f"Sample folder: {sample_folder}")
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Feature set: {config.features.feature_set} ({len(channel_indices)} channels)")
    print("delta_t is intentionally ignored for this logistic-regression baseline.")

    write_json(output_dir / "resolved_config.json", asdict(config))
    write_json(output_dir / "feature_manifest.json", feature_manifest)

    train_meta = build_sampled_matrix(config, sample_folder, cache_dir, channel_indices)
    scaler = fit_scaler(train_meta, config.training.batch_size) if config.training.standardize else None
    if scaler is not None:
        joblib.dump(scaler, output_dir / "scaler.joblib")

    trained_models: list[SGDClassifier] = []
    trained_w_values: list[float] = []
    for w_value in config.training.w_values:
        print(f"\n--- Training SGD logistic regression with W={w_value} ---")
        model = train_sgd_classifier(config, train_meta, scaler, float(w_value))
        trained_models.append(model)
        trained_w_values.append(float(w_value))

    print("\n--- Evaluating W sweep on full validation grids ---")
    val_metrics_by_w = evaluate_models_on_full_split(
        models=trained_models,
        model_labels=[f"W={w}" for w in trained_w_values],
        scaler=scaler,
        config=config,
        sample_folder=sample_folder,
        split="val",
        channel_indices=channel_indices,
        cache_dir=cache_dir,
    )

    sweep_rows: list[dict[str, Any]] = []
    best_model: SGDClassifier | None = None
    best_w: float | None = None
    best_val_auc_pr = -float("inf")
    best_val_average_precision = -float("inf")
    for w_value, model, val_metrics in zip(trained_w_values, trained_models, val_metrics_by_w):
        row = {"w": w_value, **{f"val_{key}": value for key, value in val_metrics.items()}}
        sweep_rows.append(row)
        print_metrics(f"Validation W={w_value}", val_metrics)
        if val_metrics["auc_pr"] > best_val_auc_pr:
            best_val_auc_pr = float(val_metrics["auc_pr"])
            best_val_average_precision = float(val_metrics["average_precision"])
            best_w = w_value
            best_model = model

    if best_model is None or best_w is None:
        raise RuntimeError("No logistic-regression model was trained.")

    test_metrics: dict[str, dict[str, Any]] = {}
    for split in TEST_SPLITS:
        print(f"\n--- Evaluating selected model on full {split} grids ---")
        test_metrics[split] = evaluate_models_on_full_split(
            models=[best_model],
            model_labels=[f"W={best_w}"],
            scaler=scaler,
            config=config,
            sample_folder=sample_folder,
            split=split,
            channel_indices=channel_indices,
            cache_dir=cache_dir,
        )[0]

    print(f"\nSelected W={best_w} by validation AUC-PR={best_val_auc_pr:.6f}")
    for split, metrics in test_metrics.items():
        print_metrics(split, metrics)

    write_csv(output_dir / "w_sweep_metrics.csv", sweep_rows)
    write_json(output_dir / "w_sweep_metrics.json", sweep_rows)
    write_json(
        output_dir / "final_metrics.json",
        {
            "selected_w": best_w,
            "validation_auc_pr": best_val_auc_pr,
            "validation_average_precision": best_val_average_precision,
            "test_metrics": test_metrics,
            "train_matrix": asdict(train_meta),
            "evaluation_protocol": "full_256x256_grid_streaming",
        },
    )
    joblib.dump(best_model, output_dir / "model.joblib")
    print(f"\nSaved logistic-regression artifacts to {output_dir}")


def validate_config(config: LogisticRegressionConfig) -> None:
    if config.features.feature_set not in FEATURE_SETS:
        known = ", ".join(sorted(FEATURE_SETS))
        raise ValueError(f"Unknown feature_set '{config.features.feature_set}'. Expected one of: {known}")
    if config.features.patch_size < 1 or config.features.patch_size % 2 == 0:
        raise ValueError("features.patch_size must be a positive odd integer.")
    if config.sampling.train.max_positive_pixels < 1:
        raise ValueError("sampling.train.max_positive_pixels must be positive.")
    if config.sampling.train.negative_to_positive_ratio <= 0:
        raise ValueError("sampling.train.negative_to_positive_ratio must be positive.")
    if config.training.batch_size < 1:
        raise ValueError("training.batch_size must be positive.")
    if config.training.epochs < 1:
        raise ValueError("training.epochs must be positive.")
    if not config.training.w_values:
        raise ValueError("training.w_values cannot be empty.")


def configure_output_buffering() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)


def resolve_sample_folder(configured: str | None) -> Path:
    raw = configured or os.environ.get("WILDFIRE_SAMPLE_FOLDER")
    if raw is None:
        from configs import settings

        raw = settings.SAMPLE_FOLDER
    path = Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Sample folder does not exist: {path}")
    for split in ("train", "val", *TEST_SPLITS):
        split_path = path / split
        if not split_path.exists():
            raise FileNotFoundError(f"Expected split folder does not exist: {split_path}")
    return path


def resolve_output_dir(configured: str | None, run_id: str) -> Path:
    if configured:
        return Path(os.path.expandvars(os.path.expanduser(configured))).resolve()
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch).expanduser().resolve() / "canada-wildfire-daily" / "logreg_outputs" / run_id
    return (Path("logistic_regression_outputs") / run_id).resolve()


def resolve_cache_dir(configured: str | None, run_id: str) -> Path:
    if configured:
        return Path(os.path.expandvars(os.path.expanduser(configured))).resolve()
    slurm_tmp = os.environ.get("SLURM_TMP_DIR")
    if slurm_tmp:
        return Path(slurm_tmp).expanduser().resolve() / "canada-wildfire-daily" / "logreg_cache" / run_id
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch).expanduser().resolve() / "canada-wildfire-daily" / "logreg_cache" / run_id
    return (Path("logistic_regression_cache") / run_id).resolve()


def build_feature_manifest(channel_indices: Iterable[int], patch_size: int) -> dict[str, Any]:
    pad = patch_size // 2
    flattened_features = []
    for idx in channel_indices:
        for row_offset in range(-pad, pad + 1):
            for col_offset in range(-pad, pad + 1):
                flattened_features.append(
                    {
                        "channel_index": idx,
                        "channel_name": CHANNEL_NAMES[idx],
                        "row_offset": row_offset,
                        "col_offset": col_offset,
                    }
                )
    return {
        "channel_order": list(CHANNEL_NAMES),
        "selected_channels": [{"index": idx, "name": CHANNEL_NAMES[idx]} for idx in channel_indices],
        "patch_size": patch_size,
        "flattened_features": flattened_features,
        "uses_delta_t": False,
    }


def build_sampled_matrix(
    config: LogisticRegressionConfig,
    sample_folder: Path,
    cache_dir: Path,
    channel_indices: tuple[int, ...],
) -> MatrixMetadata:
    split = "train"
    rng = np.random.default_rng(stable_seed(config.seed, split))
    files = split_files(sample_folder, split, config.sampling.max_files_per_split.train)
    signature = files_signature(files, channel_indices, config.features.patch_size, config)
    matrix_prefix = f"{split}_{config.features.feature_set}_p{config.features.patch_size}_{signature[:12]}"
    meta_path = cache_dir / f"{matrix_prefix}_metadata.json"

    if config.sampling.reuse_cache and meta_path.exists():
        raw_meta = read_json(meta_path)
        matrix_path = Path(raw_meta["matrix_path"])
        labels_path = Path(raw_meta["labels_path"])
        if matrix_path.exists() and labels_path.exists():
            print(f"Reusing cached train matrix: {matrix_path}")
            return MatrixMetadata(**raw_meta)

    target_pos = config.sampling.train.max_positive_pixels
    target_neg = int(math.ceil(target_pos * config.sampling.train.negative_to_positive_ratio))
    rows_capacity = target_pos + target_neg
    feature_dim = len(channel_indices) * config.features.patch_size * config.features.patch_size
    matrix_path = cache_dir / f"{matrix_prefix}_X.float32.memmap"
    labels_path = cache_dir / f"{matrix_prefix}_y.uint8.memmap"
    X = np.memmap(matrix_path, dtype=np.float32, mode="w+", shape=(rows_capacity, feature_dim))
    y = np.memmap(labels_path, dtype=np.uint8, mode="w+", shape=(rows_capacity,))

    print(
        f"Building train matrix from up to {len(files)} files until "
        f"{target_pos} fire and {target_neg} no-fire pixels are sampled."
    )

    cursor = 0
    sampled_pos = 0
    sampled_neg = 0
    seen_pos = 0
    seen_neg = 0
    files_scanned = 0

    for ordered_idx, file_idx in enumerate(rng.permutation(len(files)), start=1):
        if sampled_pos >= target_pos and sampled_neg >= target_neg:
            break

        x_tensor, y_mask = load_sample(files[int(file_idx)])
        pos_available = int(y_mask.sum())
        neg_available = int(y_mask.size - pos_available)
        seen_pos += pos_available
        seen_neg += neg_available
        files_scanned += 1

        n_pos = min(pos_available, target_pos - sampled_pos)
        n_neg = min(neg_available, target_neg - sampled_neg)
        if n_pos == 0 and n_neg == 0:
            continue

        row_chunks: list[np.ndarray] = []
        col_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []
        if n_pos > 0:
            rows_pos, cols_pos = choose_coords(y_mask, True, n_pos, rng)
            row_chunks.append(rows_pos)
            col_chunks.append(cols_pos)
            label_chunks.append(np.ones(n_pos, dtype=np.uint8))
        if n_neg > 0:
            rows_neg, cols_neg = choose_coords(y_mask, False, n_neg, rng)
            row_chunks.append(rows_neg)
            col_chunks.append(cols_neg)
            label_chunks.append(np.zeros(n_neg, dtype=np.uint8))

        sample_rows = np.concatenate(row_chunks)
        sample_cols = np.concatenate(col_chunks)
        sample_labels = np.concatenate(label_chunks)
        order = rng.permutation(sample_labels.size)
        sample_rows = sample_rows[order]
        sample_cols = sample_cols[order]
        sample_labels = sample_labels[order]

        features = extract_patches(x_tensor, sample_rows, sample_cols, channel_indices, config.features.patch_size)
        next_cursor = cursor + sample_labels.size
        X[cursor:next_cursor] = features
        y[cursor:next_cursor] = sample_labels
        cursor = next_cursor
        sampled_pos += n_pos
        sampled_neg += n_neg

        if ordered_idx == 1 or ordered_idx % 25 == 0 or (sampled_pos >= target_pos and sampled_neg >= target_neg):
            print(
                f"train: scanned {files_scanned} files, "
                f"sampled fire={sampled_pos}/{target_pos}, no-fire={sampled_neg}/{target_neg}"
            )

    if sampled_pos == 0 or sampled_neg == 0:
        raise ValueError("Training sampling did not collect both classes.")

    X.flush()
    y.flush()
    metadata = MatrixMetadata(
        split=split,
        matrix_path=str(matrix_path),
        labels_path=str(labels_path),
        rows=cursor,
        feature_dim=feature_dim,
        positive_rows=sampled_pos,
        negative_rows=sampled_neg,
        total_positive_pixels=seen_pos,
        total_negative_pixels=seen_neg,
        sampled_positive_pixels=sampled_pos,
        sampled_negative_pixels=sampled_neg,
        files_signature=signature,
    )
    write_json(meta_path, asdict(metadata))
    return metadata


def split_files(sample_folder: Path, split: str, max_files: int | None) -> list[Path]:
    split_dir = sample_folder / split
    files = sorted([*split_dir.glob("*.npz"), *split_dir.glob("*.pt")])
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No .npz or .pt files found for split '{split}' in {split_dir}")
    return files


def files_signature(
    files: list[Path],
    channel_indices: tuple[int, ...],
    patch_size: int,
    config: LogisticRegressionConfig,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(",".join(CHANNEL_NAMES[idx] for idx in channel_indices).encode("utf-8"))
    hasher.update(f"|patch={patch_size}|seed={config.seed}".encode("utf-8"))
    hasher.update(
        (
            f"|train_pos={config.sampling.train.max_positive_pixels}"
            f"|train_ratio={config.sampling.train.negative_to_positive_ratio}"
        ).encode("utf-8")
    )
    for file_path in files:
        stat = file_path.stat()
        hasher.update(f"\n{file_path}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8"))
    return hasher.hexdigest()


def load_sample(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    if path.suffix == ".npz":
        with np.load(path) as data:
            x_np = np.asarray(data["x"], dtype=np.float32)
            y_np = np.asarray(data["y"])
    else:
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except (RuntimeError, ValueError, EOFError, pickle.UnpicklingError, zipfile.BadZipFile):
            with np.load(path) as data:
                x_np = np.asarray(data["x"], dtype=np.float32)
                y_np = np.asarray(data["y"])
        else:
            x_np = data["x"].detach().cpu().numpy().astype(np.float32, copy=False)
            y_np = data["y"].detach().cpu().numpy()

    if x_np.ndim != 3:
        raise ValueError(f"Expected a simple sample with shape (channels, height, width), got {x_np.shape} in {path}")

    y_mask = np.asarray(y_np > 0, dtype=bool)
    y_mask = morphology.remove_small_holes(y_mask, area_threshold=1)
    return torch.from_numpy(x_np).float(), y_mask.astype(bool, copy=False)


def choose_coords(mask: np.ndarray, positive: bool, n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(mask if positive else ~mask)
    if n_samples > rows.size:
        raise ValueError(f"Requested {n_samples} pixels, but only {rows.size} are available.")
    chosen = rng.choice(rows.size, size=n_samples, replace=False)
    return rows[chosen].astype(np.int64, copy=False), cols[chosen].astype(np.int64, copy=False)


def extract_patches(
    x_tensor: torch.Tensor,
    rows: np.ndarray,
    cols: np.ndarray,
    channel_indices: tuple[int, ...],
    patch_size: int,
) -> np.ndarray:
    if max(channel_indices) >= x_tensor.shape[0]:
        raise ValueError(
            f"Selected channel index {max(channel_indices)} is out of bounds for sample with "
            f"{x_tensor.shape[0]} channels. Check the configured sample folder and feature set."
        )
    selected = x_tensor[list(channel_indices)].numpy().astype(np.float32, copy=False)
    pad = patch_size // 2
    padded = np.pad(selected, ((0, 0), (pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
    patches = np.empty((rows.size, len(channel_indices), patch_size, patch_size), dtype=np.float32)
    for patch_row in range(patch_size):
        for patch_col in range(patch_size):
            patches[:, :, patch_row, patch_col] = padded[:, rows + patch_row, cols + patch_col].T
    features = patches.reshape(rows.size, -1)
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def fit_scaler(meta: MatrixMetadata, batch_size: int) -> StandardScaler:
    print("\nFitting StandardScaler on sampled training rows...")
    scaler = StandardScaler()
    for X_batch, _ in iter_matrix_batches(meta, batch_size, shuffle=False):
        scaler.partial_fit(X_batch)
    return scaler


def train_sgd_classifier(
    config: LogisticRegressionConfig,
    train_meta: MatrixMetadata,
    scaler: StandardScaler | None,
    w_value: float,
) -> SGDClassifier:
    y_train = open_labels(train_meta)
    n_no_fire = int((y_train == 0).sum())
    n_fire = int((y_train == 1).sum())
    n_total = n_no_fire + n_fire
    no_fire_weight = (n_total / n_no_fire) * w_value
    fire_weight = n_total / n_fire

    model = SGDClassifier(
        loss="log_loss",
        penalty=config.training.penalty,
        alpha=config.training.alpha,
        learning_rate=config.training.learning_rate,
        random_state=config.seed,
    )
    rng = np.random.default_rng(stable_seed(config.seed, "train", w_value))
    classes = np.array([0, 1], dtype=np.uint8)
    first_batch = True

    for epoch in range(1, config.training.epochs + 1):
        print(f"Epoch {epoch}/{config.training.epochs}")
        for X_batch, y_batch in iter_matrix_batches(train_meta, config.training.batch_size, shuffle=True, rng=rng):
            if scaler is not None:
                X_batch = scaler.transform(X_batch)
            sample_weights = np.where(y_batch == 0, no_fire_weight, fire_weight)
            if first_batch:
                model.partial_fit(X_batch, y_batch, classes=classes, sample_weight=sample_weights)
                first_batch = False
            else:
                model.partial_fit(X_batch, y_batch, sample_weight=sample_weights)
    return model


def evaluate_models_on_full_split(
    models: list[SGDClassifier],
    model_labels: list[str],
    scaler: StandardScaler | None,
    config: LogisticRegressionConfig,
    sample_folder: Path,
    split: str,
    channel_indices: tuple[int, ...],
    cache_dir: Path,
) -> list[dict[str, float]]:
    if len(models) != len(model_labels):
        raise ValueError("models and model_labels must have the same length.")

    # All test splits share the `test` file limit.
    limit_key = "test" if split.startswith("test") else split
    files = split_files(sample_folder, split, getattr(config.sampling.max_files_per_split, limit_key))
    first_x, first_mask = load_sample(files[0])
    height, width = first_mask.shape
    pixels_per_sample = height * width
    total_rows = len(files) * pixels_per_sample
    signature = files_signature(files, channel_indices, config.features.patch_size, config)
    prefix = f"{split}_full_{config.features.feature_set}_p{config.features.patch_size}_{signature[:12]}"
    labels_path = cache_dir / f"{prefix}_labels.uint8.memmap"
    scores_path = cache_dir / f"{prefix}_scores.float32.memmap"

    label_memmap = np.memmap(labels_path, dtype=np.uint8, mode="w+", shape=(total_rows,))
    score_memmap = np.memmap(scores_path, dtype=np.float32, mode="w+", shape=(len(models), total_rows))
    row_coords, col_coords = np.indices((height, width), dtype=np.int64)
    row_coords = row_coords.reshape(-1)
    col_coords = col_coords.reshape(-1)

    true_positive = np.zeros(len(models), dtype=np.float64)
    false_positive = np.zeros(len(models), dtype=np.float64)
    false_negative = np.zeros(len(models), dtype=np.float64)
    true_negative = np.zeros(len(models), dtype=np.float64)
    total_positive = 0
    total_negative = 0
    cursor = 0

    print(
        f"Evaluating {len(models)} model(s) on full {split} grids: "
        f"{len(files)} files x {height}x{width} = {total_rows} pixels."
    )

    for file_idx, file_path in enumerate(files, start=1):
        x_tensor, y_mask = (first_x, first_mask) if file_idx == 1 else load_sample(file_path)
        if y_mask.shape != (height, width):
            raise ValueError(f"Unexpected mask shape in {file_path}: {y_mask.shape}, expected {(height, width)}")

        features = extract_patches(x_tensor, row_coords, col_coords, channel_indices, config.features.patch_size)
        if scaler is not None:
            features = scaler.transform(features)
        scores = predict_probabilities(models, features)
        y_flat = y_mask.reshape(-1).astype(np.uint8, copy=False)
        next_cursor = cursor + pixels_per_sample
        label_memmap[cursor:next_cursor] = y_flat
        score_memmap[:, cursor:next_cursor] = scores.T

        positives = y_flat.astype(bool)
        predictions = scores >= config.evaluation.threshold
        true_positive += (predictions & positives[:, None]).sum(axis=0)
        false_positive += (predictions & ~positives[:, None]).sum(axis=0)
        false_negative += (~predictions & positives[:, None]).sum(axis=0)
        true_negative += (~predictions & ~positives[:, None]).sum(axis=0)
        total_positive += int(positives.sum())
        total_negative += int((~positives).sum())
        cursor = next_cursor

        if file_idx == 1 or file_idx % 100 == 0 or file_idx == len(files):
            print(
                f"{split}/full-grid: processed {file_idx}/{len(files)} files, "
                f"pixels={cursor}/{total_rows}, fire_pixels={total_positive}, no_fire_pixels={total_negative}"
            )

    label_memmap.flush()
    score_memmap.flush()
    label_read = np.memmap(labels_path, dtype=np.uint8, mode="r", shape=(total_rows,))
    score_read = np.memmap(scores_path, dtype=np.float32, mode="r", shape=(len(models), total_rows))

    metrics_by_model: list[dict[str, float]] = []
    for model_idx, model_label in enumerate(model_labels):
        print(f"{split}/full-grid: computing exact PR AUC/AP for {model_label} over {total_rows} pixels...")
        auc_pr, average_precision = precision_recall_metrics(label_read, score_read[model_idx])
        metrics_by_model.append(
            {
                "auc_pr": auc_pr,
                "average_precision": average_precision,
                **binary_metrics_from_counts(
                    true_positive[model_idx],
                    false_positive[model_idx],
                    false_negative[model_idx],
                    true_negative[model_idx],
                ),
                "rows": float(total_rows),
                "positive_rows": float(total_positive),
                "negative_rows": float(total_negative),
            }
        )
    return metrics_by_model


def predict_probabilities(models: list[SGDClassifier], features: np.ndarray) -> np.ndarray:
    coefficients = np.vstack([model.coef_[0] for model in models]).astype(np.float32, copy=False)
    intercepts = np.array([model.intercept_[0] for model in models], dtype=np.float32)
    logits = features @ coefficients.T + intercepts
    return sigmoid(logits).astype(np.float32, copy=False)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    probabilities = np.empty_like(logits, dtype=np.float32)
    positive = logits >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)
    return probabilities


def iter_matrix_batches(
    meta: MatrixMetadata,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator | None = None,
):
    X = np.memmap(meta.matrix_path, dtype=np.float32, mode="r", shape=(meta.rows, meta.feature_dim))
    y = open_labels(meta)
    if shuffle:
        if rng is None:
            raise ValueError("A numpy random generator is required when shuffle=True.")
        indices = rng.permutation(meta.rows)
        for start in range(0, meta.rows, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield X[batch_idx], y[batch_idx]
    else:
        for start in range(0, meta.rows, batch_size):
            stop = min(start + batch_size, meta.rows)
            yield X[start:stop], y[start:stop]


def open_labels(meta: MatrixMetadata) -> np.memmap:
    return np.memmap(meta.labels_path, dtype=np.uint8, mode="r", shape=(meta.rows,))


def precision_recall_metrics(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    if int(y_true.min()) == int(y_true.max()):
        return 0.0, 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc_pr = float(auc(recall, precision))
    average_precision = float(-np.sum(np.diff(recall) * precision[:-1]))
    return auc_pr, average_precision


def binary_metrics_from_counts(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
    iou = safe_divide(tp, tp + fp + fn)
    accuracy = safe_divide(tp + tn, tp + fp + fn + tn)
    return {
        "iou": iou,
        "dice_f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def stable_seed(*parts: Any) -> int:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"|")
    return int.from_bytes(hasher.digest()[:8], "little") % (2**32)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def read_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    metric_order = ("auc_pr", "average_precision", "iou", "dice_f1", "precision", "recall", "accuracy")
    rendered = " | ".join(f"{name}={metrics[name]:.6f}" for name in metric_order)
    print(f"{label}: {rendered}")
