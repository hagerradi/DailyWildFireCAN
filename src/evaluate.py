"""Evaluate trained checkpoints on the test splits and write the metrics to a CSV.

Works with checkpoints written by ``scripts/parallel_main.py`` and with the released
benchmark checkpoints (``hf download DailyWildfireCAN/DailyWildfireCAN-Benchmark``).
Architectures can be given with their code name (``unet_time_gap``) or their
Hugging Face folder name (``unet_with_time_gap``).

Examples::

    # Released checkpoints, all runs, the three test splits
    python -m src.evaluate --checkpoint_dir benchmark_models_checkpoints \
        --architectures unet unet_with_time_gap utae

    # Only Test-Hard, run 1
    python -m src.evaluate --checkpoint_dir . --architectures unet --runs 1 --splits test_hard
"""
from __future__ import annotations

import argparse
import copy

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.builder import (
    build_dataloaders,
    build_model,
    canonical_architecture,
    find_checkpoint,
    forward_batch,
    load_weights,
    resolve_device,
)
from src.config import Config
from src.data import SPLIT_LABELS, TEST_SPLITS
from src.losses import build_loss
from src.metrics import BinaryMetrics

DEFAULT_ARCHITECTURES = (
    "unet", "unet_time_gap", "unet_attention", "unet_segformer", "asufm",
    "unet_convlstm", "utae", "simvpv2_spatiotemporal", "video_swin_unet", "frp",
)


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Mean loss and :class:`~src.metrics.BinaryMetrics` of ``model`` on ``loader``."""
    model.eval()
    loss_fn = build_loss()
    metrics = BinaryMetrics(device)
    total_loss = 0.0
    for batch in tqdm(loader, leave=False):
        labels = batch["label"].to(device)
        logits = forward_batch(model, batch, device)
        total_loss += loss_fn(logits, labels).item()
        metrics.update(logits, labels)
    return {"loss": total_loss / len(loader), **metrics.compute()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DailyWildfireCAN checkpoints on the test splits.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config (model hyper-parameters).")
    parser.add_argument("--architectures", nargs="+", default=list(DEFAULT_ARCHITECTURES),
                        help="Code or Hugging Face architecture names.")
    parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3], help="Run ids to evaluate.")
    parser.add_argument("--checkpoint_dir", default=".", help="Folder containing the checkpoint folders.")
    parser.add_argument("--splits", nargs="+", default=list(TEST_SPLITS), choices=TEST_SPLITS)
    parser.add_argument("--device", default=None, help="Override training.device (auto, cpu, cuda, mps).")
    parser.add_argument("--output", default="evaluation_results.csv", help="Output CSV.")
    args = parser.parse_args()

    base_config = Config.from_yaml(args.config)
    device = resolve_device(args.device or base_config.training.device)
    rows = []

    for name in args.architectures:
        arch = canonical_architecture(name)
        checkpoints = {run: find_checkpoint(args.checkpoint_dir, arch, run) for run in args.runs}
        checkpoints = {run: path for run, path in checkpoints.items() if path is not None}
        if not checkpoints:
            print(f"[{arch}] no checkpoint found in {args.checkpoint_dir}; skipping.")
            continue

        config = copy.deepcopy(base_config)
        config.model.architecture = arch
        loaders = build_dataloaders(config, splits=tuple(args.splits))
        model = build_model(config, device)

        for run, path in checkpoints.items():
            checkpoint = load_weights(model, path, device)
            print(f"\n[{arch}] run {run}: {path} (epoch {checkpoint.get('epoch')})")
            for split in args.splits:
                if loaders[split] is None:
                    continue
                result = evaluate_model(model, loaders[split], device)
                print(f"  {SPLIT_LABELS[split]:<12} " + " ".join(f"{k}={v:.4f}" for k, v in result.items()))
                rows.append({"architecture": arch, "run": run, "split": split,
                             "epoch": checkpoint.get("epoch"), **result})

    if not rows:
        print("No evaluation was run.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    summary = df.groupby(["architecture", "split"])["AUC_PR"].agg(["mean", "std", "count"])
    print(f"\nResults written to {args.output}\n\nAUC-PR (mean / std over runs):\n{summary.round(3).to_string()}")


if __name__ == "__main__":
    main()
