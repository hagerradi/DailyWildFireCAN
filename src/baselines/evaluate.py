"""Evaluate the heuristic baselines (persistence, dilation, cellular automata) on the test splits.

Example::

    python -m src.baselines.evaluate --stats <mono_temporal_samples>/training_stats_mono_temporal_samples.json
"""
import argparse

import pandas as pd
import torch

from src.baselines.cellular_automata import AlexandridisCA
from src.baselines.dilation import DilationBaselineModel
from src.baselines.persistence import PersistenceModel
from src.builder import build_dataloaders, resolve_device
from src.config import Config
from src.data import SPLIT_LABELS, TEST_SPLITS
from src.evaluate import evaluate_model

BASELINES = ("persistence", "dilation", "cellular_automata")


def build_baseline(name: str, stats_path: str | None = None) -> torch.nn.Module:
    if name == "persistence":
        return PersistenceModel()
    if name == "dilation":
        return DilationBaselineModel()
    if stats_path is None:
        raise ValueError("The cellular automata baseline needs --stats (training statistics JSON).")
    return AlexandridisCA.from_stats_file(stats_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--baselines", nargs="+", default=list(BASELINES), choices=BASELINES)
    parser.add_argument("--stats", default=None, help="training_stats_mono_temporal_samples.json")
    parser.add_argument("--splits", nargs="+", default=list(TEST_SPLITS), choices=TEST_SPLITS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="baseline_results.csv")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    config.model.architecture = "unet"  # mono-temporal grids layout
    device = resolve_device(args.device or config.training.device)
    loaders = build_dataloaders(config, splits=tuple(args.splits))

    rows = []
    for name in args.baselines:
        model = build_baseline(name, args.stats).to(device)
        for split in args.splits:
            if loaders[split] is None:
                continue
            result = evaluate_model(model, loaders[split], device)
            print(f"[{name}] {SPLIT_LABELS[split]:<12} AUC-PR={result['AUC_PR']:.3f} "
                  f"P={result['Precision']:.3f} R={result['Recall']:.3f}")
            rows.append({"baseline": name, "split": split, **result})
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
