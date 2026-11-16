"""Label-variation analysis (Section 5.4, Table 9 of the paper).

Re-evaluates a model on the combined test sets with two relaxed targets:

* ``spatial``: the next-day target dilated by one pixel (90 m);
* ``temporal``: the union of the burned areas of days T, T+1 and T+2.

The alternative labels are written next to each split by
``samples_generation.data_generator.generate_raw_temporal_labels``
(``<split>_temporal_labels/sample_<idx>.npz``).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.builder import build_datasets_for, resolve_device
from src.config import Config
from src.data import TEST_SPLITS
from src.evaluate import evaluate_model
from src.evaluation.common import analysis_config, iter_runs, loader, merged, summarize

VARIANTS = (None, "spatial", "temporal")
VARIANT_NAMES = {None: "original", "spatial": "spatial (1-pixel dilation)", "temporal": "temporal (3-day merge)"}


def run_variation_analysis(
    config: Config,
    architecture: str,
    checkpoint_dir: str | Path,
    runs: list[int],
    output: str | Path,
    variants: tuple = VARIANTS,
) -> pd.DataFrame:
    """AUC-PR on the combined test sets with the original and relaxed labels, for every run."""
    config = analysis_config(config, architecture)
    device = resolve_device(config.training.device)
    combined = {v: merged(list(build_datasets_for(config, TEST_SPLITS, label_variant=v).values()))
                for v in variants}

    rows = []
    for run, model in iter_runs(config, checkpoint_dir, runs, device):
        for variant, dataset in combined.items():
            metrics = evaluate_model(model, loader(dataset, config), device)
            rows.append({"Run": run, "Labels": VARIANT_NAMES[variant], **metrics})
            print(f"  {VARIANT_NAMES[variant]:<28} AUC-PR={metrics['AUC_PR']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(Path(output).with_suffix(".runs.csv"), index=False)
    summary = summarize(df, ["Labels"], ["AUC_PR", "Precision", "Recall"])
    summary.to_csv(output, index=False)
    print(f"\n{summary.round(3).to_string(index=False)}\n\nSaved to {output}")
    return summary
