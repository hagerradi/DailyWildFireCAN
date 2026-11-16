"""Robustness and generalization analysis (Section 5.2 and Appendix C.2 of the paper).

Evaluates a model on subsets of the test sets, following the split choices of the paper:

* fire size and fire age (duration stage): Test-Hard (Table 7);
* month: Test-Hard; year: Test-Medium (Table C.3);
* province / territory: Test-Easy and Test-Hard combined (Table 8).

Needs the per-sample metadata JSONs written by ``samples_generation.metadata_generator_main``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import ConcatDataset, Subset

from src.builder import build_datasets_for, resolve_device
from src.config import Config
from src.evaluate import evaluate_model
from src.evaluation import filters
from src.evaluation.common import analysis_config, iter_runs, load_split_metadata, loader, summarize

SIZE_BINS = {
    "Extra Small (<600 ha)": (0.0, 600.0),
    "Small (600-5,000 ha)": (600.0, 5000.0),
    "Medium (5,000-20,000 ha)": (5000.0, 20000.0),
    "Large (>20,000 ha)": (20000.0, float("inf")),
}
FIRE_AGE_STAGES = {
    "Early (1-3 days)": (1, 3),
    "Developing (4-7 days)": (4, 7),
    "Established (8-14 days)": (8, 14),
    "Late (>14 days)": (15, 10**6),
}
MONTHS = {3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August",
          9: "September", 10: "October", 11: "November"}
PROVINCES = ("BC", "AB", "SK", "MB", "ON", "QC", "NWT", "YT")


def build_suites(config: Config, metadata_folder: str | Path | None = None) -> list[tuple[str, str, object]]:
    """Return ``(group, condition, Subset)`` triples for every analysis condition."""
    datasets = build_datasets_for(config, ("test_easy", "test_medium", "test_hard"))
    meta = {split: load_split_metadata(config, split, ds, metadata_folder)
            for split, ds in datasets.items() if len(ds) > 0}

    suites = []
    if "test_hard" in meta:
        hard, df = datasets["test_hard"], meta["test_hard"]
        suites += [("Size (Test-Hard)", name, Subset(hard, filters.by_size(df, lo, hi)))
                   for name, (lo, hi) in SIZE_BINS.items()]
        suites += [("Fire age (Test-Hard)", name, Subset(hard, filters.by_fireday(df, lo, hi)))
                   for name, (lo, hi) in FIRE_AGE_STAGES.items()]
        suites += [("Month (Test-Hard)", name, Subset(hard, filters.by_month(df, m))) for m, name in MONTHS.items()]
    if "test_medium" in meta:
        medium, df = datasets["test_medium"], meta["test_medium"]
        suites += [("Year (Test-Medium)", str(y), Subset(medium, filters.by_year(df, y)))
                   for y in sorted(df["year"].unique())]
    union = [s for s in ("test_easy", "test_hard") if s in meta]
    if union:
        offsets = {s: sum(len(datasets[u]) for u in union[:i]) for i, s in enumerate(union)}
        combined = ConcatDataset([datasets[s] for s in union])
        label = "Province (" + " + ".join("Test-Easy" if s == "test_easy" else "Test-Hard" for s in union) + ")"
        for prov in PROVINCES:
            idx = [i + offsets[s] for s in union for i in filters.by_province(meta[s], prov)]
            suites.append((label, prov, Subset(combined, idx)))
    return [s for s in suites if len(s[2]) > 0]


def run_generalization_analysis(
    config: Config,
    architecture: str,
    checkpoint_dir: str | Path,
    runs: list[int],
    output: str | Path,
    metadata_folder: str | Path | None = None,
) -> pd.DataFrame:
    """Evaluate every run on every condition and write per-run and aggregated CSVs."""
    config = analysis_config(config, architecture)
    device = resolve_device(config.training.device)
    suites = build_suites(config, metadata_folder)

    rows = []
    for run, model in iter_runs(config, checkpoint_dir, runs, device):
        for group, condition, ds in suites:
            metrics = evaluate_model(model, loader(ds, config), device)
            rows.append({"Run": run, "Group": group, "Condition": condition, "Samples": len(ds), **metrics})
            print(f"  {group:<28} {condition:<26} n={len(ds):<5} AUC-PR={metrics['AUC_PR']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(Path(output).with_suffix(".runs.csv"), index=False)
    summary = summarize(df, ["Group", "Condition", "Samples"], ["AUC_PR", "Precision", "Recall"])
    summary.to_csv(output, index=False)
    print(f"\n{summary.round(3).to_string(index=False)}\n\nSaved to {output}")
    return summary
