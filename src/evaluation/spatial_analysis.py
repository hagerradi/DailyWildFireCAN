"""Spatial error analysis (Figure 4 and Appendix C.4 of the paper).

On the combined test sets, computes for every run:

* **Boundary F1** at 90, 180 and 270 m: a predicted (true) perimeter pixel counts as
  correct if a true (predicted) perimeter pixel lies within the tolerance.
* **Spread-distance F1** in five bands (0-90, 90-180, 180-270, 270-360, >360 m) of
  distance from the current fire perimeter, on new-burn pixels only.
* Over- and under-predicted area in hectares.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

from src.builder import build_datasets_for, forward_batch, resolve_device
from src.config import Config
from src.data import TEST_SPLITS
from src.evaluation.common import analysis_config, iter_runs, loader, merged, summarize
from src.features import FIRE_MASK_CHANNEL, STATE_FEATURES

PIXEL_SIZE_M = 90.0
PIXEL_AREA_HA = PIXEL_SIZE_M**2 / 10_000
BOUNDARY_TOLERANCES = {"90m": 1, "180m": 2, "270m": 3}  # in pixels
SPREAD_BANDS = {"0-90m": (0, 90), "90-180m": (90, 180), "180-270m": (180, 270),
                "270-360m": (270, 360), ">360m": (360, np.inf)}
EPS = 1e-8


def current_fire_mask(batch: dict) -> torch.Tensor:
    """Binary fire mask on day T (``(B, H, W)``) for any input layout."""
    if "input_grids" in batch:
        x = batch["input_grids"]
        return x[:, -1, FIRE_MASK_CHANNEL] if x.ndim == 5 else x[:, FIRE_MASK_CHANNEL]
    if "input_env" in batch:  # RGB channels (all before the fire mask) are removed
        return batch["input_env"][:, FIRE_MASK_CHANNEL - 3]
    state = batch["state"]
    channel = STATE_FEATURES.index("fire_mask")
    return state[:, -1, channel] if state.ndim == 5 else state[:, channel]


def _f1(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision, recall = tp / (tp + fp + EPS), tp / (tp + fn + EPS)
    return precision, recall, 2 * precision * recall / (precision + recall + EPS)


@torch.no_grad()
def spatial_errors(model: torch.nn.Module, data_loader, device: torch.device, threshold: float = 0.5) -> dict:
    """Boundary, spread-distance and area statistics of ``model`` on ``data_loader``."""
    boundary = {name: np.zeros(4) for name in BOUNDARY_TOLERANCES}  # tp_pred, n_pred, tp_true, n_true
    spread = {name: np.zeros(3) for name in SPREAD_BANDS}           # tp, fp, fn
    over_ha = under_ha = true_ha = 0.0

    for batch in data_loader:
        preds = (torch.sigmoid(forward_batch(model, batch, device)) > threshold).squeeze(1).cpu().numpy()
        targets = batch["label"].numpy().astype(bool)
        fires = current_fire_mask(batch).numpy() > 0

        for p, t, fire in zip(preds, targets, fires):
            over_ha += (p & ~t).sum() * PIXEL_AREA_HA
            under_ha += (~p & t).sum() * PIXEL_AREA_HA
            true_ha += t.sum() * PIXEL_AREA_HA

            p_edge, t_edge = p ^ binary_erosion(p), t ^ binary_erosion(t)
            dist_to_t = distance_transform_edt(~t_edge) if t_edge.any() else np.full(t.shape, np.inf)
            dist_to_p = distance_transform_edt(~p_edge) if p_edge.any() else np.full(p.shape, np.inf)
            for name, tol in BOUNDARY_TOLERANCES.items():
                boundary[name] += [(p_edge & (dist_to_t <= tol)).sum(), p_edge.sum(),
                                   (t_edge & (dist_to_p <= tol)).sum(), t_edge.sum()]

            new_fire = t & ~fire
            dist = distance_transform_edt(~fire) * PIXEL_SIZE_M if fire.any() else np.full(t.shape, np.inf)
            for name, (lo, hi) in SPREAD_BANDS.items():
                band = (dist > lo) & (dist <= hi) & ~fire
                spread[name] += [(p & band & new_fire).sum(), (p & band & ~new_fire).sum(),
                                 (~p & band & new_fire).sum()]

    results = {"Overpredicted_ha": over_ha, "Underpredicted_ha": under_ha, "True_burned_ha": true_ha}
    for name, (tp_p, n_p, tp_t, n_t) in boundary.items():
        precision, recall = tp_p / (n_p + EPS), tp_t / (n_t + EPS)
        results[f"Boundary_F1_{name}"] = 2 * precision * recall / (precision + recall + EPS)
    for name, (tp, fp, fn) in spread.items():
        precision, recall, f1 = _f1(tp, fp, fn)
        results.update({f"Spread_F1_{name}": f1, f"Spread_Precision_{name}": precision,
                        f"Spread_Recall_{name}": recall})
    return results


def run_spatial_analysis(
    config: Config,
    architecture: str,
    checkpoint_dir: str | Path,
    runs: list[int],
    output: str | Path,
) -> pd.DataFrame:
    """Spatial errors on the combined test sets for every run; writes per-run and mean/std CSVs."""
    config = analysis_config(config, architecture)
    device = resolve_device(config.training.device)
    combined = merged(list(build_datasets_for(config, TEST_SPLITS).values()))
    print(f"Combined test sets: {len(combined)} samples")

    rows = []
    for run, model in iter_runs(config, checkpoint_dir, runs, device):
        rows.append({"Run": run, **spatial_errors(model, loader(combined, config), device)})

    df = pd.DataFrame(rows)
    df.to_csv(Path(output).with_suffix(".runs.csv"), index=False)
    df["Split"] = "combined_test"
    value_cols = [c for c in df.columns if c.startswith(("Boundary_F1", "Spread_F1"))]
    summary = summarize(df, ["Split"], value_cols)
    summary.to_csv(output, index=False)
    print(summary.T.round(3).to_string(header=False))
    print(f"\nSaved to {output}")
    return summary
