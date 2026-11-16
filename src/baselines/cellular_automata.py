"""Cellular automata baseline following Alexandridis et al. (2008).

A cell catches fire when the summed spread probability from its burning neighbours exceeds
0.5. The probability depends on wind speed and direction, slope and NDVI; water, urban,
barren and snow land cover are not burnable.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.baselines import to_logits
from src.features import FEATURE_NAMES

UNBURNABLE_LANDCOVER = ("ccrs_16_barren", "ccrs_17_urban", "ccrs_18_water", "ccrs_19_snow")
# 8-neighbour unit offsets (dx, dy): E, NE, N, NW, W, SW, S, SE
DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


class AlexandridisCA(nn.Module):
    """Cellular automata spread model.

    Args:
        stats: Training statistics (``training_stats_mono_temporal_samples.json`` of the
            dataset), used to undo the z-score of wind speed, wind components and slope.
        steps: Number of CA iterations simulated for one day.
        p_0, c_1, c_2, a_s: Base ignition probability and wind-speed, wind-direction and
            slope coefficients of Alexandridis et al. (2008).
    """

    def __init__(self, stats: dict, steps: int = 1, p_0: float = 0.3, c_1: float = 0.045,
                 c_2: float = 0.131, a_s: float = 0.078) -> None:
        super().__init__()
        self.stats, self.steps = stats, steps
        self.p_0, self.c_1, self.c_2, self.a_s = p_0, c_1, c_2, a_s
        self.idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
        kernel = torch.ones(1, 1, 3, 3)
        kernel[0, 0, 1, 1] = 0
        self.register_buffer("neighbour_kernel", kernel)

    @classmethod
    def from_stats_file(cls, path: str | Path, **kwargs) -> "AlexandridisCA":
        with open(path) as f:
            return cls(json.load(f), **kwargs)

    def _physical(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return x[:, self.idx[name]] * self.stats[name]["std"] + self.stats[name]["mean"]

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x.float()
        ws = torch.clamp(self._physical(x, "ws"), 0, 30)
        wind_dir = torch.atan2(self._physical(x, "v10"), self._physical(x, "u10"))
        v_s = torch.exp(self.a_s * self._physical(x, "slope"))
        p_base = torch.clamp(self.p_0 + 0.2 * x[:, self.idx["ndvi"]], 0.1, 0.8)
        burnable = (sum(x[:, self.idx[n]] for n in UNBURNABLE_LANDCOVER) == 0).unsqueeze(1)

        initial = (x[:, self.idx["fire_mask"]] > 0).float().unsqueeze(1)
        fire = initial.clone()
        for _ in range(self.steps):
            has_neighbour = F.conv2d(fire, self.neighbour_kernel, padding=1) > 0
            candidates = (fire == 0) & burnable & has_neighbour
            if not candidates.any():
                break
            p_total = torch.zeros_like(fire)
            for dx, dy in DIRECTIONS:
                angle = wind_dir - torch.atan2(torch.tensor(float(dy)), torch.tensor(float(dx)))
                v_w = torch.exp(self.c_1 * ws) * torch.exp(ws * self.c_2 * (torch.cos(angle) - 1.0))
                p_dir = p_base * (1.0 + v_w) * (1.0 + v_s)
                p_total += torch.roll(fire, shifts=(dy, dx), dims=(2, 3)) * p_dir.unsqueeze(1)
            fire = torch.clamp(fire + ((p_total > 0.5) & candidates).float(), 0.0, 1.0)

        return to_logits((fire - initial) > 0)
