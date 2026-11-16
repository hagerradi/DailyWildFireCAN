"""Heuristic and simulation-based baselines of the paper (Table 5).

All baselines take mono-temporal ``input_grids`` of shape ``(B, 50, H, W)`` and return
pseudo-logits ``(B, 1, H, W)`` (``+LOGIT`` where fire is predicted, ``-LOGIT`` elsewhere)
so they can be evaluated with :func:`src.evaluate.evaluate_model`.
"""
import torch

LOGIT = 10.0


def to_logits(mask: torch.Tensor) -> torch.Tensor:
    """Map a binary mask to pseudo-logits (``sigmoid(±LOGIT)`` ≈ 1 / 0)."""
    return (mask.float() * 2 - 1) * LOGIT
