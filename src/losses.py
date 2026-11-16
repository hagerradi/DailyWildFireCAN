"""Training loss: weighted sum of a focal loss and a Dice loss (Section 4 of the paper)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss on logits (Lin et al., 2017).

    Args:
        alpha: Weight of the positive class (``1 - alpha`` for the negative class).
        gamma: Focusing parameter; larger values down-weight easy pixels more.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.type_as(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """Soft Dice loss computed over the whole batch.

    Args:
        smooth: Additive smoothing; keeps gradients alive on empty masks.
        apply_sigmoid: Apply a sigmoid to the inputs (set when the model outputs logits).
    """

    def __init__(self, smooth: float = 1.0, apply_sigmoid: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.apply_sigmoid:
            inputs = torch.sigmoid(inputs)
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1).type_as(inputs)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    """``focal_weight * FocalLoss + dice_weight * DiceLoss`` on ``(B, 1, H, W)`` logits.

    Targets of shape ``(B, H, W)`` are expanded to ``(B, 1, H, W)``.
    """

    def __init__(
        self,
        alpha: float = 0.85,
        gamma: float = 2.0,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss(smooth=1.0, apply_sigmoid=True)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        return self.focal_weight * self.focal(logits, targets) + self.dice_weight * self.dice(logits, targets)


def build_loss() -> CombinedLoss:
    """Loss used for every model in the benchmark (focal alpha=0.85, gamma=2, weights 0.5/0.5)."""
    return CombinedLoss(alpha=0.85, gamma=2.0, focal_weight=0.5, dice_weight=0.5)
