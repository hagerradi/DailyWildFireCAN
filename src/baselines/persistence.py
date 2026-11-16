"""Persistence baseline: the fire does not grow, so no new burned pixel is predicted."""
import torch
from torch import nn

from src.baselines import to_logits
from src.features import FIRE_MASK_CHANNEL


class PersistenceModel(nn.Module):
    """Predicts the day-T fire mask as the next-day fire mask.

    The targets only contain newly burned pixels, so this baseline measures how much of the
    target overlaps the current fire (expected to be close to zero).
    """

    def __init__(self, fire_mask_channel: int = FIRE_MASK_CHANNEL) -> None:
        super().__init__()
        self.fire_mask_channel = fire_mask_channel

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return to_logits(x[:, self.fire_mask_channel : self.fire_mask_channel + 1] > 0)
