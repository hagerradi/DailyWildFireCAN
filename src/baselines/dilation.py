"""Dilation baseline: uniform spread of the day-T fire by a fixed number of pixels."""
import torch
import torch.nn.functional as F
from torch import nn

from src.baselines import to_logits
from src.features import FIRE_MASK_CHANNEL


class DilationBaselineModel(nn.Module):
    """Dilates the day-T fire mask and predicts the ring of new pixels.

    Args:
        fire_mask_channel: Channel of the day-T fire mask in ``x``.
        kernel_size: Size of the square dilation window (3 = one pixel / 90 m in every direction).
    """

    def __init__(self, fire_mask_channel: int = FIRE_MASK_CHANNEL, kernel_size: int = 3) -> None:
        super().__init__()
        self.fire_mask_channel = fire_mask_channel
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        fire = (x[:, self.fire_mask_channel : self.fire_mask_channel + 1] > 0).float()
        dilated = F.max_pool2d(fire, self.kernel_size, stride=1, padding=self.kernel_size // 2)
        return to_logits((dilated - fire) > 0)
