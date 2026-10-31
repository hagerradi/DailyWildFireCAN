import torch
import torch.nn as nn
import torch.nn.functional as F

class DilationBaselineModel(nn.Module):
    def __init__(self, fire_mask_channel_idx: int = -2, kernel_size: int = 3):
        """
        A baseline model that predicts new growth by expanding the previous day's 
        fire mask outward uniformly and subtracting the original mask.
        
        Args:
            fire_mask_channel_idx: The channel index in `input_grids` containing the t-1 mask.
            kernel_size: Size of the dilation neighborhood. 
                         3 expands by 1 pixel outward. 
                         5 expands by 2 pixels outward, etc.
        """
        super().__init__()
        self.fire_mask_channel_idx = fire_mask_channel_idx
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extracts the previous day's mask, dilates it, and isolates the expansion ring.
        
        Args:
            x: Input tensor of shape (Batch, Channels, Height, Width)
        Returns:
            Tensor of shape (Batch, 1, Height, Width) representing the predicted new growth.
        """
        # Extract the t-1 fire mask
        prev_mask = x[:, self.fire_mask_channel_idx : self.fire_mask_channel_idx + 1, :, :]
        
        # Dilate the mask (max pooling over a binary mask expands the 1s)
        dilated_mask = F.max_pool2d(
            prev_mask, 
            kernel_size=self.kernel_size, 
            stride=1, 
            padding=self.padding
        )
        
        # Isolate the growth ring by subtracting the original fire area
        # F.relu ensures no negative values if there are any formatting anomalies
        new_growth_prediction = F.relu(dilated_mask - prev_mask)

        return new_growth_prediction