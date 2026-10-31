import torch
import torch.nn as nn

class PersistenceModel(nn.Module):
    def __init__(self, fire_mask_channel_idx: int = 0):
        """
        A dummy model that outputs the previous day's fire mask.
        
        Args:
            fire_mask_channel_idx: The channel index in `input_grids` 
                                   that contains the t-1 fire mask.
        """
        super().__init__()
        self.fire_mask_channel_idx = fire_mask_channel_idx

        self._printed_debug = False # Flag to ensure we only print once

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extracts and returns the previous day's fire mask.
        
        Args:
            x: The input grids tensor of shape (Batch, Channels, Height, Width) 
               or (Batch, Time, Channels, Height, Width)
        Returns:
            Tensor of shape (Batch, 1, Height, Width) representing the predicted mask.
        """
        # If x is shape (Batch, Channels, H, W)
        # Slice out the specific channel and keep the channel dimension
        prev_mask = x[:, self.fire_mask_channel_idx : self.fire_mask_channel_idx + 1, :, :]
        
        # If x has a time dimension: (Batch, Time, Channels, H, W)
        # prev_mask = x[:, -1, self.fire_mask_channel_idx : self.fire_mask_channel_idx + 1, :, :]

        # --- DEBUGGING BLOCK ---
        if not self._printed_debug:
            print("\n--- Persistence Model Tensor Debug ---")
            print(f"Input x shape:           {x.shape}")
            print(f"Channel index targeted:  {self.fire_mask_channel_idx}")
            print(f"prev_mask shape:         {prev_mask.shape}")
            print(f"prev_mask min value:     {prev_mask.min().item()}")
            print(f"prev_mask max value:     {prev_mask.max().item()}")
            print(f"prev_mask unique values: {torch.unique(prev_mask).tolist()}")
            print("--------------------------------------\n")
            self._printed_debug = True # Prevent spamming the console
        # -----------------------

        return prev_mask