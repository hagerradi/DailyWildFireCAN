import torch
import torch.nn as nn

from src.models.simvp2.simvp_model import SimVP_Model

class WildfireSimVPWrapper(nn.Module):
    def __init__(self, num_timesteps=1, channels_per_step=48, img_size=256, is_spatial_only=True):
        """
        Args:
            num_timesteps: The number of days of history provided.
            channels_per_step: The number of features per day (e.g., 48).
            img_size: The grid size (e.g., 256).
            is_spatial_only: 
                If True, expects 4D input [B, C, H, W] and adds a fake time dimension.
                If False, expects 5D input [B, T, C, H, W] directly from the dataloader.
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        self.channels_per_step = channels_per_step
        self.is_spatial_only = is_spatial_only
        
        # Initialize the core SimVP model
        self.simvp = SimVP_Model(
            in_shape=(num_timesteps, channels_per_step, img_size, img_size),
            hid_S=64,       
            hid_T=256,      
            N_S=4,          
            N_T=4,          
            model_type='gSTA' 
        )
        
        # Final layer to convert the 48 predicted channels into 1 binary fire mask logit
        self.final_conv = nn.Conv2d(channels_per_step, 1, kernel_size=1)

    def forward(self, x, **kwargs):
        
        # HANDLE INPUT SHAPE
        if self.is_spatial_only:
            # Spatial dataset case: [Batch, 48, 256, 256]
            # Add a fake temporal dimension to make it 5D: [Batch, 1, 48, 256, 256]
            x_5d = x.unsqueeze(1)
        else:
            # Spatio-Temporal dataset case: [Batch, Time, 48, 256, 256]
            # Dimensions are already 5D, so we pass it directly
            x_5d = x
            
        # FORWARD PASS
        out_5d = self.simvp(x_5d)
        
        # HANDLE OUTPUT SHAPE
        # SimVP predicts a sequence, but we only want ONE future day (Tomorrow).
        # We slice index 0 from the Time dimension.
        tomorrow = out_5d[:, 0, :, :, :]  # Shape: [Batch, 48, 256, 256]
        
        # Project down to 1 channel for the binary classification mask
        fire_mask_logits = self.final_conv(tomorrow) # Shape: [Batch, 1, 256, 256]
        
        return fire_mask_logits