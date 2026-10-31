"""Code from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py"""

import torch
import torch.nn as nn
import math

class SatelliteTimeGapBottleneck(nn.Module):
    """Generates sinusoidal Fourier embeddings to encode time gaps (satellite time gap)."""
    def __init__(self, fourier_dim=16):
        super().__init__()
        self.fourier_dim = fourier_dim
        
        # FOURIER
        half_dim = fourier_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('freqs', freqs)

    def forward(self, delta_t):
        """Encodes the time gap into a Fourier feature representation.

        Args:
          delta_t: The gap in days between the satellite image and the fire day.

        Returns:
          The sine and cosine Fourier embeddings for the given time gap.
        """
        # Calculate the Sine/Cosine waves
        # Shape: (Batch, 1) * (1, 8) -> (Batch, 8)
        args = delta_t[:, None].float() * self.freqs[None, :]
        embedding = torch.cat((args.sin(), args.cos()), dim=-1) # (Batch, 16)
        
        return embedding

class TimeGapInjectionMLP(nn.Module):
    """Projects the temporal Fourier embedding to match a network's bottleneck dimension."""
    def __init__(self, fourier_dim, bottleneck_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, fourier_dim * 4),
            nn.GELU(),
            nn.Linear(fourier_dim * 4, bottleneck_channels)
        )

    def forward(self, emb):
        """Processes the time embedding through an MLP for network injection.

        Args:
          emb: The raw Fourier time embedding.

        Returns:
          The projected embedding mapped to the bottleneck channel size.
        """
        # Turns the 16-number barcode into a vector the size of the bottleneck
        # Shape: (Batch, bottleneck_channels)
        return self.mlp(emb)