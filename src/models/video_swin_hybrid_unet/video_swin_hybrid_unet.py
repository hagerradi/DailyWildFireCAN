import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerStage, PatchMerging

class ConvDecoderBlock3D(nn.Module):
    """The standard 'Decoder Block' from your architecture diagram."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class VideoSwinHybridUNet(nn.Module):
    def __init__(self, in_channels, time_steps, input_size=(256, 256), num_classes=1, embed_dim=96):
        super().__init__()
        self.time_steps = time_steps
        
        # Calculate dynamic resolutions based on the input_size tuple
        h, w = input_size
        res1 = (h // 4, w // 4)  # 64x64 for a 256x256 input
        res2 = (res1[0] // 2, res1[1] // 2)  # 32x32
        res3 = (res2[0] // 2, res2[1] // 2)  # 16x16

        # Dynamic Window Size calculation
        w_size = 8 if res1[0] % 8 == 0 else 7

        # ==========================================
        # ENCODER PATH
        # ==========================================
        # Initial 3D Patch Embedding
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        
        # Stage 1: No downsampling (processes initial 64x64 patch features)
        self.stage1 = SwinTransformerStage(
            input_resolution=res1, dim=embed_dim, out_dim=embed_dim, 
            depth=2, num_heads=3, window_size=w_size, downsample=None
        )
        # Stage 2: Receives 64x64, downsamples to 32x32 using PatchMerging
        self.stage2 = SwinTransformerStage(
            input_resolution=res1, dim=embed_dim, out_dim=embed_dim * 2, 
            depth=2, num_heads=6, window_size=w_size, downsample=PatchMerging
        )
        # Stage 3: Receives 32x32, downsamples to 16x16 using PatchMerging
        self.stage3 = SwinTransformerStage(
            input_resolution=res2, dim=embed_dim * 2, out_dim=embed_dim * 4, 
            depth=6, num_heads=12, window_size=w_size, downsample=PatchMerging
        )
        # Bottleneck Stage: Receives 16x16, downsamples to 8x8 using PatchMerging
        self.stage4 = SwinTransformerStage(
            input_resolution=res3, dim=embed_dim * 4, out_dim=embed_dim * 8, 
            depth=2, num_heads=24, window_size=w_size, downsample=PatchMerging
        )

        # ==========================================
        # DECODER PATH & OUTPUT GENERATION
        # ==========================================
        # --- Bottom Stage ---
        self.up_trans_conv1 = nn.ConvTranspose3d(
            embed_dim * 8, embed_dim * 4, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.decoder_block1 = ConvDecoderBlock3D(in_channels=embed_dim * 8, out_channels=embed_dim * 4)
        
        # --- Middle Stage ---
        self.up_trans_conv2 = nn.ConvTranspose3d(
            embed_dim * 4, embed_dim * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.decoder_block2 = ConvDecoderBlock3D(in_channels=embed_dim * 4, out_channels=embed_dim * 2)
        
        # --- Top Stage ---
        self.up_trans_conv3 = nn.ConvTranspose3d(
            embed_dim * 2, embed_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.decoder_block3 = ConvDecoderBlock3D(in_channels=embed_dim * 2, out_channels=embed_dim)
        
        # ==========================================
        # OUTPUT GENERATION
        # ==========================================
        # Restores the original 4x spatial patch size downsampled at step 1
        self.final_restore_upsample = nn.ConvTranspose3d(
            embed_dim, embed_dim // 2, kernel_size=(1, 4, 4), stride=(1, 4, 4)
        )
        # Collapses the temporal/day dimension to map straight into your 2D prediction map
        self.final_projection = nn.Conv3d(
            embed_dim // 2, num_classes, kernel_size=(time_steps, 1, 1), stride=1, padding=0
        )

    def _forward_swin_stage(self, stage, x):
        """Helper to process 5D video tensors [B, C, T, H, W] through timm Swin blocks"""
        B, C, T, H, W = x.shape
        # Permute to Swin format: [B * T, H, W, C]
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H, W, C)
        # Run through official timm stage block logic
        x = stage(x)
        # Reshape back to standard 5D video tensor layout
        _, New_H, New_W, New_C = x.shape
        x = x.view(B, T, New_H, New_W, New_C).permute(0, 4, 1, 2, 3)
        return x

    def forward(self, x):
        # Input shape expected: [Batch, Channels, Time, Height, Width]

        # Swap Time (1) and Channels (2)
        x = x.permute(0, 2, 1, 3, 4)
        
        # ENCODER (Applying Swin processing across timelines)
        x1 = self.patch_embed(x)
        skip1 = self._forward_swin_stage(self.stage1, x1)
        skip2 = self._forward_swin_stage(self.stage2, skip1)
        skip3 = self._forward_swin_stage(self.stage3, skip2)
        bottleneck = self._forward_swin_stage(self.stage4, skip3)
        
        # DECODER (Fusing features via standard 3D skip links)
        d1 = self.up_trans_conv1(bottleneck)
        d1 = torch.cat([d1, skip3], dim=1)
        d1 = self.decoder_block1(d1)
        
        d2 = self.up_trans_conv2(d1)
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.decoder_block2(d2)
        
        d3 = self.up_trans_conv3(d2)
        d3 = torch.cat([d3, skip1], dim=1)
        d3 = self.decoder_block3(d3)
        
        # OUTPUT GENERATION
        out_features = self.final_restore_upsample(d3)
        output_tensor = self.final_projection(out_features)
        
        return output_tensor.squeeze(2)
    
def test_architecture():
    print("Initializing model...")
    # 55 channels (e.g., satellite/weather), 3 days of history
    model = VideoSwinHybridUNet(in_channels=55, time_steps=3)
    
    print("Creating dummy wildfire tensor...")
    # Shape: [Batch, Channels, Time, Height, Width]
    # Simulating a batch size of 2, 55 channels, 3 days, 256x256 grid
    # dummy_input = torch.randn(2, 55, 3, 256, 256)
    dummy_input = torch.randn(2, 3, 55, 256, 256)
    
    print("Running forward pass...")
    output = model(dummy_input)
    
    print(f"Success! Final Output Shape: {output.shape}")
    # We expect [2, 1, 256, 256] -> [Batch, Num_Classes, Height, Width]