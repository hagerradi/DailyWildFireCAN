"""Script for the Multi-Modal Model | Code from  https://github.com/dazhaxie666/FRP_Prediction/"""

import torch
import torch.nn as nn

from src.models.frp.MultimodalEncoder import StateBlock, DynamicBlock, ConstantBlock, CombinedBlock
from src.models.frp.BackboneEncoder import Encoder
from src.models.frp.BackboneDecoder import Decoder

class FullNetwork(nn.Module):
    def __init__(self, combined_block, encoder, decoder):
        super(FullNetwork, self).__init__()
        self.combined_block = combined_block
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x1, x2, x3):
        # Forward pass through the combined block with three separate inputs
        combined_x = self.combined_block(x1, x2, x3)
        
        # Forward pass through the encoder and decoder with the concatenated output
        encoded_x = self.encoder(combined_x)
        decoded_x = self.decoder(encoded_x)
        return decoded_x

# ADDED BY ME 
# TEST BLOCK
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on: {device}")

    # state_channels = 1
    # dynamic_channels = 9
    # static_channels = 6
    state_channels = 2
    dynamic_channels = 14
    static_channels = 32

    # Initialize the Blocks
    block1 = StateBlock(input_dim=state_channels,
                        hidden_dims=[4, 8, 16, 16],
                        # hidden_dims=[16, 32, 64, 64],
                        kernel_sizes=[(5, 5), (3, 3), (3, 3), (1, 1)],
                        num_layers=4,
                        num_conv_filters=[8, 16, 16],
                        # num_conv_filters=[64, 64, 64],
                        device=device)

    block2 = DynamicBlock(input_channels=dynamic_channels,
                          convlstm_hidden_channels=16,
                          conv_hidden_channels=[8, 16, 16],
                        # convlstm_hidden_channels=64,
                        # conv_hidden_channels=[64, 128, 128],
                          device=device)

    block3 = ConstantBlock(input_channels=static_channels).to(device)
    # block3 = ConstantBlock(input_channels=static_channels,
    #                         hidden_channels=[64, 64, 64]).to(device)

    # Combine and build the full network
    combined_block = CombinedBlock(block1, block2, block3).to(device)

    # fusion_channels = 256  # 64 (State) + 128 (Dynamic) + 64 (Constant)
    # encoder_dims = [128, 256, 256, 512]
    # decoder_dims = [256, 128, 64]
    # encoder = Encoder(
    #     in_channels=fusion_channels,
    #     hidden_dims=encoder_dims
    # ).to(device)
    # decoder = Decoder(
    #     in_channels=encoder_dims[-1],  # Automatically grabs the 512
    #     hidden_dims=decoder_dims,
    #     out_channels=1                 # Final predicted fire spread mask
    # ).to(device)
    # model = FullNetwork(combined_block, encoder, decoder).to(device)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    model = FullNetwork(combined_block, encoder, decoder).to(device)

    # Generate Dummy Data
    batch_size = 2
    time_steps = 3      # Looking at the past 3 days
    height, width = 256, 256 # Spatial grid size

    # x1: State (Fire Mask Sequence) -> 5D [Batch, Time, Channels, H, W]
    x1 = torch.randn(batch_size, time_steps, state_channels, height, width).to(device)
    
    # x2: Dynamic (Weather Sequence) -> 5D [Batch, Time, Channels, H, W]
    x2 = torch.randn(batch_size, time_steps, dynamic_channels, height, width).to(device)
    
    # x3: Constant (Topography) -> 4D [Batch, Channels, H, W]
    # (Notice this one does NOT have the 'time_steps' dimension)
    x3 = torch.randn(batch_size, static_channels, height, width).to(device)

    print("\n--- Input Shapes ---")
    print(f"x1 (State)   : {x1.shape}")
    print(f"x2 (Dynamic) : {x2.shape}")
    print(f"x3 (Constant): {x3.shape}")

    # Forward Pass
    output = model(x1, x2, x3)

    print("\n--- Output Shape ---")
    print(f"Output       : {output.shape}")
    # The output should be 4D: [Batch, Out_Channels, Height, Width]