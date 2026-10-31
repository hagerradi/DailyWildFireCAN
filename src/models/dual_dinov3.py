import torch
import torch.nn as nn
from transformers import AutoModel

class DualDinoV3(nn.Module):
    def __init__(self, 
                 env_channels: int = 11, 
                 num_classes: int = 1, 
                 hidden_features: list = None, 
                 use_skip_connections: bool = True, 
                 use_activation_after_upsampling: bool = False):
        """
        Args:
            input_channels: Number of input channels
            num_classes: Number of output channels
            hidden_features: List of feature maps at each level [64, 128, 256, 512]. Model adjusts accordingly
            use_skip_connections: Whether to use skip connections in the decoder
        """
        super().__init__()
        
        if hidden_features is None:
            hidden_features = [64, 128, 256, 512]
        
        self.use_skip_connections = use_skip_connections
        self.use_activation_after_upsampling = use_activation_after_upsampling

        # Initialize DINOv3
        self.dino_model = AutoModel.from_pretrained(
            # "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "facebook/dinov3-vitl16-pretrain-sat493m",
            token="")

        self.dino_patch_size = 16

        # self.dino_embed_dim = 768  # Standard for ViT-Base
        # Dynamically get the embed dimension from Hugging Face config
        if hasattr(self.dino_model, "config") and hasattr(self.dino_model.config, "hidden_size"):
            self.dino_embed_dim = self.dino_model.config.hidden_size
        else:
            self.dino_embed_dim = 1024 # Fallback for ViT-Large
        
        for param in self.dino_model.parameters():
            param.requires_grad = False  # Freeze backbone

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # encoder block: downsampling
        # in_ch = input_channels
        in_ch = env_channels
        for h_feature in hidden_features:
            # 4 blocks: n_channelsx64, 64x128, 128x256, 256x512
            self.encoder.append(self._double_conv_block(in_ch, h_feature))
            in_ch = h_feature
        
        # # bottleneck block (bottom of U): 512x1024
        # self.bottleneck_layer = self._double_conv_block(hidden_features[-1], hidden_features[-1] * 2)
        
        # bottleneck block: (hidden_features[-1] + dino_embed_dim) -> (hidden_features[-1] * 2)
        bottleneck_in_channels = hidden_features[-1] + self.dino_embed_dim
        self.bottleneck_layer = self._double_conv_block(bottleneck_in_channels, hidden_features[-1] * 2)
        
        # decoder block: upsampling
        for h_feature in reversed(hidden_features):
            # 4 downsampling blocks: 1024x512, 512x256, 256x128, 128x64
            if self.use_activation_after_upsampling:
                self.decoder.append(
                    nn.Sequential(nn.ConvTranspose2d(h_feature * 2, h_feature, kernel_size=2, stride=2),
                    nn.BatchNorm2d(h_feature),
                    nn.ReLU(inplace=True)))
            else:
                self.decoder.append(nn.ConvTranspose2d(h_feature * 2, h_feature, kernel_size=2, stride=2))
            decoder_in_channels = h_feature * 2 if use_skip_connections else h_feature
            self.decoder.append(self._double_conv_block(decoder_in_channels, h_feature))

        # output layer
        self.out_conv = nn.Conv2d(hidden_features[0], num_classes, kernel_size=1)
    
    def _double_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, rgb_x: torch.Tensor, env_x: torch.Tensor) -> torch.Tensor:

        B, _, H, W = rgb_x.shape

        skip_connections = []
        
        # 1. Encoder (Physics/Infrared Path)
        x = env_x
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.maxpool(x) # dowsnsample

        # 2. Vision Path (DINOv3)
        dino_out = self.dino_model(pixel_values=rgb_x)
        hidden_states = dino_out.last_hidden_state

        # Dynamically calculate patches to avoid hardcoding sizes
        num_patches_h = H // self.dino_patch_size
        num_patches_w = W // self.dino_patch_size
        total_patches = num_patches_h * num_patches_w
        
        # Slice off extra tokens (like CLS) and reshape to 2D grid
        patch_tokens = hidden_states[:, -total_patches:, :] 
        dino_features = patch_tokens.permute(0, 2, 1).view(B, self.dino_embed_dim, num_patches_h, num_patches_w)
        
        # 3. Bottleneck (Merge the two outputs)
        # Concatenate U-Net features and DINOv3 features along the channel dimension (dim=1)
        x = torch.cat([x, dino_features], dim=1)
        x = self.bottleneck_layer(x)
        
        # reverse skip connections for decoder
        skip_connections = skip_connections[::-1]
        
        # decoder part
        for i in range(len(self.decoder) // 2):#(0, 3)
            x = self.decoder[2 * i](x)  # upsample
            if self.use_skip_connections:
                skip_x = skip_connections[i]
                x = torch.cat([skip_x, x], dim=1)
            x = self.decoder[2 * i + 1](x)  # double conv block
        
        # output layer
        x = self.out_conv(x)
        return x

if __name__ == "__main__":

    # Define dummy tensors matching your DataLoader output
    batch_size = 2
    height, width = 256, 256
    
    print("Creating dummy inputs...")
    rgb_channels = 3
    env_channels = 52
    # DINOv3 expects exactly 3 channels (e.g., RGB)
    dummy_rgb = torch.randn(batch_size, rgb_channels, height, width)
    # Your remaining channels (NIR, SWIR, Wind, DEM, Fire Masks)
    dummy_env = torch.randn(batch_size, env_channels, height, width)
    
    # 2. Initialize the model
    print("Initializing Dual-Stream UNet (Loading DINOv3 weights)...")
    model = DualDinoV3(
        env_channels=env_channels, 
        num_classes=1, 
        use_skip_connections=True
    )
    
    # Run the forward pass
    print("Running forward pass...")
    # Disable gradients for the dummy test to save memory
    with torch.no_grad():
        output = model(dummy_rgb, dummy_env)
    
    # Verify shapes
    print("\n" + "=" * 50)
    print(f"RGB Input Shape:     {dummy_rgb.shape}")
    print(f"Env Input Shape:     {dummy_env.shape}")
    print(f"Final Output Shape:  {output.shape}")
    print("=" * 50)
    
    if output.shape == (batch_size, 1, height, width):
        print("SUCCESS: Tensors merged at the bottleneck and decoded perfectly!")
    else:
        print(f"ERROR: Expected shape ({batch_size}, 1, {height}, {width})")