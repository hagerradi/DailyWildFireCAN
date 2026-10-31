"""
Baselines: Classic U-Net encoder-decoder with optional skip connections.
"""
import torch
import torch.nn as nn

class UNetOlmo(nn.Module):
    def __init__(self, input_channels: int = 1, 
                 num_classes: int = 1, 
                 hidden_features: list = None, 
                 use_skip_connections: bool = True, 
                 use_activation_after_upsampling: bool = False,
                 use_olmo: bool = False, 
                 olmo_channels: int = 768):
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
        self.use_olmo = use_olmo

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # encoder block: downsampling
        in_ch = input_channels
        for h_feature in hidden_features:
            # 4 blocks: n_channelsx64, 64x128, 128x256, 256x512
            self.encoder.append(self._double_conv_block(in_ch, h_feature))
            in_ch = h_feature
        
        # # bottleneck block (bottom of U): 512x1024
        # self.bottleneck_layer = self._double_conv_block(hidden_features[-1], hidden_features[-1] * 2)

        # bottleneck block (bottom of U): 512x1024
        bottleneck_in_channels = hidden_features[-1]
        if self.use_olmo:
            bottleneck_in_channels += olmo_channels
            
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
    
    def forward(self, x: torch.Tensor, olmo_emb: torch.Tensor = None) -> torch.Tensor:
        skip_connections = []
        
        # Encoder: double conv block + maxpool
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.maxpool(x) # dowsnsample
        
        # # Bottleneck
        # x = self.bottleneck_layer(x)

        # Bottleneck Fusion
        if self.use_olmo and olmo_emb is not None:
            # Concatenate along the channel dimension (dim=1)
            x = torch.cat([x, olmo_emb], dim=1)
            
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