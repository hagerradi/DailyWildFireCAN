import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    """
    The Encoder module of a convolutional neural network that reduces 
    the spatial dimensions of the input image while increasing the depth.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        # Convolutional layers of the encoder with increasing depth and decreasing spatial dimensions
        self.conv1 = nn.Conv2d(48, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer to reduce dimensions
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass of the encoder. Applies a series of convolutions and downsampling.
        :param x: Input tensor to the encoder.
        :return: Transformed tensor with reduced spatial dimensions and increased depth.
        """
        x = F.relu(self.conv1(x))  # Apply convolution and activation function
        x = self.pool(x)  # Apply pooling to reduce dimensions
        x = F.relu(self.conv2(x))  # Further convolutions and pooling
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))  # Last convolution layer
        return x

# class Encoder(nn.Module):
#     """
#     The Encoder module of a convolutional neural network that reduces 
#     the spatial dimensions of the input image while increasing the depth.
#     """
#     def __init__(self, in_channels, hidden_dims=[128, 256, 256, 512]):
#         super(Encoder, self).__init__()
        
#         # We now use the arguments passed in, no hardcoded numbers!
#         self.conv1 = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(hidden_dims[2], hidden_dims[3], kernel_size=3, padding=1)
        
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer to reduce dimensions

#     def forward(self, x):
#         """
#         Forward pass of the encoder. Applies a series of convolutions and downsampling.
#         """
#         x = F.relu(self.conv1(x))  # Apply convolution and activation function
#         x = self.pool(x)           # Apply pooling to reduce dimensions
        
#         x = F.relu(self.conv2(x))  
#         x = self.pool(x)
        
#         x = F.relu(self.conv3(x))
#         x = self.pool(x)
        
#         x = F.relu(self.conv4(x))  # Last convolution layer
#         return x
