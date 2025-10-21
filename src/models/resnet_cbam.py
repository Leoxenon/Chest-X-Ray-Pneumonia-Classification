"""
ResNet18 with CBAM (Convolutional Block Attention Module) for pneumonia classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM.
    
    Args:
        in_channels: Number of input channels
        reduction_ratio: Reduction ratio for the hidden layer
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM.
    
    Args:
        kernel_size: Kernel size for the convolution
    """
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise max and average pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Combines channel and spatial attention mechanisms.
    
    Args:
        in_channels: Number of input channels
        reduction_ratio: Reduction ratio for channel attention
        kernel_size: Kernel size for spatial attention
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # Apply channel attention
        x = x * self.channel_attention(x)
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        return x


class ResNet18CBAM(nn.Module):
    """
    ResNet18 with CBAM attention modules for pneumonia classification.
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        pretrained: Whether to use pretrained ResNet18 weights
        cbam_reduction_ratio: Reduction ratio for CBAM channel attention
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        cbam_reduction_ratio: int = 16
    ):
        super(ResNet18CBAM, self).__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Extract feature extraction layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # ResNet layers with CBAM
        self.layer1 = resnet.layer1
        self.cbam1 = CBAM(64, cbam_reduction_ratio)
        
        self.layer2 = resnet.layer2
        self.cbam2 = CBAM(128, cbam_reduction_ratio)
        
        self.layer3 = resnet.layer3
        self.cbam3 = CBAM(256, cbam_reduction_ratio)
        
        self.layer4 = resnet.layer4
        self.cbam4 = CBAM(512, cbam_reduction_ratio)
        
        # Global average pooling and classifier
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks with CBAM attention
        x = self.layer1(x)
        x = self.cbam1(x)
        
        x = self.layer2(x)
        x = self.cbam2(x)
        
        x = self.layer3(x)
        x = self.cbam3(x)
        
        x = self.layer4(x)
        x = self.cbam4(x)
        
        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_features(self, x):
        """
        Extract features before the final classification layer.
        Useful for multimodal fusion.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.cbam1(x)
        
        x = self.layer2(x)
        x = self.cbam2(x)
        
        x = self.layer3(x)
        x = self.cbam3(x)
        
        x = self.layer4(x)
        x = self.cbam4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


def get_resnet_cbam(num_classes: int = 2, pretrained: bool = True) -> ResNet18CBAM:
    """
    Factory function to create ResNet18 with CBAM.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        ResNet18CBAM model
    """
    return ResNet18CBAM(num_classes=num_classes, pretrained=pretrained)
