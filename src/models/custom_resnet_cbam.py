"""
Custom ResNet18 implementation with CBAM (Convolutional Block Attention Module).

This implementation builds ResNet18 from scratch instead of using torchvision's pretrained model.
It demonstrates deep understanding of ResNet architecture and provides full control over the model.

References:
- ResNet: He et al. "Deep Residual Learning for Image Recognition" (CVPR 2016)
- CBAM: Woo et al. "CBAM: Convolutional Block Attention Module" (ECCV 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM.
    
    Learns "what" features are important by applying both average and max pooling
    followed by a shared MLP to generate channel-wise attention weights.
    
    Args:
        in_channels: Number of input channels
        reduction_ratio: Reduction ratio for the MLP hidden layer (default: 16)
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP implemented with 1x1 convolutions
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Apply both average and max pooling
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        # Element-wise sum and sigmoid activation
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM.
    
    Learns "where" to focus by applying channel-wise pooling followed by
    a convolution to generate spatial attention weights.
    
    Args:
        kernel_size: Convolution kernel size (default: 7)
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
        
        # Concatenate along channel dimension and apply convolution
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Combines channel and spatial attention in a sequential manner:
    1. Channel attention refines feature channels
    2. Spatial attention refines spatial locations
    
    Args:
        in_channels: Number of input channels
        reduction_ratio: Reduction ratio for channel attention (default: 16)
        kernel_size: Kernel size for spatial attention (default: 7)
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # Sequential refinement: channel then spatial
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet18/34.
    
    Structure:
        x -> [3x3 conv -> BN -> ReLU -> 3x3 conv -> BN] -> + -> ReLU
        |_______________________________________________|
                        (skip connection)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the first convolution (default: 1)
        downsample: Downsampling layer for skip connection (optional)
    """
    
    expansion = 1  # Output channels multiplier (1 for BasicBlock, 4 for Bottleneck)
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super(BasicBlock, self).__init__()
        
        # First convolution (may downsample)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection (with downsampling if needed)
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Element-wise addition and activation
        out += identity
        out = self.relu(out)
        
        return out


class CustomResNet18(nn.Module):
    """
    Custom implementation of ResNet18 architecture from scratch.
    
    Architecture:
        - Initial conv: 7x7 conv, 64 channels, stride 2
        - MaxPool: 3x3, stride 2
        - Layer 1: 2 BasicBlocks, 64 channels
        - Layer 2: 2 BasicBlocks, 128 channels, stride 2 (downsampling)
        - Layer 3: 2 BasicBlocks, 256 channels, stride 2 (downsampling)
        - Layer 4: 2 BasicBlocks, 512 channels, stride 2 (downsampling)
        - AvgPool: Adaptive average pooling to 1x1
        - FC: Fully connected layer for classification
    
    Total layers: 1 + (2+2+2+2)*2 + 1 = 18 layers
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        in_channels: Number of input channels (default: 3 for RGB)
    """
    
    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super(CustomResNet18, self).__init__()
        
        self.in_channels = 64  # Track current number of channels
        
        # Initial convolution layer (7x7, stride 2)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Max pooling (3x3, stride 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers (each contains 2 BasicBlocks for ResNet18)
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        """
        Create a residual layer with multiple BasicBlocks.
        
        Args:
            out_channels: Number of output channels
            num_blocks: Number of BasicBlocks in this layer
            stride: Stride for the first block (for downsampling)
        
        Returns:
            Sequential container of BasicBlocks
        """
        downsample = None
        
        # Create downsampling layer if needed (when stride != 1 or channels change)
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * BasicBlock.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
        layers = []
        
        # First block (may downsample)
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        
        # Update in_channels for subsequent blocks
        self.in_channels = out_channels * BasicBlock.expansion
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Initialize model weights using He initialization for conv layers
        and constant initialization for batch norm layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization (optimal for ReLU activation)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # Constant initialization for BN
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Normal initialization for FC layers
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_features(self, x):
        """
        Extract feature representations before the final classification layer.
        Useful for transfer learning or feature visualization.
        
        Args:
            x: Input tensor
        
        Returns:
            Feature tensor of shape (batch_size, 512)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


class CustomResNet18CBAM(nn.Module):
    """
    Custom ResNet18 with CBAM attention modules integrated after each residual layer.
    
    This combines our custom ResNet18 implementation with CBAM attention mechanisms
    to enhance feature extraction for medical image classification.
    
    Architecture Enhancement:
        - After layer1 (64 channels): CBAM
        - After layer2 (128 channels): CBAM
        - After layer3 (256 channels): CBAM
        - After layer4 (512 channels): CBAM
    
    Args:
        num_classes: Number of output classes (default: 2)
        in_channels: Number of input channels (default: 3 for RGB)
        cbam_reduction_ratio: Reduction ratio for CBAM channel attention (default: 16)
        cbam_kernel_size: Kernel size for CBAM spatial attention (default: 7)
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        cbam_reduction_ratio: int = 16,
        cbam_kernel_size: int = 7
    ):
        super(CustomResNet18CBAM, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.cbam1 = CBAM(64, cbam_reduction_ratio, cbam_kernel_size)
        
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.cbam2 = CBAM(128, cbam_reduction_ratio, cbam_kernel_size)
        
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.cbam3 = CBAM(256, cbam_reduction_ratio, cbam_kernel_size)
        
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)
        self.cbam4 = CBAM(512, cbam_reduction_ratio, cbam_kernel_size)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        """Create a residual layer with multiple BasicBlocks."""
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass with CBAM attention after each residual layer."""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks with CBAM attention
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
        """Extract features before classification."""
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


def get_custom_resnet18(num_classes: int = 2, pretrained: bool = False) -> CustomResNet18:
    """
    Factory function to create custom ResNet18.
    
    Args:
        num_classes: Number of output classes
        pretrained: Not used (custom implementation doesn't support pretrained weights)
    
    Returns:
        CustomResNet18 model
    """
    if pretrained:
        print("Warning: Custom ResNet18 doesn't support pretrained weights. "
              "Training from scratch.")
    return CustomResNet18(num_classes=num_classes)


def get_custom_resnet_cbam(
    num_classes: int = 2,
    pretrained: bool = False,
    cbam_reduction_ratio: int = 16
) -> CustomResNet18CBAM:
    """
    Factory function to create custom ResNet18 with CBAM.
    
    Args:
        num_classes: Number of output classes
        pretrained: Not used (custom implementation trains from scratch)
        cbam_reduction_ratio: Reduction ratio for CBAM
    
    Returns:
        CustomResNet18CBAM model
    """
    if pretrained:
        print("Warning: Custom ResNet18-CBAM doesn't support pretrained weights. "
              "Training from scratch with He initialization.")
    return CustomResNet18CBAM(
        num_classes=num_classes,
        cbam_reduction_ratio=cbam_reduction_ratio
    )


if __name__ == "__main__":
    # Test the model
    print("Testing Custom ResNet18-CBAM Implementation")
    print("=" * 60)
    
    # Create model
    model = CustomResNet18CBAM(num_classes=2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Feature extraction
    features = model.get_features(dummy_input)
    print(f"Feature shape: {features.shape}")
    
    print("\n✓ Model test successful!")
    
    # Architecture summary
    print("\n" + "=" * 60)
    print("Architecture Summary:")
    print("=" * 60)
    print("Layer 1: 64 channels  -> CBAM")
    print("Layer 2: 128 channels -> CBAM")
    print("Layer 3: 256 channels -> CBAM")
    print("Layer 4: 512 channels -> CBAM")
    print("Total: 18 convolutional layers + 4 CBAM modules")
