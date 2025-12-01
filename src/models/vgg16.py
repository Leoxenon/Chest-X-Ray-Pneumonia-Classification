"""
VGG16 for Pneumonia Classification (Historical Baseline).

VGG16 demonstrated that network depth is crucial for performance.
It uses very small (3x3) convolution filters throughout the network.

Reference:
- Simonyan & Zisserman "Very Deep CNNs for Large-Scale Image Recognition" (ICLR 2015)
"""

import torch
import torch.nn as nn
from torchvision import models


class VGG16(nn.Module):
    """
    VGG16 architecture for medical image classification.
    
    VGG16 (2014) demonstrated that increasing depth with small filters
    leads to better performance. It uses a simple and homogeneous architecture.
    
    Architecture:
        - 13 convolutional layers (all 3x3 filters)
        - 3 fully connected layers
        - 5 max pooling layers
        - Total depth: 16 weight layers
    
    Advantages:
        - Simple and homogeneous architecture
        - Small receptive fields with deep structure
        - Good feature learning capability
    
    Disadvantages:
        - Very large parameter count (~138M)
        - High computational cost
        - No skip connections (harder to train)
        - Memory intensive
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        pretrained: Whether to use pretrained ImageNet weights (default: True)
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(VGG16, self).__init__()
        
        # Load pretrained VGG16 from torchvision
        vgg = models.vgg16(pretrained=pretrained)
        
        # Feature extraction layers (13 conv layers)
        self.features = vgg.features
        
        # Adaptive pooling
        self.avgpool = vgg.avgpool
        
        # Classifier with modified output layer for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        """Forward pass through the network."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_features(self, x):
        """
        Extract feature representations before the final classification layer.
        
        Args:
            x: Input tensor
        
        Returns:
            Feature tensor of shape (batch_size, 4096)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def get_vgg16(num_classes: int = 2, pretrained: bool = True) -> VGG16:
    """
    Factory function to create a VGG16 model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained ImageNet weights
    
    Returns:
        VGG16 model
    """
    return VGG16(num_classes=num_classes, pretrained=pretrained)


if __name__ == "__main__":
    """Test the model."""
    print("Testing VGG16 (Historical Baseline)")
    print("=" * 80)
    
    # Create model
    model = VGG16(num_classes=2, pretrained=False)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        features = model.get_features(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Feature shape: {features.shape}")
    
    print("\n✓ Model test successful!")
    print("=" * 80)
    print("VGG16 (2014): Deep uniform architecture with 3x3 filters")
    print("16 layers, ~138M parameters")
