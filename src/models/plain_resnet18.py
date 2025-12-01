"""
Plain ResNet18 without attention mechanisms (Baseline Model).

This serves as the primary baseline to evaluate the impact of CBAM attention.
Uses pretrained ImageNet weights for fair comparison with ResNet18-CBAM.

Reference:
- He et al. "Deep Residual Learning for Image Recognition" (CVPR 2016)
"""

import torch
import torch.nn as nn
from torchvision import models


class PlainResNet18(nn.Module):
    """
    Standard ResNet18 without any attention mechanisms.
    
    This is a baseline model to compare against ResNet18-CBAM.
    The only difference is the absence of CBAM attention modules.
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        pretrained: Whether to use pretrained ImageNet weights (default: True)
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(PlainResNet18, self).__init__()
        
        # Load pretrained ResNet18 from torchvision
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Replace final FC layer for binary classification
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.resnet(x)
    
    def get_features(self, x):
        """
        Extract feature representations before the final classification layer.
        Useful for transfer learning or feature visualization.
        
        Args:
            x: Input tensor
        
        Returns:
            Feature tensor of shape (batch_size, 512)
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


def get_plain_resnet18(num_classes: int = 2, pretrained: bool = True) -> PlainResNet18:
    """
    Factory function to create a plain ResNet18 model without attention.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained ImageNet weights
    
    Returns:
        PlainResNet18 model
    """
    return PlainResNet18(num_classes=num_classes, pretrained=pretrained)


if __name__ == "__main__":
    """Test the model."""
    print("Testing Plain ResNet18 (Baseline)")
    print("=" * 80)
    
    # Create model
    model = PlainResNet18(num_classes=2, pretrained=False)
    
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
    print("Plain ResNet18: Standard ResNet without attention mechanisms")
    print("Use this as baseline to evaluate CBAM's contribution")
