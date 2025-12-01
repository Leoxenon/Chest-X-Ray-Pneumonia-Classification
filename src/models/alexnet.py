"""
AlexNet for Pneumonia Classification (Historical Baseline).

AlexNet was the breakthrough CNN that won ImageNet 2012 competition.
While outdated by modern standards, it serves as an important historical baseline
to demonstrate the evolution of CNN architectures.

Reference:
- Krizhevsky et al. "ImageNet Classification with Deep CNNs" (NIPS 2012)
"""

import torch
import torch.nn as nn
from torchvision import models


class AlexNet(nn.Module):
    """
    AlexNet architecture for medical image classification.
    
    AlexNet (2012) was the first deep CNN to win ImageNet competition,
    marking the beginning of the deep learning era in computer vision.
    
    Architecture:
        - 5 convolutional layers
        - 3 fully connected layers
        - ReLU activation and dropout for regularization
        - Total: 8 weight layers
    
    Advantages:
        - Historic significance
        - Relatively fast training
    
    Disadvantages:
        - Large parameter count (~57M)
        - No skip connections (vanishing gradient issues)
        - Outdated compared to modern architectures
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        pretrained: Whether to use pretrained ImageNet weights (default: True)
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(AlexNet, self).__init__()
        
        # Load pretrained AlexNet from torchvision
        alexnet = models.alexnet(pretrained=pretrained)
        
        # Feature extraction layers (5 conv layers)
        self.features = alexnet.features
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = alexnet.avgpool
        
        # Classifier with modified output layer for binary classification
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
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


def get_alexnet(num_classes: int = 2, pretrained: bool = True) -> AlexNet:
    """
    Factory function to create an AlexNet model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained ImageNet weights
    
    Returns:
        AlexNet model
    """
    return AlexNet(num_classes=num_classes, pretrained=pretrained)


if __name__ == "__main__":
    """Test the model."""
    print("Testing AlexNet (Historical Baseline)")
    print("=" * 80)
    
    # Create model
    model = AlexNet(num_classes=2, pretrained=False)
    
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
    print("AlexNet (2012): Historic CNN breakthrough")
    print("8 layers, ~57M parameters")
