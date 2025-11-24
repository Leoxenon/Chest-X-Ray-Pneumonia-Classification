"""
Neural network models for pneumonia classification.

Available Models:
    1. ResNet18CBAM - Standard implementation using pretrained ResNet18
    2. CustomResNet18CBAM - Custom implementation built from scratch
"""

from .resnet_cbam import ResNet18CBAM, get_resnet_cbam
from .custom_resnet_cbam import (
    CustomResNet18,
    CustomResNet18CBAM,
    get_custom_resnet18,
    get_custom_resnet_cbam
)

__all__ = [
    'ResNet18CBAM',
    'get_resnet_cbam',
    'CustomResNet18',
    'CustomResNet18CBAM',
    'get_custom_resnet18',
    'get_custom_resnet_cbam',
]