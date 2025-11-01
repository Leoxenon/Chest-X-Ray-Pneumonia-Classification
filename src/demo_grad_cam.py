#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo script for Grad-CAM visualization of pneumonia lesion localization.

This script demonstrates how to use the Grad-CAM implementation to generate
class activation maps for chest X-ray images, visualizing where the model
focuses its attention when making predictions.
"""

import os
import argparse
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from models.resnet_cbam import get_resnet_cbam


def visualize_single_image_gradcam(
    model, 
    image_path, 
    device, 
    img_size=224,
    target_layer='layer4.1.conv2',
    class_names=['NORMAL', 'PNEUMONIA']
):
    """
    Generate and display Grad-CAM visualization for a single image.
    
    Args:
        model: PyTorch model
        image_path: Path to input image
        device: Device to run inference on
        img_size: Input image size
        target_layer: Target layer for Grad-CAM
        class_names: List of class names
    """
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    image = transform(original_image).unsqueeze(0).to(device)
    
    # Register hooks to extract feature maps and gradients
    feature_maps = None
    gradients = None
    
    def forward_hook(module, input, output):
        nonlocal feature_maps
        feature_maps = output.detach()
    
    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()
    
    # Find and register hooks on target layer
    hook_handles = []
    for name, module in model.named_modules():
        if name == target_layer:
            hook_handles.append(module.register_forward_hook(forward_hook))
            hook_handles.append(module.register_backward_hook(backward_hook))
            break
    
    # Forward pass
    model.eval()
    image.requires_grad_()
    output = model(image)
    probs = torch.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probs, 1)
    
    # Backward pass to compute gradients
    one_hot = torch.zeros_like(output)
    one_hot[0, predicted_class.item()] = 1
    model.zero_grad()
    output.backward(gradient=one_hot, retain_graph=True)
    
    # Clean up hooks
    for handle in hook_handles:
        handle.remove()
    
    # Compute Grad-CAM
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * feature_maps, dim=1)
    cam = torch.relu(cam)
    
    # Resize to input image size
    cam = torch.nn.functional.interpolate(
        cam.unsqueeze(1),
        size=image.shape[2:],
        mode='bilinear',
        align_corners=False
    ).squeeze(1)
    
    # Normalize
    cam_max = torch.max(cam)
    if cam_max > 0:
        cam = cam / cam_max
    
    # Convert to numpy for visualization
    cam_np = cam.squeeze().cpu().numpy()
    img_np = np.array(original_image.resize((img_size, img_size)))
    
    # Create heatmap
    heatmap = plt.cm.jet(cam_np)
    heatmap = np.delete(heatmap, 3, 2)  # Remove alpha channel
    
    # Overlay heatmap on image
    overlay = img_np * 0.7 + heatmap[:, :, :3] * 255 * 0.3
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Create visualization
    plt.figure(figsize=(18, 6))
    
    plt.subplot(131)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cam_np, cmap='jet')
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(overlay)
    plt.title(f'Prediction: {class_names[predicted_class.item()]} ({confidence.item():.2f})')
    plt.axis('off')
    
    plt.tight_layout()
    
    return plt.gcf()


def main():
    parser = argparse.ArgumentParser(description='Demo script for Grad-CAM visualization')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet_cbam',
                        choices=['resnet_cbam'],
                        help='Model architecture (default: resnet_cbam)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classes (default: 2)')
    parser.add_argument('--class-names', type=str, nargs='+',
                        default=['NORMAL', 'PNEUMONIA'],
                        help='Class names (default: NORMAL PNEUMONIA)')
    
    # Image arguments
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--target-layer', type=str, default='layer4.1.conv2',
                        help='Target layer for Grad-CAM (default: layer4.1.conv2)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='grad_cam_demo',
                        help='Directory to save visualization (default: grad_cam_demo)')
    parser.add_argument('--save-only', action='store_true',
                        help='Save visualization without displaying it')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f'Loading {args.model} model...')
    if args.model == 'resnet_cbam':
        model = get_resnet_cbam(num_classes=args.num_classes, pretrained=False)
    else:
        raise ValueError(f'Unknown model: {args.model}')
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f'Loaded checkpoint successfully')
    
    # Generate visualization
    print(f'Generating Grad-CAM visualization for {args.image_path}...')
    fig = visualize_single_image_gradcam(
        model,
        args.image_path,
        device,
        args.img_size,
        args.target_layer,
        args.class_names
    )
    
    # Save visualization
    output_file = output_dir / f'grad_cam_{Path(args.image_path).stem}.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Visualization saved to {output_file}')
    
    # Display if not save-only mode
    if not args.save_only:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    main()