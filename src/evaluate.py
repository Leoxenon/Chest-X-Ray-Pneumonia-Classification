"""
Evaluation script for pneumonia classification models.
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import json
from PIL import Image
from torchvision import transforms

from data import get_data_loaders, ChestXRayDataset
from models.resnet_cbam import get_resnet_cbam


def evaluate_model(model, data_loader, device, class_names=['NORMAL', 'PNEUMONIA']):
    """
    Evaluate model and compute metrics.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to evaluate on
        class_names: List of class names
        
    Returns:
        Dictionary containing predictions, labels, and probabilities
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating'):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }


def compute_metrics(results, class_names=['NORMAL', 'PNEUMONIA']):
    """
    Compute and print evaluation metrics.
    
    Args:
        results: Dictionary containing predictions, labels, and probabilities
        class_names: List of class names
        
    Returns:
        Dictionary of computed metrics
    """
    predictions = results['predictions']
    labels = results['labels']
    probabilities = results['probabilities']
    
    # Accuracy
    accuracy = (predictions == labels).mean()
    
    # Classification report
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # ROC-AUC for binary classification
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall
        precision, recall, _ = precision_recall_curve(labels, probabilities[:, 1])
        avg_precision = average_precision_score(labels, probabilities[:, 1])
    else:
        roc_auc = None
        avg_precision = None
    
    metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'roc_auc': roc_auc,
        'average_precision': avg_precision
    }
    
    return metrics


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved to {save_path}')


def plot_roc_curve(labels, probabilities, save_path):
    """
    Plot and save ROC curve.
    
    Args:
        labels: True labels
        probabilities: Predicted probabilities for positive class
        save_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'ROC curve saved to {save_path}')


def plot_precision_recall_curve(labels, probabilities, save_path):
    """
    Plot and save Precision-Recall curve.
    
    Args:
        labels: True labels
        probabilities: Predicted probabilities for positive class
        save_path: Path to save the plot
    """
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    avg_precision = average_precision_score(labels, probabilities)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Precision-Recall curve saved to {save_path}')


class GradCAM:
    """
    Grad-CAM implementation for visualizing class activation maps.
    
    References:
    - Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Target convolutional layer for visualization
        """
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradient = None
        
        # Register hooks
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to extract feature maps and gradients."""
        # Forward hook to save feature maps
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()
        
        # Backward hook to save gradients
        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0].detach()
        
        # Find target layer and register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hook_handles.append(module.register_forward_hook(forward_hook))
                self.hook_handles.append(module.register_backward_hook(backward_hook))
                break
    
    def remove_hooks(self):
        """Remove registered hooks to clean up."""
        for handle in self.hook_handles:
            handle.remove()
    
    def __call__(self, x, class_idx=None):
        """
        Generate class activation map.
        
        Args:
            x: Input image tensor
            class_idx: Target class index. If None, uses the class with highest probability.
            
        Returns:
            Heatmap tensor of the same spatial dimensions as the input image.
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Forward pass
        x = x.requires_grad_()
        output = self.model(x)
        
        # If class_idx is not specified, use the class with highest probability
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # One-hot encode the target class
        one_hot = torch.zeros_like(output)
        one_hot[range(len(class_idx)), class_idx] = 1
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute weights using global average pooling on gradients
        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        
        # Compute weighted sum of feature maps
        cam = torch.sum(weights * self.feature_maps, dim=1)
        
        # Apply ReLU to ensure we only consider positive influences
        cam = F.relu(cam)
        
        # Resize CAM to match input image size
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        # Normalize CAM
        cam_max = torch.max(cam.view(cam.size(0), -1), dim=1)[0].view(-1, 1, 1)
        cam = cam / (cam_max + 1e-8)
        
        return cam


def visualize_grad_cam(model, image_path, device, class_names, img_size=224, target_layer='layer4.1.conv2'):
    """
    Generate and save Grad-CAM visualization for a single image.
    
    Args:
        model: PyTorch model
        image_path: Path to input image
        device: Device to run inference on
        class_names: List of class names
        img_size: Input image size
        target_layer: Target layer for Grad-CAM
        
    Returns:
        Tuple of (predicted_class, confidence, cam_image)
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
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Get model prediction
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
    
    # Generate CAM
    cam = grad_cam(image, predicted_class)
    
    # Clean up hooks
    grad_cam.remove_hooks()
    
    # Convert CAM to numpy array for visualization
    cam_np = cam.squeeze().cpu().numpy()
    
    # Prepare original image for visualization
    img_np = np.array(original_image.resize((img_size, img_size)))
    
    # Create heatmap
    heatmap = plt.cm.jet(cam_np)
    heatmap = np.delete(heatmap, 3, 2)  # Remove alpha channel
    
    # Overlay heatmap on image
    overlay = img_np * 0.7 + heatmap[:, :, :3] * 255 * 0.3
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Create figure with original image, heatmap, and overlay
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.imshow(img_np)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(cam_np, cmap='jet')
    ax2.set_title('Grad-CAM Heatmap')
    ax2.axis('off')
    
    ax3.imshow(overlay)
    ax3.set_title(f'Prediction: {class_names[predicted_class.item()]} ({confidence.item():.2f})')
    ax3.axis('off')
    
    plt.tight_layout()
    
    return class_names[predicted_class.item()], confidence.item(), fig


def generate_sample_visualizations(model, data_loader, device, class_names, output_dir, num_samples=10):
    """
    Generate Grad-CAM visualizations for sample images from the dataset.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation data
        device: Device to run inference on
        class_names: List of class names
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    # Create directory for visualizations
    viz_dir = output_dir / 'grad_cam_visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset with path information
    dataset_with_path = ChestXRayDataset(
        data_dir=data_loader.dataset.data_dir,
        split=data_loader.dataset.split,
        transform=data_loader.dataset.transform,
        return_path=True
    )
    
    # Select samples (balanced between classes)
    normal_samples = []
    pneumonia_samples = []
    
    for i in range(len(dataset_with_path)):
        _, label, path = dataset_with_path[i]
        if label == 0 and len(normal_samples) < num_samples // 2:
            normal_samples.append((i, path))
        elif label == 1 and len(pneumonia_samples) < num_samples - len(normal_samples):
            pneumonia_samples.append((i, path))
        
        if len(normal_samples) + len(pneumonia_samples) >= num_samples:
            break
    
    all_samples = normal_samples + pneumonia_samples
    
    print(f'Generating Grad-CAM visualizations for {len(all_samples)} samples...')
    
    # Generate visualizations
    for idx, (dataset_idx, img_path) in enumerate(all_samples):
        try:
            pred_class, confidence, fig = visualize_grad_cam(
                model, img_path, device, class_names
            )
            
            # Save visualization
            save_path = viz_dir / f'grad_cam_sample_{idx}_{os.path.basename(img_path)}'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f'Saved visualization for {img_path} -> {save_path}')
        except Exception as e:
            print(f'Error generating visualization for {img_path}: {str(e)}')
    
    print(f'All visualizations saved to {viz_dir}')


def print_metrics(metrics):
    """
    Print metrics in a readable format.
    
    Args:
        metrics: Dictionary of metrics
    """
    print('\n' + '='*60)
    print('EVALUATION RESULTS')
    print('='*60)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    
    if metrics['roc_auc'] is not None:
        print(f"ROC-AUC Score: {metrics['roc_auc']:.4f}")
        print(f"Average Precision: {metrics['average_precision']:.4f}")
    
    print('\nClassification Report:')
    print('-'*60)
    report = metrics['classification_report']
    
    for class_name in report:
        if class_name in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        print(f"\n{class_name}:")
        print(f"  Precision: {report[class_name]['precision']:.4f}")
        print(f"  Recall: {report[class_name]['recall']:.4f}")
        print(f"  F1-Score: {report[class_name]['f1-score']:.4f}")
        print(f"  Support: {report[class_name]['support']}")
    
    print(f"\nMacro Average:")
    print(f"  Precision: {report['macro avg']['precision']:.4f}")
    print(f"  Recall: {report['macro avg']['recall']:.4f}")
    print(f"  F1-Score: {report['macro avg']['f1-score']:.4f}")
    
    print(f"\nWeighted Average:")
    print(f"  Precision: {report['weighted avg']['precision']:.4f}")
    print(f"  Recall: {report['weighted avg']['recall']:.4f}")
    print(f"  F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    print('\n' + '='*60)


def evaluate(args):
    """
    Main evaluation function.
    
    Args:
        args: Command-line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print('Loading data...')
    _, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    
    # Select evaluation set
    if args.split == 'val':
        data_loader = val_loader
        print(f'Evaluating on validation set: {len(val_loader.dataset)} samples')
    else:
        data_loader = test_loader
        print(f'Evaluating on test set: {len(test_loader.dataset)} samples')
    
    # Load model
    print(f'Loading {args.model} model...')
    if args.model == 'resnet_cbam':
        model = get_resnet_cbam(num_classes=args.num_classes, pretrained=False)
    else:
        raise ValueError(f'Unknown model: {args.model}. Only "resnet_cbam" is supported.')
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f'Loaded checkpoint from epoch {checkpoint.get("epoch", "unknown")}')
    if 'val_acc' in checkpoint:
        print(f'Checkpoint validation accuracy: {checkpoint["val_acc"]:.2f}%')
    
    # Evaluate
    print('\nEvaluating model...')
    results = evaluate_model(model, data_loader, device, args.class_names)
    
    # Compute metrics
    metrics = compute_metrics(results, args.class_names)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save metrics to JSON
    metrics_file = output_dir / f'metrics_{args.split}.json'
    # Convert numpy arrays to lists for JSON serialization
    metrics_json = {
        'accuracy': float(metrics['accuracy']),
        'classification_report': metrics['classification_report'],
        'confusion_matrix': metrics['confusion_matrix'],
        'roc_auc': float(metrics['roc_auc']) if metrics['roc_auc'] is not None else None,
        'average_precision': float(metrics['average_precision']) if metrics['average_precision'] is not None else None
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics_json, f, indent=4)
    print(f'\nMetrics saved to {metrics_file}')
    
    # Plot confusion matrix
    cm_file = output_dir / f'confusion_matrix_{args.split}.png'
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        args.class_names,
        cm_file
    )
    
    # Plot ROC and PR curves (for binary classification)
    if args.num_classes == 2:
        roc_file = output_dir / f'roc_curve_{args.split}.png'
        plot_roc_curve(results['labels'], results['probabilities'][:, 1], roc_file)
        
        pr_file = output_dir / f'pr_curve_{args.split}.png'
        plot_precision_recall_curve(results['labels'], results['probabilities'][:, 1], pr_file)
    
    # Generate Grad-CAM visualizations if enabled
    if args.generate_grad_cam:
        print('\nGenerating Grad-CAM visualizations...')
        generate_sample_visualizations(
            model, data_loader, device, args.class_names, output_dir, args.num_visualizations
        )


def main():
    parser = argparse.ArgumentParser(description='Evaluate pneumonia classification model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Dataset split to evaluate (default: test)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        choices=['resnet_cbam'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classes (default: 2)')
    parser.add_argument('--class-names', type=str, nargs='+',
                        default=['NORMAL', 'PNEUMONIA'],
                        help='Class names (default: NORMAL PNEUMONIA)')
    
    # Visualization arguments
    parser.add_argument('--generate-grad-cam', action='store_true',
                        help='Generate Grad-CAM visualizations for lesion localization')
    parser.add_argument('--num-visualizations', type=int, default=10,
                        help='Number of sample visualizations to generate (default: 10)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='evaluation',
                        help='Directory to save evaluation results (default: evaluation)')
    
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
