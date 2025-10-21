"""
Evaluation script for pneumonia classification models.
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
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

from data import get_data_loaders
from models.resnet_cbam import get_resnet_cbam
from models.multimodal import get_multimodal_model


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
    elif args.model == 'multimodal':
        model = get_multimodal_model(
            num_classes=args.num_classes,
            pretrained_vision=False,
            fusion_method=args.fusion_method
        )
    else:
        raise ValueError(f'Unknown model: {args.model}')
    
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
                        choices=['resnet_cbam', 'multimodal'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classes (default: 2)')
    parser.add_argument('--fusion-method', type=str, default='concat',
                        choices=['concat', 'add', 'attention'],
                        help='Fusion method for multimodal model (default: concat)')
    parser.add_argument('--class-names', type=str, nargs='+',
                        default=['NORMAL', 'PNEUMONIA'],
                        help='Class names (default: NORMAL PNEUMONIA)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='evaluation',
                        help='Directory to save evaluation results (default: evaluation)')
    
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
