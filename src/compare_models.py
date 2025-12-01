"""
Compare results from all trained models.

This script reads the metrics from all model evaluations and creates
a comprehensive comparison table.

Usage:
    python src/compare_models.py --evaluation-dir evaluation
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd


def load_metrics(eval_dir):
    """
    Load metrics from all models in the evaluation directory.
    
    Args:
        eval_dir: Path to evaluation directory
    
    Returns:
        Dictionary mapping model names to their metrics
    """
    eval_path = Path(eval_dir)
    all_metrics = {}
    
    # Expected model directories
    model_names = [
        'alexnet',
        'vgg16',
        'plain_resnet18',
        'resnet_cbam',
        'custom_resnet18',
        'custom_resnet_cbam'
    ]
    
    for model_name in model_names:
        metrics_file = eval_path / model_name / 'metrics_test.json'
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                all_metrics[model_name] = metrics
                print(f"✓ Loaded metrics for {model_name}")
        else:
            print(f"⚠ Metrics not found for {model_name}: {metrics_file}")
    
    return all_metrics


def create_comparison_table(all_metrics):
    """
    Create a comparison table from all model metrics.
    
    Args:
        all_metrics: Dictionary of model metrics
    
    Returns:
        pandas DataFrame with comparison
    """
    rows = []
    
    for model_name, metrics in all_metrics.items():
        # Extract metrics
        accuracy = metrics['accuracy'] * 100
        report = metrics['classification_report']
        
        # Get PNEUMONIA class metrics (class 1)
        pneumonia_metrics = report.get('PNEUMONIA', report.get('1', {}))
        
        row = {
            'Model': model_name,
            'Accuracy (%)': f"{accuracy:.2f}",
            'Precision (%)': f"{pneumonia_metrics.get('precision', 0) * 100:.2f}",
            'Recall (%)': f"{pneumonia_metrics.get('recall', 0) * 100:.2f}",
            'F1-Score (%)': f"{pneumonia_metrics.get('f1-score', 0) * 100:.2f}",
            'ROC-AUC (%)': f"{metrics.get('roc_auc', 0) * 100:.2f}" if metrics.get('roc_auc') else 'N/A'
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by accuracy (convert string to float for sorting)
    df['_acc_sort'] = df['Accuracy (%)'].astype(float)
    df = df.sort_values('_acc_sort', ascending=False).drop('_acc_sort', axis=1)
    
    return df


def print_comparison(df):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print("MODEL PERFORMANCE COMPARISON (Test Set)")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    # Highlight best performing model
    best_idx = df['Accuracy (%)'].astype(float).idxmax()
    best_model = df.loc[best_idx, 'Model']
    best_acc = df.loc[best_idx, 'Accuracy (%)']
    
    print(f"\n🏆 Best Model: {best_model} (Accuracy: {best_acc}%)")
    
    # Category analysis
    print("\n📊 Category Analysis:")
    print("-" * 100)
    
    # Historical baselines
    historical = df[df['Model'].isin(['alexnet', 'vgg16'])]
    if not historical.empty:
        print("\n📚 Historical Baselines (2012-2014):")
        print(historical.to_string(index=False))
    
    # ResNet family
    resnet_family = df[df['Model'].str.contains('resnet')]
    if not resnet_family.empty:
        print("\n🔬 ResNet Family (2015+):")
        print(resnet_family.to_string(index=False))
    
    # Attention comparison
    no_attention = df[df['Model'].isin(['plain_resnet18', 'custom_resnet18'])]
    with_attention = df[df['Model'].isin(['resnet_cbam', 'custom_resnet_cbam'])]
    
    if not no_attention.empty and not with_attention.empty:
        print("\n🎯 Attention Mechanism Impact:")
        print("  Without CBAM:")
        print(no_attention[['Model', 'Accuracy (%)']].to_string(index=False))
        print("  With CBAM:")
        print(with_attention[['Model', 'Accuracy (%)']].to_string(index=False))
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description='Compare trained model results')
    parser.add_argument('--evaluation-dir', type=str, default='evaluation',
                        help='Path to evaluation directory (default: evaluation)')
    parser.add_argument('--output', type=str, default='model_comparison.csv',
                        help='Output CSV file for comparison table (default: model_comparison.csv)')
    
    args = parser.parse_args()
    
    # Load metrics
    print("Loading metrics from all models...")
    all_metrics = load_metrics(args.evaluation_dir)
    
    if not all_metrics:
        print("\n❌ No metrics found. Please train and evaluate models first.")
        print("\nQuick start:")
        print("  1. Train models: python src/train.py --model <model_name> ...")
        print("  2. Evaluate models: python src/evaluate.py --model <model_name> ...")
        print("  3. Run comparison: python src/compare_models.py")
        return
    
    # Create comparison table
    print(f"\nComparing {len(all_metrics)} models...")
    df = create_comparison_table(all_metrics)
    
    # Print comparison
    print_comparison(df)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\n💾 Comparison table saved to: {args.output}")
    
    # Additional insights
    print("\n💡 Key Insights:")
    print("  • Architecture Evolution: AlexNet (2012) → VGG16 (2014) → ResNet18 (2015)")
    print("  • Attention Impact: Compare 'plain_resnet18' vs 'resnet_cbam'")
    print("  • Ablation Study: Compare 'custom_resnet18' vs 'custom_resnet_cbam'")
    print("  • Transfer Learning: Compare pretrained vs custom (from-scratch) models")


if __name__ == '__main__':
    main()
