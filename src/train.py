"""
Training script for pneumonia classification models.
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json

from data import get_data_loaders
from models.resnet_cbam import get_resnet_cbam


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer=None):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        writer: TensorBoard writer (optional)
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
        
        # Log to TensorBoard
        if writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    if writer is not None:
        writer.add_scalar('Train/EpochLoss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
    
    return epoch_loss


def validate(model, val_loader, criterion, device, epoch, writer=None):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        writer: TensorBoard writer (optional)
        
    Returns:
        Tuple of (average validation loss, validation accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (len(pbar) if len(pbar) > 0 else 1),
                'acc': 100. * correct / total if total > 0 else 0
            })
    
    epoch_loss = running_loss / len(val_loader) if len(val_loader) > 0 else 0
    epoch_acc = 100. * correct / total if total > 0 else 0
    
    if writer is not None:
        writer.add_scalar('Val/EpochLoss', epoch_loss, epoch)
        writer.add_scalar('Val/Accuracy', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc


def train(args):
    """
    Main training function.
    
    Args:
        args: Command-line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # Load data
    print('Loading data...')
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # Create model
    print(f'Creating {args.model} model...')
    if args.model == 'resnet_cbam':
        model = get_resnet_cbam(num_classes=args.num_classes, pretrained=args.pretrained)
    else:
        raise ValueError(f'Unknown model: {args.model}. Only "resnet_cbam" is supported.')
    
    model = model.to(device)
    
    # Loss and optimizer
    if args.class_weights:
        # Calculate class weights for imbalanced dataset
        class_counts = [0, 0]
        for _, label in train_loader.dataset:
            class_counts[label] += 1
        weights = torch.tensor([1.0 / c for c in class_counts])
        weights = weights / weights.sum()
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_acc = 0.0
    best_model_path = output_dir / 'best_model.pth'
    
    print('\nStarting training...')
    for epoch in range(1, args.epochs + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{args.epochs}')
        print(f'{"="*60}')
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer)
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f'Learning rate updated from {old_lr} to {new_lr}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_model_path)
            print(f'Best model saved with validation accuracy: {val_acc:.2f}%')
        
        # Save checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
    
    # Save final model
    final_model_path = output_dir / 'final_model.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_model_path)
    
    # Save training config
    config = vars(args)
    config['best_val_acc'] = best_val_acc
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    writer.close()
    print(f'\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%')
    print(f'Model saved to: {best_model_path}')


def main():
    parser = argparse.ArgumentParser(description='Train pneumonia classification model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet_cbam',
                        choices=['resnet_cbam'],
                        help='Model architecture (default: resnet_cbam)')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classes (default: 2)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights (default: True)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--class-weights', action='store_true',
                        help='Use class weights for imbalanced dataset')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Directory to save models and logs (default: checkpoints)')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
