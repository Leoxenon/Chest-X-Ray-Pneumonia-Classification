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
import matplotlib.pyplot as plt
import numpy as np

from data import get_data_loaders, ChestXRayDataset
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
    visualizations_dir = output_dir / 'visualizations'
    visualizations_dir.mkdir(exist_ok=True)
    
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
            
        # Generate periodic visualizations if enabled
        if hasattr(args, 'generate_periodic_visualizations') and args.generate_periodic_visualizations and epoch % args.visualization_interval == 0:
            print(f'\nGenerating visualizations for epoch {epoch}...')
            generate_training_visualizations(
                model, val_loader, device, visualizations_dir, epoch,
                args.num_visualizations_per_epoch
            )
    
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


def generate_training_visualizations(model, val_loader, device, output_dir, epoch, num_samples=5):
    """
    Generate Grad-CAM visualizations during training to monitor model learning.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        device: Device to run inference on
        output_dir: Directory to save visualizations
        epoch: Current epoch number
        num_samples: Number of samples to visualize
    """
    # Create directory for current epoch visualizations
    epoch_dir = output_dir / f'epoch_{epoch}'
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset with path information
    dataset_with_path = ChestXRayDataset(
        data_dir=val_loader.dataset.data_dir,
        split=val_loader.dataset.split,
        transform=val_loader.dataset.transform,
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
    
    # Generate Grad-CAM visualizations
    for idx, (dataset_idx, img_path) in enumerate(all_samples):
        try:
            # Simple Grad-CAM implementation
            import torch.nn.functional as F
            from PIL import Image
            from torchvision import transforms
            
            # Image preprocessing
            transform = transforms.Compose([
                transforms.Resize((val_loader.dataset.transform.transforms[0].size[0], 
                                 val_loader.dataset.transform.transforms[0].size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            
            # Load and preprocess image
            original_image = Image.open(img_path).convert('RGB')
            image = transform(original_image).unsqueeze(0).to(device)
            
            # Define target layer for visualization
            target_layer = None
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) and 'layer4' in name:
                    target_layer = name
                    break
            
            if not target_layer:
                print(f'Could not find suitable target layer for Grad-CAM')
                continue
            
            # Register hooks
            feature_maps = None
            gradients = None
            
            def forward_hook(module, input, output):
                nonlocal feature_maps
                feature_maps = output.detach()
            
            def backward_hook(module, grad_in, grad_out):
                nonlocal gradients
                gradients = grad_out[0].detach()
            
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
            _, predicted = output.max(1)
            
            # Backward pass
            one_hot = torch.zeros_like(output)
            one_hot[0, predicted.item()] = 1
            model.zero_grad()
            output.backward(gradient=one_hot, retain_graph=True)
            
            # Clean up hooks
            for handle in hook_handles:
                handle.remove()
            
            # Compute weights and generate CAM
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * feature_maps, dim=1)
            cam = F.relu(cam)
            
            # Resize CAM
            cam = F.interpolate(
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
            img_np = np.array(original_image.resize((image.shape[3], image.shape[2])))
            
            # Create heatmap
            heatmap = plt.cm.jet(cam_np)
            if heatmap.shape[2] == 4:
                heatmap = np.delete(heatmap, 3, 2)  # Remove alpha channel
            
            # Overlay heatmap
            overlay = img_np * 0.7 + heatmap[:, :, :3] * 255 * 0.3
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            
            # Create figure
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            ax1.imshow(img_np)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            ax2.imshow(cam_np, cmap='jet')
            ax2.set_title('Grad-CAM Heatmap')
            ax2.axis('off')
            
            ax3.imshow(overlay)
            ax3.set_title(f'Prediction: {predicted.item()}')
            ax3.axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            save_path = epoch_dir / f'sample_{idx}_{os.path.basename(img_path)}'
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f'Error generating visualization for {img_path}: {str(e)}')
    
    print(f'Epoch {epoch} visualizations saved to {epoch_dir}')


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
    
    # Visualization arguments
    parser.add_argument('--generate-periodic-visualizations', action='store_true',
                        help='Generate Grad-CAM visualizations during training')
    parser.add_argument('--visualization-interval', type=int, default=10,
                        help='Generate visualizations every N epochs (default: 10)')
    parser.add_argument('--num-visualizations-per-epoch', type=int, default=5,
                        help='Number of samples to visualize per epoch (default: 5)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Directory to save models and logs (default: checkpoints)')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
