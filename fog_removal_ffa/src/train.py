import os
import time
import yaml
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model.ffa_net import FFA_Net
from utils.data_loader import get_data_loaders
from utils.losses import CombinedLoss
from utils.metrics import calculate_metrics

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, clip_grad=None):
    """
    Train for one epoch
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for parameter updates
        device: Device to run computations on
        epoch: Current epoch number
        clip_grad: Gradient clipping threshold (None to disable)
    
    Returns:
        Dictionary with average losses
    """
    model.train()
    running_loss = 0.0
    running_spatial_loss = 0.0
    running_edge_loss = 0.0
    
    # Progress bar for training
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}') as pbar:
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            hazy_images = batch['hazy'].to(device)
            clean_images = batch['clean'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(hazy_images)
            
            # Calculate loss
            loss, spatial_loss, edge_loss = criterion(outputs, clean_images)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (if enabled)
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            # Update weights
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            running_spatial_loss += spatial_loss.item()
            running_edge_loss += edge_loss.item()
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'spatial': f'{spatial_loss.item():.4f}',
                'edge': f'{edge_loss.item():.4f}'
            })
    
    # Calculate average losses
    avg_loss = running_loss / len(train_loader)
    avg_spatial_loss = running_spatial_loss / len(train_loader)
    avg_edge_loss = running_edge_loss / len(train_loader)
    
    return {
        'loss': avg_loss,
        'spatial_loss': avg_spatial_loss,
        'edge_loss': avg_edge_loss
    }

def validate(model, val_loader, criterion, device):
    """
    Validate the model
    
    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run computations on
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    running_loss = 0.0
    running_spatial_loss = 0.0
    running_edge_loss = 0.0
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Move data to device
            hazy_images = batch['hazy'].to(device)
            clean_images = batch['clean'].to(device)
            
            # Forward pass
            outputs = model(hazy_images)
            
            # Calculate loss
            loss, spatial_loss, edge_loss = criterion(outputs, clean_images)
            
            # Calculate metrics
            metrics = calculate_metrics(outputs, clean_images)
            
            # Update statistics
            running_loss += loss.item()
            running_spatial_loss += spatial_loss.item()
            running_edge_loss += edge_loss.item()
            psnr_values.append(metrics['PSNR'])
            ssim_values.append(metrics['SSIM'])
    
    # Calculate averages
    avg_loss = running_loss / len(val_loader)
    avg_spatial_loss = running_spatial_loss / len(val_loader)
    avg_edge_loss = running_edge_loss / len(val_loader)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    return {
        'loss': avg_loss,
        'spatial_loss': avg_spatial_loss,
        'edge_loss': avg_edge_loss,
        'PSNR': avg_psnr,
        'SSIM': avg_ssim
    }

def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train FFA-Net model for fog removal')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = FFA_Net(
        in_channels=config['model']['in_channels'],
        num_features=config['model']['num_features'],
        num_groups=config['model']['num_groups'],
        num_blocks=config['model']['num_blocks']
    ).to(device)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {num_params/1e6:.2f}M parameters")
    
    # Create data loaders
    train_loader, val_loader, _ = get_data_loaders(
        data_dir=config['dataset']['data_dir'],
        batch_size=config['dataset']['batch_size'],
        image_size=config['dataset']['image_size'],
        num_workers=config['dataset']['num_workers']
    )
    print(f"Data loaders created. Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")
    
    # Create criterion
    criterion = CombinedLoss(alpha=config['training']['edge_loss_weight']).to(device)
    
    # Create optimizer (start with Adam)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['training']['adam_betas'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create directories for checkpoints and logs
    os.makedirs(config['paths']['checkpoints_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=config['paths']['logs_dir'])
    
    # Training loop
    best_psnr = 0
    best_ssim = 0
    patience_counter = 0
    total_start_time = time.time()
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Switch optimizer from Adam to SGD if at specified epoch
        if epoch == config['training']['optimizer_switch_epoch']:
            print("Switching optimizer from Adam to SGD...")
            optimizer = optim.SGD(
                model.parameters(),
                lr=config['training']['learning_rate'] * 0.1,  # Reduce the learning rate for SGD
                momentum=config['training']['sgd_momentum'],
                weight_decay=config['training']['weight_decay']
            )
        
        # Train for one epoch
        epoch_start_time = time.time()
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch+1,
            clip_grad=config['training']['clip_gradient']
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start_time
        
        # Print validation metrics
        print(f"Validation - Loss: {val_metrics['loss']:.4f}, PSNR: {val_metrics['PSNR']:.2f} dB, SSIM: {val_metrics['SSIM']:.4f}")
        print(f"Epoch completed in {epoch_time:.1f}s")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Loss/train_spatial', train_metrics['spatial_loss'], epoch)
        writer.add_scalar('Loss/val_spatial', val_metrics['spatial_loss'], epoch)
        writer.add_scalar('Loss/train_edge', train_metrics['edge_loss'], epoch)
        writer.add_scalar('Loss/val_edge', val_metrics['edge_loss'], epoch)
        writer.add_scalar('Metrics/PSNR', val_metrics['PSNR'], epoch)
        writer.add_scalar('Metrics/SSIM', val_metrics['SSIM'], epoch)
        
        # Save checkpoint if PSNR improved
        if val_metrics['PSNR'] > best_psnr:
            best_psnr = val_metrics['PSNR']
            best_ssim = val_metrics['SSIM']
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                checkpoint_path=os.path.join(config['paths']['checkpoints_dir'], 'best_model.pth')
            )
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save regular checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                checkpoint_path=os.path.join(config['paths']['checkpoints_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            )
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"Early stopping after {patience_counter} epochs without improvement")
            break
    
    # Training complete
    total_time = time.time() - total_start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best PSNR: {best_psnr:.2f} dB, Best SSIM: {best_ssim:.4f}")
    
    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()