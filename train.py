import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
from datetime import datetime
import random
from model import SalesDataset, create_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SMAPELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        """
        Calculate SMAPE loss
        pred: [batch_size, seq_len] or [batch_size * seq_len]
        target: [batch_size, seq_len] or [batch_size * seq_len]
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        numerator = torch.abs(pred - target)
        denominator = (torch.abs(pred) + torch.abs(target)) / 2 + self.epsilon
        
        smape = torch.mean(numerator / denominator) * 100
        
        return smape

def calculate_smape(pred, target, exclude_zeros=True):
    """Calculate SMAPE metric for evaluation
    
    Args:
        pred: predictions
        target: ground truth
        exclude_zeros: if True, exclude samples where target is 0 from SMAPE calculation
    """
    pred = pred.detach().cpu().numpy().flatten()
    target = target.detach().cpu().numpy().flatten()
    
    if exclude_zeros:
        # Filter out samples where target is 0
        non_zero_mask = target != 0
        if non_zero_mask.sum() == 0:
            return 0.0  # Return 0 if all targets are 0
        pred = pred[non_zero_mask]
        target = target[non_zero_mask]
    
    numerator = np.abs(pred - target)
    denominator = (np.abs(pred) + np.abs(target)) / 2 + 1e-8
    
    smape = np.mean(numerator / denominator) * 100
    
    return smape

def train_epoch(model, dataloader, optimizer, criterion, scheduler, device, teacher_forcing_ratio):
    model.train()
    total_loss = 0
    total_smape = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch_idx, batch_data in enumerate(progress_bar):
        inputs, targets, future_features, restaurant_idx, menu_idx = batch_data
        restaurant_idx = restaurant_idx.squeeze(1).to(device)
        menu_idx = menu_idx.squeeze(1).to(device)
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        if future_features is not None:
            future_features = future_features.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs, targets, teacher_forcing_ratio, future_features=future_features,
                       restaurant_idx=restaurant_idx, menu_idx=menu_idx)
        
        loss = criterion(outputs, targets)
        smape = calculate_smape(outputs, targets, exclude_zeros=False)  # Include zeros during training
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_smape += smape
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'SMAPE': f'{smape:.2f}%'
        })
    
    if scheduler is not None:
        scheduler.step()
    
    avg_loss = total_loss / len(dataloader)
    avg_smape = total_smape / len(dataloader)
    
    return avg_loss, avg_smape

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_smape = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Evaluating')
        
        for batch_data in progress_bar:
            inputs, targets, future_features, restaurant_idx, menu_idx = batch_data
            restaurant_idx = restaurant_idx.squeeze(1).to(device)
            menu_idx = menu_idx.squeeze(1).to(device)

            inputs = inputs.to(device)
            targets = targets.to(device)
            if future_features is not None:
                future_features = future_features.to(device)
            
            outputs = model(inputs, target=None, teacher_forcing_ratio=0, future_features=future_features,
                           restaurant_idx=restaurant_idx, menu_idx=menu_idx)
            
            loss = criterion(outputs, targets)
            smape = calculate_smape(outputs, targets, exclude_zeros=True)  # Exclude zeros during evaluation
            
            total_loss += loss.item()
            total_smape += smape
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'SMAPE': f'{smape:.2f}%'
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_smape = total_smape / len(dataloader)
    
    return avg_loss, avg_smape

def main(args):
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = pd.read_csv('data_preprocessed/train_preprocessed.csv')
    
    # Parse dates
    train_data['date'] = pd.to_datetime(train_data['date'])
    
    if args.no_eval:
        print("No-eval mode: Using all data for training")
        train_dataset = SalesDataset(
            train_data,
            input_seq_len=args.input_seq_len,
            output_seq_len=args.output_seq_len,
            stride=args.stride
        )
        val_dataset = None
    else:
        print("Eval mode: Splitting data into train/val")
        # Split by date - use data before 2023-11-01 for training
        train_mask = train_data['date'] < '2023-11-01'
        train_split = train_data[train_mask]
        val_split = train_data[~train_mask]
        
        train_dataset = SalesDataset(
            train_split,
            input_seq_len=args.input_seq_len,
            output_seq_len=args.output_seq_len,
            stride=args.stride
        )
        
        val_dataset = SalesDataset(
            val_split,
            input_seq_len=args.input_seq_len,
            output_seq_len=args.output_seq_len,
            stride=args.stride * 2  # Less dense validation samples
        )
    
    print(f"Train dataset size: {len(train_dataset)}")
    if val_dataset:
        print(f"Val dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Get number of features
    num_features = len(train_dataset.feature_cols)
    print(f"Number of features: {num_features}")
    
    # Create model
    model_config = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'input_seq_len': args.input_seq_len,
        'output_seq_len': args.output_seq_len
    }
    
    model = create_model(num_features, config=model_config, use_pretrained_embeddings=True)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = SMAPELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
    
    # Training loop
    best_val_smape = float('inf')
    patience_counter = 0
    teacher_forcing_ratio = args.teacher_forcing_start
    
    for epoch in range(args.epochs):    
        # Train
        train_loss, train_smape = train_epoch(
            model, train_loader, optimizer, criterion, 
            scheduler, device, teacher_forcing_ratio
        )
        print(f"Train - Loss: {train_loss:.4f}, SMAPE: {train_smape:.2f}%")
        
        # Validation
        if val_loader:
            val_loss, val_smape = evaluate(model, val_loader, criterion, device)
            print(f"Val - Loss: {val_loss:.4f}, SMAPE: {val_smape:.2f}%")
            
            # Early stopping
            if val_smape < best_val_smape:
                best_val_smape = val_smape
                patience_counter = 0
                
                # Save best model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_smape': val_smape,
                    'train_smape': train_smape,
                    'config': model_config
                }
                torch.save(checkpoint, f'best_model_seed{args.seed}.pt')
                print(f"Saved best model (Val SMAPE: {val_smape:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        else:
            # No validation, save every few epochs
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_smape': train_smape,
                    'config': model_config
                }
                torch.save(checkpoint, f'model_epoch{epoch+1}_seed{args.seed}.pt')
                print(f"Saved checkpoint at epoch {epoch+1}")
        
        # Decay teacher forcing ratio
        teacher_forcing_ratio = max(
            0,
            teacher_forcing_ratio - (args.teacher_forcing_start / args.epochs)
        )
    
    # Save final model
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_smape': train_smape,
        'config': model_config
    }
    torch.save(final_checkpoint, f'final_model_seed{args.seed}.pt')
    print(f"\nTraining completed. Final train SMAPE: {train_smape:.2f}%")
    
    if val_loader and not args.no_eval:
        print(f"Best validation SMAPE: {best_val_smape:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Sales Prediction Model')
    
    # Data parameters
    parser.add_argument('--input_seq_len', type=int, default=28, help='Input sequence length')
    parser.add_argument('--output_seq_len', type=int, default=7, help='Output sequence length')
    parser.add_argument('--stride', type=int, default=1, help='Sliding window stride')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--teacher_forcing_start', type=float, default=0.7, help='Initial teacher forcing ratio')
    parser.add_argument('--use_scheduler', action='store_false', help='Use learning rate scheduler')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Other parameters
    parser.add_argument('--no_eval', action='store_false', help='No evaluation mode - use all data for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
    
    args = parser.parse_args()
    
    main(args)