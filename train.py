"""
Training script for AQP4 CBA Classification
"""

import os
import yaml
import argparse
import numpy as np
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from model import get_model
from dataset import create_dataloaders
from losses import get_loss_function


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class Trainer:
    """Training wrapper"""
    
    def __init__(self, config: dict, device: torch.device, logger: logging.Logger):
        self.config = config
        self.device = device
        self.logger = logger
        
        self.output_dir = config.get('logging', {}).get('output_dir', 'outputs')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir)
        
        self.model_type = config.get('mil', {}).get('enable', False) and 'mil' or 'baseline'
        self.model = get_model(config, model_type=self.model_type).to(device)
        self.loss_fn = get_loss_function(config, device)
        
        training_config = config.get('training', {})
        lr = training_config.get('learning_rate', 0.001)
        weight_decay = training_config.get('weight_decay', 1e-5)
        
        self.optimizer = AdamW(self.model.parameters(), 
                       lr=float(lr), 
                       weight_decay=float(weight_decay))
        num_epochs = training_config.get('num_epochs', 100)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        self.num_epochs = num_epochs
        self.gradient_clip = training_config.get('gradient_clip', 1.0)
        self.early_stopping_patience = config.get('validation', {}).get('early_stopping_patience', 20)
        self.early_stopping_metric = config.get('validation', {}).get('early_stopping_metric', 'val_auc')
        self.best_metric = -np.inf
        self.patience_counter = 0
        self.eval_interval = config.get('validation', {}).get('eval_interval', 5)
        self.save_interval = config.get('logging', {}).get('save_interval', 5)
        
        self.logger.info(f"Model: {self.model_type}")
        self.logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader) -> dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(images)
            loss = self.loss_fn(logits, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        metrics = {
            'train_loss': total_loss / len(train_loader),
            'train_acc': accuracy_score(all_targets, all_preds),
            'train_precision': precision_score(all_targets, all_preds, zero_division=0),
            'train_recall': recall_score(all_targets, all_preds, zero_division=0),
            'train_f1': f1_score(all_targets, all_preds, zero_division=0),
        }
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, val_loader) -> dict:
        """Evaluate on validation set"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_probs = []
        all_targets = []
        
        pbar = tqdm(val_loader, desc="Evaluating")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(images)
            loss = self.loss_fn(logits, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
        
        try:
            auc = roc_auc_score(all_targets, all_probs)
        except:
            auc = 0.0
        
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_acc': accuracy_score(all_targets, all_preds),
            'val_precision': precision_score(all_targets, all_preds, zero_division=0),
            'val_recall': recall_score(all_targets, all_preds, zero_division=0),
            'val_f1': f1_score(all_targets, all_preds, zero_division=0),
            'val_auc': auc,
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics,
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved!")
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            train_metrics = self.train_epoch(train_loader)
            self.scheduler.step()
            
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Metrics/{key}', value, epoch)
                self.logger.info(f"  {key}: {value:.4f}")
            
            if (epoch + 1) % self.eval_interval == 0:
                val_metrics = self.evaluate(val_loader)
                
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'Metrics/{key}', value, epoch)
                    self.logger.info(f"  {key}: {value:.4f}")
                
                current_metric = val_metrics.get(self.early_stopping_metric, val_metrics.get('val_auc', 0))
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                else:
                    self.patience_counter += 1
                    self.logger.info(f"  Early stop: {self.patience_counter}/{self.early_stopping_patience}")
                    
                    if self.patience_counter >= self.early_stopping_patience:
                        self.logger.info("Early stopping!")
                        break
            
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch, train_metrics, is_best=False)
        
        self.logger.info("Training completed!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train AQP4 CBA Model')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    logger = setup_logging(config.get('logging', {}).get('log_dir', 'outputs/logs'))
    
    negative_dir = os.path.join(args.data_dir, 'negative')
    positive_dir = os.path.join(args.data_dir, 'positive')
    
    model_type = config.get('mil', {}).get('enable', False) and 'mil' or 'baseline'
    train_loader, val_loader, test_loader = create_dataloaders(
        negative_dir, positive_dir, config, model_type=model_type
    )
    
    trainer = Trainer(config, device, logger)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()