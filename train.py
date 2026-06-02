"""
Training script for AQP4 CBA Classification
"""

import os
import yaml
import argparse
import numpy as np
import logging
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from model import get_model
from dataset import create_dataloaders
from losses import get_loss_function


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


def setup_logging(log_dir: str, is_master: bool) -> logging.Logger:
    """Setup logging"""
    logger = logging.getLogger(__name__)
    if not is_master:
        logger.setLevel(logging.WARNING)
        logger.addHandler(logging.NullHandler())
        return logger

    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed(backend: str) -> int:
    if dist.is_initialized():
        return dist.get_rank()

    dist.init_process_group(backend=backend, init_method='env://')
    return dist.get_rank()


def is_master_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


class Trainer:
    """Training wrapper"""
    
    def __init__(self, config: dict, device: torch.device, logger: logging.Logger, is_master: bool):
        self.config = config
        self.device = device
        self.logger = logger
        self.is_master = is_master
        
        self.output_dir = config.get('logging', {}).get('output_dir', 'outputs')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir) if is_master else NullWriter()
        
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

    def _gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return tensor

        world_size = dist.get_world_size()
        local_size = torch.tensor([tensor.numel()], device=tensor.device)
        sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(sizes, local_size)
        sizes = [int(s.item()) for s in sizes]
        max_size = max(sizes)

        if tensor.numel() < max_size:
            pad = torch.zeros(max_size - tensor.numel(), device=tensor.device, dtype=tensor.dtype)
            tensor = torch.cat([tensor.view(-1), pad], dim=0)
        else:
            tensor = tensor.view(-1)

        gathered = [torch.zeros(max_size, device=tensor.device, dtype=tensor.dtype) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        trimmed = [g[:sizes[i]] for i, g in enumerate(gathered)]
        return torch.cat(trimmed, dim=0)
    
    def train_epoch(self, train_loader) -> dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc="Training", disable=not self.is_master)
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
            
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.detach())
            all_targets.append(labels.detach())
            
            pbar.set_postfix({'loss': loss.item()})
        
        if dist.is_available() and dist.is_initialized():
            total_loss_tensor = torch.tensor([total_loss, total_samples], device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = total_loss_tensor[0].item()
            total_samples = int(total_loss_tensor[1].item())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_preds = self._gather_tensor(all_preds).cpu().numpy()
        all_targets = self._gather_tensor(all_targets).cpu().numpy()

        metrics = {
            'train_loss': total_loss / max(total_samples, 1),
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
        
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_probs = []
        all_targets = []
        
        pbar = tqdm(val_loader, desc="Evaluating", disable=not self.is_master)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(images)
            loss = self.loss_fn(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.detach())
            all_probs.append(probs[:, 1].detach())
            all_targets.append(labels.detach())
        
        if dist.is_available() and dist.is_initialized():
            total_loss_tensor = torch.tensor([total_loss, total_samples], device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = total_loss_tensor[0].item()
            total_samples = int(total_loss_tensor[1].item())

        all_preds = torch.cat(all_preds, dim=0)
        all_probs = torch.cat(all_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        all_preds = self._gather_tensor(all_preds).cpu().numpy()
        all_probs = self._gather_tensor(all_probs).cpu().numpy()
        all_targets = self._gather_tensor(all_targets).cpu().numpy()

        try:
            auc = roc_auc_score(all_targets, all_probs)
        except Exception:
            auc = 0.0
        
        metrics = {
            'val_loss': total_loss / max(total_samples, 1),
            'val_acc': accuracy_score(all_targets, all_preds),
            'val_precision': precision_score(all_targets, all_preds, zero_division=0),
            'val_recall': recall_score(all_targets, all_preds, zero_division=0),
            'val_f1': f1_score(all_targets, all_preds, zero_division=0),
            'val_auc': auc,
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save checkpoint"""
        if not self.is_master:
            return

        model_state = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
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
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

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
    parser.add_argument('--gpus', type=str, default=None, help='Comma-separated GPU ids, e.g. 0,2')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    parser.add_argument('--sync-bn', action='store_true', help='Use synchronized BatchNorm')
    
    args = parser.parse_args()

    if args.gpus:
        normalized = ','.join([item.strip() for item in args.gpus.split(',') if item.strip()])
        if normalized:
            os.environ['CUDA_VISIBLE_DEVICES'] = normalized
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    seed = config.get('data', {}).get('seed', 42)
    set_seed(seed)

    use_cuda = args.device == 'cuda' and torch.cuda.is_available()
    env_world_size = int(os.environ.get('WORLD_SIZE', '1'))
    use_distributed = args.distributed or env_world_size > 1

    if use_distributed:
        init_distributed(args.backend)
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
    else:
        device = torch.device('cuda' if use_cuda else 'cpu')

    is_master = is_master_process()
    if is_master:
        print(f"Device: {device}")

    logger = setup_logging(config.get('logging', {}).get('log_dir', 'outputs/logs'), is_master)

    augment_config = config.get('augmentation', {})
    logger.info("Augmentation config (active):\n%s", yaml.safe_dump(augment_config, sort_keys=False))
    
    negative_dir = os.path.join(args.data_dir, 'negative')
    positive_dir = os.path.join(args.data_dir, 'positive')
    
    model_type = config.get('mil', {}).get('enable', False) and 'mil' or 'baseline'
    train_loader, val_loader, test_loader = create_dataloaders(
        negative_dir,
        positive_dir,
        config,
        model_type=model_type,
        distributed=use_distributed,
        rank=dist.get_rank() if dist.is_available() and dist.is_initialized() else 0,
        world_size=dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1,
        seed=seed
    )
    
    trainer = Trainer(config, device, logger, is_master)
    if use_distributed:
        if args.sync_bn:
            trainer.model = nn.SyncBatchNorm.convert_sync_batchnorm(trainer.model)
        trainer.model = DDP(trainer.model, device_ids=[args.local_rank], output_device=args.local_rank)
    trainer.train(train_loader, val_loader)

    if use_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()