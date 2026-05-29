"""
Loss functions for handling class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        class_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - class_probs) ** self.gamma
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross-Entropy Loss for class imbalance"""
    
    def __init__(self, weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weight, reduction=self.reduction)


def get_loss_function(config: dict, device: torch.device) -> nn.Module:
    """Create loss function based on configuration"""
    training_config = config.get('training', {})
    loss_type = training_config.get('loss', 'focal')
    
    class_weights = None
    if training_config.get('use_class_weights', True):
        negative_weight = training_config.get('negative_weight', 1.0)
        positive_weight = training_config.get('positive_weight', 1.69)
        class_weights = torch.tensor([negative_weight, positive_weight], 
                                    dtype=torch.float32, device=device)
        class_weights = class_weights / class_weights.sum() * 2
    
    if loss_type == 'focal':
        loss_fn = FocalLoss(
            alpha=training_config.get('focal_alpha', 0.25),
            gamma=training_config.get('focal_gamma', 2.0),
            weight=class_weights,
            reduction='mean'
        )
    elif loss_type == 'weighted_ce':
        loss_fn = WeightedCrossEntropyLoss(weight=class_weights, reduction='mean')
    elif loss_type == 'ce':
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss_fn.to(device)
