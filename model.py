"""
AQP4 CBA Classification Model
Includes backbone networks with attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional


class AttentionModule(nn.Module):
    """Self-attention module for focusing on important regions"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super(AttentionModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, num_patches, feature_dim)
        Returns:
            aggregated_features: (batch_size, feature_dim)
            attention_weights: (batch_size, num_patches)
        """
        att = torch.tanh(self.fc1(x))
        att = self.fc2(att)
        att = att.squeeze(-1)
        att_weights = F.softmax(att, dim=1)
        aggregated = torch.sum(x * att_weights.unsqueeze(-1), dim=1)
        return aggregated, att_weights


class BaselineModel(nn.Module):
    """Baseline CNN model with optional attention mechanism"""
    
    def __init__(self, backbone: str = "resnet50", num_classes: int = 2, 
                 dropout: float = 0.3, use_attention: bool = False, 
                 attention_dim: int = 256, pretrained: bool = True):
        super(BaselineModel, self).__init__()
        
        self.use_attention = use_attention
        self.num_classes = num_classes
        
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        elif backbone == "efficientnet_b1":
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            feature_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.backbone.fc = nn.Identity()
        self.feature_dim = feature_dim
        self.dropout = nn.Dropout(dropout)
        
        if use_attention:
            self.attention = AttentionModule(feature_dim, attention_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.classifier(features)
        
        if return_features:
            return features, logits
        return logits


class MILModel(nn.Module):
    """Multiple Instance Learning Model"""
    
    def __init__(self, backbone: str = "resnet50", num_classes: int = 2,
                 dropout: float = 0.3, use_attention: bool = True,
                 attention_dim: int = 256, pretrained: bool = True,
                 mil_aggregation: str = "attention", top_k: int = 5):
        super(MILModel, self).__init__()
        
        self.num_classes = num_classes
        self.mil_aggregation = mil_aggregation
        self.top_k = top_k
        
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        elif backbone == "efficientnet_b1":
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            feature_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.backbone.fc = nn.Identity()
        self.feature_dim = feature_dim
        self.dropout_layer = nn.Dropout(dropout)
        
        if use_attention and mil_aggregation == "attention":
            self.attention = AttentionModule(feature_dim, attention_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor, return_attention_weights: bool = False) -> torch.Tensor:
        batch_size, num_patches, c, h, w = x.shape
        x_reshaped = x.view(batch_size * num_patches, c, h, w)
        patch_features = self.backbone(x_reshaped)
        patch_features = self.dropout_layer(patch_features)
        patch_features = patch_features.view(batch_size, num_patches, self.feature_dim)
        
        if self.mil_aggregation == "mean":
            aggregated_features = torch.mean(patch_features, dim=1)
        elif self.mil_aggregation == "max":
            aggregated_features, _ = torch.max(patch_features, dim=1)
        elif self.mil_aggregation == "attention":
            aggregated_features, attention_weights = self.attention(patch_features)
        else:
            aggregated_features = torch.mean(patch_features, dim=1)
        
        logits = self.classifier(aggregated_features)
        
        if return_attention_weights and self.mil_aggregation == "attention":
            return logits, attention_weights
        return logits


def get_model(config: dict, model_type: str = "baseline") -> nn.Module:
    """Factory function to create models"""
    model_config = config.get("model", {})
    
    if model_type == "baseline":
        model = BaselineModel(
            backbone=model_config.get("backbone", "resnet50"),
            num_classes=model_config.get("num_classes", 2),
            dropout=model_config.get("dropout", 0.3),
            use_attention=model_config.get("use_attention", True),
            attention_dim=model_config.get("attention_dim", 256),
            pretrained=model_config.get("pretrained", True)
        )
    elif model_type == "mil":
        mil_config = config.get("mil", {})
        model = MILModel(
            backbone=model_config.get("backbone", "resnet50"),
            num_classes=model_config.get("num_classes", 2),
            dropout=model_config.get("dropout", 0.3),
            use_attention=model_config.get("use_attention", True),
            attention_dim=model_config.get("attention_dim", 256),
            pretrained=model_config.get("pretrained", True),
            mil_aggregation=mil_config.get("aggregation", "attention"),
            top_k=mil_config.get("top_k", 5)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
