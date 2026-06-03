"""
Evaluation script
"""

import os
import argparse
import torch
import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import get_model
from dataset import create_dataloaders


def evaluate_model(model, test_loader, device, model_type='baseline'):
    """Evaluate model"""
    model.eval()
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'f1': f1_score(all_targets, all_preds, zero_division=0),
    }
    
    try:
        metrics['roc_auc'] = roc_auc_score(all_targets, all_probs)
    except:
        metrics['roc_auc'] = 0.0
    
    metrics['confusion_matrix'] = confusion_matrix(all_targets, all_preds)
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    
    precision_curve, recall_curve, _ = precision_recall_curve(all_targets, all_probs)
    metrics['precision_curve'] = precision_curve
    metrics['recall_curve'] = recall_curve
    
    metrics['predictions'] = all_preds
    metrics['probabilities'] = all_probs
    metrics['targets'] = all_targets
    
    return metrics


def print_metrics(metrics):
    """Print metrics"""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    cm = metrics['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nSensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("="*50)


def plot_metrics(metrics, output_dir='outputs/evaluation'):
    """Plot metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(metrics['fpr'], metrics['tpr'], 'b-', linewidth=2, 
             label=f"AUC = {metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
    plt.close()
    
    # PR Curve
    pr_auc = auc(metrics['recall_curve'], metrics['precision_curve'])
    plt.figure(figsize=(8, 6))
    plt.plot(metrics['recall_curve'], metrics['precision_curve'], 'b-', linewidth=2,
             label=f"AUC = {pr_auc:.4f}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=150)
    plt.close()
    
    # Confusion Matrix
    cm = metrics['confusion_matrix']
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white', fontsize=16)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model', type=str, default='outputs/checkpoints/best_model.pth')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation')
    parser.add_argument('--gpus', type=str, default=None, help='Comma-separated GPU ids, e.g. 0,2')
    
    args = parser.parse_args()

    if args.gpus:
        normalized = ','.join([item.strip() for item in args.gpus.split(',') if item.strip()])
        if normalized:
            os.environ['CUDA_VISIBLE_DEVICES'] = normalized
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    negative_dir = os.path.join(args.data_dir, 'negative')
    positive_dir = os.path.join(args.data_dir, 'positive')
    
    model_type = config.get('mil', {}).get('enable', False) and 'mil' or 'baseline'
    _, _, test_loader = create_dataloaders(negative_dir, positive_dir, config, model_type=model_type)
    
    model = get_model(config, model_type=model_type).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {args.model}")
    
    metrics = evaluate_model(model, test_loader, device, model_type=model_type)
    print_metrics(metrics)
    plot_metrics(metrics, args.output_dir)


if __name__ == '__main__':
    main()