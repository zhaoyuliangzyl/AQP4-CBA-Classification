# AQP4 CBA Classification System

A comprehensive PyTorch-based medical image classification system for identifying positive and negative cells in AQP4 CBA (Aquaporin-4 Cell-Based Assay) microscopy images.

## Overview

This project addresses the challenge of automated binary classification (positive/negative) of high-resolution medical images with:
- **Data Imbalance Handling**: 169 negative vs 100 positive images
- **Weak Supervision**: Image-level labels despite partial cell positivity
- **Multiple Instance Learning (MIL)**: Patch-based processing for better feature extraction
- **Attention Mechanism**: Focus on important image regions

## Features

✅ **Robust Model Architectures**
- ResNet50/101, EfficientNet backbone options
- Attention-based feature aggregation
- Multiple Instance Learning support

✅ **Comprehensive Data Pipeline**
- Automatic train/val/test split
- Advanced data augmentation (rotation, brightness, contrast, blur)
- Patch-based processing for high-resolution images (2448×2048)

✅ **Imbalance Handling**
- Weighted cross-entropy loss
- Focal loss for hard example mining
- Class weight balancing

✅ **Production Ready**
- Complete training pipeline with early stopping
- Inference script for new images
- Batch prediction capability
- Comprehensive evaluation metrics

## Installation

### Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU support)

### Setup

```bash
# Clone repository
git clone <repository_url>
cd AQP4-CBA-Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

Organize your data in the following structure:
```
data/
├── negative/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── positive/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 2. Train Model

```bash
# Train with baseline model (default)
python train.py --config config.yaml --data-dir data --device cuda

# Or train with MIL model (uncomment in config.yaml: mil.enable: true)
python train.py --config config.yaml --data-dir data --device cuda
```

### 3. Evaluate

```bash
python evaluate.py --config config.yaml \
                   --model outputs/checkpoints/best_model.pth \
                   --data-dir data
```

### 4. Inference

**Single Image:**
```bash
python inference.py --config config.yaml \
                   --model outputs/checkpoints/best_model.pth \
                   --image path/to/image.jpg
```

**Batch Prediction:**
```bash
python inference.py --config config.yaml \
                   --model outputs/checkpoints/best_model.pth \
                   --image-dir path/to/image/directory \
                   --output results.json
```

## Configuration

Edit `config.yaml` to customize:

### Model Configuration
```yaml
model:
  backbone: "resnet50"          # Options: resnet50, resnet101, efficientnet_b0, efficientnet_b1
  pretrained: true              # Use ImageNet pretrained weights
  use_attention: true           # Use attention mechanism
  dropout: 0.3                  # Dropout rate
```

### Training Configuration
```yaml
training:
  batch_size: 8
  learning_rate: 0.001
  num_epochs: 100
  loss: "focal"                 # Options: "ce", "focal", "weighted_ce"
  optimizer: "adamw"
  use_class_weights: true
  positive_weight: 1.69         # 169/100 for data imbalance
```

### MIL Configuration (Optional)
```yaml
mil:
  enable: true                  # Enable MIL processing
  aggregation: "attention"      # Options: "mean", "max", "top_k", "attention"
  max_patches: 16              # Max patches per image
  patch_size: 512              # Patch size in pixels
  patch_stride: 256            # Stride between patches
```

## Model Architecture

### Baseline Model
```
Input Image (512×512)
    ↓
Backbone (ResNet50)
    ↓
Feature Extraction (2048-dim)
    ↓
Attention Module (optional)
    ↓
Classification Head (512→2)
    ↓
Output (negative/positive)
```

### MIL Model
```
Input Image (2448×2048)
    ↓
Patch Extraction (512×512 patches)
    ↓
Backbone Processing (each patch)
    ↓
Attention Aggregation
    ↓
Classification Head
    ↓
Image-level Prediction
```

## Training Results

The system is designed to achieve:
- **Accuracy**: >90% on balanced validation set
- **AUC**: >0.95 on test set
- **Sensitivity**: >85% (minimizes false negatives)
- **Specificity**: >90% (minimizes false positives)

Monitor training with TensorBoard:
```bash
tensorboard --logdir outputs/logs
```

## File Structure

```
AQP4-CBA-Classification/
├── config.yaml              # Configuration file
├── model.py                 # Model definitions
├── dataset.py               # Dataset and DataLoader
├── losses.py                # Loss functions
├── train.py                 # Training script
├── inference.py             # Inference script
├── evaluate.py              # Evaluation script
├── requirements.txt         # Dependencies
├── README.md                # This file
├── data/
│   ├── negative/            # Negative images
│   └── positive/            # Positive images
└── outputs/
    ├── checkpoints/         # Saved models
    ├── logs/                # TensorBoard logs
    └── evaluation/          # Evaluation results
```

## Key Features Explained

### 1. Data Imbalance Handling

The system addresses the 169:100 negative-to-positive ratio through:

**Class Weights**
```
weight_negative = 1.0
weight_positive = 1.69  (= 169/100)
```

**Focal Loss**
```
Loss = -α(1-p)^γ log(p)
```
- Reduces weight for easy examples
- Focuses training on hard negatives
- Especially effective for imbalanced data

### 2. Weak Supervision with MIL

For images with partial positivity (only some cells are positive):

1. **Patch Extraction**: Break image into 512×512 patches with 256-pixel stride
2. **Feature Extraction**: Process each patch through CNN backbone
3. **Attention Aggregation**: Learn to weight important patches
4. **Image-level Decision**: Aggregate patch decisions → image label

### 3. Attention Mechanism

```
For each patch:
  attention_score = FC_layer(patch_features)
  attention_weight = softmax(attention_score)

Image_features = sum(attention_weight * patch_features)
```

Interpretable: Visualize which patches drove the prediction

### 4. Data Augmentation

Applied to training data:
- **Spatial**: Rotation (±15°), Flip (H/V)
- **Photometric**: Brightness ±20%, Contrast ±20%
- **Smoothing**: Gaussian blur (p=0.3)

## Advanced Usage

### Custom Loss Function

Add to `losses.py`:
```python
class CustomLoss(nn.Module):
    def forward(self, logits, targets):
        # Your loss implementation
        return loss
```

Update `config.yaml`:
```yaml
training:
  loss: "custom"
```

### Transfer Learning

Load pretrained model:
```python
from model import get_model

model = get_model(config, model_type='baseline')
checkpoint = torch.load('outputs/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Fine-tune on new data
```

### Ensemble Predictions

Combine multiple models:
```python
models = [load_model(path1), load_model(path2), load_model(path3)]
predictions = [model(x) for model in models]
ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
```

## Troubleshooting

**Out of Memory (OOM)**
- Reduce batch size in `config.yaml`
- Use smaller model: `efficientnet_b0` instead of `resnet50`
- Enable patch-based processing: `mil.enable: true`

**Poor Validation Performance**
- Check data split ratio (`data.train_split`, `val_split`)
- Verify image preprocessing (normalization)
- Try different learning rates
- Enable augmentation: `augmentation.enable: true`

**Slow Training**
- Reduce image size: `image.resize_size: 256`
- Use fewer patches: `image.num_patches_max: 8`
- Reduce number of workers: `device.num_workers: 2`

## References

- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- **ResNet**: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling", ICML 2019
- **Multiple Instance Learning**: Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018

## License

This project is provided as-is for research and educational purposes.

## Contact

For issues or questions, please create an issue in the repository.

## Version History

- **v1.0**: Initial release with baseline and MIL models
