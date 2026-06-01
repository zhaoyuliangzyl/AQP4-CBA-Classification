# AQP4 CBA 分类系统

一个基于PyTorch的综合医学图像分类系统，用于识别AQP4 CBA（水通道蛋白-4细胞基检测）显微图像中的阳性和阴性细胞。

## 概述

本项目解决了高分辨率医学图像的自动二分类（阳性/阴性）挑战，具有以下特点：
- **数据不平衡处理**：169张阴性 vs 100张阳性图像
- **弱监督学习**：尽管只有部分细胞呈阳性，仍可使用图像级标签
- **多示例学习（MIL）**：基于图像块的处理，实现更好的特征提取
- **注意力机制**：聚焦于重要的图像区域

## 功能特性

✅ **强大的模型架构**
- ResNet50/101、EfficientNet等多种主干网络选择
- 基于注意力的特征聚合
- 多示例学习支持

✅ **完善的数据流程**
- 自动划分训练/验证/测试集
- 高级数据增强（旋转、亮度、对比度、模糊）
- 针对高分辨率图像（2448×2048）的基于图像块的处理

✅ **不平衡处理**
- 加权交叉熵损失
- 针对难例挖掘的Focal Loss
- 类别权重平衡

✅ **生产就绪**
- 完整的训练流程，支持早停
- 新图像的推理脚本
- 批量预测能力
- 全面的评估指标

## 安装

### 环境要求
- Python 3.8+
- CUDA 11.8+（用于GPU支持）

### 设置

```bash
# 克隆仓库
git clone <repository_url>
cd AQP4-CBA-Classification

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows系统: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 准备数据

按以下结构组织数据：
```
data/
├── negative/          # 阴性样本
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── positive/          # 阳性样本
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 2. 训练模型

```bash
# 使用基线模型训练（默认）
python train.py --config config.yaml --data-dir data --device cuda

# 或使用MIL模型训练（在config.yaml中取消注释：mil.enable: true）
python train.py --config config.yaml --data-dir data --device cuda
```

### 3. 评估模型

```bash
python evaluate.py --config config.yaml \
                   --model outputs/checkpoints/best_model.pth \
                   --data-dir data
```

### 4. 推理预测

**单张图像：**
```bash
python inference.py --config config.yaml \
                   --model outputs/checkpoints/best_model.pth \
                   --image path/to/image.jpg
```

**批量预测：**
```bash
python inference.py --config config.yaml --model outputs/checkpoints/best_model.pth --image-dir test_images --output results.json
```

## 配置说明

编辑 `config.yaml` 进行自定义设置：

### 模型配置
```yaml
model:
  backbone: "resnet50"          # 可选: resnet50, resnet101, efficientnet_b0, efficientnet_b1
  pretrained: true              # 使用ImageNet预训练权重
  use_attention: true           # 使用注意力机制
  dropout: 0.3                  # Dropout比率
```

### 训练配置
```yaml
training:
  batch_size: 8
  learning_rate: 0.001
  num_epochs: 100
  loss: "focal"                 # 可选: "ce", "focal", "weighted_ce"
  optimizer: "adamw"
  use_class_weights: true
  positive_weight: 1.69         # 169/100，应对数据不平衡
```

### MIL配置（可选）
```yaml
mil:
  enable: true                  # 启用MIL处理
  aggregation: "attention"      # 可选: "mean", "max", "top_k", "attention"
  max_patches: 16              # 每张图像最大图像块数
  patch_size: 512              # 图像块大小（像素）
  patch_stride: 256            # 图像块之间的步长
```

## 模型架构

### 基线模型
```
输入图像 (512×512)
    ↓
主干网络 (ResNet50)
    ↓
特征提取 (2048维)
    ↓
注意力模块（可选）
    ↓
分类头 (512→2)
    ↓
输出 (阴性/阳性)
```

### MIL模型
```
输入图像 (2448×2048)
    ↓
图像块提取 (512×512块)
    ↓
主干网络处理（每个图像块）
    ↓
注意力聚合
    ↓
分类头
    ↓
图像级预测
```

## 训练结果

系统设计目标是达到：
- **准确率**：在平衡的验证集上 >90%
- **AUC**：在测试集上 >0.95
- **灵敏度**：>85%（最小化假阴性）
- **特异性**：>90%（最小化假阳性）

使用TensorBoard监控训练：
```bash
tensorboard --logdir outputs/logs
```

## 文件结构

```
AQP4-CBA-Classification/
├── config.yaml              # 配置文件
├── model.py                 # 模型定义
├── dataset.py               # 数据集和数据加载器
├── losses.py                # 损失函数
├── train.py                 # 训练脚本
├── inference.py             # 推理脚本
├── evaluate.py              # 评估脚本
├── requirements.txt         # 依赖包列表
├── README.md                # 本文件
├── data/
│   ├── negative/            # 阴性图像
│   └── positive/            # 阳性图像
└── outputs/
    ├── checkpoints/         # 保存的模型
    ├── logs/                # TensorBoard日志
    └── evaluation/          # 评估结果
```

## 核心功能详解

### 1. 数据不平衡处理

系统通过以下方式处理169:100的阴性阳性比例：

**类别权重**
```
weight_negative = 1.0
weight_positive = 1.69  (= 169/100)
```

**Focal Loss**
```
Loss = -α(1-p)^γ log(p)
```
- 降低简单样本的权重
- 聚焦于难例的训练
- 对不平衡数据特别有效

### 2. 基于MIL的弱监督学习

针对部分阳性图像（只有部分细胞呈阳性）：

1. **图像块提取**：将图像分割成512×512的图像块，步长256像素
2. **特征提取**：每个图像块通过CNN主干网络处理
3. **注意力聚合**：学习重要图像块的权重
4. **图像级决策**：聚合图像块决策 → 图像标签

### 3. 注意力机制

```
对于每个图像块：
  attention_score = FC_layer(patch_features)
  attention_weight = softmax(attention_score)

图像特征 = sum(attention_weight * patch_features)
```

可解释性：可可视化哪些图像块驱动了预测

### 4. 数据增强

应用于训练数据：
- **空间变换**：旋转（±15°）、翻转（水平/垂直）
- **光度变换**：亮度±20%、对比度±20%
- **平滑处理**：高斯模糊（p=0.3）

## 高级用法

### 自定义损失函数

添加到 `losses.py`：
```python
class CustomLoss(nn.Module):
    def forward(self, logits, targets):
        # 你的损失函数实现
        return loss
```

更新 `config.yaml`：
```yaml
training:
  loss: "custom"
```

### 迁移学习

加载预训练模型：
```python
from model import get_model

model = get_model(config, model_type='baseline')
checkpoint = torch.load('outputs/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 在新数据上进行微调
```

### 集成预测

组合多个模型：
```python
models = [load_model(path1), load_model(path2), load_model(path3)]
predictions = [model(x) for model in models]
ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
```

## 故障排除

**显存不足（OOM）**
- 减少 `config.yaml` 中的批次大小
- 使用更小的模型：用 `efficientnet_b0` 替代 `resnet50`
- 启用基于图像块的处理：`mil.enable: true`

**验证集性能差**
- 检查数据划分比例（`data.train_split`、`val_split`）
- 验证图像预处理（归一化）
- 尝试不同的学习率
- 启用数据增强：`augmentation.enable: true`

**训练速度慢**
- 减小图像尺寸：`image.resize_size: 256`
- 使用更少的图像块：`image.num_patches_max: 8`
- 减少工作进程数：`device.num_workers: 2`

## 参考文献

- **Focal Loss**：Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- **ResNet**：He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- **EfficientNet**：Tan & Le, "EfficientNet: Rethinking Model Scaling", ICML 2019
- **多示例学习**：Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018

## 许可证

本项目按原样提供，用于研究和教育目的。

## 联系方式

如有问题，请在仓库中创建issue。

## 版本历史

- **v1.0**：初始版本，包含基线和MIL模型