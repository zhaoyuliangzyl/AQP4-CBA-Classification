"""
Dataset and DataLoader for AQP4 CBA Classification
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import torch
from torch.utils.data import Dataset, DataLoader


class AQP4BaselineDataset(Dataset):
    """Dataset for baseline image classification"""
    
    def __init__(self, image_paths: List[str], labels: List[int],
                 image_size: Union[int, Tuple[int, int]] = 512,
                 augment: bool = False, normalize: bool = True,
                 augment_config: Optional[Dict] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment
        
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        
        augment_config = augment_config or {}

        if augment:
            transforms = [A.Resize(self.image_size[0], self.image_size[1])]

            if augment_config.get('random_flip', True):
                flip_prob = augment_config.get('flip_prob', 0.5)
                transforms.extend([
                    A.HorizontalFlip(p=flip_prob),
                    A.VerticalFlip(p=flip_prob),
                ])

            rotation_limit = augment_config.get('random_rotation', 15)
            rotation_prob = augment_config.get('rotation_prob', 0.5)
            if rotation_limit and rotation_prob > 0:
                transforms.append(A.Rotate(limit=rotation_limit, p=rotation_prob))

            exposure_transforms = []
            brightness_limit = augment_config.get('random_brightness', 0.2)
            contrast_limit = augment_config.get('random_contrast', 0.2)
            bc_prob = augment_config.get('brightness_contrast_prob', 0.5)
            if (brightness_limit or contrast_limit) and bc_prob > 0:
                exposure_transforms.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=brightness_limit,
                        contrast_limit=contrast_limit,
                        p=bc_prob
                    )
                )

            gamma_limit = augment_config.get('gamma_limit', [80, 120])
            gamma_prob = augment_config.get('random_gamma_prob', 0.3)
            if gamma_limit and gamma_prob > 0:
                exposure_transforms.append(
                    A.RandomGamma(gamma_limit=gamma_limit, p=gamma_prob)
                )

            clahe_config = augment_config.get('clahe', {})
            clahe_prob = clahe_config.get('prob', 0.3)
            if clahe_config.get('enable', True) and clahe_prob > 0:
                exposure_transforms.append(
                    A.CLAHE(
                        clip_limit=clahe_config.get('clip_limit', 2.0),
                        tile_grid_size=tuple(clahe_config.get('tile_grid_size', (8, 8))),
                        p=clahe_prob
                    )
                )

            tone_config = augment_config.get('tone_curve', {})
            tone_prob = tone_config.get('prob', 0.3)
            if tone_config.get('enable', True) and tone_prob > 0:
                exposure_transforms.append(
                    A.RandomToneCurve(scale=tone_config.get('scale', 0.1), p=tone_prob)
                )

            exposure_prob = augment_config.get('exposure_prob', 0.7)
            if exposure_transforms and exposure_prob > 0:
                transforms.append(A.OneOf(exposure_transforms, p=exposure_prob))

            saturation_limit = augment_config.get('random_saturation', 0.0)
            saturation_prob = augment_config.get('saturation_prob', 0.3)
            if saturation_limit and saturation_prob > 0:
                transforms.append(
                    A.HueSaturationValue(
                        hue_shift_limit=0,
                        sat_shift_limit=int(saturation_limit * 100),
                        val_shift_limit=0,
                        p=saturation_prob
                    )
                )

            if augment_config.get('gaussian_blur', True):
                blur_prob = augment_config.get('gaussian_blur_prob', 0.3)
                blur_limit = augment_config.get('gaussian_blur_limit', 3)
                if blur_prob > 0:
                    transforms.append(A.GaussBlur(blur_limit=blur_limit, p=blur_prob))

            transforms.extend([
                A.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]) if normalize else A.Normalize(),
                ToTensorV2(),
            ])

            self.transform = A.Compose(transforms)
        else:
            self.transform = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]) if normalize else A.Normalize(),
                ToTensorV2(),
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed = self.transform(image=image)
        image = transformed['image']
        
        return image, label


class AQP4MILDataset(Dataset):
    """Multiple Instance Learning Dataset with patch extraction"""
    
    def __init__(self, image_paths: List[str], labels: List[int],
                 patch_size: int = 512, patch_stride: int = 256,
                 resize_size: int = 512, max_patches: int = 16,
                 augment: bool = False, normalize: bool = True,
                 augment_config: Optional[Dict] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.resize_size = resize_size
        self.max_patches = max_patches
        
        augment_config = augment_config or {}

        if augment:
            transforms = [A.Resize(resize_size, resize_size)]

            if augment_config.get('random_flip', True):
                flip_prob = augment_config.get('flip_prob', 0.5)
                transforms.extend([
                    A.HorizontalFlip(p=flip_prob),
                    A.VerticalFlip(p=flip_prob),
                ])

            rotation_limit = augment_config.get('random_rotation', 15)
            rotation_prob = augment_config.get('rotation_prob', 0.5)
            if rotation_limit and rotation_prob > 0:
                transforms.append(A.Rotate(limit=rotation_limit, p=rotation_prob))

            exposure_transforms = []
            brightness_limit = augment_config.get('random_brightness', 0.2)
            contrast_limit = augment_config.get('random_contrast', 0.2)
            bc_prob = augment_config.get('brightness_contrast_prob', 0.5)
            if (brightness_limit or contrast_limit) and bc_prob > 0:
                exposure_transforms.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=brightness_limit,
                        contrast_limit=contrast_limit,
                        p=bc_prob
                    )
                )

            gamma_limit = augment_config.get('gamma_limit', [80, 120])
            gamma_prob = augment_config.get('random_gamma_prob', 0.3)
            if gamma_limit and gamma_prob > 0:
                exposure_transforms.append(
                    A.RandomGamma(gamma_limit=gamma_limit, p=gamma_prob)
                )

            clahe_config = augment_config.get('clahe', {})
            clahe_prob = clahe_config.get('prob', 0.3)
            if clahe_config.get('enable', True) and clahe_prob > 0:
                exposure_transforms.append(
                    A.CLAHE(
                        clip_limit=clahe_config.get('clip_limit', 2.0),
                        tile_grid_size=tuple(clahe_config.get('tile_grid_size', (8, 8))),
                        p=clahe_prob
                    )
                )

            tone_config = augment_config.get('tone_curve', {})
            tone_prob = tone_config.get('prob', 0.3)
            if tone_config.get('enable', True) and tone_prob > 0:
                exposure_transforms.append(
                    A.RandomToneCurve(scale=tone_config.get('scale', 0.1), p=tone_prob)
                )

            exposure_prob = augment_config.get('exposure_prob', 0.7)
            if exposure_transforms and exposure_prob > 0:
                transforms.append(A.OneOf(exposure_transforms, p=exposure_prob))

            saturation_limit = augment_config.get('random_saturation', 0.0)
            saturation_prob = augment_config.get('saturation_prob', 0.3)
            if saturation_limit and saturation_prob > 0:
                transforms.append(
                    A.HueSaturationValue(
                        hue_shift_limit=0,
                        sat_shift_limit=int(saturation_limit * 100),
                        val_shift_limit=0,
                        p=saturation_prob
                    )
                )

            if augment_config.get('gaussian_blur', True):
                blur_prob = augment_config.get('gaussian_blur_prob', 0.3)
                if blur_prob > 0:
                    transforms.append(A.GlassBlur(p=blur_prob))

            transforms.extend([
                A.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]) if normalize else A.Normalize(),
                ToTensorV2(),
            ])

            self.transform = A.Compose(transforms)
        else:
            self.transform = A.Compose([
                A.Resize(resize_size, resize_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]) if normalize else A.Normalize(),
                ToTensorV2(),
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        #image = cv2.imread(image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width = image.shape[:2]
        patches = []
        for y in range(0, height - self.patch_size + 1, self.patch_stride):
            for x in range(0, width - self.patch_size + 1, self.patch_stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                transformed = self.transform(image=patch)
                patch_tensor = transformed['image']
                patches.append(patch_tensor)
                
                if len(patches) >= self.max_patches:
                    break
            if len(patches) >= self.max_patches:
                break
        
        if len(patches) < self.max_patches:
            dummy_patch = torch.zeros(3, self.resize_size, self.resize_size)
            patches.extend([dummy_patch] * (self.max_patches - len(patches)))
        
        patches = torch.stack(patches[:self.max_patches], dim=0)
        return patches, label


def create_data_split(negative_dir: str, positive_dir: str,
                     train_split: float = 0.8, val_split: float = 0.1,
                     test_split: float = 0.1, seed: int = 42) -> Dict[str, Tuple[List[str], List[int]]]:
    """Create train/val/test splits from directory structure"""
    np.random.seed(seed)
    
    negative_paths = [
        os.path.join(negative_dir, f) for f in os.listdir(negative_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    positive_paths = [
        os.path.join(positive_dir, f) for f in os.listdir(positive_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    print(f"Found {len(negative_paths)} negative images")
    print(f"Found {len(positive_paths)} positive images")
    
    negative_labels = [0] * len(negative_paths)
    positive_labels = [1] * len(positive_paths)
    
    all_paths = negative_paths + positive_paths
    all_labels = negative_labels + positive_labels
    
    indices = np.random.permutation(len(all_paths))
    all_paths = [all_paths[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    train_idx = int(len(all_paths) * train_split)
    val_idx = train_idx + int(len(all_paths) * val_split)
    
    splits = {
        'train': (all_paths[:train_idx], all_labels[:train_idx]),
        'val': (all_paths[train_idx:val_idx], all_labels[train_idx:val_idx]),
        'test': (all_paths[val_idx:], all_labels[val_idx:])
    }
    
    print(f"\nData split:")
    print(f"  Train: {len(splits['train'][0])} samples")
    print(f"  Val:   {len(splits['val'][0])} samples")
    print(f"  Test:  {len(splits['test'][0])} samples")
    
    return splits


def create_dataloaders(negative_dir: str, positive_dir: str, config: dict,
                      model_type: str = "baseline") -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders"""
    data_config = config.get('data', {})
    image_config = config.get('image', {})
    augment_config = config.get('augmentation', {})
    
    splits = create_data_split(
        negative_dir, positive_dir,
        train_split=data_config.get('train_split', 0.8),
        val_split=data_config.get('val_split', 0.1),
        test_split=data_config.get('test_split', 0.1),
        seed=data_config.get('seed', 42)
    )
    
    training_config = config.get('training', {})
    batch_size = training_config.get('batch_size', 8)
    device_config = config.get('device', {})
    num_workers = device_config.get('num_workers', 4)
    pin_memory = device_config.get('pin_memory', True)
    
    if model_type == "baseline":
        train_dataset = AQP4BaselineDataset(
            splits['train'][0], splits['train'][1],
            image_size=image_config.get('resize_size', 512),
            augment=augment_config.get('enable', True),
            normalize=True,
            augment_config=augment_config
        )
        val_dataset = AQP4BaselineDataset(
            splits['val'][0], splits['val'][1],
            image_size=image_config.get('resize_size', 512),
            augment=False,
            normalize=True
        )
        test_dataset = AQP4BaselineDataset(
            splits['test'][0], splits['test'][1],
            image_size=image_config.get('resize_size', 512),
            augment=False,
            normalize=True
        )
    else:
        train_dataset = AQP4MILDataset(
            splits['train'][0], splits['train'][1],
            patch_size=image_config.get('patch_size', 512),
            patch_stride=image_config.get('patch_stride', 256),
            resize_size=image_config.get('resize_size', 512),
            max_patches=image_config.get('num_patches_max', 16),
            augment=augment_config.get('enable', True),
            normalize=True,
            augment_config=augment_config
        )
        val_dataset = AQP4MILDataset(
            splits['val'][0], splits['val'][1],
            patch_size=image_config.get('patch_size', 512),
            patch_stride=image_config.get('patch_stride', 256),
            resize_size=image_config.get('resize_size', 512),
            max_patches=image_config.get('num_patches_max', 16),
            augment=False,
            normalize=True
        )
        test_dataset = AQP4MILDataset(
            splits['test'][0], splits['test'][1],
            patch_size=image_config.get('patch_size', 512),
            patch_stride=image_config.get('patch_stride', 256),
            resize_size=image_config.get('resize_size', 512),
            max_patches=image_config.get('num_patches_max', 16),
            augment=False,
            normalize=True
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
