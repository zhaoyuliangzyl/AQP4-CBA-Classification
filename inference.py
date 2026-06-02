"""
Inference script for predictions
"""

import os
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import get_model


class Predictor:
    """Inference wrapper"""
    
    def __init__(self, config_path: str, model_path: str, device: str = 'cuda'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        self.model_type = self.config.get('mil', {}).get('enable', False) and 'mil' or 'baseline'
        self.model = get_model(self.config, model_type=self.model_type).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded: {self.model_type}")
        
        image_config = self.config.get('image', {})
        self.resize_size = image_config.get('resize_size', 512)
        self.patch_size = image_config.get('patch_size', 512)
        self.patch_stride = image_config.get('patch_stride', 256)
        self.max_patches = image_config.get('num_patches_max', 16)
        
        self.transform = A.Compose([
            A.Resize(self.resize_size, self.resize_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    @torch.no_grad()
    def predict(self, image_path: str) -> Dict:
        """Predict label for single image"""
        # image = cv2.imread(image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Cannot load: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.model_type == 'baseline':
            logits = self._predict_baseline(image)
        else:
            logits = self._predict_mil(image)
        
        logits = logits.to(self.device)
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(logits, dim=1).item()
        confidence = probs[0, pred_label].item()
        
        return {
            'label': pred_label,
            'label_name': 'positive' if pred_label == 1 else 'negative',
            'confidence': confidence,
            'probabilities': {
                'negative': probs[0, 0].item(),
                'positive': probs[0, 1].item()
            }
        }
    
    def _predict_baseline(self, image: np.ndarray) -> torch.Tensor:
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        logits = self.model(image_tensor)
        return logits
    
    def _predict_mil(self, image: np.ndarray) -> torch.Tensor:
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
        patches = patches.unsqueeze(0).to(self.device)
        logits = self.model(patches)
        return logits
    
    def predict_batch(self, image_dir: str) -> Dict[str, Dict]:
        """Predict for all images in directory"""
        results = {}
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(image_dir).glob(ext))
            # image_files.extend(Path(image_dir).glob(ext.upper()))
        
        print(f"Found {len(image_files)} images")
        
        for image_path in image_files:
            try:
                result = self.predict(str(image_path))
                results[image_path.name] = result
                safe_name = image_path.name.encode(
                    "utf-8",
                    errors="replace"
                ).decode("utf-8")

                print(f"{safe_name}: {result['label_name']} ({result['confidence']:.4f})")
            except Exception as e:
                print(f"Error: {image_path.name}: {e}")
                results[image_path.name] = {'error': str(e)}
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model', type=str, default='outputs/checkpoints/best_model.pth')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--image-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--gpus', type=str, default=None, help='Comma-separated GPU ids, e.g. 0,2')
    
    args = parser.parse_args()

    if args.gpus:
        normalized = ','.join([item.strip() for item in args.gpus.split(',') if item.strip()])
        if normalized:
            os.environ['CUDA_VISIBLE_DEVICES'] = normalized
    
    predictor = Predictor(args.config, args.model, args.device)
    
    if args.image:
        result = predictor.predict(args.image)
        print(f"\nPrediction for {args.image}:")
        print(f"  Label: {result['label_name']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probs: {result['probabilities']}")
    
    elif args.image_dir:
        results = predictor.predict_batch(args.image_dir)
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()