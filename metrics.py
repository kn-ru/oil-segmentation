"""
Метрики:
  Classification: macro-F1, per-class recall, accuracy, confusion matrix
  Segmentation:   oil IoU, oil Dice
  Negative control: false positive oil area на lookalike/no_oil
"""

import torch
import numpy as np
from typing import Dict, List
from collections import defaultdict


class MetricsAccumulator:
    """Накопитель метрик по батчам для одной эпохи."""

    def __init__(self, class_names: List[str] = None):
        if class_names is None:
            class_names = ["oil", "lookalike", "no_oil"]
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.reset()

    def reset(self):
        # Classification
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        # Segmentation (only for oil class)
        self.seg_tp = 0
        self.seg_fp = 0
        self.seg_fn = 0

        # False positive area on non-oil classes
        self.fp_area = defaultdict(list)  # class -> list of FP ratios

    @torch.no_grad()
    def update_classification(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            preds: (B,) predicted class indices
            labels: (B,) true class indices
        """
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        for p, t in zip(preds, labels):
            self.confusion[t, p] += 1

    @torch.no_grad()
    def update_segmentation(self, mask_logits: torch.Tensor,
                            mask_targets: torch.Tensor,
                            labels: torch.Tensor,
                            threshold: float = 0.5):
        """
        Args:
            mask_logits: (B, N, 1, H, W)
            mask_targets: (B, N, H, W)
            labels: (B,)
        """
        preds = (torch.sigmoid(mask_logits.squeeze(2)) > threshold).float()
        targets = mask_targets.float()

        B = labels.shape[0]
        for i in range(B):
            p = preds[i].reshape(-1)
            t = targets[i].reshape(-1)
            label = labels[i].item()

            if label == 0:  # oil class
                tp = ((p == 1) & (t == 1)).sum().item()
                fp = ((p == 1) & (t == 0)).sum().item()
                fn = ((p == 0) & (t == 1)).sum().item()
                self.seg_tp += tp
                self.seg_fp += fp
                self.seg_fn += fn
            else:
                # False positive ratio on non-oil
                fp_ratio = p.sum().item() / p.numel()
                cls_name = self.class_names[label]
                self.fp_area[cls_name].append(fp_ratio)

    def compute(self) -> Dict[str, float]:
        results = {}

        # Classification metrics
        cm = self.confusion
        total = cm.sum()
        results["accuracy"] = cm.trace() / total if total > 0 else 0

        # Per-class recall and precision
        recalls = []
        precisions = []
        f1s = []
        for i, name in enumerate(self.class_names):
            tp = cm[i, i]
            fn = cm[i].sum() - tp
            fp = cm[:, i].sum() - tp

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results[f"recall_{name}"] = recall
            results[f"precision_{name}"] = precision
            results[f"f1_{name}"] = f1
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)

        results["macro_f1"] = np.mean(f1s)
        results["macro_recall"] = np.mean(recalls)

        # Segmentation metrics (oil class only)
        iou = self.seg_tp / (self.seg_tp + self.seg_fp + self.seg_fn + 1e-7)
        dice = 2 * self.seg_tp / (2 * self.seg_tp + self.seg_fp + self.seg_fn + 1e-7)
        results["oil_iou"] = iou
        results["oil_dice"] = dice

        # False positive area on non-oil
        for cls_name in ["lookalike", "no_oil"]:
            if cls_name in self.fp_area and len(self.fp_area[cls_name]) > 0:
                results[f"fp_area_{cls_name}"] = np.mean(self.fp_area[cls_name])
            else:
                results[f"fp_area_{cls_name}"] = 0.0

        return results

    def confusion_matrix(self) -> np.ndarray:
        return self.confusion
