import torch
import numpy as np

# -------------------------------------------------------------
# Helper Functions (PyTorch Version)
# -------------------------------------------------------------

def tensor_iou(pred, gt):
    """
    Batch-wise IoU calculation on GPU
    pred, gt: (B, H, W) or (H, W) Tensor
    """
    pred = (pred > 0.5).bool()
    gt = (gt > 0.5).bool()
    
    # Flatten HW dimensions for calculation
    if pred.dim() == 3:
        inter = (pred & gt).sum(dim=(1, 2)).float()
        union = (pred | gt).sum(dim=(1, 2)).float()
    else:
        inter = (pred & gt).sum().float()
        union = (pred | gt).sum().float()
        
    iou = inter / (union + 1e-6)
    return iou.mean()

# -------------------------------------------------------------
# Refactored MetricsCalculator
# -------------------------------------------------------------

class MetricsCalculator:
    """
    Accumulates TP, FP, FN, TN on GPU to avoid CPU transfer overhead.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.pixel_count = 0
        self.img_count = 0
        
        # 专门用于记录 nIoU (Mean IoU)
        self.cumulative_iou = 0.0
    
    @torch.no_grad()
    def update(self, pred, gt):
        """
        Args:
            pred: Tensor (B, 1, H, W) or (B, H, W) [0, 1] floats
            gt: Tensor (B, 1, H, W) or (B, H, W) [0, 1] ints/floats
        """
        # 统一维度
        if pred.dim() == 4: pred = pred.squeeze(1)
        if gt.dim() == 4: gt = gt.squeeze(1)
        
        pred_mask = (pred > 0.5)
        gt_mask = (gt > 0.5)
        
        # 计算混淆矩阵元素 (Batch level sum)
        tp = (pred_mask & gt_mask).sum()
        fp = (pred_mask & ~gt_mask).sum()
        fn = (~pred_mask & gt_mask).sum()
        tn = (~pred_mask & ~gt_mask).sum()
        
        # 累加全局统计量
        self.tp += tp.item()
        self.fp += fp.item()
        self.fn += fn.item()
        self.tn += tn.item()
        
        self.pixel_count += pred.numel()
        self.img_count += pred.shape[0]
        
        # 计算 Batch 内平均 IoU 用于 nIoU
        # 这里需要逐张图算 IoU 再求和
        batch_inter = (pred_mask & gt_mask).float().sum(dim=(1,2))
        batch_union = (pred_mask | gt_mask).float().sum(dim=(1,2))
        batch_iou = batch_inter / (batch_union + 1e-6)
        self.cumulative_iou += batch_iou.sum().item()

    def get_metrics(self):
        eps = 1e-6
        
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps) # Pd
        
        f1 = 2 * precision * recall / (precision + recall + eps)
        
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        
        # Normalized IoU (这里定义为 Mean IoU)
        niou = self.cumulative_iou / (self.img_count + eps)
        
        # False Alarm Rate (Per Million Pixels)
        fa = (self.fp / (self.pixel_count + eps)) * 1e6
        
        return {
            "IoU": iou,
            "nIoU": niou,
            "Pd": recall,        # Probability of Detection
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "FA": fa             # False Alarm
        }

    def print_metrics(self):
        metrics = self.get_metrics()
        print("\n" + "="*40)
        print("  Evaluation Metrics")
        print("="*40)
        print(f"  IoU       : {metrics['IoU']:.4f}")
        print(f"  nIoU      : {metrics['nIoU']:.4f}")
        print(f"  Pd (Rec)  : {metrics['Pd']:.4f}")
        print(f"  Precision : {metrics['Precision']:.4f}")
        print(f"  F1        : {metrics['F1']:.4f}")
        print(f"  FA        : {metrics['FA']:.4f}")
        print("="*40 + "\n")