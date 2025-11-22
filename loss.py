import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMIRNetLoss(nn.Module):
    def __init__(self, det_weight=0.3, seg_weight=0.7, iou_weight=0.1):
        super().__init__()
        self.det_weight = det_weight
        self.seg_weight = seg_weight
        self.iou_weight = iou_weight
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice_loss = SoftDiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred_heatmap, pred_masks, pred_iou, gt_heatmap, gt_masks):
        det_loss = self.focal_loss(pred_heatmap, gt_heatmap)
        
        dice_loss = self.dice_loss(pred_masks, gt_masks)
        bce_loss = F.binary_cross_entropy(pred_masks, gt_masks)
        seg_loss = dice_loss + bce_loss
        
        if pred_iou is not None:
            intersection = (pred_masks * gt_masks).sum(dim=[1, 2, 3])
            union = pred_masks.sum(dim=[1, 2, 3]) + gt_masks.sum(dim=[1, 2, 3]) - intersection
            actual_iou = (intersection + 1e-6) / (union + 1e-6)
            iou_loss = F.mse_loss(pred_iou.squeeze(), actual_iou)
        else:
            iou_loss = torch.tensor(0.0, device=pred_masks.device)
        
        total_loss = (self.det_weight * det_loss + 
                     self.seg_weight * seg_loss + 
                     self.iou_weight * iou_loss)
        
        return total_loss, {
            'det_loss': det_loss.item(),
            'seg_loss': seg_loss.item()
        }

class FocalLoss(nn.Module):
    """
    Optimized Focal Loss with stable normalization
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        
        # Soft target handling
        pos_mask = (target >= 0.99).float()
        neg_mask = (target < 0.99).float()
        
        pos_loss = -self.alpha * torch.pow(1 - pred, self.gamma) * torch.log(pred) * pos_mask
        neg_loss = -(1 - self.alpha) * torch.pow(pred, self.gamma) * \
                   torch.log(1 - pred) * torch.pow(1 - target, 4) * neg_mask
        
        loss = pos_loss + neg_loss
        
        # Optimization: Normalize by mean (per pixel) or a stable count
        # Using mean() is much more stable than dividing by pos_mask.sum() when num_pos is 0 or small
        return loss.mean()

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class SoftIoULoss_Original(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, preds, gt_masks):
        pred = preds.contiguous().view(-1)
        target = gt_masks.contiguous().view(-1)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        loss = (intersection + 1) / (union + 1)
        return 1 - loss