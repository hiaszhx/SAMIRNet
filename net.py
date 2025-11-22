import torch
import torch.nn as nn
from model import *


class NetworkWrapper(nn.Module):
    """
    Wrapper for different network architectures
    Compatible with BasicIRSTD framework
    """
    
    def __init__(self, model_name, mode='train'):
        super(NetworkWrapper, self).__init__()
        
        self.model_name = model_name
        self.mode = mode
        
        # Initialize model based on name
        if model_name == 'SAMIRNet_Tiny':
            self.model = SAMIRNet_Tiny(
                pretrained_path='./pretrained/swin_tiny_patch4_window7_224.pth'
            )
            self.is_samirnet = True
            
        elif model_name == 'SAMIRNet_Small':
            self.model = SAMIRNet_Small(
                pretrained_path='./pretrained/swin_small_patch4_window7_224.pth'
            )
            self.is_samirnet = True
            
        elif model_name == 'DNANet':
            self.model = DNANet(mode=mode)
            self.is_samirnet = False
            
        elif model_name == 'ACM':
            self.model = ACM()
            self.is_samirnet = False
            
        elif model_name == 'ALCNet':
            self.model = ALCNet()
            self.is_samirnet = False
            
        elif model_name == 'UIUNet':
            self.model = UIUNet()
            self.is_samirnet = False
            
        elif model_name == 'RDIAN':
            self.model = RDIAN()
            self.is_samirnet = False
            
        elif model_name == 'Unet':
            self.model = Unet()
            self.is_samirnet = False
            
        elif model_name == 'RISTDnet':
            self.model = RISTDnet()
            self.is_samirnet = False
            
        elif model_name == 'ISTDU_Net':
            self.model = ISTDU_Net()
            self.is_samirnet = False
            
        elif model_name == 'ResUNet':
            self.model = ResUNet()
            self.is_samirnet = False
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"Initialized {model_name}")
    
    def forward(self, images, gt_centers=None, gt_masks=None):
        """
        Forward pass
        
        Args:
            images: (B, C, H, W)
            gt_centers: (B, N, 2) - for SAMIRNet only
            gt_masks: (B, 1, H, W) - for SAMIRNet only
        
        Returns:
            For SAMIRNet: (pred_heatmap, pred_masks, pred_iou)
            For others: pred_masks
        """
        if self.is_samirnet:
            mode = 'train' if self.training else 'test'
            pred_heatmap, pred_masks, pred_iou = self.model(
                images, 
                gt_centers=gt_centers,
                gt_masks=gt_masks,
                mode=mode
            )
            return pred_heatmap, pred_masks, pred_iou
        else:
            # Original BasicIRSTD models
            pred_masks = self.model(images)
            return pred_masks
    
    def get_parameter_number(self):
        """Calculate total and trainable parameters"""
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
