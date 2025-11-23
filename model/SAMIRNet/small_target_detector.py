import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallTargetDetector(nn.Module):
    """
    Dense Feature Aggregation Detector
    专为红外小目标设计：保留所有层级的特征，通过密集连接增强对微小目标的感知。
    """
    def __init__(self, in_channels=256, num_classes=1):
        super().__init__()
        
        # 融合层：输入是4层特征的拼接 (256 * 4)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # 使用 3x3 卷积进行空间融合
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 上下文增强 (Dilated Conv) 扩大感受野，区分背景噪声
        self.context = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # 预测头
        self.head = nn.Sequential(
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        # features: [P1, P2, P3, P4]
        # 统一上采样到 P1 尺寸 (1/4)
        target_size = features[0].shape[2:]
        upsampled_feats = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled_feats.append(feat)
            
        # 密集拼接
        cat_feat = torch.cat(upsampled_feats, dim=1)
        
        # 融合与预测
        x = self.fusion_conv(cat_feat)
        x = self.context(x)
        heatmap = self.head(x)
        
        # 恢复到原图尺寸
        heatmap = F.interpolate(heatmap, scale_factor=4, mode='bilinear', align_corners=False)
        return heatmap, x