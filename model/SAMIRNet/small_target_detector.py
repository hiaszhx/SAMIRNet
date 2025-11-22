import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallTargetDetector(nn.Module):
    """
    专为红外小目标设计的探测头 (Small Target Detector)
    
    架构特点：
    1. 非 FPN 结构：不使用自顶向下的逐级加和。
    2. 密集聚合 (Dense Aggregation)：将所有层级的特征统一上采样并拼接，
       最大程度保留小目标在浅层的高频空间信息。
    3. 上下文增强 (Context Enhancement)：使用膨胀卷积扩大感受野，
       帮助区分真实目标和高亮噪声。
    4. 像素注意力 (Pixel Attention)：进一步抑制背景杂波。
    """
    def __init__(self, in_channels=256, num_classes=1):
        super().__init__()
        
        # 1. 特征融合层
        # 输入是4个层级的特征拼接，所以输入通道是 in_channels * 4
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. 局部上下文增强模块 (Local Context Enhancement)
        # 使用不同膨胀率的卷积来捕获多尺度上下文
        self.context_enhancer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3. 像素级注意力 (Pixel-aware Attention)
        # 类似于 RDIAN 或 DNAnet 中的注意力机制，强调显著点
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 4. 热力图预测头
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        """
        Args:
            features: List of tensors [P1, P2, P3, P4] from Adapter.
                      P1 is the largest scale (H/4, W/4).
        """
        # 1. 统一尺度到 P1 (H/4, W/4)
        target_size = features[0].shape[2:]
        upsampled_feats = []
        
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled_feats.append(feat)
            
        # 2. 密集拼接 (Dense Concatenation)
        cat_feats = torch.cat(upsampled_feats, dim=1)
        
        # 3. 特征融合
        fused_feat = self.fusion_conv(cat_feats)
        
        # 4. 上下文增强
        enhanced_feat = self.context_enhancer(fused_feat)
        
        # 5. 注意力加权 (保留小目标高亮特征)
        att_map = self.attention(enhanced_feat)
        weighted_feat = enhanced_feat * att_map + enhanced_feat
        
        # 6. 生成热力图
        heatmap = self.heatmap_head(weighted_feat)
        
        # 恢复到原图尺寸 (H, W) - 假设输入是 1/4 尺度
        H, W = heatmap.shape[2] * 4, heatmap.shape[3] * 4
        heatmap = F.interpolate(heatmap, size=(H, W), mode='bilinear', align_corners=False)
        
        return heatmap, weighted_feat