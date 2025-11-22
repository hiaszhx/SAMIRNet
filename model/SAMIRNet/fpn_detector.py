import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNDetector(nn.Module):
    def __init__(self, in_channels=[96, 192, 384, 768], fpn_dim=256, num_classes=1):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_ch, fpn_dim, kernel_size=1) for in_ch in in_channels])
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ) for _ in in_channels
        ])
        self.attention = ChannelSpatialAttention(fpn_dim)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim // 2, fpn_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim // 4, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        laterals = [conv(features[i]) for i, conv in enumerate(self.lateral_convs)]
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i-1].shape[2:], mode='bilinear', align_corners=False)
            laterals[i-1] = laterals[i-1] + upsampled
        fpn_features = [conv(laterals[i]) for i, conv in enumerate(self.output_convs)]
        finest_feature = fpn_features[0]
        enhanced_feature = self.attention(finest_feature)
        heatmap = self.heatmap_head(enhanced_feature)
        H, W = heatmap.shape[2] * 4, heatmap.shape[3] * 4
        heatmap = F.interpolate(heatmap, size=(H, W), mode='bilinear', align_corners=False)
        return heatmap, finest_feature

class ChannelSpatialAttention(nn.Module):
    """
    Optimized for Small Targets:
    1. Channel Attention: Uses MaxPool instead of AvgPool to preserve peak features.
    2. Spatial Attention: Retains both mean and max.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Removed AvgPool for Channel Attention to prevent washing out small targets
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel attention (Max pool only)
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(max_out)
        x = x * channel_att
        
        # Spatial attention (Mean + Max)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        return x