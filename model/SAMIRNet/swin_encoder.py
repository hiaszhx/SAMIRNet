import torch
import torch.nn as nn
import timm
import os


class SwinTransformerEncoder(nn.Module):
    """
    Swin Transformer as Image Encoder
    Uses timm.create_model for robust feature extraction
    Returns multi-scale features for FPN
    """
    
    def __init__(
        self,
        img_size=256,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        pretrained_path=None,
        **kwargs
    ):
        super().__init__()
        
        # 确定模型名称
        if embed_dim == 96 and depths == [2, 2, 6, 2]:
            model_name = 'swin_tiny_patch4_window7_224'
        elif embed_dim == 96 and depths == [2, 2, 18, 2]:
            model_name = 'swin_small_patch4_window7_224'
        elif embed_dim == 128 and depths == [2, 2, 18, 2]:
            model_name = 'swin_base_patch4_window7_224'
        else:
            print(f"Warning: Custom Swin config ({embed_dim}, {depths}).")
            print("Falling back to swin_tiny_patch4_window7_224 for pretraining.")
            model_name = 'swin_tiny_patch4_window7_224'
        
        # 检查 pretrained_path 是否存在
        use_pretrained = False
        if pretrained_path is not None and os.path.exists(pretrained_path):
            use_pretrained = True
            print(f"Loading pretrained weights from {pretrained_path}...")
        else:
            print(f"⚠️ Pretrained weights not found at {pretrained_path}. Training from scratch...")

        self.swin = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            pretrained_cfg_overlay=dict(file=pretrained_path) if use_pretrained else None,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=3,
            img_size=img_size,
        )
        
        print(f"Initialized Swin encoder ({model_name}) with feature dims:")
        for i, dim in enumerate(self.swin.feature_info.channels()):
            print(f"  Stage {i+1}: {dim}")

    def forward(self, x):
        """
        Forward pass to extract multi-scale features
        
        Args:
            x: (B, 3, H, W)
        
        Returns:
            List of 4 features maps (B, C, H, W) at different scales
        """
        # timm's features_only=True returns (B, H, W, C)
        features_channels_last = self.swin(x)
        
        # Convert to (B, C, H, W) for compatibility with FPN
        features = [
            f.permute(0, 3, 1, 2).contiguous() 
            for f in features_channels_last
        ]
        
        return features


# Test function
if __name__ == '__main__':
    # Test encoder
    encoder = SwinTransformerEncoder(
        img_size=256,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        pretrained_path='./pretrained/swin_tiny_patch4_window7_224.pth'
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    features = encoder(x)
    
    print("\nOutput feature shapes:")
    for i, feat in enumerate(features):
        print(f"  Stage {i+1}: {feat.shape}")