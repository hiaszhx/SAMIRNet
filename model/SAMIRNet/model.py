import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_encoder import SwinTransformerEncoder
# 修改导入：不再使用 FPN，改用新的 SmallTargetDetector
from .small_target_detector import SmallTargetDetector
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder

class MultiScaleAdapter(nn.Module):
    """
    探测头适配器：
    将 Swin 的不同通道数的特征统一映射到固定维度，保持多尺度列表结构
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=1),
                nn.GroupNorm(16, out_channels), # 使用 GroupNorm 适应小 batch
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            ) for in_ch in in_channels_list
        ])
        
    def forward(self, features):
        return [adapter(f) for adapter, f in zip(self.adapters, features)]

class SegFeatureAdapter(nn.Module):
    """
    分割头适配器：
    将多尺度特征融合为单一的高分辨率特征图 (Image Embedding)，供 Mask Decoder 使用
    """
    def __init__(self, in_channels_list, embed_dim):
        super().__init__()
        # 先投影到 embed_dim
        self.projections = nn.ModuleList([
            nn.Conv2d(in_ch, embed_dim, kernel_size=1) for in_ch in in_channels_list
        ])
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels_list), embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        )
        
    def forward(self, features):
        target_size = features[0].shape[2:]
        projected_feats = []
        
        for i, feat in enumerate(features):
            x = self.projections[i](feat)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            projected_feats.append(x)
            
        cat_feats = torch.cat(projected_feats, dim=1)
        out = self.fusion(cat_feats)
        return out

class SAMIRNet(nn.Module):
    def __init__(
        self,
        img_size=256,
        pretrained_path=None,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        fpn_dim=256,      # 探测头内部特征维度
        decoder_dim=256,  # Mask Decoder 维度
        num_mask_tokens=3,
        use_scheduled_sampling=True
    ):
        super(SAMIRNet, self).__init__()
        
        self.img_size = img_size
        self.use_scheduled_sampling = use_scheduled_sampling
        self.training_step = 0
        
        # 1. Image Encoder (Swin Transformer)
        self.encoder = SwinTransformerEncoder(
            img_size=img_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            pretrained_path=pretrained_path
        )
        
        # --- 冻结 Image Encoder ---
        # 我们希望只训练探测头、适配器和分割解码器
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("INFO: Image Encoder parameters frozen (Swin Transformer).")
        
        encoder_dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        
        # 2. Adapters (新增)
        # 探测头适配层：输出多尺度列表
        self.det_adapter = MultiScaleAdapter(encoder_dims, out_channels=fpn_dim)
        # 分割头适配层：输出单一特征图
        self.seg_adapter = SegFeatureAdapter(encoder_dims, embed_dim=decoder_dim)
        
        # 3. Detection Head (更换为 SmallTargetDetector)
        self.detector = SmallTargetDetector(
            in_channels=fpn_dim,
            num_classes=1
        )
        
        # 4. Prompt Encoder
        self.prompt_encoder = PromptEncoder(
            embed_dim=decoder_dim,
            image_embedding_size=(img_size // 4, img_size // 4)
        )
        
        # 5. Mask Decoder
        self.mask_decoder = MaskDecoder(
            transformer_dim=decoder_dim,
            num_multimask_outputs=num_mask_tokens
        )
        
    def forward(self, images, gt_centers=None, gt_masks=None, mode='train'):
        B = images.size(0)
        H, W = images.shape[-2:]
        
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # 1. 获取 Encoder 特征 (冻结状态)
        # 使用 no_grad 确保不计算梯度，节省显存
        with torch.no_grad():
            multi_scale_features = self.encoder(images)
            
        # 2. 分支一：探测分支 (Detection Branch)
        # 适配 -> 探测 -> 热力图
        det_features = self.det_adapter(multi_scale_features)
        pred_heatmap, _ = self.detector(det_features)
        
        # 3. 生成 Prompts
        if mode == 'train':
            use_gt_prompt = self._should_use_gt_prompt()
            if use_gt_prompt and gt_centers is not None:
                prompt_points = gt_centers.clone()
                prompt_points[:, :, 0] = prompt_points[:, :, 0] / W
                prompt_points[:, :, 1] = prompt_points[:, :, 1] / H
            else:
                prompt_points = self._extract_centers_from_heatmap(
                    pred_heatmap.detach(), top_k=3
                )
        else:
            prompt_points = self._extract_centers_from_heatmap(
                pred_heatmap, top_k=5, threshold=0.3
            )
        
        # 4. 分支二：分割分支 (Segmentation Branch)
        # 适配(融合) -> Image Embedding
        image_embeddings = self.seg_adapter(multi_scale_features)
        
        # 5. Prompt Encoding
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=prompt_points, boxes=None, masks=None
        )
        
        # 6. Mask Decoding
        # 注意：这里直接传入 image_embeddings，它已经是 (B, 256, H/4, W/4)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        
        # 7. 后处理
        pred_masks = F.interpolate(
            low_res_masks,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )
        pred_masks = torch.sigmoid(pred_masks)
        
        if mode == 'train':
            self.training_step += 1
        
        return pred_heatmap, pred_masks, iou_predictions
    
    def _should_use_gt_prompt(self):
        if not self.use_scheduled_sampling: return True
        max_steps = 10000
        min_prob = 0.5
        if self.training_step < max_steps:
            prob = 1.0 - (1.0 - min_prob) * (self.training_step / max_steps)
        else:
            prob = min_prob
        return torch.rand(1).item() < prob
    
    def _extract_centers_from_heatmap(self, heatmap, top_k=3, threshold=0.3):
        """
        Extracts and normalizes center points to [0, 1]
        """
        B, _, H, W = heatmap.shape
        device = heatmap.device
        centers_list = []
        
        for b in range(B):
            hm = heatmap[b, 0]
            # 使用 MaxPool 进行非极大值抑制 (NMS) 的近似
            max_pooled = F.max_pool2d(
                hm.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1
            ).squeeze()
            
            peaks = (hm == max_pooled) & (hm > threshold)
            y_coords, x_coords = torch.where(peaks)
            scores = hm[y_coords, x_coords]
            
            if len(scores) == 0:
                # 如果没有检测到，使用中心点作为默认 Prompt
                centers = torch.tensor([[0.5, 0.5]], device=device).repeat(top_k, 1)
            else:
                top_indices = torch.argsort(scores, descending=True)[:top_k]
                x_top = x_coords[top_indices].float()
                y_top = y_coords[top_indices].float()
                
                if len(x_top) < top_k:
                    pad_size = top_k - len(x_top)
                    x_top = torch.cat([x_top, x_top[-1:].repeat(pad_size)])
                    y_top = torch.cat([y_top, y_top[-1:].repeat(pad_size)])
                
                # Stack (x, y)
                centers = torch.stack([x_top, y_top], dim=1)
                
                # Normalize to [0, 1]
                centers[:, 0] = centers[:, 0] / W
                centers[:, 1] = centers[:, 1] / H
            
            centers_list.append(centers)
        
        return torch.stack(centers_list, dim=0)

def SAMIRNet_Tiny(pretrained_path='./pretrained/swin_tiny_patch4_window7_224.pth'):
    return SAMIRNet(img_size=256, pretrained_path=pretrained_path, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])

def SAMIRNet_Small(pretrained_path='./pretrained/swin_small_patch4_window7_224.pth'):
    return SAMIRNet(img_size=256, pretrained_path=pretrained_path, embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24])