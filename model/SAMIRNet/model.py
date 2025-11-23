import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_encoder import SwinTransformerEncoder
from .small_target_detector import SmallTargetDetector
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .fusion import UpBlock_attention # 确保你已经创建了 fusion.py

class CrossScaleAdapter(nn.Module):
    """
    升级版适配器：使用 Cross Attention 融合多尺度特征
    学习 SAM-SPL 的 Decoder 结构
    """
    def __init__(self, in_channels_list, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. 统一通道数
        self.projections = nn.ModuleList([
            nn.Conv2d(in_ch, embed_dim, kernel_size=1) for in_ch in in_channels_list
        ])
        
        # 2. 交叉注意力上采样块 (从深层到浅层逐级融合)
        # P4 -> P3 -> P2 -> P1
        self.up_blocks = nn.ModuleList([
            UpBlock_attention(embed_dim, embed_dim, nb_Conv=1, MC=True), # P4+P3
            UpBlock_attention(embed_dim, embed_dim, nb_Conv=1, MC=True), # +P2
            UpBlock_attention(embed_dim, embed_dim, nb_Conv=1, MC=True)  # +P1
        ])
        
        # 最终输出调整
        self.final_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        # features: [P1, P2, P3, P4] (scales: 1/4, 1/8, 1/16, 1/32)
        # 投影所有特征到统一维度
        projs = [p(f) for p, f in zip(self.projections, features)]
        p1, p2, p3, p4 = projs
        
        # 逐级融合 (Deep-to-Shallow)
        # p4 (1/32) -> p3 (1/16)
        x = self.up_blocks[0](p4, p3)
        # -> p2 (1/8)
        x = self.up_blocks[1](x, p2)
        # -> p1 (1/4)
        x = self.up_blocks[2](x, p1)
        
        out = self.final_conv(x)
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
        fpn_dim=256,
        decoder_dim=256,
        num_mask_tokens=3,
        use_scheduled_sampling=True
    ):
        super(SAMIRNet, self).__init__()
        
        self.img_size = img_size
        self.use_scheduled_sampling = use_scheduled_sampling
        self.training_step = 0
        
        # 1. Image Encoder
        self.encoder = SwinTransformerEncoder(
            img_size=img_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            pretrained_path=pretrained_path
        )
        
        # --- 策略调整：解冻高层参数 ---
        # 冻结前两个 Stage (提取基础纹理)，解冻后两个 Stage (适应红外语义)
        for name, param in self.encoder.named_parameters():
            if "layers.0" in name or "layers.1" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True # 允许训练深层
        print("INFO: Image Encoder partially unfrozen (Stages 3 & 4 trainable).")
        
        encoder_dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        
        # 2. Adapters
        # 探测分支适配 (保持简单，保留高频)
        self.det_adapter = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, fpn_dim, 1),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU()
            ) for dim in encoder_dims
        ])
        
        # 分割分支适配 (升级为 Cross Attention 融合)
        self.seg_adapter = CrossScaleAdapter(encoder_dims, embed_dim=decoder_dim)
        
        # 3. Detection Head (独立分支)
        self.detector = SmallTargetDetector(in_channels=fpn_dim)
        
        # 4. Prompt Encoder & Mask Decoder
        self.prompt_encoder = PromptEncoder(
            embed_dim=decoder_dim,
            image_embedding_size=(img_size // 4, img_size // 4)
        )
        self.mask_decoder = MaskDecoder(
            transformer_dim=decoder_dim,
            num_multimask_outputs=num_mask_tokens
        )
        
    def forward(self, images, gt_centers=None, gt_masks=None, mode='train'):
        B = images.size(0)
        H, W = images.shape[-2:]
        if images.size(1) == 1: images = images.repeat(1, 3, 1, 1)
        
        # 1. Encoder (部分解冻)
        multi_scale_features = self.encoder(images)
            
        # 2. 探测分支 (独立)
        det_feats = [adapt(f) for adapt, f in zip(self.det_adapter, multi_scale_features)]
        pred_heatmap, _ = self.detector(det_feats)
        
        # 3. 生成 Prompt
        if mode == 'train':
            use_gt = self._should_use_gt_prompt()
            if use_gt and gt_centers is not None:
                prompt_points = self._norm_points(gt_centers, H, W)
            else:
                prompt_points = self._extract_centers_from_heatmap(pred_heatmap.detach(), top_k=3)
        else:
            prompt_points = self._extract_centers_from_heatmap(pred_heatmap, top_k=5, threshold=0.3)
        
        # 4. 分割分支 (Cross Attention 融合)
        image_embeddings = self.seg_adapter(multi_scale_features) # Output: (B, 256, H/4, W/4)
        
        # 5. SAM Decoding
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=prompt_points, boxes=None, masks=None)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        
        pred_masks = F.interpolate(low_res_masks, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        pred_masks = torch.sigmoid(pred_masks)
        
        if mode == 'train': self.training_step += 1
        return pred_heatmap, pred_masks, iou_predictions

    def _norm_points(self, points, H, W):
        norm_points = points.clone()
        norm_points[:, :, 0] /= W
        norm_points[:, :, 1] /= H
        return norm_points

    def _should_use_gt_prompt(self):
        if not self.use_scheduled_sampling: return True
        # 随着训练进行，逐渐减少对 GT Prompt 的依赖
        prob = max(0.3, 1.0 - (self.training_step / 15000)) 
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

# Helpers
def SAMIRNet_Tiny(pretrained_path='./pretrained/swin_tiny_patch4_window7_224.pth'):
    return SAMIRNet(img_size=256, pretrained_path=pretrained_path, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])

def SAMIRNet_Small(pretrained_path='./pretrained/swin_small_patch4_window7_224.pth'):
    return SAMIRNet(img_size=256, pretrained_path=pretrained_path, embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24])