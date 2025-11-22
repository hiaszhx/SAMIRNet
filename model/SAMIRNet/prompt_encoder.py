import torch
import torch.nn as nn
import numpy as np


class PromptEncoder(nn.Module):
    """
    Encode different types of prompts (points, boxes, masks)
    Similar to SAM's prompt encoder
    """
    
    def __init__(
        self,
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        
        # Positional encoding for points
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        
        # Point embeddings (for different point types)
        self.num_point_embeddings = 4  # pos/neg point, box corners
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)
        ])
        
        # Not-a-point embedding
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        
        # Mask downscaling
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            nn.LayerNorm([mask_in_chans // 4, 
                         image_embedding_size[0], 
                         image_embedding_size[1]]),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            nn.LayerNorm([mask_in_chans, 
                         image_embedding_size[0] // 2, 
                         image_embedding_size[1] // 2]),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        
        # No mask embedding
        self.no_mask_embed = nn.Embedding(1, embed_dim)
    
    def get_dense_pe(self):
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)
    
    def forward(self, points=None, boxes=None, masks=None):
        """
        Args:
            points: (B, N, 2) - point coordinates in [0, 1] range
            boxes: (B, N, 4) - box coordinates in [0, 1] range
            masks: (B, 1, H, W) - mask prompts
        
        Returns:
            sparse_embeddings: (B, N, embed_dim)
            dense_embeddings: (B, embed_dim, H, W)
        """
        bs = 1
        sparse_embeddings_list = []
        
        # Encode points
        if points is not None:
            bs = points.shape[0]
            coords = points.clone()
            # Scale to image embedding size
            coords[:, :, 0] *= self.image_embedding_size[1]
            coords[:, :, 1] *= self.image_embedding_size[0]
            
            # Add positional encoding
            point_embedding = self.pe_layer.forward_with_coords(
                coords, self.image_embedding_size
            )
            
            # Add point type embedding (all positive points for center detection)
            point_embedding = point_embedding + self.point_embeddings[0].weight.reshape(1, 1, -1)
            
            sparse_embeddings_list.append(point_embedding)
        
        # Encode boxes (if needed)
        if boxes is not None:
            bs = boxes.shape[0]
            # Convert boxes to corner points
            box_coords = self._boxes_to_corners(boxes)
            # Scale and encode
            box_coords[:, :, 0] *= self.image_embedding_size[1]
            box_coords[:, :, 1] *= self.image_embedding_size[0]
            
            corner_embedding = self.pe_layer.forward_with_coords(
                box_coords, self.image_embedding_size
            )
            corner_embedding[:, 0::2] += self.point_embeddings[2].weight.reshape(1, 1, -1)
            corner_embedding[:, 1::2] += self.point_embeddings[3].weight.reshape(1, 1, -1)
            
            sparse_embeddings_list.append(corner_embedding)
        
        # Combine sparse embeddings
        if len(sparse_embeddings_list) > 0:
            sparse_embeddings = torch.cat(sparse_embeddings_list, dim=1)
        else:
            # No prompts, use not-a-point embedding
            sparse_embeddings = self.not_a_point_embed.weight.reshape(1, 1, -1).expand(
                bs, 1, -1
            )
        
        # Encode masks
        if masks is not None:
            dense_embeddings = self.mask_downscaling(masks)
        else:
            # No mask prompt
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        
        return sparse_embeddings, dense_embeddings
    
    def _boxes_to_corners(self, boxes):
        """Convert boxes (x1, y1, x2, y2) to corners"""
        corners = torch.stack([
            boxes[:, :, [0, 1]],  # top-left
            boxes[:, :, [2, 1]],  # top-right
            boxes[:, :, [0, 3]],  # bottom-left
            boxes[:, :, [2, 3]],  # bottom-right
        ], dim=2).flatten(1, 2)
        return corners


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """
    
    def __init__(self, num_pos_feats=64, scale=None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )
    
    def _pe_encoding(self, coords):
        """Positionally encode points that are normalized to [0,1]."""
        # Assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # Outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    
    def forward(self, size):
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W
    
    def forward_with_coords(self, coords_input, image_size):
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))
