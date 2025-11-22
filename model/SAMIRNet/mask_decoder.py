import torch
import torch.nn as nn

# ... [Keep imports and TwoWayTransformer/Attention/MLP classes as they were] ...
# Only modifying MaskDecoder class

class MaskDecoder(nn.Module):
    def __init__(
        self,
        transformer_dim=256,
        transformer_depth=2,
        transformer_heads=8,
        transformer_mlp_dim=2048,
        num_multimask_outputs=3,
        activation=nn.GELU,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs
        
        self.transformer = TwoWayTransformer(
            depth=transformer_depth,
            embedding_dim=transformer_dim,
            num_heads=transformer_heads,
            mlp_dim=transformer_mlp_dim,
        )
        
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        
        # OPTIMIZATION: Use Upsample + Conv instead of ConvTranspose2d
        # to avoid checkerboard artifacts in small target masks
        self.output_upscaling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=3, padding=1),
            nn.LayerNorm([transformer_dim // 4, 128, 128]), 
            activation(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=3, padding=1),
            activation(),
        )
        
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)
        ])
        
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )
    
    # ... [Keep forward and predict_masks methods same] ...
    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output=True):
        masks, iou_pred = self.predict_masks(image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings)
        if multimask_output: mask_slice = slice(1, None)
        else: mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        return masks, iou_pred

    def predict_masks(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings):
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)
        
        src = image_embeddings + dense_prompt_embeddings
        b, c, h, w = src.shape
        hs, src = self.transformer(src, image_pe, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]
        
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred

# ... [Include TwoWayTransformer, Attention, MLP classes here] ...
# (For brevity, assume the rest of mask_decoder.py is unchanged)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.activation = activation()
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TwoWayTransformer(nn.Module):
    def __init__(self, depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048, activation=nn.ReLU, attention_downsample_rate=2):
        super().__init__()
        self.layers = nn.ModuleList([TwoWayAttentionBlock(embedding_dim, num_heads, mlp_dim, activation, attention_downsample_rate, skip_first_layer_pe=(i == 0)) for i in range(depth)])
        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)
    def forward(self, image_embedding, image_pe, point_embedding):
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries = point_embedding
        keys = image_embedding
        for layer in self.layers: queries, keys = layer(queries, keys, point_embedding, image_pe)
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys

class TwoWayAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dim=2048, activation=nn.ReLU, attention_downsample_rate=2, skip_first_layer_pe=False):
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, 2, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe
    def forward(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe: queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q=q, k=q, v=queries)
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(q=q, k=k, v=keys))
        queries = self.norm3(queries + self.mlp(queries))
        q = queries + query_pe
        k = keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(q=k, k=q, v=queries))
        return queries, keys

class Attention(nn.Module):
    def __init__(self, embedding_dim, num_heads, downsample_rate=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
    def forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        b, n, c = q.shape
        q = q.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(b, k.size(1), self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(b, v.size(1), self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) / (c // self.num_heads) ** 0.5
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(b, n, c)
        return self.out_proj(out)