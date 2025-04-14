import torch
import torch.nn as nn

import numpy as np

from math import sqrt


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention # 是否返回attention, 用于自注意力可视化
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask:TriangularCausalMask = None):
        B, L, H, dk = queries.shape
        _, S, _, _ = values.shape
        scale = self.scale or 1./sqrt(dk) # 缩放因子, scale默认为dk的平方根, 防止自注意力分数过大导致学习梯度过大
        
        scores = torch.einsum("blhd,bshd->bhls", queries, keys) # bhls: batch_size * n_heads * L * S
        
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device).mask

            scores.masked_fill_(attn_mask, -np.inf)
        
        softmax_scores = self.dropout(torch.softmax(scores * scale, dim=-1)) # dim = technical feature
        output = torch.einsum("bhls,bshd->blhd", softmax_scores, values)
        
        if self.output_attention:
            return output.contiguous(), softmax_scores
        else:
            return output.contiguous(), None
    
class AttentionLayer(nn.Module):
    def __init__(self, attention: FullAttention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads) # Wq: 可学习的权重矩阵，且已包含多注意力头
        self.key_projection = nn.Linear(d_model, d_keys * n_heads) # Wk: 可学习的权重矩阵，且已包含多注意力头
        self.value_projection = nn.Linear(d_model, d_values * n_heads) # Wv: 可学习的权重矩阵，且已包含多注意力头
        self.output_projection = nn.Linear(d_values * n_heads, d_model)
        
        self.n_heads = n_heads
        
    def forward(self, queries, keys, values, attn_mask:TriangularCausalMask = None):
        B, L, _ = queries.shape
        _, S, _ = values.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        out, attn = self.inner_attention(
            queries, keys, values, attn_mask=attn_mask
        )
        out = out.view(B, L, -1)
        out = self.output_projection(out)
        
        return out, attn