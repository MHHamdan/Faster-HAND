"""
Decoder attention modules of the proposed HAND architecture (manuscript Eq. 11, 12, 14):
MemoryAugmentedAttention, SparseAttention, AdaptiveFeatureFusion.

INACTIVE. Their only consumer is `hand.models.experimental.hand_decoder`, which no
training entry point imports. No released checkpoint contains a `memory`, `sparse`
or `fusion` key. No released checkpoint carries one; see docs/ablations.md section 1.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Linear, LayerNorm, Dropout
from torch.nn.init import xavier_uniform_
import math


class MemoryAugmentedAttention(Module):
    """
    Memory-Augmented Attention (Equation 11 from HAND paper)
    A_M(Q, K, V, M) = σ(Q[K; M]^T / √d_k)[V; M]
    """
    def __init__(self, embed_dim, num_heads, memory_size=64, dropout=0.1):
        super(MemoryAugmentedAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Learnable memory matrix
        self.memory = nn.Parameter(torch.randn(memory_size, embed_dim))
        nn.init.xavier_uniform_(self.memory)

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

        self.dropout = Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        B, T, C = query.shape

        # Project Q, K, V
        Q = self.q_proj(query).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Expand memory for batch
        M = self.memory.unsqueeze(0).expand(B, -1, -1)
        M = M.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Concatenate memory with K and V
        K_aug = torch.cat([K, M], dim=2)  # [B, num_heads, seq_len + mem_size, head_dim]
        V_aug = torch.cat([V, M], dim=2)

        # Attention scores
        attn = (Q @ K_aug.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            # Expand mask for memory
            mem_mask = torch.zeros(B, T, M.size(2), device=attn.device, dtype=attn_mask.dtype)
            attn_mask_aug = torch.cat([attn_mask, mem_mask], dim=-1)
            attn = attn + attn_mask_aug.unsqueeze(1)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ V_aug).transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)

        return out, attn


class SparseAttention(Module):
    """
    Sparse Attention (Equation 12 from HAND paper)
    A_S(Q, K, V) = σ(QK^T / √d_k ⊙ W)V
    where W is a sparse attention mask
    """
    def __init__(self, embed_dim, num_heads, window_size=100, dropout=0.1):
        super(SparseAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

        self.dropout = Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        B, T, C = query.shape
        S = key.shape[1]

        # Project
        Q = self.q_proj(query).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Create combined attention mask for memory-efficient attention
        combined_mask = None

        # Create sparse mask (local window)
        if self.window_size is not None and self.window_size < S:
            sparse_mask = torch.zeros(T, S, device=query.device, dtype=Q.dtype)
            for i in range(T):
                # For cross-attention, center the window around position i scaled to S
                center_pos = int(i * S / T) if T != S else i
                start = max(0, center_pos - self.window_size // 2)
                end = min(S, center_pos + self.window_size // 2)
                # Set positions outside window to -inf
                if start > 0:
                    sparse_mask[i, :start] = float('-inf')
                if end < S:
                    sparse_mask[i, end:] = float('-inf')
            combined_mask = sparse_mask

        # Add additional attention mask if provided
        if attn_mask is not None:
            if combined_mask is None:
                combined_mask = attn_mask
            else:
                combined_mask = combined_mask + attn_mask

        # Use memory-efficient attention (PyTorch 2.0+)
        # This avoids materializing the full attention matrix
        try:
            out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=combined_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=self.scale
            )
            # For compatibility, compute attention weights only when needed (e.g., visualization)
            # In training, we skip this to save memory
            attn = None
        except:
            # Fallback to manual computation if scaled_dot_product_attention not available
            attn = (Q @ K.transpose(-2, -1)) * self.scale
            if combined_mask is not None:
                attn = attn + combined_mask.unsqueeze(0).unsqueeze(0)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = attn @ V

        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)

        return out, attn


class AdaptiveFeatureFusion(Module):
    """
    Adaptive Feature Fusion (Equation 14 from HAND paper)
    C = Σ_l λ_l · MultiHeadAttn_l(Q, K_l, V_l)
    where λ_l are learnable weights
    """
    def __init__(self, embed_dim, num_levels=3):
        super(AdaptiveFeatureFusion, self).__init__()

        self.num_levels = num_levels
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(num_levels))

    def forward(self, features_list):
        """
        features_list: list of feature tensors from different levels
        """
        # Normalize weights
        weights = F.softmax(self.fusion_weights, dim=0)

        # Weighted sum
        fused = sum(w * f for w, f in zip(weights, features_list))

        return fused
