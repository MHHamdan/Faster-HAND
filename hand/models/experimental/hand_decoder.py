"""
HAND Hierarchical Attention Decoder
Based on HAND paper Section III-B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, LayerNorm, Dropout, Embedding, Conv1d
from hand.models.experimental.memory_attention import (
    MemoryAugmentedAttention,
    SparseAttention,
    AdaptiveFeatureFusion
)
from hand.models.baseline.attention import PositionalEncoding1DOnTheFly


class HANDDecoderLayer(Module):
    """
    Single HAND Decoder Layer with hierarchical attention mechanisms
    Implements equations 9-16 from the paper
    """
    def __init__(self, params):
        super(HANDDecoderLayer, self).__init__()

        self.embed_dim = params["enc_dim"]
        self.num_heads = params["dec_num_heads"]
        self.dim_feedforward = params.get("dec_dim_feedforward", 256)

        # Equation 9: Masked Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(
            self.embed_dim,
            self.num_heads,
            dropout=params.get("dec_att_dropout", 0.1),
            batch_first=False
        )
        self.norm1 = LayerNorm(self.embed_dim)
        self.dropout1 = Dropout(params.get("dec_res_dropout", 0.1))

        # Equation 10: Multi-Head Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            self.embed_dim,
            self.num_heads,
            dropout=params.get("dec_att_dropout", 0.1),
            batch_first=False
        )
        self.norm2 = LayerNorm(self.embed_dim)
        self.dropout2 = Dropout(params.get("dec_res_dropout", 0.1))

        # Equation 11: Memory-Augmented Attention
        memory_size = params.get("memory_size", 64)
        self.memory_attn = MemoryAugmentedAttention(
            self.embed_dim,
            self.num_heads,
            memory_size=memory_size,
            dropout=params.get("dec_att_dropout", 0.1)
        )

        # Equation 12: Sparse Attention
        window_size = params.get("attention_win", 100)
        self.sparse_attn = SparseAttention(
            self.embed_dim,
            self.num_heads,
            window_size=window_size,
            dropout=params.get("dec_att_dropout", 0.1)
        )

        # Equation 13: Combine attention mechanisms
        self.attention_combine = Linear(self.embed_dim * 2, self.embed_dim)

        # Equation 16: Position-Wise Feed-Forward Network
        self.ffn = nn.Sequential(
            Linear(self.embed_dim, self.dim_feedforward),
            nn.ReLU(),
            Dropout(params.get("dec_res_dropout", 0.1)),
            Linear(self.dim_feedforward, self.embed_dim)
        )
        self.norm3 = LayerNorm(self.embed_dim)
        self.dropout3 = Dropout(params.get("dec_res_dropout", 0.1))

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: target sequence [T, B, E]
            memory: encoder memory [S, B, E]
            tgt_mask: target mask
            memory_mask: memory mask
            tgt_key_padding_mask: target padding mask
            memory_key_padding_mask: memory padding mask
        """
        # Equation 9: Masked Self-Attention with residual
        tgt2, self_attn_weights = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Equation 10: Cross-Attention with residual
        tgt2, cross_attn_weights = self.cross_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Convert to batch-first for custom attention modules
        tgt_bf = tgt.transpose(0, 1)  # [B, T, E]
        memory_bf = memory.transpose(0, 1)  # [B, S, E]

        # Equation 11: Memory-Augmented Attention
        mem_out, _ = self.memory_attn(tgt_bf, memory_bf, memory_bf)

        # Equation 12: Sparse Attention
        sparse_out, _ = self.sparse_attn(tgt_bf, memory_bf, memory_bf)

        # Equation 13: Combine attention outputs
        combined = torch.cat([mem_out, sparse_out], dim=-1)
        combined = self.attention_combine(combined)
        combined = combined.transpose(0, 1)  # Back to [T, B, E]

        # Add combined attention with residual
        tgt = tgt + self.dropout2(combined)

        # Equation 16: Feed-Forward Network with residual
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, cross_attn_weights, self_attn_weights


class HAND_Decoder(Module):
    """
    Complete HAND Hierarchical Attention Decoder
    Stack of HANDDecoderLayer modules with adaptive feature fusion
    """
    def __init__(self, params):
        super(HAND_Decoder, self).__init__()

        self.params = params
        self.enc_dim = params["enc_dim"]
        self.num_layers = params.get("dec_num_layers", 6)
        vocab_size = params["vocab_size"]

        # Token embedding
        self.embedding = Embedding(vocab_size + 3, self.enc_dim)

        # Positional encoding
        use_line_indices = params.get("use_line_indices", True)
        if use_line_indices:
            if params.get("two_step_pos_enc_mode", "cat") == "add":
                self.pe_1d = PositionalEncoding1DOnTheFly(self.enc_dim, params["device"])
            else:
                self.pe_1d = PositionalEncoding1DOnTheFly(self.enc_dim // 2, params["device"])
        else:
            from hand.models.baseline.attention import PositionalEncoding1D
            self.pe_1d = PositionalEncoding1D(self.enc_dim, params["l_max"], params["device"])

        self.use_line_indices = use_line_indices

        # Stack of decoder layers
        self.layers = ModuleList([
            HANDDecoderLayer(params) for _ in range(self.num_layers)
        ])

        # Equation 14: Adaptive Feature Fusion
        self.adaptive_fusion = AdaptiveFeatureFusion(self.enc_dim, num_levels=self.num_layers // 2)

        # Output projection (Equation 17)
        # Need vocab_size + 3: charset + EOT + SOT + padding
        # But padding is -1 and handled by ignore_index, so we need vocab_size + 2
        self.dropout = Dropout(params.get("dec_pred_dropout", 0.1))
        self.output_proj = Conv1d(self.enc_dim, vocab_size + 2, kernel_size=1)

    def forward(self, tokens, features_2d, token_len=None, features_size=None,
                line_indices=None, index_in_lines=None, tgt_mask=None,
                memory_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            tokens: input token indices [B, T]
            features_2d: encoded features from HAND encoder
            line_indices: line position indices for hierarchical encoding
            index_in_lines: position within each line
            Other args: various masks and metadata
        """
        device = tokens.device
        B, T = tokens.shape

        # Replace padding tokens (-1) with 0 before embedding lookup
        # This prevents index out of range errors
        # Clamp to valid range: [0, vocab_size + 2]
        vocab_size = self.params["vocab_size"]
        max_valid_token = vocab_size + 2  # embedding has vocab_size + 3 entries (0 to vocab_size + 2)
        tokens_safe = torch.clamp(tokens, min=0, max=max_valid_token)

        # Debug: check for out of range tokens
        if (tokens > max_valid_token).any() or (tokens < -1).any():
            invalid_count = ((tokens > max_valid_token) | (tokens < -1)).sum().item()
            max_token = tokens.max().item()
            min_token = tokens.min().item()
            print(f"⚠️  Warning: {invalid_count} invalid tokens found (range: [{min_token}, {max_token}], valid: [-1, {max_valid_token}])")

        # Token Embedding (Equation 8)
        emb_tokens = self.embedding(tokens_safe).permute(1, 0, 2)  # [T, B, E]

        # Add Positional Encoding
        if self.use_line_indices and line_indices is not None:
            if self.params.get("two_step_pos_enc_mode", "cat") == "cat":
                # Concatenate line and position encodings
                line_pe = self.pe_1d(line_indices)  # [B, E/2, T]
                pos_pe = self.pe_1d(index_in_lines)  # [B, E/2, T]
                pe = torch.cat([line_pe, pos_pe], dim=1).permute(2, 0, 1)  # [T, B, E]
            else:
                # Add line and position encodings
                line_pe = self.pe_1d(line_indices).permute(2, 0, 1)  # [T, B, E/2]
                pos_pe = self.pe_1d(index_in_lines).permute(2, 0, 1)  # [T, B, E/2]
                pe = torch.cat([line_pe, pos_pe], dim=-1)  # [T, B, E]
            tgt = emb_tokens + pe
        else:
            # Standard 1D positional encoding
            indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            pe = self.pe_1d(indices)  # [B, E, T]
            pe = pe.permute(2, 0, 1)  # [T, B, E]
            tgt = emb_tokens + pe

        # Flatten 2D features to 1D for cross-attention
        B, C, H, W = features_2d.shape
        memory = features_2d.view(B, C, H * W).permute(2, 0, 1)  # [H*W, B, C]

        # Pass through decoder layers and collect multi-level features
        multi_level_features = []
        all_cross_attn_weights = []
        all_self_attn_weights = []

        for i, layer in enumerate(self.layers):
            tgt, cross_weights, self_weights = layer(
                tgt, memory,
                tgt_mask=tgt_mask,
                memory_mask=None,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

            # Collect features from specific layers for adaptive fusion
            if i % (self.num_layers // 3) == 0:
                multi_level_features.append(tgt)

            all_cross_attn_weights.append(cross_weights)
            all_self_attn_weights.append(self_weights)

        # Equation 14: Adaptive Feature Fusion
        if len(multi_level_features) > 1:
            fused_features = self.adaptive_fusion(multi_level_features)
            # Blend fused features with final output
            tgt = 0.7 * tgt + 0.3 * fused_features

        # Equation 17: Output projection
        output = tgt.permute(1, 2, 0)  # [B, E, T]
        output = F.relu(output)
        output = self.dropout(output)
        logits = self.output_proj(output)  # [B, vocab_size, T]

        return logits, all_cross_attn_weights[-1], tgt

    def generate_square_subsequent_mask(self, sz, device):
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
