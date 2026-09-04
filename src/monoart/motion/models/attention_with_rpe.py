"""
Multi-head Attention with Relative Position Encoding (RPE).

This module extends the standard multi-head attention to support
contextual relative position encoding for 3D point cloud transformers.

Key features:
- Compatible with existing additive position encoding
- Optional RPE that adds position bias to attention logits
- Numerical stability with attention logit clamping
- Support for lookup table, MLP-based, and separable RPE

RPE Types:
- "table": Lookup table based (original MAFT approach)
  - Sparse gradients, needs higher learning rate
  - Fast inference, fixed resolution
- "mlp": MLP-based (continuous position encoding)
  - Dense gradients, easier to train
  - Smooth continuous output, better extrapolation
  - HIGH MEMORY: O(Q×N×D) - not recommended for large point clouds
- "mlp_chunked": MLP with key-based chunking (RECOMMENDED - exact + memory efficient)
  - MATHEMATICALLY EQUIVALENT to "mlp"
  - Memory: O(Q×chunk_n×D) - ~100x less than MLP
  - Slightly slower due to chunked computation
- "mlp_double_chunked": MLP with double chunking (for extreme memory constraints)
  - MATHEMATICALLY EQUIVALENT to "mlp"
  - Memory: O(chunk_q×chunk_n×D) - ~1000x less than MLP
  - Slowest due to many kernel launches
- "mlp_checkpoint": MLP with gradient checkpointing (RECOMMENDED - exact + low memory)
  - MATHEMATICALLY EQUIVALENT to "mlp"
  - Memory: ~10x less than mlp_chunked (no intermediate storage)
  - ~25-30% slower (recomputes during backward)
- "separable": Separable approximation (fastest, but approximate)
  - Dense gradients, easy to train
  - LOW MEMORY: O(Q×D + N×D) - ~200x less than MLP
  - Approximates cos(ω(x-y)) ≈ cos(ωx)cos(ωy) + sin(ωx)sin(ωy)
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rpe import RelativePositionBias3D
from .rpe_mlp import (
    RelativePositionMLPCheckpoint,
    RelativePositionMLPDoubleChunked,
    RelativePositionMLPEfficient,
    RelativePositionMLPKeyChunked,
)
from .rpe_mlp_efficient import SeparableRelativePositionBias, SeparableRelativePositionBiasV2


class MultiheadAttentionWithRPE(nn.Module):
    """
    Multi-head Attention with optional Relative Position Encoding.

    This module computes attention as:
        attn = softmax((QK^T / sqrt(d)) + pos_bias)

    where pos_bias is computed from the relative 3D positions between
    queries and keys using learnable position embedding tables.

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_rpe: Whether to use relative position encoding
        rpe_config: Configuration for RPE module
        attn_clamp_value: Value to clamp attention logits for numerical stability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rpe: bool = True,
        rpe_config: Optional[Dict[str, Any]] = None,
        attn_clamp_value: float = 50.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.attn_clamp_value = attn_clamp_value
        self.scale = self.head_dim**-0.5

        assert self.head_dim * num_heads == embed_dim, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # RPE module
        self.use_rpe = use_rpe
        self.rpe = None
        self.rpe_type = None  # "table", "mlp", "mlp_chunked", "mlp_double_chunked", "mlp_checkpoint", or "separable"

        if use_rpe:
            rpe_config = rpe_config or {}
            self.rpe_type = rpe_config.get(
                "type", "table"
            )  # Default to table for backward compatibility

            if self.rpe_type == "mlp_checkpoint":
                # Gradient checkpointing MLP RPE: MATHEMATICALLY EQUIVALENT to MLP
                # Best memory efficiency with exact results, ~25-30% slower
                self.rpe = RelativePositionMLPCheckpoint(
                    num_heads=num_heads,
                    hidden_dim=rpe_config.get("mlp_hidden_dim", 64),
                    num_layers=rpe_config.get("mlp_num_layers", 2),
                    use_fourier=rpe_config.get("mlp_use_fourier", True),
                    num_frequencies=rpe_config.get("mlp_num_frequencies", 8),
                    max_freq=rpe_config.get("mlp_max_freq", 10.0),
                    chunk_size=rpe_config.get("chunk_n", 1000),
                )
            elif self.rpe_type == "mlp_chunked":
                # Key-chunked MLP RPE: MATHEMATICALLY EQUIVALENT to MLP
                # Recommended for large point clouds - balances memory and speed
                self.rpe = RelativePositionMLPKeyChunked(
                    num_heads=num_heads,
                    hidden_dim=rpe_config.get("mlp_hidden_dim", 64),
                    num_layers=rpe_config.get("mlp_num_layers", 2),
                    use_fourier=rpe_config.get("mlp_use_fourier", True),
                    num_frequencies=rpe_config.get("mlp_num_frequencies", 8),
                    max_freq=rpe_config.get("mlp_max_freq", 10.0),
                    chunk_size=rpe_config.get("chunk_n", 1000),  # Keys per chunk
                )
            elif self.rpe_type == "mlp_double_chunked":
                # Double-chunked MLP RPE: MATHEMATICALLY EQUIVALENT to MLP
                # For extreme memory constraints
                self.rpe = RelativePositionMLPDoubleChunked(
                    num_heads=num_heads,
                    hidden_dim=rpe_config.get("mlp_hidden_dim", 64),
                    num_layers=rpe_config.get("mlp_num_layers", 2),
                    use_fourier=rpe_config.get("mlp_use_fourier", True),
                    num_frequencies=rpe_config.get("mlp_num_frequencies", 8),
                    max_freq=rpe_config.get("mlp_max_freq", 10.0),
                    chunk_q=rpe_config.get("chunk_q", 20),
                    chunk_n=rpe_config.get("chunk_n", 1000),
                )
            elif self.rpe_type == "separable":
                # Separable RPE: memory-efficient approximation
                # Reduces memory from O(Q×N×D) to O(Q×D + N×D)
                # RECOMMENDED for large point clouds (N > 10000)
                separable_version = rpe_config.get("separable_version", "v1")  # 'v1' or 'v2'
                if separable_version == "v2":
                    # V2: Additive bias (simpler, fewer parameters)
                    self.rpe = SeparableRelativePositionBiasV2(
                        num_heads=num_heads,
                        num_frequencies=rpe_config.get("mlp_num_frequencies", 8),
                        max_freq=rpe_config.get("mlp_max_freq", 10.0),
                        learnable_freqs=rpe_config.get("learnable_freqs", False),
                    )
                else:
                    # V1: Multiplicative bias (more expressive, default)
                    self.rpe = SeparableRelativePositionBias(
                        num_heads=num_heads,
                        embed_dim=rpe_config.get("separable_embed_dim", 64),
                        num_frequencies=rpe_config.get("mlp_num_frequencies", 8),
                        max_freq=rpe_config.get("mlp_max_freq", 10.0),
                        use_bias=True,
                    )
            elif self.rpe_type == "mlp":
                # MLP-based RPE: continuous position encoding with dense gradients
                # WARNING: High memory usage O(Q×N×D), not recommended for large point clouds
                self.rpe = RelativePositionMLPEfficient(
                    num_heads=num_heads,
                    hidden_dim=rpe_config.get("mlp_hidden_dim", 64),
                    num_layers=rpe_config.get("mlp_num_layers", 2),
                    use_fourier=rpe_config.get("mlp_use_fourier", True),
                    num_frequencies=rpe_config.get("mlp_num_frequencies", 8),
                    max_freq=rpe_config.get("mlp_max_freq", 10.0),
                    chunk_size=rpe_config.get("chunk_size", 10),
                )
            else:
                # Lookup table RPE (original approach)
                self.rpe = RelativePositionBias3D(
                    num_heads=num_heads,
                    head_dim=self.head_dim,
                    grid_size=rpe_config.get("grid_size", 0.05),
                    num_buckets=rpe_config.get("num_buckets", 24),
                    use_query_bias=rpe_config.get("use_query_bias", True),
                    use_key_bias=rpe_config.get("use_key_bias", True),
                    use_value_bias=rpe_config.get("use_value_bias", False),
                    chunk_size=rpe_config.get("chunk_size", 10),
                )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0.0)
        nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos_3d: Optional[torch.Tensor] = None,
        key_pos_3d: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            query: [B, Q, D] query features
            key: [B, N, D] key features
            value: [B, N, D] value features
            query_pos_3d: [B, Q, 3] query 3D positions (for RPE)
            key_pos_3d: [B, N, 3] key 3D positions (for RPE)
            key_padding_mask: [B, N] True = ignore this position
            attn_mask: [Q, N] or [B*H, Q, N] additional attention mask
            need_weights: Whether to return attention weights

        Returns:
            output: [B, Q, D] attention output
            attn_weights: [B, H, Q, N] attention weights if need_weights else None
        """
        B, Q, _ = query.shape
        N = key.shape[1]

        # Project Q, K, V
        q = self.q_proj(query)  # [B, Q, D]
        k = self.k_proj(key)  # [B, N, D]
        v = self.v_proj(value)  # [B, N, D]

        # Reshape for multi-head attention
        # [B, L, D] -> [B, L, H, head_dim] -> [B, H, L, head_dim]
        q = q.view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores: [B, H, Q, N]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # ========== Add position bias (RPE) ==========
        if self.use_rpe and self.rpe is not None:
            if query_pos_3d is not None and key_pos_3d is not None:
                if self.rpe_type in (
                    "mlp",
                    "mlp_chunked",
                    "mlp_double_chunked",
                    "mlp_checkpoint",
                    "separable",
                ):
                    # MLP-based, chunked, checkpoint, and Separable RPE: only needs positions
                    pos_bias = self.rpe(query_pos_3d, key_pos_3d)
                else:
                    # Table-based RPE: needs query/key features and positions
                    pos_bias = self.rpe(q, k, query_pos_3d, key_pos_3d)
                attn_logits = attn_logits + pos_bias

        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [Q, N] -> [1, 1, Q, N]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # [B*H, Q, N] -> [B, H, Q, N]
                attn_mask = attn_mask.view(B, self.num_heads, Q, N)

            if attn_mask.dtype == torch.bool:
                attn_logits = attn_logits.masked_fill(attn_mask, float("-inf"))
            else:
                attn_logits = attn_logits + attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            # [B, N] -> [B, 1, 1, N]
            attn_logits = attn_logits.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # Numerical stability: clamp attention logits
        attn_logits = attn_logits.clamp(min=-self.attn_clamp_value, max=self.attn_clamp_value)

        # Softmax
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Dropout
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [B, H, Q, head_dim]

        # Add value bias (if using table-based RPE with value_bias)
        # Note: MLP-based RPE does not support value_bias
        if self.use_rpe and self.rpe is not None and self.rpe_type == "table":
            if hasattr(self.rpe, "use_value_bias") and self.rpe.use_value_bias:
                if query_pos_3d is not None and key_pos_3d is not None:
                    rel_idx = self.rpe.compute_relative_indices(query_pos_3d, key_pos_3d)
                    value_bias = self.rpe.forward_value_bias(attn_weights, rel_idx)
                    if value_bias is not None:
                        output = output + value_bias

        # Reshape back: [B, H, Q, head_dim] -> [B, Q, D]
        output = output.transpose(1, 2).contiguous().view(B, Q, self.embed_dim)

        # Output projection
        output = self.out_proj(output)

        if need_weights:
            return output, attn_weights
        return output, None

    def extra_repr(self) -> str:
        rpe_info = f"use_rpe={self.use_rpe}"
        if self.use_rpe:
            rpe_info += f", rpe_type={self.rpe_type}"
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"dropout={self.dropout}, {rpe_info}"
        )


class TransformerDecoderLayerWithRPE(nn.Module):
    """
    Transformer Decoder Layer with optional RPE in cross-attention.

    Structure:
        1. Self-Attention (queries attend to queries, additive pos encoding)
        2. Cross-Attention (queries attend to memory, RPE for position bias)
        3. Feed-Forward Network

    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: FFN intermediate dimension
        dropout: Dropout probability
        use_rpe: Whether to use RPE in cross-attention
        rpe_config: Configuration for RPE module
    """

    def __init__(
        self,
        d_model: int = 448,
        nhead: int = 8,
        dim_feedforward: int = 1792,
        dropout: float = 0.1,
        use_rpe: bool = True,
        rpe_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.use_rpe = use_rpe

        # Self-attention (no RPE, uses additive position encoding)
        self.self_attn = MultiheadAttentionWithRPE(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            use_rpe=False,  # Self-attention doesn't use RPE
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention (with RPE)
        self.cross_attn = MultiheadAttentionWithRPE(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            use_rpe=use_rpe,
            rpe_config=rpe_config,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor: torch.Tensor, pos: Optional[torch.Tensor]):
        """Add positional embedding to tensor."""
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        query_pos_3d: Optional[torch.Tensor] = None,
        memory_pos_3d: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            query: [B, Q, D] query features (content queries)
            memory: [B, N, D] memory features (point features)
            query_pos: [B, Q, D] query position embedding (for additive encoding)
            memory_pos: [B, N, D] memory position embedding (for additive encoding)
            query_pos_3d: [B, Q, 3] query 3D positions (for RPE)
            memory_pos_3d: [B, N, 3] memory 3D positions (for RPE, usually = points)
            memory_key_padding_mask: [B, N] padding mask for memory

        Returns:
            query: [B, Q, D] updated query features
        """
        # ========== 1. Self-Attention ==========
        # Queries attend to each other using additive position encoding
        q = k = self.with_pos_embed(query, query_pos)
        query2, _ = self.self_attn(q, k, query)
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # ========== 2. Cross-Attention ==========
        # Queries attend to memory with RPE for position bias
        q = self.with_pos_embed(query, query_pos)
        k = self.with_pos_embed(memory, memory_pos)
        query2, _ = self.cross_attn(
            q,
            k,
            memory,
            query_pos_3d=query_pos_3d,
            key_pos_3d=memory_pos_3d,
            key_padding_mask=memory_key_padding_mask,
        )
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # ========== 3. Feed-Forward Network ==========
        query2 = self.ffn(query)
        query = query + query2
        query = self.norm3(query)

        return query

    def extra_repr(self) -> str:
        rpe_info = f"use_rpe={self.use_rpe}"
        if self.use_rpe and hasattr(self.cross_attn, "rpe_type"):
            rpe_info += f", rpe_type={self.cross_attn.rpe_type}"
        return f"d_model={self.d_model}, nhead={self.nhead}, {rpe_info}"
