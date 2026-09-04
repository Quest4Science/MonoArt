"""
Relative Position Encoding (RPE) for 3D Point Cloud Transformers.

This module implements the contextual relative position encoding as described in
the MAFT paper, which computes position bias based on the relative 3D positions
between queries and keys.

Key formula:
    r = Qp - P                           # relative position
    r_hat = floor(r / s) + L             # quantization
    f_pos = table[0, r_hat_x] + table[1, r_hat_y] + table[2, r_hat_z]
    pos_bias = f_pos · f_q + f_pos · f_k
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Check PyTorch version for checkpoint compatibility
_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
_CHECKPOINT_SUPPORTS_USE_REENTRANT = _TORCH_VERSION >= (1, 11)


class RelativePositionBias3D(nn.Module):
    """
    3D Relative Position Bias module.

    Computes position bias for attention based on the relative 3D positions
    between query and key positions. The relative positions are quantized
    into discrete buckets and used to index learnable position embedding tables.

    Args:
        num_heads: Number of attention heads
        head_dim: Dimension per head (d_model // num_heads)
        grid_size: Quantization step size in meters (default: 0.05)
        num_buckets: Number of buckets on each side (L), total buckets = 2L
        use_query_bias: Whether to use query position bias
        use_key_bias: Whether to use key position bias
        use_value_bias: Whether to use value position bias (for output)
    """

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 56,
        grid_size: float = 0.05,
        num_buckets: int = 24,
        use_query_bias: bool = True,
        use_key_bias: bool = True,
        use_value_bias: bool = False,
        chunk_size: int = 10,  # Number of queries to process at once (higher = faster but more memory)
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.grid_size = grid_size
        self.num_buckets = num_buckets
        self.total_buckets = 2 * num_buckets
        self.chunk_size = chunk_size  # Configurable chunk size for memory-speed tradeoff

        self.use_query_bias = use_query_bias
        self.use_key_bias = use_key_bias
        self.use_value_bias = use_value_bias

        # Learnable position embedding tables
        # Shape: [num_heads, head_dim, 3 * total_buckets]
        # We store all 3 axes in a single tensor for efficiency
        if use_query_bias:
            self.table_query = nn.Parameter(
                torch.zeros(num_heads, head_dim, 3 * self.total_buckets)
            )
            self._init_table(self.table_query)
        else:
            self.register_parameter("table_query", None)

        if use_key_bias:
            self.table_key = nn.Parameter(torch.zeros(num_heads, head_dim, 3 * self.total_buckets))
            self._init_table(self.table_key)
        else:
            self.register_parameter("table_key", None)

        if use_value_bias:
            self.table_value = nn.Parameter(
                torch.zeros(num_heads, head_dim, 3 * self.total_buckets)
            )
            self._init_table(self.table_value)
        else:
            self.register_parameter("table_value", None)

    def _init_table(self, table: torch.Tensor):
        """Initialize table with truncated normal distribution."""
        nn.init.trunc_normal_(table, std=0.02)

    @torch.no_grad()
    def compute_relative_indices(
        self,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute quantized relative position indices.

        Note: This function runs without gradient computation since
        indices don't need gradients (they are just integer lookups).

        Args:
            query_pos: [B, Q, 3] query positions in 3D space
            key_pos: [B, N, 3] key positions in 3D space

        Returns:
            rel_idx: [B, Q, N, 3] integer indices in range [0, 2L-1]
        """
        # Compute relative positions: [B, Q, N, 3]
        # query_pos[:, :, None, :] -> [B, Q, 1, 3]
        # key_pos[:, None, :, :] -> [B, 1, N, 3]
        rel_pos = query_pos.unsqueeze(2) - key_pos.unsqueeze(1)

        # Quantize: floor(rel_pos / grid_size)
        rel_idx = torch.div(rel_pos, self.grid_size, rounding_mode="floor").long()

        # Clamp to [-L, L-1] range
        rel_idx = rel_idx.clamp(-self.num_buckets, self.num_buckets - 1)

        # Shift to [0, 2L-1] range (non-negative for indexing)
        rel_idx = rel_idx + self.num_buckets

        return rel_idx

    def _gather_position_embedding_chunked(
        self,
        table: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        features: torch.Tensor,
        chunk_size: int = 10,
    ) -> torch.Tensor:
        """
        Memory-efficient version using chunked computation.

        Avoids creating [B, H, Q, N, D] tensor by processing queries in chunks.
        Also computes rel_idx on-the-fly to avoid storing the full [B, Q, N, 3] tensor.

        IMPORTANT: Uses list accumulation instead of in-place tensor accumulation
        to avoid autograd graph explosion that causes OOM during training.

        Args:
            table: [num_heads, head_dim, 3 * total_buckets]
            query_pos: [B, Q, 3] query positions
            key_pos: [B, N, 3] key positions
            features: [B, num_heads, L, head_dim] query or key features
            chunk_size: Number of queries to process at once

        Returns:
            bias: [B, num_heads, Q, N] position bias
        """
        B, Q, _ = query_pos.shape
        N = key_pos.shape[1]
        H = self.num_heads
        D = self.head_dim

        # Reshape table: [H, D, 3, L]
        table_3d = table.view(H, D, 3, self.total_buckets)

        # Collect chunk results in list (avoid autograd graph explosion)
        chunk_results = []

        # Determine if we're processing query or key features
        is_query = features.shape[2] == Q

        # Process in chunks to save memory
        for q_start in range(0, Q, chunk_size):
            q_end = min(q_start + chunk_size, Q)
            chunk_q = q_end - q_start

            # Compute rel_idx for this chunk only (saves memory!)
            # query_pos_chunk: [B, chunk_q, 3]
            query_pos_chunk = query_pos[:, q_start:q_end, :]

            # Compute relative positions for chunk: [B, chunk_q, N, 3]
            with torch.no_grad():
                rel_pos_chunk = query_pos_chunk.unsqueeze(2) - key_pos.unsqueeze(1)
                rel_idx_chunk = torch.div(
                    rel_pos_chunk, self.grid_size, rounding_mode="floor"
                ).long()
                rel_idx_chunk = rel_idx_chunk.clamp(-self.num_buckets, self.num_buckets - 1)
                rel_idx_chunk = rel_idx_chunk + self.num_buckets

            # Accumulate bias for this chunk across all 3 axes
            chunk_bias = None

            # For each axis
            for axis in range(3):
                # Get indices for this axis: [B, chunk_q, N]
                idx = rel_idx_chunk[..., axis]

                # Get table for this axis: [H, D, L]
                table_axis = table_3d[:, :, axis, :]

                # Flatten indices: [B * chunk_q * N]
                idx_flat = idx.flatten()

                # Gather: [H, D, B * chunk_q * N]
                pos_embed_flat = table_axis[:, :, idx_flat]

                # Reshape: [H, D, B, chunk_q, N] -> [B, H, chunk_q, N, D]
                pos_embed = pos_embed_flat.view(H, D, B, chunk_q, N).permute(2, 0, 3, 4, 1)

                if is_query:
                    # Query features chunk: [B, H, chunk_q, D]
                    feat_chunk = features[:, :, q_start:q_end, :]
                    # [B, H, chunk_q, 1, D] * [B, H, chunk_q, N, D] -> sum -> [B, H, chunk_q, N]
                    dot_prod = torch.einsum("bhqd,bhqnd->bhqn", feat_chunk, pos_embed)
                else:
                    # Key features: [B, H, N, D]
                    # [B, H, 1, N, D] * [B, H, chunk_q, N, D] -> sum -> [B, H, chunk_q, N]
                    dot_prod = torch.einsum("bhnd,bhqnd->bhqn", features, pos_embed)

                # Accumulate within chunk (only 3 operations, manageable)
                if chunk_bias is None:
                    chunk_bias = dot_prod
                else:
                    chunk_bias = chunk_bias + dot_prod

                # Clean up intermediate tensors
                del pos_embed_flat, pos_embed, dot_prod

            # Store chunk result
            chunk_results.append(chunk_bias)

            # Explicitly delete chunk tensors to free memory
            del rel_pos_chunk, rel_idx_chunk, chunk_bias

        # Concatenate all chunks along Q dimension: [B, H, Q, N]
        bias = torch.cat(chunk_results, dim=2)

        return bias

    def _gather_position_embedding(
        self,
        table: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather position embeddings and compute dot product with features.

        Uses memory-efficient chunked computation for large point clouds.

        Args:
            table: [num_heads, head_dim, 3 * total_buckets]
            query_pos: [B, Q, 3] query positions
            key_pos: [B, N, 3] key positions
            features: [B, num_heads, L, head_dim] query or key features
                      where L = Q for query, L = N for key

        Returns:
            bias: [B, num_heads, Q, N] position bias
        """
        B, Q, _ = query_pos.shape
        N = key_pos.shape[1]

        # Use chunked version for large N to avoid OOM
        # Threshold: if Q * N * D > 50M elements, use chunked
        estimated_elements = Q * N * self.head_dim
        if estimated_elements > 50_000_000:
            # Compute optimal chunk size based on available memory
            # Target: ~200MB for intermediate tensor (conservative for tight memory)
            # [B, H, chunk_q, N, D] * 4 bytes < 200MB
            # chunk_q < 200MB / (B * H * N * D * 4)
            target_bytes = 200 * 1024 * 1024  # 200MB (conservative)
            chunk_size = max(1, target_bytes // (B * self.num_heads * N * self.head_dim * 4))
            chunk_size = min(chunk_size, Q, 5)  # Cap at 5 for very tight memory
            return self._gather_position_embedding_chunked(
                table, query_pos, key_pos, features, chunk_size
            )

        # Original fast implementation for smaller inputs
        # Compute rel_idx (with no_grad since indices don't need gradients)
        rel_idx = self.compute_relative_indices(query_pos, key_pos)

        H = self.num_heads
        D = self.head_dim
        L = self.total_buckets

        # Reshape table for easier indexing
        table_3d = table.view(H, D, 3, L)

        bias = features.new_zeros(B, H, Q, N)

        for axis in range(3):
            idx = rel_idx[..., axis]  # [B, Q, N]
            table_axis = table_3d[:, :, axis, :]
            idx_flat = idx.flatten()
            pos_embed_flat = table_axis[:, :, idx_flat]
            pos_embed = pos_embed_flat.view(H, D, B, Q, N)
            pos_embed = pos_embed.permute(2, 0, 3, 4, 1)

            if features.shape[2] == Q:
                feat_expanded = features.unsqueeze(3)
                dot_prod = (feat_expanded * pos_embed).sum(-1)
            else:
                feat_expanded = features.unsqueeze(2)
                dot_prod = (feat_expanded * pos_embed).sum(-1)

            bias = bias + dot_prod

        return bias

    def _compute_bias_memory_efficient(
        self,
        table: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Memory-efficient position bias computation.

        Strategy:
        - Query bias: Use vectorized gather (fast, Q is small)
        - Key bias: Use per-q loop with list accumulation (memory-safe for DDP)

        The chunk_size parameter controls batching for query bias only.
        Key bias always uses chunk_size=1 for DDP compatibility with old PyTorch.
        """
        B, Q, _ = query_pos.shape
        N = key_pos.shape[1]
        H = self.num_heads
        L = self.total_buckets
        chunk_size = self.chunk_size

        # Compute relative indices (no grad needed for integer indices)
        with torch.no_grad():
            rel_pos = query_pos.unsqueeze(2) - key_pos.unsqueeze(1)  # [B, Q, N, 3]
            rel_idx = torch.div(rel_pos, self.grid_size, rounding_mode="floor").long()
            rel_idx = rel_idx.clamp(-self.num_buckets, self.num_buckets - 1)
            rel_idx = rel_idx + self.num_buckets  # [B, Q, N, 3], values in [0, 2L-1]

        # Reshape table: [H, D, 3*L] -> [H, D, 3, L]
        table_3d = table.view(H, self.head_dim, 3, L)

        is_query = features.shape[2] == Q

        if is_query:
            # ===== Query bias: can use larger chunks (vectorized) =====
            # features: [B, H, Q, D]
            # Compute: features @ table -> [B, H, Q, 3, L]
            f_table = torch.einsum("bhqd,hdal->bhqal", features, table_3d)  # [B, H, Q, 3, L]

            # Process in chunks for better GPU utilization
            chunk_results = []
            for q_start in range(0, Q, chunk_size):
                q_end = min(q_start + chunk_size, Q)
                chunk_q = q_end - q_start

                # Chunk bias accumulator
                chunk_bias = features.new_zeros(B, H, chunk_q, N)

                for axis in range(3):
                    # f_table for this chunk: [B, H, chunk_q, L]
                    f_chunk = f_table[:, :, q_start:q_end, axis, :]
                    # idx for this chunk: [B, chunk_q, N]
                    idx_chunk = rel_idx[:, q_start:q_end, :, axis]

                    # Vectorized gather: flatten B*H*chunk_q, gather, reshape back
                    f_flat = f_chunk.reshape(B * H * chunk_q, L)
                    idx_exp = (
                        idx_chunk.unsqueeze(1).expand(-1, H, -1, -1).reshape(B * H * chunk_q, N)
                    )

                    # Gather: [B*H*chunk_q, N]
                    gathered = f_flat.gather(dim=-1, index=idx_exp)

                    # Reshape back: [B, H, chunk_q, N]
                    chunk_bias = chunk_bias + gathered.view(B, H, chunk_q, N)

                chunk_results.append(chunk_bias)

            bias = torch.cat(chunk_results, dim=2)  # [B, H, Q, N]

        else:
            # ===== Key bias: use per-q loop for DDP compatibility =====
            # For PyTorch 1.10 + DDP, complex chunk graphs cause OOM
            # Simple per-q loop with list accumulation is more memory-stable
            #
            # features: [B, H, N, D] - key features
            # Compute: features @ table -> [B, H, N, 3, L]
            f_table = torch.einsum("bhnd,hdal->bhnal", features, table_3d)  # [B, H, N, 3, L]

            # Optimization: process all 3 axes together per q (reduce loop overhead)
            # f_table: [B, H, N, 3, L] -> [B, H, N, 3*L] for combined gather
            f_combined = f_table.view(B, H, N, 3 * L)

            # Process each q independently (flat autograd graph)
            q_results = []
            for q in range(Q):
                # idx for this q: [B, N, 3] -> need to offset for combined table
                idx_q_3axis = rel_idx[:, q, :, :]  # [B, N, 3]

                # Offset indices: axis0 stays, axis1 += L, axis2 += 2L
                offsets = torch.tensor([0, L, 2 * L], device=idx_q_3axis.device)
                idx_q_offset = idx_q_3axis + offsets  # [B, N, 3]

                # Expand to [B, H, N, 3]
                idx_exp = idx_q_offset.unsqueeze(1).expand(-1, H, -1, -1)

                # Gather from combined table: [B, H, N, 3*L] with idx [B, H, N, 3]
                gathered = f_combined.gather(dim=-1, index=idx_exp)  # [B, H, N, 3]

                # Sum over axes
                q_bias = gathered.sum(dim=-1)  # [B, H, N]

                q_results.append(q_bias.unsqueeze(2))  # [B, H, 1, N]

            bias = torch.cat(q_results, dim=2)  # [B, H, Q, N]

        return bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        use_checkpoint: bool = True,
    ) -> torch.Tensor:
        """
        Compute position bias for attention.

        Args:
            query: [B, num_heads, Q, head_dim] query features
            key: [B, num_heads, N, head_dim] key features
            query_pos: [B, Q, 3] query 3D positions
            key_pos: [B, N, 3] key 3D positions
            use_checkpoint: Use gradient checkpointing to save memory

        Returns:
            pos_bias: [B, num_heads, Q, N] position bias to add to attention logits
        """
        # Use gradient checkpointing for training to save memory
        # This trades compute for memory by not saving intermediate activations
        #
        # NOTE: PyTorch 1.10's checkpoint uses reentrant mode by default,
        # which conflicts with DDP (causes "variable ready twice" error).
        # Only PyTorch 1.11+ supports use_reentrant=False to fix this.
        # So we disable checkpoint for old PyTorch versions to ensure DDP compatibility.
        #
        should_checkpoint = (
            use_checkpoint
            and self.training
            and _CHECKPOINT_SUPPORTS_USE_REENTRANT  # Only checkpoint with PyTorch 1.11+
        )

        if should_checkpoint:
            pos_bias = checkpoint(
                self._forward_impl, query, key, query_pos, key_pos, use_reentrant=False
            )
        else:
            pos_bias = self._forward_impl(query, key, query_pos, key_pos)

        return pos_bias

    def _forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Internal forward implementation, can be wrapped by checkpoint."""
        B, H, Q, D = query.shape
        N = key.shape[2]

        # Initialize bias
        pos_bias = query.new_zeros(B, H, Q, N)

        # Query bias: f_pos · f_q
        if self.table_query is not None:
            pos_bias = pos_bias + self._compute_bias_memory_efficient(
                self.table_query, query_pos, key_pos, query
            )

        # Key bias: f_pos · f_k
        if self.table_key is not None:
            pos_bias = pos_bias + self._compute_bias_memory_efficient(
                self.table_key, query_pos, key_pos, key
            )

        return pos_bias

    def forward_value_bias(
        self,
        attn_weights: torch.Tensor,
        rel_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute value position bias (applied to attention output).

        This is optional and adds f_pos to the attention-weighted values.

        Args:
            attn_weights: [B, num_heads, Q, N] attention weights (after softmax)
            rel_idx: [B, Q, N, 3] relative position indices

        Returns:
            value_bias: [B, num_heads, Q, head_dim] bias to add to attention output
        """
        if self.table_value is None:
            return None

        B, H, Q, N = attn_weights.shape
        D = self.head_dim
        L = self.total_buckets

        # Reshape table: [H, D, 3, L]
        table_3d = self.table_value.view(H, D, 3, L)

        # For each axis, gather and weight by attention
        value_bias = attn_weights.new_zeros(B, H, Q, D)

        for axis in range(3):
            idx = rel_idx[..., axis]  # [B, Q, N]
            table_axis = table_3d[:, :, axis, :]  # [H, D, L]

            # Gather: [H, D, B*Q*N]
            idx_flat = idx.flatten()
            pos_embed_flat = table_axis[:, :, idx_flat]

            # Reshape: [B, H, Q, N, D]
            pos_embed = pos_embed_flat.view(H, D, B, Q, N).permute(2, 0, 3, 4, 1)

            # Weight by attention: [B, H, Q, N, 1] * [B, H, Q, N, D] -> sum over N
            weighted = attn_weights.unsqueeze(-1) * pos_embed
            value_bias = value_bias + weighted.sum(3)

        return value_bias

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"grid_size={self.grid_size}, num_buckets={self.num_buckets}, "
            f"use_query_bias={self.use_query_bias}, use_key_bias={self.use_key_bias}, "
            f"use_value_bias={self.use_value_bias}"
        )


class RelativePositionBias3DFast(nn.Module):
    """
    Optimized version of RelativePositionBias3D using einsum operations.

    This version is more memory-efficient for large point clouds.
    """

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 56,
        grid_size: float = 0.05,
        num_buckets: int = 24,
        use_query_bias: bool = True,
        use_key_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.grid_size = grid_size
        self.num_buckets = num_buckets
        self.total_buckets = 2 * num_buckets

        # Tables: [num_heads, head_dim, 3, total_buckets]
        if use_query_bias:
            self.table_query = nn.Parameter(torch.zeros(num_heads, head_dim, 3, self.total_buckets))
            nn.init.trunc_normal_(self.table_query, std=0.02)
        else:
            self.register_parameter("table_query", None)

        if use_key_bias:
            self.table_key = nn.Parameter(torch.zeros(num_heads, head_dim, 3, self.total_buckets))
            nn.init.trunc_normal_(self.table_key, std=0.02)
        else:
            self.register_parameter("table_key", None)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute position bias using optimized operations.

        Args:
            query: [B, H, Q, D]
            key: [B, H, N, D]
            query_pos: [B, Q, 3]
            key_pos: [B, N, 3]

        Returns:
            pos_bias: [B, H, Q, N]
        """
        B, H, Q, D = query.shape
        N = key.shape[2]

        # Compute relative positions and quantize
        rel_pos = query_pos.unsqueeze(2) - key_pos.unsqueeze(1)  # [B, Q, N, 3]
        rel_idx = torch.div(rel_pos, self.grid_size, rounding_mode="floor").long()
        rel_idx = rel_idx.clamp(-self.num_buckets, self.num_buckets - 1)
        rel_idx = rel_idx + self.num_buckets  # [B, Q, N, 3]

        pos_bias = query.new_zeros(B, H, Q, N)

        # Process query bias
        if self.table_query is not None:
            # query: [B, H, Q, D]
            # We compute: q @ table[axis, rel_idx] for each axis and position
            # This is: sum_d q[b,h,q,d] * table[h,d,axis,rel_idx[b,q,n,axis]]

            # First, compute q @ table -> [B, H, Q, 3, L]
            # table_query: [H, D, 3, L]
            q_table = torch.einsum("bhqd,hdal->bhqal", query, self.table_query)
            # q_table: [B, H, Q, 3, L]

            # Now gather using rel_idx
            for axis in range(3):
                idx = rel_idx[..., axis]  # [B, Q, N]
                # q_table[..., axis, :]: [B, H, Q, L]
                # We want to gather: q_table[b, h, q, axis, idx[b,q,n]]
                idx_expanded = idx.unsqueeze(1).expand(-1, H, -1, -1)  # [B, H, Q, N]
                bias_axis = torch.gather(
                    q_table[..., axis, :].unsqueeze(3).expand(-1, -1, -1, N, -1),
                    dim=-1,
                    index=idx_expanded.unsqueeze(-1),
                ).squeeze(-1)  # [B, H, Q, N]
                pos_bias = pos_bias + bias_axis

        # Process key bias
        if self.table_key is not None:
            # key: [B, H, N, D]
            # k @ table -> [B, H, N, 3, L]
            k_table = torch.einsum("bhnd,hdal->bhnal", key, self.table_key)

            for axis in range(3):
                idx = rel_idx[..., axis]  # [B, Q, N]
                # k_table[..., axis, :]: [B, H, N, L]
                # We want: k_table[b, h, n, axis, idx[b,q,n]]
                idx_expanded = idx.unsqueeze(1).expand(-1, H, -1, -1)  # [B, H, Q, N]
                # Need to transpose for proper indexing
                k_table_axis = k_table[..., axis, :]  # [B, H, N, L]
                # Expand Q dimension
                k_table_expanded = k_table_axis.unsqueeze(2).expand(
                    -1, -1, Q, -1, -1
                )  # [B, H, Q, N, L]
                bias_axis = torch.gather(
                    k_table_expanded, dim=-1, index=idx_expanded.unsqueeze(-1)
                ).squeeze(-1)  # [B, H, Q, N]
                pos_bias = pos_bias + bias_axis

        return pos_bias
