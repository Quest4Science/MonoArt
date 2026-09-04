"""
MLP-based Relative Position Encoding for 3D Point Cloud Transformers.

This module implements continuous relative position encoding using MLPs,
as an alternative to the lookup table approach. MLP-based RPE has denser
gradients and learns faster, making it easier to train.

Key advantages over lookup table:
- Dense gradients: all parameters receive gradients every forward pass
- Continuous output: smooth position bias without quantization artifacts
- Better extrapolation: can handle positions outside training range
- Simpler training: no need for higher learning rates

Architecture options:
1. Direct MLP: r -> MLP -> bias
2. Fourier + MLP: r -> sin/cos encoding -> MLP -> bias (better for high-freq patterns)
3. Gaussian RBF + Linear: r -> RBF features -> Linear -> bias (interpretable)
"""

import math

import torch
import torch.nn as nn


class FourierEncoding(nn.Module):
    """
    Fourier feature encoding for continuous positions.

    Maps 3D position to higher-dimensional space using sin/cos functions.
    This helps MLPs learn high-frequency patterns.

    f(r) = [sin(2πσ₁r), cos(2πσ₁r), sin(2πσ₂r), cos(2πσ₂r), ...]

    Args:
        input_dim: Input dimension (3 for 3D positions)
        num_frequencies: Number of frequency bands
        max_freq: Maximum frequency (controls the finest detail)
        include_input: Whether to include original input in output
    """

    def __init__(
        self,
        input_dim: int = 3,
        num_frequencies: int = 8,
        max_freq: float = 10.0,
        include_input: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Frequency bands: logarithmically spaced from 1 to max_freq
        freqs = 2.0 ** torch.linspace(0, math.log2(max_freq), num_frequencies)
        self.register_buffer("freqs", freqs)  # [num_frequencies]

        # Output dimension
        self.output_dim = input_dim * num_frequencies * 2  # sin + cos
        if include_input:
            self.output_dim += input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., input_dim] input positions

        Returns:
            [..., output_dim] Fourier features
        """
        # x: [..., 3]
        # freqs: [num_frequencies]
        # x_freq: [..., 3, num_frequencies]
        x_freq = x.unsqueeze(-1) * self.freqs * 2 * math.pi

        # Compute sin and cos: [..., 3, num_frequencies, 2]
        encoded = torch.stack([x_freq.sin(), x_freq.cos()], dim=-1)

        # Flatten: [..., 3 * num_frequencies * 2]
        encoded = encoded.flatten(-3)

        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)

        return encoded


class RelativePositionMLP(nn.Module):
    """
    MLP-based Relative Position Bias module.

    Computes position bias for attention based on relative 3D positions
    using a small MLP network instead of lookup tables.

    Args:
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension of MLP
        num_layers: Number of MLP layers (2-3 recommended)
        use_fourier: Use Fourier encoding for input positions
        num_frequencies: Number of Fourier frequency bands
        max_freq: Maximum frequency for Fourier encoding
        dropout: Dropout probability in MLP
    """

    def __init__(
        self,
        num_heads: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_fourier: bool = True,
        num_frequencies: int = 8,
        max_freq: float = 10.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.use_fourier = use_fourier

        # Position encoding
        if use_fourier:
            self.pos_encoder = FourierEncoding(
                input_dim=3,
                num_frequencies=num_frequencies,
                max_freq=max_freq,
                include_input=True,
            )
            input_dim = self.pos_encoder.output_dim
        else:
            self.pos_encoder = None
            input_dim = 3

        # Build MLP
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = num_heads if i == num_layers - 1 else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))

            if i < num_layers - 1:  # No activation/dropout on last layer
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

        # Initialize output layer to small values
        # This ensures initial pos_bias is small and doesn't disrupt attention
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=0.01)

    def forward(
        self,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute position bias for attention.

        Args:
            query_pos: [B, Q, 3] query 3D positions
            key_pos: [B, N, 3] key 3D positions

        Returns:
            pos_bias: [B, num_heads, Q, N] position bias to add to attention logits
        """
        # Compute relative positions: [B, Q, N, 3]
        # query_pos: [B, Q, 1, 3]
        # key_pos: [B, 1, N, 3]
        rel_pos = query_pos.unsqueeze(2) - key_pos.unsqueeze(1)

        # Encode positions
        if self.pos_encoder is not None:
            rel_encoded = self.pos_encoder(rel_pos)  # [B, Q, N, encoded_dim]
        else:
            rel_encoded = rel_pos  # [B, Q, N, 3]

        # MLP forward: [B, Q, N, num_heads]
        pos_bias = self.mlp(rel_encoded)

        # Transpose to [B, num_heads, Q, N]
        pos_bias = pos_bias.permute(0, 3, 1, 2)

        return pos_bias

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, hidden_dim={self.hidden_dim}, "
            f"use_fourier={self.use_fourier}"
        )


class RelativePositionMLPEfficient(nn.Module):
    """
    Memory-efficient version of RelativePositionMLP.

    For large point clouds (N > 10000), computing all pairwise relative
    positions at once may cause OOM. This version processes in chunks.

    Args:
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension of MLP
        num_layers: Number of MLP layers
        use_fourier: Use Fourier encoding
        num_frequencies: Number of Fourier frequency bands
        max_freq: Maximum frequency
        chunk_size: Number of queries to process at once
    """

    def __init__(
        self,
        num_heads: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_fourier: bool = True,
        num_frequencies: int = 8,
        max_freq: float = 10.0,
        chunk_size: int = 10,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.chunk_size = chunk_size

        # Core MLP module
        self.rpe_mlp = RelativePositionMLP(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_fourier=use_fourier,
            num_frequencies=num_frequencies,
            max_freq=max_freq,
        )

    def forward(
        self,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute position bias with chunked processing.

        Args:
            query_pos: [B, Q, 3] query 3D positions
            key_pos: [B, N, 3] key 3D positions

        Returns:
            pos_bias: [B, num_heads, Q, N] position bias
        """
        B, Q, _ = query_pos.shape
        N = key_pos.shape[1]

        # Estimate memory usage: B * Q * N * encoded_dim * 4 bytes
        # If too large, use chunked processing
        estimated_bytes = B * Q * N * 64 * 4  # Assume ~64 encoded dim
        memory_threshold = 500 * 1024 * 1024  # 500 MB

        if estimated_bytes < memory_threshold:
            # Direct computation
            return self.rpe_mlp(query_pos, key_pos)

        # Chunked computation
        chunk_results = []
        for q_start in range(0, Q, self.chunk_size):
            q_end = min(q_start + self.chunk_size, Q)
            query_chunk = query_pos[:, q_start:q_end, :]  # [B, chunk_q, 3]

            # Compute bias for this chunk
            chunk_bias = self.rpe_mlp(query_chunk, key_pos)  # [B, H, chunk_q, N]
            chunk_results.append(chunk_bias)

        # Concatenate: [B, H, Q, N]
        pos_bias = torch.cat(chunk_results, dim=2)

        return pos_bias

    def extra_repr(self) -> str:
        return f"chunk_size={self.chunk_size}, " + self.rpe_mlp.extra_repr()


class RelativePositionMLPKeyChunked(nn.Module):
    """
    Memory-efficient MLP RPE with Key-based chunking.

    This implementation chunks along the Key (point cloud) dimension,
    which is more effective when N >> Q (typical case: N=100k, Q=200).

    MATHEMATICALLY EQUIVALENT to the original MLP approach!

    Memory comparison (Q=200, N=100000, chunk_n=1000):
        Original: Q × N × D = 200 × 100000 × 64 × 4 = 4.8 GB
        Chunked:  Q × chunk_n × D = 200 × 1000 × 64 × 4 = 51 MB
        Reduction: ~100x

    Args:
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension of MLP
        num_layers: Number of MLP layers
        use_fourier: Use Fourier encoding
        num_frequencies: Number of Fourier frequency bands
        max_freq: Maximum frequency
        chunk_size: Number of keys to process at once (default: 1000)
    """

    def __init__(
        self,
        num_heads: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_fourier: bool = True,
        num_frequencies: int = 8,
        max_freq: float = 10.0,
        chunk_size: int = 1000,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.use_fourier = use_fourier

        # Position encoding
        if use_fourier:
            self.pos_encoder = FourierEncoding(
                input_dim=3,
                num_frequencies=num_frequencies,
                max_freq=max_freq,
                include_input=True,
            )
            input_dim = self.pos_encoder.output_dim
        else:
            self.pos_encoder = None
            input_dim = 3

        # Build MLP
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = num_heads if i == num_layers - 1 else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))

            if i < num_layers - 1:
                layers.append(nn.GELU())

        self.mlp = nn.Sequential(*layers)

        # Initialize output layer to small values
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=0.01)

    def _compute_chunk(
        self,
        query_pos: torch.Tensor,
        key_pos_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute position bias for a chunk of keys.

        Args:
            query_pos: [B, Q, 3] all query positions
            key_pos_chunk: [B, chunk_n, 3] chunk of key positions

        Returns:
            pos_bias_chunk: [B, num_heads, Q, chunk_n]
        """
        # Compute relative positions: [B, Q, chunk_n, 3]
        rel_pos = query_pos.unsqueeze(2) - key_pos_chunk.unsqueeze(1)

        # Encode positions
        if self.pos_encoder is not None:
            rel_encoded = self.pos_encoder(rel_pos)
        else:
            rel_encoded = rel_pos

        # MLP forward: [B, Q, chunk_n, num_heads]
        pos_bias = self.mlp(rel_encoded)

        # Transpose to [B, num_heads, Q, chunk_n]
        pos_bias = pos_bias.permute(0, 3, 1, 2)

        return pos_bias

    def forward(
        self,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute position bias with key-based chunking.

        This is MATHEMATICALLY EQUIVALENT to the original MLP approach,
        but uses much less memory by processing keys in chunks.

        Args:
            query_pos: [B, Q, 3] query 3D positions
            key_pos: [B, N, 3] key 3D positions

        Returns:
            pos_bias: [B, num_heads, Q, N] position bias
        """
        B, Q, _ = query_pos.shape
        N = key_pos.shape[1]

        # If small enough, compute directly
        estimated_bytes = B * Q * N * 64 * 4  # Assume ~64 encoded dim
        memory_threshold = 200 * 1024 * 1024  # 200 MB threshold

        if estimated_bytes < memory_threshold:
            return self._compute_chunk(query_pos, key_pos)

        # Chunked computation along key dimension
        chunk_results = []
        for n_start in range(0, N, self.chunk_size):
            n_end = min(n_start + self.chunk_size, N)
            key_chunk = key_pos[:, n_start:n_end, :]  # [B, chunk_n, 3]

            # Compute bias for this chunk: [B, H, Q, chunk_n]
            chunk_bias = self._compute_chunk(query_pos, key_chunk)
            chunk_results.append(chunk_bias)

        # Concatenate along key dimension: [B, H, Q, N]
        pos_bias = torch.cat(chunk_results, dim=-1)

        return pos_bias

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, chunk_size={self.chunk_size}, "
            f"use_fourier={self.use_fourier}"
        )


class RelativePositionMLPDoubleChunked(nn.Module):
    """
    Ultra memory-efficient MLP RPE with double chunking (both Q and N).

    For extreme memory constraints, this chunks both query and key dimensions.

    MATHEMATICALLY EQUIVALENT to the original MLP approach!

    Memory comparison (Q=200, N=100000, chunk_q=20, chunk_n=1000):
        Original: Q × N × D = 200 × 100000 × 64 × 4 = 4.8 GB
        Double chunked: chunk_q × chunk_n × D = 20 × 1000 × 64 × 4 = 5 MB
        Reduction: ~1000x

    Trade-off: More kernel launches = slower computation

    Args:
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension of MLP
        num_layers: Number of MLP layers
        use_fourier: Use Fourier encoding
        num_frequencies: Number of Fourier frequency bands
        max_freq: Maximum frequency
        chunk_q: Number of queries to process at once
        chunk_n: Number of keys to process at once
    """

    def __init__(
        self,
        num_heads: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_fourier: bool = True,
        num_frequencies: int = 8,
        max_freq: float = 10.0,
        chunk_q: int = 20,
        chunk_n: int = 1000,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.chunk_q = chunk_q
        self.chunk_n = chunk_n
        self.use_fourier = use_fourier

        # Position encoding
        if use_fourier:
            self.pos_encoder = FourierEncoding(
                input_dim=3,
                num_frequencies=num_frequencies,
                max_freq=max_freq,
                include_input=True,
            )
            input_dim = self.pos_encoder.output_dim
        else:
            self.pos_encoder = None
            input_dim = 3

        # Build MLP
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = num_heads if i == num_layers - 1 else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))

            if i < num_layers - 1:
                layers.append(nn.GELU())

        self.mlp = nn.Sequential(*layers)

        # Initialize output layer to small values
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=0.01)

    def _compute_chunk(
        self,
        query_pos_chunk: torch.Tensor,
        key_pos_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute position bias for a small chunk.

        Args:
            query_pos_chunk: [B, chunk_q, 3]
            key_pos_chunk: [B, chunk_n, 3]

        Returns:
            pos_bias_chunk: [B, num_heads, chunk_q, chunk_n]
        """
        # Compute relative positions: [B, chunk_q, chunk_n, 3]
        rel_pos = query_pos_chunk.unsqueeze(2) - key_pos_chunk.unsqueeze(1)

        # Encode positions
        if self.pos_encoder is not None:
            rel_encoded = self.pos_encoder(rel_pos)
        else:
            rel_encoded = rel_pos

        # MLP forward: [B, chunk_q, chunk_n, num_heads]
        pos_bias = self.mlp(rel_encoded)

        # Transpose to [B, num_heads, chunk_q, chunk_n]
        pos_bias = pos_bias.permute(0, 3, 1, 2)

        return pos_bias

    def forward(
        self,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute position bias with double chunking.

        MATHEMATICALLY EQUIVALENT to the original MLP approach.

        Args:
            query_pos: [B, Q, 3] query 3D positions
            key_pos: [B, N, 3] key 3D positions

        Returns:
            pos_bias: [B, num_heads, Q, N] position bias
        """
        B, Q, _ = query_pos.shape
        N = key_pos.shape[1]
        H = self.num_heads

        # Allocate output tensor
        pos_bias = query_pos.new_zeros(B, H, Q, N)

        # Double loop: chunk both Q and N
        for q_start in range(0, Q, self.chunk_q):
            q_end = min(q_start + self.chunk_q, Q)
            query_chunk = query_pos[:, q_start:q_end, :]

            for n_start in range(0, N, self.chunk_n):
                n_end = min(n_start + self.chunk_n, N)
                key_chunk = key_pos[:, n_start:n_end, :]

                # Compute this small chunk
                chunk_bias = self._compute_chunk(query_chunk, key_chunk)

                # Write to output
                pos_bias[:, :, q_start:q_end, n_start:n_end] = chunk_bias

        return pos_bias

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, chunk_q={self.chunk_q}, "
            f"chunk_n={self.chunk_n}, use_fourier={self.use_fourier}"
        )


class RelativePositionMLPCheckpoint(nn.Module):
    """
    Memory-efficient MLP RPE with Gradient Checkpointing.

    This implementation uses torch.utils.checkpoint to avoid storing
    intermediate activations during forward pass. Instead, they are
    recomputed during backward pass.

    MATHEMATICALLY EQUIVALENT to the original MLP approach!

    Trade-off:
        Memory: ↓ ~70% (no intermediate storage)
        Speed:  ↑ ~25-30% slower (recompute during backward)

    Memory comparison (Q=100, N=100000, chunk_n=1000):
        Without checkpoint: ~7.4 GB (stores all chunk intermediates)
        With checkpoint:    ~0.7 GB (only stores output + recomputes)
        Reduction: ~10x

    Args:
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension of MLP
        num_layers: Number of MLP layers
        use_fourier: Use Fourier encoding
        num_frequencies: Number of Fourier frequency bands
        max_freq: Maximum frequency
        chunk_size: Number of keys to process at once
    """

    def __init__(
        self,
        num_heads: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_fourier: bool = True,
        num_frequencies: int = 8,
        max_freq: float = 10.0,
        chunk_size: int = 1000,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.use_fourier = use_fourier
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_frequencies = num_frequencies
        self.max_freq = max_freq

        # Position encoding
        if use_fourier:
            self.pos_encoder = FourierEncoding(
                input_dim=3,
                num_frequencies=num_frequencies,
                max_freq=max_freq,
                include_input=True,
            )
            input_dim = self.pos_encoder.output_dim
        else:
            self.pos_encoder = None
            input_dim = 3

        # Build MLP
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = num_heads if i == num_layers - 1 else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))

            if i < num_layers - 1:
                layers.append(nn.GELU())

        self.mlp = nn.Sequential(*layers)

        # Initialize output layer to small values
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=0.01)

    def _compute_chunk_fn(
        self,
        query_pos: torch.Tensor,
        key_pos_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute position bias for a chunk of keys.
        This function will be wrapped with checkpoint.

        Args:
            query_pos: [B, Q, 3] all query positions
            key_pos_chunk: [B, chunk_n, 3] chunk of key positions

        Returns:
            pos_bias_chunk: [B, num_heads, Q, chunk_n]
        """
        # Compute relative positions: [B, Q, chunk_n, 3]
        rel_pos = query_pos.unsqueeze(2) - key_pos_chunk.unsqueeze(1)

        # Encode positions
        if self.pos_encoder is not None:
            rel_encoded = self.pos_encoder(rel_pos)
        else:
            rel_encoded = rel_pos

        # MLP forward: [B, Q, chunk_n, num_heads]
        pos_bias = self.mlp(rel_encoded)

        # Transpose to [B, num_heads, Q, chunk_n]
        pos_bias = pos_bias.permute(0, 3, 1, 2)

        return pos_bias

    def forward(
        self,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute position bias with gradient checkpointing.

        This is MATHEMATICALLY EQUIVALENT to the original MLP approach,
        but uses much less memory by not storing intermediate activations.

        Args:
            query_pos: [B, Q, 3] query 3D positions
            key_pos: [B, N, 3] key 3D positions

        Returns:
            pos_bias: [B, num_heads, Q, N] position bias
        """
        from torch.utils.checkpoint import checkpoint

        B, Q, _ = query_pos.shape
        N = key_pos.shape[1]
        H = self.num_heads

        # Pre-allocate output tensor to avoid cat() holding references
        pos_bias = query_pos.new_zeros(B, H, Q, N)

        # Chunked computation with gradient checkpointing
        for n_start in range(0, N, self.chunk_size):
            n_end = min(n_start + self.chunk_size, N)
            key_chunk = key_pos[:, n_start:n_end, :]  # [B, chunk_n, 3]

            if self.training:
                # Use checkpoint during training to save memory
                # use_reentrant=False is recommended for newer PyTorch versions
                chunk_bias = checkpoint(
                    self._compute_chunk_fn,
                    query_pos,
                    key_chunk,
                    use_reentrant=False,
                )
            else:
                # No checkpoint during inference (faster)
                chunk_bias = self._compute_chunk_fn(query_pos, key_chunk)

            # Write directly to output (no list append + cat)
            pos_bias[:, :, :, n_start:n_end] = chunk_bias

        return pos_bias

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, chunk_size={self.chunk_size}, "
            f"use_fourier={self.use_fourier}, checkpoint=True"
        )
