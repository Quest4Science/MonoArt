"""
Part Geometric Feature Module for Motion Head.

Computes geometric features for each query's corresponding part region,
including spatial distribution information to help predict axis positions.

Features computed:
- basic_feat: Weighted average of semantic-reasoner features [d_model]
- pos_mean_enc: Fourier-encoded offset from query position to weighted center [pos_encoding_dim]
- pos_std_proj: Linear-projected spatial distribution (std) [std_proj_dim]

Memory-optimized implementation using Var = E[X²] - E[X]² to avoid [B, Q, N, 3] tensors.
"""

import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal position embedding for 3D coordinates (lightweight version for pos_mean)."""

    def __init__(self, d_model: int = 64, temperature: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.dim_per_coord = (d_model + 2) // 3

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: [..., 3] coordinates

        Returns:
            pos_embed: [..., d_model] Fourier embeddings
        """
        device = xyz.device
        dim_t = torch.arange(self.dim_per_coord, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.dim_per_coord)

        pos_embeds = []
        for i in range(3):
            pos = xyz[..., i : i + 1] / dim_t
            pos_sin = pos[..., 0::2].sin()
            pos_cos = pos[..., 1::2].cos()

            if pos_sin.shape[-1] == pos_cos.shape[-1]:
                pos_embed = torch.stack([pos_sin, pos_cos], dim=-1).flatten(-2)
            else:
                pos_embed = torch.cat(
                    [
                        torch.stack([pos_sin[..., :-1], pos_cos], dim=-1).flatten(-2),
                        pos_sin[..., -1:],
                    ],
                    dim=-1,
                )
            pos_embeds.append(pos_embed)

        result = torch.cat(pos_embeds, dim=-1)

        # Truncate or pad to exact d_model
        if result.shape[-1] > self.d_model:
            result = result[..., : self.d_model]
        elif result.shape[-1] < self.d_model:
            pad_size = self.d_model - result.shape[-1]
            result = torch.nn.functional.pad(result, (0, pad_size), mode="constant", value=0)

        return result


class PartGeometricFeatureModule(nn.Module):
    """
    Computes geometric features for each query's corresponding part region.

    For each query, uses the predicted segmentation mask to aggregate:
    1. basic_feat: Weighted average of semantic-reasoner features (part shape info)
    2. pos_mean: Weighted center offset from query position (spatial bias)
    3. pos_std: Spatial distribution range (boundary distance hint)

    Memory-optimized: Uses Var = E[X²] - E[X]² to avoid [B, Q, N, 3] large tensors.

    Output dimension: d_model + pos_encoding_dim + std_proj_dim = 448 + 64 + 64 = 576
    """

    def __init__(
        self,
        d_model: int = 448,
        pos_encoding_dim: int = 64,
        std_proj_dim: int = 64,
    ):
        """
        Args:
            d_model: Reasoner feature dimension (basic_feat output dim)
            pos_encoding_dim: Fourier encoding dimension for pos_mean
            std_proj_dim: Linear projection dimension for pos_std
        """
        super().__init__()

        self.d_model = d_model
        self.pos_encoding_dim = pos_encoding_dim
        self.std_proj_dim = std_proj_dim

        # pos_mean: Fourier encoding (coordinate offsets benefit from multi-frequency encoding)
        self.pos_embed = PositionEmbeddingSine(d_model=pos_encoding_dim)

        # pos_std: Linear projection + LayerNorm (preserve linear scale relationship)
        self.std_proj = nn.Sequential(
            nn.Linear(3, std_proj_dim),
            nn.LayerNorm(std_proj_dim),
        )

        # Output dimension: d_model + pos_encoding_dim + std_proj_dim
        self.output_dim = d_model + pos_encoding_dim + std_proj_dim

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Small initialization for std_proj to start with small influence
        nn.init.xavier_uniform_(self.std_proj[0].weight, gain=0.1)
        nn.init.zeros_(self.std_proj[0].bias)

    def forward(
        self,
        points: torch.Tensor,  # [B, N, 3]
        partfield: torch.Tensor,  # [B, N, d_model]
        mask_logits: torch.Tensor,  # [B, Q, N]
        query_position: torch.Tensor,  # [B, Q, 3]
    ) -> torch.Tensor:
        """
        Compute part geometric features for each query.

        CRITICAL: mask_logits is detached to prevent motion loss from interfering
        with segmentation learning.

        Args:
            points: [B, N, 3] point cloud coordinates
            partfield: [B, N, d_model] semantic-reasoner features per point
            mask_logits: [B, Q, N] segmentation mask logits (will be detached)
            query_position: [B, Q, 3] query positions (part centers)

        Returns:
            part_feat: [B, Q, output_dim] geometric features for motion head
                       Contains: [basic_feat, pos_mean_enc, pos_std_proj]
        """
        # ========== Step 1: Normalize mask (DETACH to prevent gradient interference) ==========
        # Critical: detach() prevents motion loss from distorting segmentation masks
        mask = torch.sigmoid(mask_logits.detach())  # [B, Q, N]
        mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, Q, 1]
        mask_norm = mask / mask_sum  # [B, Q, N] normalized weights

        # ========== Step 2: Weighted center (using einsum to avoid [B,Q,N,3]) ==========
        # mask_norm: [B, Q, N], points: [B, N, 3] -> center: [B, Q, 3]
        center = torch.einsum("bqn,bnd->bqd", mask_norm, points)  # [B, Q, 3]

        # ========== Step 3: Relative position mean ==========
        # pos_mean = weighted center - query position
        # Represents: "the part's points are biased towards +X direction relative to query"
        pos_mean = center - query_position  # [B, Q, 3]

        # ========== Step 4: Variance using Var = E[X²] - E[X]² ==========
        # This avoids creating [B, Q, N, 3] tensor for (points - center)

        # E[X²]: weighted mean of point coordinates squared
        points_sq = points**2  # [B, N, 3]
        weighted_sq = torch.einsum("bqn,bnd->bqd", mask_norm, points_sq)  # [B, Q, 3]

        # E[X]² = center²
        center_sq = center**2  # [B, Q, 3]

        # Var = E[X²] - E[X]², clamp to prevent numerical issues
        pos_var = (weighted_sq - center_sq).clamp(min=1e-6)  # [B, Q, 3]
        pos_std = torch.sqrt(pos_var)  # [B, Q, 3]

        # ========== Step 5: Aggregate reasoner features ==========
        # mask_norm: [B, Q, N], partfield: [B, N, d_model] -> basic_feat: [B, Q, d_model]
        basic_feat = torch.einsum("bqn,bnd->bqd", mask_norm, partfield)  # [B, Q, d_model]

        # ========== Step 6: Encode/Project ==========
        # pos_mean: Fourier encoding (coordinate offsets need multi-frequency)
        pos_mean_enc = self.pos_embed(pos_mean)  # [B, Q, pos_encoding_dim]

        # pos_std: Linear projection (scale values, preserve linear relationship)
        pos_std_proj = self.std_proj(pos_std)  # [B, Q, std_proj_dim]

        # ========== Step 7: Concatenate final features ==========
        part_feat = torch.cat(
            [
                basic_feat,  # [d_model] = 448
                pos_mean_enc,  # [pos_encoding_dim] = 64
                pos_std_proj,  # [std_proj_dim] = 64
            ],
            dim=-1,
        )  # [576]

        return part_feat

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"pos_encoding_dim={self.pos_encoding_dim}, "
            f"std_proj_dim={self.std_proj_dim}, "
            f"output_dim={self.output_dim}"
        )
