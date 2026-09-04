"""Sampling operations for the Part-Aware Semantic Reasoner."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_triplane_features(
    feature_triplane: torch.Tensor,
    normalized_positions: torch.Tensor,
) -> torch.Tensor:
    """Sample and sum XY, YZ, and XZ planes at 3D point positions.

    Args:
        feature_triplane: Tensor shaped ``[B, 3, C, H, W]``.
        normalized_positions: Tensor shaped ``[B, N, 3]`` in grid coordinates.

    Returns:
        A tensor shaped ``[B, N, C]``.
    """
    xy_plane, yz_plane, xz_plane = torch.unbind(feature_triplane, dim=1)

    def sample(plane: torch.Tensor, axes: tuple[int, int]) -> torch.Tensor:
        grid = normalized_positions[..., list(axes)].unsqueeze(1)
        return F.grid_sample(
            plane,
            grid,
            padding_mode="border",
            align_corners=True,
        )

    sampled = sample(xy_plane, (0, 1)) + sample(yz_plane, (1, 2)) + sample(xz_plane, (0, 2))
    return sampled.squeeze(2).permute(0, 2, 1).contiguous()
