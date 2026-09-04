# Copyright (c) 2023-2024, Zexin He
#
# Portions of this module derive from OpenLRM and are licensed under the
# Apache License, Version 2.0. See THIRD_PARTY.md for attribution.

"""Transformer refinement of rasterized three-plane point features."""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    """Pre-normalized self-attention transformer block."""

    def __init__(
        self,
        inner_dim: int,
        num_heads: int,
        eps: float,
        attn_drop: float = 0.0,
        attn_bias: bool = False,
        mlp_ratio: float = 4.0,
        mlp_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(inner_dim, eps=eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=attn_bias,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        x = (
            x
            + self.self_attn(
                normalized,
                normalized,
                normalized,
                need_weights=False,
            )[0]
        )
        return x + self.mlp(self.norm2(x))


class TransformerDecoder(nn.Module):
    """Checkpoint-compatible stack of basic transformer blocks."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        inner_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BasicBlock(inner_dim=inner_dim, num_heads=num_heads, eps=eps)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(inner_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TriplaneTransformer(nn.Module):
    """Refine triplanes through residual transformer and pointwise MLP branches."""

    def __init__(
        self,
        input_dim: int,
        transformer_dim: int,
        transformer_layers: int,
        transformer_heads: int,
        triplane_low_res: int,
        triplane_high_res: int,
        triplane_dim: int,
    ) -> None:
        super().__init__()
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim

        token_count = 3 * triplane_low_res**2
        self.pos_embed = nn.Parameter(
            torch.randn(1, token_count, transformer_dim) * (1.0 / transformer_dim) ** 0.5
        )
        self.transformer = TransformerDecoder(
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            inner_dim=transformer_dim,
        )
        self.downsampler = nn.Sequential(
            nn.Conv2d(input_dim, transformer_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(transformer_dim, transformer_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.upsampler = nn.ConvTranspose2d(
            transformer_dim,
            triplane_dim,
            kernel_size=4,
            stride=4,
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, triplane_dim),
            nn.ReLU(),
            nn.Linear(triplane_dim, triplane_dim),
        )

    def _downsample(self, triplanes: torch.Tensor) -> torch.Tensor:
        batch = triplanes.shape[0]
        height = width = self.triplane_high_res
        x = triplanes.view(batch, 3, -1, height, width)
        x = torch.einsum("nidhw->indhw", x).contiguous()
        x = self.downsampler(x.view(3 * batch, *x.shape[2:]))
        x = x.view(3, batch, *x.shape[-3:])
        return torch.einsum("indhw->nidhw", x).contiguous()

    def _run_transformer(self, triplanes: torch.Tensor) -> torch.Tensor:
        batch = triplanes.shape[0]
        tokens = torch.einsum("nidhw->nihwd", triplanes).reshape(batch, self.pos_embed.shape[1], -1)
        return self.transformer(self.pos_embed.repeat(batch, 1, 1) + tokens)

    def _upsample(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        height = width = self.triplane_low_res
        x = tokens.view(batch, 3, height, width, -1)
        x = torch.einsum("nihwd->indhw", x).contiguous()
        x = self.upsampler(x.view(3 * batch, *x.shape[2:]))
        x = x.view(3, batch, *x.shape[-3:])
        return torch.einsum("indhw->nidhw", x).contiguous()

    def forward(self, triplanes: torch.Tensor) -> torch.Tensor:
        residual = self._upsample(self._run_transformer(self._downsample(triplanes)))
        projected = self.mlp(triplanes.permute(0, 1, 3, 4, 2).contiguous())
        projected = projected.permute(0, 1, 4, 2, 3).contiguous()
        return projected + residual
