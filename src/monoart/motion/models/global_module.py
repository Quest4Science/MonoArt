"""
Global Feature Module for Articulated Object Segmentation.

Based on VAE features (8D) for:
1. Global feature extraction via enhanced pooling (mean + max + std)
2. Object category prediction (auxiliary supervision)
3. Dynamic Query generation (Position + Content)

Architecture:
    VAE [B, N, 8] -> Enhanced Pool (mean+max+std) -> [B, 24]
                            |
                    Global Encoder -> global_emb [B, d_global]
                            |
        +-------------------+-------------------+
        |                   |                   |
    Category Head    Position Generator   Content Generator
        |                   |                   |
    [B, 46]            [B, Q, 3]           [B, Q, 448]

Enhanced Pooling:
    - Mean: captures average shape characteristics
    - Max: captures shape boundaries/extremes
    - Std: captures shape variation/complexity
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class GlobalFeatureModule(nn.Module):
    """
    Global feature extraction + Category prediction + Query generation.

    Uses VAE features (8D) and optionally semantic-reasoner features (448D) for:
    - VAE: global shape information
    - Semantic reasoner: part composition information (what types of parts exist)

    Both features are pooled (mean + max + std) and fused for category prediction.
    """

    def __init__(
        self,
        vae_dim: int = 8,
        d_global: int = 64,
        d_content: int = 448,
        num_categories: int = 7,
        num_queries: int = 100,
        use_learnable_fallback: bool = True,
        dropout: float = 0.1,
        # Semantic-reasoner fusion for category classification
        partfield_dim: int = 448,
        use_partfield_for_category: bool = True,
        partfield_proj_dim: int = 32,
    ):
        """
        Args:
            vae_dim: VAE feature dimension (default: 8)
            d_global: Global embedding dimension
            d_content: Content query dimension (should match the reasoner dimension)
            num_categories: Number of object categories
            num_queries: Number of query slots
            use_learnable_fallback: Use learnable fallback for training stability
            dropout: Dropout rate
            partfield_dim: Reasoner feature dimension (default: 448)
            use_partfield_for_category: Use reasoner features for category classification
            partfield_proj_dim: Projected reasoner dimension (default: 32)
        """
        super().__init__()

        self.vae_dim = vae_dim
        self.d_global = d_global
        self.d_content = d_content
        self.num_categories = num_categories
        self.num_queries = num_queries
        self.use_partfield_for_category = use_partfield_for_category

        # Enhanced pooling: mean + max + std = 3x dim
        self.vae_pooled_dim = vae_dim * 3  # 8 * 3 = 24

        # Reasoner projection and pooling
        if use_partfield_for_category:
            self.partfield_proj_dim = partfield_proj_dim
            self.pf_pooled_dim = partfield_proj_dim * 3  # 32 * 3 = 96

            # Project reasoner features to a lower dimension before pooling
            # This reduces computation and focuses on category-relevant features
            self.partfield_proj = nn.Sequential(
                nn.Linear(partfield_dim, partfield_proj_dim),
                nn.LayerNorm(partfield_proj_dim),
                nn.ReLU(inplace=True),
            )

            # Total input to global encoder: VAE (24) + reasoner features (96) = 120
            total_pooled_dim = self.vae_pooled_dim + self.pf_pooled_dim
        else:
            self.partfield_proj = None
            total_pooled_dim = self.vae_pooled_dim

        self.total_pooled_dim = total_pooled_dim

        # 1. Global feature encoder (Fused features -> d_global)
        # Input: 24 (VAE only) or 120 (VAE + reasoner features)
        self.global_encoder = nn.Sequential(
            nn.Linear(total_pooled_dim, 128),  # 120 -> 128
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, d_global),  # 128 -> 64
            nn.LayerNorm(d_global),
            nn.ReLU(inplace=True),
        )

        # 2. Category prediction head (larger capacity for 46 classes)
        self.category_head = nn.Sequential(
            nn.Linear(d_global, 128),  # 64 -> 128
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),  # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_categories),  # 64 -> 46
        )

        # 3. Position query generator (d_global -> Q * 3)
        self.pos_generator = nn.Sequential(
            nn.Linear(d_global, d_global * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_global * 2, num_queries * 3),
        )

        # 4. Content query generator (d_global -> Q * d_content)
        # Use a larger hidden dim to generate high-dim content
        content_hidden = min(d_global * 4, 512)
        self.content_generator = nn.Sequential(
            nn.Linear(d_global, content_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(content_hidden, content_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(content_hidden, num_queries * d_content),
        )

        # 5. Learnable fallback for training stability
        self.use_learnable_fallback = use_learnable_fallback
        if use_learnable_fallback:
            # Initialize with small random values
            self.fallback_pos = nn.Parameter(torch.randn(num_queries, 3) * 0.1)
            self.fallback_content = nn.Parameter(torch.randn(num_queries, d_content) * 0.02)
            # Gate to blend between generated and fallback
            self.fallback_gate = nn.Sequential(
                nn.Linear(d_global, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize position generator output to cover [0, 1] range
        nn.init.xavier_uniform_(self.pos_generator[-1].weight, gain=0.1)

    def forward(
        self,
        vae_features: torch.Tensor,
        partfield_features: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            vae_features: [B, N, 8] VAE point features
            partfield_features: [B, N, 448] reasoner features (optional, for category fusion)
            return_intermediate: If True, return intermediate values for debugging

        Returns:
            Dictionary containing:
            - global_emb: [B, d_global] global embedding
            - category_logits: [B, num_categories] category prediction
            - pos_queries: [B, Q, 3] position queries (normalized to [0, 1])
            - content_queries: [B, Q, d_content] content queries
        """
        B, N, _ = vae_features.shape

        # Enhanced global pooling for VAE: mean + max + std
        # - Mean: average shape characteristics
        # - Max: shape boundaries/extremes
        # - Std: shape variation/complexity
        global_vae_mean = vae_features.mean(dim=1)  # [B, 8]
        global_vae_max = vae_features.max(dim=1)[0]  # [B, 8]
        global_vae_std = vae_features.std(dim=1)  # [B, 8]
        global_vae = torch.cat([global_vae_mean, global_vae_max, global_vae_std], dim=-1)  # [B, 24]

        # Fuse reasoner features if enabled
        if self.use_partfield_for_category and partfield_features is not None:
            # Project reasoner features to a lower dimension: [B, N, 448] -> [B, N, 32]
            pf_proj = self.partfield_proj(partfield_features)  # [B, N, 32]

            # Enhanced pooling for reasoner features
            # This captures "part composition" information:
            # - Mean: average part type in the object
            # - Max: most prominent part features
            # - Std: part diversity (complex objects have higher std)
            global_pf_mean = pf_proj.mean(dim=1)  # [B, 32]
            global_pf_max = pf_proj.max(dim=1)[0]  # [B, 32]
            global_pf_std = pf_proj.std(dim=1)  # [B, 32]
            global_pf = torch.cat([global_pf_mean, global_pf_max, global_pf_std], dim=-1)  # [B, 96]

            # Concatenate VAE and reasoner pooled features
            global_fused = torch.cat([global_vae, global_pf], dim=-1)  # [B, 120]
        else:
            global_fused = global_vae  # [B, 24]
            global_pf = None

        # Encode global features
        global_emb = self.global_encoder(global_fused)  # [B, d_global]

        # Category prediction
        category_logits = self.category_head(global_emb)  # [B, num_categories]

        # Generate position queries
        pos_raw = self.pos_generator(global_emb)  # [B, Q*3]
        pos_queries = pos_raw.view(B, self.num_queries, 3)
        pos_queries = torch.sigmoid(pos_queries)  # Normalize to [0, 1]

        # Generate content queries
        content_raw = self.content_generator(global_emb)  # [B, Q*d_content]
        content_queries = content_raw.view(B, self.num_queries, self.d_content)

        # Apply fallback blending if enabled
        if self.use_learnable_fallback:
            gate = self.fallback_gate(global_emb)  # [B, 1]
            gate = gate.unsqueeze(1)  # [B, 1, 1]

            # Blend with fallback
            fallback_pos = torch.sigmoid(self.fallback_pos)  # [Q, 3]
            fallback_pos = fallback_pos.unsqueeze(0).expand(B, -1, -1)  # [B, Q, 3]
            pos_queries = gate * pos_queries + (1 - gate) * fallback_pos

            fallback_content = self.fallback_content.unsqueeze(0).expand(
                B, -1, -1
            )  # [B, Q, d_content]
            content_queries = gate * content_queries + (1 - gate) * fallback_content

        result = {
            "global_emb": global_emb,
            "category_logits": category_logits,
            "pos_queries": pos_queries,
            "content_queries": content_queries,
        }

        if return_intermediate:
            result["global_vae"] = global_vae
            result["global_fused"] = global_fused
            if global_pf is not None:
                result["global_pf"] = global_pf
            if self.use_learnable_fallback:
                result["fallback_gate"] = gate.squeeze()

        return result
