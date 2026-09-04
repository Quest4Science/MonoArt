"""
Articulated MAFT Model for Part Segmentation and Motion Prediction.

Dual-feature architecture:
- VAE features (8D) -> GlobalFeatureModule -> Category prediction + Query generation
- Semantic-reasoner features (448D) -> MAFT Decoder -> Part segmentation + Motion prediction

Architecture:
    Input:
    ├── Reasoner [B, N, 448] -> memory
    ├── VAE [B, N, 8] -> GlobalFeatureModule -> queries + category
    └── XYZ [B, N, 3] -> PositionEmbedding -> memory_pos

    Processing:
    GlobalFeatureModule -> pos_queries, content_queries, category_logits
    MAFTDecoder(memory, memory_pos, queries) -> refined_queries, refined_positions
    [Optional] SemanticFusionModule -> part_class_logits, refined_queries

    Output Heads:
    ├── MaskHead -> mask_logits [B, Q, N]
    ├── ScoreHead -> scores [B, Q]
    └── MotionHead -> motion_type (4-class), axis_direction, axis_origin, limits

Semantic Fusion (optional, configurable):
    - Part Classification Head: 18-class part classification
    - Motion Prior: Statistical prior from class-motion correlation
    - Semantic Refiner: EASE-style gated CLIP embedding injection
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .global_module import GlobalFeatureModule
from .motion_head import MotionHead

logger = logging.getLogger(__name__)


# Part Geometric Feature Module (for enhanced motion prediction)
try:
    from monoart.motion.models.part_geometric_feature import PartGeometricFeatureModule

    PART_GEOMETRIC_FEATURE_AVAILABLE = True
except ImportError:
    PART_GEOMETRIC_FEATURE_AVAILABLE = False
    logger.warning("PartGeometricFeatureModule is unavailable")

# Semantic Fusion Module (optional)
try:
    from monoart.motion.semantic_fusion import (
        IterativeSemanticFusion,
        IterativeSemanticFusionConfig,
        MotionTypeHeadWithResidual,
        SemanticFusionModule,
    )

    SEMANTIC_FUSION_AVAILABLE = True
except ImportError:
    SEMANTIC_FUSION_AVAILABLE = False
    logger.warning("Semantic fusion modules are unavailable")

# Relative Position Encoding Module (optional)
try:
    from monoart.motion.models.attention_with_rpe import TransformerDecoderLayerWithRPE

    RPE_AVAILABLE = True
except ImportError:
    RPE_AVAILABLE = False
    logger.warning("Relative-position encoding modules are unavailable")


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal position embedding for 3D coordinates."""

    def __init__(self, d_model: int = 448, temperature: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

        # Calculate dimension per coordinate to ensure total is exactly d_model
        # We use ceil division and will pad/truncate the last dimension
        self.dim_per_coord = (d_model + 2) // 3  # Ensure enough dims

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: [B, N, 3] or [N, 3] coordinates

        Returns:
            pos_embed: [B, N, d_model] or [N, d_model]
        """
        # Create frequency bands
        dim_t = torch.arange(self.dim_per_coord, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.dim_per_coord)

        # Encode each coordinate
        pos_embeds = []
        for i in range(3):
            pos = xyz[..., i : i + 1] / dim_t  # [..., dim_per_coord]
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

        result = torch.cat(pos_embeds, dim=-1)  # [..., dim_per_coord * 3]

        # Truncate or pad to exact d_model
        if result.shape[-1] > self.d_model:
            result = result[..., : self.d_model]
        elif result.shape[-1] < self.d_model:
            pad_size = self.d_model - result.shape[-1]
            result = F.pad(result, (0, pad_size), mode="constant", value=0)

        return result


class SafeMultiheadAttention(nn.Module):
    """
    MultiheadAttention wrapper with numerical stability protection.

    Prevents NaN by clamping attention logits before softmax.
    This is a drop-in replacement for nn.MultiheadAttention.

    The key insight: softmax(x) only cares about relative differences.
    softmax([50, 0]) ≈ softmax([100, 50]) ≈ [1, 0]
    But e^100 = inf causes NaN, while e^50 is safe.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        attn_clamp_value: float = 50.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.attn_clamp_value = attn_clamp_value
        self.scale = self.head_dim**-0.5

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Use xavier initialization like nn.MultiheadAttention
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
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: [B, Q, D] if batch_first else [Q, B, D]
            key: [B, S, D] if batch_first else [S, B, D]
            value: [B, S, D] if batch_first else [S, B, D]
            key_padding_mask: [B, S] True = ignore
            need_weights: return attention weights
            attn_mask: additional attention mask

        Returns:
            output: same shape as query
            attn_weights: [B, Q, S] if need_weights else None
        """
        if self.batch_first:
            B, Q, _ = query.shape
            _, S, _ = key.shape
        else:
            Q, B, _ = query.shape
            S, _, _ = key.shape
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Project Q, K, V
        q = self.q_proj(query)  # [B, Q, D]
        k = self.k_proj(key)  # [B, S, D]
        v = self.v_proj(value)  # [B, S, D]

        # Reshape for multi-head attention
        q = q.view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Q, head_dim]
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, head_dim]
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, head_dim]

        # Compute attention scores
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, Q, S]

        # ========== KEY FIX: Clamp before softmax ==========
        attn_logits = attn_logits.clamp(min=-self.attn_clamp_value, max=self.attn_clamp_value)

        # Apply masks
        if key_padding_mask is not None:
            # key_padding_mask: [B, S] -> [B, 1, 1, S]
            attn_logits = attn_logits.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        if attn_mask is not None:
            attn_logits = attn_logits + attn_mask

        # Softmax (safe now because logits are clamped)
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Check for NaN after softmax (shouldn't happen with clamp)
        if torch.isnan(attn_weights).any():
            logger.error("SafeMultiheadAttention produced NaN attention weights")

        # Dropout
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [B, H, Q, head_dim]

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(B, Q, self.embed_dim)  # [B, Q, D]

        # Output projection
        output = self.out_proj(output)

        if not self.batch_first:
            output = output.transpose(0, 1)

        if need_weights:
            # Average over heads
            attn_weights_avg = attn_weights.mean(dim=1)  # [B, Q, S]
            return output, attn_weights_avg

        return output, None


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with numerical stability."""

    def __init__(
        self,
        d_model: int = 448,
        nhead: int = 8,
        dim_feedforward: int = 1792,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention (with numerical stability protection)
        self.self_attn = SafeMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention (with numerical stability protection)
        self.cross_attn = SafeMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, Q, D] query features
            memory: [B, N, D] memory (point) features
            query_pos: [B, Q, D] query position embeddings
            memory_pos: [B, N, D] memory position embeddings
            memory_key_padding_mask: [B, N] mask for padded positions

        Returns:
            query: [B, Q, D] updated query features
        """
        # Self-attention
        q = k = self.with_pos_embed(query, query_pos)
        query2, _ = self.self_attn(q, k, query)
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # Cross-attention
        q = self.with_pos_embed(query, query_pos)
        k = self.with_pos_embed(memory, memory_pos)
        query2, _ = self.cross_attn(q, k, memory, key_padding_mask=memory_key_padding_mask)
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # FFN
        query2 = self.ffn(query)
        query = query + query2
        query = self.norm3(query)

        return query


class ArticulatedMAFT(nn.Module):
    """
    Articulated MAFT for Part Segmentation and Motion Prediction.

    Dual-feature architecture using:
    - VAE (8D) for global/category encoding and query generation
    - Semantic-reasoner features (448D) for part-level segmentation
    """

    def __init__(
        self,
        # Feature dimensions
        partfield_dim: int = 448,
        vae_dim: int = 8,
        d_model: int = 448,
        d_global: int = 64,
        # Architecture
        num_queries: int = 100,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 1792,
        dropout: float = 0.1,
        # Task settings
        num_categories: int = 7,
        num_motion_types: int = 3,
        # Options
        use_learnable_fallback: bool = True,
        use_position_refinement: bool = True,
        # Category classification fusion
        use_partfield_for_category: bool = True,
        partfield_proj_dim: int = 32,
        # Semantic Fusion (new)
        semantic_fusion_config: Optional[Dict[str, Any]] = None,
        # Part Geometric Feature for Motion Head (new in v2)
        part_geometric_config: Optional[Dict[str, Any]] = None,
        # Iterative Semantic Fusion (new in v3)
        iterative_fusion_config: Optional[Dict[str, Any]] = None,
        # Relative Position Encoding (new in v4)
        rpe_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.partfield_dim = partfield_dim
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.use_partfield_for_category = use_partfield_for_category

        # Semantic Fusion configuration
        self.use_semantic_fusion = False
        self.use_motion_prior_head = False
        if semantic_fusion_config is not None and SEMANTIC_FUSION_AVAILABLE:
            self.use_semantic_fusion = semantic_fusion_config.get("enabled", False)
            self.use_motion_prior_head = semantic_fusion_config.get("use_motion_prior_head", False)

        # Part Geometric Feature configuration (new in v2)
        self.use_part_geometric_feat = False
        self.part_geometric_config = part_geometric_config or {}
        if part_geometric_config is not None and PART_GEOMETRIC_FEATURE_AVAILABLE:
            self.use_part_geometric_feat = part_geometric_config.get("enabled", False)

        # Relative Position Encoding configuration (new in v4)
        self.use_rpe = False
        self.rpe_config = rpe_config or {}
        if rpe_config is not None and RPE_AVAILABLE:
            self.use_rpe = rpe_config.get("enabled", False)

        # Feature projection (only if dims don't match)
        if partfield_dim != d_model:
            self.feature_proj = nn.Linear(partfield_dim, d_model)
        else:
            self.feature_proj = nn.Identity()

        # Global feature module (VAE + semantic reasoner -> category + queries)
        self.global_module = GlobalFeatureModule(
            vae_dim=vae_dim,
            d_global=d_global,
            d_content=d_model,
            num_categories=num_categories,
            num_queries=num_queries,
            use_learnable_fallback=use_learnable_fallback,
            dropout=dropout,
            partfield_dim=partfield_dim,
            use_partfield_for_category=use_partfield_for_category,
            partfield_proj_dim=partfield_proj_dim,
        )

        # Position embedding for point coordinates
        self.pos_embed = PositionEmbeddingSine(d_model=d_model)
        self.pos_embed_proj = nn.Linear(d_model, d_model)

        # Transformer decoder layers
        # Use RPE-enabled layers if configured, otherwise use standard layers
        if self.use_rpe and RPE_AVAILABLE:
            logger.info(
                "Using RPE decoder layers (grid_size=%s, num_buckets=%s)",
                self.rpe_config.get("grid_size", 0.05),
                self.rpe_config.get("num_buckets", 24),
            )
            self.decoder_layers = nn.ModuleList(
                [
                    TransformerDecoderLayerWithRPE(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        use_rpe=True,
                        rpe_config=self.rpe_config,
                    )
                    for _ in range(num_decoder_layers)
                ]
            )
        else:
            logger.info("Using standard decoder layers without RPE")
            self.decoder_layers = nn.ModuleList(
                [
                    TransformerDecoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                    )
                    for _ in range(num_decoder_layers)
                ]
            )
        self.decoder_norm = nn.LayerNorm(d_model)

        # Position refinement (predict offset to refine query positions)
        self.use_position_refinement = use_position_refinement
        if use_position_refinement:
            self.pos_refinement = nn.ModuleList(
                [nn.Linear(d_model, 3) for _ in range(num_decoder_layers - 1)]
            )
            for layer in self.pos_refinement:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Output heads
        # 1. Mask head
        self.mask_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        # 2. Score head
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

        # 3. Part Geometric Feature Module (optional, new in v2)
        self.part_geometric_module = None
        d_part_feat = 0
        if self.use_part_geometric_feat and PART_GEOMETRIC_FEATURE_AVAILABLE:
            pos_encoding_dim = self.part_geometric_config.get("pos_encoding_dim", 64)
            std_proj_dim = self.part_geometric_config.get("std_proj_dim", 64)
            self.part_geometric_module = PartGeometricFeatureModule(
                d_model=d_model,
                pos_encoding_dim=pos_encoding_dim,
                std_proj_dim=std_proj_dim,
            )
            d_part_feat = self.part_geometric_module.output_dim

        # 4. Motion head
        self.motion_head = MotionHead(
            d_model=d_model,
            d_global=d_global,
            num_motion_types=num_motion_types,
            dropout=dropout,
            use_part_geometric_feat=self.use_part_geometric_feat,
            d_part_feat=d_part_feat if d_part_feat > 0 else 576,  # default 576 if not computed
        )

        self.d_global = d_global

        # 4. Semantic Fusion Module (optional)
        self.semantic_fusion = None
        self.motion_prior_head = None
        if self.use_semantic_fusion and SEMANTIC_FUSION_AVAILABLE:
            # Build semantic fusion config
            sf_config = {
                "d_model": d_model,
                "enable_part_classification": semantic_fusion_config.get(
                    "enable_part_classification", True
                ),
                "enable_motion_prior": semantic_fusion_config.get("enable_motion_prior", True),
                "enable_semantic_refiner": semantic_fusion_config.get(
                    "enable_semantic_refiner", False
                ),
                "dropout": dropout,
                "clip_embedding_path": semantic_fusion_config.get("clip_embedding_path", None),
            }
            self.semantic_fusion = SemanticFusionModule(config=sf_config)
            logger.info("Initialized %s", self.semantic_fusion)

            # Optional: Motion type head with residual learning
            if self.use_motion_prior_head:
                self.motion_prior_head = MotionTypeHeadWithResidual(
                    d_model=d_model,
                    num_motion_types=num_motion_types,
                    use_adaptive_mix=semantic_fusion_config.get("motion_prior_adaptive_mix", False),
                )
                logger.info("Initialized MotionTypeHeadWithResidual")

        # 5. Iterative Semantic Fusion (optional, new in v3)
        # Wraps SemanticFusionModule to apply at each decoder layer
        self.iterative_fusion = None
        if iterative_fusion_config is not None and SEMANTIC_FUSION_AVAILABLE:
            iter_cfg = IterativeSemanticFusionConfig.from_dict(iterative_fusion_config)
            if iter_cfg.enabled and self.semantic_fusion is not None:
                self.iterative_fusion = IterativeSemanticFusion(
                    config=iter_cfg,
                    semantic_fusion=self.semantic_fusion,
                    num_layers=num_decoder_layers,
                )
                logger.info(
                    "Enabled iterative semantic fusion: %s",
                    self.iterative_fusion.get_gate_info(),
                )

    def forward(
        self,
        partfield_features: torch.Tensor,
        vae_features: torch.Tensor,
        points: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            partfield_features: [B, N, 448] semantic-reasoner features
            vae_features: [B, N, 8] VAE features
            points: [B, N, 3] point coordinates
            return_intermediate: Return intermediate outputs for debugging

        Returns:
            Dictionary containing:
            - category_logits: [B, num_categories]
            - mask_logits: [B, Q, N]
            - scores: [B, Q]
            - query_positions: [B, Q, 3]
            - motion_type_logits: [B, Q, 4] (F=0, P=1, R=2, C=3)
            - axis_direction: [B, Q, 3]
            - axis_origin: [B, Q, 3]
            - revolute_limit: [B, Q, 2] (center, span) for R type
            - prismatic_limit: [B, Q, 2] (center, span) for P type
        """
        B, N, _ = partfield_features.shape

        # 1. Project semantic-reasoner features to d_model
        memory = self.feature_proj(partfield_features)  # [B, N, d_model]

        # 2. Compute position embeddings for points
        pos_embed = self.pos_embed(points)  # [B, N, d_model]
        memory_pos = self.pos_embed_proj(pos_embed)  # [B, N, d_model]

        # 3. Get global features and initial queries from VAE (+ reasoner features for category)
        # If use_partfield_for_category is True, partfield_features will be used
        # for category classification along with VAE features
        global_output = self.global_module(
            vae_features, partfield_features if self.use_partfield_for_category else None
        )
        category_logits = global_output["category_logits"]  # [B, num_categories]
        global_emb = global_output["global_emb"]  # [B, d_global]

        # Initial queries
        query = global_output["content_queries"]  # [B, Q, d_model]
        query_pos_3d = global_output["pos_queries"]  # [B, Q, 3] in [0, 1]

        # Denormalize query positions to match point cloud range
        points_min = points.min(dim=1, keepdim=True)[0]  # [B, 1, 3]
        points_max = points.max(dim=1, keepdim=True)[0]  # [B, 1, 3]
        query_pos_3d = query_pos_3d * (points_max - points_min) + points_min

        # 4. Decoder forward
        intermediate_queries = []
        intermediate_positions = []
        iterative_aux_outputs = []  # Collect auxiliary outputs from iterative fusion

        for i, layer in enumerate(self.decoder_layers):
            # Compute query position embedding
            query_pos_embed = self.pos_embed_proj(self.pos_embed(query_pos_3d))

            # Decoder layer
            # For RPE-enabled layers, pass 3D coordinates for relative position encoding
            if self.use_rpe:
                query = layer(
                    query=query,
                    memory=memory,
                    query_pos=query_pos_embed,
                    memory_pos=memory_pos,
                    query_pos_3d=query_pos_3d,  # 3D coordinates for RPE
                    memory_pos_3d=points,  # Point cloud coordinates for RPE
                )
            else:
                query = layer(
                    query=query,
                    memory=memory,
                    query_pos=query_pos_embed,
                    memory_pos=memory_pos,
                )

            # Position refinement (except last layer)
            if self.use_position_refinement and i < len(self.pos_refinement):
                pos_offset = self.pos_refinement[i](query)  # [B, Q, 3]
                query_pos_3d = query_pos_3d + pos_offset

            # ========== Iterative Semantic Fusion (new in v3) ==========
            # Apply semantic fusion after each decoder layer (if enabled)
            if self.iterative_fusion is not None:
                # Need to recompute query_pos_embed after position refinement
                query_pos_embed = self.pos_embed_proj(self.pos_embed(query_pos_3d))
                query, aux_out = self.iterative_fusion(
                    query=query,
                    query_pos=query_pos_embed,
                    layer_idx=i,
                )
                if aux_out is not None:
                    iterative_aux_outputs.append(aux_out)

            if return_intermediate:
                intermediate_queries.append(query)
                intermediate_positions.append(query_pos_3d.clone())

        # Final normalization
        query = self.decoder_norm(query)

        # 5. Semantic Fusion (optional, after decoder before output heads)
        semantic_output = {}
        query_for_heads = query  # Default: use decoder output directly

        if self.semantic_fusion is not None:
            # Get query position embedding for semantic fusion
            query_pos_embed = self.pos_embed_proj(self.pos_embed(query_pos_3d))

            # Apply semantic fusion
            # If iterative fusion is enabled:
            #   - query has already been updated in the decoder loop
            #   - here we only need class_logits and motion_prior (skip_motion_prior=False)
            #   - query_for_heads uses the already-updated query (don't update again)
            # If iterative fusion is disabled:
            #   - standard behavior: use query_refined from semantic_fusion
            semantic_output = self.semantic_fusion(
                query=query,
                query_pos=query_pos_embed,
                skip_motion_prior=False,  # Need motion_prior for MotionHead
            )

            if self.iterative_fusion is not None:
                # Iterative mode: query already updated in loop, don't update again
                # Only use semantic_output for class_logits and motion_prior
                query_for_heads = query
            else:
                # Standard mode: use refined query for output heads
                query_for_heads = semantic_output.get("query", query)

        # 6. Output heads
        # Mask prediction
        mask_embed = self.mask_embed(query_for_heads)  # [B, Q, d_model]
        mask_logits = torch.bmm(mask_embed, memory.transpose(1, 2))  # [B, Q, N]

        # Score prediction (output logits, sigmoid applied in inference/loss)
        score_logits = self.score_head(query_for_heads).squeeze(-1)  # [B, Q]

        # Part geometric features (optional, new in v2)
        part_feat = None
        if self.part_geometric_module is not None:
            part_feat = self.part_geometric_module(
                points=points,
                partfield=partfield_features,  # Use original reasoner features (not projected)
                mask_logits=mask_logits,
                query_position=query_pos_3d,
            )  # [B, Q, d_part_feat]

        # Motion prediction
        motion_output = self.motion_head(
            content_query=query_for_heads,
            position_query=query_pos_3d,
            global_emb=global_emb,
            part_feat=part_feat,  # Pass part_feat to motion head
        )

        # Motion type: optionally use prior-based head
        motion_type_logits = motion_output["motion_type_logits"]
        if self.motion_prior_head is not None and "class_logits" in semantic_output:
            # Use residual learning head with motion prior
            motion_type_logits = self.motion_prior_head(
                query=query_for_heads,
                class_logits=semantic_output["class_logits"],
            )

        # Collect outputs
        output = {
            # Category
            "category_logits": category_logits,
            # Segmentation
            "mask_logits": mask_logits,
            "scores": score_logits,  # Output logits, apply sigmoid in loss/inference
            "query_positions": query_pos_3d,
            "query_features": query_for_heads,
            # Motion
            "motion_type_logits": motion_type_logits,
            "axis_direction": motion_output["axis_direction"],
            "axis_origin": motion_output["axis_origin"],
            "axis_origin_offset": motion_output["axis_origin_offset"],
            # Motion limits (center-span parameterization)
            "revolute_limit": motion_output["revolute_limit"],
            "prismatic_limit": motion_output["prismatic_limit"],
            # Global
            "global_emb": global_emb,
        }

        # Add semantic fusion outputs
        if "class_logits" in semantic_output:
            output["part_class_logits"] = semantic_output["class_logits"]
            output["part_class_probs"] = semantic_output["class_probs"]
        if "motion_prior" in semantic_output:
            output["motion_prior"] = semantic_output["motion_prior"]
        if "query_original" in semantic_output:
            output["query_before_semantic"] = semantic_output["query_original"]

        # Add iterative fusion auxiliary outputs (for deep supervision)
        if len(iterative_aux_outputs) > 0:
            output["iterative_aux_outputs"] = iterative_aux_outputs

        if return_intermediate:
            output["intermediate_queries"] = intermediate_queries
            output["intermediate_positions"] = intermediate_positions
            output["initial_pos_queries"] = global_output["pos_queries"]

        return output

    def get_predictions(
        self,
        outputs: Dict[str, torch.Tensor],
        score_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert raw outputs to final predictions.

        Args:
            outputs: Output dictionary from forward()
            score_threshold: Score threshold for filtering predictions

        Returns:
            List of prediction dictionaries (one per batch)
        """
        B = outputs["scores"].shape[0]
        predictions = []

        for b in range(B):
            scores = torch.sigmoid(outputs["scores"][b])  # [Q] - apply sigmoid to logits
            valid_mask = scores > score_threshold

            pred = {
                "scores": scores[valid_mask],
                "masks": torch.sigmoid(outputs["mask_logits"][b][valid_mask]),  # [K, N]
                "positions": outputs["query_positions"][b][valid_mask],  # [K, 3]
                "motion_types": outputs["motion_type_logits"][b][valid_mask].argmax(dim=-1),  # [K]
                "motion_type_probs": F.softmax(
                    outputs["motion_type_logits"][b][valid_mask], dim=-1
                ),
                "axis_directions": outputs["axis_direction"][b][valid_mask],  # [K, 3]
                "axis_origins": outputs["axis_origin"][b][valid_mask],  # [K, 3]
                "revolute_limits": outputs["revolute_limit"][b][valid_mask],  # [K, 2]
                "prismatic_limits": outputs["prismatic_limit"][b][valid_mask],  # [K, 2]
            }
            predictions.append(pred)

        return predictions
