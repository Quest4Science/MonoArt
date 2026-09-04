"""
Semantic Fusion Module - Unified Integration.

Integrates all semantic fusion components:
1. Part Classification Head (18 classes)
2. Motion Prior Lookup + Residual Learning
3. Semantic-Gated Query Refiner (EASE-style)

All components can be individually enabled/disabled via configuration
for easy ablation experiments.

Usage:
    # Create module
    semantic_fusion = SemanticFusionModule(
        d_model=448,
        enable_part_classification=True,
        enable_motion_prior=True,
        enable_semantic_refiner=False,  # Disabled for ablation
    )

    # Forward pass (after decoder, before output heads)
    outputs = semantic_fusion(
        query=query_features,          # [B, Q, D]
        query_pos=query_positions,     # [B, Q, D] or [B, Q, 3]
    )

    # Outputs:
    # - query: [B, Q, D] (possibly refined by semantic refiner)
    # - class_logits: [B, Q, 18] (if part classification enabled)
    # - motion_prior: [B, Q, 4] (if motion prior enabled)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEFAULT_CLIP_EMBEDDING_PATH, NUM_PART_CLASSES
from .motion_prior import MotionPriorLookup, MotionTypeHeadWithResidual
from .part_classification import PartClassificationHead
from .semantic_refiner import SemanticGatedQueryRefiner


@dataclass
class SemanticFusionConfig:
    """Configuration for Semantic Fusion Module."""

    # Enable flags (for ablation)
    enable_part_classification: bool = True
    enable_motion_prior: bool = True
    enable_semantic_refiner: bool = False  # Start disabled, enable after validation

    # Model dimensions
    d_model: int = 448
    num_part_classes: int = NUM_PART_CLASSES
    num_motion_types: int = 4

    # Part classification settings
    part_class_use_mlp: bool = False
    part_class_hidden_dim: Optional[int] = None

    # Motion prior settings
    motion_prior_use_adaptive_mix: bool = False
    motion_prior_hidden_dim: Optional[int] = None

    # Semantic refiner settings
    clip_dim: int = 512
    clip_embedding_path: Optional[str] = None
    refiner_use_position_in_gate: bool = True
    refiner_hidden_dim: Optional[int] = None

    # Dropout
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SemanticFusionConfig":
        """Create config from dictionary (e.g., from YAML)."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


class SemanticFusionModule(nn.Module):
    """
    Unified Semantic Fusion Module.

    Integrates part classification, motion prior, and semantic refinement
    with configurable enable/disable switches for each component.

    Args:
        config: SemanticFusionConfig or dict with configuration
        **kwargs: Override config values
    """

    def __init__(
        self,
        config: Optional[SemanticFusionConfig] = None,
        **kwargs,
    ):
        super().__init__()

        # Handle config
        if config is None:
            config = SemanticFusionConfig(**kwargs)
        elif isinstance(config, dict):
            config = SemanticFusionConfig.from_dict(config)

        self.config = config

        # Initialize components based on config
        self._init_components()

    def _init_components(self):
        """Initialize enabled components."""
        cfg = self.config

        # 1. Part Classification Head
        self.part_class_head = None
        if cfg.enable_part_classification:
            self.part_class_head = PartClassificationHead(
                d_model=cfg.d_model,
                num_classes=cfg.num_part_classes,
                hidden_dim=cfg.part_class_hidden_dim,
                use_mlp=cfg.part_class_use_mlp,
                dropout=cfg.dropout,
            )

        # 2. Motion Prior Lookup
        self.motion_prior_lookup = None
        if cfg.enable_motion_prior:
            self.motion_prior_lookup = MotionPriorLookup()

        # 3. Semantic-Gated Query Refiner
        self.semantic_refiner = None
        if cfg.enable_semantic_refiner:
            self.semantic_refiner = SemanticGatedQueryRefiner(
                d_model=cfg.d_model,
                num_classes=cfg.num_part_classes,
                clip_dim=cfg.clip_dim,
                embedding_path=cfg.clip_embedding_path or DEFAULT_CLIP_EMBEDDING_PATH,
                use_position_in_gate=cfg.refiner_use_position_in_gate,
                hidden_dim=cfg.refiner_hidden_dim,
                dropout=cfg.dropout,
            )

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        skip_motion_prior: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            query: [B, N_query, D_model] query features (Q_content)
            query_pos: [B, N_query, D_model] or [B, N_query, 3] position features
            skip_motion_prior: If True, skip MotionPriorLookup (used in iterative fusion
                               where motion prior is only needed at the final layer)

        Returns:
            Dictionary containing:
            - query: [B, N_query, D_model] (possibly refined)
            - class_logits: [B, N_query, 18] (if part classification enabled)
            - class_probs: [B, N_query, 18] (if part classification enabled)
            - motion_prior: [B, N_query, 4] (if motion prior enabled and not skipped)
        """
        outputs = {"query": query}
        class_logits = None
        class_probs = None

        # 1. Part Classification
        if self.part_class_head is not None:
            class_logits = self.part_class_head(query)
            class_probs = F.softmax(class_logits, dim=-1)
            outputs["class_logits"] = class_logits
            outputs["class_probs"] = class_probs

        # 2. Motion Prior Lookup (can be skipped for iterative fusion intermediate layers)
        if (
            self.motion_prior_lookup is not None
            and class_logits is not None
            and not skip_motion_prior
        ):
            motion_prior = self.motion_prior_lookup.forward_with_probs(class_probs)
            outputs["motion_prior"] = motion_prior

        # 3. Semantic Refinement
        if self.semantic_refiner is not None and class_logits is not None:
            # Handle different query_pos formats
            if query_pos.shape[-1] == 3:
                # Preserve the released checkpoint behavior when only raw XYZ is supplied.
                pos_features = query
            else:
                pos_features = query_pos

            query_refined = self.semantic_refiner(
                q_content=query,
                q_position=pos_features,
                class_logits=class_logits,
            )
            outputs["query"] = query_refined
            outputs["query_original"] = query

        return outputs

    def get_motion_type_head(self) -> Optional[MotionTypeHeadWithResidual]:
        """
        Create a MotionTypeHeadWithResidual that uses this module's prior lookup.

        This is a factory method to create a motion head that works with
        the semantic fusion module.

        Returns:
            MotionTypeHeadWithResidual or None if motion prior is disabled
        """
        if not self.config.enable_motion_prior:
            return None

        return MotionTypeHeadWithResidual(
            d_model=self.config.d_model,
            num_motion_types=self.config.num_motion_types,
            hidden_dim=self.config.motion_prior_hidden_dim,
            use_adaptive_mix=self.config.motion_prior_use_adaptive_mix,
        )

    def get_enabled_components(self) -> Dict[str, bool]:
        """Get status of all components."""
        return {
            "part_classification": self.part_class_head is not None,
            "motion_prior": self.motion_prior_lookup is not None,
            "semantic_refiner": self.semantic_refiner is not None,
        }

    def __repr__(self) -> str:
        enabled = self.get_enabled_components()
        components = [k for k, v in enabled.items() if v]
        return f"SemanticFusionModule(enabled=[{', '.join(components)}])"


# =============================================================================
# Test
# =============================================================================
