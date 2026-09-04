"""
Semantic Fusion Module for Articulated Object Part Understanding.

This module provides semantic-guided enhancement for motion prediction,
leveraging the strong correlation between part categories and motion types.

Main Components:
- PartClassificationHead: Classifies queries into 18 part classes
- MotionPriorLookup: Soft lookup of motion priors from classification
- MotionTypeHeadWithResidual: Motion prediction with prior + residual learning
- SemanticGatedQueryRefiner: EASE-style gated semantic injection
- SemanticFusionModule: Unified module integrating all components

Usage:
    from monoart.motion.semantic_fusion import SemanticFusionModule

    # Create module with configurable components
    semantic_fusion = SemanticFusionModule(
        d_model=448,
        enable_part_classification=True,
        enable_motion_prior=True,
        enable_semantic_refiner=False,  # Can be toggled for ablation
    )

    # Forward pass (after decoder, before output heads)
    outputs = semantic_fusion(
        query=query_features,          # [B, Q, D]
        query_pos=query_positions,     # [B, Q, D]
    )

    # outputs['query']: refined query features
    # outputs['class_logits']: part classification logits
    # outputs['motion_prior']: motion type prior probabilities
"""

# Config
from .config import (
    CLASS_CERTAINTY,
    CLASS_COUNTS,
    CLASS_NAMES,
    CLASS_TO_IDX,
    CLASS_WEIGHTS,
    DEFAULT_CLIP_EMBEDDING_PATH,
    IDX_TO_CLASS,
    MERGE_RULES,
    MOTION_PRIOR_TABLE,
    NUM_PART_CLASSES,
    REVERSE_MERGE_MAP,
    compute_class_weights,
    get_merged_class,
    get_merged_class_idx,
)

# Iterative Semantic Fusion
from .iterative_fusion import (
    IterativeSemanticFusion,
    IterativeSemanticFusionConfig,
    compute_iterative_deep_supervision_loss,
)

# Motion Prior
from .motion_prior import (
    MotionPriorLookup,
    MotionTypeHeadWithResidual,
)

# Part Classification
from .part_classification import (
    PartClassificationHead,
    PartClassificationLoss,
)

# Unified Module
from .semantic_fusion_module import (
    SemanticFusionConfig,
    SemanticFusionModule,
)

# Semantic Refiner
from .semantic_refiner import (
    SemanticGatedQueryRefiner,
)

__all__ = [
    # Config
    "NUM_PART_CLASSES",
    "CLASS_NAMES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "MOTION_PRIOR_TABLE",
    "CLASS_CERTAINTY",
    "CLASS_WEIGHTS",
    "CLASS_COUNTS",
    "MERGE_RULES",
    "REVERSE_MERGE_MAP",
    "get_merged_class",
    "get_merged_class_idx",
    "compute_class_weights",
    "DEFAULT_CLIP_EMBEDDING_PATH",
    # Part Classification
    "PartClassificationHead",
    "PartClassificationLoss",
    # Motion Prior
    "MotionPriorLookup",
    "MotionTypeHeadWithResidual",
    # Semantic Refiner
    "SemanticGatedQueryRefiner",
    # Unified Module
    "SemanticFusionConfig",
    "SemanticFusionModule",
    # Iterative Semantic Fusion
    "IterativeSemanticFusionConfig",
    "IterativeSemanticFusion",
    "compute_iterative_deep_supervision_loss",
]
