"""
Parent-Child Relation Prediction Module.

This module predicts the kinematic tree structure of articulated objects,
specifically the parent-child relationships between parts.

Design Document: parent_child_design.md

Two-Stage Training Strategy:
- Stage 1: Train segmentation + motion prediction (must complete first)
- Stage 2: Train parent-child prediction with frozen Stage 1 modules

Main Components:
- ParentPredictionHead: Bilinear attention + semantic affinity
- ParentPredictionLoss: Cross-entropy with masked loss for unmatched parents
- resolve_cycles: Post-processing to ensure tree structure (no cycles)
- build_kinematic_tree: Build tree structure from predictions

Usage:
    from monoart.motion.relation import (
        ParentPredictionHead,
        ParentPredictionLoss,
        resolve_cycles,
        build_kinematic_tree,
        ParentPredictionConfig,
    )

    # Stage 2 training
    parent_head = ParentPredictionHead(d_model=448)
    parent_loss = ParentPredictionLoss(weight=1.0)

    # Forward
    parent_logits = parent_head(q_content, q_position, class_probs)

    # Loss (requires Hungarian matcher results)
    losses = parent_loss(parent_logits, matched_indices, gt_parent_info, gt_link_ids)

    # Inference with cycle resolution
    parent_pred = resolve_cycles(parent_logits[0])
    tree = build_kinematic_tree(parent_pred)
"""

from .config import (
    DEFAULT_AFFINITY,
    NUM_PART_CLASSES,
    ParentPredictionConfig,
    compute_parent_affinity_matrix,
    get_affinity_matrix,
)
from .cycle_resolution import (
    detect_cycles,
    greedy_tree_from_scores,
    resolve_cycles,
    resolve_cycles_batch,
)
from .parent_head import ParentPredictionHead
from .parent_loss import (
    ParentPredictionLoss,
    build_parent_targets,
)
from .tree_utils import (
    build_kinematic_tree,
    compute_tree_metrics,
    get_subtree_nodes,
    get_tree_depth,
    tree_to_string,
    validate_tree,
)

__all__ = [
    # Config
    "ParentPredictionConfig",
    "DEFAULT_AFFINITY",
    "NUM_PART_CLASSES",
    "compute_parent_affinity_matrix",
    "get_affinity_matrix",
    # Head
    "ParentPredictionHead",
    # Loss
    "ParentPredictionLoss",
    "build_parent_targets",
    # Cycle resolution
    "resolve_cycles",
    "resolve_cycles_batch",
    "detect_cycles",
    "greedy_tree_from_scores",
    # Tree utils
    "build_kinematic_tree",
    "get_tree_depth",
    "get_subtree_nodes",
    "compute_tree_metrics",
    "tree_to_string",
    "validate_tree",
]
