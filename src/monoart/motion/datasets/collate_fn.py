"""
Custom collate function for ArticulatedDataset.

Handles variable-length motion labels across different samples
by keeping them as lists instead of stacking into tensors.
"""

from typing import Any, Dict, List

import torch


def articulated_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for ArticulatedDataset.

    Fixed-size tensors are stacked into batches.
    Variable-size tensors (motion labels) are kept as lists.

    Args:
        batch: List of sample dictionaries from ArticulatedDataset

    Returns:
        Batched dictionary with:
        - Fixed-size: stacked tensors [B, ...]
        - Variable-size: lists of tensors [tensor, tensor, ...]
        - Metadata: lists of values
    """
    if not batch:
        return {}

    # Keys to stack (fixed-size across samples)
    stack_keys = [
        "points",  # [B, N, 3]
        "normals",  # [B, N, 3]
        "group_ids",  # [B, N]
        "face_ids",  # [B, N] - per-point face IDs from input PLY
        "partfield_features",  # [B, N, 448]
        "vae_features",  # [B, N, 8]
    ]

    # Keys to keep as lists (variable-size)
    list_keys = [
        "gt_motion_types",  # List of [G_i] tensors (0=F, 1=P, 2=R, 3=C)
        "gt_axis_directions",  # List of [G_i, 3] tensors
        "gt_axis_positions",  # List of [G_i, 3] tensors
        "gt_motion_limits",  # List of [G_i, 2] tensors (min, max limits)
        "gt_link_ids",  # List of [G_i] tensors
        "gt_is_movable",  # List of [G_i] tensors
        "gt_part_class_labels",  # List of [G_i] tensors (0-17 for 18 classes)
        "gt_part_names",  # List of List[str]
        "gt_parent_info",  # List of Dict[int, int] (link_id -> parent_link_id)
        "gt_projected_origins",  # List of [G_i, 3] tensors (part center projection on axis, for anchor loss)
    ]

    # Metadata keys (strings, ints)
    meta_keys = [
        "anno_id",
        "category_idx",
        "category_name",
        "num_parts",
    ]

    result = {}

    # Stack fixed-size tensors
    for key in stack_keys:
        if key in batch[0]:
            result[key] = torch.stack([sample[key] for sample in batch], dim=0)

    # Keep variable-size tensors as lists
    for key in list_keys:
        if key in batch[0]:
            result[key] = [sample[key] for sample in batch]

    # Collect metadata as lists
    for key in meta_keys:
        if key in batch[0]:
            result[key] = [sample[key] for sample in batch]

    # Add batch size info
    result["batch_size"] = len(batch)

    # Compute batch_offsets for MAFT decoder (cumulative point counts)
    n_points = result["points"].shape[1]
    batch_offsets = torch.tensor([i * n_points for i in range(len(batch) + 1)], dtype=torch.long)
    result["batch_offsets"] = batch_offsets

    # Stack category indices as tensor for classification
    if "category_idx" in result:
        result["category_idx_tensor"] = torch.tensor(result["category_idx"], dtype=torch.long)

    return result
