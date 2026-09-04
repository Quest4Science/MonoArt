#!/usr/bin/env python3
"""
Inference script for Articulated Object Part Segmentation and Motion Prediction.

Outputs:
- segmentation.npz: Per-point segmentation labels and scores
- motion.json: Motion parameters in the same format as training input
- visualization.ply: Colored point cloud for 3D visualization

Usage:
    python -m monoart.motion.scripts.inference --config configs/inference.yaml
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from monoart.checkpoints import load_torch, motion_payload
from monoart.motion.datasets.motion_parser import (
    MOTION_TYPE_NAMES,
    get_category_names,
)
from monoart.motion.datasets.world_transform import inverse_transform_axis
from monoart.motion.models.articulated_maft import ArticulatedMAFT

# Stage 2: Parent prediction imports
try:
    from monoart.motion.relation import (
        ParentPredictionHead,
        build_kinematic_tree,
        resolve_cycles,
        tree_to_string,
    )

    PARENT_PREDICTION_AVAILABLE = True
except ImportError:
    PARENT_PREDICTION_AVAILABLE = False
    print("Warning: Parent prediction module not available. Stage 2 features disabled.")

# Semantic Fusion: Part class names (18 classes)
try:
    from monoart.motion.semantic_fusion import CLASS_NAMES as PART_CLASS_NAMES
    from monoart.motion.semantic_fusion import IDX_TO_CLASS

    PART_CLASS_AVAILABLE = True
except ImportError:
    PART_CLASS_AVAILABLE = False
    PART_CLASS_NAMES = None
    IDX_TO_CLASS = None
    print("Warning: Semantic fusion module not available. Part class names disabled.")


# Motion type mapping for output JSON (4-class system)
MOTION_TYPE_TO_LETTER = {
    0: "F",  # Fixed
    1: "P",  # Prismatic (Translation)
    2: "R",  # Revolute (Rotation with limits)
    3: "C",  # Continuous (Unlimited rotation)
}

# π for limit denormalization
PI = math.pi


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> torch.nn.Module:
    """Create ArticulatedMAFT model."""
    model_config = config["model"]

    # Build semantic_fusion_config if enabled
    semantic_fusion_config = None
    sf_config = config.get("semantic_fusion", {})
    if sf_config.get("enabled", False):
        semantic_fusion_config = {
            "enabled": True,
            "num_classes": 18,  # Fixed 18 part classes
            "enable_part_classification": sf_config.get("enable_part_classification", True),
            "part_class_use_mlp": sf_config.get("part_class_use_mlp", True),
            "enable_motion_prior": sf_config.get("enable_motion_prior", True),
            "use_motion_prior_head": sf_config.get("use_motion_prior_head", True),
            "motion_prior_adaptive_mix": sf_config.get("motion_prior_adaptive_mix", False),
            "enable_semantic_refiner": sf_config.get("enable_semantic_refiner", True),
            "clip_embedding_path": sf_config.get("clip_embedding_path", None),
        }

    # Part Geometric Feature config (for enhanced motion origin prediction)
    part_geometric_config = config.get("part_geometric", None)
    if part_geometric_config is not None:
        print("Part Geometric Feature config:")
        print(f"  - Enabled: {part_geometric_config.get('enabled', False)}")

    # Iterative Semantic Fusion config
    iterative_fusion_config = config.get("iterative_semantic_fusion", None)
    if iterative_fusion_config is not None:
        print("Iterative Semantic Fusion config:")
        print(f"  - Enabled: {iterative_fusion_config.get('enabled', False)}")
        print(f"  - Mode: {iterative_fusion_config.get('mode', 'all_layers')}")
        print(f"  - Gate strategy: {iterative_fusion_config.get('gate_strategy', 'increasing')}")

    # RPE (Relative Position Encoding) config
    rpe_config = config.get("rpe", None)
    if rpe_config is not None:
        rpe_type = rpe_config.get("type", "table")
        print("RPE (Relative Position Encoding) config:")
        print(f"  - Enabled: {rpe_config.get('enabled', False)}")
        print(f"  - Type: {rpe_type}")
        if rpe_type == "table":
            print(f"  - Grid size: {rpe_config.get('grid_size', 0.05)}")
            print(f"  - Num buckets: {rpe_config.get('num_buckets', 24)}")
        elif rpe_type == "mlp":
            print(f"  - Hidden dim: {rpe_config.get('mlp_hidden_dim', 64)}")
            print(f"  - Num layers: {rpe_config.get('mlp_num_layers', 2)}")
            print(f"  - Use Fourier: {rpe_config.get('mlp_use_fourier', True)}")

    model = ArticulatedMAFT(
        partfield_dim=model_config.get("partfield_dim", 448),
        vae_dim=model_config.get("vae_dim", 8),
        d_model=model_config.get("d_model", 448),
        d_global=model_config.get("d_global", 64),
        num_queries=model_config.get("num_queries", 100),
        num_decoder_layers=model_config.get("num_decoder_layers", 6),
        nhead=model_config.get("nhead", 8),
        dim_feedforward=model_config.get("dim_feedforward", 1792),
        dropout=model_config.get("dropout", 0.1),
        num_categories=model_config.get("num_categories", 46),  # 46 categories
        num_motion_types=model_config.get("num_motion_types", 4),  # 4 classes: F, P, R, C
        use_learnable_fallback=model_config.get("use_learnable_fallback", True),
        use_position_refinement=model_config.get("use_position_refinement", True),
        semantic_fusion_config=semantic_fusion_config,
        part_geometric_config=part_geometric_config,
        iterative_fusion_config=iterative_fusion_config,
        rpe_config=rpe_config,
    )

    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    """Load model checkpoint (Stage 1 only, backward compatible)."""
    checkpoint = motion_payload(load_torch(checkpoint_path, map_location=device))

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", -1)
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights")

    return model


def load_checkpoint_with_parent_head(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    config: dict,
):
    """
    Load checkpoint with automatic detection of ParentPredictionHead.

    Returns:
        model: Loaded ArticulatedMAFT model
        parent_head: ParentPredictionHead if available, else None
        has_parent_prediction: bool
    """
    checkpoint = motion_payload(load_torch(checkpoint_path, map_location=device))

    # Load main model
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", -1)
        print(f"Loaded model checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights")

    # Check for parent_head_state_dict (Stage 2)
    parent_head = None
    has_parent_prediction = False

    if "parent_head_state_dict" in checkpoint and PARENT_PREDICTION_AVAILABLE:
        pp_config = config.get("parent_prediction", {})
        head_config = pp_config.get("head", {})

        parent_head = ParentPredictionHead(
            d_model=config["model"].get("d_model", 448),
            use_position=head_config.get("use_position", True),
            use_semantic=head_config.get("use_semantic", True),
            lambda_semantic_init=head_config.get("lambda_semantic_init", 0.3),
        )
        parent_head.load_state_dict(checkpoint["parent_head_state_dict"])
        parent_head = parent_head.to(device)
        parent_head.eval()

        has_parent_prediction = True
        print("=" * 60)
        print("STAGE 2 MODE: ParentPredictionHead loaded")
        print(f"  use_position: {head_config.get('use_position', True)}")
        print(f"  use_semantic: {head_config.get('use_semantic', True)}")
        print("=" * 60)
    else:
        if "parent_head_state_dict" in checkpoint and not PARENT_PREDICTION_AVAILABLE:
            print("WARNING: Checkpoint contains parent_head but relation module not available!")
        print("STAGE 1 MODE: No parent prediction (all parents = 'base')")

    return model, parent_head, has_parent_prediction


def apply_nms(masks: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> list:
    """
    Apply Non-Maximum Suppression to remove overlapping predictions.

    Args:
        masks: [K, N] binary masks
        scores: [K] prediction scores
        iou_threshold: IoU threshold for suppression

    Returns:
        List of indices to keep
    """
    if len(scores) == 0:
        return []

    # Sort by score (descending)
    sorted_indices = torch.argsort(scores, descending=True)

    keep = []
    masks_binary = masks > 0.5

    while len(sorted_indices) > 0:
        # Keep the highest scoring prediction
        idx = sorted_indices[0].item()
        keep.append(idx)

        if len(sorted_indices) == 1:
            break

        # Compute IoU with remaining predictions
        current_mask = masks_binary[idx]
        remaining_indices = sorted_indices[1:]
        remaining_masks = masks_binary[remaining_indices]

        # Compute IoU
        intersection = (current_mask.unsqueeze(0) & remaining_masks).sum(dim=1).float()
        union = (current_mask.unsqueeze(0) | remaining_masks).sum(dim=1).float()
        iou = intersection / (union + 1e-8)

        # Remove predictions with high IoU
        keep_mask = iou < iou_threshold
        sorted_indices = remaining_indices[keep_mask]

    return keep


def masks_to_per_point_labels(
    masks: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    force_assign: bool = True,
) -> tuple:
    """
    Convert per-query masks to per-point labels.

    Uses Argmax competition mechanism (standard in Panoptic Segmentation like Mask2Former):
    For each point, assign it to the Query with the strongest response.

    Args:
        masks: [K, N] mask probabilities
        scores: [K] prediction scores
        threshold: Mask binarization threshold (only used when force_assign=False)
        force_assign: If True (default), every point is assigned to some part (no background).
                      This is appropriate for single-object segmentation where all points
                      belong to the object. If False, points with all mask values < threshold
                      are marked as background (-1).

    Returns:
        labels: [N] int, part label for each point (-1 for background only if force_assign=False)
        confidences: [N] float, confidence for each point's label
    """
    K, N = masks.shape

    if K == 0:
        return np.full(N, -1, dtype=np.int32), np.zeros(N, dtype=np.float32)

    # Weight masks by scores
    weighted_masks = masks * scores[:, np.newaxis]  # [K, N]

    # For each point, find the part with highest weighted mask value (Argmax competition)
    labels = np.argmax(weighted_masks, axis=0)  # [N]
    confidences = np.max(weighted_masks, axis=0)  # [N]

    # Optionally mark background points (where no mask > threshold)
    # When force_assign=True, skip this step - every point gets assigned
    if not force_assign:
        max_mask_value = np.max(masks, axis=0)  # [N]
        background_mask = max_mask_value < threshold
        labels[background_mask] = -1
        confidences[background_mask] = 0.0

    return labels.astype(np.int32), confidences.astype(np.float32)


def recover_limit(motion_type_idx: int, revolute_limit: list, prismatic_limit: list) -> list:
    """
    Recover motion limits from center-span parameterization.

    Args:
        motion_type_idx: 0=F, 1=P, 2=R, 3=C
        revolute_limit: [center, span] for revolute (normalized to π)
        prismatic_limit: [center, span] for prismatic (scene-normalized)

    Returns:
        [min_limit, max_limit] or None for Fixed/Continuous
    """
    if motion_type_idx == 0:  # Fixed - no motion
        return None
    elif motion_type_idx == 1:  # Prismatic
        center, span = prismatic_limit
        return [center - span, center + span]
    elif motion_type_idx == 2:  # Revolute
        center, span = revolute_limit
        # Denormalize from π scale
        return [(center - span) * PI, (center + span) * PI]
    else:  # Continuous - unlimited rotation
        return None  # Or could return [-inf, +inf]


def create_motion_json(
    anno_id: str,
    category_name: str,
    parts: list,
    points: np.ndarray,
    output_local_coords: bool = False,
) -> dict:
    """
    Create motion JSON in the same format as training input.

    Output format (new 4-class system with parent prediction):
    {
        "object_name": "StorageFurniture",
        "predicted": true,
        "parts": [{"label": 0, "name": "part_0"}, ...],
        "group_info": {
            "0": ["link_0", "base", "F"],                              // Fixed, parent=base
            "1": ["link_1", "link_0", [dx,dy,dz,ox,oy,oz,min,max], "R"], // Revolute, parent=link_0
            "2": ["link_2", "link_1", [dx,dy,dz,ox,oy,oz,min,max], "P"], // Prismatic, parent=link_1
        }
    }

    Motion Types:
    - F: Fixed (no motion)
    - P: Prismatic (translation with limits)
    - R: Revolute (rotation with limits)
    - C: Continuous (unlimited rotation)

    Parent Prediction (Stage 2):
    - If part has 'parent_name', use it (e.g., "link_0", "base")
    - If not available (Stage 1), default to "base"

    Args:
        anno_id: Annotation ID
        category_name: Object category name
        parts: List of part dictionaries with predictions
        points: Point cloud [N, 3]
        output_local_coords: If True, convert world coordinates to local (parent-relative)
                            coordinates for URDF-compatible output. Default: False (world coords)
    """
    output = {
        "object_name": category_name,
        "predicted": True,
        "anno_id": anno_id,
        "coordinate_frame": "local" if output_local_coords else "world",
        "parts": [],
        "group_info": {},
    }

    # Build parent map for inverse transform (if needed)
    parent_transforms = {}
    if output_local_coords:
        # First pass: collect world positions for each part
        for i, part in enumerate(parts):
            axis_origin = np.array(part["axis_origin"])
            # Store as simple translation transform
            T = np.eye(4)
            T[:3, 3] = axis_origin
            parent_transforms[i] = T
        # Add base transform (identity)
        parent_transforms[-1] = np.eye(4)

    for i, part in enumerate(parts):
        # Parts info
        output["parts"].append(
            {
                "label": i,
                "name": f"part_{i}",
                "score": float(part["score"]),
                "num_points": int(part["num_points"]),
            }
        )

        # Group info (motion parameters)
        motion_type_idx = part["motion_type_idx"]
        motion_letter = MOTION_TYPE_TO_LETTER.get(motion_type_idx, "F")

        axis_dir = part["axis_direction"]
        axis_origin = part["axis_origin"]

        # Get parent name (Stage 2) or default to "base" (Stage 1)
        parent_name = part.get("parent_name", "base")

        if motion_letter == "F":
            # Fixed part: ["link_X", "parent", "F"]
            group_data = [f"link_{i}", parent_name, "F"]
        else:
            # Convert to local coordinates if requested
            if output_local_coords:
                # Get parent's world transform
                if parent_name == "base":
                    parent_id = -1
                else:
                    try:
                        parent_id = int(parent_name.split("_")[1])
                    except (ValueError, IndexError):
                        parent_id = -1

                T_parent = parent_transforms.get(parent_id, np.eye(4))

                # Inverse transform
                axis_dir_local, axis_origin_local = inverse_transform_axis(
                    np.array(axis_dir), np.array(axis_origin), T_parent
                )
                axis_dir = axis_dir_local.tolist()
                axis_origin = axis_origin_local.tolist()

            # Movable part: ["link_X", "parent", [axis_dir, axis_pos, limits], "P/R/C"]
            # Get limits from center-span prediction
            limits = recover_limit(
                motion_type_idx,
                part.get("revolute_limit", [0.0, 0.5]),
                part.get("prismatic_limit", [0.0, 0.1]),
            )

            if limits is None:
                # Continuous type: use placeholder limits
                limits = [-PI, PI]

            # Combine axis parameters: [dx, dy, dz, ox, oy, oz, limit_min, limit_max]
            motion_params = axis_dir + axis_origin + limits
            group_data = [f"link_{i}", parent_name, motion_params, motion_letter]

        output["group_info"][str(i)] = group_data

    return output


def create_colored_ply(
    points: np.ndarray,
    normals: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    colormap: str = "tab20",
    face_ids: np.ndarray = None,
):
    """
    Create a colored PLY file for visualization.

    Args:
        points: [N, 3] point coordinates
        normals: [N, 3] point normals
        labels: [N] part labels (-1 for background)
        output_path: Output PLY file path
        colormap: Matplotlib colormap name
        face_ids: [N] per-point face IDs from input PLY (optional, preserved if provided)
    """
    import matplotlib.pyplot as plt

    N = points.shape[0]

    # Get colormap
    cmap = plt.get_cmap(colormap)

    # Assign colors based on labels
    unique_labels = np.unique(labels[labels >= 0])
    num_parts = len(unique_labels)

    colors = np.zeros((N, 3), dtype=np.uint8)

    # Background color (gray)
    colors[labels < 0] = [128, 128, 128]

    # Part colors
    for i, label in enumerate(unique_labels):
        color = cmap(i / max(num_parts, 1))[:3]
        color = (np.array(color) * 255).astype(np.uint8)
        colors[labels == label] = color

    # Check if face_ids is available and valid
    has_face_ids = face_ids is not None and len(face_ids) == N and not np.all(face_ids == -1)

    # Write PLY file
    with open(output_path, "w") as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int label\n")  # Part label (corresponds to motion.json parts[].label)
        if has_face_ids:
            f.write("property int face_id\n")  # Preserved from input PLY
        f.write("end_header\n")

        # Data
        for i in range(N):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} ")
            f.write(f"{normals[i, 0]:.6f} {normals[i, 1]:.6f} {normals[i, 2]:.6f} ")
            f.write(f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]} ")
            f.write(f"{labels[i]}")
            if has_face_ids:
                f.write(f" {face_ids[i]}")
            f.write("\n")


def compute_global_transforms(parts: list) -> dict:
    """
    Compute global transforms for each part based on parent-child structure.

    In URDF, joint origin and axis are defined relative to parent link frame.
    This function computes the global position and orientation for each joint.

    Args:
        parts: List of part dictionaries with 'parent_name', 'axis_origin', 'axis_direction'

    Returns:
        Dictionary mapping part_id -> {
            'global_origin': [x, y, z],
            'global_direction': [dx, dy, dz],
            'global_rotation': 3x3 rotation matrix (identity for now, can extend for rotations)
        }
    """
    num_parts = len(parts)

    # Build parent-child mapping
    # part_id -> parent_part_id (None means root/base)
    parent_map = {}
    for part in parts:
        part_id = part["part_id"]
        parent_name = part.get("parent_name", "base")

        if parent_name == "base":
            parent_map[part_id] = None  # Root
        else:
            # Extract parent_id from "link_X"
            try:
                parent_id = int(parent_name.split("_")[1])
                if parent_id < num_parts:
                    parent_map[part_id] = parent_id
                else:
                    parent_map[part_id] = None  # Invalid parent, treat as root
            except (ValueError, IndexError):
                parent_map[part_id] = None  # Parse error, treat as root

    # Initialize global transforms
    global_transforms = {}

    # Topological sort: process parents before children
    # Using iterative approach to handle arbitrary depth
    processed = set()

    def compute_transform(part_id):
        if part_id in processed:
            return global_transforms[part_id]

        part = parts[part_id]
        local_origin = np.array(part["axis_origin"])
        local_direction = np.array(part["axis_direction"])
        local_direction = local_direction / (np.linalg.norm(local_direction) + 1e-8)

        parent_id = parent_map.get(part_id)

        if parent_id is None:
            # Root node: local = global
            global_origin = local_origin
            global_direction = local_direction
            global_rotation = np.eye(3)
        else:
            # Ensure parent is processed first
            if parent_id not in processed:
                compute_transform(parent_id)

            parent_transform = global_transforms[parent_id]
            parent_global_origin = np.array(parent_transform["global_origin"])
            parent_global_rotation = np.array(parent_transform["global_rotation"])

            # Transform local to global
            # Global position = Parent_rotation @ Local_origin + Parent_position
            # For now, we assume parent rotation is based on its axis direction
            # This is a simplification - full FK would need joint angles
            global_origin = parent_global_rotation @ local_origin + parent_global_origin
            global_direction = parent_global_rotation @ local_direction
            global_direction = global_direction / (np.linalg.norm(global_direction) + 1e-8)

            # For visualization, we use identity rotation (no joint angle applied)
            # In reality, this would depend on current joint configuration
            global_rotation = parent_global_rotation  # Inherit parent rotation

        global_transforms[part_id] = {
            "global_origin": global_origin.tolist(),
            "global_direction": global_direction.tolist(),
            "global_rotation": global_rotation.tolist(),
            "local_origin": local_origin.tolist(),
            "local_direction": local_direction.tolist(),
        }
        processed.add(part_id)
        return global_transforms[part_id]

    # Process all parts
    for part in parts:
        compute_transform(part["part_id"])

    return global_transforms


def create_axes_ply(
    parts: list,
    output_path: str,
    axis_length: float = 0.3,
    points_per_axis: int = 50,
    use_global_transform: bool = True,
):
    """
    Create a PLY file visualizing motion axes.

    Args:
        parts: List of part dictionaries
        output_path: Output PLY file path
        axis_length: Length of axis visualization
        points_per_axis: Number of points per axis line
        use_global_transform: If True, compute global coordinates based on parent-child structure
    """
    all_points = []
    all_colors = []

    # Compute global transforms if enabled
    if use_global_transform and len(parts) > 0:
        global_transforms = compute_global_transforms(parts)
    else:
        global_transforms = None

    for i, part in enumerate(parts):
        motion_type = part["motion_type_idx"]

        # Skip fixed parts
        if motion_type == 0:
            continue

        part_id = part["part_id"]

        if global_transforms and part_id in global_transforms:
            # Use global coordinates
            origin = np.array(global_transforms[part_id]["global_origin"])
            direction = np.array(global_transforms[part_id]["global_direction"])
        else:
            # Fallback to local coordinates (original behavior)
            origin = np.array(part["axis_origin"])
            direction = np.array(part["axis_direction"])

        direction = direction / (np.linalg.norm(direction) + 1e-8)

        # Create points along the axis
        t = np.linspace(-axis_length / 2, axis_length / 2, points_per_axis)
        axis_points = origin + np.outer(t, direction)

        # Color based on motion type:
        # 0=F: Skip (above)
        # 1=P: Blue (prismatic/translation)
        # 2=R: Red (revolute with limits)
        # 3=C: Green (continuous rotation)
        if motion_type == 1:  # Prismatic
            color = [0, 0, 255]
        elif motion_type == 2:  # Revolute
            color = [255, 0, 0]
        else:  # Continuous (3)
            color = [0, 255, 0]

        all_points.extend(axis_points.tolist())
        all_colors.extend([color] * points_per_axis)

    if not all_points:
        return

    # Write PLY
    N = len(all_points)
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(N):
            f.write(f"{all_points[i][0]:.6f} {all_points[i][1]:.6f} {all_points[i][2]:.6f} ")
            f.write(f"{all_colors[i][0]} {all_colors[i][1]} {all_colors[i][2]}\n")


def save_per_link_ply(
    points: np.ndarray,
    normals: np.ndarray,
    labels: np.ndarray,
    parts: list,
    output_dir: str,
    colormap: str = "tab20",
):
    """
    Save each link as a separate PLY file in a subdirectory.

    Args:
        points: [N, 3] point coordinates
        normals: [N, 3] point normals
        labels: [N] per-point labels
        parts: List of part dictionaries
        output_dir: Output directory for per-link PLY files
        colormap: Matplotlib colormap name
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Get colormap
    cmap = plt.get_cmap(colormap)
    num_parts = len(parts)

    for part in parts:
        part_id = part["part_id"]
        motion_type = part.get("motion_type", "Unknown")
        parent_name = part.get("parent_name", "base")
        score = part.get("score", 0.0)

        # Get points belonging to this part
        mask = labels == part_id
        part_points = points[mask]
        part_normals = normals[mask]

        if len(part_points) == 0:
            continue

        # Get color for this part
        color_float = cmap(part_id % 20)[:3]
        color = [int(c * 255) for c in color_float]

        # Output filename
        ply_filename = f"link_{part_id}_{motion_type}_parent_{parent_name}.ply"
        ply_path = os.path.join(output_dir, ply_filename)

        # Write PLY
        N = len(part_points)
        with open(ply_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"comment link_id: {part_id}\n")
            f.write(f"comment motion_type: {motion_type}\n")
            f.write(f"comment parent: {parent_name}\n")
            f.write(f"comment score: {score:.4f}\n")
            f.write(f"comment num_points: {N}\n")
            f.write(f"element vertex {N}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for i in range(N):
                f.write(f"{part_points[i, 0]:.6f} {part_points[i, 1]:.6f} {part_points[i, 2]:.6f} ")
                f.write(
                    f"{part_normals[i, 0]:.6f} {part_normals[i, 1]:.6f} {part_normals[i, 2]:.6f} "
                )
                f.write(f"{color[0]} {color[1]} {color[2]}\n")

    # Also save a summary JSON
    summary = {
        "num_parts": num_parts,
        "parts": [
            {
                "part_id": p["part_id"],
                "motion_type": p.get("motion_type", "Unknown"),
                "parent_name": p.get("parent_name", "base"),
                "score": p.get("score", 0.0),
                "num_points": int((labels == p["part_id"]).sum()),
                "ply_file": f"link_{p['part_id']}_{p.get('motion_type', 'Unknown')}_parent_{p.get('parent_name', 'base')}.ply",
            }
            for p in parts
        ],
    }
    summary_path = os.path.join(output_dir, "parts_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
    config: dict,
    parent_head: torch.nn.Module = None,
) -> dict:
    """
    Run inference on a single sample.

    Args:
        model: ArticulatedMAFT model
        batch: Input batch
        device: Device to run on
        config: Configuration dictionary
        parent_head: ParentPredictionHead for Stage 2 (optional)

    Returns:
        Dictionary with all prediction results
    """
    inference_config = config.get("inference", {})
    score_threshold = inference_config.get("score_threshold", 0.5)
    mask_threshold = inference_config.get("mask_threshold", 0.5)
    use_nms = inference_config.get("use_nms", True)
    nms_iou_threshold = inference_config.get("nms_iou_threshold", 0.5)
    force_assign = inference_config.get("force_assign", True)  # Argmax competition, no background

    # Parent prediction config
    pp_config = config.get("parent_prediction", {})
    use_cycle_resolution = pp_config.get("inference", {}).get("use_cycle_resolution", True)

    # Move to device
    partfield_features = batch["partfield_features"].to(device)
    vae_features = batch["vae_features"].to(device)
    points = batch["points"].to(device)

    # Forward pass
    outputs = model(partfield_features, vae_features, points)

    # Get raw predictions
    mask_logits = outputs["mask_logits"][0]  # [Q, N]
    score_logits = outputs["scores"][0]  # [Q]

    # Apply sigmoid
    masks = torch.sigmoid(mask_logits)  # [Q, N]
    scores = torch.sigmoid(score_logits)  # [Q]

    # Filter by score threshold
    valid_mask = scores > score_threshold
    valid_indices = torch.where(valid_mask)[0]  # Keep track of original indices
    masks = masks[valid_mask]
    scores = scores[valid_mask]

    motion_type_logits = outputs["motion_type_logits"][0][valid_mask]
    axis_directions = outputs["axis_direction"][0][valid_mask]
    axis_origins = outputs["axis_origin"][0][valid_mask]
    query_positions = outputs["query_positions"][0][valid_mask]
    revolute_limits = outputs["revolute_limit"][0][valid_mask]  # [K, 2] (center, span)
    prismatic_limits = outputs["prismatic_limit"][0][valid_mask]  # [K, 2] (center, span)

    # Get query features for parent prediction (before filtering)
    query_features_all = outputs.get("query_features")  # [B, Q, D]
    query_positions_all = outputs.get("query_positions")  # [B, Q, 3]
    class_probs_all = outputs.get("part_class_probs")  # [B, Q, 18] or None

    # Apply NMS
    if use_nms and len(scores) > 0:
        keep_indices = apply_nms(masks, scores, nms_iou_threshold)
        masks = masks[keep_indices]
        scores = scores[keep_indices]
        motion_type_logits = motion_type_logits[keep_indices]
        axis_directions = axis_directions[keep_indices]
        axis_origins = axis_origins[keep_indices]
        query_positions = query_positions[keep_indices]
        revolute_limits = revolute_limits[keep_indices]
        prismatic_limits = prismatic_limits[keep_indices]
        # Update valid_indices after NMS
        valid_indices = valid_indices[keep_indices]

    K = len(scores)  # Number of final predictions

    # Get part class probs for valid queries (after NMS filtering)
    part_class_probs_filtered = None
    if class_probs_all is not None and K > 0:
        part_class_probs_filtered = class_probs_all[0, valid_indices]  # [K, 18]

    # Category prediction
    category_logits = outputs["category_logits"][0]
    category_probs = F.softmax(category_logits, dim=0)
    pred_category_idx = category_logits.argmax().item()
    # Use dynamic category names based on num_categories from config
    num_categories = config.get("model", {}).get("num_categories", 46)
    category_names_map = get_category_names(num_categories)
    pred_category_name = category_names_map.get(pred_category_idx, f"Unknown_{pred_category_idx}")

    # Convert to numpy
    masks_np = masks.cpu().numpy()
    scores_np = scores.cpu().numpy()

    # Per-point labels (Argmax competition mechanism)
    labels, confidences = masks_to_per_point_labels(
        masks_np, scores_np, mask_threshold, force_assign=force_assign
    )

    # ========== Stage 2: Parent Prediction ==========
    parent_predictions = None  # [K] parent index for each part, K means root
    kinematic_tree = None
    parent_confidences = None

    if parent_head is not None and K > 0:
        # Get filtered query features
        query_features_filtered = query_features_all[0, valid_indices]  # [K, D]
        query_positions_filtered = query_positions_all[0, valid_indices]  # [K, 3]

        # Get class_probs for filtered queries
        class_probs_filtered = None
        if class_probs_all is not None:
            class_probs_filtered = class_probs_all[0, valid_indices]  # [K, 18]
        else:
            print(
                "WARNING: part_class_probs is None. SemanticFusion may not be enabled. "
                "Falling back to use_semantic=False for parent prediction."
            )

        # Compute position embedding
        q_position = model.pos_embed_proj(
            model.pos_embed(query_positions_filtered.unsqueeze(0))
        )  # [1, K, D]

        # Run parent head
        parent_logits = parent_head(
            query_features_filtered.unsqueeze(0),  # [1, K, D]
            q_position,  # [1, K, D]
            class_probs_filtered.unsqueeze(0) if class_probs_filtered is not None else None,
        )  # [1, K, K+1]

        # Get parent predictions (argmax)
        parent_probs = F.softmax(parent_logits[0], dim=-1)  # [K, K+1]
        parent_predictions_raw = parent_logits[0].argmax(dim=-1).cpu().numpy()  # [K]
        parent_confidences = parent_probs.max(dim=-1)[0].cpu().numpy()  # [K]

        # Apply cycle resolution if enabled
        if use_cycle_resolution and PARENT_PREDICTION_AVAILABLE:
            parent_predictions = resolve_cycles(parent_logits[0])  # [K] numpy
        else:
            parent_predictions = parent_predictions_raw

        # Build kinematic tree
        if PARENT_PREDICTION_AVAILABLE:
            kinematic_tree = build_kinematic_tree(parent_predictions)
            kinematic_tree["tree_string"] = tree_to_string(kinematic_tree)

    # Build parts list
    parts = []
    for i in range(len(scores_np)):
        motion_type_idx = motion_type_logits[i].argmax().item()
        motion_probs = F.softmax(motion_type_logits[i], dim=0).cpu().numpy()

        # Normalize axis direction
        axis_dir = axis_directions[i].cpu().numpy()
        axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-8)

        # Get limit predictions (center, span format)
        rev_limit = revolute_limits[i].cpu().numpy().tolist()  # [center, span]
        pri_limit = prismatic_limits[i].cpu().numpy().tolist()  # [center, span]

        part = {
            "part_id": i,
            "score": float(scores_np[i]),
            "num_points": int((labels == i).sum()),
            "motion_type_idx": motion_type_idx,
            "motion_type": MOTION_TYPE_NAMES.get(motion_type_idx, "Unknown"),
            "motion_probs": motion_probs.tolist(),
            "axis_direction": axis_dir.tolist(),
            "axis_origin": axis_origins[i].cpu().numpy().tolist(),
            "query_position": query_positions[i].cpu().numpy().tolist(),
            "revolute_limit": rev_limit,  # [center, span] normalized to π
            "prismatic_limit": pri_limit,  # [center, span] scene-normalized
        }

        # Add parent prediction if available
        if parent_predictions is not None:
            parent_idx = int(parent_predictions[i])
            part["parent_idx"] = parent_idx  # K means root/base
            part["parent_name"] = "base" if parent_idx == K else f"link_{parent_idx}"
            if parent_confidences is not None:
                part["parent_confidence"] = float(parent_confidences[i])

        # Add part semantic class prediction (18 classes) if available
        if part_class_probs_filtered is not None:
            part_probs = part_class_probs_filtered[i].cpu().numpy()
            part_class_idx = int(part_probs.argmax())
            part["part_class_idx"] = part_class_idx
            part["part_class_confidence"] = float(part_probs[part_class_idx])
            # Add class name if available
            if PART_CLASS_AVAILABLE and IDX_TO_CLASS is not None:
                part["part_class_name"] = IDX_TO_CLASS.get(
                    part_class_idx, f"unknown_{part_class_idx}"
                )
            # Optionally add full probability distribution (top-3)
            top3_indices = part_probs.argsort()[-3:][::-1]
            part["part_class_top3"] = [
                {
                    "idx": int(idx),
                    "name": IDX_TO_CLASS.get(int(idx), f"unknown_{idx}")
                    if PART_CLASS_AVAILABLE
                    else str(idx),
                    "prob": float(part_probs[idx]),
                }
                for idx in top3_indices
            ]

        parts.append(part)

    result = {
        "anno_id": batch["anno_id"][0],
        "category_idx": pred_category_idx,
        "category_name": pred_category_name,
        "category_probs": category_probs.cpu().numpy().tolist(),
        "parts": parts,
        "labels": labels,
        "confidences": confidences,
        "masks": masks_np,
        "scores": scores_np,
    }

    # Add Stage 2 results
    if parent_predictions is not None:
        result["parent_predictions"] = parent_predictions.tolist()
        result["kinematic_tree"] = kinematic_tree

    return result


def save_outputs(
    result: dict,
    batch: dict,
    output_dir: str,
    config: dict,
):
    """Save inference outputs in various formats."""
    output_config = config.get("output", {})
    save_npz = output_config.get("save_npz", True)
    save_json = output_config.get("save_json", True)
    save_ply = output_config.get("save_ply", True)
    vis_config = output_config.get("visualization", {})

    anno_id = result["anno_id"]
    sample_dir = os.path.join(output_dir, anno_id)
    os.makedirs(sample_dir, exist_ok=True)

    # 1. Save NPZ (per-point segmentation)
    if save_npz:
        npz_path = os.path.join(sample_dir, "segmentation.npz")
        np.savez_compressed(
            npz_path,
            labels=result["labels"],  # [N] int32, part labels (-1 for background)
            confidences=result["confidences"],  # [N] float32, confidence scores
            masks=result["masks"],  # [K, N] float32, per-query masks
            scores=result["scores"],  # [K] float32, per-query scores
            category_idx=result["category_idx"],
            category_name=result["category_name"],
        )

    # 2. Save motion JSON (like training input format)
    if save_json:
        json_path = os.path.join(sample_dir, "motion.json")
        points_np = batch["points"][0].numpy()

        # Get output coordinate frame option from config
        output_local_coords = config.get("output", {}).get("output_local_coords", False)

        motion_json = create_motion_json(
            anno_id=anno_id,
            category_name=result["category_name"],
            parts=result["parts"],
            points=points_np,
            output_local_coords=output_local_coords,
        )
        with open(json_path, "w") as f:
            json.dump(motion_json, f, indent=2)

        # Also save detailed predictions
        detail_path = os.path.join(sample_dir, "predictions_detail.json")
        # Use dynamic category names based on num_categories from config
        num_categories = config.get("model", {}).get("num_categories", 46)
        category_names_map = get_category_names(num_categories)
        detail = {
            "anno_id": anno_id,
            "category": {
                "idx": result["category_idx"],
                "name": result["category_name"],
                "probs": {
                    category_names_map.get(i, str(i)): p
                    for i, p in enumerate(result["category_probs"])
                },
            },
            "parts": result["parts"],
        }
        with open(detail_path, "w") as f:
            json.dump(detail, f, indent=2)

    # 3. Save visualization PLY
    if save_ply and vis_config.get("enabled", True):
        points_np = batch["points"][0].numpy()
        normals_np = batch["normals"][0].numpy()
        # Get face_ids from batch (preserved from input PLY)
        face_ids_np = batch["face_ids"][0].numpy() if "face_ids" in batch else None

        # Segmentation visualization
        ply_path = os.path.join(sample_dir, "segmentation.ply")
        create_colored_ply(
            points=points_np,
            normals=normals_np,
            labels=result["labels"],
            output_path=ply_path,
            colormap=vis_config.get("colormap", "tab20"),
            face_ids=face_ids_np,
        )

        # Motion axes visualization (with global transform based on parent-child structure)
        if vis_config.get("show_axes", True):
            axes_path = os.path.join(sample_dir, "motion_axes.ply")
            use_global_transform = vis_config.get("use_global_transform", True)
            create_axes_ply(
                parts=result["parts"],
                output_path=axes_path,
                axis_length=vis_config.get("axis_length", 0.3),
                use_global_transform=use_global_transform,
            )

        # Per-link PLY files (each link as separate file in subdirectory)
        save_per_link = output_config.get("save_per_link_ply", True)
        if save_per_link and len(result["parts"]) > 0:
            per_link_dir = os.path.join(sample_dir, "per_link")
            save_per_link_ply(
                points=points_np,
                normals=normals_np,
                labels=result["labels"],
                parts=result["parts"],
                output_dir=per_link_dir,
                colormap=vis_config.get("colormap", "tab20"),
            )

    # 4. Save kinematic tree (Stage 2)
    save_kinematic_tree = output_config.get("save_kinematic_tree", True)
    if save_kinematic_tree and "kinematic_tree" in result and result["kinematic_tree"] is not None:
        tree_path = os.path.join(sample_dir, "kinematic_tree.json")
        tree_output = {
            "anno_id": anno_id,
            "num_parts": len(result["parts"]),
            "kinematic_tree": result["kinematic_tree"],
            "parent_predictions": result.get("parent_predictions", []),
            "parts_summary": [
                {
                    "part_id": p["part_id"],
                    "parent_name": p.get("parent_name", "base"),
                    "motion_type": p["motion_type"],
                    "score": p["score"],
                }
                for p in result["parts"]
            ],
        }
        with open(tree_path, "w") as f:
            json.dump(tree_output, f, indent=2)


def compute_metrics(result: dict, batch: dict, config: dict) -> dict:
    """Compute evaluation metrics if GT is available."""
    from scipy.optimize import linear_sum_assignment

    # Ground truth
    gt_category_idx = batch["category_idx"][0]
    if isinstance(gt_category_idx, torch.Tensor):
        gt_category_idx = gt_category_idx.item()

    gt_group_ids = batch["group_ids"][0].numpy()
    gt_motion_types = batch["gt_motion_types"][0].numpy()
    gt_axis_directions = batch["gt_axis_directions"][0].numpy()
    gt_axis_positions = batch["gt_axis_positions"][0].numpy()
    gt_link_ids = batch["gt_link_ids"][0].numpy()
    gt_is_movable = batch["gt_is_movable"][0].numpy()

    metrics = {
        "category_correct": result["category_idx"] == gt_category_idx,
    }

    # Build GT masks
    unique_groups = np.unique(gt_group_ids[gt_group_ids >= 0])
    G = len(unique_groups)
    N = gt_group_ids.shape[0]

    gt_masks = np.zeros((G, N), dtype=bool)
    for i, gid in enumerate(unique_groups):
        gt_masks[i] = gt_group_ids == gid

    # Match predictions to GT
    pred_masks = result["masks"]
    K = len(pred_masks)

    if K > 0 and G > 0:
        # Compute IoU matrix
        pred_binary = pred_masks > 0.5

        iou_matrix = np.zeros((K, G))
        for i in range(K):
            for j in range(G):
                intersection = (pred_binary[i] & gt_masks[j]).sum()
                union = (pred_binary[i] | gt_masks[j]).sum()
                iou_matrix[i, j] = intersection / (union + 1e-8)

        # Hungarian matching
        pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)

        ious = []
        motion_correct = []
        dir_errors = []
        origin_errors = []

        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            iou = iou_matrix[pred_idx, gt_idx]
            if iou < 0.1:
                continue

            ious.append(iou)

            # Find motion info for this GT group
            gt_gid = unique_groups[gt_idx]
            motion_idx = np.where(gt_link_ids == gt_gid)[0]

            if len(motion_idx) > 0:
                motion_idx = motion_idx[0]

                # Motion type accuracy
                pred_motion = result["parts"][pred_idx]["motion_type_idx"]
                gt_motion = gt_motion_types[motion_idx]
                motion_correct.append(pred_motion == gt_motion)

                # Axis errors (only for movable parts)
                if gt_is_movable[motion_idx]:
                    pred_dir = np.array(result["parts"][pred_idx]["axis_direction"])
                    gt_dir = gt_axis_directions[motion_idx]

                    # Direction error (angle in degrees)
                    cos_angle = np.abs(np.dot(pred_dir, gt_dir))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    dir_error = np.degrees(np.arccos(cos_angle))
                    dir_errors.append(dir_error)

                    # Origin error (point-to-line distance)
                    pred_origin = np.array(result["parts"][pred_idx]["axis_origin"])
                    gt_origin = gt_axis_positions[motion_idx]
                    diff = pred_origin - gt_origin
                    proj = np.dot(diff, gt_dir) * gt_dir
                    perp = diff - proj
                    origin_errors.append(np.linalg.norm(perp))

        metrics["mean_iou"] = np.mean(ious) if ious else 0.0
        metrics["motion_type_accuracy"] = np.mean(motion_correct) if motion_correct else 0.0
        metrics["mean_axis_direction_error"] = np.mean(dir_errors) if dir_errors else 0.0
        metrics["mean_axis_origin_error"] = np.mean(origin_errors) if origin_errors else 0.0
        metrics["matched_parts"] = len(ious)
        metrics["num_gt_parts"] = G
        metrics["num_pred_parts"] = K
    else:
        metrics["mean_iou"] = 0.0
        metrics["motion_type_accuracy"] = 0.0
        metrics["mean_axis_direction_error"] = 0.0
        metrics["mean_axis_origin_error"] = 0.0
        metrics["matched_parts"] = 0
        metrics["num_gt_parts"] = G
        metrics["num_pred_parts"] = K

    return metrics


def run_inference_pipeline(config: dict, checkpoint_path: str = None, anno_ids: list = None):
    """Run the complete inference pipeline."""
    from torch.utils.data import DataLoader

    from monoart.motion.datasets.articulated_dataset import ArticulatedDataset
    from monoart.motion.datasets.collate_fn import articulated_collate_fn

    inference_config = config.get("inference", {})
    output_config = config.get("output", {})
    eval_config = config.get("evaluation", {})

    # Device
    device_str = inference_config.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directory
    output_dir = output_config.get("dir", "./results/inference")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Checkpoint
    if checkpoint_path is None:
        checkpoint_path = config.get("checkpoint", {}).get("path")
    if checkpoint_path is None:
        raise ValueError("No checkpoint path provided")

    # Create model and load checkpoint (with optional parent_head for Stage 2)
    print("\nLoading model...")
    model = create_model(config)
    model, parent_head, has_parent_prediction = load_checkpoint_with_parent_head(
        model, checkpoint_path, device, config
    )
    model = model.to(device)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if parent_head is not None:
        print(f"ParentHead parameters: {sum(p.numel() for p in parent_head.parameters()):,}")

    # Create dataset
    data_config = config["data"]
    split = inference_config.get("split", "val")
    csv_key = f"{split}_csv"

    # Check if inference_only mode is enabled (skip GT loading)
    inference_only = inference_config.get("inference_only", False)
    if inference_only:
        print("[INFERENCE-ONLY MODE] Skipping GT data loading (group_id, motion JSON)")

    print(f"\nLoading {split} dataset...")
    dataset = ArticulatedDataset(
        csv_path=data_config[csv_key],
        json_dir=data_config["json_dir"],
        ply_dir=data_config["ply_dir"],
        vae_dir=data_config["vae_dir"],
        partfield_dir=data_config["partfield_dir"],
        n_points=data_config.get("n_points", 100000),
        augment=False,
        include_fixed_motion=data_config.get("include_fixed_motion", True),
        verbose=False,
        inference_only=inference_only,
        # Configurable filenames
        ply_filename=data_config.get("ply_filename", "sample_100k.ply"),
        vae_filename=data_config.get("vae_filename", None),
        vae_in_anno_dir=data_config.get("vae_in_anno_dir", False),
        partfield_filename=data_config.get("partfield_filename", "points_100000_feat.npy"),
    )

    # Filter by anno_ids if specified
    if anno_ids:
        indices = [i for i, aid in enumerate(dataset.anno_ids) if aid in anno_ids]
        if not indices:
            print("Warning: No matching anno_ids found")
            return
        # Create subset
        from torch.utils.data import Subset

        dataset = Subset(dataset, indices)
        print(f"Running inference on {len(indices)} specified samples")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=inference_config.get("num_workers", 4),
        collate_fn=articulated_collate_fn,
        pin_memory=True,
    )

    print(f"Running inference on {len(dataset)} samples...")

    # Inference loop
    all_metrics = []

    for batch in tqdm(dataloader, desc="Inference"):
        # Run inference (with optional parent_head for Stage 2)
        result = run_inference(model, batch, device, config, parent_head=parent_head)

        # Save outputs
        save_outputs(result, batch, output_dir, config)

        # Compute metrics if enabled
        if eval_config.get("enabled", True):
            metrics = compute_metrics(result, batch, config)
            metrics["anno_id"] = result["anno_id"]
            all_metrics.append(metrics)

    # Save summary metrics
    if all_metrics and output_config.get("save_metrics", True):
        # Aggregate metrics
        summary = {
            "total_samples": len(all_metrics),
            "category_accuracy": np.mean([m["category_correct"] for m in all_metrics]),
            "mean_iou": np.mean([m["mean_iou"] for m in all_metrics if m["mean_iou"] > 0]),
            "motion_type_accuracy": np.mean(
                [m["motion_type_accuracy"] for m in all_metrics if m["motion_type_accuracy"] > 0]
            ),
            "mean_axis_direction_error": np.mean(
                [
                    m["mean_axis_direction_error"]
                    for m in all_metrics
                    if m["mean_axis_direction_error"] > 0
                ]
            ),
            "mean_axis_origin_error": np.mean(
                [
                    m["mean_axis_origin_error"]
                    for m in all_metrics
                    if m["mean_axis_origin_error"] > 0
                ]
            ),
        }

        print("\n" + "=" * 60)
        print("INFERENCE SUMMARY")
        print("=" * 60)
        print(f"Total samples:          {summary['total_samples']}")
        print(f"Category Accuracy:      {summary['category_accuracy']:.4f}")
        print(f"Mean IoU:               {summary['mean_iou']:.4f}")
        print(f"Motion Type Accuracy:   {summary['motion_type_accuracy']:.4f}")
        print(f"Axis Direction Error:   {summary['mean_axis_direction_error']:.2f}°")
        print(f"Axis Origin Error:      {summary['mean_axis_origin_error']:.4f}")

        # Save summary
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to: {summary_path}")

        # Save per-sample metrics
        metrics_path = os.path.join(output_dir, "per_sample_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

    print(f"\nInference complete! Results saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Run inference with ArticulatedMAFT")
    parser.add_argument("--config", type=str, required=True, help="Path to inference config file")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint (overrides config)"
    )
    parser.add_argument(
        "--anno_id", type=str, nargs="+", default=None, help="Specific anno_ids to process"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["val", "test"],
        help="Data split to use (overrides config)",
    )
    parser.add_argument(
        "--score_threshold", type=float, default=None, help="Score threshold (overrides config)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command line args
    if args.output_dir:
        config["output"]["dir"] = args.output_dir
    if args.split:
        config["inference"]["split"] = args.split
    if args.score_threshold:
        config["inference"]["score_threshold"] = args.score_threshold

    # Run inference
    run_inference_pipeline(
        config=config,
        checkpoint_path=args.checkpoint,
        anno_ids=args.anno_id,
    )


if __name__ == "__main__":
    main()
