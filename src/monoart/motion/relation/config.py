"""
Configuration for Parent-Child Relation Prediction Module.

Contains:
- Default configuration values
- Semantic affinity matrix computation
- Part class statistics for parent-child relationships
"""

import glob
import json
import os
from dataclasses import dataclass
from typing import Optional

import torch

# Import part class config from semantic_fusion
try:
    from monoart.motion.semantic_fusion.config import (
        CLASS_NAMES,
        get_merged_class_idx,
    )
    from monoart.motion.semantic_fusion.config import (
        NUM_CLASSES as NUM_PART_CLASSES,
    )
except ImportError:
    # Fallback definitions
    NUM_PART_CLASSES = 18
    CLASS_NAMES = [
        "base",
        "button",
        "dial",
        "display",
        "door",
        "drawer",
        "frame",
        "handle",
        "hinge",
        "knob",
        "lever",
        "lid",
        "other",
        "pedal",
        "shelf",
        "switch",
        "wheel",
        "window",
    ]

    def get_merged_class_idx(name: str) -> int:
        """Fallback: return -1 for unknown classes."""
        return -1


@dataclass
class ParentPredictionConfig:
    """Configuration for parent-child prediction head."""

    # Model architecture
    d_model: int = 448
    use_position: bool = True  # Use Q_position in addition to Q_content
    use_semantic: bool = True  # Use semantic affinity branch
    lambda_semantic_init: float = 0.3  # Initial weight for semantic branch
    init_affinity_from_stats: bool = False  # Initialize affinity from statistics

    # Root token
    use_root_token: bool = True

    # Training
    freeze_decoder: bool = False  # Whether to freeze decoder in Stage 2
    decoder_lr_scale: float = 0.1  # LR multiplier for decoder if not frozen
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    parent_loss_weight: float = 1.0

    # Inference
    use_cycle_resolution: bool = True  # Post-process to ensure tree structure


# Default prior affinity matrix (18x18)
# A[i, j] = P(class_i is child of class_j), rough estimates
# Most parts connect to 'base' (index 0), 'door' (4), 'drawer' (5), 'frame' (6)
DEFAULT_AFFINITY = torch.zeros(NUM_PART_CLASSES, NUM_PART_CLASSES)

# Common parent-child patterns (rough priors)
# Index mapping: base=0, button=1, dial=2, display=3, door=4, drawer=5,
#                frame=6, handle=7, hinge=8, knob=9, lever=10, lid=11,
#                other=12, pedal=13, shelf=14, switch=15, wheel=16, window=17

_affinity_rules = [
    # (child_class, parent_class, weight)
    # button usually on base/door/drawer
    (1, 0, 0.5),
    (1, 4, 0.3),
    (1, 5, 0.2),
    # dial usually on base
    (2, 0, 0.8),
    (2, 4, 0.2),
    # display usually on base
    (3, 0, 0.9),
    (3, 4, 0.1),
    # door usually on base/frame
    (4, 0, 0.6),
    (4, 6, 0.4),
    # drawer usually on base/frame
    (5, 0, 0.6),
    (5, 6, 0.4),
    # frame usually on base
    (6, 0, 1.0),
    # handle usually on door/drawer/lid
    (7, 4, 0.5),
    (7, 5, 0.3),
    (7, 11, 0.2),
    # hinge usually on door
    (8, 4, 0.8),
    (8, 0, 0.2),
    # knob usually on door/drawer
    (9, 4, 0.5),
    (9, 5, 0.3),
    (9, 0, 0.2),
    # lever usually on base
    (10, 0, 0.8),
    (10, 4, 0.2),
    # lid usually on base
    (11, 0, 0.9),
    (11, 6, 0.1),
    # other usually on base
    (12, 0, 0.8),
    (12, 6, 0.2),
    # pedal usually on base
    (13, 0, 1.0),
    # shelf usually on frame/base
    (14, 6, 0.6),
    (14, 0, 0.4),
    # switch usually on base
    (15, 0, 0.9),
    (15, 4, 0.1),
    # wheel usually on base
    (16, 0, 1.0),
    # window usually on door/base
    (17, 4, 0.6),
    (17, 0, 0.4),
]

for child, parent, weight in _affinity_rules:
    if child < NUM_PART_CLASSES and parent < NUM_PART_CLASSES:
        DEFAULT_AFFINITY[child, parent] = weight


def compute_parent_affinity_matrix(
    json_dir: str,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Compute parent-child affinity matrix from dataset statistics.

    Scans all JSON files and counts (child_class, parent_class) pairs.

    Args:
        json_dir: Directory containing motion JSON files
        save_path: If provided, save the computed matrix
        verbose: Print progress

    Returns:
        affinity: [18, 18] tensor where A[i,j] = P(class_i is child of class_j)
    """
    counts = torch.zeros(NUM_PART_CLASSES, NUM_PART_CLASSES)
    root_counts = torch.zeros(NUM_PART_CLASSES)  # Count of parts connected to base

    json_files = glob.glob(os.path.join(json_dir, "**/*.json"), recursive=True)

    if verbose:
        print(f"Scanning {len(json_files)} JSON files...")

    for json_path in json_files:
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        # Build link_id -> class_idx mapping
        link_to_class = {}
        for part in data.get("parts", []):
            link_id = part.get("label", -1)
            name = part.get("name", "unknown")
            class_idx = get_merged_class_idx(name)
            if link_id >= 0 and class_idx >= 0:
                link_to_class[link_id] = class_idx

        # Count parent-child relationships
        for group_id, group_data in data.get("group_info", {}).items():
            if len(group_data) < 2:
                continue

            # Get child link_id
            link_name = group_data[0]
            if link_name.startswith("link_"):
                child_link_id = int(link_name.split("_")[1])
            else:
                continue

            if child_link_id not in link_to_class:
                continue
            child_class = link_to_class[child_link_id]

            # Get parent
            parent_link = group_data[1]
            if parent_link == "base":
                root_counts[child_class] += 1
            elif parent_link.startswith("link_"):
                parent_link_id = int(parent_link.split("_")[1])
                if parent_link_id in link_to_class:
                    parent_class = link_to_class[parent_link_id]
                    counts[child_class, parent_class] += 1

    # Normalize to probabilities
    # Add root counts to diagonal (self-loop for visualization) or handle separately
    total_per_child = counts.sum(dim=1) + root_counts
    total_per_child = total_per_child.clamp(min=1)

    affinity = counts / total_per_child.unsqueeze(1)

    if verbose:
        print(f"Computed affinity matrix from {len(json_files)} files")
        print(f"Root connections per class: {root_counts.tolist()}")

    if save_path:
        torch.save(
            {
                "affinity": affinity,
                "counts": counts,
                "root_counts": root_counts,
                "class_names": CLASS_NAMES,
            },
            save_path,
        )
        if verbose:
            print(f"Saved to {save_path}")

    return affinity


def get_affinity_matrix(
    init_from_stats: bool = False,
    json_dir: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> torch.Tensor:
    """
    Get the affinity matrix for parent-child prediction.

    Args:
        init_from_stats: Whether to compute from statistics
        json_dir: Required if init_from_stats=True
        cache_path: Path to cached affinity matrix

    Returns:
        affinity: [18, 18] tensor
    """
    if cache_path and os.path.exists(cache_path):
        data = torch.load(cache_path)
        return data["affinity"]

    if init_from_stats and json_dir:
        return compute_parent_affinity_matrix(json_dir, save_path=cache_path)

    return DEFAULT_AFFINITY.clone()
