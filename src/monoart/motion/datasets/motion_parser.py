"""
Motion annotation parser for articulated object dataset.
Parses JSON files containing motion information and maps to 4-class system.

Motion Types (4-class):
- 0: Fixed (static, no motion)
- 1: Prismatic (translation, e.g., drawer)
- 2: Revolute (rotation with limits, e.g., door hinge)
- 3: Continuous (unlimited rotation, e.g., fan blade)

Original Types Mapping:
- 'F' (Fixed) -> 0 (Fixed)
- 'P' (Prismatic/Translation) -> 1 (Prismatic)
- 'R' (Revolute) -> 2 (Revolute)
- 'C' (Continuous) -> 3 (Continuous)
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Import world transform utilities
try:
    from .world_transform import WorldTransformComputer

    WORLD_TRANSFORM_AVAILABLE = True
except ImportError:
    WORLD_TRANSFORM_AVAILABLE = False
    logger.warning("world_transform module not available, using local coordinates")


# Motion type mapping: original letter -> 4-class index
MOTION_TYPE_MAP_4CLASS = {
    "F": 0,  # Fixed
    "P": 1,  # Prismatic (Translation)
    "R": 2,  # Revolute (Rotation with limits)
    "C": 3,  # Continuous (Unlimited rotation)
}

# Alias for backward compatibility
MOTION_TYPE_MAP_3CLASS = MOTION_TYPE_MAP_4CLASS

MOTION_TYPE_NAMES = {
    0: "Fixed",
    1: "Prismatic",
    2: "Revolute",
    3: "Continuous",
}

# Which motion types need limit prediction
# F: no motion, no limit
# P: prismatic limit (translation range)
# R: revolute limit (rotation angle range)
# C: continuous, no limit (infinite rotation)
MOTION_TYPES_WITH_LIMIT = {1, 2}  # P and R need limit prediction

# =============================================================================
# Category Mappings
# =============================================================================

# Full category mapping: 46 categories (alphabetical order)
CATEGORY_MAP_FULL = {
    "Bottle": 0,
    "Box": 1,
    "Bucket": 2,
    "Camera": 3,
    "Cart": 4,
    "Chair": 5,
    "Clock": 6,
    "CoffeeMachine": 7,
    "Dishwasher": 8,
    "Dispenser": 9,
    "Display": 10,
    "Door": 11,
    "Eyeglasses": 12,
    "Fan": 13,
    "Faucet": 14,
    "FoldingChair": 15,
    "Globe": 16,
    "Kettle": 17,
    "Keyboard": 18,
    "KitchenPot": 19,
    "Knife": 20,
    "Lamp": 21,
    "Laptop": 22,
    "Lighter": 23,
    "Microwave": 24,
    "Mouse": 25,
    "Oven": 26,
    "Pen": 27,
    "Phone": 28,
    "Pliers": 29,
    "Printer": 30,
    "Refrigerator": 31,
    "Remote": 32,
    "Safe": 33,
    "Scissors": 34,
    "Stapler": 35,
    "StorageFurniture": 36,
    "Suitcase": 37,
    "Switch": 38,
    "Table": 39,
    "Toaster": 40,
    "Toilet": 41,
    "TrashCan": 42,
    "USB": 43,
    "WashingMachine": 44,
    "Window": 45,
}

# Singapo category mapping: 7 categories (alphabetical order)
CATEGORY_MAP_SINGAPO = {
    "Dishwasher": 0,
    "Microwave": 1,
    "Oven": 2,
    "Refrigerator": 3,
    "StorageFurniture": 4,
    "Table": 5,
    "WashingMachine": 6,
}

# Default to full mapping for backward compatibility
CATEGORY_MAP = CATEGORY_MAP_FULL
CATEGORY_NAMES = {v: k for k, v in CATEGORY_MAP.items()}

# Global variable to track current category map type
_current_num_categories = 46


def get_category_map(num_categories: int = None) -> Dict[str, int]:
    """
    Get the appropriate category mapping based on number of categories.

    Args:
        num_categories: Number of categories (7 for singapo, 46 for full).
                       If None, returns the current active map.

    Returns:
        Category name to index mapping dictionary.
    """
    if num_categories is None:
        num_categories = _current_num_categories

    if num_categories == 7:
        return CATEGORY_MAP_SINGAPO
    elif num_categories == 46:
        return CATEGORY_MAP_FULL
    else:
        # For other values, try to match closest
        if num_categories <= 7:
            return CATEGORY_MAP_SINGAPO
        return CATEGORY_MAP_FULL


def get_category_names(num_categories: int = None) -> Dict[int, str]:
    """
    Get the reverse mapping (index to name) based on number of categories.

    Args:
        num_categories: Number of categories (7 for singapo, 46 for full).

    Returns:
        Category index to name mapping dictionary.
    """
    cat_map = get_category_map(num_categories)
    return {v: k for k, v in cat_map.items()}


def set_category_map(num_categories: int):
    """
    Set the global default category map.

    This updates the module-level CATEGORY_MAP and CATEGORY_NAMES variables
    for backward compatibility with code that imports them directly.

    Args:
        num_categories: Number of categories (7 for singapo, 46 for full).
    """
    global CATEGORY_MAP, CATEGORY_NAMES, _current_num_categories

    _current_num_categories = num_categories
    CATEGORY_MAP = get_category_map(num_categories)
    CATEGORY_NAMES = {v: k for k, v in CATEGORY_MAP.items()}

    logger.info(f"Category map set to {num_categories} categories")


class MotionParser:
    """
    Parser for motion annotation JSON files.

    JSON Structure:
    {
        "object_name": "Remote",
        "parts": [{"label": 0, "name": "button"}, ...],
        "group_info": {
            "0": ["link_0", "link_39", [axis_dir(3), axis_pos(3), limit(2)], "B"],
            "39": ["link_39", "base", "F"]  # Fixed parts have only 3 elements
        }
    }
    """

    def __init__(
        self,
        use_3class: bool = True,
        verbose: bool = False,
        use_world_coordinates: bool = True,
    ):
        """
        Args:
            use_3class: If True, map to 3-class system. Otherwise keep original.
            verbose: If True, print debug information.
            use_world_coordinates: If True, transform axis coordinates from local
                (relative to parent) to world frame using Forward Kinematics.
                This is essential for proper training as point clouds are in world frame.
        """
        self.use_3class = use_3class
        self.verbose = verbose
        self.use_world_coordinates = use_world_coordinates

        # Initialize world transform computer if needed
        self.world_transform_computer = None
        if use_world_coordinates and WORLD_TRANSFORM_AVAILABLE:
            self.world_transform_computer = WorldTransformComputer(verbose=verbose)

    def parse(self, json_path: str) -> Optional[Dict[str, Any]]:
        """
        Parse motion annotation from JSON file.

        Args:
            json_path: Path to the JSON file

        Returns:
            Dictionary containing:
            - object_name: str
            - category_idx: int (0-6)
            - parts: List of part info
            - motions: List of motion parameters for each group
            - num_parts: int
            - num_movable: int (number of non-fixed parts)
        """
        if not os.path.exists(json_path):
            logger.warning(f"Motion JSON not found: {json_path}")
            return None

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading JSON {json_path}: {e}")
            return None

        # Parse object category
        object_name = data.get("object_name", "unknown")
        # Handle category name variations (e.g., "Storage Furniture" vs "StorageFurniture")
        category_key = object_name.replace(" ", "")
        category_idx = CATEGORY_MAP.get(category_key, -1)

        if category_idx == -1 and self.verbose:
            logger.warning(f"Unknown category: {object_name}")

        # Parse parts information
        parts_info = []
        if "parts" in data:
            for part in data["parts"]:
                parts_info.append(
                    {
                        "label": part.get("label", -1),
                        "name": part.get("name", "unknown"),
                    }
                )

        # Parse group/motion information
        motions = []
        group_to_link = {}  # Map group_id (joint_id) to link_id

        if "group_info" in data:
            for group_id_str, group_data in data["group_info"].items():
                group_id = int(group_id_str)

                # Parse link name to get link_id
                link_name = group_data[0]  # e.g., "link_0"
                if link_name.startswith("link_"):
                    link_id = int(link_name.split("_")[1])
                else:
                    link_id = group_id

                group_to_link[group_id] = link_id

                # Check if this is a Fixed part (only 3 elements) or movable (4 elements)
                if len(group_data) == 3:
                    # Fixed part: ["link_39", "base", "F"]
                    motion_type_letter = group_data[2]
                    motion = {
                        "group_id": group_id,
                        "link_id": link_id,
                        "parent_link": group_data[1],
                        "motion_type_letter": motion_type_letter,
                        "motion_type_idx": self._map_motion_type(motion_type_letter),
                        "axis_direction": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                        "axis_position": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                        "motion_limit": np.array([0.0, 0.0], dtype=np.float32),
                        "is_movable": False,
                    }
                elif len(group_data) >= 4:
                    # Movable part: ["link_0", "link_39", [8 numbers], "B"]
                    parent_link = group_data[1]
                    motion_params = group_data[2]
                    motion_type_letter = group_data[3]

                    # Extract motion parameters
                    if isinstance(motion_params, list) and len(motion_params) >= 8:
                        axis_dir = np.array(motion_params[0:3], dtype=np.float32)
                        axis_pos = np.array(motion_params[3:6], dtype=np.float32)
                        motion_limit = np.array(motion_params[6:8], dtype=np.float32)
                    else:
                        axis_dir = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                        axis_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                        motion_limit = np.array([0.0, 0.0], dtype=np.float32)

                    # Normalize axis direction
                    axis_norm = np.linalg.norm(axis_dir)
                    if axis_norm > 1e-6:
                        axis_dir = axis_dir / axis_norm

                    motion = {
                        "group_id": group_id,
                        "link_id": link_id,
                        "parent_link": parent_link,
                        "motion_type_letter": motion_type_letter,
                        "motion_type_idx": self._map_motion_type(motion_type_letter),
                        "axis_direction": axis_dir,
                        "axis_position": axis_pos,
                        "motion_limit": motion_limit,
                        "is_movable": motion_type_letter != "F",
                    }
                else:
                    # Unknown format, skip
                    if self.verbose:
                        logger.warning(
                            f"Unknown group_data format for group {group_id}: {group_data}"
                        )
                    continue

                motions.append(motion)

        # Sort by link_id for consistency
        motions.sort(key=lambda x: x["link_id"])

        # ========== World Coordinate Transformation ==========
        # Transform axis coordinates from local (relative to parent) to world frame
        # This is essential because:
        # - JSON/URDF stores coordinates relative to parent link
        # - Point clouds are in normalized world coordinates
        # - Without this transformation, training targets are incorrect
        if self.world_transform_computer is not None and "group_info" in data:
            try:
                world_data = self.world_transform_computer.compute_world_coordinates(
                    data["group_info"]
                )

                # Update motions with world coordinates
                for motion in motions:
                    gid_str = str(motion["group_id"])
                    if gid_str in world_data:
                        wd = world_data[gid_str]

                        # Store local coordinates for reference
                        motion["axis_direction_local"] = motion["axis_direction"].copy()
                        motion["axis_position_local"] = motion["axis_position"].copy()

                        # Update with world coordinates
                        if wd["axis_dir_world"] is not None:
                            motion["axis_direction"] = wd["axis_dir_world"].astype(np.float32)
                        if wd["axis_pos_world"] is not None:
                            motion["axis_position"] = wd["axis_pos_world"].astype(np.float32)

                        # Store world transform for potential inverse transformation later
                        motion["world_transform"] = wd["world_transform"]

                if self.verbose:
                    logger.info(f"Applied Forward Kinematics to {len(motions)} motions")

            except Exception as e:
                logger.warning(f"Failed to compute world coordinates: {e}")
                logger.warning("Using local coordinates instead")

        # Count movable parts
        num_movable = sum(1 for m in motions if m["is_movable"])

        result = {
            "object_name": object_name,
            "category_idx": category_idx,
            "parts": parts_info,
            "motions": motions,
            "group_to_link": group_to_link,
            "num_parts": len(parts_info),
            "num_groups": len(motions),
            "num_movable": num_movable,
            "raw_group_info": data.get("group_info", {}),  # Store raw data for reference
        }

        if self.verbose:
            self._print_summary(result)

        return result

    def _map_motion_type(self, letter: str) -> int:
        """Map motion type letter to index (4-class system)."""
        if self.use_3class:
            return MOTION_TYPE_MAP_4CLASS.get(letter, 0)  # Default to Fixed
        else:
            # Original 6-class mapping if needed (legacy)
            return {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}.get(letter, 5)

    def _print_summary(self, result: Dict):
        """Print summary of parsed motion data."""
        print(f"\n{'=' * 50}")
        print(f"Object: {result['object_name']} (category_idx: {result['category_idx']})")
        print(
            f"Parts: {result['num_parts']}, Groups: {result['num_groups']}, Movable: {result['num_movable']}"
        )
        print(f"{'=' * 50}")
        for m in result["motions"]:
            motion_name = MOTION_TYPE_NAMES.get(m["motion_type_idx"], "Unknown")
            print(f"  Link {m['link_id']:3d}: {motion_name:10s} ({m['motion_type_letter']})")
            if m["is_movable"]:
                print(f"           dir={m['axis_direction']}, pos={m['axis_position']}")
        print()

    def get_motion_labels(
        self, parsed_motion: Dict, include_fixed: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Convert parsed motion data to training labels.

        Args:
            parsed_motion: Output from parse()
            include_fixed: If True, include Fixed parts. Otherwise only movable parts.

        Returns:
            Dictionary of arrays:
            - motion_types: [G] int64, motion type indices
            - axis_directions: [G, 3] float32, normalized axis directions
            - axis_positions: [G, 3] float32, axis origin positions
            - motion_limits: [G, 2] float32, motion range limits
            - link_ids: [G] int32, corresponding link IDs
            - is_movable: [G] bool, whether each part is movable
            - part_names: List[str], part names for each group (for part classification)
        """
        if parsed_motion is None:
            return {
                "motion_types": np.array([], dtype=np.int64),
                "axis_directions": np.zeros((0, 3), dtype=np.float32),
                "axis_positions": np.zeros((0, 3), dtype=np.float32),
                "motion_limits": np.zeros((0, 2), dtype=np.float32),
                "link_ids": np.array([], dtype=np.int32),
                "is_movable": np.array([], dtype=bool),
                "part_names": [],
            }

        # Build link_id -> part_name mapping from parts info
        link_to_name = {}
        for part in parsed_motion.get("parts", []):
            label = part.get("label", -1)
            name = part.get("name", "unknown")
            if label >= 0:
                link_to_name[label] = name

        motion_types = []
        axis_directions = []
        axis_positions = []
        motion_limits = []
        link_ids = []
        is_movable = []
        part_names = []

        for motion in parsed_motion["motions"]:
            if not include_fixed and not motion["is_movable"]:
                continue

            motion_types.append(motion["motion_type_idx"])
            axis_directions.append(motion["axis_direction"])
            axis_positions.append(motion["axis_position"])
            motion_limits.append(motion["motion_limit"])
            link_ids.append(motion["link_id"])
            is_movable.append(motion["is_movable"])
            # Get part name for this group (default to 'unknown' if not found)
            part_names.append(link_to_name.get(motion["link_id"], "unknown"))

        return {
            "motion_types": np.array(motion_types, dtype=np.int64),
            "axis_directions": np.array(axis_directions, dtype=np.float32)
            if axis_directions
            else np.zeros((0, 3), dtype=np.float32),
            "axis_positions": np.array(axis_positions, dtype=np.float32)
            if axis_positions
            else np.zeros((0, 3), dtype=np.float32),
            "motion_limits": np.array(motion_limits, dtype=np.float32)
            if motion_limits
            else np.zeros((0, 2), dtype=np.float32),
            "link_ids": np.array(link_ids, dtype=np.int32),
            "is_movable": np.array(is_movable, dtype=bool),
            "part_names": part_names,
        }

    def get_parent_info(self, parsed_motion: Dict, include_fixed: bool = False) -> Dict[int, int]:
        """
        Extract parent-child relationships from parsed motion data.

        Args:
            parsed_motion: Output from parse()
            include_fixed: If True, include Fixed parts. Otherwise only movable parts.

        Returns:
            Dictionary mapping link_id -> parent_link_id
            - parent_link_id = -1 means root (connected to "base")
            - parent_link_id >= 0 means connected to another link
        """
        if parsed_motion is None:
            return {}

        parent_info = {}

        for motion in parsed_motion["motions"]:
            if not include_fixed and not motion["is_movable"]:
                continue

            link_id = motion["link_id"]
            parent_link = motion["parent_link"]

            if parent_link == "base":
                # Connected to base/root
                parent_info[link_id] = -1
            elif parent_link.startswith("link_"):
                # Connected to another link
                parent_link_id = int(parent_link.split("_")[1])
                parent_info[link_id] = parent_link_id
            else:
                # Unknown format, treat as root
                parent_info[link_id] = -1

        return parent_info
