#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forward Kinematics for URDF-based articulated objects.
Converts local coordinates (relative to parent) to world coordinates.

Problem:
    URDF/JSON stores axis_pos and axis_dir relative to parent link frame.
    Point clouds are in normalized world coordinates.
    This mismatch causes training issues.

Solution:
    Apply Forward Kinematics to transform all coordinates to world frame.

Usage:
    from monoart.motion.datasets.world_transform import WorldTransformComputer

    computer = WorldTransformComputer()
    world_data = computer.compute_world_coordinates(group_info)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def build_transform_matrix(
    xyz: List[float],
    rpy: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Build 4x4 homogeneous transformation matrix.

    Args:
        xyz: [x, y, z] translation
        rpy: [roll, pitch, yaw] rotation in radians (default: [0,0,0])

    Returns:
        4x4 transformation matrix T where:
        - T[:3, :3] is the rotation matrix
        - T[:3, 3] is the translation vector
        - T[3, :] = [0, 0, 0, 1]
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = xyz

    if rpy is not None and any(abs(r) > 1e-10 for r in rpy):
        # Build rotation matrix from Euler angles (XYZ order)
        roll, pitch, yaw = rpy

        # Rotation around X (roll)
        Rx = np.array(
            [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
        )

        # Rotation around Y (pitch)
        Ry = np.array(
            [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
        )

        # Rotation around Z (yaw)
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

        # Combined rotation: R = Rz @ Ry @ Rx (URDF convention)
        R = Rz @ Ry @ Rx
        T[:3, :3] = R

    return T


def transform_point(point: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transform a 3D point using a 4x4 transformation matrix.

    Args:
        point: [3] or [N, 3] point(s) in local frame
        T: [4, 4] transformation matrix

    Returns:
        Transformed point(s) in world frame
    """
    point = np.asarray(point)
    if point.ndim == 1:
        # Single point
        p_homo = np.array([point[0], point[1], point[2], 1.0])
        p_world = T @ p_homo
        return p_world[:3]
    else:
        # Multiple points [N, 3]
        N = point.shape[0]
        ones = np.ones((N, 1))
        p_homo = np.hstack([point, ones])  # [N, 4]
        p_world = (T @ p_homo.T).T  # [N, 4]
        return p_world[:, :3]


def transform_direction(direction: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transform a direction vector using a 4x4 transformation matrix.
    Only rotation is applied (no translation).

    Args:
        direction: [3] direction vector in local frame
        T: [4, 4] transformation matrix

    Returns:
        Transformed direction in world frame (normalized)
    """
    R = T[:3, :3]
    d_world = R @ direction
    norm = np.linalg.norm(d_world)
    if norm > 1e-8:
        d_world = d_world / norm
    return d_world


class WorldTransformComputer:
    """
    Computes world coordinates for all links using Forward Kinematics.

    Given the hierarchical structure in group_info (from URDF), this class:
    1. Builds the kinematic tree
    2. Computes world transform for each link
    3. Transforms axis_pos and axis_dir to world frame
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def compute_world_coordinates(
        self,
        group_info: Dict[str, List],
        base_transform: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute world coordinates for all links.

        Args:
            group_info: JSON group_info dict with structure:
                {
                    "0": ["link_0", "parent_link", [axis_dir(3), axis_pos(3), limits(2)], "R"],
                    "1": ["link_1", "base", "F"],  # Fixed
                    ...
                }
            base_transform: Optional 4x4 base frame transform (default: identity)

        Returns:
            Dict mapping link_id (str) to:
            {
                'link_name': str,
                'parent_name': str,
                'motion_type': str,
                'world_transform': np.ndarray [4, 4],
                'axis_dir_local': np.ndarray [3] or None,
                'axis_pos_local': np.ndarray [3] or None,
                'axis_dir_world': np.ndarray [3] or None,
                'axis_pos_world': np.ndarray [3] or None,
                'limits': np.ndarray [2] or None,
            }
        """
        if base_transform is None:
            base_transform = np.eye(4, dtype=np.float64)

        # Step 1: Parse group_info and build kinematic tree
        link_data = {}
        parent_map = {}  # link_name -> parent_name

        for gid_str, gdata in group_info.items():
            link_name = gdata[0]
            parent_name = gdata[1]
            motion_type = gdata[-1]  # Last element is motion type

            parent_map[link_name] = parent_name

            # Parse motion parameters
            axis_dir_local = None
            axis_pos_local = None
            limits = None

            if len(gdata) >= 4 and isinstance(gdata[2], list):
                params = gdata[2]
                # Check if it's a list of lists (multi-motion) or single list
                if isinstance(params[0], list):
                    # Multi-motion case: [[params1], [params2]]
                    # Use the first motion's parameters
                    if len(params) > 0 and len(params[0]) >= 6:
                        axis_dir_local = np.array(params[0][0:3], dtype=np.float64)
                        axis_pos_local = np.array(params[0][3:6], dtype=np.float64)
                        if len(params[0]) >= 8:
                            limits = np.array(params[0][6:8], dtype=np.float64)
                elif len(params) >= 6:
                    # Single motion case: [axis_dir(3), axis_pos(3), limits(2)]
                    axis_dir_local = np.array(params[0:3], dtype=np.float64)
                    axis_pos_local = np.array(params[3:6], dtype=np.float64)
                    if len(params) >= 8:
                        limits = np.array(params[6:8], dtype=np.float64)

            link_data[link_name] = {
                "gid": gid_str,
                "link_name": link_name,
                "parent_name": parent_name,
                "motion_type": motion_type,
                "axis_dir_local": axis_dir_local,
                "axis_pos_local": axis_pos_local,
                "limits": limits,
            }

        # Step 2: Compute world transforms using Forward Kinematics
        world_transforms = {"base": base_transform}

        def get_world_transform(link_name: str) -> np.ndarray:
            """Recursively compute world transform for a link."""
            if link_name in world_transforms:
                return world_transforms[link_name]

            if link_name not in link_data:
                # Unknown link, assume identity relative to base
                world_transforms[link_name] = base_transform.copy()
                return world_transforms[link_name]

            data = link_data[link_name]
            parent_name = data["parent_name"]

            # Get parent's world transform
            T_parent = get_world_transform(parent_name)

            # Build local transform from axis_pos (translation)
            # Note: In URDF, the joint origin defines the child frame relative to parent
            if data["axis_pos_local"] is not None:
                T_local = build_transform_matrix(data["axis_pos_local"].tolist())
            else:
                T_local = np.eye(4, dtype=np.float64)

            # World transform = Parent transform @ Local transform
            T_world = T_parent @ T_local
            world_transforms[link_name] = T_world

            return T_world

        # Compute transforms for all links
        for link_name in link_data.keys():
            get_world_transform(link_name)

        # Step 3: Transform axis to world coordinates
        result = {}
        for gid_str, gdata in group_info.items():
            link_name = gdata[0]
            data = link_data[link_name]
            T_world = world_transforms.get(link_name, np.eye(4))

            # Get parent's world transform for proper axis transformation
            parent_name = data["parent_name"]
            T_parent = world_transforms.get(parent_name, np.eye(4))

            axis_dir_world = None
            axis_pos_world = None

            if data["axis_dir_local"] is not None:
                # Direction: transform using parent's rotation
                # (axis is defined in parent frame, not child frame)
                axis_dir_world = transform_direction(data["axis_dir_local"], T_parent)

            if data["axis_pos_local"] is not None:
                # Position: transform using parent's full transform
                axis_pos_world = transform_point(data["axis_pos_local"], T_parent)

            result[gid_str] = {
                "link_name": link_name,
                "parent_name": parent_name,
                "motion_type": data["motion_type"],
                "world_transform": T_world,
                "axis_dir_local": data["axis_dir_local"],
                "axis_pos_local": data["axis_pos_local"],
                "axis_dir_world": axis_dir_world,
                "axis_pos_world": axis_pos_world,
                "limits": data["limits"],
            }

            if self.verbose and axis_pos_world is not None:
                logger.info(
                    f"Link {gid_str} ({link_name}): "
                    f"local=[{data['axis_pos_local'][0]:.3f}, {data['axis_pos_local'][1]:.3f}, {data['axis_pos_local'][2]:.3f}] -> "
                    f"world=[{axis_pos_world[0]:.3f}, {axis_pos_world[1]:.3f}, {axis_pos_world[2]:.3f}]"
                )

        return result


def compute_projected_origin(
    part_center: np.ndarray,
    axis_pos: np.ndarray,
    axis_dir: np.ndarray,
) -> np.ndarray:
    """
    Compute the projection of part center onto the axis line.

    This is the ideal target for axis origin prediction:
    - Lies exactly on the axis line
    - Is closest to the part center

    Args:
        part_center: [3] center of the part's point cloud
        axis_pos: [3] a point on the axis line
        axis_dir: [3] normalized axis direction

    Returns:
        [3] projected point on axis line

    Math:
        Line: P = axis_pos + t * axis_dir
        Projection: t = (part_center - axis_pos) · axis_dir
        Result: axis_pos + t * axis_dir
    """
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-8)
    diff = part_center - axis_pos
    t = np.dot(diff, axis_dir)
    projected = axis_pos + t * axis_dir
    return projected


def inverse_transform_axis(
    axis_dir_world: np.ndarray,
    axis_pos_world: np.ndarray,
    T_parent_world: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse transform: Convert axis from world frame back to parent-relative (local) frame.

    This is the inverse of transform_axis_to_world.
    Used during inference to convert predictions back to URDF-compatible format.

    Args:
        axis_dir_world: [3] axis direction in world frame
        axis_pos_world: [3] axis position in world frame
        T_parent_world: [4, 4] parent's world transform matrix

    Returns:
        (axis_dir_local, axis_pos_local): Axis in parent's local frame
    """
    # Compute inverse of parent transform
    R = T_parent_world[:3, :3]
    t = T_parent_world[:3, 3]

    R_inv = R.T  # For rotation matrix, inverse = transpose
    t_inv = -R_inv @ t

    # Direction: only rotation (inverse)
    axis_dir_local = R_inv @ axis_dir_world
    axis_dir_local = axis_dir_local / (np.linalg.norm(axis_dir_local) + 1e-8)

    # Position: full inverse transformation
    axis_pos_local = R_inv @ axis_pos_world + t_inv

    return axis_dir_local, axis_pos_local


def convert_predictions_to_local(
    predictions: Dict[int, Dict],
    parent_map: Dict[int, int],
) -> Dict[int, Dict]:
    """
    Convert predicted world coordinates back to URDF-style local coordinates.

    Args:
        predictions: Dict mapping part_id -> {
            'axis_dir_world': [3],
            'axis_pos_world': [3],
            'motion_type': str,
            ...
        }
        parent_map: Dict mapping part_id -> parent_part_id (-1 or None for base)

    Returns:
        Dict with same structure but with 'axis_dir_local' and 'axis_pos_local' added
    """
    result = {}

    # First pass: compute world transforms for each part
    # (needed for inverse transformation of children)
    world_transforms = {-1: np.eye(4)}  # base has identity transform

    def build_transform_from_axis(axis_dir, axis_pos):
        """Build a simple transform matrix from axis info."""
        T = np.eye(4)
        T[:3, 3] = axis_pos
        # Note: For full accuracy, would need to build rotation from axis
        # Here we use translation only (sufficient for most cases)
        return T

    # Build world transforms (simplified - just using positions)
    for part_id, pred in predictions.items():
        if "axis_pos_world" in pred:
            T = build_transform_from_axis(
                np.array(pred.get("axis_dir_world", [0, 0, 1])), np.array(pred["axis_pos_world"])
            )
            world_transforms[part_id] = T

    # Second pass: inverse transform
    for part_id, pred in predictions.items():
        result[part_id] = pred.copy()

        if pred.get("motion_type") == "F":
            # Fixed parts don't have meaningful axis
            result[part_id]["axis_dir_local"] = np.zeros(3)
            result[part_id]["axis_pos_local"] = np.zeros(3)
            continue

        parent_id = parent_map.get(part_id, -1)
        if parent_id is None:
            parent_id = -1

        T_parent = world_transforms.get(parent_id, np.eye(4))

        axis_dir_world = np.array(pred.get("axis_dir_world", [0, 0, 1]))
        axis_pos_world = np.array(pred.get("axis_pos_world", [0, 0, 0]))

        axis_dir_local, axis_pos_local = inverse_transform_axis(
            axis_dir_world, axis_pos_world, T_parent
        )

        result[part_id]["axis_dir_local"] = axis_dir_local
        result[part_id]["axis_pos_local"] = axis_pos_local

    return result
