"""
Articulated Object Dataset for Part Segmentation and Motion Prediction.

Loads:
- Semantic-reasoner features [N, 448] - for part segmentation
- VAE features [N, 8] - for global/category encoding
- PLY point cloud (XYZ, Normal, group_id)
- Motion JSON annotations

Dual-feature architecture:
- VAE (8D) -> Global feature -> Category prediction + Query generation
- Semantic reasoner (448D) -> Point features -> MAFT Decoder -> Part segmentation
"""

import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from plyfile import PlyData
from torch.utils.data import Dataset

from .motion_parser import MotionParser, get_category_map, get_category_names
from .world_transform import compute_projected_origin

# Import semantic fusion for part class mapping (optional)
try:
    from monoart.motion.semantic_fusion import NUM_PART_CLASSES, get_merged_class_idx

    SEMANTIC_FUSION_AVAILABLE = True
except ImportError:
    SEMANTIC_FUSION_AVAILABLE = False
    NUM_PART_CLASSES = 18

    def get_merged_class_idx(name: str) -> int:
        """Fallback: return -1 for unknown classes."""
        return -1


logger = logging.getLogger(__name__)


class ArticulatedDataset(Dataset):
    """
    Dataset for articulated object part segmentation and motion prediction.

    Data files:
    - CSV: anno_id, model_cat, source
    - JSON: {json_dir}/{model_id}/{config}.json
    - PLY: {ply_dir}/{anno_id}/sample_100k.ply
    - VAE: {vae_dir}/{anno_id}_100k_features.npz
    - Reasoner: {partfield_dir}/{anno_id}/points_100000_feat.npy

    Output dict keys:
    - anno_id: str
    - category_idx: int (0-6)
    - category_name: str
    - points: [N, 3] float32, XYZ coordinates
    - normals: [N, 3] float32, surface normals
    - group_ids: [N] int32, per-point part labels
    - partfield_features: [N, 448] float32, semantic-reasoner features
    - vae_features: [N, 8] float32, VAE features
    - gt_motion_types: [G] int64, motion type labels (0=F, 1=P, 2=R, 3=C)
    - gt_axis_directions: [G, 3] float32, normalized axis directions
    - gt_axis_positions: [G, 3] float32, axis origin positions
    - gt_motion_limits: [G, 2] float32, motion limits (min, max)
    - gt_link_ids: [G] int32, link IDs for each motion
    - gt_is_movable: [G] bool, whether each part is movable
    - gt_part_class_labels: [G] int64, part class labels (0-17 for 18 classes, -1 for unknown)
    - gt_part_names: List[str], original part names for each group
    - gt_parent_info: Dict[int, int], mapping link_id -> parent_link_id (-1 for root)
    - num_parts: int, number of unique parts
    """

    def __init__(
        self,
        csv_path: str,
        json_dir: str,
        ply_dir: str,
        vae_dir: str,
        partfield_dir: str,
        n_points: int = 100000,
        augment: bool = False,
        include_fixed_motion: bool = True,
        verbose: bool = False,
        num_categories: int = 46,
        use_world_coordinates: bool = True,
        inference_only: bool = False,
        ply_filename: str = "sample_100k.ply",
        vae_filename: str = None,
        vae_in_anno_dir: bool = False,
        partfield_filename: str = "points_100000_feat.npy",
    ):
        """
        Args:
            csv_path: Path to CSV file with anno_id, model_cat, source columns
            json_dir: Directory containing motion JSON files
            ply_dir: Directory containing PLY point cloud files
            vae_dir: Directory containing VAE feature files
            partfield_dir: Directory containing semantic-reasoner feature files
            n_points: Number of points per sample (for validation)
            augment: Whether to apply data augmentation
            include_fixed_motion: Whether to include Fixed parts in motion labels
            verbose: Whether to print debug information
            num_categories: Number of object categories (7 for singapo, 46 for full)
            use_world_coordinates: Whether to transform axis coordinates from local
                (relative to parent) to world frame using Forward Kinematics.
                This is essential for proper training. Default: True
            inference_only: If True, skip loading GT labels (group_id, motion JSON).
                Useful for pure inference without GT data. Default: False
            ply_filename: PLY filename within anno_id folder. Default: "sample_100k.ply"
            vae_filename: VAE filename. If None, uses "{anno_id}_100k_features.npz" pattern.
                If specified with vae_in_anno_dir=True, uses "{vae_dir}/{anno_id}/{vae_filename}"
            vae_in_anno_dir: If True, VAE file is inside anno_id folder. Default: False
            partfield_filename: Reasoner filename within anno_id folder. Default: "points_100000_feat.npy"
        """
        super().__init__()

        self.inference_only = inference_only

        self.json_dir = json_dir
        self.ply_dir = ply_dir
        self.vae_dir = vae_dir
        self.partfield_dir = partfield_dir
        self.n_points = n_points
        self.augment = augment
        self.include_fixed_motion = include_fixed_motion
        self.verbose = verbose
        self.num_categories = num_categories
        self.use_world_coordinates = use_world_coordinates

        # Configurable filenames
        self.ply_filename = ply_filename
        self.vae_filename = vae_filename
        self.vae_in_anno_dir = vae_in_anno_dir
        self.partfield_filename = partfield_filename

        # Get category map based on num_categories
        self.category_map = get_category_map(num_categories)
        self.category_names = get_category_names(num_categories)

        # Load CSV
        self.df = pd.read_csv(csv_path)
        # Ensure anno_ids are strings (pandas may convert numeric IDs to int)
        self.anno_ids = [str(aid) for aid in self.df["anno_id"].tolist()]
        self.categories = self.df["model_cat"].tolist()

        # Motion parser with world coordinate transformation
        self.motion_parser = MotionParser(
            use_3class=True,
            verbose=False,
            use_world_coordinates=use_world_coordinates,
        )

        # Validate data availability
        self._validate_data()

        logger.info(f"Loaded {len(self)} samples from {csv_path}")
        if use_world_coordinates:
            logger.info("Using world coordinates (Forward Kinematics enabled)")

    def _validate_data(self):
        """Check that all required files exist for each sample."""
        missing = []
        # In inference_only mode, skip json validation
        skip_keys = {"json"} if self.inference_only else set()

        for idx, anno_id in enumerate(self.anno_ids):
            paths = self._get_paths(anno_id)
            for name, path in paths.items():
                if name in skip_keys:
                    continue
                if not os.path.exists(path):
                    missing.append((anno_id, name, path))
                    if len(missing) <= 5:  # Only show first 5
                        logger.warning(f"Missing {name} for {anno_id}: {path}")

        if missing:
            logger.warning(f"Total missing files: {len(missing)}")
            # Filter out samples with missing files
            valid_ids = []
            valid_cats = []
            for anno_id, cat in zip(self.anno_ids, self.categories):
                paths = self._get_paths(anno_id)
                # In inference_only mode, skip json check
                if all(os.path.exists(p) for name, p in paths.items() if name not in skip_keys):
                    valid_ids.append(anno_id)
                    valid_cats.append(cat)

            logger.info(
                f"Filtered to {len(valid_ids)} valid samples (removed {len(self.anno_ids) - len(valid_ids)})"
            )
            self.anno_ids = valid_ids
            self.categories = valid_cats

    def _get_paths(self, anno_id: str) -> Dict[str, str]:
        """Get file paths for a given anno_id."""
        # Parse anno_id: e.g., "23724_config_0_3_10"
        parts = anno_id.split("_config_")
        if len(parts) == 2:
            model_id = parts[0]
            config_name = "config_" + parts[1]
        else:
            model_id = anno_id
            config_name = anno_id

        # VAE path: configurable location and filename
        if self.vae_in_anno_dir:
            # VAE file inside anno_id folder: {vae_dir}/{anno_id}/{vae_filename}
            vae_filename = self.vae_filename or "features_glb.npz"
            vae_path = os.path.join(self.vae_dir, anno_id, vae_filename)
        else:
            # VAE file with anno_id prefix: {vae_dir}/{anno_id}_100k_features.npz
            vae_path = os.path.join(self.vae_dir, f"{anno_id}_100k_features.npz")

        return {
            "json": os.path.join(self.json_dir, model_id, f"{config_name}.json"),
            "ply": os.path.join(self.ply_dir, anno_id, self.ply_filename),
            "vae": vae_path,
            "partfield": os.path.join(self.partfield_dir, anno_id, self.partfield_filename),
        }

    def __len__(self) -> int:
        return len(self.anno_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        anno_id = self.anno_ids[idx]
        category_name = self.categories[idx]
        paths = self._get_paths(anno_id)

        # Load PLY (points, normals, group_ids, face_ids)
        points, normals, group_ids, face_ids = self._load_ply(paths["ply"])

        # Load features
        partfield_features = self._load_partfield(paths["partfield"])
        vae_features = self._load_vae(paths["vae"])

        # Get category index (using instance category_map for correct mapping)
        category_idx = self.category_map.get(category_name, -1)

        # ========== Inference-only mode: skip GT loading ==========
        if self.inference_only:
            # Return dummy GT data for inference-only mode
            num_parts = 0
            return {
                "anno_id": anno_id,
                "category_idx": category_idx,
                "category_name": category_name,
                "points": torch.from_numpy(points).float(),
                "normals": torch.from_numpy(normals).float(),
                "group_ids": torch.from_numpy(group_ids).long(),
                "face_ids": torch.from_numpy(face_ids).long(),  # Preserve face_id from input PLY
                "partfield_features": torch.from_numpy(partfield_features).float(),
                "vae_features": torch.from_numpy(vae_features).float(),
                # Dummy GT data (not used in inference-only mode)
                "gt_motion_types": torch.zeros(0, dtype=torch.long),
                "gt_axis_directions": torch.zeros(0, 3, dtype=torch.float),
                "gt_axis_positions": torch.zeros(0, 3, dtype=torch.float),
                "gt_motion_limits": torch.zeros(0, 2, dtype=torch.float),
                "gt_link_ids": torch.zeros(0, dtype=torch.long),
                "gt_is_movable": torch.zeros(0, dtype=torch.bool),
                "gt_part_class_labels": torch.zeros(0, dtype=torch.long),
                "gt_part_names": [],
                "gt_parent_info": {},
                "gt_projected_origins": torch.zeros(0, 3, dtype=torch.float),
                "num_parts": num_parts,
            }

        # ========== Normal mode: load GT data ==========
        # Load motion annotations
        motion_data = self.motion_parser.parse(paths["json"])
        motion_labels = self.motion_parser.get_motion_labels(
            motion_data, include_fixed=self.include_fixed_motion
        )
        # Get parent-child relationships
        parent_info = self.motion_parser.get_parent_info(
            motion_data, include_fixed=self.include_fixed_motion
        )

        # Apply augmentation if enabled
        if self.augment:
            points, normals, motion_labels = self._augment(points, normals, motion_labels)

        # Count unique parts
        num_parts = len(np.unique(group_ids[group_ids >= 0]))

        # Map part names to 18 merged classes for part classification
        part_names = motion_labels.get("part_names", [])
        part_class_labels = (
            np.array([get_merged_class_idx(name) for name in part_names], dtype=np.int64)
            if part_names
            else np.array([], dtype=np.int64)
        )

        # ========== Compute Projected Origins ==========
        # For each part, compute the projection of its center onto the axis line.
        # This is the ideal training target for axis origin:
        # - Lies exactly on the axis line
        # - Is closest to the part center (intuitive and stable)
        gt_projected_origins = self._compute_projected_origins(points, group_ids, motion_labels)

        return {
            "anno_id": anno_id,
            "category_idx": category_idx,
            "category_name": category_name,
            "points": torch.from_numpy(points).float(),
            "normals": torch.from_numpy(normals).float(),
            "group_ids": torch.from_numpy(group_ids).long(),
            "face_ids": torch.from_numpy(face_ids).long(),  # Preserve face_id from input PLY
            "partfield_features": torch.from_numpy(partfield_features).float(),
            "vae_features": torch.from_numpy(vae_features).float(),
            "gt_motion_types": torch.from_numpy(motion_labels["motion_types"]).long(),
            "gt_axis_directions": torch.from_numpy(motion_labels["axis_directions"]).float(),
            "gt_axis_positions": torch.from_numpy(motion_labels["axis_positions"]).float(),
            "gt_motion_limits": torch.from_numpy(
                motion_labels["motion_limits"]
            ).float(),  # [G, 2] (min, max)
            "gt_link_ids": torch.from_numpy(motion_labels["link_ids"]).long(),
            "gt_is_movable": torch.from_numpy(motion_labels["is_movable"]).bool(),
            "gt_part_class_labels": torch.from_numpy(
                part_class_labels
            ).long(),  # [G] (0-17 for 18 classes)
            "gt_part_names": part_names,  # List[str]
            "gt_parent_info": parent_info,  # Dict[int, int]: link_id -> parent_link_id (-1 for root)
            "gt_projected_origins": torch.from_numpy(
                gt_projected_origins
            ).float(),  # [G, 3] projected anchor points
            "num_parts": num_parts,
        }

    def _load_ply(self, ply_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load point cloud from PLY file.

        Returns:
            points: [N, 3] point coordinates
            normals: [N, 3] surface normals
            group_ids: [N] per-point part labels
            face_ids: [N] per-point face IDs (or -1 if not available)
        """
        try:
            ply_data = PlyData.read(ply_path)
        except Exception as e:
            print(f"\n[ERROR] Failed to load PLY file: {ply_path}")
            print(f"[ERROR] Exception: {type(e).__name__}: {e}")
            raise RuntimeError(f"PLY load failed: {ply_path}") from e
        vertex = ply_data["vertex"]
        property_names = [p.name for p in vertex.properties]

        # Extract coordinates
        points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)

        # Extract normals
        normals = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=-1).astype(np.float32)

        # Extract group_ids (part labels)
        # In inference_only mode, group_id may not exist - return all -1
        if self.inference_only and "group_id" not in property_names:
            group_ids = np.full(len(points), -1, dtype=np.int32)
        else:
            group_ids = np.array(vertex["group_id"]).astype(np.int32)

        # Extract face_ids (if available)
        if "face_id" in property_names:
            face_ids = np.array(vertex["face_id"]).astype(np.int32)
        else:
            face_ids = np.full(len(points), -1, dtype=np.int32)

        return points, normals, group_ids, face_ids

    def _load_partfield(self, path: str) -> np.ndarray:
        """Load semantic-reasoner features [N, 448]."""
        features = np.load(path).astype(np.float32)
        assert features.shape[1] == 448, (
            f"Expected 448D reasoner features, got {features.shape[1]}D"
        )
        return features

    def _load_vae(self, path: str) -> np.ndarray:
        """Load VAE features [N, 8]."""
        data = np.load(path)
        features = data["features"].astype(np.float32)
        assert features.shape[1] == 8, f"Expected 8D VAE, got {features.shape[1]}D"
        return features

    def _compute_projected_origins(
        self,
        points: np.ndarray,
        group_ids: np.ndarray,
        motion_labels: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Compute projected origin for each part.

        For each movable part:
        1. Compute the part's point cloud center
        2. Project the center onto the axis line
        3. Use this projected point as the training target

        This is better than using the raw axis_position because:
        - Raw axis_position can be anywhere on the axis line
        - Projected point is closest to the part (intuitive)
        - Provides stable, consistent training targets

        Args:
            points: [N, 3] point cloud coordinates
            group_ids: [N] per-point part labels
            motion_labels: Dict containing axis_directions, axis_positions, link_ids

        Returns:
            [G, 3] projected origin for each group
        """
        axis_directions = motion_labels["axis_directions"]  # [G, 3]
        axis_positions = motion_labels["axis_positions"]  # [G, 3]
        link_ids = motion_labels["link_ids"]  # [G]
        is_movable = motion_labels["is_movable"]  # [G]

        G = len(link_ids)
        projected_origins = np.zeros((G, 3), dtype=np.float32)

        for g in range(G):
            link_id = link_ids[g]
            mask = group_ids == link_id

            if mask.sum() == 0:
                # No points for this part, use original axis_position
                projected_origins[g] = axis_positions[g]
                continue

            if not is_movable[g]:
                # Fixed parts don't have meaningful axis, use zero
                projected_origins[g] = np.zeros(3, dtype=np.float32)
                continue

            # Part center in world coordinates
            part_center = points[mask].mean(axis=0)

            # Project onto axis line
            projected_origins[g] = compute_projected_origin(
                part_center=part_center,
                axis_pos=axis_positions[g],
                axis_dir=axis_directions[g],
            )

        return projected_origins

    def _augment(
        self, points: np.ndarray, normals: np.ndarray, motion_labels: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Apply data augmentation.

        Augmentations:
        - Random rotation around Z-axis
        - Random scaling
        - Random jitter

        Note: Motion parameters (axis_direction, axis_position) must be
        transformed consistently with the point cloud.
        """
        # Random rotation around Z-axis
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)

        # Apply rotation to points and normals
        points = points @ rot_matrix.T
        normals = normals @ rot_matrix.T

        # Apply rotation to motion parameters
        if len(motion_labels["axis_directions"]) > 0:
            motion_labels["axis_directions"] = motion_labels["axis_directions"] @ rot_matrix.T
            motion_labels["axis_positions"] = motion_labels["axis_positions"] @ rot_matrix.T

        # Random scaling (0.9 - 1.1)
        scale = np.random.uniform(0.9, 1.1)
        points = points * scale
        motion_labels["axis_positions"] = motion_labels["axis_positions"] * scale

        # Random jitter (small noise)
        jitter = np.random.normal(0, 0.005, points.shape).astype(np.float32)
        points = points + jitter

        return points, normals, motion_labels

    def get_category_weights(self) -> torch.Tensor:
        """Compute class weights for category classification."""
        counts = np.zeros(self.num_categories)
        for cat in self.categories:
            idx = self.category_map.get(cat, -1)
            if idx >= 0 and idx < self.num_categories:
                counts[idx] += 1

        # Inverse frequency weighting
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * self.num_categories
        return torch.from_numpy(weights).float()
