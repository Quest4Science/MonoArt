"""
Motion Prediction Loss for Articulated Object Motion Estimation.

Loss components:
1. Type Loss: CrossEntropy classification (Fixed/Prismatic/Revolute/Continuous)
2. Direction Loss: Unsigned direction loss: 1 - |pred · gt| (direction-agnostic)
3. Origin Loss: Point-to-line distance (origin should lie on GT axis line)
4. Anchor Loss: Constrain origin to be near the part center projection on axis
5. Limit Loss: Center-span prediction for motion range (P and R types only)

Motion Types (4-class):
- 0: Fixed (no motion, no axis/limit)
- 1: Prismatic (translation, has direction but origin position is irrelevant)
- 2: Revolute (rotation with limits, has axis/origin/limit)
- 3: Continuous (unlimited rotation, has axis/origin but NO limit)

Loss computation by motion type:
- Direction Loss: P, R, C (all movable parts need correct axis direction)
- Origin Loss: R, C only (rotation needs precise origin; P only needs direction)
- Anchor Loss: R, C only (same reason as origin loss)
- Limit Loss: P, R only (C has unlimited rotation)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionPredictionLoss(nn.Module):
    """
    Motion prediction loss module.

    Computes losses for:
    - Motion type classification (4 classes: F=0, P=1, R=2, C=3)
    - Axis direction (unsigned, direction-agnostic)
    - Axis origin (point-to-line distance)

    Note: Limit loss is handled separately by LimitLoss class.
    """

    def __init__(
        self,
        type_weight: float = 1.0,
        direction_weight: float = 1.0,
        origin_weight: float = 0.5,
        anchor_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        ignore_fixed_for_axis: bool = True,
        direction_loss_type: str = "cosine",  # 'cosine' or 'l2'
        origin_loss_type: str = "point_to_line",  # 'point_to_line' or 'l2'
        use_anchor_loss: bool = True,
    ):
        """
        Args:
            type_weight: Weight for motion type classification loss
            direction_weight: Weight for axis direction loss
            origin_weight: Weight for axis origin loss (point-to-line constraint)
            anchor_weight: Weight for anchor loss (proximity to part center)
            class_weights: Class weights for CrossEntropy [3], e.g., for imbalanced data
            ignore_fixed_for_axis: If True, don't compute axis loss for Fixed parts
            direction_loss_type: 'cosine' (1-|cos|) or 'l2' (MSE)
            origin_loss_type: 'point_to_line' or 'l2' (direct MSE)
            use_anchor_loss: If True, add anchor loss to constrain predicted origin
                to be close to the part center's projection on the axis line.
                This prevents the predicted origin from "drifting" to arbitrary
                positions along the axis line.
        """
        super().__init__()

        self.type_weight = type_weight
        self.direction_weight = direction_weight
        self.origin_weight = origin_weight
        self.anchor_weight = anchor_weight
        self.ignore_fixed_for_axis = ignore_fixed_for_axis
        self.direction_loss_type = direction_loss_type
        self.origin_loss_type = origin_loss_type
        self.use_anchor_loss = use_anchor_loss

        # CrossEntropy for type classification
        self.type_criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction="none",  # We'll reduce manually for matched pairs
        )

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        matched_indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute motion prediction losses.

        Args:
            predictions: Model outputs containing:
                - motion_type_logits: [B, Q, 4] type classification logits
                - axis_direction: [B, Q, 3] predicted axis direction (normalized)
                - axis_origin: [B, Q, 3] predicted axis origin
            targets: Ground truth containing:
                - gt_motion_types: List[Tensor] of shape [G_b] per batch
                - gt_axis_directions: List[Tensor] of shape [G_b, 3] per batch
                - gt_axis_positions: List[Tensor] of shape [G_b, 3] per batch
                - gt_projected_origins: List[Tensor] of shape [G_b, 3] per batch (optional)
                    The projection of part center onto axis line. Used for anchor loss.
            matched_indices: Hungarian matching results
                List of (pred_indices, gt_indices) tuples per batch

        Returns:
            Dictionary with loss values:
            - motion_loss: Total weighted motion loss
            - motion_type_loss: Type classification loss
            - motion_dir_loss: Direction loss (for P, R, C)
            - motion_origin_loss: Origin loss (for R, C only)
            - motion_anchor_loss: Anchor loss (for R, C only)
            - motion_num_matched: Number of matched pairs (for logging)
            - motion_num_movable: Number of movable parts P+R+C (for direction loss)
            - motion_num_rotation: Number of rotation parts R+C (for origin/anchor loss)
        """
        device = predictions["motion_type_logits"].device

        total_type_loss = torch.tensor(0.0, device=device)
        total_dir_loss = torch.tensor(0.0, device=device)
        total_origin_loss = torch.tensor(0.0, device=device)
        total_anchor_loss = torch.tensor(0.0, device=device)

        num_matched = 0
        num_movable = 0  # For direction loss (P, R, C)
        num_rotation = 0  # For origin/anchor loss (R, C only)

        # Check if projected origins are available
        has_projected_origins = "gt_projected_origins" in targets and self.use_anchor_loss

        for batch_idx, (pred_idx, gt_idx) in enumerate(matched_indices):
            if len(pred_idx) == 0:
                continue

            # Get matched GT arrays first to check bounds
            gt_types = targets["gt_motion_types"][batch_idx]  # [G]
            gt_dirs = targets["gt_axis_directions"][batch_idx]  # [G, 3]
            gt_origins = targets["gt_axis_positions"][batch_idx]  # [G, 3]

            # ========== Boundary Check ==========
            # gt_masks may have more groups than gt_motion_types if PLY and JSON are inconsistent
            # Filter out indices that exceed gt_motion_types bounds
            gt_motion_count = len(gt_types)
            if len(gt_idx) > 0 and gt_idx.max() >= gt_motion_count:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"[MotionLoss] Batch {batch_idx}: gt_idx.max()={gt_idx.max().item()} >= "
                    f"gt_motion_count={gt_motion_count}. Filtering out-of-bounds indices."
                )
                valid_mask = gt_idx < gt_motion_count
                pred_idx = pred_idx[valid_mask]
                gt_idx = gt_idx[valid_mask]
                if len(pred_idx) == 0:
                    continue

            # Get matched predictions
            pred_type_logits = predictions["motion_type_logits"][batch_idx, pred_idx]  # [K, 4]
            pred_dir = predictions["axis_direction"][batch_idx, pred_idx]  # [K, 3]
            pred_origin = predictions["axis_origin"][batch_idx, pred_idx]  # [K, 3]

            # Index into GT with matched indices (now guaranteed to be in bounds)
            gt_type = gt_types[gt_idx].to(device)  # [K]
            gt_dir = gt_dirs[gt_idx].to(device)  # [K, 3]
            gt_origin = gt_origins[gt_idx].to(device)  # [K, 3]

            # Get projected origins if available
            gt_projected = None
            if has_projected_origins:
                gt_projected_all = targets["gt_projected_origins"][batch_idx]  # [G, 3]
                gt_projected = gt_projected_all[gt_idx].to(device)  # [K, 3]

            # 1. Type Loss (compute for all matched pairs)
            type_loss = self.type_criterion(pred_type_logits, gt_type)
            total_type_loss = total_type_loss + type_loss.sum()
            num_matched += len(pred_idx)

            # 2. Direction Loss (for movable parts: P=1, R=2, C=3)
            # All movable parts need correct axis direction
            if self.ignore_fixed_for_axis:
                movable_mask = gt_type != 0  # 0 = Fixed, exclude
            else:
                movable_mask = torch.ones_like(gt_type, dtype=torch.bool)

            if movable_mask.sum() > 0:
                pred_dir_m = pred_dir[movable_mask]
                gt_dir_m = gt_dir[movable_mask]

                # Direction loss
                if self.direction_loss_type == "cosine":
                    dir_loss = self._direction_loss_cosine(pred_dir_m, gt_dir_m)
                else:
                    dir_loss = self._direction_loss_l2(pred_dir_m, gt_dir_m)
                total_dir_loss = total_dir_loss + dir_loss.sum()
                num_movable += movable_mask.sum().item()

            # 3. Origin Loss (for rotation parts only: R=2, C=3)
            # Prismatic (P=1) only needs direction, origin position is irrelevant
            # because translation happens along an infinite line
            rotation_mask = (gt_type == 2) | (gt_type == 3)  # R=2, C=3

            if rotation_mask.sum() > 0:
                pred_origin_r = pred_origin[rotation_mask]
                gt_origin_r = gt_origin[rotation_mask]
                gt_dir_r = gt_dir[rotation_mask]

                # Origin loss (point-to-line distance)
                if self.origin_loss_type == "point_to_line":
                    origin_loss = self._point_to_line_distance(pred_origin_r, gt_origin_r, gt_dir_r)
                else:
                    origin_loss = F.mse_loss(pred_origin_r, gt_origin_r, reduction="none").sum(
                        dim=-1
                    )
                total_origin_loss = total_origin_loss + origin_loss.sum()

                # 4. Anchor Loss (proximity to projected origin)
                # This constrains the predicted origin to be close to a specific point
                # on the axis line (the projection of part center), preventing
                # the prediction from drifting to arbitrary positions on the line.
                # Only for R and C types (rotation parts).
                if has_projected_origins and gt_projected is not None:
                    gt_projected_r = gt_projected[rotation_mask]
                    anchor_loss = self._anchor_loss(pred_origin_r, gt_projected_r)
                    total_anchor_loss = total_anchor_loss + anchor_loss.sum()

                num_rotation += rotation_mask.sum().item()

        # Average losses
        if num_matched > 0:
            total_type_loss = total_type_loss / num_matched
        if num_movable > 0:
            total_dir_loss = total_dir_loss / num_movable
        if num_rotation > 0:
            total_origin_loss = total_origin_loss / num_rotation
            if has_projected_origins:
                total_anchor_loss = total_anchor_loss / num_rotation

        # Weighted total
        total_loss = (
            self.type_weight * total_type_loss
            + self.direction_weight * total_dir_loss
            + self.origin_weight * total_origin_loss
        )

        # Add anchor loss if enabled
        if has_projected_origins:
            total_loss = total_loss + self.anchor_weight * total_anchor_loss

        return {
            "motion_loss": total_loss,
            "motion_type_loss": total_type_loss,
            "motion_dir_loss": total_dir_loss,
            "motion_origin_loss": total_origin_loss,
            "motion_anchor_loss": total_anchor_loss,
            "motion_num_matched": torch.tensor(num_matched, device=device, dtype=torch.float),
            "motion_num_movable": torch.tensor(num_movable, device=device, dtype=torch.float),
            "motion_num_rotation": torch.tensor(num_rotation, device=device, dtype=torch.float),
        }

    @staticmethod
    def _anchor_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Anchor loss: L2 distance between predicted origin and target (projected origin).

        This loss constrains the predicted axis origin to be close to a specific
        point on the axis line, preventing it from drifting to arbitrary positions.

        Args:
            pred: [K, 3] predicted axis origins
            target: [K, 3] target origins (projection of part center on axis line)

        Returns:
            [K] squared L2 distance values
        """
        return ((pred - target) ** 2).sum(dim=-1)

    @staticmethod
    def _direction_loss_cosine(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Unsigned direction loss using cosine similarity.

        Loss = 1 - |cos(theta)| = 1 - |pred · gt|

        This is direction-agnostic: pred and -pred have the same loss.

        Args:
            pred: [K, 3] predicted directions (should be normalized)
            gt: [K, 3] ground truth directions (should be normalized)

        Returns:
            [K] loss values in range [0, 1]
        """
        # Ensure normalization
        pred = F.normalize(pred, p=2, dim=-1, eps=1e-6)
        gt = F.normalize(gt, p=2, dim=-1, eps=1e-6)

        cos_sim = (pred * gt).sum(dim=-1)  # [K]
        return 1 - cos_sim.abs()  # Unsigned: |cos|

    @staticmethod
    def _direction_loss_l2(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Direction loss using L2 distance.

        Computes min(||pred - gt||, ||pred + gt||) to handle direction ambiguity.

        Args:
            pred: [K, 3] predicted directions
            gt: [K, 3] ground truth directions

        Returns:
            [K] squared L2 distance values
        """
        pred = F.normalize(pred, p=2, dim=-1, eps=1e-6)
        gt = F.normalize(gt, p=2, dim=-1, eps=1e-6)

        loss_pos = ((pred - gt) ** 2).sum(dim=-1)  # [K]
        loss_neg = ((pred + gt) ** 2).sum(dim=-1)  # [K]

        return torch.minimum(loss_pos, loss_neg)

    @staticmethod
    def _point_to_line_distance(
        point: torch.Tensor,
        line_point: torch.Tensor,
        line_direction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute squared distance from point to line.

        The line is defined by a point and direction.
        Distance = ||(point - line_point) x line_direction|| / ||line_direction||

        Args:
            point: [K, 3] query points
            line_point: [K, 3] points on the lines
            line_direction: [K, 3] line directions (will be normalized)

        Returns:
            [K] squared distances
        """
        # Normalize direction
        line_direction = F.normalize(line_direction, p=2, dim=-1, eps=1e-6)

        # Vector from line point to query point
        diff = point - line_point  # [K, 3]

        # Cross product gives perpendicular component
        cross = torch.cross(diff, line_direction, dim=-1)  # [K, 3]

        # Squared distance = ||cross||^2 (since direction is normalized)
        dist_sq = (cross**2).sum(dim=-1)  # [K]

        return dist_sq


class CategoryLoss(nn.Module):
    """
    Category classification loss (auxiliary supervision).

    Supports automatic class weight computation for imbalanced datasets.
    Weight formula: weight_i = total_samples / (num_classes * count_i)
    """

    def __init__(
        self,
        num_categories: int = 7,
        class_weights: Optional[torch.Tensor] = None,
        class_counts: Optional[List[int]] = None,
        weight: float = 0.3,
    ):
        """
        Args:
            num_categories: Number of object categories
            class_weights: Optional pre-computed class weights for imbalanced data
            class_counts: Optional list of sample counts per class for auto weight computation.
                          If provided, class_weights will be computed automatically.
                          Order should match category indices (0: StorageFurniture, 1: Table, ...)
            weight: Loss weight multiplier
        """
        super().__init__()
        self.weight = weight
        self.num_categories = num_categories

        # Auto-compute class weights from counts if provided
        if class_counts is not None and class_weights is None:
            class_weights = self._compute_class_weights(class_counts)
            print("CategoryLoss: Auto-computed class weights from counts")
            for i, (cnt, w) in enumerate(zip(class_counts, class_weights)):
                print(f"  Class {i}: count={cnt}, weight={w:.3f}")

        self.register_buffer("class_weights", class_weights)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def _compute_class_weights(self, class_counts: List[int]) -> torch.Tensor:
        """
        Compute class weights using inverse frequency.

        Formula: weight_i = total_samples / (num_classes * count_i)

        Args:
            class_counts: List of sample counts per class

        Returns:
            Tensor of class weights
        """
        total_samples = sum(class_counts)
        num_classes = len(class_counts)

        weights = []
        for count in class_counts:
            if count > 0:
                w = total_samples / (num_classes * count)
            else:
                w = 1.0  # Default weight for classes with no samples
            weights.append(w)

        return torch.tensor(weights, dtype=torch.float32)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute category classification loss.

        Args:
            predictions: Contains 'category_logits' [B, num_categories]
            targets: Contains 'categories' [B]

        Returns:
            Dictionary with:
            - category_loss: Classification loss value
        """
        category_logits = predictions["category_logits"]  # [B, 7]
        gt_categories = targets["categories"].to(category_logits.device)  # [B]

        loss = self.criterion(category_logits, gt_categories)

        return {
            "category_loss": self.weight * loss,
            "category_loss_unweighted": loss,
        }


class LimitLoss(nn.Module):
    """
    Motion limit prediction loss with center-span parameterization.

    Uses separate expert heads for revolute and prismatic motions:
    - Revolute: rotation limits (normalized to π)
    - Prismatic: translation limits (scene-normalized scale)

    Loss is only computed for:
    - P (Prismatic, type=1): Uses prismatic_limit head
    - R (Revolute, type=2): Uses revolute_limit head

    Loss is NOT computed for:
    - F (Fixed, type=0): No motion
    - C (Continuous, type=3): Unlimited rotation

    Center-Span parameterization:
    - center: midpoint of [min, max] range
    - span: half-width of range (always > 0, enforced by Softplus)
    - Recovers: min = center - span, max = center + span
    """

    def __init__(
        self,
        weight: float = 0.5,
        loss_type: str = "l1",  # 'l1' or 'smooth_l1'
        revolute_scale: float = 3.14159,  # π for revolute normalization
    ):
        """
        Args:
            weight: Loss weight multiplier
            loss_type: 'l1' or 'smooth_l1'
            revolute_scale: Scale factor for revolute limits (default: π)
        """
        super().__init__()
        self.weight = weight
        self.loss_type = loss_type
        self.revolute_scale = revolute_scale

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        matched_indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute limit prediction loss.

        Args:
            predictions: Model outputs containing:
                - revolute_limit: [B, Q, 2] (center, span) for revolute motion
                - prismatic_limit: [B, Q, 2] (center, span) for prismatic motion
            targets: Ground truth containing:
                - gt_motion_types: List[Tensor] of shape [G_b] per batch (0=F, 1=P, 2=R, 3=C)
                - gt_motion_limits: List[Tensor] of shape [G_b, 2] per batch (min, max)
            matched_indices: Hungarian matching results
                List of (pred_indices, gt_indices) tuples per batch

        Returns:
            Dictionary with loss values:
            - limit_loss: Total weighted limit loss
            - revolute_limit_loss: Loss for revolute parts only
            - prismatic_limit_loss: Loss for prismatic parts only
            - limit_num_revolute: Number of revolute parts (for logging)
            - limit_num_prismatic: Number of prismatic parts (for logging)
        """
        device = predictions["revolute_limit"].device

        total_rev_loss = torch.tensor(0.0, device=device)
        total_pri_loss = torch.tensor(0.0, device=device)

        num_revolute = 0
        num_prismatic = 0

        for batch_idx, (pred_idx, gt_idx) in enumerate(matched_indices):
            if len(pred_idx) == 0:
                continue

            # Get matched GT arrays first to check bounds
            gt_types = targets["gt_motion_types"][batch_idx]  # [G]
            gt_limits = targets["gt_motion_limits"][batch_idx]  # [G, 2]

            # ========== Boundary Check ==========
            # Filter out indices that exceed gt_motion_types bounds
            gt_motion_count = len(gt_types)
            if len(gt_idx) > 0 and gt_idx.max() >= gt_motion_count:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"[LimitLoss] Batch {batch_idx}: gt_idx.max()={gt_idx.max().item()} >= "
                    f"gt_motion_count={gt_motion_count}. Filtering out-of-bounds indices."
                )
                valid_mask = gt_idx < gt_motion_count
                pred_idx = pred_idx[valid_mask]
                gt_idx = gt_idx[valid_mask]
                if len(pred_idx) == 0:
                    continue

            # Get matched predictions
            pred_rev_limit = predictions["revolute_limit"][batch_idx, pred_idx]  # [K, 2]
            pred_pri_limit = predictions["prismatic_limit"][batch_idx, pred_idx]  # [K, 2]

            # Index into GT with matched indices (now guaranteed to be in bounds)
            gt_type = gt_types[gt_idx].to(device)  # [K]
            gt_limit = gt_limits[gt_idx].to(device)  # [K, 2] (min, max)

            # Convert GT limits to center-span format
            gt_center = (gt_limit[..., 0] + gt_limit[..., 1]) / 2  # [K]
            gt_span = (gt_limit[..., 1] - gt_limit[..., 0]) / 2  # [K]

            # Prismatic loss (type=1)
            prismatic_mask = gt_type == 1
            if prismatic_mask.sum() > 0:
                pred_p = pred_pri_limit[prismatic_mask]  # [K_p, 2]
                gt_c_p = gt_center[prismatic_mask]  # [K_p]
                gt_s_p = gt_span[prismatic_mask]  # [K_p]

                # L1 loss for center and span
                loss_center = self._compute_loss(pred_p[:, 0], gt_c_p)
                loss_span = self._compute_loss(pred_p[:, 1], gt_s_p)
                total_pri_loss = total_pri_loss + (loss_center + loss_span).sum()
                num_prismatic += prismatic_mask.sum().item()

            # Revolute loss (type=2)
            revolute_mask = gt_type == 2
            if revolute_mask.sum() > 0:
                pred_r = pred_rev_limit[revolute_mask]  # [K_r, 2]

                # Normalize GT to π scale
                gt_c_r = gt_center[revolute_mask] / self.revolute_scale  # [K_r]
                gt_s_r = gt_span[revolute_mask] / self.revolute_scale  # [K_r]

                # L1 loss for center and span (normalized)
                loss_center = self._compute_loss(pred_r[:, 0], gt_c_r)
                loss_span = self._compute_loss(pred_r[:, 1], gt_s_r)
                total_rev_loss = total_rev_loss + (loss_center + loss_span).sum()
                num_revolute += revolute_mask.sum().item()

        # Average losses
        if num_prismatic > 0:
            total_pri_loss = total_pri_loss / num_prismatic
        if num_revolute > 0:
            total_rev_loss = total_rev_loss / num_revolute

        # Total weighted loss
        total_loss = self.weight * (total_pri_loss + total_rev_loss)

        return {
            "limit_loss": total_loss,
            "revolute_limit_loss": total_rev_loss,
            "prismatic_limit_loss": total_pri_loss,
            "limit_num_revolute": torch.tensor(num_revolute, device=device, dtype=torch.float),
            "limit_num_prismatic": torch.tensor(num_prismatic, device=device, dtype=torch.float),
        }

    def _compute_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute element-wise loss."""
        if self.loss_type == "l1":
            return F.l1_loss(pred, gt, reduction="none")
        elif self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, gt, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
