"""
Combined Loss for Articulated Object Part Segmentation and Motion Prediction.

Total Loss = λ_seg * L_seg + λ_motion * L_motion + λ_cls * L_cls + L_limit + L_center

Where:
- L_seg = L_mask (focal + dice) + L_score
- L_motion = L_type + L_direction + L_origin + L_anchor
- L_cls = Category classification loss
- L_limit = Motion limit prediction loss (center-span parameterization, P and R only)
- L_center = Query position supervision loss (query_pos -> GT part center)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .motion_loss import CategoryLoss, LimitLoss, MotionPredictionLoss

logger = logging.getLogger(__name__)


# Import semantic fusion for part classification (optional)
try:
    from monoart.motion.semantic_fusion import (
        CLASS_WEIGHTS,
        NUM_PART_CLASSES,
        PartClassificationLoss,
    )

    SEMANTIC_FUSION_AVAILABLE = True
except ImportError:
    SEMANTIC_FUSION_AVAILABLE = False
    NUM_PART_CLASSES = 18
    CLASS_WEIGHTS = None
    PartClassificationLoss = None


class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for bipartite matching between predictions and ground truth.

    Matches queries to GT parts based on mask IoU and classification scores.

    Supports high-order matching cost from Rank-DETR:
        Cost = -score * (IoU)^alpha
    This penalizes "high-score but low-quality" predictions.

    Uses chunked computation to avoid OOM with large point clouds (100k+) and
    many GT parts (100+). Instead of creating [Q, G, N] tensors at once,
    computes cost in chunks of queries.
    """

    def __init__(
        self,
        cost_mask: float = 1.0,
        cost_dice: float = 1.0,
        cost_score: float = 1.0,
        # High-order matching (Rank-DETR style)
        use_high_order_matching: bool = True,
        iou_power_alpha: float = 3.0,
        # Chunked computation to avoid OOM
        chunk_size: int = 10,
    ):
        """
        Args:
            cost_mask: Weight for mask BCE cost
            cost_dice: Weight for mask dice cost
            cost_score: Weight for score cost
            use_high_order_matching: Use multiplicative high-order matching cost
            iou_power_alpha: Power for IoU in high-order matching (2~4 recommended)
            chunk_size: Number of queries to process at once for cost computation.
                        Smaller = less memory, larger = faster. Default 10.
        """
        super().__init__()
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_score = cost_score
        self.use_high_order_matching = use_high_order_matching
        self.iou_power_alpha = iou_power_alpha
        self.chunk_size = chunk_size

    @torch.no_grad()
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, Any],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Hungarian matching.

        Args:
            predictions:
                - mask_logits: [B, Q, N] predicted mask logits
                - scores: [B, Q] predicted scores
            targets:
                - gt_masks: List[Tensor] of [G_b, N] GT masks per batch
                - or group_ids: [B, N] with integer part IDs

        Returns:
            List of (pred_indices, gt_indices) tuples for each batch
        """
        B = predictions["mask_logits"].shape[0]
        Q = predictions["mask_logits"].shape[1]

        matched_indices = []

        for b in range(B):
            pred_masks = predictions["mask_logits"][b]  # [Q, N]
            pred_scores = predictions["scores"][b]  # [Q]

            # Get GT masks
            if "gt_masks" in targets:
                gt_masks = targets["gt_masks"][b]  # [G, N]
            else:
                # Build GT masks from group_ids
                group_ids = targets["group_ids"][b]  # [N]
                unique_ids = torch.unique(group_ids)
                unique_ids = unique_ids[unique_ids >= 0]  # Remove invalid
                gt_masks = (group_ids.unsqueeze(0) == unique_ids.unsqueeze(1)).float()  # [G, N]

            G = gt_masks.shape[0]

            if G == 0:
                matched_indices.append(
                    (
                        torch.tensor([], dtype=torch.long, device=pred_masks.device),
                        torch.tensor([], dtype=torch.long, device=pred_masks.device),
                    )
                )
                continue

            # Compute cost matrix using chunked computation to avoid OOM
            # Original: [Q, G, N] tensor would be 100 * 116 * 100000 * 4 = 4.6GB
            # Chunked: [chunk_size, G, N] tensor is much smaller
            with torch.amp.autocast("cuda", enabled=False):
                pred_masks_float = pred_masks.float()
                gt_masks_float = gt_masks.float()

                # Clamp logits to prevent extreme values that cause NaN in BCE
                pred_masks_float = pred_masks_float.clamp(-50, 50)

                pred_probs = pred_masks_float.sigmoid()  # [Q, N]

                # Initialize cost matrices
                cost_mask = torch.zeros(Q, G, device=pred_masks.device)
                cost_dice = torch.zeros(Q, G, device=pred_masks.device)
                iou_matrix = torch.zeros(Q, G, device=pred_masks.device)

                # Chunked computation
                chunk_size = self.chunk_size
                for i in range(0, Q, chunk_size):
                    end_i = min(i + chunk_size, Q)

                    # Get chunk of predictions
                    pred_chunk_logits = pred_masks_float[i:end_i]  # [chunk, N]
                    pred_chunk_probs = pred_probs[i:end_i]  # [chunk, N]
                    chunk_len = end_i - i

                    # Mask BCE cost for this chunk: [chunk, G, N] -> [chunk, G]
                    cost_mask[i:end_i] = F.binary_cross_entropy_with_logits(
                        pred_chunk_logits.unsqueeze(1).expand(-1, G, -1),  # [chunk, G, N]
                        gt_masks_float.unsqueeze(0).expand(chunk_len, -1, -1),  # [chunk, G, N]
                        reduction="none",
                    ).mean(dim=-1)  # [chunk, G]

                    # Dice cost for this chunk
                    cost_dice[i:end_i] = self._dice_cost_chunk(
                        pred_chunk_probs, gt_masks_float
                    )  # [chunk, G]

                    # IoU for this chunk
                    iou_matrix[i:end_i] = self._compute_iou_chunk(
                        pred_chunk_probs, gt_masks_float
                    )  # [chunk, G]

                # Score cost
                pred_scores_prob = pred_scores.float().sigmoid()  # [Q]

                if self.use_high_order_matching:
                    # High-order matching from Rank-DETR:
                    # Cost = -score * (IoU)^alpha
                    # This penalizes high-score but low-IoU predictions
                    # Clamp IoU to [0, 1] before pow to prevent NaN
                    iou_clamped = iou_matrix.clamp(0, 1)
                    iou_powered = iou_clamped.pow(self.iou_power_alpha)  # [Q, G]

                    cost_score = -pred_scores_prob.unsqueeze(1) * iou_powered  # [Q, G]
                else:
                    # Traditional linear score cost
                    cost_score = -pred_scores_prob.unsqueeze(1).expand(-1, G)  # [Q, G]

            # Total cost
            C = (
                self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
                + self.cost_score * cost_score
            )

            # Check for NaN/Inf and handle gracefully
            if torch.isnan(C).any() or torch.isinf(C).any():
                nan_count = torch.isnan(C).sum().item()
                inf_count = torch.isinf(C).sum().item()
                logger.warning(
                    "Cost matrix contains %d NaN and %d Inf values; replacing them",
                    nan_count,
                    inf_count,
                )

                # Replace NaN/Inf with large values to avoid matching errors
                C = torch.where(
                    torch.isnan(C) | torch.isinf(C), torch.tensor(1e6, device=C.device), C
                )

            # Hungarian matching
            C_np = C.cpu().numpy()
            pred_idx, gt_idx = linear_sum_assignment(C_np)

            matched_indices.append(
                (
                    torch.tensor(pred_idx, dtype=torch.long, device=pred_masks.device),
                    torch.tensor(gt_idx, dtype=torch.long, device=pred_masks.device),
                )
            )

        return matched_indices

    def _dice_cost(self, pred_probs: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """
        Compute dice cost between predictions and GT.

        Args:
            pred_probs: [Q, N] predicted probabilities
            gt_masks: [G, N] GT masks

        Returns:
            cost: [Q, G] dice cost matrix
        """
        Q, N = pred_probs.shape
        pred_probs = pred_probs.unsqueeze(1)  # [Q, 1, N]
        gt_masks = gt_masks.unsqueeze(0)  # [1, G, N]

        intersection = (pred_probs * gt_masks).sum(dim=-1)  # [Q, G]
        union = pred_probs.sum(dim=-1) + gt_masks.sum(dim=-1)  # [Q, G]

        dice = 2 * intersection / (union + 1e-6)
        return 1 - dice

    def _compute_iou(self, pred_probs: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU matrix between predictions and GT.

        Args:
            pred_probs: [Q, N] predicted probabilities
            gt_masks: [G, N] GT masks

        Returns:
            iou: [Q, G] IoU matrix
        """
        Q, N = pred_probs.shape
        pred_probs = pred_probs.unsqueeze(1)  # [Q, 1, N]
        gt_masks = gt_masks.unsqueeze(0)  # [1, G, N]

        intersection = (pred_probs * gt_masks).sum(dim=-1)  # [Q, G]
        union = pred_probs.sum(dim=-1) + gt_masks.sum(dim=-1) - intersection  # [Q, G]

        iou = intersection / (union + 1e-6)
        return iou.clamp(0, 1)

    def _dice_cost_chunk(self, pred_probs: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """
        Compute dice cost for a chunk of predictions.

        Memory-efficient version that processes a subset of queries at a time.

        Args:
            pred_probs: [chunk, N] predicted probabilities for chunk of queries
            gt_masks: [G, N] GT masks (full)

        Returns:
            cost: [chunk, G] dice cost matrix for this chunk
        """
        chunk_size, N = pred_probs.shape
        # Expand for broadcasting: [chunk, G, N]
        pred_expanded = pred_probs.unsqueeze(1)  # [chunk, 1, N]
        gt_expanded = gt_masks.unsqueeze(0)  # [1, G, N]

        intersection = (pred_expanded * gt_expanded).sum(dim=-1)  # [chunk, G]
        union = pred_expanded.sum(dim=-1) + gt_expanded.sum(dim=-1)  # [chunk, G]

        dice = 2 * intersection / (union + 1e-6)
        return 1 - dice

    def _compute_iou_chunk(self, pred_probs: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU for a chunk of predictions.

        Memory-efficient version that processes a subset of queries at a time.

        Args:
            pred_probs: [chunk, N] predicted probabilities for chunk of queries
            gt_masks: [G, N] GT masks (full)

        Returns:
            iou: [chunk, G] IoU matrix for this chunk
        """
        chunk_size, N = pred_probs.shape
        # Expand for broadcasting: [chunk, G, N]
        pred_expanded = pred_probs.unsqueeze(1)  # [chunk, 1, N]
        gt_expanded = gt_masks.unsqueeze(0)  # [1, G, N]

        intersection = (pred_expanded * gt_expanded).sum(dim=-1)  # [chunk, G]
        union = pred_expanded.sum(dim=-1) + gt_expanded.sum(dim=-1) - intersection  # [chunk, G]

        iou = intersection / (union + 1e-6)
        return iou.clamp(0, 1)


class MaskLoss(nn.Module):
    """
    Mask prediction loss combining focal loss and dice loss.
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        matched_indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mask loss.

        Args:
            pred_masks: [B, Q, N] predicted mask logits
            gt_masks: List[Tensor] of [G_b, N] GT masks per batch
            matched_indices: Hungarian matching results

        Returns:
            Dictionary with focal_loss and dice_loss
        """
        device = pred_masks.device
        total_focal = torch.tensor(0.0, device=device)
        total_dice = torch.tensor(0.0, device=device)
        num_matched = 0

        for b, (pred_idx, gt_idx) in enumerate(matched_indices):
            if len(pred_idx) == 0:
                continue

            pred_m = pred_masks[b, pred_idx]  # [K, N]
            gt_m = gt_masks[b][gt_idx].to(device)  # [K, N]

            # Focal loss
            focal = self._focal_loss(pred_m, gt_m)
            total_focal = total_focal + focal.sum()

            # Dice loss
            dice = self._dice_loss(pred_m, gt_m)
            total_dice = total_dice + dice.sum()

            num_matched += len(pred_idx)

        if num_matched > 0:
            total_focal = total_focal / num_matched
            total_dice = total_dice / num_matched

        return {
            "mask_focal_loss": self.focal_weight * total_focal,
            "mask_dice_loss": self.dice_weight * total_dice,
            "mask_loss": self.focal_weight * total_focal + self.dice_weight * total_dice,
        }

    def _focal_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Binary focal loss."""
        p = pred.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(pred, gt, reduction="none")

        p_t = p * gt + (1 - p) * (1 - gt)
        alpha_t = self.focal_alpha * gt + (1 - self.focal_alpha) * (1 - gt)
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma

        focal_loss = focal_weight * ce_loss
        return focal_loss.mean(dim=-1)  # [K]

    def _dice_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Dice loss."""
        pred_probs = pred.sigmoid()
        intersection = (pred_probs * gt).sum(dim=-1)
        union = pred_probs.sum(dim=-1) + gt.sum(dim=-1)
        dice = 2 * intersection / (union + 1e-6)
        return 1 - dice  # [K]


class SpatialAffinityLoss(nn.Module):
    """
    Spatial Affinity Loss for encouraging spatially coherent mask predictions.

    This loss penalizes predictions where spatially adjacent points have
    inconsistent mask values, helping to eliminate isolated noisy predictions.

    L_affinity = Σ_i Σ_j∈N(i) w_ij * |mask_i - mask_j|

    Where:
    - N(i) is the K nearest neighbors of point i
    - w_ij = exp(-||pos_i - pos_j||² / σ²) is the spatial distance weight

    For large point clouds (100k+), we use downsampling for efficiency.
    """

    def __init__(
        self,
        k_neighbors: int = 16,
        sigma: float = 0.1,
        weight: float = 0.5,
        use_distance_weight: bool = True,
        max_points: int = 10000,  # Downsample if exceeds this
    ):
        """
        Args:
            k_neighbors: Number of nearest neighbors to consider
            sigma: Bandwidth for distance weighting (smaller = more local)
            weight: Loss weight
            use_distance_weight: Whether to weight by spatial distance
            max_points: Maximum points to use (downsample if larger)
        """
        super().__init__()
        self.k_neighbors = k_neighbors
        self.sigma = sigma
        self.weight = weight
        self.use_distance_weight = use_distance_weight
        self.max_points = max_points

    def forward(
        self,
        pred_masks: torch.Tensor,
        points: torch.Tensor,
        matched_indices: List[Tuple[torch.Tensor, torch.Tensor]],
        gt_masks: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute spatial affinity loss.

        Args:
            pred_masks: [B, Q, N] predicted mask logits
            points: [B, N, 3] point coordinates
            matched_indices: Hungarian matching results
            gt_masks: Optional GT masks for selective computation

        Returns:
            Dictionary with affinity_loss
        """
        device = pred_masks.device
        B, Q, N = pred_masks.shape

        total_loss = torch.tensor(0.0, device=device)
        num_matched = 0

        for b, (pred_idx, gt_idx) in enumerate(matched_indices):
            if len(pred_idx) == 0:
                continue

            # Get matched predictions [K, N]
            pred_m = pred_masks[b, pred_idx].sigmoid()  # [K, N]
            pts = points[b]  # [N, 3]

            # Use actual point count (may differ from pred_masks.shape[2] in some cases)
            N_actual = min(pred_m.shape[1], pts.shape[0])
            pred_m = pred_m[:, :N_actual]
            pts = pts[:N_actual]

            # Downsample for large point clouds
            if N_actual > self.max_points:
                sample_idx = torch.randperm(N_actual, device=device)[: self.max_points]
                pts_sampled = pts[sample_idx]  # [max_points, 3]
                pred_m_sampled = pred_m[:, sample_idx]  # [K, max_points]
            else:
                sample_idx = None
                pts_sampled = pts
                pred_m_sampled = pred_m

            N_sampled = pts_sampled.shape[0]
            k = min(self.k_neighbors, N_sampled - 1)

            # Compute KNN indices on sampled points
            knn_idx = self._compute_knn_simple(pts_sampled, k)  # [N_sampled, k]

            # Compute affinity loss for each matched query
            for ki in range(pred_m_sampled.shape[0]):
                mask_i = pred_m_sampled[ki]  # [N_sampled]

                # Get neighbor mask values [N_sampled, k]
                mask_j = mask_i[knn_idx]  # [N_sampled, k]

                # Compute difference
                diff = torch.abs(mask_i.unsqueeze(-1) - mask_j)  # [N_sampled, k]

                if self.use_distance_weight:
                    # Compute distance weights
                    pts_i = pts_sampled.unsqueeze(1)  # [N_sampled, 1, 3]
                    pts_j = pts_sampled[knn_idx]  # [N_sampled, k, 3]
                    dist_sq = ((pts_i - pts_j) ** 2).sum(dim=-1)  # [N_sampled, k]
                    weights = torch.exp(-dist_sq / (2 * self.sigma**2))  # [N_sampled, k]

                    # Weighted difference
                    weighted_diff = weights * diff
                    loss = weighted_diff.sum() / (weights.sum() + 1e-6)
                else:
                    loss = diff.mean()

                total_loss = total_loss + loss
                num_matched += 1

        if num_matched > 0:
            total_loss = total_loss / num_matched

        return {
            "affinity_loss": self.weight * total_loss,
        }

    def _compute_knn_simple(self, points: torch.Tensor, k: int) -> torch.Tensor:
        """
        Simple KNN computation for reasonably sized point clouds.

        Args:
            points: [N, 3] point coordinates (N should be <= max_points)
            k: Number of neighbors

        Returns:
            knn_idx: [N, k] indices of K nearest neighbors
        """
        N = points.shape[0]

        # For small enough point clouds, compute directly on GPU
        if N <= 5000:
            with torch.amp.autocast("cuda", enabled=False):
                dist = torch.cdist(
                    points.float().unsqueeze(0), points.float().unsqueeze(0)
                ).squeeze(0)
            dist.fill_diagonal_(float("inf"))
            _, knn_idx = dist.topk(k, dim=-1, largest=False)
            return knn_idx
        else:
            # Use CPU for medium-sized point clouds
            return self._chunked_knn_cpu(points, k)

    def _compute_knn(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute K nearest neighbors for each point.

        For large point clouds (100k+), we use random sampling to reduce computation.

        Args:
            points: [N, 3] point coordinates

        Returns:
            knn_idx: [N, k] indices of K nearest neighbors (or sampled subset)
        """
        N = points.shape[0]
        k = min(self.k_neighbors, N - 1)
        # For very large point clouds, use sampling-based approach
        if N > 30000:
            # Sample a subset of points for KNN computation
            # This is much faster and still effective for affinity loss
            return self._sampled_knn(points, k)
        elif N > 10000:
            # Chunked KNN for medium point clouds
            return self._chunked_knn_cpu(points, k)
        else:
            # Direct computation for smaller point clouds
            dist = torch.cdist(points.unsqueeze(0), points.unsqueeze(0)).squeeze(0)
            dist.fill_diagonal_(float("inf"))
            _, knn_idx = dist.topk(k, dim=-1, largest=False)
            return knn_idx

    def _sampled_knn(self, points: torch.Tensor, k: int, sample_size: int = 5000) -> torch.Tensor:
        """
        Compute approximate KNN using random sampling.

        For each point, we only compare with a random subset of points.
        This is much faster for large point clouds and still effective.

        Args:
            points: [N, 3] point coordinates
            k: Number of neighbors
            sample_size: Number of candidate points to sample

        Returns:
            knn_idx: [N, k] indices of K nearest neighbors
        """
        N = points.shape[0]
        device = points.device
        k = min(k, sample_size - 1)

        # Process in chunks to avoid OOM
        chunk_size = 2000
        knn_idx = torch.zeros(N, k, dtype=torch.long, device=device)

        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            chunk_points = points[i:end_i]  # [chunk, 3]
            chunk_n = end_i - i

            # Sample random candidate points (include nearby indices for locality)
            # Mix of random global samples and local neighborhood
            local_range = min(sample_size // 2, N)
            local_start = max(0, i - local_range // 2)
            local_end = min(N, i + chunk_n + local_range // 2)

            # Local indices
            local_indices = torch.arange(local_start, local_end, device=device)

            # Random global indices
            num_global = sample_size - len(local_indices)
            if num_global > 0:
                global_indices = torch.randperm(N, device=device)[:num_global]
                candidate_indices = torch.cat([local_indices, global_indices])
            else:
                candidate_indices = local_indices[:sample_size]

            candidate_indices = torch.unique(candidate_indices)
            candidate_points = points[candidate_indices]  # [S, 3]

            # Compute distances: [chunk, S]
            # Use float32 on CPU if GPU memory is tight
            with torch.amp.autocast("cuda", enabled=False):
                chunk_f = chunk_points.float()
                cand_f = candidate_points.float()
                dist = torch.cdist(chunk_f, cand_f)

            # Mask self-distances
            for j in range(chunk_n):
                self_idx = (candidate_indices == (i + j)).nonzero(as_tuple=True)[0]
                if len(self_idx) > 0:
                    dist[j, self_idx] = float("inf")

            # Get K nearest from candidates
            _, topk_in_candidates = dist.topk(k, dim=-1, largest=False)

            # Map back to original indices
            knn_idx[i:end_i] = candidate_indices[topk_in_candidates]

        return knn_idx

    def _chunked_knn_cpu(
        self, points: torch.Tensor, k: int, chunk_size: int = 2000
    ) -> torch.Tensor:
        """
        Compute KNN in chunks, using CPU for distance computation to avoid GPU OOM.

        Args:
            points: [N, 3] point coordinates
            k: Number of neighbors
            chunk_size: Size of each chunk

        Returns:
            knn_idx: [N, k] indices of K nearest neighbors
        """
        N = points.shape[0]
        device = points.device

        # Move to CPU for computation
        points_cpu = points.cpu().float()
        knn_idx = torch.zeros(N, k, dtype=torch.long)

        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            chunk_points = points_cpu[i:end_i]  # [chunk, 3]

            # Compute distances on CPU
            dist = torch.cdist(chunk_points.unsqueeze(0), points_cpu.unsqueeze(0)).squeeze(0)

            # Set self-distance to inf
            for j in range(end_i - i):
                dist[j, i + j] = float("inf")

            # Get K nearest neighbors
            _, chunk_knn = dist.topk(k, dim=-1, largest=False)
            knn_idx[i:end_i] = chunk_knn

        return knn_idx.to(device)


class ScoreLoss(nn.Module):
    """
    Score prediction loss with Quality Focal Loss (QFL) from Rank-DETR.

    Improvements over standard BCE:
    1. Quality Focal Loss: Focuses gradient on hard positives, reduces easy negative impact
    2. GIoU target: Provides gradient even when masks don't overlap (avoids dead gradient)
    3. Unmatched Query suppression: Train unmatched queries to have score=0

    QFL formula:
        loss = |pred - target|^beta * BCE(pred, target)

    GIoU target formula:
        target = clamp((GIoU + 1) / 2, 0, 1)
    """

    def __init__(
        self,
        weight: float = 2.0,
        use_quality_focal: bool = True,
        focal_beta: float = 2.0,
        use_giou_target: bool = True,
        use_unmatched_loss: bool = True,
        unmatched_weight: float = 0.5,
    ):
        """
        Args:
            weight: Loss weight (higher than before since QFL reduces magnitude)
            use_quality_focal: Use Quality Focal Loss instead of BCE
            focal_beta: Power for focal weight (similar to gamma in Focal Loss)
            use_giou_target: Use normalized GIoU as target instead of IoU
            use_unmatched_loss: Train unmatched queries to have score=0
            unmatched_weight: Weight for unmatched query loss (relative to matched)
        """
        super().__init__()
        self.weight = weight
        self.use_quality_focal = use_quality_focal
        self.focal_beta = focal_beta
        self.use_giou_target = use_giou_target
        self.use_unmatched_loss = use_unmatched_loss
        self.unmatched_weight = unmatched_weight

    def forward(
        self,
        pred_scores: torch.Tensor,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        matched_indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute score loss.

        Args:
            pred_scores: [B, Q] predicted score logits
            pred_masks: [B, Q, N] predicted mask logits
            gt_masks: List[Tensor] of [G_b, N] GT masks
            matched_indices: Hungarian matching results

        Returns:
            Dictionary with score_loss
        """
        device = pred_scores.device
        B, Q = pred_scores.shape

        total_matched_loss = torch.tensor(0.0, device=device)
        total_unmatched_loss = torch.tensor(0.0, device=device)
        num_matched = 0
        num_unmatched = 0

        for b, (pred_idx, gt_idx) in enumerate(matched_indices):
            # 1. Matched Query loss: score → IoU/GIoU
            if len(pred_idx) > 0:
                score_logits = pred_scores[b, pred_idx]  # [K] - logits
                pred_m = pred_masks[b, pred_idx].sigmoid()  # [K, N]
                gt_m = gt_masks[b][gt_idx].to(device)  # [K, N]

                # Compute target quality
                with torch.no_grad():
                    if self.use_giou_target:
                        target_quality = self._compute_giou_target(pred_m, gt_m)
                    else:
                        intersection = (pred_m * gt_m).sum(dim=-1)
                        union = pred_m.sum(dim=-1) + gt_m.sum(dim=-1) - intersection
                        target_quality = (intersection / (union + 1e-6)).clamp(0, 1)

                # Compute loss for matched queries
                if self.use_quality_focal:
                    loss = self._quality_focal_loss(score_logits, target_quality)
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        score_logits, target_quality, reduction="none"
                    )

                total_matched_loss = total_matched_loss + loss.sum()
                num_matched += len(pred_idx)

            # 2. Unmatched Query loss: score → 0 (suppress false positives)
            if self.use_unmatched_loss:
                matched_set = set(pred_idx.tolist()) if len(pred_idx) > 0 else set()
                unmatched_idx = [i for i in range(Q) if i not in matched_set]

                if len(unmatched_idx) > 0:
                    unmatched_scores = pred_scores[b, unmatched_idx]  # [U]
                    target_zeros = torch.zeros_like(unmatched_scores)

                    # Use same loss type for consistency
                    if self.use_quality_focal:
                        unmatched_loss = self._quality_focal_loss(unmatched_scores, target_zeros)
                    else:
                        unmatched_loss = F.binary_cross_entropy_with_logits(
                            unmatched_scores, target_zeros, reduction="none"
                        )

                    total_unmatched_loss = total_unmatched_loss + unmatched_loss.sum()
                    num_unmatched += len(unmatched_idx)

        # Average losses
        if num_matched > 0:
            total_matched_loss = total_matched_loss / num_matched
        if num_unmatched > 0:
            total_unmatched_loss = total_unmatched_loss / num_unmatched

        # Combine losses
        total_loss = total_matched_loss + self.unmatched_weight * total_unmatched_loss

        return {
            "score_loss": self.weight * total_loss,
            "score_matched_loss": total_matched_loss,
            "score_unmatched_loss": total_unmatched_loss,
        }

    def _quality_focal_loss(
        self,
        pred_logits: torch.Tensor,
        target_quality: torch.Tensor,
    ) -> torch.Tensor:
        """
        Quality Focal Loss from Rank-DETR.

        QFL = |pred_score - target_quality|^beta * BCE(pred_logits, target_quality)

        This focuses gradient on:
        - Hard positives (high target, low pred)
        - Reduces gradient for easy negatives (low target, low pred)

        Args:
            pred_logits: [K] predicted score logits
            target_quality: [K] target quality (IoU or GIoU)

        Returns:
            loss: [K] per-sample loss
        """
        pred_score = pred_logits.sigmoid()

        # Scale factor: |pred - target|^beta
        # For positives: penalizes if pred is far from target
        # For negatives (target~0): becomes |pred|^beta, penalizes false positives
        scale_factor = (pred_score - target_quality).abs().pow(self.focal_beta)

        # BCE loss (with logits for numerical stability)
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target_quality, reduction="none")

        return scale_factor * bce_loss

    def _compute_giou_target(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute normalized GIoU as target.

        For point cloud masks, we use a soft approximation:
        - IoU is computed as usual
        - For GIoU, we estimate the "enclosing" region based on mask coverage

        GIoU = IoU - (C - Union) / C
        where C is the enclosing region (all points covered by either mask)

        Normalized GIoU = (GIoU + 1) / 2, mapped to [0, 1]

        Args:
            pred_masks: [K, N] predicted mask probabilities
            gt_masks: [K, N] GT masks

        Returns:
            target: [K] normalized GIoU values in [0, 1]
        """
        # Soft IoU
        intersection = (pred_masks * gt_masks).sum(dim=-1)  # [K]
        pred_sum = pred_masks.sum(dim=-1)  # [K]
        gt_sum = gt_masks.sum(dim=-1)  # [K]
        union = pred_sum + gt_sum - intersection  # [K]

        iou = intersection / (union + 1e-6)

        # For point clouds, the "enclosing" region C is approximated as
        # the number of points that have non-zero probability in either mask
        # C = max(pred_sum, gt_sum) for simplicity, or use union-based estimate
        # Here we use: C = points where either pred > 0.1 or gt > 0
        pred_coverage = (pred_masks > 0.1).float()
        gt_coverage = gt_masks
        enclosing = ((pred_coverage + gt_coverage) > 0).float().sum(dim=-1)  # [K]

        # GIoU = IoU - (C - Union) / C
        # When masks are far apart, (C - Union) / C is large, making GIoU negative
        giou = iou - (enclosing - union) / (enclosing + 1e-6)

        # Normalize to [0, 1]: (GIoU + 1) / 2
        # GIoU range is [-1, 1], so normalized range is [0, 1]
        normalized_giou = ((giou + 1) / 2).clamp(0, 1)

        return normalized_giou


class CenterLoss(nn.Module):
    """
    Center Loss for supervising query positions to match GT part centers.

    This loss explicitly supervises query positions to converge to the
    geometric centers of their matched GT parts, rather than relying on
    implicit gradients from mask loss.

    L_center = (1/K) * Σ ||query_pos[i] - gt_center[σ(i)]||_p

    Where:
    - query_pos[i]: Predicted query position for matched query i
    - gt_center[σ(i)]: GT part center for the matched GT part
    - p: Norm type (1 for L1, 2 for L2)

    Benefits:
    - Faster convergence: Direct supervision on query positions
    - Better localization: Queries more accurately target parts
    - Improved motion prediction: Since origin = query_pos + offset
    """

    def __init__(
        self,
        weight: float = 0.5,
        loss_type: str = "l1",  # 'l1', 'l2', 'smooth_l1'
        normalize_by_scale: bool = False,  # Normalize by point cloud scale
    ):
        """
        Args:
            weight: Loss weight
            loss_type: Type of distance loss ('l1', 'l2', 'smooth_l1')
            normalize_by_scale: If True, normalize distance by point cloud bbox diagonal
        """
        super().__init__()
        self.weight = weight
        self.loss_type = loss_type
        self.normalize_by_scale = normalize_by_scale

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, Any],
        matched_indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute center loss for matched queries.

        Args:
            predictions: Must contain 'query_positions' [B, Q, 3]
            targets: Must contain 'points' [B, N, 3] and 'group_ids' [B, N]
            matched_indices: List of (pred_idx, gt_idx) from Hungarian matcher

        Returns:
            Dictionary with:
            - center_loss: Weighted loss value
            - center_dist: Mean distance (unweighted, for monitoring)
        """
        device = predictions["query_positions"].device

        # Check required inputs
        if "query_positions" not in predictions:
            return {
                "center_loss": torch.tensor(0.0, device=device),
                "center_dist": torch.tensor(0.0, device=device),
            }

        query_positions = predictions["query_positions"]  # [B, Q, 3]
        points = targets.get("points")
        group_ids = targets.get("group_ids")

        if points is None or group_ids is None:
            return {
                "center_loss": torch.tensor(0.0, device=device),
                "center_dist": torch.tensor(0.0, device=device),
            }

        total_loss = torch.tensor(0.0, device=device)
        total_dist = torch.tensor(0.0, device=device)
        num_matched = 0

        for b, (pred_idx, gt_idx) in enumerate(matched_indices):
            if len(pred_idx) == 0:
                continue

            # Get matched query positions [K, 3]
            pred_centers = query_positions[b, pred_idx]

            # Compute GT part centers [K, 3]
            pts_b = points[b]  # [N, 3]
            gids_b = group_ids[b]  # [N]

            gt_centers_list = []
            valid_mask = []

            for g_idx in gt_idx:
                g = g_idx.item() if isinstance(g_idx, torch.Tensor) else g_idx
                part_mask = gids_b == g

                if part_mask.sum() > 0:
                    # Compute geometric center of the part
                    gt_center = pts_b[part_mask].mean(dim=0)  # [3]
                    gt_centers_list.append(gt_center)
                    valid_mask.append(True)
                else:
                    # Part has no points (shouldn't happen, but handle gracefully)
                    gt_centers_list.append(torch.zeros(3, device=device))
                    valid_mask.append(False)

            if not any(valid_mask):
                continue

            gt_centers = torch.stack(gt_centers_list)  # [K, 3]
            valid_mask = torch.tensor(valid_mask, device=device)

            # Filter to valid pairs only
            pred_centers_valid = pred_centers[valid_mask]
            gt_centers_valid = gt_centers[valid_mask]

            # Compute scale for normalization (optional)
            if self.normalize_by_scale:
                bbox_min = pts_b.min(dim=0)[0]
                bbox_max = pts_b.max(dim=0)[0]
                scale = (bbox_max - bbox_min).norm() + 1e-6  # bbox diagonal
            else:
                scale = 1.0

            # Compute distance
            diff = pred_centers_valid - gt_centers_valid  # [K', 3]

            if self.loss_type == "l1":
                dist = diff.abs().sum(dim=-1)  # [K']
            elif self.loss_type == "l2":
                dist = (diff**2).sum(dim=-1)  # [K']
            elif self.loss_type == "smooth_l1":
                dist = F.smooth_l1_loss(pred_centers_valid, gt_centers_valid, reduction="none").sum(
                    dim=-1
                )  # [K']
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            # Normalize by scale
            if self.normalize_by_scale:
                dist = dist / scale

            total_loss = total_loss + dist.sum()
            total_dist = total_dist + diff.abs().sum(dim=-1).sum()  # Always L1 for monitoring
            num_matched += valid_mask.sum().item()

        # Average over matched pairs
        if num_matched > 0:
            total_loss = total_loss / num_matched
            total_dist = total_dist / num_matched
        else:
            total_loss = torch.tensor(0.0, device=device)
            total_dist = torch.tensor(0.0, device=device)

        return {
            "center_loss": self.weight * total_loss,
            "center_dist": total_dist,  # Unweighted, for monitoring
        }


class MatchedPartClassLoss(nn.Module):
    """
    Part Classification Loss for matched queries.

    Computes cross-entropy loss between predicted part class logits
    and ground truth part class labels, only for matched queries.

    Args:
        num_classes: Number of part classes (default: 18)
        use_class_weights: Use class weights for imbalanced data
        weight: Loss weight
        label_smoothing: Label smoothing factor
    """

    def __init__(
        self,
        num_classes: int = NUM_PART_CLASSES,
        use_class_weights: bool = True,
        weight: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_class_weights = use_class_weights
        self.weight = weight
        self.label_smoothing = label_smoothing

        if use_class_weights and SEMANTIC_FUSION_AVAILABLE and CLASS_WEIGHTS is not None:
            self.register_buffer("class_weights", CLASS_WEIGHTS.clone())
        else:
            self.class_weights = None

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, Any],
        matched_indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute part classification loss for matched queries.

        Args:
            predictions: Must contain 'part_class_logits' [B, Q, num_classes]
            targets: Must contain 'gt_part_class_labels' List[Tensor] [G_b] for each batch
            matched_indices: List of (pred_idx, gt_idx) tuples for each batch

        Returns:
            Dictionary with:
            - part_class_loss: Weighted loss value
            - part_class_accuracy: Classification accuracy on matched queries
        """
        device = predictions["part_class_logits"].device

        # Check if predictions have part_class_logits
        if "part_class_logits" not in predictions:
            return {
                "part_class_loss": torch.tensor(0.0, device=device),
                "part_class_accuracy": torch.tensor(0.0, device=device),
            }

        part_class_logits = predictions["part_class_logits"]  # [B, Q, C]
        gt_part_class_labels = targets.get("gt_part_class_labels", None)

        if gt_part_class_labels is None:
            return {
                "part_class_loss": torch.tensor(0.0, device=device),
                "part_class_accuracy": torch.tensor(0.0, device=device),
            }

        total_loss = torch.tensor(0.0, device=device)
        total_correct = 0
        total_valid = 0

        for b, (pred_idx, gt_idx) in enumerate(matched_indices):
            if len(pred_idx) == 0:
                continue

            # Get matched predictions [K, C]
            pred_logits = part_class_logits[b, pred_idx]

            # Get GT labels [K]
            gt_labels = gt_part_class_labels[b][gt_idx].to(device)

            # Filter out unknown labels (label == -1)
            valid_mask = gt_labels >= 0

            if valid_mask.sum() == 0:
                continue

            valid_logits = pred_logits[valid_mask]  # [K', C]
            valid_labels = gt_labels[valid_mask]  # [K']

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                valid_logits,
                valid_labels,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
                reduction="mean",
            )
            total_loss = total_loss + loss * valid_mask.sum()

            # Compute accuracy
            pred_classes = valid_logits.argmax(dim=-1)
            total_correct += (pred_classes == valid_labels).sum().item()
            total_valid += valid_mask.sum().item()

        # Average loss
        if total_valid > 0:
            total_loss = total_loss / total_valid
            accuracy = total_correct / total_valid
        else:
            accuracy = 0.0

        return {
            "part_class_loss": self.weight * total_loss,
            "part_class_accuracy": torch.tensor(accuracy, device=device),
        }


class CombinedLoss(nn.Module):
    """
    Combined loss for part segmentation and motion prediction.

    L_total = L_seg + L_motion + L_category + L_affinity + L_limit + L_part_class

    Where:
    - L_seg = L_mask (focal + dice) + L_score
    - L_motion = L_type + L_direction + L_origin
    - L_category = CrossEntropy for category classification
    - L_affinity = Spatial affinity loss (optional, for coherent masks)
    - L_limit = Motion limit prediction (center-span, P and R only)
    - L_part_class = Part classification loss (18 classes, for semantic fusion)

    Motion Types (4-class):
    - 0: Fixed (F) - no motion
    - 1: Prismatic (P) - translation with limit
    - 2: Revolute (R) - rotation with limit
    - 3: Continuous (C) - unlimited rotation (no limit prediction)

    Improvements from Rank-DETR:
    - High-order matching: Cost = -score * IoU^alpha (penalizes high-score low-quality)
    - Quality Focal Loss: Focuses on hard positives
    - GIoU target: Provides gradient even when masks don't overlap

    Spatial Affinity Loss:
    - Encourages spatially adjacent points to have similar mask predictions
    - Helps eliminate isolated noisy predictions (fine-grained noise)
    """

    def __init__(
        self,
        # Mask loss weights
        mask_focal_weight: float = 0.5,
        mask_dice_weight: float = 2.0,
        score_weight: float = 2.0,  # Increased from 0.7 for QFL
        # Motion loss weights
        motion_type_weight: float = 1.0,
        motion_direction_weight: float = 1.0,
        motion_origin_weight: float = 0.5,
        motion_anchor_weight: float = 0.5,
        use_anchor_loss: bool = True,
        # Category loss weight
        category_weight: float = 0.3,
        # Motion limit loss weight
        motion_limit_weight: float = 0.5,
        # Overall task weights
        seg_weight: float = 1.0,
        motion_weight: float = 1.0,
        # Options
        ignore_fixed_for_axis: bool = True,
        motion_warmup_epochs: int = 5,
        num_categories: int = 46,  # Updated to 46 categories
        # Category class balancing
        category_class_counts: Optional[List[int]] = None,
        # Motion type class balancing (for imbalanced F/P/R/C distribution)
        motion_type_class_counts: Optional[List[int]] = None,
        # Rank-DETR improvements
        use_high_order_matching: bool = True,
        iou_power_alpha: float = 3.0,
        use_quality_focal_loss: bool = True,
        quality_focal_beta: float = 2.0,
        use_giou_target: bool = True,
        # Unmatched Query suppression (no background)
        use_unmatched_loss: bool = True,
        unmatched_weight: float = 0.5,
        # Spatial Affinity Loss
        use_affinity_loss: bool = True,
        affinity_k: int = 16,
        affinity_sigma: float = 0.1,
        affinity_weight: float = 0.5,
        # Motion Limit Loss
        use_limit_loss: bool = True,
        limit_loss_type: str = "l1",
        # Matcher chunked computation (for OOM prevention)
        matcher_chunk_size: int = 10,
        # Part Classification Loss (for semantic fusion)
        use_part_class_loss: bool = False,
        part_class_weight: float = 0.5,
        part_class_balance: bool = True,
        # Center Loss (for query position supervision)
        use_center_loss: bool = False,
        center_weight: float = 0.5,
        center_loss_type: str = "l1",
        # Iterative Deep Supervision (for iterative semantic fusion)
        use_iterative_deep_supervision: bool = False,
        iterative_deep_supervision_weight: float = 0.1,
    ):
        """
        Args:
            mask_focal_weight: Weight for mask focal loss
            mask_dice_weight: Weight for mask dice loss
            score_weight: Weight for score loss (2.0 recommended with QFL)
            motion_type_weight: Weight for motion type loss
            motion_direction_weight: Weight for motion direction loss
            motion_origin_weight: Weight for motion origin loss (point-to-line constraint)
            motion_anchor_weight: Weight for motion anchor loss (proximity to projected origin)
            use_anchor_loss: Enable anchor loss to constrain origin near part center
            category_weight: Weight for category classification loss
            motion_limit_weight: Weight for motion limit loss (P and R only)
            seg_weight: Overall weight for segmentation losses
            motion_weight: Overall weight for motion losses
            ignore_fixed_for_axis: Skip axis losses for Fixed parts
            motion_warmup_epochs: Epochs to warmup motion loss
            num_categories: Number of object categories (46 for full PartNet-Mobility)
            category_class_counts: List of sample counts per category for class balancing.
                                   If provided, auto-computes weights as: total / (num_classes * count_i)
            motion_type_class_counts: List of sample counts per motion type [F, P, R, C] for class balancing.
                                      If provided, auto-computes weights to balance F/P/R/C distribution.
                                      This helps improve Prismatic prediction which is often underrepresented.
            use_high_order_matching: Use IoU^alpha in matching cost (Rank-DETR)
            iou_power_alpha: Power for IoU in matching (2~4 recommended)
            use_quality_focal_loss: Use Quality Focal Loss for score
            quality_focal_beta: Beta for QFL (similar to gamma)
            use_giou_target: Use normalized GIoU as score target
            use_unmatched_loss: Train unmatched queries to have score=0 (no background)
            unmatched_weight: Weight for unmatched query loss relative to matched
            use_affinity_loss: Enable spatial affinity loss for coherent masks
            affinity_k: Number of nearest neighbors for affinity loss
            affinity_sigma: Bandwidth for distance weighting in affinity loss
            affinity_weight: Weight for affinity loss
            use_limit_loss: Enable motion limit prediction loss
            limit_loss_type: Loss type for limit prediction ('l1' or 'smooth_l1')
            matcher_chunk_size: Number of queries to process at once in Hungarian matcher
                                cost computation. Smaller = less memory, larger = faster.
                                Default 10 works well for 100k points with 100+ GT parts.
            use_part_class_loss: Enable part classification loss (for semantic fusion)
            part_class_weight: Weight for part classification loss
            part_class_balance: Use class weights for imbalanced data
            use_center_loss: Enable center loss for query position supervision
            center_weight: Weight for center loss
            center_loss_type: Type of distance loss ('l1', 'l2', 'smooth_l1')
            use_iterative_deep_supervision: Enable deep supervision for iterative semantic fusion
            iterative_deep_supervision_weight: Weight for intermediate layer losses (final=1.0)
        """
        super().__init__()

        self.seg_weight = seg_weight
        self.motion_weight = motion_weight
        self.motion_warmup_epochs = motion_warmup_epochs
        self.use_affinity_loss = use_affinity_loss
        self.use_limit_loss = use_limit_loss
        self.use_part_class_loss = use_part_class_loss
        self.use_center_loss = use_center_loss
        self.use_iterative_deep_supervision = use_iterative_deep_supervision
        self.iterative_deep_supervision_weight = iterative_deep_supervision_weight

        # Hungarian matcher with high-order matching and chunked computation
        self.matcher = HungarianMatcher(
            cost_mask=1.0,
            cost_dice=1.0,
            cost_score=1.0,
            use_high_order_matching=use_high_order_matching,
            iou_power_alpha=iou_power_alpha,
            chunk_size=matcher_chunk_size,
        )

        # Mask loss
        self.mask_loss = MaskLoss(
            focal_weight=mask_focal_weight,
            dice_weight=mask_dice_weight,
        )

        # Spatial Affinity loss
        if use_affinity_loss:
            self.affinity_loss = SpatialAffinityLoss(
                k_neighbors=affinity_k,
                sigma=affinity_sigma,
                weight=affinity_weight,
            )
        else:
            self.affinity_loss = None

        # Score loss with Quality Focal Loss, GIoU target, and unmatched suppression
        self.score_loss = ScoreLoss(
            weight=score_weight,
            use_quality_focal=use_quality_focal_loss,
            focal_beta=quality_focal_beta,
            use_giou_target=use_giou_target,
            use_unmatched_loss=use_unmatched_loss,
            unmatched_weight=unmatched_weight,
        )

        # Compute motion type class weights if counts provided
        motion_type_class_weights = None
        if motion_type_class_counts is not None:
            motion_type_class_weights = self._compute_motion_type_weights(motion_type_class_counts)

        # Motion loss with anchor loss support and optional class balancing
        self.motion_loss = MotionPredictionLoss(
            type_weight=motion_type_weight,
            direction_weight=motion_direction_weight,
            origin_weight=motion_origin_weight,
            anchor_weight=motion_anchor_weight,
            class_weights=motion_type_class_weights,  # For F/P/R/C balancing
            ignore_fixed_for_axis=ignore_fixed_for_axis,
            use_anchor_loss=use_anchor_loss,
        )

        # Category loss with optional class balancing
        self.category_loss = CategoryLoss(
            num_categories=num_categories,
            class_counts=category_class_counts,
            weight=category_weight,
        )

        # Motion Limit loss (center-span parameterization, P and R only)
        if use_limit_loss:
            self.limit_loss = LimitLoss(
                weight=motion_limit_weight,
                loss_type=limit_loss_type,
            )
        else:
            self.limit_loss = None

        # Part Classification loss (for semantic fusion)
        if use_part_class_loss:
            self.part_class_loss = MatchedPartClassLoss(
                num_classes=NUM_PART_CLASSES,
                use_class_weights=part_class_balance,
                weight=part_class_weight,
            )
        else:
            self.part_class_loss = None

        # Center Loss (for query position supervision)
        if use_center_loss:
            self.center_loss_module = CenterLoss(
                weight=center_weight,
                loss_type=center_loss_type,
            )
        else:
            self.center_loss_module = None

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, Any],
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            predictions: Model outputs containing:
                - mask_logits: [B, Q, N]
                - scores: [B, Q]
                - category_logits: [B, num_categories]
                - motion_type_logits: [B, Q, 4] (F=0, P=1, R=2, C=3)
                - axis_direction: [B, Q, 3]
                - axis_origin: [B, Q, 3]
                - revolute_limit: [B, Q, 2] (center, span) for R type
                - prismatic_limit: [B, Q, 2] (center, span) for P type
            targets: Ground truth containing:
                - group_ids: [B, N] or gt_masks: List[Tensor]
                - categories: [B] category indices
                - gt_motion_types: List[Tensor] (0=F, 1=P, 2=R, 3=C)
                - gt_axis_directions: List[Tensor]
                - gt_axis_positions: List[Tensor]
                - gt_motion_limits: List[Tensor] [G, 2] (min, max) limits
            epoch: Current epoch (for motion warmup)

        Returns:
            Dictionary with all loss values and total loss
        """
        device = predictions["mask_logits"].device

        # Build GT masks from group_ids if needed
        if "gt_masks" not in targets:
            gt_masks = self._build_gt_masks(targets["group_ids"])
            targets["gt_masks"] = gt_masks

        # Hungarian matching
        matched_indices = self.matcher(predictions, targets)

        # Initialize loss dict
        losses = {}

        # 1. Mask loss
        mask_losses = self.mask_loss(
            predictions["mask_logits"],
            targets["gt_masks"],
            matched_indices,
        )
        losses.update(mask_losses)

        # 2. Score loss
        score_losses = self.score_loss(
            predictions["scores"],
            predictions["mask_logits"],
            targets["gt_masks"],
            matched_indices,
        )
        losses.update(score_losses)

        # 3. Category loss
        category_losses = self.category_loss(predictions, targets)
        losses.update(category_losses)

        # 4. Motion loss (with warmup)
        motion_scale = self._get_motion_scale(epoch)
        motion_losses = self.motion_loss(predictions, targets, matched_indices)

        # Save raw motion losses (before scaling) for monitoring true performance
        for key in list(motion_losses.keys()):
            if key.startswith("motion_") and "num" not in key:
                losses[f"{key}_raw"] = motion_losses[key].clone()

        # Scale motion losses
        for key in motion_losses:
            if key.startswith("motion_") and "num" not in key:
                motion_losses[key] = motion_losses[key] * motion_scale
        losses.update(motion_losses)

        # 5. Spatial Affinity loss (for coherent masks)
        if self.use_affinity_loss and self.affinity_loss is not None:
            # Need points for KNN computation
            points = targets.get("points")
            if points is not None:
                affinity_losses = self.affinity_loss(
                    predictions["mask_logits"],
                    points,
                    matched_indices,
                )
                losses.update(affinity_losses)
            else:
                losses["affinity_loss"] = torch.tensor(0.0, device=device)
        else:
            losses["affinity_loss"] = torch.tensor(0.0, device=device)

        # 6. Motion Limit loss (P and R types only)
        if self.use_limit_loss and self.limit_loss is not None:
            # Check if gt_motion_limits exists in targets
            if "gt_motion_limits" in targets:
                limit_losses = self.limit_loss(predictions, targets, matched_indices)
                # Save raw limit losses (before scaling) for monitoring
                for key in list(limit_losses.keys()):
                    if key.startswith("limit_") and "num" not in key:
                        losses[f"{key}_raw"] = limit_losses[key].clone()
                # Scale limit loss with motion warmup
                for key in limit_losses:
                    if key.startswith("limit_") and "num" not in key:
                        limit_losses[key] = limit_losses[key] * motion_scale
                losses.update(limit_losses)
            else:
                losses["limit_loss"] = torch.tensor(0.0, device=device)
                losses["revolute_limit_loss"] = torch.tensor(0.0, device=device)
                losses["prismatic_limit_loss"] = torch.tensor(0.0, device=device)
        else:
            losses["limit_loss"] = torch.tensor(0.0, device=device)
            losses["revolute_limit_loss"] = torch.tensor(0.0, device=device)
            losses["prismatic_limit_loss"] = torch.tensor(0.0, device=device)

        # 7. Part Classification loss (for semantic fusion)
        if self.use_part_class_loss and self.part_class_loss is not None:
            # Check if part_class_logits exists in predictions
            if "part_class_logits" in predictions:
                part_class_losses = self.part_class_loss(predictions, targets, matched_indices)
                losses.update(part_class_losses)
            else:
                losses["part_class_loss"] = torch.tensor(0.0, device=device)
                losses["part_class_accuracy"] = torch.tensor(0.0, device=device)
        else:
            losses["part_class_loss"] = torch.tensor(0.0, device=device)
            losses["part_class_accuracy"] = torch.tensor(0.0, device=device)

        # 8. Center Loss (for query position supervision)
        if self.use_center_loss and self.center_loss_module is not None:
            # Check if query_positions exists in predictions and points in targets
            if "query_positions" in predictions and "points" in targets:
                center_losses = self.center_loss_module(predictions, targets, matched_indices)
                losses.update(center_losses)
            else:
                losses["center_loss"] = torch.tensor(0.0, device=device)
                losses["center_dist"] = torch.tensor(0.0, device=device)
        else:
            losses["center_loss"] = torch.tensor(0.0, device=device)
            losses["center_dist"] = torch.tensor(0.0, device=device)

        # 9. Iterative Deep Supervision Loss (for iterative semantic fusion)
        if self.use_iterative_deep_supervision and "iterative_aux_outputs" in predictions:
            iterative_aux_outputs = predictions["iterative_aux_outputs"]
            gt_part_class_labels = targets.get("gt_part_class_labels", None)

            if iterative_aux_outputs and gt_part_class_labels is not None:
                iterative_ds_loss = self._compute_iterative_deep_supervision_loss(
                    iterative_aux_outputs=iterative_aux_outputs,
                    gt_part_class_labels=gt_part_class_labels,
                    matched_indices=matched_indices,
                    intermediate_weight=self.iterative_deep_supervision_weight,
                    device=device,
                )
                losses["iterative_ds_loss"] = iterative_ds_loss
            else:
                losses["iterative_ds_loss"] = torch.tensor(0.0, device=device)
        else:
            losses["iterative_ds_loss"] = torch.tensor(0.0, device=device)

        # Compute total loss
        seg_loss = losses["mask_loss"] + losses["score_loss"]
        motion_loss = losses["motion_loss"]
        category_loss = losses["category_loss"]
        affinity_loss = losses["affinity_loss"]
        limit_loss = losses["limit_loss"]
        part_class_loss = losses["part_class_loss"]
        center_loss = losses["center_loss"]
        iterative_ds_loss = losses["iterative_ds_loss"]

        total_loss = (
            self.seg_weight * seg_loss
            + self.motion_weight * motion_loss
            + category_loss
            + affinity_loss  # Already weighted in SpatialAffinityLoss
            + limit_loss  # Already weighted in LimitLoss
            + part_class_loss  # Already weighted in MatchedPartClassLoss
            + center_loss  # Already weighted in CenterLoss
            + iterative_ds_loss  # Already weighted with intermediate_weight
        )

        losses["seg_loss"] = seg_loss
        losses["total_loss"] = total_loss
        losses["motion_scale"] = torch.tensor(motion_scale, device=device)

        return losses

    def _compute_motion_type_weights(self, class_counts: List[int]) -> torch.Tensor:
        """
        Compute class weights for motion type classification.

        Uses inverse frequency weighting: weight_i = total / (num_classes * count_i)
        This balances the loss contribution from minority classes (Prismatic, Continuous).

        Args:
            class_counts: List of [count_F, count_P, count_R, count_C] sample counts

        Returns:
            Tensor of class weights [4]
        """
        total = sum(class_counts)
        num_classes = len(class_counts)

        weights = []
        for i, count in enumerate(class_counts):
            if count > 0:
                w = total / (num_classes * count)
            else:
                w = 1.0  # Default weight for classes with no samples
            weights.append(w)

        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        # Print computed weights for debugging
        motion_type_names = ["Fixed", "Prismatic", "Revolute", "Continuous"]
        print("Motion Type Class Balancing enabled:")
        print(f"  Total samples: {total}")
        for i, (name, cnt, w) in enumerate(zip(motion_type_names, class_counts, weights)):
            print(f"  {name}({i}): count={cnt}, weight={w:.3f}")

        return weights_tensor

    def _build_gt_masks(self, group_ids: torch.Tensor) -> List[torch.Tensor]:
        """Build GT masks from group_ids."""
        B = group_ids.shape[0]
        gt_masks = []

        for b in range(B):
            gids = group_ids[b]  # [N]
            unique_ids = torch.unique(gids)
            unique_ids = unique_ids[unique_ids >= 0]

            if len(unique_ids) == 0:
                masks = torch.zeros(0, gids.shape[0], device=gids.device)
            else:
                masks = (gids.unsqueeze(0) == unique_ids.unsqueeze(1)).float()

            gt_masks.append(masks)

        return gt_masks

    def _get_motion_scale(self, epoch: int) -> float:
        """Get motion loss scale based on warmup."""
        if self.motion_warmup_epochs <= 0:
            return 1.0
        if epoch >= self.motion_warmup_epochs:
            return 1.0
        return epoch / self.motion_warmup_epochs

    def _compute_iterative_deep_supervision_loss(
        self,
        iterative_aux_outputs: List[Dict],
        gt_part_class_labels: List[torch.Tensor],
        matched_indices: List[Tuple[torch.Tensor, torch.Tensor]],
        intermediate_weight: float = 0.1,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Compute deep supervision loss for iterative semantic fusion.

        Uses the same matching as the final layer (from Hungarian matcher)
        to supervise intermediate layer part classifications.

        Args:
            iterative_aux_outputs: List of auxiliary outputs from each layer,
                                   each containing 'class_logits' [B, Q, C]
            gt_part_class_labels: [B] list of GT part class labels [G_b]
            matched_indices: List of (pred_idx, gt_idx) tuples for each batch
            intermediate_weight: Weight for each layer's loss
            device: Target device

        Returns:
            total_loss: Weighted sum of intermediate layer losses
        """
        if device is None:
            device = gt_part_class_labels[0].device if gt_part_class_labels else torch.device("cpu")

        total_loss = torch.tensor(0.0, device=device)
        num_valid_layers = 0

        for aux_out in iterative_aux_outputs:
            if aux_out is None:
                continue

            class_logits = aux_out.get("class_logits", None)
            if class_logits is None:
                continue

            # class_logits: [B, Q, C]
            B = class_logits.shape[0]
            layer_loss = torch.tensor(0.0, device=device)
            valid_samples = 0

            for b in range(B):
                if matched_indices[b] is None or len(matched_indices[b][0]) == 0:
                    continue

                pred_idx, gt_idx = matched_indices[b]

                # Get matched query logits
                matched_logits = class_logits[b, pred_idx]  # [K, C]

                # Get GT labels
                gt_labels = gt_part_class_labels[b][gt_idx].to(device)  # [K]

                # Filter out invalid labels (label == -1)
                valid_mask = gt_labels >= 0
                if valid_mask.sum() == 0:
                    continue

                valid_logits = matched_logits[valid_mask]
                valid_targets = gt_labels[valid_mask]

                # Compute cross-entropy loss
                sample_loss = F.cross_entropy(
                    valid_logits,
                    valid_targets,
                    reduction="mean",
                )

                layer_loss = layer_loss + sample_loss
                valid_samples += 1

            if valid_samples > 0:
                layer_loss = layer_loss / valid_samples
                total_loss = total_loss + intermediate_weight * layer_loss
                num_valid_layers += 1

        return total_loss
