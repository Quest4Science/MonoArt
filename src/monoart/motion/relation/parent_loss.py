"""
Parent Prediction Loss for Kinematic Tree Structure.

Handles:
1. Building GT targets from Hungarian matcher results
2. Masked loss for unmatched parent nodes (edge case)
3. Cross-entropy loss with accuracy tracking
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class ParentPredictionLoss(nn.Module):
    """
    Loss function for parent-child relationship prediction.

    Key considerations:
    1. Only compute loss for matched queries (from Hungarian matcher)
    2. Mask out queries whose GT parent is not matched (parent missed)
    3. Root token index = N (last column in logits)
    """

    def __init__(
        self,
        weight: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            weight: Loss weight multiplier
            label_smoothing: Label smoothing factor for CE loss
        """
        super().__init__()
        self.weight = weight
        self.ce_loss = nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)

    def forward(
        self,
        parent_logits: torch.Tensor,
        matched_indices: List[Tuple[torch.Tensor, torch.Tensor]],
        gt_parent_info: List[Dict[int, int]],
        gt_link_ids: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute parent prediction loss.

        Args:
            parent_logits: [B, N, N+1] parent prediction logits
            matched_indices: List of (pred_idx, gt_idx) tuples from Hungarian matcher
                pred_idx: [M] indices of matched predictions
                gt_idx: [M] indices of matched GT parts
            gt_parent_info: List of dicts {link_id: parent_link_id}
                parent_link_id = -1 means root/base
            gt_link_ids: List of [G] tensors with GT link IDs

        Returns:
            Dictionary with:
            - parent_loss: Weighted average loss
            - parent_accuracy: Prediction accuracy
            - parent_num_valid: Number of valid samples
        """
        device = parent_logits.device
        B, N, _ = parent_logits.shape

        total_loss = torch.tensor(0.0, device=device)
        total_correct = 0
        total_valid = 0

        for b, (pred_idx, gt_idx) in enumerate(matched_indices):
            if len(pred_idx) == 0:
                continue

            # Build mappings for this sample
            # gt_link_id -> query_idx
            link_to_query = {}
            for q_idx, g_idx in zip(pred_idx.tolist(), gt_idx.tolist()):
                link_id = gt_link_ids[b][g_idx].item()
                link_to_query[link_id] = q_idx

            parent_info = gt_parent_info[b]

            # For each matched query, compute loss
            for q_idx, g_idx in zip(pred_idx.tolist(), gt_idx.tolist()):
                link_id = gt_link_ids[b][g_idx].item()

                # Get GT parent link_id
                if link_id not in parent_info:
                    # No parent info for this link, skip
                    continue

                gt_parent_link_id = parent_info[link_id]

                # Determine target index
                if gt_parent_link_id == -1:
                    # Connected to root/base
                    target = N  # Root token index
                elif gt_parent_link_id in link_to_query:
                    # Parent is matched to a query
                    target = link_to_query[gt_parent_link_id]
                else:
                    # Parent is NOT matched (missed detection)
                    # Skip this sample - we can't train on incorrect targets
                    continue

                # Get logits for this query
                logits = parent_logits[b, q_idx]  # [N+1]
                target_tensor = torch.tensor([target], device=device, dtype=torch.long)

                # Compute loss
                loss = self.ce_loss(logits.unsqueeze(0), target_tensor)
                total_loss = total_loss + loss.squeeze()

                # Compute accuracy
                pred = logits.argmax().item()
                if pred == target:
                    total_correct += 1
                total_valid += 1

        # Average loss
        if total_valid > 0:
            avg_loss = total_loss / total_valid
            accuracy = total_correct / total_valid
        else:
            # Keep the zero attached to the graph so DDP sees the parent head.
            avg_loss = parent_logits.sum() * 0.0
            accuracy = 0.0

        return {
            "parent_loss": self.weight * avg_loss,
            "parent_accuracy": torch.tensor(accuracy, device=device),
            "parent_num_valid": torch.tensor(total_valid, device=device),
        }


def build_parent_targets(
    matched_indices: List[Tuple[torch.Tensor, torch.Tensor]],
    gt_parent_info: List[Dict[int, int]],
    gt_link_ids: List[torch.Tensor],
    num_queries: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build target tensors for parent prediction.

    This is a batch-level version for efficient training.

    Args:
        matched_indices: List of (pred_idx, gt_idx) tuples
        gt_parent_info: List of {link_id: parent_link_id} dicts
        gt_link_ids: List of GT link ID tensors
        num_queries: Number of queries (N)
        device: Target device

    Returns:
        targets: [B, N] parent indices (N = root, -1 = invalid)
        mask: [B, N] valid query mask (1 = valid, 0 = invalid)
    """
    B = len(matched_indices)
    N = num_queries

    targets = torch.full((B, N), -1, dtype=torch.long, device=device)
    mask = torch.zeros((B, N), dtype=torch.bool, device=device)

    for b, (pred_idx, gt_idx) in enumerate(matched_indices):
        if len(pred_idx) == 0:
            continue

        # Build link_id -> query_idx mapping
        link_to_query = {}
        for q_idx, g_idx in zip(pred_idx.tolist(), gt_idx.tolist()):
            link_id = gt_link_ids[b][g_idx].item()
            link_to_query[link_id] = q_idx

        parent_info = gt_parent_info[b]

        for q_idx, g_idx in zip(pred_idx.tolist(), gt_idx.tolist()):
            link_id = gt_link_ids[b][g_idx].item()

            if link_id not in parent_info:
                continue

            gt_parent_link_id = parent_info[link_id]

            if gt_parent_link_id == -1:
                # Root
                targets[b, q_idx] = N
                mask[b, q_idx] = True
            elif gt_parent_link_id in link_to_query:
                # Matched parent
                targets[b, q_idx] = link_to_query[gt_parent_link_id]
                mask[b, q_idx] = True
            # else: parent not matched, leave as invalid

    return targets, mask
