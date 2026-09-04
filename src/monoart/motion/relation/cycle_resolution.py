"""
Cycle Resolution for Parent Prediction.

Ensures the predicted parent-child relationships form a valid tree
(no cycles, all nodes connected to root).

Algorithm:
1. Rank every valid child-parent edge globally
2. Select the strongest edges that do not introduce a cycle
3. Connect any remaining valid nodes to the root
"""

from typing import List, Optional

import torch


def resolve_cycles(
    parent_logits: torch.Tensor,
    score_threshold: float = float("-inf"),
) -> torch.Tensor:
    """
    Build a valid tree from parent prediction logits.

    Uses global greedy edge selection to ensure no cycles.

    Args:
        parent_logits: [N, N+1] logits where last column is root
        score_threshold: Minimum score to consider an edge

    Returns:
        parent_pred: [N] parent indices (N = root)
    """
    if parent_logits.ndim != 2 or parent_logits.shape[1] != parent_logits.shape[0] + 1:
        raise ValueError(f"Expected parent logits shaped [N, N+1], got {parent_logits.shape}")
    probabilities = torch.softmax(parent_logits, dim=-1)
    return greedy_tree_from_scores(probabilities, score_threshold=score_threshold)


def resolve_cycles_batch(
    parent_logits: torch.Tensor,
    score_threshold: float = float("-inf"),
) -> torch.Tensor:
    """
    Batch version of cycle resolution.

    Args:
        parent_logits: [B, N, N+1] logits
        score_threshold: Minimum score threshold

    Returns:
        parent_pred: [B, N] parent indices
    """
    if parent_logits.ndim != 3 or parent_logits.shape[2] != parent_logits.shape[1] + 1:
        raise ValueError(f"Expected parent logits shaped [B, N, N+1], got {parent_logits.shape}")
    B, N, _ = parent_logits.shape
    device = parent_logits.device

    parent_pred = torch.zeros((B, N), dtype=torch.long, device=device)

    for b in range(B):
        parent_pred[b] = resolve_cycles(parent_logits[b], score_threshold)

    return parent_pred


def detect_cycles(parent_pred: torch.Tensor) -> List[List[int]]:
    """
    Detect cycles in parent prediction.

    Args:
        parent_pred: [N] parent indices

    Returns:
        List of cycles (each cycle is a list of node indices)
    """
    N = len(parent_pred)
    root_idx = N

    visited = set()
    cycles = []

    for start_node in range(N):
        if start_node in visited:
            continue

        # Trace path from start_node
        path = []
        path_set = set()
        current = start_node

        while current not in visited and current != root_idx:
            if current in path_set:
                # Found a cycle
                cycle_start = path.index(current)
                cycle = path[cycle_start:]
                cycles.append(cycle)
                break

            path.append(current)
            path_set.add(current)
            current = parent_pred[current].item()

        # Mark all nodes in path as visited
        visited.update(path_set)

    return cycles


def greedy_tree_from_scores(
    parent_scores: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    score_threshold: float = float("-inf"),
) -> torch.Tensor:
    """
    Build tree using global greedy edge selection.

    Algorithm:
    1. Sort all edges by score (descending)
    2. Add edges greedily, skipping if it would create a cycle
    3. Connect remaining nodes to root

    Args:
        parent_scores: [N, N+1] unnormalized scores
        valid_mask: [N] mask for valid queries (optional)

    Returns:
        parent_pred: [N] parent indices
    """
    if parent_scores.ndim != 2 or parent_scores.shape[1] != parent_scores.shape[0] + 1:
        raise ValueError(f"Expected parent scores shaped [N, N+1], got {parent_scores.shape}")
    N = parent_scores.shape[0]
    root_idx = N
    device = parent_scores.device

    if valid_mask is None:
        valid_mask = torch.ones(N, dtype=torch.bool, device=device)
    elif valid_mask.shape != (N,):
        raise ValueError(f"Expected valid_mask shaped [{N}], got {valid_mask.shape}")
    else:
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)

    parent_pred = torch.full((N,), -1, dtype=torch.long, device=device)

    # Get all valid edges: (child, parent, score)
    edges = []
    for child in range(N):
        if not valid_mask[child]:
            continue
        for parent in range(N + 1):  # Include root
            if parent == child:
                continue
            if parent < N and not valid_mask[parent]:
                continue
            score = parent_scores[child, parent].item()
            if score >= score_threshold:
                edges.append((score, child, parent))

    # Sort by score descending
    edges.sort(reverse=True)

    # Greedy edge selection
    def would_create_cycle(child: int, parent: int) -> bool:
        """Check if adding child->parent would create a cycle."""
        if parent == root_idx:
            return False

        # Trace from parent to see if we reach child
        current = parent
        visited = {child}
        while current != root_idx and current >= 0:
            if current in visited:
                return True
            visited.add(current)
            current = parent_pred[current].item()
        return False

    for score, child, parent in edges:
        if parent_pred[child] != -1:
            continue  # Already assigned
        if would_create_cycle(child, parent):
            continue
        parent_pred[child] = parent

    # Connect remaining nodes to root
    for node in range(N):
        if valid_mask[node] and parent_pred[node] == -1:
            parent_pred[node] = root_idx

    return parent_pred
