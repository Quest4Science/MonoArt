"""
Tree Structure Utilities for Kinematic Tree.

Provides functions for:
1. Building tree structure from parent predictions
2. Computing tree metrics
3. Visualization utilities
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch


def build_kinematic_tree(
    parent_pred: torch.Tensor,
) -> Dict:
    """
    Build tree structure from parent predictions.

    Args:
        parent_pred: [N] parent indices where N = root

    Returns:
        tree: Dictionary containing:
            - root: Root index (N)
            - children: {parent_id: [child_id, ...]}
            - depth: {node_id: depth}
            - parents: {node_id: parent_id}
            - num_nodes: Number of nodes (excluding root)
    """
    N = len(parent_pred)
    root_idx = N

    # Build children mapping
    children: Dict[int, List[int]] = defaultdict(list)
    parents: Dict[int, int] = {}

    for child_idx in range(N):
        parent_idx = parent_pred[child_idx].item()
        children[parent_idx].append(child_idx)
        parents[child_idx] = parent_idx

    # Compute depth using BFS
    depth: Dict[int, int] = {root_idx: 0}
    queue = [(root_idx, 0)]

    while queue:
        node, d = queue.pop(0)
        for child in children[node]:
            depth[child] = d + 1
            queue.append((child, d + 1))

    # Handle nodes not reachable from root (shouldn't happen after cycle resolution)
    for node in range(N):
        if node not in depth:
            depth[node] = -1  # Mark as unreachable

    return {
        "root": root_idx,
        "children": dict(children),
        "depth": depth,
        "parents": parents,
        "num_nodes": N,
    }


def get_tree_depth(tree: Dict) -> int:
    """Get maximum depth of the tree."""
    if not tree["depth"]:
        return 0
    depths = [d for d in tree["depth"].values() if d >= 0]
    return max(depths) if depths else 0


def get_subtree_nodes(
    tree: Dict,
    node_id: int,
) -> Set[int]:
    """
    Get all nodes in the subtree rooted at node_id.

    Args:
        tree: Tree structure from build_kinematic_tree
        node_id: Root of subtree

    Returns:
        Set of node IDs in subtree (including node_id)
    """
    subtree = {node_id}
    queue = [node_id]

    while queue:
        current = queue.pop(0)
        for child in tree["children"].get(current, []):
            if child not in subtree:
                subtree.add(child)
                queue.append(child)

    return subtree


def compute_tree_metrics(
    pred_tree: Dict,
    gt_tree: Dict,
    matched_pred_to_gt: Optional[Dict[int, int]] = None,
) -> Dict[str, float]:
    """
    Compute metrics comparing predicted and GT trees.

    Args:
        pred_tree: Predicted tree from build_kinematic_tree
        gt_tree: Ground truth tree
        matched_pred_to_gt: Mapping from pred node to GT node (if different)

    Returns:
        metrics: Dictionary of metric values
    """
    # If no mapping, assume identity (pred_i corresponds to gt_i)
    if matched_pred_to_gt is None:
        matched_pred_to_gt = {i: i for i in range(pred_tree["num_nodes"])}

    pred_parents = pred_tree["parents"]
    gt_parents = gt_tree["parents"]
    root_idx = pred_tree["root"]

    # Edge accuracy: fraction of correct parent predictions
    correct_edges = 0
    total_edges = 0

    for pred_node, gt_node in matched_pred_to_gt.items():
        if pred_node not in pred_parents or gt_node not in gt_parents:
            continue

        pred_parent = pred_parents[pred_node]
        gt_parent = gt_parents[gt_node]

        # Map predicted parent to GT space
        if pred_parent == root_idx:
            pred_parent_gt = gt_tree["root"]
        elif pred_parent in matched_pred_to_gt:
            pred_parent_gt = matched_pred_to_gt[pred_parent]
        else:
            pred_parent_gt = -999  # No match

        if pred_parent_gt == gt_parent:
            correct_edges += 1
        total_edges += 1

    edge_accuracy = correct_edges / total_edges if total_edges > 0 else 0.0

    # Root children accuracy: fraction of correct root children
    pred_root_children = set(pred_tree["children"].get(root_idx, []))
    gt_root_children = set(gt_tree["children"].get(gt_tree["root"], []))

    # Map to GT space
    pred_root_children_gt = {matched_pred_to_gt.get(c, -1) for c in pred_root_children}

    if len(gt_root_children) > 0:
        root_precision = (
            len(pred_root_children_gt & gt_root_children) / len(pred_root_children_gt)
            if pred_root_children_gt
            else 0
        )
        root_recall = len(pred_root_children_gt & gt_root_children) / len(gt_root_children)
        root_f1 = (
            2 * root_precision * root_recall / (root_precision + root_recall)
            if (root_precision + root_recall) > 0
            else 0
        )
    else:
        root_precision = root_recall = root_f1 = 0.0

    # Depth accuracy
    pred_depths = pred_tree["depth"]
    gt_depths = gt_tree["depth"]

    depth_correct = 0
    depth_total = 0
    for pred_node, gt_node in matched_pred_to_gt.items():
        if pred_node in pred_depths and gt_node in gt_depths:
            if pred_depths[pred_node] == gt_depths[gt_node]:
                depth_correct += 1
            depth_total += 1

    depth_accuracy = depth_correct / depth_total if depth_total > 0 else 0.0

    return {
        "edge_accuracy": edge_accuracy,
        "root_precision": root_precision,
        "root_recall": root_recall,
        "root_f1": root_f1,
        "depth_accuracy": depth_accuracy,
        "max_pred_depth": get_tree_depth(pred_tree),
        "max_gt_depth": get_tree_depth(gt_tree),
    }


def tree_to_string(
    tree: Dict,
    node_names: Optional[Dict[int, str]] = None,
) -> str:
    """
    Convert tree to string representation for visualization.

    Args:
        tree: Tree structure from build_kinematic_tree
        node_names: Optional mapping from node_id to name

    Returns:
        String representation of tree
    """
    if node_names is None:
        node_names = {}

    root_idx = tree["root"]
    children = tree["children"]

    def format_node(node_id: int) -> str:
        name = node_names.get(node_id, f"Node_{node_id}")
        if node_id == root_idx:
            return "Root"
        return name

    def print_subtree(node: int, prefix: str = "", is_last: bool = True) -> str:
        connector = "└── " if is_last else "├── "
        line = prefix + connector + format_node(node) + "\n"

        child_list = children.get(node, [])
        for i, child in enumerate(child_list):
            is_child_last = i == len(child_list) - 1
            extension = "    " if is_last else "│   "
            line += print_subtree(child, prefix + extension, is_child_last)

        return line

    # Start from root
    result = format_node(root_idx) + "\n"
    root_children = children.get(root_idx, [])
    for i, child in enumerate(root_children):
        is_last = i == len(root_children) - 1
        result += print_subtree(child, "", is_last)

    return result


def validate_tree(parent_pred: torch.Tensor) -> Tuple[bool, str]:
    """
    Validate that parent predictions form a valid tree.

    Args:
        parent_pred: [N] parent indices

    Returns:
        (is_valid, message)
    """
    N = len(parent_pred)
    root_idx = N

    # Check 1: No self-loops
    for i in range(N):
        if parent_pred[i].item() == i:
            return False, f"Self-loop at node {i}"

    # Check 2: All nodes reachable from root (no cycles)
    visited = set()
    for start in range(N):
        path = []
        current = start

        while current not in visited and current != root_idx:
            if current in path:
                return False, f"Cycle detected: {path[path.index(current) :]}"
            path.append(current)
            current = parent_pred[current].item()

        visited.update(path)

    # Check 3: All nodes have valid parent
    for i in range(N):
        parent = parent_pred[i].item()
        if parent < 0 or parent > N:
            return False, f"Invalid parent {parent} for node {i}"

    return True, "Valid tree"
