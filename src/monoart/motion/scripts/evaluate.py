#!/usr/bin/env python3
"""
Evaluation script for Articulated Object Part Segmentation and Motion Prediction.

Usage:
    python -m monoart.motion.scripts.evaluate --config configs/train_motion.yaml --checkpoint checkpoints/epoch_20.pth
    python -m monoart.motion.scripts.evaluate --config configs/train_motion.yaml --checkpoint checkpoints/epoch_20.pth --output_dir results/
    python -m monoart.motion.scripts.evaluate --config configs/train_motion.yaml --checkpoint checkpoints/epoch_20.pth --split test
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from monoart.checkpoints import load_torch, motion_payload
from monoart.motion.datasets.articulated_dataset import ArticulatedDataset
from monoart.motion.datasets.collate_fn import articulated_collate_fn
from monoart.motion.datasets.motion_parser import (
    MOTION_TYPE_NAMES,
    get_category_names,
)
from monoart.motion.models.articulated_maft import ArticulatedMAFT

# Stage 2: Parent prediction imports
try:
    from monoart.motion.relation import (
        ParentPredictionHead,
        resolve_cycles,
    )

    PARENT_PREDICTION_AVAILABLE = True
except ImportError:
    PARENT_PREDICTION_AVAILABLE = False


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


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
            "num_classes": 18,
            "enable_part_classification": sf_config.get("enable_part_classification", True),
            "part_class_use_mlp": sf_config.get("part_class_use_mlp", True),
            "enable_motion_prior": sf_config.get("enable_motion_prior", True),
            "use_motion_prior_head": sf_config.get("use_motion_prior_head", True),
            "motion_prior_adaptive_mix": sf_config.get("motion_prior_adaptive_mix", False),
            "enable_semantic_refiner": sf_config.get("enable_semantic_refiner", True),
            "clip_embedding_path": sf_config.get("clip_embedding_path", None),
        }

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
        num_categories=model_config.get("num_categories", 7),
        num_motion_types=model_config.get("num_motion_types", 3),
        use_learnable_fallback=model_config.get("use_learnable_fallback", True),
        use_position_refinement=model_config.get("use_position_refinement", True),
        semantic_fusion_config=semantic_fusion_config,
    )

    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    """Load model checkpoint."""
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
        model, parent_head (or None), has_parent_prediction
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
        print("=" * 60)
    else:
        print("STAGE 1 MODE: No parent prediction evaluation")

    return model, parent_head, has_parent_prediction


def evaluate_parent_prediction(
    pred_parent: np.ndarray,
    gt_parent_info: dict,
    matched_indices: list,
    gt_link_ids: np.ndarray,
    num_pred_parts: int,
) -> dict:
    """
    Evaluate parent-child prediction accuracy.

    Args:
        pred_parent: [K] predicted parent indices (K means root)
        gt_parent_info: {link_id: parent_link_id} from batch
        matched_indices: List of (pred_idx, gt_idx) tuples
        gt_link_ids: [G] ground truth link IDs
        num_pred_parts: K, number of predicted parts

    Returns:
        Dictionary with parent prediction metrics
    """
    if pred_parent is None or len(matched_indices) == 0:
        return {
            "parent_edge_accuracy": 0.0,
            "root_precision": 0.0,
            "root_recall": 0.0,
            "root_f1": 0.0,
            "num_parent_correct": 0,
            "num_parent_total": 0,
        }

    # Build reverse mapping: gt_idx -> pred_idx
    gt_to_pred = {}
    for pred_idx, gt_idx in matched_indices:
        gt_to_pred[gt_idx] = pred_idx

    # Evaluate each matched prediction
    num_correct = 0
    num_total = 0
    pred_roots = []
    gt_roots = []

    for pred_idx, gt_idx in matched_indices:
        # Get GT link_id and parent_link_id
        gt_link_id = int(gt_link_ids[gt_idx])
        gt_parent_link_id = gt_parent_info.get(gt_link_id, -1)  # -1 means root

        # Get predicted parent
        pred_parent_idx = pred_parent[pred_idx]
        is_pred_root = pred_parent_idx == num_pred_parts
        pred_roots.append(is_pred_root)

        # Check if GT parent is root
        is_gt_root = gt_parent_link_id == -1
        gt_roots.append(is_gt_root)

        if is_gt_root:
            # GT is connected to root
            if is_pred_root:
                num_correct += 1
        else:
            # GT is connected to another part
            # Find which pred_idx corresponds to gt_parent_link_id
            gt_parent_idx = None
            for gi, link_id in enumerate(gt_link_ids):
                if link_id == gt_parent_link_id:
                    if gi in gt_to_pred:
                        gt_parent_idx = gt_to_pred[gi]
                    break

            if gt_parent_idx is not None:
                if pred_parent_idx == gt_parent_idx:
                    num_correct += 1
            # If parent not matched, can't evaluate this edge

        num_total += 1

    # Compute metrics
    edge_accuracy = num_correct / num_total if num_total > 0 else 0.0

    # Root F1
    pred_roots = np.array(pred_roots)
    gt_roots = np.array(gt_roots)

    if len(pred_roots) > 0:
        tp = np.sum(pred_roots & gt_roots)
        fp = np.sum(pred_roots & ~gt_roots)
        fn = np.sum(~pred_roots & gt_roots)

        root_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        root_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        root_f1 = (
            2 * root_precision * root_recall / (root_precision + root_recall)
            if (root_precision + root_recall) > 0
            else 0.0
        )
    else:
        root_precision = 0.0
        root_recall = 0.0
        root_f1 = 0.0

    return {
        "parent_edge_accuracy": edge_accuracy,
        "root_precision": root_precision,
        "root_recall": root_recall,
        "root_f1": root_f1,
        "num_parent_correct": num_correct,
        "num_parent_total": num_total,
    }


def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Compute IoU between two binary masks."""
    pred_mask = pred_mask > 0.5
    gt_mask = gt_mask > 0.5

    intersection = (pred_mask & gt_mask).sum().float()
    union = (pred_mask | gt_mask).sum().float()

    if union < 1:
        return 1.0 if intersection < 1 else 0.0

    return (intersection / union).item()


def compute_axis_direction_error(pred_dir: np.ndarray, gt_dir: np.ndarray) -> float:
    """
    Compute axis direction error (angle in degrees).
    Since axis direction is undirected, we take min of angle and 180-angle.
    """
    pred_dir = pred_dir / (np.linalg.norm(pred_dir) + 1e-8)
    gt_dir = gt_dir / (np.linalg.norm(gt_dir) + 1e-8)

    cos_angle = np.abs(np.dot(pred_dir, gt_dir))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def compute_axis_origin_error(
    pred_origin: np.ndarray, gt_origin: np.ndarray, gt_dir: np.ndarray
) -> float:
    """
    Compute axis origin error (point-to-line distance).
    Since origin can be anywhere along the axis, we compute perpendicular distance.
    """
    gt_dir = gt_dir / (np.linalg.norm(gt_dir) + 1e-8)

    # Vector from GT origin to pred origin
    diff = pred_origin - gt_origin

    # Project onto axis direction
    proj_len = np.dot(diff, gt_dir)
    proj = proj_len * gt_dir

    # Perpendicular component
    perp = diff - proj
    distance = np.linalg.norm(perp)

    return distance


def hungarian_match(
    pred_masks: torch.Tensor, gt_masks: torch.Tensor, pred_scores: torch.Tensor
) -> list:
    """
    Match predictions to ground truth using Hungarian algorithm.

    Args:
        pred_masks: [K, N] predicted masks (after sigmoid)
        gt_masks: [G, N] ground truth masks
        pred_scores: [K] prediction scores

    Returns:
        List of (pred_idx, gt_idx, iou) tuples
    """
    from scipy.optimize import linear_sum_assignment

    K, N = pred_masks.shape
    G = gt_masks.shape[0]

    if K == 0 or G == 0:
        return []

    # Compute IoU matrix
    pred_binary = pred_masks > 0.5
    gt_binary = gt_masks > 0.5

    # [K, G]
    intersection = (pred_binary.unsqueeze(1) & gt_binary.unsqueeze(0)).sum(dim=-1).float()
    union = (pred_binary.unsqueeze(1) | gt_binary.unsqueeze(0)).sum(dim=-1).float()
    iou_matrix = intersection / (union + 1e-8)

    # Cost matrix (negative IoU for minimization)
    cost_matrix = -iou_matrix.cpu().numpy()

    # Hungarian matching
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    matches = []
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        iou = iou_matrix[pred_idx, gt_idx].item()
        if iou > 0.1:  # Only keep matches with reasonable IoU
            matches.append((pred_idx, gt_idx, iou))

    return matches


@torch.no_grad()
def evaluate_sample(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    parent_head: torch.nn.Module = None,
    config: dict = None,
) -> dict:
    """
    Evaluate a single sample.

    Args:
        model: ArticulatedMAFT model
        batch: Input batch
        device: Device to run on
        score_threshold: Score threshold for predictions
        iou_threshold: IoU threshold for matching
        parent_head: ParentPredictionHead for Stage 2 (optional)
        config: Configuration dictionary (optional)

    Returns:
        Dictionary with evaluation results
    """
    # Move to device
    partfield_features = batch["partfield_features"].to(device)
    vae_features = batch["vae_features"].to(device)
    points = batch["points"].to(device)

    B = partfield_features.shape[0]
    assert B == 1, "Evaluation should be done with batch size 1"

    # Forward pass
    outputs = model(partfield_features, vae_features, points)

    # Get predictions
    predictions = model.get_predictions(outputs, score_threshold=score_threshold)
    pred = predictions[0]

    # Ground truth
    gt_group_ids = batch["group_ids"][0]  # [N]
    gt_category_idx = batch["category_idx"][0]
    if isinstance(gt_category_idx, torch.Tensor):
        gt_category_idx = gt_category_idx.item()
    gt_motion_types = batch["gt_motion_types"][0]  # [G]
    gt_axis_directions = batch["gt_axis_directions"][0]  # [G, 3]
    gt_axis_positions = batch["gt_axis_positions"][0]  # [G, 3]
    gt_link_ids = batch["gt_link_ids"][0]  # [G]
    gt_is_movable = batch["gt_is_movable"][0]  # [G]

    # Build GT masks from group_ids
    unique_groups = torch.unique(gt_group_ids[gt_group_ids >= 0])
    G = len(unique_groups)
    N = gt_group_ids.shape[0]

    gt_masks = torch.zeros(G, N, dtype=torch.bool, device=device)
    group_to_gt_idx = {}
    for i, gid in enumerate(unique_groups):
        gt_masks[i] = gt_group_ids == gid.item()
        group_to_gt_idx[gid.item()] = i

    # Category prediction
    pred_category_logits = outputs["category_logits"][0]  # [num_categories]
    pred_category_idx = pred_category_logits.argmax().item()
    category_correct = pred_category_idx == gt_category_idx

    K = len(pred["scores"])  # Number of predictions

    results = {
        "anno_id": batch["anno_id"][0],
        "category_name": batch["category_name"][0],
        "gt_category_idx": gt_category_idx,
        "pred_category_idx": pred_category_idx,
        "category_correct": category_correct,
        "num_gt_parts": G,
        "num_pred_parts": K,
    }

    # Match predictions to GT
    if K > 0 and G > 0:
        pred_masks = pred["masks"].to(device)  # [K, N]
        matches = hungarian_match(pred_masks, gt_masks.float(), pred["scores"])

        # Compute metrics for matched parts
        ious = []
        motion_type_correct = []
        axis_direction_errors = []
        axis_origin_errors = []

        matched_predictions = []

        for pred_idx, gt_idx, iou in matches:
            ious.append(iou)

            # Get GT group_id and find corresponding motion info
            gt_group_id = unique_groups[gt_idx].item()

            # Find motion info for this group
            # Motion info index may differ from gt_idx
            motion_idx = None
            for mi, lid in enumerate(gt_link_ids):
                if lid.item() == gt_group_id:
                    motion_idx = mi
                    break

            if motion_idx is not None:
                # Motion type comparison
                pred_motion_type = pred["motion_types"][pred_idx].item()
                gt_motion_type = gt_motion_types[motion_idx].item()
                motion_type_correct.append(pred_motion_type == gt_motion_type)

                # Only compute axis errors for movable parts
                if gt_is_movable[motion_idx]:
                    pred_axis_dir = pred["axis_directions"][pred_idx].cpu().numpy()
                    gt_axis_dir = gt_axis_directions[motion_idx].cpu().numpy()

                    pred_axis_origin = pred["axis_origins"][pred_idx].cpu().numpy()
                    gt_axis_origin = gt_axis_positions[motion_idx].cpu().numpy()

                    dir_error = compute_axis_direction_error(pred_axis_dir, gt_axis_dir)
                    origin_error = compute_axis_origin_error(
                        pred_axis_origin, gt_axis_origin, gt_axis_dir
                    )

                    axis_direction_errors.append(dir_error)
                    axis_origin_errors.append(origin_error)

                matched_predictions.append(
                    {
                        "pred_idx": pred_idx,
                        "gt_group_id": gt_group_id,
                        "iou": iou,
                        "pred_motion_type": MOTION_TYPE_NAMES.get(
                            pred_motion_type, str(pred_motion_type)
                        ),
                        "gt_motion_type": MOTION_TYPE_NAMES.get(
                            gt_motion_type, str(gt_motion_type)
                        ),
                        "score": pred["scores"][pred_idx].item(),
                    }
                )

        results["matched_parts"] = len(matches)
        results["mean_iou"] = np.mean(ious) if ious else 0.0
        results["motion_type_accuracy"] = (
            np.mean(motion_type_correct) if motion_type_correct else 0.0
        )
        results["mean_axis_direction_error"] = (
            np.mean(axis_direction_errors) if axis_direction_errors else 0.0
        )
        results["mean_axis_origin_error"] = (
            np.mean(axis_origin_errors) if axis_origin_errors else 0.0
        )
        results["matched_predictions"] = matched_predictions

        # AP@IoU threshold
        tp = sum(1 for iou in ious if iou >= iou_threshold)
        results["precision"] = tp / len(pred["scores"]) if len(pred["scores"]) > 0 else 0.0
        results["recall"] = tp / G if G > 0 else 0.0

        # ========== Stage 2: Parent Prediction Evaluation ==========
        if parent_head is not None and K > 0:
            # Get query features for parent prediction
            query_features = outputs.get("query_features")  # [B, Q, D]
            query_positions = outputs.get("query_positions")  # [B, Q, 3]
            class_probs = outputs.get("part_class_probs")  # [B, Q, 18] or None

            # Get valid query indices (matching predictions)
            valid_indices = pred.get("indices", torch.arange(K))
            if not isinstance(valid_indices, torch.Tensor):
                valid_indices = torch.tensor(valid_indices, device=device)

            # Get filtered features
            query_features_filtered = query_features[0, valid_indices]  # [K, D]
            query_positions_filtered = query_positions[0, valid_indices]  # [K, 3]

            class_probs_filtered = None
            if class_probs is not None:
                class_probs_filtered = class_probs[0, valid_indices]  # [K, 18]

            # Compute position embedding
            q_position = model.pos_embed_proj(
                model.pos_embed(query_positions_filtered.unsqueeze(0))
            )

            # Run parent head
            parent_logits = parent_head(
                query_features_filtered.unsqueeze(0),
                q_position,
                class_probs_filtered.unsqueeze(0) if class_probs_filtered is not None else None,
            )

            # Get predictions with cycle resolution
            pp_config = config.get("parent_prediction", {}) if config else {}
            use_cycle_resolution = pp_config.get("inference", {}).get("use_cycle_resolution", True)

            if use_cycle_resolution and PARENT_PREDICTION_AVAILABLE:
                pred_parent = resolve_cycles(parent_logits[0])
            else:
                pred_parent = parent_logits[0].argmax(dim=-1).cpu().numpy()

            # Get GT parent info
            gt_parent_info = batch.get("gt_parent_info", [{}])[0]
            gt_link_ids_np = (
                gt_link_ids.cpu().numpy() if isinstance(gt_link_ids, torch.Tensor) else gt_link_ids
            )

            # Convert matches to list of (pred_idx, gt_idx) tuples
            match_pairs = [(m[0], m[1]) for m in matches]

            # Evaluate parent prediction
            parent_metrics = evaluate_parent_prediction(
                pred_parent=pred_parent,
                gt_parent_info=gt_parent_info,
                matched_indices=match_pairs,
                gt_link_ids=gt_link_ids_np,
                num_pred_parts=K,
            )
            results.update(parent_metrics)
        else:
            # No parent prediction
            results["parent_edge_accuracy"] = None
            results["root_f1"] = None
    else:
        results["matched_parts"] = 0
        results["mean_iou"] = 0.0
        results["motion_type_accuracy"] = 0.0
        results["mean_axis_direction_error"] = 0.0
        results["mean_axis_origin_error"] = 0.0
        results["matched_predictions"] = []
        results["precision"] = 0.0
        results["recall"] = 1.0 if G == 0 else 0.0
        results["parent_edge_accuracy"] = None
        results["root_f1"] = None

    return results


def export_predictions(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
    score_threshold: float = 0.5,
    num_categories: int = 46,
) -> dict:
    """
    Export predictions for a single sample to JSON-serializable format.
    """
    with torch.no_grad():
        partfield_features = batch["partfield_features"].to(device)
        vae_features = batch["vae_features"].to(device)
        points = batch["points"].to(device)

        outputs = model(partfield_features, vae_features, points)
        predictions = model.get_predictions(outputs, score_threshold=score_threshold)
        pred = predictions[0]

    # Category
    pred_category_logits = outputs["category_logits"][0]
    pred_category_idx = pred_category_logits.argmax().item()
    category_probs = F.softmax(pred_category_logits, dim=0).cpu().numpy().tolist()

    # Get dynamic category names based on num_categories
    category_names = get_category_names(num_categories)

    # Build output
    output = {
        "anno_id": batch["anno_id"][0],
        "category": {
            "predicted": category_names.get(pred_category_idx, str(pred_category_idx)),
            "predicted_idx": pred_category_idx,
            "confidence": category_probs[pred_category_idx],
            "probabilities": {
                category_names.get(i, str(i)): p for i, p in enumerate(category_probs)
            },
        },
        "parts": [],
    }

    # Parts
    if len(pred["scores"]) > 0:
        pred_masks = pred["masks"].cpu().numpy()  # [K, N]

        for k in range(len(pred["scores"])):
            # Get point indices for this part (mask > 0.5)
            mask = pred_masks[k] > 0.5
            point_indices = np.where(mask)[0].tolist()

            part_info = {
                "part_id": k,
                "score": float(pred["scores"][k].item()),
                "num_points": int(mask.sum()),
                "point_indices": point_indices,  # Can be large, consider saving separately
                "motion": {
                    "type": MOTION_TYPE_NAMES.get(pred["motion_types"][k].item(), "Unknown"),
                    "type_idx": int(pred["motion_types"][k].item()),
                    "type_probs": pred["motion_type_probs"][k].cpu().numpy().tolist(),
                    "axis_direction": pred["axis_directions"][k].cpu().numpy().tolist(),
                    "axis_origin": pred["axis_origins"][k].cpu().numpy().tolist(),
                },
                "query_position": pred["positions"][k].cpu().numpy().tolist(),
            }
            output["parts"].append(part_info)

    return output


def evaluate(
    config: dict,
    checkpoint_path: str,
    split: str = "val",
    output_dir: str = None,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    save_predictions: bool = True,
    device: torch.device = None,
):
    """
    Run evaluation on the specified split.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

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

    # Get num_categories for dynamic category name mapping
    num_categories = config.get("model", {}).get("num_categories", 46)

    # Create dataset
    data_config = config["data"]
    csv_key = f"{split}_csv"
    if csv_key not in data_config:
        csv_key = "val_csv"  # Fallback

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
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Evaluate one by one
        shuffle=False,
        num_workers=config["inference"].get("num_workers", 4),
        collate_fn=articulated_collate_fn,
        pin_memory=True,
    )

    print(f"Evaluating {len(dataset)} samples...")

    # Evaluation loop
    all_results = []
    all_predictions = []

    # Per-category metrics
    category_metrics = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "ious": [],
            "motion_acc": [],
            "dir_errors": [],
            "origin_errors": [],
        }
    )

    # Track parent prediction metrics
    parent_edge_accuracies = []
    root_f1_scores = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        # Evaluate (with optional parent_head for Stage 2)
        result = evaluate_sample(
            model,
            batch,
            device,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            parent_head=parent_head,
            config=config,
        )
        all_results.append(result)

        # Track parent prediction metrics if available
        if result.get("parent_edge_accuracy") is not None:
            parent_edge_accuracies.append(result["parent_edge_accuracy"])
        if result.get("root_f1") is not None:
            root_f1_scores.append(result["root_f1"])

        # Update per-category metrics
        cat_name = result["category_name"]
        category_metrics[cat_name]["total"] += 1
        if result["category_correct"]:
            category_metrics[cat_name]["correct"] += 1
        if result["mean_iou"] > 0:
            category_metrics[cat_name]["ious"].append(result["mean_iou"])
        if result["motion_type_accuracy"] > 0:
            category_metrics[cat_name]["motion_acc"].append(result["motion_type_accuracy"])
        if result["mean_axis_direction_error"] > 0:
            category_metrics[cat_name]["dir_errors"].append(result["mean_axis_direction_error"])
        if result["mean_axis_origin_error"] > 0:
            category_metrics[cat_name]["origin_errors"].append(result["mean_axis_origin_error"])

        # Export predictions
        if save_predictions:
            pred_output = export_predictions(model, batch, device, score_threshold, num_categories)
            all_predictions.append(pred_output)

    # Compute overall metrics
    overall_metrics = {
        "total_samples": len(all_results),
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        # Category classification
        "category_accuracy": np.mean([r["category_correct"] for r in all_results]),
        # Segmentation
        "mean_iou": np.mean([r["mean_iou"] for r in all_results if r["mean_iou"] > 0]),
        "mean_precision": np.mean([r["precision"] for r in all_results]),
        "mean_recall": np.mean([r["recall"] for r in all_results]),
        # Motion prediction
        "motion_type_accuracy": np.mean(
            [r["motion_type_accuracy"] for r in all_results if r["motion_type_accuracy"] > 0]
        ),
        "mean_axis_direction_error_deg": np.mean(
            [
                r["mean_axis_direction_error"]
                for r in all_results
                if r["mean_axis_direction_error"] > 0
            ]
        ),
        "mean_axis_origin_error": np.mean(
            [r["mean_axis_origin_error"] for r in all_results if r["mean_axis_origin_error"] > 0]
        ),
    }

    # Add parent prediction metrics if available (Stage 2)
    if parent_edge_accuracies:
        overall_metrics["parent_edge_accuracy"] = np.mean(parent_edge_accuracies)
        overall_metrics["root_f1"] = np.mean(root_f1_scores) if root_f1_scores else 0.0
        overall_metrics["has_parent_prediction"] = True
    else:
        overall_metrics["has_parent_prediction"] = False

    # Per-category summary
    per_category_metrics = {}
    for cat_name, metrics in category_metrics.items():
        per_category_metrics[cat_name] = {
            "total": metrics["total"],
            "category_accuracy": metrics["correct"] / metrics["total"]
            if metrics["total"] > 0
            else 0,
            "mean_iou": np.mean(metrics["ious"]) if metrics["ious"] else 0,
            "motion_type_accuracy": np.mean(metrics["motion_acc"]) if metrics["motion_acc"] else 0,
            "mean_axis_direction_error_deg": np.mean(metrics["dir_errors"])
            if metrics["dir_errors"]
            else 0,
            "mean_axis_origin_error": np.mean(metrics["origin_errors"])
            if metrics["origin_errors"]
            else 0,
        }

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nOverall Metrics ({len(all_results)} samples):")
    print(f"  Category Accuracy:      {overall_metrics['category_accuracy']:.4f}")
    print(f"  Mean IoU:               {overall_metrics['mean_iou']:.4f}")
    print(f"  Precision@{iou_threshold}:          {overall_metrics['mean_precision']:.4f}")
    print(f"  Recall@{iou_threshold}:             {overall_metrics['mean_recall']:.4f}")
    print(f"  Motion Type Accuracy:   {overall_metrics['motion_type_accuracy']:.4f}")
    print(f"  Axis Direction Error:   {overall_metrics['mean_axis_direction_error_deg']:.2f}°")
    print(f"  Axis Origin Error:      {overall_metrics['mean_axis_origin_error']:.4f}")

    # Print parent prediction metrics if available (Stage 2)
    if overall_metrics.get("has_parent_prediction", False):
        print("\nParent Prediction Metrics (Stage 2):")
        print(f"  Parent Edge Accuracy:   {overall_metrics['parent_edge_accuracy']:.4f}")
        print(f"  Root F1:                {overall_metrics['root_f1']:.4f}")

    print("\nPer-Category Metrics:")
    print("-" * 70)
    print(f"{'Category':<20} {'Count':>6} {'CatAcc':>8} {'mIoU':>8} {'MotAcc':>8} {'DirErr':>8}")
    print("-" * 70)
    for cat_name in sorted(per_category_metrics.keys()):
        m = per_category_metrics[cat_name]
        print(
            f"{cat_name:<20} {m['total']:>6} {m['category_accuracy']:>8.4f} {m['mean_iou']:>8.4f} {m['motion_type_accuracy']:>8.4f} {m['mean_axis_direction_error_deg']:>7.2f}°"
        )

    # Save results
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(
            convert_to_serializable(
                {
                    "overall_metrics": overall_metrics,
                    "per_category_metrics": per_category_metrics,
                    "config": {
                        "checkpoint": checkpoint_path,
                        "split": split,
                        "score_threshold": score_threshold,
                        "iou_threshold": iou_threshold,
                    },
                }
            ),
            f,
            indent=2,
        )
    print(f"\nSaved evaluation results to: {results_file}")

    # Save per-sample results
    per_sample_file = os.path.join(output_dir, "per_sample_results.json")
    # Remove large fields for per-sample
    for r in all_results:
        if "matched_predictions" in r:
            r["matched_predictions"] = len(r.get("matched_predictions", []))
    with open(per_sample_file, "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"Saved per-sample results to: {per_sample_file}")

    # Save predictions
    if save_predictions:
        predictions_file = os.path.join(output_dir, "predictions.json")
        # Simplify predictions (remove large point_indices)
        for pred in all_predictions:
            for part in pred["parts"]:
                part["point_indices"] = f"[{part['num_points']} points]"  # Save space
        with open(predictions_file, "w") as f:
            json.dump(convert_to_serializable(all_predictions), f, indent=2)
        print(f"Saved predictions to: {predictions_file}")

    return overall_metrics, per_category_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ArticulatedMAFT")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Data split to evaluate",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument(
        "--score_threshold", type=float, default=0.5, help="Score threshold for predictions"
    )
    parser.add_argument(
        "--iou_threshold", type=float, default=0.5, help="IoU threshold for AP calculation"
    )
    parser.add_argument(
        "--no_save_predictions", action="store_true", help="Do not save detailed predictions"
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run evaluation
    evaluate(
        config=config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        save_predictions=not args.no_save_predictions,
        device=device,
    )


if __name__ == "__main__":
    main()
