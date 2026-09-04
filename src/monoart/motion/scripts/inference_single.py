#!/usr/bin/env python3
"""
Single sample inference script for ArticulatedMAFT.

Directly loads PLY, VAE, and semantic-reasoner features from a specified directory
without requiring CSV-based dataset loading.

Usage:
    python -m monoart.motion.scripts.inference_single --config configs/inference.yaml
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from plyfile import PlyData

from monoart.motion.datasets.motion_parser import MOTION_TYPE_NAMES, get_category_names
from monoart.motion.scripts.inference import (
    apply_nms,
    create_axes_ply,
    create_colored_ply,
    create_model,
    create_motion_json,
    load_checkpoint_with_parent_head,
    masks_to_per_point_labels,
    save_per_link_ply,
)


def load_single_sample(config: dict) -> dict:
    """
    Load a single sample from specified paths.

    Expected config.data.single_sample structure:
        ply_path: Path to points_100000.ply
        partfield_path: Path to a reasoner-feature PLY or NPY file. The argument
            keeps its historical name for checkpoint/configuration compatibility.
        vae_path: Path to vae_features.npz or *_100k_features.npz
        sample_id: Sample identifier for output naming
    """
    single_config = config["data"]["single_sample"]

    ply_path = single_config["ply_path"]
    partfield_path = single_config["partfield_path"]
    vae_path = single_config["vae_path"]
    sample_id = single_config.get("sample_id", "sample")

    # Load PLY point cloud
    print(f"Loading PLY: {ply_path}")
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    x = np.array(vertex["x"], dtype=np.float32)
    y = np.array(vertex["y"], dtype=np.float32)
    z = np.array(vertex["z"], dtype=np.float32)
    points = np.stack([x, y, z], axis=1)

    # Try to get normals
    try:
        nx = np.array(vertex["nx"], dtype=np.float32)
        ny = np.array(vertex["ny"], dtype=np.float32)
        nz = np.array(vertex["nz"], dtype=np.float32)
        normals = np.stack([nx, ny, nz], axis=1)
    except (KeyError, ValueError):
        print("No normals in PLY, computing from points...")
        normals = np.zeros_like(points)

    try:
        face_ids = np.array(vertex["face_id"], dtype=np.int64)
    except (KeyError, ValueError):
        face_ids = np.full(len(points), -1, dtype=np.int64)

    N = points.shape[0]
    print(f"Loaded {N} points")

    # Load semantic-reasoner features
    print(f"Loading reasoner features: {partfield_path}")
    if partfield_path.endswith(".npy"):
        partfield_features = np.load(partfield_path).astype(np.float32)
    elif partfield_path.endswith(".ply"):
        # Load from PLY with features
        pf_plydata = PlyData.read(partfield_path)
        pf_vertex = pf_plydata["vertex"]

        # Extract feature dimensions (f_0, f_1, ..., f_447)
        feature_names = [p.name for p in pf_vertex.properties if p.name.startswith("f_")]
        feature_dim = len(feature_names)
        print(f"Found {feature_dim} feature dimensions in reasoner PLY")

        partfield_features = np.zeros((len(pf_vertex.data), feature_dim), dtype=np.float32)
        for i, fname in enumerate(sorted(feature_names, key=lambda x: int(x.split("_")[1]))):
            partfield_features[:, i] = np.array(pf_vertex[fname], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported reasoner format: {partfield_path}")

    print(f"Reasoner features shape: {partfield_features.shape}")

    # Load VAE features
    print(f"Loading VAE: {vae_path}")
    vae_data = np.load(vae_path)
    if "features" in vae_data:
        vae_features = vae_data["features"].astype(np.float32)
    elif "vae_features" in vae_data:
        vae_features = vae_data["vae_features"].astype(np.float32)
    else:
        # Try first array
        keys = list(vae_data.keys())
        vae_features = vae_data[keys[0]].astype(np.float32)

    print(f"VAE features shape: {vae_features.shape}")

    # Validate shapes
    assert points.shape[0] == partfield_features.shape[0], (
        f"Point count mismatch: {points.shape[0]} vs {partfield_features.shape[0]}"
    )
    assert points.shape[0] == vae_features.shape[0], (
        f"Point count mismatch: {points.shape[0]} vs {vae_features.shape[0]}"
    )

    # Convert to tensors and add batch dimension
    batch = {
        "anno_id": [sample_id],
        "points": torch.from_numpy(points).unsqueeze(0),  # [1, N, 3]
        "normals": torch.from_numpy(normals).unsqueeze(0),  # [1, N, 3]
        "face_ids": torch.from_numpy(face_ids).unsqueeze(0),  # [1, N]
        "partfield_features": torch.from_numpy(partfield_features).unsqueeze(0),  # [1, N, 448]
        "vae_features": torch.from_numpy(vae_features).unsqueeze(0),  # [1, N, 8]
    }

    return batch


@torch.no_grad()
def run_single_inference(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
    config: dict,
    parent_head: torch.nn.Module = None,
) -> dict:
    """Run inference on a single sample."""
    inference_config = config.get("inference", {})
    score_threshold = inference_config.get("score_threshold", 0.5)
    mask_threshold = inference_config.get("mask_threshold", 0.5)
    use_nms = inference_config.get("use_nms", True)
    nms_iou_threshold = inference_config.get("nms_iou_threshold", 0.5)
    force_assign = inference_config.get("force_assign", True)

    pp_config = config.get("parent_prediction", {})
    use_cycle_resolution = pp_config.get("inference", {}).get("use_cycle_resolution", True)

    # Move to device
    partfield_features = batch["partfield_features"].to(device)
    vae_features = batch["vae_features"].to(device)
    points = batch["points"].to(device)

    # Forward pass
    outputs = model(partfield_features, vae_features, points)

    # Get predictions
    mask_logits = outputs["mask_logits"][0]
    score_logits = outputs["scores"][0]

    masks = torch.sigmoid(mask_logits)
    scores = torch.sigmoid(score_logits)

    # Filter by score
    valid_mask = scores > score_threshold
    valid_indices = torch.where(valid_mask)[0]
    masks = masks[valid_mask]
    scores = scores[valid_mask]

    motion_type_logits = outputs["motion_type_logits"][0][valid_mask]
    axis_directions = outputs["axis_direction"][0][valid_mask]
    axis_origins = outputs["axis_origin"][0][valid_mask]
    query_positions = outputs["query_positions"][0][valid_mask]
    revolute_limits = outputs["revolute_limit"][0][valid_mask]
    prismatic_limits = outputs["prismatic_limit"][0][valid_mask]

    query_features_all = outputs.get("query_features")
    query_positions_all = outputs.get("query_positions")
    class_probs_all = outputs.get("part_class_probs")

    # Apply NMS
    if use_nms and len(scores) > 0:
        keep_indices = apply_nms(masks, scores, nms_iou_threshold)
        masks = masks[keep_indices]
        scores = scores[keep_indices]
        motion_type_logits = motion_type_logits[keep_indices]
        axis_directions = axis_directions[keep_indices]
        axis_origins = axis_origins[keep_indices]
        query_positions = query_positions[keep_indices]
        revolute_limits = revolute_limits[keep_indices]
        prismatic_limits = prismatic_limits[keep_indices]
        valid_indices = valid_indices[keep_indices]

    K = len(scores)

    # Category prediction
    category_logits = outputs["category_logits"][0]
    category_probs = F.softmax(category_logits, dim=0)
    pred_category_idx = category_logits.argmax().item()
    num_categories = config.get("model", {}).get("num_categories", 46)
    category_names_map = get_category_names(num_categories)
    pred_category_name = category_names_map.get(pred_category_idx, f"Unknown_{pred_category_idx}")

    # Convert to numpy
    masks_np = masks.cpu().numpy()
    scores_np = scores.cpu().numpy()

    # Per-point labels
    labels, confidences = masks_to_per_point_labels(
        masks_np, scores_np, mask_threshold, force_assign=force_assign
    )

    # Parent prediction (Stage 2)
    parent_predictions = None
    kinematic_tree = None
    parent_confidences = None

    if parent_head is not None and K > 0:
        try:
            from monoart.motion.relation import build_kinematic_tree, resolve_cycles, tree_to_string

            PARENT_PREDICTION_AVAILABLE = True
        except ImportError:
            PARENT_PREDICTION_AVAILABLE = False

        query_features_filtered = query_features_all[0, valid_indices]
        query_positions_filtered = query_positions_all[0, valid_indices]

        class_probs_filtered = None
        if class_probs_all is not None:
            class_probs_filtered = class_probs_all[0, valid_indices]

        q_position = model.pos_embed_proj(model.pos_embed(query_positions_filtered.unsqueeze(0)))

        parent_logits = parent_head(
            query_features_filtered.unsqueeze(0),
            q_position,
            class_probs_filtered.unsqueeze(0) if class_probs_filtered is not None else None,
        )

        parent_probs = F.softmax(parent_logits[0], dim=-1)
        parent_predictions_raw = parent_logits[0].argmax(dim=-1).cpu().numpy()
        parent_confidences = parent_probs.max(dim=-1)[0].cpu().numpy()

        if use_cycle_resolution and PARENT_PREDICTION_AVAILABLE:
            parent_predictions = resolve_cycles(parent_logits[0])
        else:
            parent_predictions = parent_predictions_raw

        if PARENT_PREDICTION_AVAILABLE:
            kinematic_tree = build_kinematic_tree(parent_predictions)
            kinematic_tree["tree_string"] = tree_to_string(kinematic_tree)

    # Build parts list
    parts = []
    for i in range(len(scores_np)):
        motion_type_idx = motion_type_logits[i].argmax().item()
        motion_probs = F.softmax(motion_type_logits[i], dim=0).cpu().numpy()

        axis_dir = axis_directions[i].cpu().numpy()
        axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-8)

        rev_limit = revolute_limits[i].cpu().numpy().tolist()
        pri_limit = prismatic_limits[i].cpu().numpy().tolist()

        part = {
            "part_id": i,
            "score": float(scores_np[i]),
            "num_points": int((labels == i).sum()),
            "motion_type_idx": motion_type_idx,
            "motion_type": MOTION_TYPE_NAMES.get(motion_type_idx, "Unknown"),
            "motion_probs": motion_probs.tolist(),
            "axis_direction": axis_dir.tolist(),
            "axis_origin": axis_origins[i].cpu().numpy().tolist(),
            "query_position": query_positions[i].cpu().numpy().tolist(),
            "revolute_limit": rev_limit,
            "prismatic_limit": pri_limit,
        }

        if parent_predictions is not None:
            parent_idx = int(parent_predictions[i])
            part["parent_idx"] = parent_idx
            part["parent_name"] = "base" if parent_idx == K else f"link_{parent_idx}"
            if parent_confidences is not None:
                part["parent_confidence"] = float(parent_confidences[i])

        parts.append(part)

    result = {
        "anno_id": batch["anno_id"][0],
        "category_idx": pred_category_idx,
        "category_name": pred_category_name,
        "category_probs": category_probs.cpu().numpy().tolist(),
        "parts": parts,
        "labels": labels,
        "confidences": confidences,
        "masks": masks_np,
        "scores": scores_np,
    }

    if parent_predictions is not None:
        result["parent_predictions"] = parent_predictions.tolist()
        result["kinematic_tree"] = kinematic_tree

    return result


def save_single_outputs(result: dict, batch: dict, output_dir: str, config: dict):
    """Save inference outputs."""
    output_config = config.get("output", {})
    vis_config = output_config.get("visualization", {})

    os.makedirs(output_dir, exist_ok=True)

    # 1. Save NPZ
    if output_config.get("save_npz", True):
        npz_path = os.path.join(output_dir, "segmentation.npz")
        np.savez_compressed(
            npz_path,
            labels=result["labels"],
            confidences=result["confidences"],
            masks=result["masks"],
            scores=result["scores"],
            category_idx=result["category_idx"],
            category_name=result["category_name"],
        )
        print(f"Saved: {npz_path}")

    # 2. Save motion JSON
    if output_config.get("save_json", True):
        json_path = os.path.join(output_dir, "motion.json")
        points_np = batch["points"][0].numpy()
        output_local_coords = config.get("output", {}).get("output_local_coords", False)

        motion_json = create_motion_json(
            anno_id=result["anno_id"],
            category_name=result["category_name"],
            parts=result["parts"],
            points=points_np,
            output_local_coords=output_local_coords,
        )
        with open(json_path, "w") as f:
            json.dump(motion_json, f, indent=2)
        print(f"Saved: {json_path}")

        # Detail predictions
        detail_path = os.path.join(output_dir, "predictions_detail.json")
        num_categories = config.get("model", {}).get("num_categories", 46)
        category_names_map = get_category_names(num_categories)
        detail = {
            "anno_id": result["anno_id"],
            "category": {
                "idx": result["category_idx"],
                "name": result["category_name"],
                "probs": {
                    category_names_map.get(i, str(i)): p
                    for i, p in enumerate(result["category_probs"])
                },
            },
            "parts": result["parts"],
        }
        with open(detail_path, "w") as f:
            json.dump(detail, f, indent=2)
        print(f"Saved: {detail_path}")

    # 3. Save visualization PLY
    if output_config.get("save_ply", True) and vis_config.get("enabled", True):
        points_np = batch["points"][0].numpy()
        normals_np = batch["normals"][0].numpy()
        face_ids_np = batch["face_ids"][0].numpy() if "face_ids" in batch else None

        ply_path = os.path.join(output_dir, "segmentation.ply")
        create_colored_ply(
            points=points_np,
            normals=normals_np,
            labels=result["labels"],
            output_path=ply_path,
            colormap=vis_config.get("colormap", "tab20"),
            face_ids=face_ids_np,
        )
        print(f"Saved: {ply_path}")

        if vis_config.get("show_axes", True):
            axes_path = os.path.join(output_dir, "motion_axes.ply")
            use_global_transform = vis_config.get("use_global_transform", True)
            create_axes_ply(
                parts=result["parts"],
                output_path=axes_path,
                axis_length=vis_config.get("axis_length", 0.3),
                use_global_transform=use_global_transform,
            )
            print(f"Saved: {axes_path}")

        # Per-link PLY
        save_per_link = output_config.get("save_per_link_ply", True)
        if save_per_link and len(result["parts"]) > 0:
            per_link_dir = os.path.join(output_dir, "per_link")
            save_per_link_ply(
                points=points_np,
                normals=normals_np,
                labels=result["labels"],
                parts=result["parts"],
                output_dir=per_link_dir,
                colormap=vis_config.get("colormap", "tab20"),
            )
            print(f"Saved per-link PLYs to: {per_link_dir}")

    # 4. Save kinematic tree
    if output_config.get("save_kinematic_tree", True) and result.get("kinematic_tree"):
        tree_path = os.path.join(output_dir, "kinematic_tree.json")
        tree_output = {
            "anno_id": result["anno_id"],
            "num_parts": len(result["parts"]),
            "kinematic_tree": result["kinematic_tree"],
            "parent_predictions": result.get("parent_predictions", []),
            "parts_summary": [
                {
                    "part_id": p["part_id"],
                    "parent_name": p.get("parent_name", "base"),
                    "motion_type": p["motion_type"],
                    "score": p["score"],
                }
                for p in result["parts"]
            ],
        }
        with open(tree_path, "w") as f:
            json.dump(tree_output, f, indent=2)
        print(f"Saved: {tree_path}")


def main():
    parser = argparse.ArgumentParser(description="Single sample inference with ArticulatedMAFT")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Device
    device_str = config.get("inference", {}).get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load sample
    print("\nLoading single sample...")
    batch = load_single_sample(config)
    print(f"Sample ID: {batch['anno_id'][0]}")

    # Create model
    print("\nLoading model...")
    checkpoint_path = args.checkpoint or config.get("checkpoint", {}).get("path")
    if not checkpoint_path:
        raise ValueError("No checkpoint path provided")

    model = create_model(config)
    model, parent_head, has_parent_prediction = load_checkpoint_with_parent_head(
        model, checkpoint_path, device, config
    )
    model = model.to(device)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run inference
    print("\nRunning inference...")
    result = run_single_inference(model, batch, device, config, parent_head=parent_head)

    print("\nResults:")
    print(f"  Category: {result['category_name']} (idx={result['category_idx']})")
    print(f"  Num parts: {len(result['parts'])}")
    for part in result["parts"]:
        print(
            f"    Part {part['part_id']}: {part['motion_type']} (score={part['score']:.3f}, points={part['num_points']})"
        )

    # Save outputs
    output_dir = config.get("output", {}).get("dir", "./results/single_inference")
    print(f"\nSaving outputs to: {output_dir}")
    save_single_outputs(result, batch, output_dir, config)

    print("\nInference complete!")


if __name__ == "__main__":
    main()
