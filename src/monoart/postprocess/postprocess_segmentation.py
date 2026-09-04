#!/usr/bin/env python3
"""
Post-process per-point segmentation to remove fragments and noisy boundaries.

This script:
1. Keeps only the largest connected component for each label
2. Reassigns orphan points to nearby labels using KNN voting
3. Smooths boundaries using neighborhood voting
4. Merges small regions into nearby labels

Usage:
    python postprocess_segmentation.py --anno_id 47185_config_0_2_10
    python postprocess_segmentation.py --all
"""

import argparse
import json
import os
from collections import Counter
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from tqdm import tqdm

# Portable CLI defaults
INPUT_DIR = "outputs/motion"
OUTPUT_DIR = "outputs/postprocess"
MIN_POINTS = 200  # Minimum points to keep a label


def read_segmentation_ply(ply_path):
    """
    Read segmentation.ply file.

    Returns:
        points: [N, 3] coordinates
        normals: [N, 3] normals
        colors: [N, 3] RGB colors
        labels: [N] labels
        face_ids: [N] face IDs (or None)
    """
    with open(ply_path, "r") as f:
        lines = f.readlines()

    # Find header end
    header_end = 0
    has_face_id = False
    for i, line in enumerate(lines):
        if "property int face_id" in line:
            has_face_id = True
        if line.strip() == "end_header":
            header_end = i + 1
            break

    # Parse data
    points = []
    normals = []
    colors = []
    labels = []
    face_ids = []

    for line in lines[header_end:]:
        parts = line.strip().split()
        if len(parts) >= 10:
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            normals.append([float(parts[3]), float(parts[4]), float(parts[5])])
            colors.append([int(parts[6]), int(parts[7]), int(parts[8])])
            labels.append(int(parts[9]))
            if has_face_id and len(parts) >= 11:
                face_ids.append(int(parts[10]))

    return (
        np.array(points, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(colors, dtype=np.uint8),
        np.array(labels, dtype=np.int32),
        np.array(face_ids, dtype=np.int32) if face_ids else None,
    )


def write_segmentation_ply(ply_path, points, normals, colors, labels, face_ids=None):
    """Write segmentation.ply file."""
    N = len(points)
    has_face_id = face_ids is not None and len(face_ids) == N

    with open(ply_path, "w") as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int label\n")
        if has_face_id:
            f.write("property int face_id\n")
        f.write("end_header\n")

        # Data
        for i in range(N):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} ")
            f.write(f"{normals[i, 0]:.6f} {normals[i, 1]:.6f} {normals[i, 2]:.6f} ")
            f.write(f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]} ")
            f.write(f"{labels[i]}")
            if has_face_id:
                f.write(f" {face_ids[i]}")
            f.write("\n")


def find_connected_components(points, labels, label, k=15):
    """
    Find connected components for a specific label using KNN graph.

    Args:
        points: [N, 3] all points
        labels: [N] all labels
        label: target label to analyze
        k: number of neighbors for connectivity

    Returns:
        component_labels: [n_label_points] component ID for each point of this label
        n_components: number of components
        label_indices: global indices of points with this label
    """
    mask = labels == label
    label_indices = np.where(mask)[0]
    label_points = points[mask]
    n_label = len(label_points)

    if n_label < 2:
        return np.zeros(n_label, dtype=np.int32), 1, label_indices

    # Build KD-Tree for label points only
    tree = cKDTree(label_points)
    k_use = min(k, n_label - 1)
    distances, neighbors = tree.query(label_points, k=k_use + 1)

    # Build adjacency graph
    rows = []
    cols = []
    for i in range(n_label):
        for j in range(1, k_use + 1):
            neighbor_idx = neighbors[i, j]
            rows.append(i)
            cols.append(neighbor_idx)

    data = np.ones(len(rows))
    graph = csr_matrix((data, (rows, cols)), shape=(n_label, n_label))

    # Find connected components
    n_components, component_labels = connected_components(graph, directed=False)

    return component_labels, n_components, label_indices


def keep_largest_component(points, labels, min_component_ratio=0.1):
    """
    Keep only the largest connected component for each label.
    Points from smaller components are marked as -1 (to be reassigned).

    Args:
        points: [N, 3] coordinates
        labels: [N] labels
        min_component_ratio: minimum ratio of largest component to keep the label

    Returns:
        new_labels: [N] labels with small components marked as -1
        stats: dict with statistics
    """
    new_labels = labels.copy()
    unique_labels = np.unique(labels[labels >= 0])
    stats = {}

    for label in unique_labels:
        component_labels, n_components, label_indices = find_connected_components(
            points, labels, label
        )

        if n_components <= 1:
            stats[str(int(label))] = {
                "n_components": 1,
                "kept": int(len(label_indices)),
                "removed": 0,
            }
            continue

        # Find largest component
        component_sizes = Counter(component_labels)
        largest_component = max(component_sizes, key=component_sizes.get)
        largest_size = component_sizes[largest_component]

        # Mark points from non-largest components as -1
        removed = 0
        for i, comp in enumerate(component_labels):
            if comp != largest_component:
                new_labels[label_indices[i]] = -1
                removed += 1

        stats[str(int(label))] = {
            "n_components": int(n_components),
            "kept": int(largest_size),
            "removed": int(removed),
        }

    return new_labels, stats


def reassign_orphan_points(points, labels, k=20):
    """
    Reassign points with label -1 to nearby labels using KNN voting.

    Args:
        points: [N, 3] coordinates
        labels: [N] labels (with -1 for orphans)
        k: number of neighbors for voting

    Returns:
        new_labels: [N] labels with orphans reassigned
    """
    new_labels = labels.copy()
    orphan_mask = labels == -1
    orphan_indices = np.where(orphan_mask)[0]

    if len(orphan_indices) == 0:
        return new_labels

    # Build KD-Tree for all points
    tree = cKDTree(points)

    # For each orphan, find k nearest neighbors and vote
    orphan_points = points[orphan_mask]
    distances, neighbors = tree.query(orphan_points, k=k + 1)

    for i, orphan_idx in enumerate(orphan_indices):
        # Get labels of neighbors (excluding self)
        neighbor_labels = labels[neighbors[i, 1:]]

        # Only consider valid labels (>= 0)
        valid_labels = neighbor_labels[neighbor_labels >= 0]

        if len(valid_labels) > 0:
            # Vote for most common label
            label_counts = Counter(valid_labels)
            new_labels[orphan_idx] = label_counts.most_common(1)[0][0]

    return new_labels


def smooth_boundaries(points, labels, k=10, iterations=2):
    """
    Smooth boundaries using neighborhood voting.

    For each point, if its label differs from the majority of its neighbors,
    change it to the majority label.

    Args:
        points: [N, 3] coordinates
        labels: [N] labels
        k: number of neighbors for voting
        iterations: number of smoothing iterations

    Returns:
        new_labels: [N] smoothed labels
    """
    new_labels = labels.copy()
    tree = cKDTree(points)

    for iteration in range(iterations):
        changed = 0
        distances, neighbors = tree.query(points, k=k + 1)

        for i in range(len(points)):
            current_label = new_labels[i]

            # Get neighbor labels (excluding self)
            neighbor_labels = new_labels[neighbors[i, 1:]]
            valid_labels = neighbor_labels[neighbor_labels >= 0]

            if len(valid_labels) == 0:
                continue

            # Count labels
            label_counts = Counter(valid_labels)
            majority_label, majority_count = label_counts.most_common(1)[0]

            # Change if majority is different and strong enough (> 60%)
            if majority_label != current_label and majority_count > k * 0.6:
                new_labels[i] = majority_label
                changed += 1

        if changed == 0:
            break

    return new_labels


def merge_small_regions(points, labels, min_points=500, k=20):
    """
    Merge small regions (labels with few points) into nearby labels.

    Args:
        points: [N, 3] coordinates
        labels: [N] labels
        min_points: minimum points to keep a label
        k: number of neighbors for voting

    Returns:
        new_labels: [N] labels with small regions merged
    """
    new_labels = labels.copy()
    # Count points per label
    label_counts = Counter(labels[labels >= 0])

    # Find small labels
    small_labels = [label for label, count in label_counts.items() if count < min_points]

    if not small_labels:
        return new_labels

    # Build KD-Tree
    tree = cKDTree(points)

    for small_label in small_labels:
        mask = new_labels == small_label
        small_indices = np.where(mask)[0]
        small_points = points[mask]

        # Find nearest neighbors for each point
        distances, neighbors = tree.query(small_points, k=k + 1)

        for i, idx in enumerate(small_indices):
            neighbor_labels = new_labels[neighbors[i, 1:]]
            # Exclude the small label itself
            valid_labels = neighbor_labels[
                (neighbor_labels >= 0) & (neighbor_labels != small_label)
            ]

            if len(valid_labels) > 0:
                label_counts = Counter(valid_labels)
                new_labels[idx] = label_counts.most_common(1)[0][0]

    return new_labels


def relabel_sequential(labels):
    """
    Relabel to sequential integers starting from 0.

    Args:
        labels: [N] labels with possible gaps

    Returns:
        new_labels: [N] sequential labels (0, 1, 2, ...)
        mapping: dict old_label -> new_label
    """
    unique_labels = sorted(np.unique(labels[labels >= 0]))
    mapping = {old: new for new, old in enumerate(unique_labels)}

    new_labels = labels.copy()
    for old, new in mapping.items():
        new_labels[labels == old] = new

    return new_labels, mapping


def find_boundary_points(points, labels, k=15):
    """
    Find boundary points - points whose neighbors have different labels.

    Args:
        points: [N, 3] coordinates
        labels: [N] labels
        k: number of neighbors to check

    Returns:
        boundary_mask: [N] boolean mask of boundary points
        boundary_pairs: list of (label1, label2) pairs that share boundaries
    """
    tree = cKDTree(points)
    distances, neighbors = tree.query(points, k=k + 1)

    boundary_mask = np.zeros(len(points), dtype=bool)
    boundary_pairs = set()

    for i in range(len(points)):
        current_label = labels[i]
        if current_label < 0:
            continue

        neighbor_labels = labels[neighbors[i, 1:]]
        different_labels = neighbor_labels[
            (neighbor_labels >= 0) & (neighbor_labels != current_label)
        ]

        if len(different_labels) > 0:
            boundary_mask[i] = True
            for other_label in np.unique(different_labels):
                pair = tuple(sorted([current_label, other_label]))
                boundary_pairs.add(pair)

    return boundary_mask, list(boundary_pairs)


def fit_boundary_plane_ransac(
    points, labels, label1, label2, boundary_mask, inlier_threshold=0.02, min_boundary_points=50
):
    """
    Fit a plane to the boundary between two labels using RANSAC.

    Args:
        points: [N, 3] coordinates
        labels: [N] labels
        label1, label2: the two labels sharing the boundary
        boundary_mask: [N] boolean mask of boundary points
        inlier_threshold: distance threshold for RANSAC inliers
        min_boundary_points: minimum points needed to fit plane

    Returns:
        plane_normal: [3] normal vector of the plane
        plane_point: [3] a point on the plane
        success: whether fitting succeeded
    """
    # Get boundary points between these two labels
    mask1 = (labels == label1) & boundary_mask
    mask2 = (labels == label2) & boundary_mask

    boundary_points = (
        np.vstack([points[mask1], points[mask2]])
        if np.any(mask1) and np.any(mask2)
        else np.array([])
    )

    if len(boundary_points) < min_boundary_points:
        return None, None, False

    # Use PCA to find the plane normal
    # The plane normal is the eigenvector with smallest eigenvalue
    centroid = np.mean(boundary_points, axis=0)
    centered = boundary_points - centroid

    try:
        # SVD to find principal components
        U, S, Vt = np.linalg.svd(centered)

        # The normal is the last row of Vt (smallest singular value direction)
        plane_normal = Vt[-1]
        plane_point = centroid

        # Make sure normal points from label1 to label2
        center1 = np.mean(points[labels == label1], axis=0)
        center2 = np.mean(points[labels == label2], axis=0)
        direction = center2 - center1

        if np.dot(plane_normal, direction) < 0:
            plane_normal = -plane_normal

        return plane_normal, plane_point, True

    except Exception:
        return None, None, False


def straighten_boundary_with_plane(
    points, labels, label1, label2, plane_normal, plane_point, boundary_width=0.05
):
    """
    Straighten the boundary between two labels using the fitted plane.

    Points within boundary_width of the plane are reassigned based on
    which side of the plane they fall on.

    Args:
        points: [N, 3] coordinates
        labels: [N] labels
        label1, label2: the two labels
        plane_normal: [3] normal vector (pointing from label1 to label2)
        plane_point: [3] a point on the plane
        boundary_width: width of boundary region to adjust

    Returns:
        new_labels: [N] updated labels
        n_changed: number of points reassigned
    """
    new_labels = labels.copy()

    # Only consider points that are label1 or label2
    mask = (labels == label1) | (labels == label2)
    candidate_indices = np.where(mask)[0]

    if len(candidate_indices) == 0:
        return new_labels, 0

    # Calculate signed distance to plane for each candidate point
    candidate_points = points[candidate_indices]
    distances = np.dot(candidate_points - plane_point, plane_normal)

    # Find points near the boundary (within boundary_width)
    near_boundary = np.abs(distances) < boundary_width

    n_changed = 0
    for i, idx in enumerate(candidate_indices):
        if not near_boundary[i]:
            continue

        # Assign based on which side of plane
        if distances[i] < 0:
            # Should be label1
            if labels[idx] != label1:
                new_labels[idx] = label1
                n_changed += 1
        else:
            # Should be label2
            if labels[idx] != label2:
                new_labels[idx] = label2
                n_changed += 1

    return new_labels, n_changed


def straighten_all_boundaries(points, labels, boundary_width=0.03, verbose=True):
    """
    Straighten all boundaries between labels using plane fitting.

    Args:
        points: [N, 3] coordinates
        labels: [N] labels
        boundary_width: width of boundary region to adjust
        verbose: print progress

    Returns:
        new_labels: [N] labels with straightened boundaries
        stats: dict with statistics
    """
    new_labels = labels.copy()
    stats = {"boundaries_processed": 0, "points_changed": 0, "boundary_pairs": []}

    # Find boundary points and pairs
    boundary_mask, boundary_pairs = find_boundary_points(points, new_labels)

    if verbose:
        print(f"  Found {len(boundary_pairs)} boundary pairs to process")

    for label1, label2 in boundary_pairs:
        # Fit plane to boundary
        plane_normal, plane_point, success = fit_boundary_plane_ransac(
            points, new_labels, label1, label2, boundary_mask
        )

        if not success:
            continue

        # Straighten boundary
        new_labels, n_changed = straighten_boundary_with_plane(
            points, new_labels, label1, label2, plane_normal, plane_point, boundary_width
        )

        stats["boundaries_processed"] += 1
        stats["points_changed"] += n_changed
        stats["boundary_pairs"].append(
            {"labels": [int(label1), int(label2)], "points_changed": int(n_changed)}
        )

        # Update boundary mask after changes
        if n_changed > 0:
            boundary_mask, _ = find_boundary_points(points, new_labels)

    return new_labels, stats


def update_colors_by_label(labels):
    """
    Generate colors based on labels using tab20 colormap.

    Args:
        labels: [N] labels

    Returns:
        colors: [N, 3] RGB colors
    """
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    unique_labels = np.unique(labels[labels >= 0])

    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    colors[labels < 0] = [128, 128, 128]  # Gray for invalid

    for i, label in enumerate(unique_labels):
        color = cmap(i % 20)[:3]
        color = (np.array(color) * 255).astype(np.uint8)
        colors[labels == label] = color

    return colors


def postprocess_segmentation(
    points,
    normals,
    colors,
    labels,
    face_ids,
    min_component_ratio=0.1,
    min_points=500,
    smooth_iterations=2,
    smooth_k=10,
    boundary_width=0.03,
    straighten_boundaries=True,
    verbose=True,
):
    """
    Full post-processing pipeline.

    Args:
        points, normals, colors, labels, face_ids: input data
        min_component_ratio: minimum ratio to keep component
        min_points: minimum points to keep a label
        smooth_iterations: number of smoothing passes
        smooth_k: neighbors for smoothing
        boundary_width: width of boundary region for plane fitting
        straighten_boundaries: whether to straighten boundaries using plane fitting
        verbose: print progress

    Returns:
        new_labels: [N] cleaned labels
        new_colors: [N, 3] updated colors
        stats: dict with statistics
    """
    stats = {"original": {}}

    # Original stats
    for label in np.unique(labels[labels >= 0]):
        stats["original"][str(int(label))] = int(np.sum(labels == label))

    if verbose:
        print("Step 1: Keeping largest connected component per label...")
    labels_step1, component_stats = keep_largest_component(points, labels, min_component_ratio)
    stats["component_cleanup"] = component_stats

    orphan_count = np.sum(labels_step1 == -1)
    if verbose:
        print(f"  Orphan points to reassign: {orphan_count}")

    if verbose:
        print("Step 2: Reassigning orphan points using KNN voting...")
    labels_step2 = reassign_orphan_points(points, labels_step1, k=20)

    remaining_orphans = np.sum(labels_step2 == -1)
    if verbose:
        print(f"  Remaining orphans: {remaining_orphans}")

    if verbose:
        print(f"Step 3: Smoothing boundaries ({smooth_iterations} iterations)...")
    labels_step3 = smooth_boundaries(points, labels_step2, k=smooth_k, iterations=smooth_iterations)

    if verbose:
        print(f"Step 4: Merging small regions (< {min_points} points)...")
    labels_step4 = merge_small_regions(points, labels_step3, min_points=min_points)

    # Step 5: Straighten boundaries using plane fitting
    if straighten_boundaries:
        if verbose:
            print(
                f"Step 5: Straightening boundaries using plane fitting (width={boundary_width})..."
            )
        labels_step5, boundary_stats = straighten_all_boundaries(
            points, labels_step4, boundary_width=boundary_width, verbose=verbose
        )
        stats["boundary_straightening"] = boundary_stats
    else:
        labels_step5 = labels_step4

    # Use labels_step5 as final (no relabeling to preserve URDF correspondence)
    labels_final = labels_step5

    # Final stats
    stats["final"] = {}
    for label in np.unique(labels_final[labels_final >= 0]):
        stats["final"][str(int(label))] = int(np.sum(labels_final == label))

    # Update colors
    new_colors = update_colors_by_label(labels_final)

    return labels_final, new_colors, stats


def process_single_sample(
    anno_id: str,
    input_dir: str,
    output_dir: str,
    min_component_ratio: float = 0.1,
    min_points: int = 500,
    smooth_iterations: int = 2,
    smooth_k: int = 10,
    boundary_width: float = 0.03,
    straighten_boundaries: bool = True,
) -> dict:
    """
    Process a single sample.

    Args:
        anno_id: Annotation ID
        input_dir: Input directory with segmentation results
        output_dir: Output directory

    Returns:
        Result dict
    """
    # Paths
    input_ply = os.path.join(input_dir, anno_id, "segmentation.ply")
    output_ply = os.path.join(output_dir, anno_id, "segmentation.ply")

    if not os.path.exists(input_ply):
        return {"error": f"Input not found: {input_ply}", "anno_id": anno_id}

    # Check if already processed (skip_existing is passed via global or closure)
    if getattr(process_single_sample, "skip_existing", False) and os.path.exists(output_ply):
        return {"skipped": True, "reason": "Already processed", "anno_id": anno_id}

    # Read input
    points, normals, colors, labels, face_ids = read_segmentation_ply(input_ply)

    # Post-process
    new_labels, new_colors, stats = postprocess_segmentation(
        points,
        normals,
        colors,
        labels,
        face_ids,
        min_component_ratio=min_component_ratio,
        min_points=min_points,
        smooth_iterations=smooth_iterations,
        smooth_k=smooth_k,
        boundary_width=boundary_width,
        straighten_boundaries=straighten_boundaries,
        verbose=False,
    )

    # Create output directory
    sample_output_dir = os.path.join(output_dir, anno_id)
    os.makedirs(sample_output_dir, exist_ok=True)

    # Write output
    output_ply = os.path.join(sample_output_dir, "segmentation.ply")
    write_segmentation_ply(output_ply, points, normals, new_colors, new_labels, face_ids)

    # Copy motion.json if exists
    input_motion = os.path.join(input_dir, anno_id, "motion.json")
    if os.path.exists(input_motion):
        import shutil

        shutil.copy(input_motion, os.path.join(sample_output_dir, "motion.json"))

    # Copy motion_axes.ply if exists
    input_axes = os.path.join(input_dir, anno_id, "motion_axes.ply")
    if os.path.exists(input_axes):
        import shutil

        shutil.copy(input_axes, os.path.join(sample_output_dir, "motion_axes.ply"))

    # Save stats
    stats["anno_id"] = anno_id
    stats_path = os.path.join(sample_output_dir, "postprocess_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return {
        "anno_id": anno_id,
        "original_labels": len(stats["original"]),
        "final_labels": len(stats["final"]),
        "output_path": output_ply,
    }


def main():
    parser = argparse.ArgumentParser(description="Post-process segmentation results")
    parser.add_argument("--anno_id", type=str, default=None, help="Single annotation ID to process")
    parser.add_argument("--all", action="store_true", help="Process all samples")
    parser.add_argument(
        "--csv", type=str, default=None, help="CSV file with anno_id column to filter samples"
    )
    parser.add_argument("--input_dir", type=str, default=INPUT_DIR, help="Input directory")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--min_points", type=int, default=MIN_POINTS, help="Minimum points to keep a label"
    )
    parser.add_argument(
        "--smooth_iterations", type=int, default=2, help="Number of smoothing iterations"
    )
    parser.add_argument("--smooth_k", type=int, default=10, help="Neighbors for smoothing")
    parser.add_argument(
        "--boundary_width",
        type=float,
        default=0.03,
        help="Width of boundary region for plane fitting",
    )
    parser.add_argument(
        "--straighten",
        action="store_true",
        help="Enable boundary straightening using plane fitting (disabled by default)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of parallel workers (0 for serial)"
    )
    parser.add_argument(
        "--skip_existing", action="store_true", help="Skip samples that already have output"
    )

    args = parser.parse_args()

    # Set skip_existing flag on process function
    process_single_sample.skip_existing = args.skip_existing

    # Get list of anno_ids
    if args.anno_id:
        anno_ids = [args.anno_id]
    elif args.csv:
        # Read anno_ids from CSV file
        import pandas as pd

        df = pd.read_csv(args.csv)
        if "anno_id" not in df.columns:
            parser.error(f'CSV file must have "anno_id" column, found: {df.columns.tolist()}')
        anno_ids = df["anno_id"].tolist()
        anno_ids = sorted(anno_ids)
        print(f"Loaded {len(anno_ids)} anno_ids from CSV: {args.csv}")
    elif args.all:
        anno_ids = [
            d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))
        ]
        anno_ids = sorted(anno_ids)
    else:
        parser.error("Must specify --anno_id, --csv, or --all")

    print(f"Processing {len(anno_ids)} samples...")
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Min points: {args.min_points}")
    print(f"Smooth iterations: {args.smooth_iterations}")
    print(f"Straighten boundaries: {args.straighten}")
    if args.straighten:
        print(f"Boundary width: {args.boundary_width}")
    print(f"Num workers: {args.num_workers}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Create partial function with fixed arguments
    process_fn = partial(
        process_single_sample,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_points=args.min_points,
        smooth_iterations=args.smooth_iterations,
        smooth_k=args.smooth_k,
        boundary_width=args.boundary_width,
        straighten_boundaries=args.straighten,
    )

    # Process samples
    if args.num_workers > 1 and len(anno_ids) > 1:
        # Parallel processing
        with Pool(processes=args.num_workers) as pool:
            results = list(
                tqdm(pool.imap(process_fn, anno_ids), total=len(anno_ids), desc="Post-processing")
            )
    else:
        # Serial processing
        results = []
        for anno_id in tqdm(anno_ids, desc="Post-processing"):
            result = process_fn(anno_id)
            results.append(result)

    # Print errors
    for result in results:
        if "error" in result:
            print(f"  {result.get('anno_id', 'unknown')}: ERROR - {result['error']}")

    # Summary
    success = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {len(results)}")
    print(f"Success: {len(success)}")
    print(f"Errors: {len(errors)}")

    if success:
        print(f"\nFirst successful sample: {success[0]}")


if __name__ == "__main__":
    main()
