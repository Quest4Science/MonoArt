#!/usr/bin/env python3
"""
Map segmentation results back to GLB mesh.

This script takes the segmentation.ply output from inference and maps the
per-point labels back to the original mesh faces, then exports a colored GLB.

Data flow:
1. Read segmentation.ply (points with face_id and label)
2. For each face, determine label by majority voting
3. Assign every unresolved face to a synthetic fixed link
4. Load original mesh_textured.glb
5. Split all faces by label and export a complete GLB

Usage:
    python -m monoart.postprocess.map_segmentation_to_glb --anno_id sample

"""

import argparse
import json
import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import numpy as np
import trimesh
from plyfile import PlyData
from scipy.spatial import cKDTree
from tqdm import tqdm

# ============================================================================
# DEFAULT CONFIGURATION - Modify these paths as needed
# ============================================================================
DEFAULT_SEGMENTATION_DIR = "outputs/postprocess"
DEFAULT_OUTPUT_DIR = "outputs/mesh"
DEFAULT_MESH_DIR = "outputs/generation"
# ============================================================================


# Color palette for different labels (tab20 colormap)
LABEL_COLORS = [
    [31, 119, 180, 255],  # Label 0: blue
    [255, 127, 14, 255],  # Label 1: orange
    [44, 160, 44, 255],  # Label 2: green
    [214, 39, 40, 255],  # Label 3: red
    [148, 103, 189, 255],  # Label 4: purple
    [140, 86, 75, 255],  # Label 5: brown
    [227, 119, 194, 255],  # Label 6: pink
    [127, 127, 127, 255],  # Label 7: gray
    [188, 189, 34, 255],  # Label 8: olive
    [23, 190, 207, 255],  # Label 9: cyan
    [174, 199, 232, 255],  # Label 10: light blue
    [255, 187, 120, 255],  # Label 11: light orange
    [152, 223, 138, 255],  # Label 12: light green
    [255, 152, 150, 255],  # Label 13: light red
    [197, 176, 213, 255],  # Label 14: light purple
    [196, 156, 148, 255],  # Label 15: light brown
    [247, 182, 210, 255],  # Label 16: light pink
    [199, 199, 199, 255],  # Label 17: light gray
    [219, 219, 141, 255],  # Label 18: light olive
    [158, 218, 229, 255],  # Label 19: light cyan
]


def read_segmentation_ply(ply_path: str) -> tuple:
    """
    Read segmentation.ply and extract face_id -> label mapping.
    Supports both ASCII and binary PLY formats.

    Args:
        ply_path: Path to segmentation.ply

    Returns:
        face_ids: [N] array of face IDs
        labels: [N] array of labels
    """
    # Try using plyfile first (supports both ASCII and binary)
    try:
        ply_data = PlyData.read(ply_path)
        vertex = ply_data["vertex"]
        face_ids = np.array(vertex["face_id"])
        labels = np.array(vertex["label"])
        return face_ids, labels
    except Exception:
        pass

    # Fallback to manual ASCII parsing
    with open(ply_path, "r") as f:
        lines = f.readlines()

    # Find header end
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            header_end = i + 1
            break

    # Parse data (format: x y z nx ny nz r g b label face_id)
    labels = []
    face_ids = []

    for line in lines[header_end:]:
        parts = line.strip().split()
        if len(parts) >= 11:
            label = int(parts[9])
            face_id = int(parts[10])
            labels.append(label)
            face_ids.append(face_id)

    return np.array(face_ids), np.array(labels)


def compute_face_labels(
    face_ids: np.ndarray,
    labels: np.ndarray,
    num_faces: int,
    mesh: trimesh.Trimesh = None,
    fill_uncovered: bool = True,
) -> np.ndarray:
    """
    Compute per-face labels using majority voting, with optional nearest-neighbor filling.

    Args:
        face_ids: [N] array of face IDs for each point
        labels: [N] array of labels for each point
        num_faces: Total number of faces in the mesh
        mesh: Trimesh object (required if fill_uncovered=True)
        fill_uncovered: If True, fill uncovered faces using nearest neighbor

    Returns:
        face_labels: [num_faces] array of labels for each face
    """
    # Group points by face_id
    face_to_labels = defaultdict(list)
    for fid, label in zip(face_ids, labels):
        face_to_labels[fid].append(label)

    # Majority voting for each face
    face_labels = np.full(num_faces, -1, dtype=np.int32)

    for fid, point_labels in face_to_labels.items():
        if 0 <= fid < num_faces:
            # Find most common label
            unique, counts = np.unique(point_labels, return_counts=True)
            dominant_label = unique[np.argmax(counts)]
            face_labels[fid] = dominant_label

    # Fill uncovered faces using nearest neighbor
    if fill_uncovered and mesh is not None:
        covered_mask = face_labels >= 0
        uncovered_mask = ~covered_mask
        num_uncovered = np.sum(uncovered_mask)

        if num_uncovered > 0 and np.any(covered_mask):
            # Compute face centroids
            face_centroids = mesh.triangles_center

            # Build KD-tree from covered face centroids
            covered_indices = np.where(covered_mask)[0]
            covered_centroids = face_centroids[covered_indices]

            tree = cKDTree(covered_centroids)

            # For each uncovered face, find nearest covered face
            uncovered_indices = np.where(uncovered_mask)[0]
            uncovered_centroids = face_centroids[uncovered_indices]

            _, nearest_idx = tree.query(uncovered_centroids, k=1)

            # Assign labels from nearest covered face
            for i, uncovered_fid in enumerate(uncovered_indices):
                nearest_covered_fid = covered_indices[nearest_idx[i]]
                face_labels[uncovered_fid] = face_labels[nearest_covered_fid]

    return face_labels


def split_mesh_by_labels(mesh: trimesh.Trimesh, face_labels: np.ndarray) -> trimesh.Scene:
    """
    Split mesh into multiple sub-meshes based on face labels.

    Each label becomes a separate geometry node (link_0, link_1, etc.)
    that can be seen as separate objects in Blender.

    Preserves texture/material information from the original mesh.

    Args:
        mesh: Original mesh
        face_labels: [num_faces] array of labels

    Returns:
        trimesh.Scene with separate geometries for each label
    """
    if len(face_labels) != len(mesh.faces):
        raise ValueError(
            f"Expected one label per face ({len(mesh.faces)}), received {len(face_labels)}"
        )
    if np.any(face_labels < 0):
        unresolved = int(np.sum(face_labels < 0))
        raise ValueError(f"Refusing to drop {unresolved} unresolved mesh faces")

    unique_labels = np.unique(face_labels)

    scene = trimesh.Scene()

    # Check if mesh has texture
    has_texture = (
        mesh.visual is not None
        and isinstance(mesh.visual, trimesh.visual.TextureVisuals)
        and mesh.visual.uv is not None
    )

    for label in unique_labels:
        # Get faces belonging to this label
        face_mask = face_labels == label
        face_indices = np.where(face_mask)[0]

        if len(face_indices) == 0:
            continue

        # Extract faces for this label
        label_faces = mesh.faces[face_indices]

        # Get unique vertices used by these faces
        unique_verts = np.unique(label_faces.flatten())

        # Create vertex index mapping (old -> new)
        vert_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_verts)}

        # Remap face indices
        new_faces = np.array([[vert_map[v] for v in face] for face in label_faces])

        # Extract vertices
        new_vertices = mesh.vertices[unique_verts]

        # Create sub-mesh
        sub_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

        # Preserve texture/material if available
        if has_texture:
            # Extract UV coordinates for the vertices we're keeping
            new_uv = mesh.visual.uv[unique_verts]

            # Create new TextureVisuals with the same material
            sub_mesh.visual = trimesh.visual.TextureVisuals(
                uv=new_uv,
                material=mesh.visual.material,
            )
        elif mesh.visual is not None and hasattr(mesh.visual, "vertex_colors"):
            # Fallback to vertex colors if no texture
            if mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) == len(
                mesh.vertices
            ):
                sub_mesh.visual = trimesh.visual.ColorVisuals(
                    vertex_colors=mesh.visual.vertex_colors[unique_verts]
                )

        # Add to scene with link_X name
        geometry_name = f"link_{label}"
        scene.add_geometry(sub_mesh, node_name=geometry_name, geom_name=geometry_name)

    return scene


def _load_motion_data(motion_json_path: str, anno_id: str) -> dict:
    if os.path.exists(motion_json_path):
        with open(motion_json_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "object_name": anno_id,
        "predicted": True,
        "anno_id": anno_id,
        "coordinate_frame": "world",
        "parts": [],
        "group_info": {},
    }


def _next_label(labels: np.ndarray, motion_data: dict) -> int:
    used_labels = {int(label) for label in labels if int(label) >= 0}
    for raw_label in motion_data.get("group_info", {}):
        try:
            used_labels.add(int(raw_label))
        except (TypeError, ValueError):
            continue
    for part in motion_data.get("parts", []):
        try:
            used_labels.add(int(part["label"]))
        except (KeyError, TypeError, ValueError):
            continue
    return max(used_labels, default=-1) + 1


def _update_motion_data(
    motion_data: dict,
    point_labels: np.ndarray,
    static_label: int | None,
    static_face_count: int,
) -> dict:
    """Synchronize point counts and describe an optional fixed fallback link."""
    point_counts = {
        int(label): int(count)
        for label, count in zip(*np.unique(point_labels[point_labels >= 0], return_counts=True))
    }
    parts = motion_data.setdefault("parts", [])
    for part in parts:
        try:
            label = int(part["label"])
        except (KeyError, TypeError, ValueError):
            continue
        part["num_points"] = point_counts.get(label, 0)

    if static_label is None:
        return motion_data

    static_point_count = int(np.sum(point_labels < 0))
    parts.append(
        {
            "label": static_label,
            "name": "static_fallback",
            "score": 0.0,
            "num_points": static_point_count,
            "num_faces": static_face_count,
            "fallback": True,
        }
    )
    motion_data.setdefault("group_info", {})[str(static_label)] = [
        f"link_{static_label}",
        "base",
        "F",
    ]
    motion_data["static_fallback"] = {
        "label": static_label,
        "link_name": f"link_{static_label}",
        "num_points": static_point_count,
        "num_faces": static_face_count,
        "motion_type": "F",
    }
    return motion_data


def _scene_face_count(scene: trimesh.Scene) -> int:
    return sum(
        len(geometry.faces)
        for geometry in scene.geometry.values()
        if isinstance(geometry, trimesh.Trimesh)
    )


def process_single_sample(
    anno_id: str,
    segmentation_dir: str,
    mesh_dir: str,
    output_dir: str,
) -> dict:
    """
    Process a single sample: map segmentation to GLB.

    Args:
        anno_id: Annotation ID (e.g., "12587_config_1_6_10")
        segmentation_dir: Directory containing segmentation results
        mesh_dir: Directory containing mesh_textured.glb files
        output_dir: Output directory

    Returns:
        Result dictionary with statistics
    """
    # Paths
    seg_ply_path = os.path.join(segmentation_dir, anno_id, "segmentation.ply")
    motion_json_path = os.path.join(segmentation_dir, anno_id, "motion.json")
    mesh_glb_path = os.path.join(mesh_dir, anno_id, "mesh_textured.glb")
    output_glb_path = os.path.join(output_dir, anno_id, "segmented_mesh.glb")

    # Check if already processed (skip_existing)
    if getattr(process_single_sample, "skip_existing", False) and os.path.exists(output_glb_path):
        return {"skipped": True, "reason": "Already processed", "anno_id": anno_id}

    # Check files exist
    if not os.path.exists(seg_ply_path):
        return {"error": f"segmentation.ply not found: {seg_ply_path}"}
    if not os.path.exists(mesh_glb_path):
        return {"error": f"mesh_textured.glb not found: {mesh_glb_path}"}

    # Read segmentation
    face_ids, labels = read_segmentation_ply(seg_ply_path)

    if len(face_ids) != len(labels):
        return {"error": "face_id and label arrays have different lengths", "anno_id": anno_id}

    motion_data = _load_motion_data(motion_json_path, anno_id)

    # Load mesh (preserve texture by not merging)
    scene = trimesh.load(mesh_glb_path)
    if isinstance(scene, trimesh.Scene):
        # Get the first (usually only) mesh - don't concatenate to preserve texture
        meshes = [geom for geom in scene.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if meshes:
            if len(meshes) == 1:
                mesh = meshes[0]
            else:
                # Multiple meshes - need to concatenate but will lose texture
                # Try to preserve texture from first mesh if possible
                mesh = trimesh.util.concatenate(meshes)
                print(f"  Warning: {anno_id} has {len(meshes)} meshes, texture may be lost")
        else:
            return {"error": "No mesh found in GLB"}
    else:
        mesh = scene

    num_faces = len(mesh.faces)
    if num_faces == 0:
        return {"error": "The source mesh has no faces", "anno_id": anno_id}

    # Check face_id range validity
    if len(face_ids) > 0:
        max_face_id = face_ids.max()
        min_face_id = face_ids.min()
        if max_face_id >= num_faces or min_face_id < 0:
            return {
                "skipped": True,
                "reason": f"Face ID mismatch: face_ids range [{min_face_id}, {max_face_id}], but mesh has {num_faces} faces",
                "anno_id": anno_id,
            }

    # Preserve uncertainty instead of borrowing the nearest moving label. Every
    # unresolved face receives a dedicated identity-motion link below.
    face_labels = compute_face_labels(face_ids, labels, num_faces, mesh=mesh, fill_uncovered=False)
    unresolved_mask = face_labels < 0
    static_face_count = int(np.sum(unresolved_mask))
    static_label = None
    if static_face_count:
        static_label = _next_label(face_labels, motion_data)
        face_labels[unresolved_mask] = static_label

    if len(face_labels) != num_faces or np.any(face_labels < 0):
        return {"error": "Failed to assign every source mesh face", "anno_id": anno_id}

    # Split mesh by labels (each label becomes a separate geometry: link_0, link_1, etc.)
    segmented_scene = split_mesh_by_labels(mesh, face_labels)
    scene_face_count = _scene_face_count(segmented_scene)
    if scene_face_count != num_faces:
        return {
            "error": f"Face conservation failed before export: {scene_face_count} != {num_faces}",
            "anno_id": anno_id,
        }

    # Check if scene is empty
    if len(segmented_scene.geometry) == 0:
        return {
            "skipped": True,
            "reason": "Empty segmented scene (no geometries created)",
            "anno_id": anno_id,
        }

    # Create output directory
    sample_output_dir = os.path.join(output_dir, anno_id)
    os.makedirs(sample_output_dir, exist_ok=True)

    # Export GLB (Scene with multiple geometries)
    output_glb_path = os.path.join(sample_output_dir, "segmented_mesh.glb")
    segmented_scene.export(output_glb_path)

    exported_scene = trimesh.load(output_glb_path, force="scene")
    exported_face_count = _scene_face_count(exported_scene)
    if exported_face_count != num_faces:
        return {
            "error": f"Face conservation failed after export: {exported_face_count} != {num_faces}",
            "anno_id": anno_id,
        }

    # The mesh-stage motion file is authoritative for animation and future
    # URDF export because it includes the synthetic fixed link when required.
    motion_data = _update_motion_data(
        motion_data,
        labels,
        static_label=static_label,
        static_face_count=static_face_count,
    )
    with open(os.path.join(sample_output_dir, "motion.json"), "w", encoding="utf-8") as handle:
        json.dump(motion_data, handle, indent=2)

    # Statistics
    unique_labels = np.unique(face_labels[face_labels >= 0])
    stats = {
        "anno_id": anno_id,
        "num_faces": num_faces,
        "exported_num_faces": exported_face_count,
        "num_points": len(labels),
        "unique_labels": unique_labels.tolist(),
        "label_counts": {int(label): int(np.sum(face_labels == label)) for label in unique_labels},
        "unresolved_faces_before_fallback": static_face_count,
        "static_fallback_label": static_label,
        "static_fallback_faces": static_face_count,
        "all_faces_assigned": bool(np.all(face_labels >= 0)),
        "face_count_conserved": exported_face_count == num_faces,
        "output_path": output_glb_path,
    }

    # Save stats
    stats_path = os.path.join(sample_output_dir, "mapping_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Map segmentation results to GLB mesh")
    parser.add_argument("--anno_id", type=str, default=None, help="Single annotation ID to process")
    parser.add_argument(
        "--all", action="store_true", help="Process all samples in segmentation directory"
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="CSV file with anno_id column to filter samples"
    )
    parser.add_argument(
        "--segmentation_dir",
        type=str,
        default=DEFAULT_SEGMENTATION_DIR,
        help="Directory containing segmentation results",
    )
    parser.add_argument(
        "--mesh_dir",
        type=str,
        default=DEFAULT_MESH_DIR,
        help="Directory containing mesh_textured.glb files",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--skip_existing", action="store_true", help="Skip samples that already have output"
    )
    args = parser.parse_args()

    # Set skip_existing flag on process function
    process_single_sample.skip_existing = args.skip_existing

    # Get list of anno_ids to process
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
        # Get all subdirectories in segmentation_dir
        anno_ids = [
            d
            for d in os.listdir(args.segmentation_dir)
            if os.path.isdir(os.path.join(args.segmentation_dir, d))
        ]
        anno_ids = sorted(anno_ids)
    else:
        parser.error("Must specify --anno_id, --csv, or --all")

    print(f"Processing {len(anno_ids)} samples...")
    print(f"Segmentation dir: {args.segmentation_dir}")
    print(f"Mesh dir: {args.mesh_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Num workers: {args.num_workers}")
    print()

    # Create partial function with fixed arguments
    process_fn = partial(
        process_single_sample,
        segmentation_dir=args.segmentation_dir,
        mesh_dir=args.mesh_dir,
        output_dir=args.output_dir,
    )

    # Process samples
    if args.num_workers > 1 and len(anno_ids) > 1:
        # Parallel processing
        with Pool(processes=args.num_workers) as pool:
            results = list(
                tqdm(pool.imap(process_fn, anno_ids), total=len(anno_ids), desc="Mapping")
            )
    else:
        # Serial processing
        results = []
        for anno_id in tqdm(anno_ids, desc="Mapping"):
            result = process_fn(anno_id)
            results.append(result)

    # Print errors
    for result in results:
        if "error" in result:
            print(f"  {result.get('anno_id', 'unknown')}: ERROR - {result['error']}")

    # Summary
    success = [r for r in results if "error" not in r and "skipped" not in r]
    skipped = [r for r in results if r.get("skipped")]
    errors = [r for r in results if "error" in r]

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {len(results)}")
    print(f"Success: {len(success)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Errors: {len(errors)}")

    # Show skip reasons breakdown
    if skipped:
        skip_reasons = defaultdict(list)
        for r in skipped:
            skip_reasons[r.get("reason", "Unknown")].append(r.get("anno_id"))
        print("\nSkip reasons:")
        for reason, ids in skip_reasons.items():
            print(f"  - {reason}: {len(ids)} samples")

    if success:
        print("\nFirst successful sample:")
        print(f"  {success[0]}")


if __name__ == "__main__":
    main()
