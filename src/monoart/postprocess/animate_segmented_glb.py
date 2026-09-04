#!/usr/bin/env python3
"""
Animate segmented GLB based on motion parameters from JSON.

This script:
1. Loads segmented GLB (with link_0, link_1, etc.)
2. Applies motion transformations based on motion.json
3. Exports multiple GLB files at different motion ratios
4. Renders animation video

Usage:
    python animate_segmented_glb.py --anno_id 45135_config_1_6_10
    python animate_segmented_glb.py --anno_id 45135_config_1_6_10 --ratios 0.0 0.5 1.0
    python animate_segmented_glb.py --all
"""

import argparse
import json
import math
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ============================================================================
# DEFAULT CONFIGURATION - Modify these paths as needed
# ============================================================================
DEFAULT_SEGMENTED_GLB_DIR = "outputs/mesh"
DEFAULT_MOTION_JSON_DIR = "outputs/postprocess"
DEFAULT_OUTPUT_DIR = "outputs/animation"


def rotation_matrix_from_axis_angle(axis, angle):
    """
    Create a 4x4 rotation matrix from axis and angle.

    Args:
        axis: [3] normalized axis vector
        angle: rotation angle in radians

    Returns:
        4x4 transformation matrix
    """
    axis = np.array(axis)
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    rot = Rotation.from_rotvec(axis * angle)
    mat = np.eye(4)
    mat[:3, :3] = rot.as_matrix()
    return mat


def translation_matrix(direction, distance):
    """
    Create a 4x4 translation matrix.

    Args:
        direction: [3] normalized direction vector
        distance: translation distance

    Returns:
        4x4 transformation matrix
    """
    direction = np.array(direction)
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    mat = np.eye(4)
    mat[:3, 3] = direction * distance
    return mat


def apply_motion_to_mesh(mesh, motion_type, axis_dir, axis_origin, limits, ratio):
    """
    Apply motion transformation to a mesh.

    Args:
        mesh: trimesh.Trimesh object
        motion_type: 'R' (revolute), 'P' (prismatic), 'C' (continuous), 'F' (fixed)
        axis_dir: [3] axis direction
        axis_origin: [3] axis origin point
        limits: [lower, upper] motion limits
        ratio: motion ratio (0.0 to 1.0)

    Returns:
        Transformed mesh (copy)
    """
    if motion_type == "F":
        return mesh.copy()

    # Make a copy to avoid modifying original
    transformed = mesh.copy()

    axis_dir = np.array(axis_dir)
    axis_origin = np.array(axis_origin)

    if motion_type in ["R", "C"]:  # Revolute or Continuous
        # Calculate rotation angle
        if limits is not None and len(limits) >= 2:
            # Convert normalized values to radians (multiply by pi)
            # JSON stores values normalized by pi for training convenience
            lower = limits[0] * math.pi
            upper = limits[1] * math.pi
            angle = lower + (upper - lower) * ratio
        else:
            # Default rotation range for continuous: 0 to 2*pi
            angle = 2 * math.pi * ratio

        # Create rotation around axis at origin
        # 1. Translate to origin
        # 2. Rotate
        # 3. Translate back
        T_to_origin = np.eye(4)
        T_to_origin[:3, 3] = -axis_origin

        R_mat = rotation_matrix_from_axis_angle(axis_dir, angle)

        T_from_origin = np.eye(4)
        T_from_origin[:3, 3] = axis_origin

        # Combined transform: T_from @ R @ T_to
        transform = T_from_origin @ R_mat @ T_to_origin
        transformed.apply_transform(transform)

    elif motion_type == "P":  # Prismatic
        # Calculate translation distance
        if limits is not None and len(limits) >= 2:
            lower, upper = limits[0], limits[1]
            distance = lower + (upper - lower) * ratio
        else:
            distance = ratio * 0.5  # Default 0.5 unit range

        T_mat = translation_matrix(axis_dir, distance)
        transformed.apply_transform(T_mat)

    return transformed


def load_motion_params(motion_json_path):
    """
    Load motion parameters from JSON file.

    Returns:
        dict: {label: {'type': 'R/P/C/F', 'axis_dir': [...], 'axis_origin': [...], 'limits': [...]}}
    """
    with open(motion_json_path, "r") as f:
        data = json.load(f)

    motion_params = {}
    group_info = data.get("group_info", {})

    for label, joint_data in group_info.items():
        label = int(label)
        parent_link = joint_data[1]  # e.g., "base"

        if len(joint_data) == 3 and joint_data[2] == "F":
            # Fixed joint
            motion_params[label] = {
                "type": "F",
                "axis_dir": None,
                "axis_origin": None,
                "limits": None,
                "parent": parent_link,
            }
        elif len(joint_data) == 4:
            params = joint_data[2]
            motion_type = joint_data[3]

            motion_params[label] = {
                "type": motion_type,
                "axis_dir": params[:3],
                "axis_origin": params[3:6],
                "limits": params[6:8] if len(params) >= 8 else None,
                "parent": parent_link,
            }

    return motion_params, data


def animate_glb(glb_path, motion_params, ratio):
    """
    Create an animated version of the GLB at a specific motion ratio.

    Args:
        glb_path: Path to segmented GLB
        motion_params: Motion parameters dict
        ratio: Motion ratio (0.0 to 1.0)

    Returns:
        trimesh.Scene with animated meshes
    """
    # Load the segmented GLB
    scene = trimesh.load(glb_path)

    if not isinstance(scene, trimesh.Scene):
        print(f"Warning: {glb_path} is not a Scene, skipping")
        return scene

    # Create new scene with animated meshes
    new_scene = trimesh.Scene()

    for geom_name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue

        # Extract label from geometry name (e.g., "link_0" -> 0)
        if geom_name.startswith("link_"):
            try:
                label = int(geom_name.split("_")[1])
            except (ValueError, IndexError):
                label = -1
        else:
            label = -1

        # Apply motion if we have params for this label
        if label in motion_params:
            params = motion_params[label]
            transformed = apply_motion_to_mesh(
                geom,
                params["type"],
                params["axis_dir"],
                params["axis_origin"],
                params["limits"],
                ratio,
            )
        else:
            transformed = geom.copy()

        new_scene.add_geometry(transformed, node_name=geom_name, geom_name=geom_name)

    return new_scene


def texture_to_vertex_colors(mesh):
    """
    Convert texture to vertex colors for rendering compatibility.

    Args:
        mesh: trimesh.Trimesh with TextureVisuals

    Returns:
        trimesh.Trimesh with ColorVisuals (vertex colors)
    """
    try:
        if not (
            hasattr(mesh, "visual")
            and isinstance(mesh.visual, trimesh.visual.TextureVisuals)
            and mesh.visual.uv is not None
            and mesh.visual.material is not None
        ):
            return mesh

        # Get the texture image
        material = mesh.visual.material
        if hasattr(material, "baseColorTexture") and material.baseColorTexture is not None:
            texture = material.baseColorTexture
        elif hasattr(material, "image") and material.image is not None:
            texture = material.image
        else:
            return mesh

        # Convert to numpy array
        tex_array = np.array(texture)
        tex_h, tex_w = tex_array.shape[:2]

        # Get UV coordinates
        uv = mesh.visual.uv
        # Clamp UV to [0, 1]
        uv = np.clip(uv, 0, 1)

        # Convert UV to pixel coordinates
        px = (uv[:, 0] * (tex_w - 1)).astype(int)
        py = ((1 - uv[:, 1]) * (tex_h - 1)).astype(int)  # Flip Y

        # Sample colors from texture
        if len(tex_array.shape) == 3:
            vertex_colors = tex_array[py, px]
            # Add alpha channel if not present
            if vertex_colors.shape[1] == 3:
                alpha = np.full((len(vertex_colors), 1), 255, dtype=np.uint8)
                vertex_colors = np.hstack([vertex_colors, alpha])
        else:
            # Grayscale
            gray = tex_array[py, px]
            vertex_colors = np.stack([gray, gray, gray, np.full_like(gray, 255)], axis=1)

        # Create new mesh with vertex colors
        colored_mesh = trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces.copy(),
            vertex_colors=vertex_colors,
            process=False,
        )
        return colored_mesh

    except Exception:
        return mesh


def render_scene(scene, output_path, resolution=(512, 512), use_texture=True):
    """
    Render scene to image using pyrender with original textures.

    Args:
        scene: trimesh.Scene
        output_path: Output image path
        resolution: (width, height)
        use_texture: Whether to try using original textures (converted to vertex colors)
    """
    try:
        import pyrender
        from PIL import Image

        os.environ["PYOPENGL_PLATFORM"] = "egl"  # For headless rendering

        # Create pyrender scene
        pr_scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.4, 0.4, 0.4])

        # Colors for different links (used when texture fails or disabled)
        fallback_colors = [
            [0.12, 0.47, 0.71, 1.0],  # Blue
            [1.0, 0.5, 0.05, 1.0],  # Orange
            [0.17, 0.63, 0.17, 1.0],  # Green
            [0.84, 0.15, 0.16, 1.0],  # Red
        ]

        for i, (geom_name, geom) in enumerate(scene.geometry.items()):
            if not isinstance(geom, trimesh.Trimesh):
                continue

            render_geom = geom

            # Convert texture to vertex colors for reliable rendering
            if use_texture:
                render_geom = texture_to_vertex_colors(geom)

            # Check if we have vertex colors now
            has_vertex_colors = (
                hasattr(render_geom, "visual")
                and hasattr(render_geom.visual, "vertex_colors")
                and render_geom.visual.vertex_colors is not None
            )

            if has_vertex_colors:
                # Render with vertex colors
                pr_mesh = pyrender.Mesh.from_trimesh(render_geom, smooth=True)
            else:
                # Fallback to solid color
                color = fallback_colors[i % len(fallback_colors)]
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=color,
                    metallicFactor=0.2,
                    roughnessFactor=0.8,
                )
                pr_mesh = pyrender.Mesh.from_trimesh(render_geom, material=material)

            pr_scene.add(pr_mesh)

        # Calculate camera position based on scene bounds
        bounds = scene.bounds
        center = (bounds[0] + bounds[1]) / 2
        size = np.linalg.norm(bounds[1] - bounds[0])

        # Camera looking at center from front-top-right
        camera_distance = size * 1.8
        camera_pos = center + np.array(
            [camera_distance * 0.5, camera_distance * 0.3, camera_distance * 0.8]
        )

        # Create camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)

        # Camera pose (look at center)
        forward = center - camera_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.array([0, 1, 0]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        camera_pose = np.eye(4)
        camera_pose[:3, 0] = right
        camera_pose[:3, 1] = up
        camera_pose[:3, 2] = -forward
        camera_pose[:3, 3] = camera_pos

        pr_scene.add(camera, pose=camera_pose)

        # Add lights
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        pr_scene.add(light, pose=camera_pose)

        # Add ambient light from opposite direction
        light2_pose = np.eye(4)
        light2_pose[:3, 3] = center - (camera_pos - center)
        light2 = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
        pr_scene.add(light2, pose=light2_pose)

        # Render
        renderer = pyrender.OffscreenRenderer(*resolution)
        color_img, _ = renderer.render(pr_scene)
        renderer.delete()

        # Save image
        Image.fromarray(color_img).save(output_path)
        return True

    except Exception as e:
        # If texture rendering fails, try again without textures
        if use_texture:
            return render_scene(scene, output_path, resolution, use_texture=False)
        print(f"Rendering failed: {e}")
        return False


def create_animation_video(image_dir, output_path, fps=10):
    """
    Create video from rendered images.

    Args:
        image_dir: Directory containing frame images
        output_path: Output video path
        fps: Frames per second
    """
    try:
        import cv2

        # Get all frame images
        frames = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

        if not frames:
            print("No frames found for video")
            return False

        # Read first frame to get dimensions
        first_frame = cv2.imread(os.path.join(image_dir, frames[0]))
        height, width = first_frame.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame_file in frames:
            frame = cv2.imread(os.path.join(image_dir, frame_file))
            video.write(frame)

        # Add reverse for smooth loop
        for frame_file in reversed(frames[1:-1]):
            frame = cv2.imread(os.path.join(image_dir, frame_file))
            video.write(frame)

        video.release()
        return True

    except ImportError:
        print("OpenCV not available, skipping video creation")
        return False
    except Exception as e:
        print(f"Video creation failed: {e}")
        return False


def process_single_sample(
    anno_id: str,
    segmented_glb_dir: str,
    motion_json_dir: str,
    output_dir: str,
    ratios: list = None,
    render_video: bool = True,
    include_fixed: bool = False,
) -> dict:
    """
    Process a single sample: animate and render.

    Args:
        anno_id: Annotation ID
        segmented_glb_dir: Directory containing segmented GLB files
        motion_json_dir: Directory containing motion JSON files
        output_dir: Output directory
        ratios: List of motion ratios (default: 30 frames from 0.0 to 1.0)
        render_video: Whether to render video
        include_fixed: Whether to include samples with all fixed parts (generate identical GLBs)

    Returns:
        Result dictionary
    """
    if ratios is None:
        # Default: 30 frames for smooth animation
        ratios = [i / 29 for i in range(30)]

    # Paths
    glb_path = os.path.join(segmented_glb_dir, anno_id, "segmented_mesh.glb")
    motion_json_path = os.path.join(motion_json_dir, anno_id, "motion.json")

    # Check files exist
    if not os.path.exists(glb_path):
        return {"error": f"segmented_mesh.glb not found: {glb_path}", "anno_id": anno_id}
    if not os.path.exists(motion_json_path):
        return {"error": f"motion.json not found: {motion_json_path}", "anno_id": anno_id}

    # Load motion parameters
    motion_params, motion_data = load_motion_params(motion_json_path)

    # Check if all parts are fixed (no movable parts)
    has_movable = any(p["type"] != "F" for p in motion_params.values())
    if not has_movable and not include_fixed:
        return {
            "skipped": True,
            "reason": "All parts are fixed (no movable joints)",
            "anno_id": anno_id,
        }

    # Create output directory
    sample_output_dir = os.path.join(output_dir, anno_id)
    os.makedirs(sample_output_dir, exist_ok=True)

    # Create frames directory for rendering
    frames_dir = os.path.join(sample_output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Process each ratio
    output_glbs = []
    for i, ratio in enumerate(ratios):
        # Animate
        animated_scene = animate_glb(glb_path, motion_params, ratio)

        # Export GLB
        # Include the frame index and sufficient precision to avoid overwriting
        # nearby ratios when exporting more than eleven frames.
        ratio_str = f"{i:03d}_{ratio:.3f}".replace(".", "_")
        glb_output_path = os.path.join(sample_output_dir, f"motion_{ratio_str}.glb")
        animated_scene.export(glb_output_path)
        output_glbs.append(glb_output_path)

        # Render frame
        if render_video:
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            render_scene(animated_scene, frame_path)

    # Create video
    video_path = None
    if render_video:
        video_path = os.path.join(sample_output_dir, "animation.mp4")
        create_animation_video(frames_dir, video_path)

    # Copy motion.json
    import shutil

    shutil.copy(motion_json_path, os.path.join(sample_output_dir, "motion.json"))

    # Result
    result = {
        "anno_id": anno_id,
        "ratios": ratios,
        "output_glbs": output_glbs,
        "video_path": video_path,
        "motion_types": {k: v["type"] for k, v in motion_params.items()},
    }

    # Save result
    with open(os.path.join(sample_output_dir, "animation_info.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Animate segmented GLB based on motion parameters")
    parser.add_argument("--anno_id", type=str, default=None, help="Single annotation ID to process")
    parser.add_argument("--all", action="store_true", help="Process all samples")
    parser.add_argument(
        "--csv", type=str, default=None, help="CSV file with anno_id column to filter samples"
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=None,  # Will use num_frames to generate
        help="Motion ratios to export (default: auto-generate from num_frames)",
    )
    parser.add_argument(
        "--num_frames", type=int, default=30, help="Number of frames for animation (default: 30)"
    )
    parser.add_argument("--no-video", action="store_true", help="Skip video rendering")
    parser.add_argument(
        "--segmented_glb_dir",
        type=str,
        default=DEFAULT_SEGMENTED_GLB_DIR,
        help="Directory containing segmented GLB files",
    )
    parser.add_argument(
        "--motion_json_dir",
        type=str,
        default=DEFAULT_MOTION_JSON_DIR,
        help="Directory containing motion JSON files",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 1, video rendering may not work with >1)",
    )
    parser.add_argument(
        "--include-fixed",
        action="store_true",
        help="Include samples with all fixed parts (generate identical GLBs for each ratio)",
    )

    args = parser.parse_args()

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
            d
            for d in os.listdir(args.segmented_glb_dir)
            if os.path.isdir(os.path.join(args.segmented_glb_dir, d))
        ]
        anno_ids = sorted(anno_ids)
    else:
        parser.error("Must specify --anno_id, --csv, or --all")

    # Generate ratios from num_frames if not specified
    if args.ratios is None:
        ratios = [i / (args.num_frames - 1) for i in range(args.num_frames)]
    else:
        ratios = args.ratios

    print(f"Processing {len(anno_ids)} samples...")
    print(f"Segmented GLB dir: {args.segmented_glb_dir}")
    print(f"Motion JSON dir: {args.motion_json_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Num frames: {len(ratios)}")
    print(f"Num workers: {args.num_workers}")
    if args.num_workers > 1 and not args.no_video:
        print(
            "  Warning: Video rendering may not work with multiple workers. Consider using --no-video"
        )
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Create partial function with fixed arguments
    process_fn = partial(
        process_single_sample,
        segmented_glb_dir=args.segmented_glb_dir,
        motion_json_dir=args.motion_json_dir,
        output_dir=args.output_dir,
        ratios=ratios,
        render_video=not args.no_video,
        include_fixed=args.include_fixed,
    )

    # Process samples
    if args.num_workers > 1 and len(anno_ids) > 1:
        # Parallel processing
        with Pool(processes=args.num_workers) as pool:
            results = list(
                tqdm(pool.imap(process_fn, anno_ids), total=len(anno_ids), desc="Animating")
            )
    else:
        # Serial processing
        results = []
        for anno_id in tqdm(anno_ids, desc="Animating"):
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
    print(f"Skipped (all fixed): {len(skipped)}")
    print(f"Errors: {len(errors)}")

    if success:
        print("\nFirst successful sample:")
        r = success[0]
        print(f"  Anno ID: {r['anno_id']}")
        print(f"  Motion types: {r['motion_types']}")
        print(f"  Output GLBs: {len(r['output_glbs'])} files")


if __name__ == "__main__":
    main()
