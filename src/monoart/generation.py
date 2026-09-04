"""TRELLIS reconstruction and point-aligned SLAT feature extraction."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import numpy as np
import torch
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement

# This row-vector rotation matches the feature extraction used to train MonoArt.
# It maps a GLB point (x, y, z) to (x, -z, y) before querying SLAT.
GLB_TO_SLAT = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


class SparseFeatureGrid:
    """Dense lookup grid for trilinear interpolation of sparse SLAT features."""

    def __init__(self, slat: Any, resolution: int = 64) -> None:
        coordinates = slat.coords.detach().cpu().numpy().astype(np.int64)
        features = slat.feats.detach().float().cpu().numpy()
        if coordinates.ndim != 2 or coordinates.shape[1] not in (3, 4):
            raise ValueError(f"Unexpected SLAT coordinates: {coordinates.shape}")
        if features.ndim != 2 or features.shape[1] != 8:
            raise ValueError(f"Expected 8D SLAT features, got {features.shape}")
        if coordinates.shape[1] == 4:
            coordinates = coordinates[:, 1:]

        self.resolution = resolution
        self.grid = np.zeros((resolution, resolution, resolution, 8), np.float32)
        self.occupied = np.zeros((resolution, resolution, resolution), dtype=bool)
        valid = np.all((coordinates >= 0) & (coordinates < resolution), axis=1)
        coordinates = coordinates[valid]
        features = features[valid]
        x, y, z = coordinates.T
        self.grid[x, y, z] = features
        self.occupied[x, y, z] = True

    def sample(self, points: np.ndarray) -> np.ndarray:
        """Trilinearly interpolate features at points in TRELLIS world space."""
        coordinates = (np.asarray(points, np.float32) + 0.5) * self.resolution - 0.5
        coordinates = np.clip(coordinates, 0.0, self.resolution - 1.0)
        lower = np.floor(coordinates).astype(np.int64)
        upper = np.minimum(lower + 1, self.resolution - 1)
        fraction = coordinates - lower

        output = np.zeros((len(points), 8), dtype=np.float32)
        for x_bit in (0, 1):
            for y_bit in (0, 1):
                for z_bit in (0, 1):
                    index = np.column_stack(
                        [
                            upper[:, 0] if x_bit else lower[:, 0],
                            upper[:, 1] if y_bit else lower[:, 1],
                            upper[:, 2] if z_bit else lower[:, 2],
                        ]
                    )
                    weight = (
                        (fraction[:, 0] if x_bit else 1.0 - fraction[:, 0])
                        * (fraction[:, 1] if y_bit else 1.0 - fraction[:, 1])
                        * (fraction[:, 2] if z_bit else 1.0 - fraction[:, 2])
                    )
                    ix, iy, iz = index.T
                    values = self.grid[ix, iy, iz]
                    values = values * self.occupied[ix, iy, iz, None]
                    output += values * weight[:, None]
        return output


def _sample_texture(mesh: trimesh.Trimesh, face_ids: np.ndarray) -> np.ndarray:
    colors = np.full((len(face_ids), 3), 128, dtype=np.uint8)
    uv = getattr(mesh.visual, "uv", None)
    material = getattr(mesh.visual, "material", None)
    texture = getattr(material, "baseColorTexture", None)
    if uv is None or texture is None:
        return colors

    texture_array = np.asarray(texture)
    if texture_array.ndim != 3 or texture_array.shape[2] < 3:
        return colors
    height, width = texture_array.shape[:2]
    sample_uv = np.asarray(uv)[mesh.faces[face_ids]].mean(axis=1)
    x = np.clip((sample_uv[:, 0] * width).astype(int), 0, width - 1)
    y = np.clip(((1.0 - sample_uv[:, 1]) * height).astype(int), 0, height - 1)
    return texture_array[y, x, :3].astype(np.uint8)


def sample_mesh(
    mesh: trimesh.Trimesh,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a textured mesh while retaining the originating face index."""
    if count <= 0:
        raise ValueError("Point count must be positive")
    rng = np.random.default_rng(seed)
    face_count = len(mesh.faces)
    centroids = mesh.vertices[mesh.faces].mean(axis=1)

    if face_count >= count:
        selected = rng.choice(face_count, size=count, replace=False)
        points = centroids[selected]
        face_ids = selected
    else:
        # trimesh uses NumPy's legacy global generator internally.
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            area_points, area_faces = trimesh.sample.sample_surface(mesh, count=count - face_count)
        finally:
            np.random.set_state(state)
        points = np.vstack([centroids, area_points])
        face_ids = np.concatenate([np.arange(face_count), area_faces])

    face_ids = np.asarray(face_ids, dtype=np.int32)
    normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float32)
    colors = _sample_texture(mesh, face_ids)
    return np.asarray(points, np.float32), normals, colors, face_ids


def write_point_cloud(
    path: Path,
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    face_ids: np.ndarray,
) -> None:
    """Write the point attributes consumed by all downstream MonoArt stages."""
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("face_id", "i4"),
        ],
    )
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, index]
    for index, name in enumerate(("nx", "ny", "nz")):
        vertices[name] = normals[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, index]
    vertices["face_id"] = face_ids
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


class TrellisGenerator:
    """Generate a textured object and point-aligned SLAT features from one image."""

    def __init__(
        self,
        model: str = "JeffreyXiang/TRELLIS-image-large",
        trellis_root: str | Path | None = None,
    ) -> None:
        if trellis_root is not None:
            root = str(Path(trellis_root).expanduser().resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
        try:
            from trellis.pipelines import TrellisImageTo3DPipeline
            from trellis.utils import postprocessing_utils
        except ImportError as exc:
            raise RuntimeError(
                "TRELLIS is not importable. Pass --trellis-root or install the "
                "official TRELLIS repository in this environment."
            ) from exc

        self.postprocessing = postprocessing_utils
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(model)
        self.pipeline.cuda()

    def generate(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        *,
        seed: int = 42,
        num_points: int = 100_000,
        sparse_steps: int = 25,
        sparse_cfg: float = 5.0,
        slat_steps: int = 25,
        slat_cfg: float = 5.0,
        simplify: float = 0.95,
        texture_size: int = 1024,
        save_gaussian: bool = False,
    ) -> Path:
        image_path = Path(image_path).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()

        image = Image.open(image_path).convert("RGBA")
        shutil.copy2(image_path, output_dir / f"input{image_path.suffix.lower()}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Sampling and decoding do not need gradients. Texture baking below does:
        # TRELLIS optimizes a texture tensor with back-propagation inside ``to_glb``.
        with torch.no_grad():
            processed = self.pipeline.preprocess_image(image)
            condition = self.pipeline.get_cond([processed])
            coordinates = self.pipeline.sample_sparse_structure(
                condition,
                num_samples=1,
                sampler_params={"steps": sparse_steps, "cfg_strength": sparse_cfg},
            )
            slat = self.pipeline.sample_slat(
                condition,
                coordinates,
                sampler_params={"steps": slat_steps, "cfg_strength": slat_cfg},
            )
            decoded = self.pipeline.decode_slat(slat, formats=["mesh", "gaussian"])
        np.savez_compressed(
            output_dir / "slat_latent.npz",
            coords=slat.coords.detach().cpu().numpy(),
            feats=slat.feats.detach().float().cpu().numpy(),
        )

        mesh_result = decoded["mesh"][0]
        gaussian_result = decoded["gaussian"][0]
        if not mesh_result.success:
            raise RuntimeError("TRELLIS failed to decode a mesh")

        mesh_result.vertices = mesh_result.vertices.detach()
        mesh_result.faces = mesh_result.faces.detach()
        if mesh_result.vertex_attrs is not None:
            mesh_result.vertex_attrs = mesh_result.vertex_attrs.detach()
        raw_mesh = trimesh.Trimesh(
            vertices=mesh_result.vertices.cpu().numpy(),
            faces=mesh_result.faces.cpu().numpy(),
            process=False,
        )
        raw_mesh.export(output_dir / "mesh.ply")

        glb_mesh = self.postprocessing.to_glb(
            gaussian_result,
            mesh_result,
            simplify=simplify,
            texture_size=texture_size,
            verbose=True,
        )
        glb_path = output_dir / "mesh_textured.glb"
        glb_mesh.export(glb_path)

        points, normals, colors, face_ids = sample_mesh(glb_mesh, num_points, seed)
        point_path = output_dir / f"sampled_glb_{num_points}.ply"
        write_point_cloud(point_path, points, normals, colors, face_ids)

        feature_grid = SparseFeatureGrid(slat)
        features = feature_grid.sample(points @ GLB_TO_SLAT)
        feature_path = output_dir / f"features_glb_{num_points}.npz"
        np.savez_compressed(
            feature_path,
            features=features,
            num_points=np.int64(num_points),
            feat_dim=np.int64(features.shape[1]),
        )

        if save_gaussian:
            gaussian_result.save_ply(str(output_dir / "gaussian.ply"))

        metadata = {
            "input_image": str(image_path),
            "seed": seed,
            "num_points": num_points,
            "model_inputs": {
                "sparse_steps": sparse_steps,
                "sparse_cfg": sparse_cfg,
                "slat_steps": slat_steps,
                "slat_cfg": slat_cfg,
            },
            "mesh": {
                "vertices": int(len(glb_mesh.vertices)),
                "faces": int(len(glb_mesh.faces)),
            },
            "slat": {
                "voxels": int(slat.feats.shape[0]),
                "feature_dim": int(slat.feats.shape[1]),
            },
            "elapsed_seconds": time.perf_counter() - start,
        }
        (output_dir / "generation.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        del decoded, mesh_result, gaussian_result, slat
        gc.collect()
        torch.cuda.empty_cache()
        return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trellis-root", type=Path)
    parser.add_argument("--model", default="JeffreyXiang/TRELLIS-image-large")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-points", type=int, default=100_000)
    parser.add_argument("--sparse-steps", type=int, default=25)
    parser.add_argument("--sparse-cfg", type=float, default=5.0)
    parser.add_argument("--slat-steps", type=int, default=25)
    parser.add_argument("--slat-cfg", type=float, default=5.0)
    parser.add_argument("--simplify", type=float, default=0.95)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--save-gaussian", action="store_true")
    args = parser.parse_args()

    generator = TrellisGenerator(args.model, args.trellis_root)
    result = generator.generate(
        args.image,
        args.output,
        seed=args.seed,
        num_points=args.num_points,
        sparse_steps=args.sparse_steps,
        sparse_cfg=args.sparse_cfg,
        slat_steps=args.slat_steps,
        slat_cfg=args.slat_cfg,
        simplify=args.simplify,
        texture_size=args.texture_size,
        save_gaussian=args.save_gaussian,
    )
    print(f"Saved TRELLIS outputs to {result}")


if __name__ == "__main__":
    main()
