"""End-to-end orchestration across the TRELLIS and MonoArt environments."""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .motion.inference import infer_motion
from .postprocess.animate_segmented_glb import process_single_sample as animate_sample
from .postprocess.export_urdf import AssetPaths, export_urdf_asset
from .postprocess.map_segmentation_to_glb import process_single_sample as map_sample
from .postprocess.postprocess_segmentation import process_single_sample as clean_sample
from .reasoner.inference import infer_part_features


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    sample_id: str

    @property
    def generation(self) -> Path:
        return self.root / "generation" / self.sample_id

    @property
    def reasoner(self) -> Path:
        return self.root / "reasoner" / self.sample_id

    @property
    def motion(self) -> Path:
        return self.root / "motion" / self.sample_id

    @property
    def postprocess(self) -> Path:
        return self.root / "postprocess" / self.sample_id

    @property
    def mesh(self) -> Path:
        return self.root / "mesh" / self.sample_id

    @property
    def animation(self) -> Path:
        return self.root / "animation" / self.sample_id

    def as_dict(self) -> dict[str, str]:
        """Return every stage directory for the run manifest."""
        return {
            "root": str(self.root),
            "generation": str(self.generation),
            "reasoner": str(self.reasoner),
            "motion": str(self.motion),
            "postprocess": str(self.postprocess),
            "mesh": str(self.mesh),
            "animation": str(self.animation),
        }


def _sample_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not normalized:
        raise ValueError("The sample ID is empty after normalization")
    return normalized


def _require_success(stage: str, result: dict) -> None:
    if "error" in result:
        raise RuntimeError(f"{stage} failed: {result['error']}")
    if result.get("skipped") and result.get("reason") != "Already processed":
        raise RuntimeError(f"{stage} skipped the sample: {result.get('reason')}")


def _prepare_workspace(root: Path, *, resume: bool) -> Path:
    """Create or resume the generated workspace beside the final asset directory."""
    if not root.name:
        raise ValueError("The filesystem root cannot be used as an output directory")
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        if not root.is_dir():
            raise NotADirectoryError(root)
        if any(root.iterdir()):
            raise FileExistsError(
                f"Final output directory is not empty: {root}. Choose a new directory."
            )

    workspace = root.with_name(f".{root.name}.monoart-work")
    if resume:
        if not workspace.is_dir():
            raise FileNotFoundError(
                f"No resumable workspace exists at {workspace}. Run without --resume first."
            )
    else:
        if workspace.exists():
            raise FileExistsError(
                f"A previous workspace exists at {workspace}. Use --resume or move it aside."
            )
        workspace.mkdir(parents=True)
    return workspace


def _publish_asset(asset_stage: Path, root: Path) -> AssetPaths:
    """Atomically replace an empty destination with a validated asset directory."""
    if root.exists():
        if any(root.iterdir()):
            raise FileExistsError(f"Final output directory became non-empty: {root}")
        root.rmdir()
    asset_stage.replace(root)
    return AssetPaths(root)


def _run_trellis(
    image: Path,
    output: Path,
    *,
    trellis_python: Path,
    trellis_root: Path | None,
    seed: int,
    num_points: int,
    sparse_steps: int,
    slat_steps: int,
    texture_size: int,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    python_paths = [str(source_root)]
    if trellis_root is not None:
        python_paths.append(str(trellis_root.resolve()))
    if existing := os.environ.get("PYTHONPATH"):
        python_paths.append(existing)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["PYTHONNOUSERSITE"] = "1"

    command = [
        str(trellis_python),
        "-m",
        "monoart.generation",
        "--image",
        str(image),
        "--output",
        str(output),
        "--seed",
        str(seed),
        "--num-points",
        str(num_points),
        "--sparse-steps",
        str(sparse_steps),
        "--slat-steps",
        str(slat_steps),
        "--texture-size",
        str(texture_size),
    ]
    if trellis_root is not None:
        command.extend(["--trellis-root", str(trellis_root)])
    subprocess.run(command, check=True, env=environment)
    expected = [
        output / "mesh_textured.glb",
        output / f"sampled_glb_{num_points}.ply",
        output / f"features_glb_{num_points}.npz",
    ]
    missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"TRELLIS completed without producing valid outputs: {missing}")


def run_pipeline(
    image: str | Path,
    output: str | Path,
    checkpoint: str | Path,
    *,
    motion_config: str | Path | None = None,
    trellis_python: str | Path = sys.executable,
    trellis_root: str | Path | None = None,
    sample_id: str | None = None,
    device: str = "cuda",
    seed: int = 42,
    num_points: int = 100_000,
    sparse_steps: int = 25,
    slat_steps: int = 25,
    texture_size: int = 1024,
    animation_frames: int = 0,
    resume: bool = False,
    keep_intermediates: bool = False,
) -> AssetPaths:
    """Run inference and publish a compact URDF asset bundle.

    By default, the final directory contains only ``model.urdf``, ``whole.glb``,
    and ``meshes/link_*.glb``. Generated stage outputs are retained only when
    ``keep_intermediates`` is enabled.
    """
    image = Path(image).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    root = Path(output).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if animation_frames < 0:
        raise ValueError("animation_frames cannot be negative")
    if animation_frames > 0 and not keep_intermediates:
        raise ValueError("animation_frames requires keep_intermediates=True")
    identifier = _sample_id(sample_id or image.stem)
    workspace = _prepare_workspace(root, resume=resume)
    paths = PipelinePaths(root=workspace, sample_id=identifier)
    started = time.perf_counter()

    generation_outputs = [
        paths.generation / "mesh_textured.glb",
        paths.generation / f"sampled_glb_{num_points}.ply",
        paths.generation / f"features_glb_{num_points}.npz",
    ]
    if not (resume and all(path.is_file() for path in generation_outputs)):
        _run_trellis(
            image,
            paths.generation,
            trellis_python=Path(trellis_python),
            trellis_root=Path(trellis_root) if trellis_root else None,
            seed=seed,
            num_points=num_points,
            sparse_steps=sparse_steps,
            slat_steps=slat_steps,
            texture_size=texture_size,
        )

    point_cloud = paths.generation / f"sampled_glb_{num_points}.ply"
    trellis_features = paths.generation / f"features_glb_{num_points}.npz"
    part_features = paths.reasoner / f"points_{num_points}_feat.npy"
    if not (resume and part_features.is_file()):
        infer_part_features(
            point_cloud,
            trellis_features,
            checkpoint,
            part_features,
            device=device,
        )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    motion_json = paths.motion / "motion.json"
    has_tree = False
    if not (resume and motion_json.is_file()):
        _, has_tree = infer_motion(
            point_cloud,
            part_features,
            trellis_features,
            checkpoint,
            paths.motion,
            config=motion_config,
            sample_id=identifier,
            device=device,
        )
    elif (paths.motion / "kinematic_tree.json").is_file():
        has_tree = True
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not (resume and (paths.postprocess / "segmentation.ply").is_file()):
        result = clean_sample(
            identifier,
            str(workspace / "motion"),
            str(workspace / "postprocess"),
            min_points=200,
        )
        _require_success("Segmentation post-processing", result)

    if not (resume and (paths.mesh / "segmented_mesh.glb").is_file()):
        result = map_sample(
            identifier,
            str(workspace / "postprocess"),
            str(workspace / "generation"),
            str(workspace / "mesh"),
        )
        _require_success("Mesh mapping", result)

    if animation_frames > 0:
        animation_info = paths.animation / "animation_info.json"
        if not (resume and animation_info.is_file()):
            if animation_frames == 1:
                ratios = [0.0]
            else:
                ratios = [index / (animation_frames - 1) for index in range(animation_frames)]
            result = animate_sample(
                identifier,
                str(workspace / "mesh"),
                str(workspace / "mesh"),
                str(workspace / "animation"),
                ratios=ratios,
                render_video=False,
                include_fixed=True,
            )
            _require_success("Animation export", result)

    manifest = {
        "sample_id": identifier,
        "input_image": str(image),
        "checkpoint": str(checkpoint),
        "contains_kinematic_estimator": has_tree,
        "num_points": num_points,
        "seed": seed,
        "elapsed_seconds": time.perf_counter() - started,
        "paths": paths.as_dict(),
    }
    (workspace / "run.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    asset_stage = workspace / "asset"
    if asset_stage.exists():
        if not resume:
            raise FileExistsError(asset_stage)
        shutil.rmtree(asset_stage)
    export_urdf_asset(
        paths.mesh / "segmented_mesh.glb",
        paths.mesh / "motion.json",
        asset_stage,
    )
    assets = _publish_asset(asset_stage, root)

    if keep_intermediates:
        workspace.replace(root / "work")
    else:
        shutil.rmtree(workspace)
    return assets
