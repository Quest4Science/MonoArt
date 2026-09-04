"""Direct single-object inference for the Dual-Query Motion Decoder."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
import yaml

from monoart.checkpoints import load_torch, motion_payload

from .scripts.inference import create_model, load_checkpoint_with_parent_head
from .scripts.inference_single import (
    load_single_sample,
    run_single_inference,
    save_single_outputs,
)


def load_inference_config(
    checkpoint: str | Path,
    config: str | Path | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load architecture settings from a bundle or a standalone YAML file."""
    if isinstance(config, dict):
        return copy.deepcopy(config)
    if config is not None:
        return yaml.safe_load(Path(config).read_text(encoding="utf-8"))

    payload = load_torch(checkpoint)
    motion = motion_payload(payload)
    if "config" not in motion:
        raise ValueError("A legacy motion checkpoint requires --config; MonoArt bundles embed it.")
    return copy.deepcopy(motion["config"])


def infer_motion(
    point_cloud: str | Path,
    part_features: str | Path,
    trellis_features: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    config: str | Path | dict[str, Any] | None = None,
    sample_id: str = "sample",
    device: str = "cuda",
) -> tuple[Path, bool]:
    """Infer per-point parts, joint parameters, and an optional kinematic tree."""
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    inference_config = load_inference_config(checkpoint, config)
    inference_config.setdefault("data", {})["single_sample"] = {
        "ply_path": str(Path(point_cloud)),
        "partfield_path": str(Path(part_features)),
        "vae_path": str(Path(trellis_features)),
        "sample_id": sample_id,
    }
    inference_config["data"].setdefault("n_points", 100_000)
    inference_config.setdefault("inference", {})["device"] = device
    inference_config.setdefault("output", {})["dir"] = str(output_dir)

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    batch = load_single_sample(inference_config)
    model = create_model(inference_config)
    model, parent_head, has_parent_prediction = load_checkpoint_with_parent_head(
        model,
        str(checkpoint),
        requested,
        inference_config,
    )
    model = model.to(requested).eval()
    result = run_single_inference(
        model,
        batch,
        requested,
        inference_config,
        parent_head=parent_head,
    )
    save_single_outputs(result, batch, str(output_dir), inference_config)
    return output_dir, has_parent_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-cloud", required=True, type=Path)
    parser.add_argument("--part-features", required=True, type=Path)
    parser.add_argument("--trellis-features", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-id", default="sample")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output, has_tree = infer_motion(
        args.point_cloud,
        args.part_features,
        args.trellis_features,
        args.checkpoint,
        args.output,
        config=args.config,
        sample_id=args.sample_id,
        device=args.device,
    )
    print(f"Saved motion predictions to {output}")
    if not has_tree:
        print(
            "Warning: this checkpoint has no Kinematic Estimator weights; "
            "all links are attached to the base."
        )


if __name__ == "__main__":
    main()
