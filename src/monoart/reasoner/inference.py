"""Inference interface for MonoArt's Part-Aware Semantic Reasoner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from plyfile import PlyData

from .builder import PartAwareSemanticReasoner, build_reasoner, load_component_state


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("monoart_format_version") == 1:
        reasoner = payload["reasoner"]
        return reasoner["config"], reasoner["state_dict"]

    try:
        config = payload["hyper_parameters"]["config"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "The reasoner checkpoint does not contain its training configuration. "
            "Use a MonoArt bundle or the original Lightning checkpoint."
        ) from exc
    return config, payload.get("state_dict", payload)


def _read_points(path: Path) -> np.ndarray:
    vertex = PlyData.read(path)["vertex"]
    return np.column_stack([np.asarray(vertex[axis], dtype=np.float32) for axis in ("x", "y", "z")])


def _read_features(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "features" not in payload:
            raise KeyError(f"{path} does not contain an array named 'features'")
        return np.asarray(payload["features"], dtype=np.float32)


class PartReasoner:
    """Load the semantic reasoner once and apply it to point-feature pairs."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda") -> None:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.device = requested
        config, state_dict = _load_checkpoint(Path(checkpoint))
        self.components = build_reasoner(config, self.device)
        load_component_state(self.components["encoder"], state_dict, "encoder")
        load_component_state(
            self.components["triplane_transformer"],
            state_dict,
            "triplane_transformer",
        )
        load_component_state(
            self.components["part_decoder"],
            state_dict,
            "part_decoder",
            required=False,
        )
        self.model = PartAwareSemanticReasoner(self.components).eval()

    @torch.inference_mode()
    def __call__(self, points: np.ndarray, features: np.ndarray) -> np.ndarray:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points shaped [N, 3], got {points.shape}")
        if features.ndim != 2 or features.shape[1] != 8:
            raise ValueError(f"Expected TRELLIS features shaped [N, 8], got {features.shape}")
        if len(points) != len(features):
            raise ValueError(f"Point/feature count mismatch: {len(points)} versus {len(features)}")

        points_tensor = torch.as_tensor(points, device=self.device).unsqueeze(0)
        features_tensor = torch.as_tensor(features, device=self.device).unsqueeze(0)
        point_features = self.model(points_tensor, features_tensor)
        return point_features[0].float().cpu().numpy()


def infer_part_features(
    point_cloud: str | Path,
    trellis_features: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    device: str = "cuda",
) -> Path:
    """Run semantic reasoning and save an ``[N, 448]`` NumPy array."""
    point_cloud = Path(point_cloud)
    trellis_features = Path(trellis_features)
    output = Path(output)
    points = _read_points(point_cloud)
    features = _read_features(trellis_features)
    result = PartReasoner(checkpoint, device=device)(points, features)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-cloud", required=True, type=Path)
    parser.add_argument("--trellis-features", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = infer_part_features(
        args.point_cloud,
        args.trellis_features,
        args.checkpoint,
        args.output,
        args.device,
    )
    print(f"Saved semantic features to {output}")


if __name__ == "__main__":
    main()
