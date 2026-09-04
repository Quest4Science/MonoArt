"""Creation and inspection of self-contained MonoArt inference bundles."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml

FORMAT_VERSION = 1


def load_torch(path: str | Path, map_location: Any = "cpu") -> Any:
    """Load a trusted project checkpoint with an explicit pickle policy."""
    return torch.load(path, map_location=map_location, weights_only=False)


def motion_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the motion section of a bundle or a legacy motion checkpoint."""
    if payload.get("monoart_format_version") == FORMAT_VERSION:
        return payload["motion"]
    return payload


def _reasoner_payload(checkpoint: Path) -> dict[str, Any]:
    payload = load_torch(checkpoint)
    try:
        source_config = payload["hyper_parameters"]["config"]
    except (KeyError, TypeError) as exc:
        raise ValueError("The reasoner checkpoint must embed hyper_parameters.config") from exc
    state = payload.get("state_dict", payload)
    prefixes = ("encoder.", "triplane_transformer.", "part_decoder.")
    state = {key: value.cpu() for key, value in state.items() if key.startswith(prefixes)}
    if not state:
        raise ValueError("No reasoner parameters were found")
    try:
        model_config = deepcopy(source_config["model"])
        model_config["feature_encoder"]["feature_root_paths"] = {}
    except (KeyError, TypeError) as exc:
        raise ValueError("The reasoner checkpoint has no usable model configuration") from exc
    config = {"model": model_config}
    return {"config": config, "state_dict": state}


def _motion_payload(checkpoint: Path, config: Path) -> dict[str, Any]:
    payload = motion_payload(load_torch(checkpoint))
    state = payload.get("model_state_dict")
    if state is None:
        if not all(isinstance(value, torch.Tensor) for value in payload.values()):
            raise ValueError("No model_state_dict was found in the motion checkpoint")
        state = payload
    result: dict[str, Any] = {
        "config": yaml.safe_load(config.read_text(encoding="utf-8")),
        "model_state_dict": {key: value.cpu() for key, value in state.items()},
        "epoch": payload.get("epoch", -1),
    }
    if "parent_head_state_dict" in payload:
        result["parent_head_state_dict"] = {
            key: value.cpu() for key, value in payload["parent_head_state_dict"].items()
        }
    return result


def pack_checkpoint(
    reasoner_checkpoint: str | Path,
    motion_checkpoint: str | Path,
    motion_config: str | Path,
    output: str | Path,
) -> Path:
    """Strip optimizer state and combine the two MonoArt-owned modules."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "monoart_format_version": FORMAT_VERSION,
        "reasoner": _reasoner_payload(Path(reasoner_checkpoint)),
        "motion": _motion_payload(Path(motion_checkpoint), Path(motion_config)),
        "metadata": {
            "contains_kinematic_estimator": False,
            "note": (
                "Set automatically from parent_head_state_dict presence; "
                "TRELLIS weights are distributed separately."
            ),
        },
    }
    bundle["metadata"]["contains_kinematic_estimator"] = (
        "parent_head_state_dict" in bundle["motion"]
    )
    torch.save(bundle, output)
    return output


def sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    """Return a JSON-serializable checkpoint summary without tensor values."""
    path = Path(path)
    payload = load_torch(path)
    if payload.get("monoart_format_version") == FORMAT_VERSION:
        motion = payload["motion"]
        return {
            "path": path.name,
            "format_version": payload["monoart_format_version"],
            "size_bytes": path.stat().st_size,
            "reasoner_tensors": len(payload["reasoner"]["state_dict"]),
            "motion_tensors": len(motion["model_state_dict"]),
            "contains_kinematic_estimator": "parent_head_state_dict" in motion,
            "motion_epoch": motion.get("epoch", -1),
        }
    return {
        "path": path.name,
        "format_version": "legacy",
        "size_bytes": path.stat().st_size,
        "top_level_keys": sorted(payload.keys()),
    }


def write_manifest(checkpoint: str | Path, output: str | Path) -> Path:
    summary = inspect_checkpoint(checkpoint)
    summary["sha256"] = sha256(checkpoint)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output


def verify_checkpoint(
    checkpoint: str | Path,
    manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Verify checkpoint size and SHA-256 without deserializing the checkpoint."""
    checkpoint = Path(checkpoint)
    manifest = (
        Path(manifest)
        if manifest is not None
        else checkpoint.with_suffix(checkpoint.suffix + ".json")
    )
    expected = json.loads(manifest.read_text(encoding="utf-8"))

    failures = []
    if expected.get("path") != checkpoint.name:
        failures.append(f"filename is {checkpoint.name!r}, expected {expected.get('path')!r}")
    size = checkpoint.stat().st_size
    if expected.get("size_bytes") != size:
        failures.append(f"size is {size}, expected {expected.get('size_bytes')}")
    digest = sha256(checkpoint)
    if expected.get("sha256") != digest:
        failures.append(f"SHA-256 is {digest}, expected {expected.get('sha256')}")
    if failures:
        raise ValueError("Checkpoint verification failed: " + "; ".join(failures))

    return {
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "size_bytes": size,
        "sha256": digest,
        "verified": True,
    }
