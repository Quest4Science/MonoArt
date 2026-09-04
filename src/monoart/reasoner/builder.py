"""Build the semantic reasoner architecture stored in a MonoArt checkpoint."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .config import ConfigNode
from .model import FeatureBasedEncoder, TriplaneTransformer, VanillaMLP, sample_triplane_features


class PartAwareSemanticReasoner(nn.Module):
    """Trainable composition of the released semantic-reasoner components."""

    def __init__(self, components: dict[str, nn.Module | None]) -> None:
        super().__init__()
        encoder = components["encoder"]
        transformer = components["triplane_transformer"]
        if encoder is None or transformer is None:
            raise ValueError("The encoder and triplane transformer are required")
        self.encoder = encoder
        self.triplane_transformer = transformer
        self.part_decoder = components["part_decoder"]

    def forward(self, points: torch.Tensor, trellis_features: torch.Tensor) -> torch.Tensor:
        triplanes, valid_indices = self.encoder(
            points,
            preloaded_features=trellis_features,
            sample_ids=None,
            sources=None,
        )
        expected_indices = list(range(points.shape[0]))
        if triplanes is None or valid_indices != expected_indices:
            raise RuntimeError("The semantic encoder rejected one or more batch samples")

        enhanced = self.triplane_transformer(triplanes)
        if enhanced.shape[2] <= 64:
            raise RuntimeError(f"Unexpected triplane shape: {tuple(enhanced.shape)}")
        point_features = sample_triplane_features(enhanced[:, :, 64:], points)
        if self.part_decoder is not None:
            batch, count, channels = point_features.shape
            point_features = self.part_decoder(point_features.reshape(-1, channels)).reshape(
                batch, count, -1
            )
        return point_features


def build_reasoner(config: dict[str, Any], device: torch.device) -> dict[str, nn.Module | None]:
    """Construct checkpoint-compatible reasoner components.

    The released model uses the feature-based encoder. The legacy PVCNN branch is
    intentionally excluded because it is neither referenced by the released weights
    nor by the image-to-articulation inference path.
    """
    model_cfg = config["model"]
    if not model_cfg.get("use_feature_encoder", False):
        raise ValueError(
            "This release supports the feature-based reasoner encoder only; "
            "the checkpoint requests the legacy PVCNN encoder."
        )

    feature_cfg = ConfigNode(model_cfg["feature_encoder"])
    feature_cfg.z_triplane_channels = model_cfg["pvcnn"]["z_triplane_channels"]
    feature_cfg.z_triplane_resolution = model_cfg["pvcnn"]["z_triplane_resolution"]
    if not hasattr(feature_cfg, "feature_root_paths"):
        feature_cfg.feature_root_paths = {}

    encoder = FeatureBasedEncoder(
        cfg=feature_cfg,
        device=device,
        use_2d_feat=model_cfg.get("use_2d_feat", False),
    )
    triplane_transformer = TriplaneTransformer(
        input_dim=model_cfg["pvcnn"]["z_triplane_channels"],
        transformer_dim=model_cfg["transformer"]["dim"],
        transformer_layers=model_cfg["transformer"]["layers"],
        transformer_heads=model_cfg["transformer"]["heads"],
        triplane_low_res=model_cfg["triplane_low_res"],
        triplane_high_res=model_cfg["triplane_high_res"],
        triplane_dim=model_cfg["triplane_channels_high"],
    )

    part_decoder: nn.Module | None = None
    if model_cfg.get("use_final_mlp", True):
        part_decoder = VanillaMLP(
            input_dim=model_cfg["triplane_channels_high"] - 64,
            output_dim=model_cfg["feature_dim"],
            out_activation="RELU",
            n_hidden_layers=model_cfg.get("mlp_num_layers", 4),
            n_neurons=model_cfg.get("mlp_hidden_dim", 64),
            activation="ReLU",
        )

    components: dict[str, nn.Module | None] = {
        "encoder": encoder,
        "triplane_transformer": triplane_transformer,
        "part_decoder": part_decoder,
    }
    for component in components.values():
        if component is not None:
            component.to(device).eval()
    return components


def load_component_state(
    component: nn.Module | None,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    *,
    required: bool = True,
) -> None:
    """Load one prefixed component and reject incompatible checkpoints."""
    if component is None:
        return
    prefix_with_dot = f"{prefix}."
    component_state = {
        key.removeprefix(prefix_with_dot): value
        for key, value in state_dict.items()
        if key.startswith(prefix_with_dot)
    }
    if required and not component_state:
        raise KeyError(f"Checkpoint has no parameters with prefix '{prefix_with_dot}'")
    if component_state:
        missing, unexpected = component.load_state_dict(component_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint is incompatible with {prefix}: "
                f"missing={list(missing)}, unexpected={list(unexpected)}"
            )
