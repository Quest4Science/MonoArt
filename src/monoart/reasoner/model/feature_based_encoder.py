"""Encode point coordinates and TRELLIS features into three feature planes."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_mean


def generate_plane_features(p, c, resolution, plane="xz"):
    """Mean-pool per-point features into one orthographic feature plane."""
    padding = 0.0
    c_dim = c.size(1)
    batch_size = p.size(0)

    xy = normalize_coordinate(p, plane=plane, padding=padding)
    index = coordinate2index(xy, resolution)
    plane_features = c.new_zeros(batch_size, c_dim, resolution**2)
    max_idx = resolution**2 - 1
    index = torch.clamp(index, 0, max_idx)
    plane_features = scatter_mean(c, index, out=plane_features, dim=2)
    return plane_features.reshape(batch_size, c_dim, resolution, resolution)


def normalize_coordinate(p, padding=0.1, plane="xz"):
    """Project ``[B, 3, N]`` coordinates and map them into ``[0, 1)``."""
    axes = {"xz": (0, 2), "xy": (0, 1), "yz": (1, 2)}
    if plane not in axes:
        raise ValueError(f"Unknown plane: {plane}")
    xy = p[:, axes[plane], :].transpose(1, 2)
    return (xy / (1 + padding + 1e-5) + 0.5).clamp(0.0, 1.0 - 1e-5)


def coordinate2index(x, resolution):
    """Convert normalized 2D coordinates to flattened grid indices."""
    cells = (x * resolution).long()
    return (cells[:, :, 0] + resolution * cells[:, :, 1])[:, None, :]


class FeatureMLP(nn.Module):
    """Apply the same multilayer perceptron independently to every point."""

    def __init__(self, input_dim, hidden_dims, output_dim, use_bn=True, dropout=0.1):
        super().__init__()

        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                if use_bn:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        original_shape = x.shape
        if x.ndim == 3:
            batch, count, channels = x.shape
            x = x.reshape(batch * count, channels)

        x = self.mlp(x)

        if len(original_shape) == 3:
            x = x.reshape(original_shape[0], original_shape[1], -1)

        return x


class FeatureBasedEncoder(nn.Module):
    """Fuse 8D TRELLIS features with XYZ and rasterize the result to triplanes."""

    def __init__(
        self,
        cfg,
        device="cuda",
        shape_min=-1.0,
        shape_length=2.0,
        use_2d_feat=False,
    ):
        super().__init__()

        self.device = device
        self.cfg = cfg
        self.shape_min = shape_min
        self.shape_length = shape_length

        input_feature_dim = cfg.feature_dim  # 8
        hidden_dims = cfg.hidden_dims  # [128, 256]
        output_dim = cfg.z_triplane_channels  # 256
        self.z_triplane_resolution = cfg.z_triplane_resolution  # 128
        self.use_residual = cfg.get("use_residual", True)

        self.feature_root_paths = cfg.get("feature_root_paths", {})

        self.feature_mlp = FeatureMLP(
            input_dim=input_feature_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            use_bn=True,
            dropout=cfg.get("dropout", 0.1),
        )

        self.coord_mlp = FeatureMLP(
            input_dim=3,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            use_bn=True,
            dropout=cfg.get("dropout", 0.1),
        )

        if self.use_residual:
            self.fusion_layer = nn.Conv1d(output_dim * 2, output_dim, 1)
        else:
            self.fusion_layer = None

        self.feature_cache = {}

        self._initialize_weights()

    def load_features(self, sample_ids, sources):
        """Load feature arrays for the requested dataset samples.

        This path is used by training and batch evaluation. Single-image inference
        passes preloaded arrays directly and does not perform filesystem lookups here.
        """
        batch_features = []
        valid_indices = []

        for idx, (sample_id, source) in enumerate(zip(sample_ids, sources)):
            cache_key = f"{source}_{sample_id}"
            features = None

            if cache_key in self.feature_cache:
                features = self.feature_cache[cache_key]
            else:
                feature_root = self.feature_root_paths.get(source)
                if feature_root is None:
                    print(f"Sample {idx}: unknown source {source}; skipping {sample_id}")
                    continue

                feature_path = os.path.join(feature_root, f"{sample_id}_100k_features.npz")

                if not os.path.exists(feature_path):
                    print(f"Sample {idx}: feature file not found: {feature_path}")
                    continue
                try:
                    with np.load(feature_path, allow_pickle=False) as data:
                        if "features" not in data:
                            print(
                                f"Sample {idx}: {feature_path} has no 'features' array; "
                                f"available keys: {list(data.keys())}"
                            )
                            continue
                        features_np = np.asarray(data["features"], dtype=np.float32)
                    if features_np.ndim != 2 or features_np.shape[1] != 8:
                        print(
                            f"Sample {idx}: expected [N, 8] features, got "
                            f"{features_np.shape} in {feature_path}"
                        )
                        continue
                    features = torch.from_numpy(features_np).to(self.device)
                except (OSError, ValueError) as exc:
                    print(f"Sample {idx}: failed to read {feature_path}: {exc}")
                    continue

                if features is not None:
                    self.feature_cache[cache_key] = features

            if features is not None:
                batch_features.append(features)
                valid_indices.append(idx)

        if len(batch_features) > 0:
            return torch.stack(batch_features), valid_indices
        return None, []

    def subsample_features(self, features, indices):
        if indices is not None:
            B = features.shape[0]
            sampled_features = []

            for b in range(B):
                sampled = features[b][indices[b]]
                sampled_features.append(sampled)

            return torch.stack(sampled_features)
        else:
            return features

    def encode(
        self,
        point_cloud_xyz,
        point_cloud_feature=None,
        sample_ids=None,
        sources=None,
        sample_indices=None,
        preloaded_features=None,
    ):
        B, N, _ = point_cloud_xyz.shape

        if preloaded_features is not None:
            features_8d = preloaded_features  # [B, N, 8]
            valid_indices = list(range(B))
        elif sample_ids is not None and sources is not None:
            full_features, valid_indices = self.load_features(sample_ids, sources)

            if full_features is None or len(valid_indices) == 0:
                return None, []

            point_cloud_xyz = point_cloud_xyz[valid_indices]  # [valid_B, N, 3]

            if sample_indices is not None:
                sample_indices = [sample_indices[i] for i in valid_indices]
                features_8d = self.subsample_features(
                    full_features, sample_indices
                )  # [valid_B, N, 8]
            else:
                features_8d = full_features[:, :N, :]
        else:
            print("No sample IDs or preloaded features were provided")
            return None, []

        features_256d = self.feature_mlp(features_8d)  # [B, N, 256]

        coord_features = self.coord_mlp(point_cloud_xyz)  # [B, N, 256]

        if self.use_residual:
            combined_features = torch.cat([features_256d, coord_features], dim=-1)  # [B, N, 512]
            combined_features = combined_features.transpose(1, 2)  # [B, 512, N]
            fused_features = self.fusion_layer(combined_features)  # [B, 256, N]
        else:
            fused_features = (features_256d + coord_features).transpose(1, 2)  # [B, 256, N]

        point_cloud_xyz_norm = (point_cloud_xyz - self.shape_min) / self.shape_length
        point_cloud_xyz_norm = point_cloud_xyz_norm - 0.5  # [-0.5, 0.5]

        point_cloud_xyz_norm = point_cloud_xyz_norm.transpose(1, 2)  # [B, 3, N]
        plane_xy = generate_plane_features(
            point_cloud_xyz_norm, fused_features, resolution=self.z_triplane_resolution, plane="xy"
        )

        plane_yz = generate_plane_features(
            point_cloud_xyz_norm, fused_features, resolution=self.z_triplane_resolution, plane="yz"
        )

        plane_xz = generate_plane_features(
            point_cloud_xyz_norm, fused_features, resolution=self.z_triplane_resolution, plane="xz"
        )

        triplane_features = torch.stack(
            [plane_xy, plane_yz, plane_xz], dim=1
        )  # [valid_B, 3, 256, 128, 128]

        return triplane_features, valid_indices

    def forward(self, point_cloud_xyz, point_cloud_feature=None, **kwargs):
        return self.encode(point_cloud_xyz, point_cloud_feature, **kwargs)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
