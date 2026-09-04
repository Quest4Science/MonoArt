from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from monoart.generation import GLB_TO_SLAT, SparseFeatureGrid
from monoart.reasoner.builder import PartAwareSemanticReasoner, build_reasoner
from monoart.reasoner.training import hard_part_infonce


class GenerationContractTests(unittest.TestCase):
    def test_glb_to_slat_rotation(self) -> None:
        point = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        np.testing.assert_array_equal(point @ GLB_TO_SLAT, [[1.0, -3.0, 2.0]])

    def test_sparse_grid_samples_boundary_voxels(self) -> None:
        slat = SimpleNamespace(
            coords=torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]]),
            feats=torch.stack([torch.ones(8), torch.full((8,), 2.0)]),
        )
        grid = SparseFeatureGrid(slat, resolution=2)
        sampled = grid.sample(np.asarray([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float32))
        np.testing.assert_array_equal(sampled[0], np.ones(8))
        np.testing.assert_array_equal(sampled[1], np.full(8, 2.0))


class ReasonerContractTests(unittest.TestCase):
    @staticmethod
    def _config() -> dict:
        return {
            "model": {
                "use_feature_encoder": True,
                "use_2d_feat": False,
                "use_final_mlp": False,
                "triplane_low_res": 4,
                "triplane_high_res": 16,
                "triplane_channels_high": 72,
                "feature_dim": 8,
                "feature_encoder": {
                    "feature_dim": 8,
                    "hidden_dims": [8],
                    "use_residual": True,
                    "dropout": 0.0,
                    "feature_root_paths": {},
                },
                "pvcnn": {
                    "z_triplane_channels": 8,
                    "z_triplane_resolution": 16,
                },
                "transformer": {"dim": 16, "layers": 1, "heads": 4},
                "mlp_num_layers": 2,
                "mlp_hidden_dim": 8,
            }
        }

    def test_small_forward_and_contrastive_backward(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PartAwareSemanticReasoner(build_reasoner(self._config(), device)).train()
        points = torch.rand(1, 32, 3, device=device) - 0.5
        trellis_features = torch.randn(1, 32, 8, device=device)
        output = model(points, trellis_features)
        self.assertEqual(tuple(output.shape), (1, 32, 8))
        labels = torch.arange(4, device=device).repeat_interleave(8).unsqueeze(0)
        loss = hard_part_infonce(
            output,
            labels,
            temperature=0.07,
            anchors_per_object=16,
            negatives_per_anchor=8,
            max_candidates=32,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
