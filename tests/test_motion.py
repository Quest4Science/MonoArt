import unittest

import torch

from monoart.motion.losses.combined_loss import CombinedLoss
from monoart.motion.models.articulated_maft import ArticulatedMAFT
from monoart.motion.relation import ParentPredictionHead, ParentPredictionLoss, resolve_cycles


class MotionContractTests(unittest.TestCase):
    def test_small_forward_loss_and_backward(self) -> None:
        torch.manual_seed(7)
        model = ArticulatedMAFT(
            partfield_dim=16,
            vae_dim=8,
            d_model=32,
            d_global=16,
            num_queries=4,
            num_decoder_layers=2,
            nhead=4,
            dim_feedforward=64,
            dropout=0.0,
            num_categories=3,
            num_motion_types=4,
            use_partfield_for_category=True,
            partfield_proj_dim=8,
            semantic_fusion_config=None,
            part_geometric_config={"enabled": False},
            iterative_fusion_config=None,
            rpe_config={"enabled": False},
        )
        points = torch.rand(1, 24, 3) - 0.5
        predictions = model(
            partfield_features=torch.randn(1, 24, 16),
            vae_features=torch.randn(1, 24, 8),
            points=points,
        )
        self.assertEqual(tuple(predictions["mask_logits"].shape), (1, 4, 24))

        targets = {
            "points": points,
            "group_ids": torch.tensor([[0] * 12 + [1] * 12]),
            "categories": torch.tensor([1]),
            "gt_motion_types": [torch.tensor([0, 2])],
            "gt_axis_directions": [torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])],
            "gt_axis_positions": [torch.zeros(2, 3)],
            "gt_projected_origins": [torch.zeros(2, 3)],
        }
        loss_fn = CombinedLoss(
            num_categories=3,
            motion_warmup_epochs=1,
            matcher_chunk_size=2,
            use_affinity_loss=False,
            use_limit_loss=False,
            use_part_class_loss=False,
            use_center_loss=False,
        )
        losses = loss_fn(predictions, targets, epoch=1)
        self.assertTrue(torch.isfinite(losses["total_loss"]))
        losses["total_loss"].backward()
        gradients = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_kinematic_head_loss_and_cycle_resolution(self) -> None:
        torch.manual_seed(11)
        head = ParentPredictionHead(d_model=16, use_position=True, use_semantic=True)
        content = torch.randn(1, 3, 16)
        position = torch.randn(1, 3, 16)
        classes = torch.softmax(torch.randn(1, 3, 18), dim=-1)
        logits = head(content, position, classes)
        self.assertEqual(tuple(logits.shape), (1, 3, 4))

        loss = ParentPredictionLoss()(
            logits,
            [(torch.tensor([0, 1]), torch.tensor([0, 1]))],
            [{10: -1, 20: 10}],
            [torch.tensor([10, 20])],
        )
        self.assertEqual(int(loss["parent_num_valid"]), 2)
        loss["parent_loss"].backward()
        self.assertIsNotNone(head.child_proj.weight.grad)

        cyclic_logits = torch.full((3, 4), -10.0)
        cyclic_logits[0, 1] = 10.0
        cyclic_logits[1, 0] = 9.0
        cyclic_logits[2, 1] = 8.0
        parents = resolve_cycles(cyclic_logits)
        self.assertEqual(tuple(parents.shape), (3,))
        self.assertTrue(any(int(parent) == 3 for parent in parents))


if __name__ == "__main__":
    unittest.main()
