from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from monoart.checkpoints import inspect_checkpoint, sha256, verify_checkpoint, write_manifest


class CheckpointContractTests(unittest.TestCase):
    def test_bundle_inspection_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "test.pt"
            torch.save(
                {
                    "monoart_format_version": 1,
                    "reasoner": {
                        "config": {"model": {}},
                        "state_dict": {"encoder.weight": torch.ones(1)},
                    },
                    "motion": {
                        "config": {"model": {}},
                        "model_state_dict": {"weight": torch.ones(1)},
                        "epoch": 3,
                    },
                    "metadata": {"contains_kinematic_estimator": False},
                },
                checkpoint,
            )
            summary = inspect_checkpoint(checkpoint)
            self.assertEqual(summary["format_version"], 1)
            self.assertFalse(summary["contains_kinematic_estimator"])
            manifest = write_manifest(checkpoint, root / "test.pt.json")
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["sha256"], sha256(checkpoint))
            verification = verify_checkpoint(checkpoint)
            self.assertTrue(verification["verified"])

            with checkpoint.open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "verification failed"):
                verify_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
