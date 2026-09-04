from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from PIL import Image

from monoart.asset_validation import validate_asset


class VersionedExampleTests(unittest.TestCase):
    def test_100028_input_and_reference_asset(self) -> None:
        example = Path(__file__).resolve().parents[1] / "examples" / "100028"
        reference = json.loads((example / "reference.json").read_text(encoding="utf-8"))
        input_path = example / reference["input"]["path"]
        digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
        self.assertEqual(digest, reference["input"]["sha256"])
        with Image.open(input_path) as image:
            self.assertEqual(
                image.size,
                (reference["input"]["width"], reference["input"]["height"]),
            )
            self.assertEqual(image.mode, reference["input"]["mode"])

        expected = reference["validated_reference"]
        report = validate_asset(example / expected["path"])
        self.assertTrue(report["ok"])
        self.assertEqual(report["whole_faces"], expected["whole_faces"])
        self.assertEqual(report["independent_faces"], expected["whole_faces"])
        self.assertEqual(report["num_links"], expected["num_links"])
        self.assertLessEqual(
            report["max_raw_alignment_error"],
            expected["max_raw_alignment_error"],
        )
        self.assertLessEqual(
            report["max_urdf_zero_pose_error"],
            expected["max_urdf_zero_pose_error"],
        )
        self.assertEqual(
            {item["link"]: item["joint_type"] for item in report["links"]},
            {item["name"]: item["joint_type"] for item in expected["links"]},
        )
        self.assertEqual(
            {item["link"]: item["faces"] for item in report["links"]},
            {item["name"]: item["faces"] for item in expected["links"]},
        )


if __name__ == "__main__":
    unittest.main()
