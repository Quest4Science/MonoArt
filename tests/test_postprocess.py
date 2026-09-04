from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement

from monoart.postprocess.map_segmentation_to_glb import process_single_sample


class MeshMappingContractTests(unittest.TestCase):
    @staticmethod
    def _write_case(root: Path, labels: np.ndarray) -> tuple[Path, Path, Path]:
        sample = "sample"
        segmentation_root = root / "postprocess"
        generation_root = root / "generation"
        output_root = root / "mesh"
        (segmentation_root / sample).mkdir(parents=True)
        (generation_root / sample).mkdir(parents=True)

        mesh = trimesh.creation.box()
        mesh.export(generation_root / sample / "mesh_textured.glb")

        vertices = np.empty(len(labels), dtype=[("face_id", "i4"), ("label", "i4")])
        vertices["face_id"] = np.arange(len(labels), dtype=np.int32)
        vertices["label"] = labels
        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(
            segmentation_root / sample / "segmentation.ply"
        )
        motion = {
            "object_name": "box",
            "predicted": True,
            "anno_id": sample,
            "coordinate_frame": "world",
            "parts": [{"label": 0, "name": "part_0", "score": 0.9, "num_points": 99}],
            "group_info": {"0": ["link_0", "base", "F"]},
        }
        (segmentation_root / sample / "motion.json").write_text(
            json.dumps(motion), encoding="utf-8"
        )
        return segmentation_root, generation_root, output_root

    def test_unassigned_faces_become_fixed_and_all_faces_survive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.asarray([0] * 6 + [-1] * 6, dtype=np.int32)
            segmentation, generation, output = self._write_case(root, labels)

            result = process_single_sample(
                "sample", str(segmentation), str(generation), str(output)
            )

            self.assertEqual(result["num_faces"], 12)
            self.assertEqual(result["exported_num_faces"], 12)
            self.assertEqual(result["static_fallback_faces"], 6)
            self.assertTrue(result["all_faces_assigned"])
            self.assertTrue(result["face_count_conserved"])

            scene = trimesh.load(output / "sample" / "segmented_mesh.glb", force="scene")
            exported_faces = sum(len(mesh.faces) for mesh in scene.geometry.values())
            self.assertEqual(exported_faces, 12)

            motion = json.loads((output / "sample" / "motion.json").read_text())
            static_label = str(result["static_fallback_label"])
            self.assertEqual(
                motion["group_info"][static_label], [f"link_{static_label}", "base", "F"]
            )
            self.assertEqual(motion["static_fallback"]["num_faces"], 6)
            self.assertEqual(motion["parts"][0]["num_points"], 6)

    def test_all_unassigned_faces_produce_one_fixed_complete_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.full(12, -1, dtype=np.int32)
            segmentation, generation, output = self._write_case(root, labels)

            result = process_single_sample(
                "sample", str(segmentation), str(generation), str(output)
            )

            self.assertEqual(result["static_fallback_faces"], 12)
            self.assertEqual(result["label_counts"], {1: 12})
            self.assertTrue(result["face_count_conserved"])


if __name__ == "__main__":
    unittest.main()
