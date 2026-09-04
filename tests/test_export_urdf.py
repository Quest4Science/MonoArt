from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from monoart.asset_validation import validate_asset
from monoart.pipeline import run_pipeline
from monoart.postprocess.export_urdf import export_urdf_asset


def _world_bounds(path: Path) -> np.ndarray:
    scene = trimesh.load(path, force="scene")
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node)
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    return trimesh.util.concatenate(meshes).bounds


class UrdfAssetExportTests(unittest.TestCase):
    @staticmethod
    def _write_input(
        root: Path,
        parent: str = "base",
        motion_type: str = "R",
    ) -> tuple[Path, Path]:
        scene = trimesh.Scene()
        fixed = trimesh.creation.box(extents=[1.0, 2.0, 3.0])
        moving = trimesh.creation.box(extents=[0.5, 0.75, 1.0])
        fixed_transform = trimesh.transformations.translation_matrix([0.5, 0.0, 0.0])
        moving_transform = trimesh.transformations.translation_matrix([2.0, 3.0, 4.0])
        scene.add_geometry(
            fixed,
            node_name="link_0",
            geom_name="link_0",
            transform=fixed_transform,
        )
        scene.add_geometry(
            moving,
            node_name="link_1",
            geom_name="link_1",
            transform=moving_transform,
        )
        segmented = root / "segmented.glb"
        scene.export(segmented)
        motion = {
            "object_name": "Test Object",
            "coordinate_frame": "world",
            "group_info": {
                "0": ["link_0", "base", "F"],
                "1": [
                    "link_1",
                    parent,
                    [0.0, 0.0, 2.0, 1.0, 2.0, 3.0, -0.25, 0.75],
                    motion_type,
                ],
            },
        }
        motion_path = root / "motion.json"
        motion_path.write_text(json.dumps(motion), encoding="utf-8")
        return segmented, motion_path

    def test_compact_bundle_preserves_faces_and_zero_pose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segmented, motion = self._write_input(root)
            output = root / "asset"

            paths = export_urdf_asset(segmented, motion, output)

            files = sorted(
                path.relative_to(output).as_posix()
                for path in output.rglob("*")
                if path.is_file()
            )
            self.assertEqual(
                files,
                ["meshes/link_0.glb", "meshes/link_1.glb", "model.urdf", "whole.glb"],
            )
            whole = trimesh.load(paths.whole, force="scene")
            individual_faces = sum(
                sum(
                    len(mesh.faces)
                    for mesh in trimesh.load(path, force="scene").geometry.values()
                )
                for path in paths.meshes.glob("*.glb")
            )
            self.assertEqual(
                individual_faces,
                sum(len(mesh.faces) for mesh in whole.geometry.values()),
            )

            moving_world_bounds = None
            for node in whole.graph.nodes_geometry:
                if str(node) == "link_1":
                    transform, geometry_name = whole.graph.get(node)
                    moving_mesh = whole.geometry[geometry_name].copy()
                    moving_mesh.apply_transform(transform)
                    moving_world_bounds = moving_mesh.bounds
            self.assertIsNotNone(moving_world_bounds)
            independent_bounds = _world_bounds(paths.meshes / "link_1.glb")
            np.testing.assert_allclose(
                independent_bounds,
                moving_world_bounds,
                atol=1e-6,
            )

            robot = ET.parse(paths.urdf).getroot()
            self.assertEqual(robot.attrib["name"], "Test_Object")
            moving_joint = robot.find("./joint[@name='joint_link_1']")
            self.assertIsNotNone(moving_joint)
            self.assertEqual(moving_joint.attrib["type"], "revolute")
            self.assertEqual(moving_joint.find("origin").attrib["xyz"], "1 2 3")
            self.assertEqual(moving_joint.find("axis").attrib["xyz"], "0 0 1")
            moving_link = robot.find("./link[@name='link_1']")
            self.assertEqual(
                moving_link.find("visual/origin").attrib["xyz"],
                "-1 -2 -3",
            )
            self.assertEqual(
                moving_link.find("collision/origin").attrib["xyz"],
                "-1 -2 -3",
            )
            report = validate_asset(paths.root)
            self.assertTrue(report["ok"])
            self.assertEqual(report["num_links"], 2)
            self.assertEqual(report["whole_faces"], 24)
            self.assertEqual(report["independent_faces"], 24)

    def test_prismatic_origin_is_ignored_and_mesh_stays_world_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segmented, motion = self._write_input(root, motion_type="P")

            paths = export_urdf_asset(segmented, motion, root / "asset")

            whole = trimesh.load(paths.whole, force="scene")
            moving_world_bounds = None
            for node in whole.graph.nodes_geometry:
                if str(node) == "link_1":
                    transform, geometry_name = whole.graph.get(node)
                    moving_mesh = whole.geometry[geometry_name].copy()
                    moving_mesh.apply_transform(transform)
                    moving_world_bounds = moving_mesh.bounds
            self.assertIsNotNone(moving_world_bounds)
            np.testing.assert_allclose(
                _world_bounds(paths.meshes / "link_1.glb"),
                moving_world_bounds,
                atol=1e-6,
            )

            robot = ET.parse(paths.urdf).getroot()
            joint = robot.find("./joint[@name='joint_link_1']")
            link = robot.find("./link[@name='link_1']")
            self.assertEqual(joint.attrib["type"], "prismatic")
            self.assertEqual(joint.find("origin").attrib["xyz"], "0 0 0")
            self.assertEqual(link.find("visual/origin").attrib["xyz"], "0 0 0")
            self.assertEqual(link.find("collision/origin").attrib["xyz"], "0 0 0")

    def test_asset_validator_rejects_shifted_standalone_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segmented, motion = self._write_input(root, motion_type="P")
            paths = export_urdf_asset(segmented, motion, root / "asset")
            link_path = paths.meshes / "link_1.glb"
            shifted = trimesh.load(link_path, force="scene")
            for geometry in shifted.geometry.values():
                geometry.apply_translation([0.1, 0.0, 0.0])
            shifted.export(link_path)

            with self.assertRaisesRegex(ValueError, "is offset from whole.glb"):
                validate_asset(paths.root)

    def test_missing_motion_entry_defaults_to_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segmented, motion_path = self._write_input(root)
            motion = json.loads(motion_path.read_text(encoding="utf-8"))
            del motion["group_info"]["1"]
            motion_path.write_text(json.dumps(motion), encoding="utf-8")

            paths = export_urdf_asset(segmented, motion_path, root / "asset")

            robot = ET.parse(paths.urdf).getroot()
            joint = robot.find("./joint[@name='joint_link_1']")
            self.assertIsNotNone(joint)
            self.assertEqual(joint.attrib["type"], "fixed")

    def test_hierarchical_parent_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segmented, motion = self._write_input(root, parent="link_0")
            with self.assertRaisesRegex(ValueError, "Hierarchical URDF export"):
                export_urdf_asset(segmented, motion, root / "asset")

    def test_pipeline_default_publishes_only_compact_asset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "input.png"
            checkpoint = root / "checkpoint.pt"
            image.write_bytes(b"input")
            checkpoint.write_bytes(b"checkpoint")
            output = root / "result"
            workspace = root / ".result.monoart-work"
            sample = "sample"

            segmented, motion = self._write_input(root)
            generation = workspace / "generation" / sample
            reasoner = workspace / "reasoner" / sample
            motion_stage = workspace / "motion" / sample
            postprocess = workspace / "postprocess" / sample
            mesh_stage = workspace / "mesh" / sample
            for path in (generation, reasoner, motion_stage, postprocess, mesh_stage):
                path.mkdir(parents=True)
            shutil.copy2(segmented, generation / "mesh_textured.glb")
            (generation / "sampled_glb_4.ply").write_bytes(b"point cloud")
            (generation / "features_glb_4.npz").write_bytes(b"features")
            (reasoner / "points_4_feat.npy").write_bytes(b"features")
            shutil.copy2(motion, motion_stage / "motion.json")
            (postprocess / "segmentation.ply").write_bytes(b"segmentation")
            shutil.copy2(segmented, mesh_stage / "segmented_mesh.glb")
            shutil.copy2(motion, mesh_stage / "motion.json")

            paths = run_pipeline(
                image,
                output,
                checkpoint,
                sample_id=sample,
                num_points=4,
                resume=True,
            )

            self.assertEqual(paths.root, output.resolve())
            self.assertFalse(workspace.exists())
            files = sorted(
                path.relative_to(output).as_posix()
                for path in output.rglob("*")
                if path.is_file()
            )
            self.assertEqual(
                files,
                ["meshes/link_0.glb", "meshes/link_1.glb", "model.urdf", "whole.glb"],
            )


if __name__ == "__main__":
    unittest.main()
