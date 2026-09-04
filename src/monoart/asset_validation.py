"""Validation for compact MonoArt GLB and URDF asset bundles."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _vector(value: str, field: str) -> np.ndarray:
    vector = np.fromstring(value, sep=" ", dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Invalid {field}: {value!r}")
    return vector


def _origin_matrix(element: ET.Element | None) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    if element is None:
        return matrix
    xyz = _vector(element.attrib.get("xyz", "0 0 0"), "origin xyz")
    rpy = _vector(element.attrib.get("rpy", "0 0 0"), "origin rpy")
    matrix = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
    matrix[:3, 3] = xyz
    return matrix


def _world_link_meshes(path: Path) -> dict[str, trimesh.Trimesh]:
    scene = trimesh.load(path, force="scene")
    links: dict[str, trimesh.Trimesh] = {}
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node_name)
        link_name = str(node_name)
        if not link_name.startswith("link_"):
            link_name = str(geometry_name)
        if not link_name.startswith("link_"):
            raise ValueError(f"Unexpected mesh node name in {path}: {node_name!r}")
        if link_name in links:
            raise ValueError(f"Duplicate mesh node {link_name!r} in {path}")
        geometry = scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh):
            raise ValueError(f"Node {link_name!r} in {path} is not a triangle mesh")
        mesh = geometry.copy()
        mesh.apply_transform(transform)
        if len(mesh.faces) == 0:
            raise ValueError(f"Node {link_name!r} in {path} has no faces")
        links[link_name] = mesh
    if not links:
        raise ValueError(f"No link meshes found in {path}")
    return links


def _point_set_error(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) == 0 or len(second) == 0:
        return float("inf")
    first_to_second = cKDTree(second).query(first, k=1)[0]
    second_to_first = cKDTree(first).query(second, k=1)[0]
    return float(max(first_to_second.max(), second_to_first.max()))


def validate_asset(output: str | Path, *, tolerance: float = 1e-5) -> dict[str, object]:
    """Validate file layout, geometry alignment, and the URDF zero pose."""
    root = Path(output).expanduser().resolve()
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if not root.is_dir():
        raise FileNotFoundError(root)

    urdf_path = root / "model.urdf"
    whole_path = root / "whole.glb"
    meshes_path = root / "meshes"
    for required in (urdf_path, whole_path, meshes_path):
        if not required.exists():
            raise FileNotFoundError(required)

    whole_links = _world_link_meshes(whole_path)
    expected_files = {"model.urdf", "whole.glb"}
    expected_files.update(f"meshes/{link_name}.glb" for link_name in whole_links)
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ValueError(f"Asset file mismatch; missing={missing}, unexpected={unexpected}")

    robot = ET.parse(urdf_path).getroot()
    if robot.tag != "robot":
        raise ValueError("model.urdf does not contain a robot root element")
    urdf_links = {element.attrib.get("name", ""): element for element in robot.findall("link")}
    joint_by_child: dict[str, ET.Element] = {}
    for joint in robot.findall("joint"):
        child = joint.find("child")
        if child is None or "link" not in child.attrib:
            raise ValueError("URDF joint is missing its child link")
        child_name = child.attrib["link"]
        if child_name in joint_by_child:
            raise ValueError(f"Multiple joints use child link {child_name!r}")
        joint_by_child[child_name] = joint
    if set(joint_by_child) != set(whole_links):
        raise ValueError("URDF joint children do not match whole.glb link nodes")

    total_whole_faces = 0
    total_independent_faces = 0
    max_raw_error = 0.0
    max_zero_pose_error = 0.0
    link_reports = []
    for link_name, whole_mesh in sorted(whole_links.items()):
        link_path = meshes_path / f"{link_name}.glb"
        independent_links = _world_link_meshes(link_path)
        if set(independent_links) != {link_name}:
            raise ValueError(f"{link_path} must contain only node {link_name!r}")
        independent_mesh = independent_links[link_name]
        if len(independent_mesh.faces) != len(whole_mesh.faces):
            raise ValueError(f"Face count mismatch for {link_name!r}")

        raw_error = _point_set_error(independent_mesh.vertices, whole_mesh.vertices)
        if raw_error > tolerance:
            raise ValueError(
                f"Standalone mesh {link_name!r} is offset from whole.glb "
                f"(error={raw_error:.6g})"
            )

        urdf_link = urdf_links.get(link_name)
        if urdf_link is None:
            raise ValueError(f"URDF link {link_name!r} is missing")
        visual = urdf_link.find("visual")
        collision = urdf_link.find("collision")
        if visual is None or collision is None:
            raise ValueError(f"URDF link {link_name!r} needs visual and collision elements")
        expected_reference = f"meshes/{link_name}.glb"
        for element_name, element in (("visual", visual), ("collision", collision)):
            mesh = element.find("geometry/mesh")
            if mesh is None or mesh.attrib.get("filename") != expected_reference:
                raise ValueError(f"Invalid {element_name} mesh reference for {link_name!r}")

        joint = joint_by_child[link_name]
        parent = joint.find("parent")
        if parent is None or parent.attrib.get("link") != "base":
            raise ValueError("The Stage-1 asset validator requires base-relative joints")
        joint_origin = _origin_matrix(joint.find("origin"))
        visual_origin = _origin_matrix(visual.find("origin"))
        collision_origin = _origin_matrix(collision.find("origin"))
        if not np.allclose(visual_origin, collision_origin, rtol=0.0, atol=tolerance):
            raise ValueError(f"Visual and collision origins differ for {link_name!r}")

        zero_pose_mesh = independent_mesh.copy()
        zero_pose_mesh.apply_transform(joint_origin @ visual_origin)
        zero_pose_error = _point_set_error(zero_pose_mesh.vertices, whole_mesh.vertices)
        if zero_pose_error > tolerance:
            raise ValueError(
                f"URDF zero pose does not match whole.glb for {link_name!r} "
                f"(error={zero_pose_error:.6g})"
            )

        total_whole_faces += len(whole_mesh.faces)
        total_independent_faces += len(independent_mesh.faces)
        max_raw_error = max(max_raw_error, raw_error)
        max_zero_pose_error = max(max_zero_pose_error, zero_pose_error)
        link_reports.append(
            {
                "link": link_name,
                "joint_type": joint.attrib.get("type"),
                "faces": len(independent_mesh.faces),
            }
        )

    if total_independent_faces != total_whole_faces:
        raise ValueError("Independent GLBs do not preserve the whole.glb face count")
    return {
        "ok": True,
        "root": str(root),
        "files": sorted(actual_files),
        "num_links": len(whole_links),
        "whole_faces": total_whole_faces,
        "independent_faces": total_independent_faces,
        "max_raw_alignment_error": max_raw_error,
        "max_urdf_zero_pose_error": max_zero_pose_error,
        "links": link_reports,
    }
