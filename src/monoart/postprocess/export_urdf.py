"""Export the compact, geometry-complete MonoArt asset bundle."""

from __future__ import annotations

import json
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


_LINK_PATTERN = re.compile(r"link_[0-9]+")
_MOTION_TYPES = {
    "F": "fixed",
    "P": "prismatic",
    "R": "revolute",
    "C": "continuous",
}


@dataclass(frozen=True)
class AssetPaths:
    """Paths in the default public inference result."""

    root: Path

    @property
    def whole(self) -> Path:
        return self.root / "whole.glb"

    @property
    def urdf(self) -> Path:
        return self.root / "model.urdf"

    @property
    def meshes(self) -> Path:
        return self.root / "meshes"


@dataclass(frozen=True)
class _Joint:
    link: str
    parent: str
    motion_type: str
    axis: np.ndarray
    origin: np.ndarray
    lower: float | None
    upper: float | None

    @property
    def mesh_origin(self) -> np.ndarray:
        """Offset a world-coordinate mesh into the URDF joint frame."""
        if self.motion_type in {"revolute", "continuous"}:
            return -self.origin
        return np.zeros(3)


def _as_scene(path: Path) -> trimesh.Scene:
    loaded = trimesh.load(path, force="scene")
    if not isinstance(loaded, trimesh.Scene):
        loaded = trimesh.Scene(loaded)
    return loaded


def _scene_face_count(scene: trimesh.Scene) -> int:
    return sum(
        len(geometry.faces)
        for geometry in scene.geometry.values()
        if isinstance(geometry, trimesh.Trimesh)
    )


def _world_link_meshes(scene: trimesh.Scene) -> dict[str, trimesh.Trimesh]:
    """Return each segmented link with its scene-graph transform applied."""
    links: dict[str, trimesh.Trimesh] = {}
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node_name)
        link_name = str(node_name)
        if not _LINK_PATTERN.fullmatch(link_name):
            link_name = str(geometry_name)
        if not _LINK_PATTERN.fullmatch(link_name):
            raise ValueError(
                f"Expected segmented geometry names like link_0, received {node_name!r}"
            )
        if link_name in links:
            raise ValueError(f"Segmented GLB contains duplicate node {link_name!r}")
        geometry = scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh):
            raise TypeError(f"Geometry {geometry_name!r} is not a triangle mesh")
        mesh = geometry.copy()
        mesh.apply_transform(transform)
        if len(mesh.faces) == 0:
            raise ValueError(f"Segmented link {link_name!r} contains no faces")
        links[link_name] = mesh
    if not links:
        raise ValueError("The segmented GLB contains no link meshes")
    return links


def _parse_joint(raw: object, link_name: str) -> _Joint:
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError(f"Invalid motion entry for {link_name!r}: {raw!r}")
    declared_link = str(raw[0])
    parent = str(raw[1])
    motion_code = str(raw[-1]).upper()
    if declared_link != link_name:
        raise ValueError(
            f"Motion entry declares {declared_link!r}, expected {link_name!r}"
        )
    if parent != "base":
        raise ValueError(
            "Hierarchical URDF export requires parent-relative joint coordinates; "
            f"{link_name!r} has parent {parent!r} instead of 'base'"
        )
    if motion_code not in _MOTION_TYPES:
        raise ValueError(f"Unsupported motion type {motion_code!r} for {link_name!r}")

    if motion_code == "F":
        return _Joint(
            link=link_name,
            parent=parent,
            motion_type="fixed",
            axis=np.asarray([1.0, 0.0, 0.0]),
            origin=np.zeros(3),
            lower=None,
            upper=None,
        )

    if len(raw) != 4 or not isinstance(raw[2], list) or len(raw[2]) < 8:
        raise ValueError(f"Missing axis, origin, or limits for moving link {link_name!r}")
    parameters = np.asarray(raw[2], dtype=np.float64)
    if not np.all(np.isfinite(parameters)):
        raise ValueError(f"Non-finite motion parameters for {link_name!r}")
    axis = parameters[:3]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8:
        raise ValueError(f"Near-zero joint axis for {link_name!r}")
    lower, upper = sorted((float(parameters[6]), float(parameters[7])))
    motion_type = _MOTION_TYPES[motion_code]
    # A prismatic axis is an infinite line: only its direction and displacement
    # matter. The decoder does not supervise its origin, so the URDF frame must
    # remain at the identity zero pose instead of using that arbitrary value.
    origin = (
        parameters[3:6]
        if motion_type in {"revolute", "continuous"}
        else np.zeros(3)
    )
    return _Joint(
        link=link_name,
        parent=parent,
        motion_type=motion_type,
        axis=axis / axis_norm,
        origin=origin,
        lower=lower,
        upper=upper,
    )


def _joint_for_link(link_name: str, group_info: dict[str, object]) -> _Joint:
    label = link_name.removeprefix("link_")
    raw = group_info.get(label)
    if raw is None:
        # Geometry without a confident motion prediction remains present and static.
        return _Joint(
            link=link_name,
            parent="base",
            motion_type="fixed",
            axis=np.asarray([1.0, 0.0, 0.0]),
            origin=np.zeros(3),
            lower=None,
            upper=None,
        )
    return _parse_joint(raw, link_name)


def _format_numbers(values: np.ndarray | list[float]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _robot_xml(robot_name: str, joints: list[_Joint]) -> bytes:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", robot_name).strip("._") or "monoart"
    robot = ET.Element("robot", {"name": safe_name})
    robot.append(
        ET.Comment(
            " Mesh coordinates and prismatic limits use MonoArt's normalized object units. "
        )
    )
    ET.SubElement(robot, "link", {"name": "base"})

    for joint in joints:
        link = ET.SubElement(robot, "link", {"name": joint.link})
        mesh_uri = f"meshes/{joint.link}.glb"
        for element_name in ("visual", "collision"):
            element = ET.SubElement(link, element_name)
            ET.SubElement(
                element,
                "origin",
                {"xyz": _format_numbers(joint.mesh_origin), "rpy": "0 0 0"},
            )
            geometry = ET.SubElement(element, "geometry")
            ET.SubElement(geometry, "mesh", {"filename": mesh_uri})

        joint_element = ET.SubElement(
            robot,
            "joint",
            {"name": f"joint_{joint.link}", "type": joint.motion_type},
        )
        ET.SubElement(joint_element, "parent", {"link": joint.parent})
        ET.SubElement(joint_element, "child", {"link": joint.link})
        ET.SubElement(
            joint_element,
            "origin",
            {"xyz": _format_numbers(joint.origin), "rpy": "0 0 0"},
        )
        if joint.motion_type != "fixed":
            ET.SubElement(joint_element, "axis", {"xyz": _format_numbers(joint.axis)})
            limit = {"effort": "1", "velocity": "1"}
            if joint.motion_type != "continuous":
                if joint.lower is None or joint.upper is None:
                    raise ValueError(f"Joint limits are missing for {joint.link!r}")
                limit.update(
                    {"lower": f"{joint.lower:.9g}", "upper": f"{joint.upper:.9g}"}
                )
            ET.SubElement(joint_element, "limit", limit)

    ET.indent(robot, space="  ")
    return ET.tostring(robot, encoding="utf-8", xml_declaration=True) + b"\n"


def _validate_bundle(
    paths: AssetPaths,
    expected_links: dict[str, trimesh.Trimesh],
    joints: list[_Joint],
) -> None:
    expected_faces = sum(len(mesh.faces) for mesh in expected_links.values())
    tree = ET.parse(paths.urdf)
    references = {element.attrib["filename"] for element in tree.findall(".//mesh")}
    expected_references = {f"meshes/{joint.link}.glb" for joint in joints}
    if references != expected_references:
        raise RuntimeError(
            f"URDF mesh references do not match exported links: {references} != "
            f"{expected_references}"
        )
    exported_faces = 0
    for reference in sorted(references):
        mesh_path = paths.root / reference
        if not mesh_path.is_file() or mesh_path.stat().st_size == 0:
            raise RuntimeError(f"URDF references a missing or empty mesh: {mesh_path}")
        exported = _world_link_meshes(_as_scene(mesh_path))
        link_name = Path(reference).stem
        if set(exported) != {link_name}:
            raise RuntimeError(f"Independent mesh has unexpected nodes: {mesh_path}")
        exported_mesh = exported[link_name]
        expected_mesh = expected_links[link_name]
        exported_faces += len(exported_mesh.faces)
        if len(exported_mesh.faces) != len(expected_mesh.faces) or not np.allclose(
            exported_mesh.bounds,
            expected_mesh.bounds,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise RuntimeError(
                f"Independent mesh {link_name!r} is not in the whole.glb coordinate frame"
            )
    if exported_faces != expected_faces:
        raise RuntimeError(
            f"Independent mesh face count changed: {exported_faces} != {expected_faces}"
        )
    if _scene_face_count(_as_scene(paths.whole)) != expected_faces:
        raise RuntimeError("whole.glb does not preserve every segmented mesh face")


def export_urdf_asset(
    segmented_glb: str | Path,
    motion_json: str | Path,
    output: str | Path,
) -> AssetPaths:
    """Write only ``whole.glb``, ``model.urdf``, and per-link GLBs.

    Motion predictions and link meshes use world coordinates. Independent GLBs
    preserve those coordinates, so importing them with identity transforms exactly
    reconstructs ``whole.glb``. URDF visual offsets express revolute joint frames
    without modifying mesh vertices. A link missing from the motion metadata is
    conservatively exported as fixed.
    """
    segmented_glb = Path(segmented_glb).expanduser().resolve()
    motion_json = Path(motion_json).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not segmented_glb.is_file():
        raise FileNotFoundError(segmented_glb)
    if not motion_json.is_file():
        raise FileNotFoundError(motion_json)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Asset output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = AssetPaths(output)
    paths.meshes.mkdir()

    scene = _as_scene(segmented_glb)
    links = _world_link_meshes(scene)
    expected_faces = sum(len(mesh.faces) for mesh in links.values())
    if expected_faces != _scene_face_count(scene):
        raise RuntimeError("The segmented scene graph contains unaccounted mesh geometry")

    motion = json.loads(motion_json.read_text(encoding="utf-8"))
    if motion.get("coordinate_frame", "world") != "world":
        raise ValueError("URDF export currently requires world-frame motion predictions")
    group_info = motion.get("group_info", {})
    if not isinstance(group_info, dict):
        raise ValueError("motion.json group_info must be an object")

    joints: list[_Joint] = []
    for link_name in sorted(links, key=lambda name: int(name.removeprefix("link_"))):
        joint = _joint_for_link(link_name, group_info)
        joints.append(joint)
        link_scene = trimesh.Scene()
        link_scene.add_geometry(
            links[link_name].copy(),
            node_name=link_name,
            geom_name=link_name,
        )
        link_scene.export(paths.meshes / f"{link_name}.glb")

    shutil.copy2(segmented_glb, paths.whole)
    paths.urdf.write_bytes(_robot_xml(str(motion.get("object_name", "monoart")), joints))
    _validate_bundle(paths, links, joints)
    return paths
