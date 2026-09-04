#!/usr/bin/env python3
"""Run source, configuration, checkpoint, and test checks for a release tree."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from monoart.checkpoints import inspect_checkpoint, load_torch, sha256

TEXT_SUFFIXES = {".py", ".sh", ".md", ".yaml", ".yml", ".toml", ".cff", ".txt"}
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
PRIVATE_PATH = re.compile(r"/(?:home|mnt\d*|media/home)/[A-Za-z0-9_.-]+/")
LOCALIZED_DOCUMENTS = {Path("README_CN.md")}


def _source_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in TEXT_SUFFIXES
        and "third_party" not in path.parts
        and "__pycache__" not in path.parts
    )


def _embedded_strings(value: object, location: str = "checkpoint") -> list[tuple[str, str]]:
    """Collect metadata strings without traversing checkpoint tensors."""
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_embedded_strings(child, f"{location}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_embedded_strings(child, f"{location}[{index}]"))
    elif isinstance(value, str):
        found.append((location, value))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    failures: list[str] = []

    for path in _source_files(root):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(root)
        if HAN.search(text) and relative not in LOCALIZED_DOCUMENTS:
            failures.append(f"Non-English Han text: {relative}")
        if relative.parts[0] in {"src", "configs", "scripts"} and PRIVATE_PATH.search(text):
            failures.append(f"Private absolute path: {relative}")
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=str(relative))
            except SyntaxError as exc:
                failures.append(f"Syntax error in {relative}: {exc}")
        elif path.suffix in {".yaml", ".yml"}:
            try:
                yaml.safe_load(text)
            except yaml.YAMLError as exc:
                failures.append(f"Invalid YAML in {relative}: {exc}")

    if args.checkpoint:
        summary = inspect_checkpoint(args.checkpoint)
        payload = load_torch(args.checkpoint)
        for location, value in _embedded_strings(payload):
            if HAN.search(value):
                failures.append(f"Non-English Han text in {location}")
            if PRIVATE_PATH.search(value):
                failures.append(f"Private absolute path in {location}")
        manifest_path = args.checkpoint.with_suffix(args.checkpoint.suffix + ".json")
        if not manifest_path.is_file():
            failures.append(f"Missing checkpoint manifest: {manifest_path}")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            digest = sha256(args.checkpoint)
            if digest != manifest.get("sha256"):
                failures.append("Checkpoint SHA-256 does not match its manifest")
            if manifest.get("path") != args.checkpoint.name:
                failures.append("Checkpoint manifest path must be a portable filename")
        print(json.dumps(summary, indent=2))

    test_environment = os.environ.copy()
    python_path = [str(root / "src")]
    if existing := test_environment.get("PYTHONPATH"):
        python_path.append(existing)
    test_environment["PYTHONPATH"] = os.pathsep.join(python_path)
    test_environment["PYTHONNOUSERSITE"] = "1"
    test = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests"), "-v"],
        cwd=root,
        env=test_environment,
        check=False,
    )
    if test.returncode:
        failures.append("Unit tests failed")

    report = {"root": str(root), "files_checked": len(_source_files(root)), "failures": failures}
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
