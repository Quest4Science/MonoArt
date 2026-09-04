#!/usr/bin/env python3
"""Check the active environment without downloading models or modifying it."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import site
import sys
from pathlib import Path


def _import(name: str) -> tuple[object | None, str | None]:
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            module = importlib.import_module(name)
        if message := output.getvalue():
            print(message, end="", file=sys.stderr)
        return module, None
    except Exception as exc:  # Import failures can originate in compiled extensions.
        if message := output.getvalue():
            print(message, end="", file=sys.stderr)
        return None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("all", "main", "trellis"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--trellis-root", type=Path)
    args = parser.parse_args()

    if args.trellis_root:
        sys.path.insert(0, str(args.trellis_root.expanduser().resolve()))
    if args.profile in {"all", "trellis"}:
        os.environ.setdefault("ATTN_BACKEND", "xformers")
        os.environ.setdefault("SPCONV_ALGO", "native")

    profiles = {
        "main": ("torch", "numpy", "scipy", "sklearn", "trimesh", "plyfile", "yaml"),
        "trellis": (
            "torch",
            "numpy",
            "trimesh",
            "trellis",
            "spconv",
            "xformers",
            "kaolin",
            "nvdiffrast",
            "diff_gaussian_rasterization",
        ),
    }
    required = profiles["main"] if args.profile == "main" else profiles["trellis"]
    if args.profile == "all":
        required = tuple(dict.fromkeys(profiles["main"] + profiles["trellis"]))
    if args.profile in {"all", "main"}:
        required += ("torch_scatter", "monoart")

    report: dict[str, object] = {
        "profile": args.profile,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "user_site_enabled": bool(site.ENABLE_USER_SITE),
        "modules": {},
    }
    failures = []
    loaded = {}
    for name in required:
        module, error = _import(name)
        loaded[name] = module
        if error:
            report["modules"][name] = {"error": error}
            failures.append(name)
        else:
            report["modules"][name] = {
                "version": getattr(module, "__version__", "unknown"),
                "path": getattr(module, "__file__", "built-in"),
            }

    torch = loaded.get("torch")
    if torch is not None:
        cuda_available = bool(torch.cuda.is_available())
        report["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
        }
        if args.device == "cuda" and not cuda_available:
            failures.append("cuda")

    report["ok"] = not failures
    if failures:
        report["failures"] = failures
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
