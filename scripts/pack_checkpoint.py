#!/usr/bin/env python3
"""Combine MonoArt-owned inference weights and discard optimizer state."""

from __future__ import annotations

import argparse
from pathlib import Path

from monoart.checkpoints import pack_checkpoint, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasoner", required=True, type=Path)
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--motion-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    result = pack_checkpoint(
        args.reasoner,
        args.motion,
        args.motion_config,
        args.output,
    )
    manifest = args.manifest or result.with_suffix(result.suffix + ".json")
    write_manifest(result, manifest)
    print(f"Saved checkpoint bundle to {result}")
    print(f"Saved manifest to {manifest}")


if __name__ == "__main__":
    main()
