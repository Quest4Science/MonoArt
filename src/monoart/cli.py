"""Command-line interface for the MonoArt release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .checkpoints import inspect_checkpoint, pack_checkpoint, verify_checkpoint, write_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="monoart", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the complete single-image pipeline")
    run.add_argument("image", type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--checkpoint", required=True, type=Path)
    run.add_argument("--motion-config", type=Path)
    run.add_argument("--trellis-python", type=Path, default=Path(sys.executable))
    run.add_argument("--trellis-root", type=Path)
    run.add_argument("--sample-id")
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--num-points", type=int, default=100_000)
    run.add_argument("--sparse-steps", type=int, default=25)
    run.add_argument("--slat-steps", type=int, default=25)
    run.add_argument("--texture-size", type=int, default=1024)
    run.add_argument(
        "--animation-frames",
        type=int,
        default=0,
        help="Generate debug animation frames (requires --keep-intermediates)",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="Resume the hidden workspace left by an interrupted run",
    )
    run.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Retain stage outputs under OUTPUT/work (off by default)",
    )

    validate_asset = subparsers.add_parser(
        "validate-asset",
        help="Validate a compact GLB and URDF inference result",
    )
    validate_asset.add_argument("output", type=Path)
    validate_asset.add_argument("--tolerance", type=float, default=1e-5)

    pack = subparsers.add_parser(
        "pack-checkpoint", help="Combine and strip MonoArt-owned checkpoints"
    )
    pack.add_argument("--reasoner", required=True, type=Path)
    pack.add_argument("--motion", required=True, type=Path)
    pack.add_argument("--motion-config", required=True, type=Path)
    pack.add_argument("--output", required=True, type=Path)

    inspect = subparsers.add_parser("inspect-checkpoint", help="Print checkpoint metadata")
    inspect.add_argument("checkpoint", type=Path)

    verify = subparsers.add_parser(
        "verify-checkpoint", help="Verify a checkpoint against its SHA-256 manifest"
    )
    verify.add_argument("checkpoint", type=Path)
    verify.add_argument("--manifest", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "run":
        from .pipeline import run_pipeline

        paths = run_pipeline(
            args.image,
            args.output,
            args.checkpoint,
            motion_config=args.motion_config,
            trellis_python=args.trellis_python,
            trellis_root=args.trellis_root,
            sample_id=args.sample_id,
            device=args.device,
            seed=args.seed,
            num_points=args.num_points,
            sparse_steps=args.sparse_steps,
            slat_steps=args.slat_steps,
            texture_size=args.texture_size,
            animation_frames=args.animation_frames,
            resume=args.resume,
            keep_intermediates=args.keep_intermediates,
        )
        print(f"MonoArt completed. Results: {paths.root}")
    elif args.command == "validate-asset":
        from .asset_validation import validate_asset

        print(json.dumps(validate_asset(args.output, tolerance=args.tolerance), indent=2))
    elif args.command == "pack-checkpoint":
        output = pack_checkpoint(
            args.reasoner,
            args.motion,
            args.motion_config,
            args.output,
        )
        manifest = write_manifest(output, output.with_suffix(output.suffix + ".json"))
        print(f"Saved {output}")
        print(f"Saved {manifest}")
    elif args.command == "inspect-checkpoint":
        print(json.dumps(inspect_checkpoint(args.checkpoint), indent=2))
    elif args.command == "verify-checkpoint":
        print(json.dumps(verify_checkpoint(args.checkpoint, args.manifest), indent=2))


if __name__ == "__main__":
    main()
