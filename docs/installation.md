# Installation

## Requirements

- Linux with an NVIDIA GPU and a working CUDA driver;
- Conda or Mamba;
- Git with submodule support; and
- enough disk space for TRELLIS and the 557 MiB MonoArt checkpoint.

The end-to-end path was validated on an NVIDIA A100 80 GB. Lower-memory GPUs may work, but TRELLIS memory use depends on the generated sparse structure and texture settings and is not asserted here.

## Unified environment

All MonoArt stages use one environment. The generator still runs as a child process to release its CUDA memory cleanly before later stages.

```bash
conda env create -f environment/monoart.yml
conda activate monoart
bash scripts/install_dependencies.sh
```

The bootstrap installs MonoArt, installs the pinned TRELLIS Python requirements, checks out the validated TRELLIS revision, installs all GPU extensions into the active `monoart` environment, and runs the combined environment check. Pass a target path if TRELLIS should live elsewhere: `bash scripts/install_dependencies.sh /path/to/TRELLIS`. To test another TRELLIS revision intentionally, set `MONOART_TRELLIS_REVISION` to that commit before running the script.

`torch-scatter` and several TRELLIS dependencies are compiled CUDA extensions. Keep PyTorch, the CUDA runtime, the CUDA toolkit, and every extension on the versions selected by this environment. An ABI mismatch commonly appears as an undefined C++ symbol while importing an extension. If installation fails, verify the compiler and CUDA toolkit against TRELLIS's current installation guide.

Run Python with user-site packages disabled so an older package under `~/.local` cannot shadow the Conda environment.

## Checkpoint

Place `monoart_stage1.pt` and `monoart_stage1.pt.json` under `checkpoints/`. Verify the file before loading it:

```bash
sha256sum checkpoints/monoart_stage1.pt
monoart verify-checkpoint checkpoints/monoart_stage1.pt
monoart inspect-checkpoint checkpoints/monoart_stage1.pt
```

Expected SHA-256 for the prepared artifact:

```text
ab523603fed5b03807219a08284cdea22c4e82969cef98a72f7f2b2ace068fbe
```

PyTorch checkpoint loading uses pickle. Do not load untrusted files.

## Troubleshooting

### `torch_scatter` reports an undefined symbol

Confirm which copy Python resolved:

```bash
PYTHONNOUSERSITE=1 python -c \
  "import torch, torch_scatter; print(torch.__version__); print(torch_scatter.__file__)"
```

Reinstall the wheel matching the active PyTorch build. Do not mix a user-site wheel with a Conda environment wheel.

### TRELLIS is not importable

Pass the clone explicitly with `--trellis-root`. `--trellis-python` is only needed for an advanced split-environment installation; the unified installation uses the active Python executable automatically.

### Texture baking says a tensor has no gradient

Use this release's `monoart.generation` implementation. TRELLIS texture baking performs optimization and must not be wrapped in a process-wide `torch.inference_mode()` context.

### Out of memory

Run TRELLIS and MonoArt in separate processes as the CLI does. Reduce `--texture-size` first. Reducing `--num-points` changes the motion model's expected input contract and is not recommended for the released checkpoint.
