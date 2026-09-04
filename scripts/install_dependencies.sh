#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
trellis_root="${1:-$project_root/third_party/TRELLIS}"
trellis_revision="${MONOART_TRELLIS_REVISION:-442aa1e1afb9014e80681d3bf604e8d728a86ee7}"
extension_root="${MONOART_EXTENSION_ROOT:-$project_root/third_party/.build}"
nvdiffrast_revision="253ac4fcea7de5f396371124af597e6cc957bfae"
mip_splatting_revision="dda02ab5ecf45d6edb8c540d9bb65c7e451345a9"

ensure_checkout() {
  local name="$1"
  local repository="$2"
  local revision="$3"
  local target="$extension_root/$name"

  if [[ ! -e "$target" ]]; then
    git clone --recursive "$repository" "$target"
    git -C "$target" checkout --detach "$revision"
    git -C "$target" submodule update --init --recursive
  elif [[ ! -d "$target/.git" ]]; then
    printf 'Dependency target exists but is not a Git clone: %s\n' "$target" >&2
    exit 1
  fi

  local current_revision
  current_revision="$(git -C "$target" rev-parse HEAD)"
  if [[ "$current_revision" != "$revision" ]]; then
    printf '%s revision is %s; expected %s. Use a fresh extension cache.\n' \
      "$name" "$current_revision" "$revision" >&2
    exit 1
  fi
  git -C "$target" submodule update --init --recursive
}

python -c 'import sys, torch; assert sys.version_info[:2] == (3, 10), sys.version; assert torch.__version__.startswith("2.4.0"), torch.__version__; assert torch.version.cuda == "12.1", torch.version.cuda'
python -m pip install --no-user -e "$project_root"
python -m pip install --no-user -r "$project_root/environment/trellis-requirements.txt"

if [[ ! -e "$trellis_root" ]]; then
  git clone --recursive https://github.com/microsoft/TRELLIS.git "$trellis_root"
  git -C "$trellis_root" checkout --detach "$trellis_revision"
  git -C "$trellis_root" submodule update --init --recursive
elif [[ ! -d "$trellis_root/trellis" ]]; then
  printf 'TRELLIS target exists but is not a valid clone: %s\n' "$trellis_root" >&2
  exit 1
fi

if [[ -d "$trellis_root/.git" ]]; then
  current_revision="$(git -C "$trellis_root" rev-parse HEAD)"
  if [[ "$current_revision" != "$trellis_revision" ]]; then
    printf 'TRELLIS revision is %s; expected %s.\n' \
      "$current_revision" "$trellis_revision" >&2
    printf 'Use a fresh target, or set MONOART_TRELLIS_REVISION=%s intentionally.\n' \
      "$current_revision" >&2
    exit 1
  fi
fi

python -m pip install --no-user xformers==0.0.27.post2 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-user kaolin==0.18.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
python -m pip install --no-user spconv-cu120==2.3.6

mkdir -p "$extension_root"
if ! python -c 'import nvdiffrast' >/dev/null 2>&1; then
  ensure_checkout nvdiffrast https://github.com/NVlabs/nvdiffrast.git \
    "$nvdiffrast_revision"
  python -m pip install --no-user --no-build-isolation "$extension_root/nvdiffrast"
fi
if ! python -c 'import diff_gaussian_rasterization' >/dev/null 2>&1; then
  ensure_checkout mip-splatting https://github.com/autonomousvision/mip-splatting.git \
    "$mip_splatting_revision"
  python -m pip install --no-user --no-build-isolation \
    "$extension_root/mip-splatting/submodules/diff-gaussian-rasterization"
fi

if ! python -m pip install --no-user --only-binary=:all: torch-scatter \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html; then
  printf 'PyG wheel download failed; building torch-scatter 2.1.2 from source.\n' >&2
  python -m pip install --no-user --no-build-isolation \
    'git+https://github.com/rusty1s/pytorch_scatter.git@2.1.2'
fi

PYTHONNOUSERSITE=1 python "$project_root/scripts/check_environment.py" \
  --profile all --device cuda --trellis-root "$trellis_root"
python -m pip check

printf 'MonoArt environment is ready. TRELLIS revision: '
git -C "$trellis_root" rev-parse HEAD
