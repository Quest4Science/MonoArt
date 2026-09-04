<h1 align="center">MonoArt</h1>

<p align="center">
  <strong>English</strong> | <a href="README_CN.md">Chinese</a>
</p>

<h3 align="center">Progressive Structural Reasoning for Monocular Articulated 3D Reconstruction</h3>

<p align="center"><strong>🎉 Accepted to ECCV 2026</strong></p>

<p align="center">
  <a href="https://quest4science.github.io/">Haitian Li</a><sup>*</sup> ·
  <a href="https://haozhexie.com">Haozhe Xie</a><sup>*</sup> ·
  <a href="https://linkedin.com/in/junxiang-xu-324812328">Junxiang Xu</a> ·
  <a href="https://github.com/wenbc21">Beichen Wen</a> ·
  <a href="https://hongfz16.github.io/">Fangzhou Hong</a> ·
  <a href="https://liuziwei7.github.io/">Ziwei Liu</a><sup>†</sup>
</p>

<p align="center">
  S-Lab, Nanyang Technological University<br>
  <sup>*</sup> Equal contribution    <sup>†</sup> Corresponding author
</p>

<p align="center">
  <a href="https://eccv.ecva.net/Conferences/2026">
    <img src="https://img.shields.io/badge/ECCV-2026-6f42c1?style=flat-square" alt="ECCV 2026">
  </a>
  <a href="https://arxiv.org/abs/2603.19231">
    <img src="https://img.shields.io/badge/arXiv-2603.19231-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv paper">
  </a>
  <a href="https://lihaitian.com/MonoArt">
    <img src="https://img.shields.io/badge/Project-Page-2563eb?style=flat-square&logo=googlechrome&logoColor=white" alt="Project page">
  </a>
  <a href="https://github.com/Quest4Science/MonoArt/releases">
    <img src="https://img.shields.io/badge/Model-Checkpoints-f59e0b?style=flat-square&logo=github&logoColor=white" alt="Model checkpoints">
  </a>
</p>

<p align="center">
  <img src="assets/teaser0301.jpg" width="100%" alt="MonoArt overview">
</p>

<p align="center"><em>From a single image to a textured, segmented, and articulated 3D asset.</em></p>

## 🔥 News

- **ECCV 2026:** MonoArt has been accepted to the European Conference on Computer Vision! 🎉
- **Newer work:** [PhysX-Omni](https://arxiv.org/abs/2605.21572) delivers stronger results and
  extends simulation-ready physical 3D generation to a unified setting covering rigid, deformable,
  and articulated objects. [[Project page](https://physx-omni.github.io/)]

## ✨ Overview

MonoArt is the official implementation of **MonoArt: Progressive Structural Reasoning for
Monocular Articulated 3D Reconstruction**. It turns a single object image into a textured,
segmented, and articulated 3D asset.

## 🗓️ TODO

- [x] ~~Publish the initial code and checkpoint release (organized and functional, but not yet fully polished).~~
- [ ] **October 2026:** Fix the remaining minor bugs and publish the final release.

## 🛠️ Installation

MonoArt uses one Conda environment for all stages. TRELLIS still runs in a child process so its GPU allocations are released before the reasoning and motion stages begin.

```bash
conda env create -f environment/monoart.yml
conda activate monoart
bash scripts/install_dependencies.sh
```

The examples below use `python -m monoart` so that the active Python interpreter is explicit.
After installation, the shorter `monoart ...` form is exactly equivalent: it is a Python console
entry point installed into the active Conda environment, not a shell alias. Installation and use do
not require `sudo` or change repository file permissions. `PYTHONNOUSERSITE=1` only prevents
packages from the user's global Python site directory from leaking into this environment.

More details and troubleshooting are in [docs/installation.md](docs/installation.md).

## 📦 Checkpoints

Download the release asset and its manifest into `checkpoints/`:

```bash
curl -L -o checkpoints/monoart_stage1.pt \
  https://github.com/Quest4Science/MonoArt/releases/latest/download/monoart_stage1.pt
curl -L -o checkpoints/monoart_stage1.pt.json \
  https://github.com/Quest4Science/MonoArt/releases/latest/download/monoart_stage1.pt.json
python -m monoart verify-checkpoint checkpoints/monoart_stage1.pt
python -m monoart inspect-checkpoint checkpoints/monoart_stage1.pt
```

### TRELLIS Weights

`third_party/TRELLIS/` is the TRELLIS source checkout created by the installation script, not a
model-weight directory. On the first inference run, MonoArt automatically downloads
[`JeffreyXiang/TRELLIS-image-large`](https://huggingface.co/JeffreyXiang/TRELLIS-image-large) from
Hugging Face and stores it in the standard cache (`~/.cache/huggingface/hub/` by default). Set
`HF_HOME=/path/to/huggingface-cache` before running MonoArt to use another cache location. The first
run requires network access unless this cache has already been populated.

## 🚀 Inference

Run every stage from the unified `monoart` environment:

```bash
PYTHONNOUSERSITE=1 python -m monoart run path/to/object.png \
  --checkpoint checkpoints/monoart_stage1.pt \
  --output outputs/demo \
  --trellis-root third_party/TRELLIS
```

The default result is a compact articulated asset: `model.urdf`, the complete `whole.glb`, and one URDF-ready `meshes/link_*.glb` per predicted link. No point clouds, feature arrays, JSON diagnostics, or animation frames are published by default. An interrupted run can continue from its hidden workspace with `--resume`; use `--keep-intermediates` only when stage-level debugging artifacts are needed.

### ✅ End-to-End Smoke Test

The repository includes the validated KitchenPot input and its compact reference asset. A quick
CPU-only check of the expected file layout and geometry is:

```bash
PYTHONNOUSERSITE=1 python -m monoart validate-asset examples/100028/expected
```

From the repository root, run the complete image-to-URDF pipeline with:

```bash
PYTHONNOUSERSITE=1 python -m monoart run examples/100028/input.png \
  --sample-id 100028 \
  --checkpoint checkpoints/monoart_stage1.pt \
  --output outputs/100028 \
  --trellis-root third_party/TRELLIS \
  --seed 42
```

Stage-specific commands, file contracts, and output layout are documented in [docs/inference.md](docs/inference.md).

Checkpoint publication and download commands are documented in [docs/releasing.md](docs/releasing.md).

## 🎓 Training and Evaluation

Portable configurations are provided for the released 200-query architecture:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m monoart.motion.scripts.train --config configs/train_motion.yaml

torchrun --standalone --nproc-per-node=4 \
  -m monoart.motion.scripts.train --config configs/train_kinematic.yaml
```

Dataset preparation and training details are recorded in [docs/training.md](docs/training.md) and [docs/datasets.md](docs/datasets.md).

## 🗂️ Repository Layout

```text
configs/             inference and training configurations
docs/                installation, data, inference, training, and release notes
environment/         reproducible Conda specifications
examples/            versioned end-to-end smoke-test inputs and reference metadata
scripts/             checkpoint and release utilities
src/monoart/         importable implementation
tests/               fast contract and numerical tests
third_party/         external dependencies created during installation
```

## 📝 Citation

```bibtex
@inproceedings{li2026monoart,
  title   = {MonoArt: Progressive Structural Reasoning for Monocular Articulated 3D Reconstruction},
  author  = {Li, Haitian and Xie, Haozhe and Xu, Junxiang and Wen, Beichen and Hong, Fangzhou and Liu, Ziwei},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year    = {2026}
}
```

## ⚖️ Licensing

Distributed under the S-Lab License. See `LICENSE` for more information.
