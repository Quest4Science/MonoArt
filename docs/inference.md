# Inference

## End-to-end command

```bash
PYTHONNOUSERSITE=1 monoart run image.png \
  --checkpoint checkpoints/monoart_stage1.pt \
  --output outputs/example \
  --trellis-root third_party/TRELLIS \
  --seed 42
```

The TRELLIS stage runs in its own process while using the same environment. Releasing its CUDA allocations before the semantic reasoner starts makes the pipeline more robust. Advanced users can still select another executable with `--trellis-python`.

Inference stages run in a hidden sibling workspace. After successful validation, the compact asset directory is published and the workspace is removed. If a run is interrupted, use `--resume` to skip only stages whose required files already exist. The final output directory must be new or empty so an existing asset is never overwritten implicitly.

## Output layout

By default, the command writes only:

```text
outputs/example/
├── model.urdf
├── whole.glb
└── meshes/
    ├── link_0.glb
    ├── link_1.glb
    └── ...
```

`whole.glb` preserves the zero-pose object as one textured, multi-node scene. Every independent
GLB keeps the same object/world coordinates as its node in `whole.glb`, so importing all link GLBs
with identity transforms reconstructs the whole object. The exporter verifies both this spatial
alignment and exact face-count conservation.

For debugging, add `--keep-intermediates`. This retains generation, reasoner, motion, post-processing, mesh-mapping, and run-manifest files under `outputs/example/work/`. Debug animation frames can be requested with `--keep-intermediates --animation-frames 30`.

Validate any default asset bundle with `monoart validate-asset outputs/example`. This checks the
exact compact file layout, face conservation, standalone-link alignment, URDF mesh references, and
the reconstructed zero pose.

## Stage contracts

### TRELLIS generation

```bash
PYTHONNOUSERSITE=1 python \
  -m monoart.generation \
  --image image.png \
  --output outputs/generation/image \
  --trellis-root third_party/TRELLIS
```

The feature lookup applies the same row-vector rotation used during training: `(x, y, z) -> (x, -z, y)`. Sparse SLAT features are trilinearly interpolated and missing voxels contribute zero.

### Part-Aware Semantic Reasoner

```bash
PYTHONNOUSERSITE=1 python -m monoart.reasoner.inference \
  --point-cloud outputs/generation/image/sampled_glb_100000.ply \
  --trellis-features outputs/generation/image/features_glb_100000.npz \
  --checkpoint checkpoints/monoart_stage1.pt \
  --output outputs/reasoner/image/points_100000_feat.npy
```

The output is a finite `float32` array shaped `[100000, 448]`.

### Motion decoder

```bash
PYTHONNOUSERSITE=1 python -m monoart.motion.inference \
  --point-cloud outputs/generation/image/sampled_glb_100000.ply \
  --trellis-features outputs/generation/image/features_glb_100000.npz \
  --part-features outputs/reasoner/image/points_100000_feat.npy \
  --checkpoint checkpoints/monoart_stage1.pt \
  --output outputs/motion/image \
  --sample-id image
```

A MonoArt bundle embeds its motion architecture. A legacy `.pth` file requires `--config configs/inference.yaml`.

### Geometry-complete mesh mapping

Every successful mesh export contains exactly the same number of faces as the source textured
mesh. Faces without a valid point label are assigned to a synthetic `static_fallback` link whose
motion type is `F`; they are never omitted or borrowed by a nearby moving link. The mesh-stage
metadata records the source/exported face counts and fallback count internally. These diagnostic
files are available under `work/mesh/` only when `--keep-intermediates` is enabled.

## URDF coordinate convention

The released Stage-1 model predicts independent base-relative joints in the reconstructed object's
world frame. Independent mesh vertices are never shifted during export. Fixed and prismatic links
use identity joint and visual origins; prismatic motion depends only on its predicted direction and
limit, because an origin on an infinite translation axis is irrelevant and is not supervised during
training. Revolute and continuous joints place the joint frame at the predicted world-space pivot
and use the inverse pivot as the URDF visual/collision origin. This represents
`T(pivot) * R(q) * T(-pivot)` while keeping each standalone GLB world-aligned.

Mesh coordinates and prismatic limits remain in MonoArt's normalized object units, as stated in a
comment inside `model.urdf`; they are not calibrated metric measurements. Hierarchical parents are
rejected rather than silently converted with an invalid frame assumption. A future Stage-2 release
must define its parent-link coordinate conversion before hierarchical URDF export is enabled.

## Kinematic tree behavior

If a checkpoint contains `parent_head_state_dict`, MonoArt resolves cycles and emits a kinematic tree. The recovered Stage-1 artifact does not contain that state, so the default URDF contains independent base-relative joints rather than a predicted hierarchy. The debug run manifest records `contains_kinematic_estimator: false` when `--keep-intermediates` is enabled.

## Determinism

The generator seeds PyTorch, CUDA, NumPy mesh sampling, and the point sampler. Exact bitwise equality across GPU models, CUDA versions, or extension builds is not guaranteed. For reproducibility studies, use `--keep-intermediates` and record `work/generation/*/generation.json`, `work/run.json`, the checkpoint SHA-256, and software versions.
