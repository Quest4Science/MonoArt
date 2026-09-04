# Reproducibility record

## Paper settings

The paper reports 100,000 surface points, 8D TRELLIS features, 448D reasoner features, 100 motion queries, six decoder layers, AdamW with learning rate `5e-5`, weight decay `0.01`, ten warm-up epochs, cosine decay to `1e-6`, mixed precision, gradient clipping at `0.1`, batch size one per GPU, and four A100 GPUs. Its progressive schedule is described as 100 reasoner epochs, 20 category-initialization epochs, 100 joint motion epochs with a 40-epoch motion ramp, and 30 Kinematic Estimator epochs.

## Recovered artifacts

The files available during release preparation do not match every paper-level number:

| Item | Paper | Recovered artifact |
| --- | --- | --- |
| motion queries | 100 | 200 required by all three 46-category motion checkpoints |
| reasoner phase | 100 epochs | chosen checkpoint labeled epoch 122; embedded config allows 500 |
| reasoner optimizer | AdamW, `5e-5`, weight decay `0.01`, clip `0.1` | Adam, `1e-3`, weight decay `1e-4`, clip `1.0` in the embedded training config |
| motion checkpoint | not filename-specific | `we_useepoch_16.pth` stores epoch index 15 |
| kinematic head | reported module | no `parent_head_state_dict` found |

The release does not silently coerce these differences. `configs/inference.yaml` matches the tensors strictly, while the paper settings remain recorded for new experiments.

## Validation performed on the prepared release

The following checks were run on a real 100,000-point KitchenPot sample and the recovered weights:

- strict loading of all 107 reasoner tensors;
- reasoner output shape `[100000, 448]`, finite `float32` values;
- numerical comparison with the legacy output: mean absolute difference `7.53e-8`, RMSE `7.83e-7`, maximum absolute difference `6.56e-5`;
- strict loading of all 259 motion tensors and successful 46-category/200-query inference;
- preservation of `face_id` through segmentation and post-processing;
- GLB face labeling and three distinct animation-frame exports;
- a full TRELLIS 25+25-step generation, 100-view rendering, and 2,500-step texture bake;
- output validation for a 100,000-point PLY, `[100000, 8]` finite SLAT features, and a textured GLB;
- successful downstream inference from a combined checkpoint bundle (583,908,148 bytes after removal of private training paths);
- clean reasoner and Kinematic Estimator training smoke tests, including optimizer/scheduler resume; and
- source, configuration, unit-test, and installable-wheel validation in a clean temporary environment.

The checked-in unified environment was recreated from scratch and validated with Python 3.10.18, PyTorch 2.4.0 with CUDA 12.1 at runtime, NumPy 1.26.4, xFormers 0.0.27.post2, spconv 2.3.6, Kaolin 0.18.0, and `torch-scatter` 2.1.2. The source extensions were compiled with CUDA Toolkit 12.4 on an A100 server with driver 570.158.01. The same Python executable completed all image-to-motion stages for the versioned 100028 example in roughly two minutes. Its compact asset is checked in under `examples/100028/expected` and passes `monoart validate-asset` with exact face conservation and zero alignment error. TRELLIS was fixed at commit `442aa1e1afb9014e80681d3bf604e8d728a86ee7`.

## What is not claimed

- No paper benchmark was rerun because the final licensed split manifests were not recovered in a publishable form.
- The Kinematic Estimator cannot be validated without its trained head.
- A fresh six-day training reproduction was not run during code preparation.
- Cross-hardware bitwise equality is not expected.

These limits are publication metadata, not hidden fallbacks. See [OPEN_QUESTIONS.md](../OPEN_QUESTIONS.md).
