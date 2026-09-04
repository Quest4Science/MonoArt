# Training

MonoArt trains its three owned phases separately. TRELLIS is used as a frozen external generator.

## 1. Part-Aware Semantic Reasoner

Prepare explicit train/validation manifests as described in [datasets.md](datasets.md), then run:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m monoart.reasoner.training --config configs/train_reasoner.yaml
```

The trainer rasterizes aligned XYZ and 8D TRELLIS features to three planes, refines them with the six-layer transformer, samples 448D features back at the input points, and applies hard-negative part InfoNCE. It supports DDP, mixed precision, warm-up plus cosine decay, gradient clipping, validation, atomic checkpoints, and exact resume state.

`best.ckpt` and `last.ckpt` use the same `encoder.*`, `triplane_transformer.*`, and optional `part_decoder.*` key prefixes consumed by inference and the bundle packer.

## 2. Dual-Query Motion Decoder

Prepare the four aligned data products and motion annotations, edit only the data paths if necessary, and run:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m monoart.motion.scripts.train --config configs/train_motion.yaml
```

The provided config follows the progressive schedule: 20 category warm-up epochs, 100 joint segmentation/motion epochs, and a 40-epoch motion-loss ramp. It uses AdamW at `5e-5`, weight decay `0.01`, ten warm-up epochs, AMP, batch size one per process, and gradient clipping at `0.1`.

The Stage-1 checkpoint uses 200 motion queries and 46 object categories.

## 3. Kinematic Estimator

Set `parent_prediction.stage1_checkpoint` in `configs/train_kinematic.yaml` to a completed Stage-1 motion checkpoint, then run:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m monoart.motion.scripts.train --config configs/train_kinematic.yaml
```

Stage 2 runs for 30 epochs, freezes the configured Stage-1 heads, fine-tunes the decoder at a reduced learning rate, and uses the same warm-up plus cosine schedule. It saves atomic checkpoints containing the model, parent head, optimizer, scheduler, scaler, and best-validation state, so `--resume` continues the exact training state. Completed checkpoints enable hierarchical inference and can be supplied to `monoart pack-checkpoint` as the motion input.

## Evaluation

```bash
python -m monoart.motion.scripts.evaluate \
  --config configs/train_motion.yaml \
  --checkpoint outputs/motion_checkpoints/best.pth \
  --split test \
  --output_dir outputs/evaluation
```

For comparable evaluations, keep the dataset split, preprocessing cache, checkpoint SHA-256, and environment fixed.

## Building a release bundle

```bash
monoart pack-checkpoint \
  --reasoner outputs/reasoner_training/best.ckpt \
  --motion outputs/motion_checkpoints/best.pth \
  --motion-config configs/inference.yaml \
  --output checkpoints/monoart_stage1.pt
```

For a Stage-2 motion checkpoint, name the resulting artifact to reflect that it contains the Kinematic Estimator and verify `contains_kinematic_estimator: true` with `monoart inspect-checkpoint`.
