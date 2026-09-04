# Checkpoints

Large weights are distributed as GitHub Release assets and are ignored by Git.

`monoart_stage1.pt` contains:

- the Part-Aware Semantic Reasoner configuration and 107 tensors;
- the Dual-Query Motion Decoder configuration and 259 tensors;
- no optimizer or scheduler state;
- no TRELLIS weights.

Verify the downloaded bytes before deserializing the trusted artifact, then inspect it:

```bash
monoart verify-checkpoint checkpoints/monoart_stage1.pt
monoart inspect-checkpoint checkpoints/monoart_stage1.pt
```

To build the bundle from component checkpoints:

```bash
monoart pack-checkpoint \
  --reasoner path/to/reasoner.ckpt \
  --motion path/to/motion.pth \
  --motion-config configs/inference.yaml \
  --output checkpoints/monoart_stage1.pt
```

PyTorch checkpoints use pickle. Only load files obtained from a trusted release and verify the SHA-256 manifest first.
