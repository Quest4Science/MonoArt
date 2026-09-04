# Dataset contracts

No training data is redistributed in this preparation tree. Review the licenses of PartNet, PartNet-Mobility, and every derived asset before publishing splits or caches.

## Motion decoder

The CSV split must contain:

```csv
anno_id,model_cat,source
100028_config_0_3_10,KitchenPot,partnet_mobility
```

`source` is retained for provenance; the current motion loader uses `anno_id` and `model_cat`.

Given `anno_id = 100028_config_0_3_10`, the default loader resolves:

| Item | Path |
| --- | --- |
| motion annotation | `<json_dir>/100028/config_0_3_10.json` |
| point cloud | `<ply_dir>/<anno_id>/sample_100k.ply` |
| TRELLIS feature | `<vae_dir>/<anno_id>_100k_features.npz` |
| reasoner feature | `<reasoner_feature_dir>/<anno_id>/points_100000_feat.npy` |

The PLY must contain `x y z nx ny nz group_id`; `face_id` is optional for training and required for reliable GLB mapping. TRELLIS NPZ files contain a `features` array shaped `[N, 8]`. Reasoner NPY files are shaped `[N, 448]`. The released motion checkpoint expects `N = 100000`.

Alternative PLY/NPZ filenames can be set in the YAML data section. Some recovered configuration keys retain their historical names for strict checkpoint and loader compatibility.

The annotation JSON is parsed into per-link joint type, axis direction, axis position, limits, semantic part name, and parent relationship. Training uses world-coordinate axes after forward-kinematic transformation by default.

## Semantic reasoner

The clean reasoner trainer uses a manifest CSV with explicit paths, avoiding assumptions about a private directory hierarchy:

```csv
sample_id,point_cloud,features,labels
sample_000,objects/sample_000.ply,features/sample_000.npz,labels/sample_000.npy
```

Paths are resolved relative to the manifest. Point clouds contain XYZ; feature files contain `[N, 8]` under `features`; label files contain integer part IDs shaped `[N]`, with negative IDs ignored. Each trainable object needs at least two valid parts and at least two points per sampled part.

## Preprocessing invariants

- Point order must be identical across the PLY, 8D features, 448D features, and labels.
- Coordinates must use the normalized object frame expected by TRELLIS and MonoArt.
- Do not recompute or shuffle one array independently.
- Preserve `face_id` when sampling a GLB.
- Record split files and checksums; paths alone are not a reproducible dataset definition.
