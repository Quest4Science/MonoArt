# Open questions before publication

This file records decisions that cannot be inferred safely from the recovered code and checkpoints.

## 1. Kinematic Estimator weights

No searched checkpoint contains `parent_head_state_dict`. The available motion files (`epoch_30.pth`, `epoch_38.pth`, and `we_useepoch_16.pth`) are Stage-1 checkpoints. The release bundle is therefore named `monoart_stage1.pt`, and inference explicitly reports that every predicted link is attached to the base.

**Author action:** provide the final Stage-2 checkpoint or confirm that Stage-1-only weights should be the public release.

## 2. Checkpoint/configuration mismatch

The paper states 100 motion queries, while the recovered public-inference checkpoints require 200 queries and 46 object categories. The selected motion checkpoint is internally labeled epoch 15 despite its filename `we_useepoch_16.pth`; it is an ordinary checkpoint, not a weighted average. The reasoner checkpoint is epoch 122 and embeds a 500-epoch training configuration, while the paper describes a 100-epoch reasoner phase.

**Author action:** confirm whether the paper should be amended, a different final checkpoint should be released, or these implementation differences should remain documented.

## 3. Default motion checkpoint

The recovered README calls `we_useepoch_16.pth` recommended, but recorded validation losses are lower for `epoch_38.pth` (approximately 1.285 versus 1.321). No benchmark table tying these files to the paper was recovered.

**Author action:** choose the public default based on the final evaluation protocol. The prepared bundle follows the recovered README and uses `we_useepoch_16.pth`.

## 4. Dataset release paths and redistribution

Training metadata references local PartNet/PartNet-Mobility derivatives, CSV splits, motion JSON, and cached generator/reasoner features. Their redistribution permissions and the exact final split files were not present in a clean release form.

**Author action:** publish permitted split manifests and preprocessing instructions, or provide official download links and checksums. Do not redistribute restricted source assets.
