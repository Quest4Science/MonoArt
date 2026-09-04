#!/usr/bin/env python3
"""
Main training script for Articulated Object Part Segmentation and Motion Prediction.

Usage:
    # Single GPU
    python -m monoart.motion.scripts.train --config configs/train_motion.yaml

    # Multi-GPU with torchrun
    torchrun --nproc-per-node=2 -m monoart.motion.scripts.train --config configs/train_motion.yaml

    # Resume training
    python -m monoart.motion.scripts.train --config configs/train_motion.yaml --resume checkpoints/epoch_10.pth
"""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from monoart.checkpoints import load_torch, motion_payload
from monoart.motion.datasets.articulated_dataset import ArticulatedDataset
from monoart.motion.datasets.collate_fn import articulated_collate_fn
from monoart.motion.losses.combined_loss import CombinedLoss
from monoart.motion.models.articulated_maft import ArticulatedMAFT

# Parent-child relation prediction (Stage 2)
from monoart.motion.relation import (
    ParentPredictionHead,
    ParentPredictionLoss,
)
from monoart.motion.trainers.base_trainer import BaseTrainer
from monoart.motion.utils.distributed import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict, distributed: bool = False) -> tuple:
    """Create training and validation dataloaders."""
    data_config = config["data"]
    model_config = config.get("model", {})

    if is_main_process():
        print(f"Batch size per process: {config['training']['batch_size']}")
        print(f"Training manifest: {data_config['train_csv']}")

    # Get num_categories from config (7 for singapo, 46 for full)
    num_categories = model_config.get("num_categories", 46)

    # Training dataset
    train_dataset = ArticulatedDataset(
        csv_path=data_config["train_csv"],
        json_dir=data_config["json_dir"],
        ply_dir=data_config["ply_dir"],
        vae_dir=data_config["vae_dir"],
        partfield_dir=data_config["partfield_dir"],
        n_points=data_config.get("n_points", 100000),
        augment=data_config.get("augment_train", data_config.get("augment", False)),
        include_fixed_motion=data_config.get("include_fixed_motion", True),
        verbose=False,
        num_categories=num_categories,
        use_world_coordinates=data_config.get("use_world_coordinates", True),
        ply_filename=data_config.get("ply_filename", "sample_100k.ply"),
        vae_filename=data_config.get("vae_filename"),
        vae_in_anno_dir=data_config.get("vae_in_anno_dir", False),
        partfield_filename=data_config.get("partfield_filename", "points_100000_feat.npy"),
    )

    # Create sampler for distributed training
    train_sampler = None
    shuffle = True
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
        )
        shuffle = False  # Sampler handles shuffling

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=config["training"].get("num_workers", 4),
        collate_fn=articulated_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Validation dataset
    val_loader = None
    if "val_csv" in data_config and data_config["val_csv"]:
        val_dataset = ArticulatedDataset(
            csv_path=data_config["val_csv"],
            json_dir=data_config["json_dir"],
            ply_dir=data_config["ply_dir"],
            vae_dir=data_config["vae_dir"],
            partfield_dir=data_config["partfield_dir"],
            n_points=data_config.get("n_points", 100000),
            augment=data_config.get("augment_val", False),
            include_fixed_motion=data_config.get("include_fixed_motion", True),
            verbose=False,
            num_categories=num_categories,
            use_world_coordinates=data_config.get("use_world_coordinates", True),
            ply_filename=data_config.get("ply_filename", "sample_100k.ply"),
            vae_filename=data_config.get("vae_filename"),
            vae_in_anno_dir=data_config.get("vae_in_anno_dir", False),
            partfield_filename=data_config.get("partfield_filename", "points_100000_feat.npy"),
        )

        # Validation sampler for distributed training
        val_sampler = None
        if distributed:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=False,
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            sampler=val_sampler,
            num_workers=config["training"].get("num_workers", 4),
            collate_fn=articulated_collate_fn,
            pin_memory=True,
        )

    if is_main_process():
        print(f"Training samples: {len(train_dataset)}")
        if val_loader:
            print(f"Validation samples: {len(val_dataset)}")
        if distributed:
            print(f"Samples per GPU: {len(train_dataset) // get_world_size()}")

    return train_loader, val_loader


def create_model(config: dict) -> torch.nn.Module:
    """Create ArticulatedMAFT model."""
    model_config = config["model"]

    # Build semantic_fusion_config from config if enabled
    semantic_fusion_config = None
    sf_config = config.get("semantic_fusion", {})
    if sf_config.get("enabled", False):
        semantic_fusion_config = {
            "enabled": True,  # Required for ArticulatedMAFT to initialize semantic fusion
            "d_model": model_config.get("d_model", 448),
            "enable_part_classification": sf_config.get("enable_part_classification", True),
            "enable_motion_prior": sf_config.get("enable_motion_prior", True),
            "enable_semantic_refiner": sf_config.get("enable_semantic_refiner", False),
            "part_class_use_mlp": sf_config.get("part_class_use_mlp", False),
            "motion_prior_adaptive_mix": sf_config.get("motion_prior_adaptive_mix", False),
            "clip_embedding_path": sf_config.get("clip_embedding_path", None),
            "dropout": model_config.get("dropout", 0.1),
            # use_motion_prior_head is read from this dict by ArticulatedMAFT
            "use_motion_prior_head": sf_config.get("use_motion_prior_head", False),
        }
        if is_main_process():
            print("Semantic Fusion enabled:")
            print(f"  - Part Classification: {sf_config.get('enable_part_classification', True)}")
            print(f"  - Motion Prior: {sf_config.get('enable_motion_prior', True)}")
            print(
                f"  - Motion Prior Head (Residual): {sf_config.get('use_motion_prior_head', False)}"
            )
            print(f"  - Semantic Refiner (CLIP): {sf_config.get('enable_semantic_refiner', False)}")

    _num_categories = model_config.get("num_categories", 46)
    if is_main_process():
        print(f"Object categories: {_num_categories}")

    # Part Geometric Feature config (for enhanced motion origin prediction)
    part_geometric_config = config.get("part_geometric", None)
    if part_geometric_config is not None and is_main_process():
        print("Part Geometric Feature config:")
        print(f"  - Enabled: {part_geometric_config.get('enabled', False)}")
        print(f"  - pos_encoding_dim: {part_geometric_config.get('pos_encoding_dim', 64)}")
        print(f"  - std_proj_dim: {part_geometric_config.get('std_proj_dim', 64)}")

    # Iterative Semantic Fusion config (EASE-style iterative fusion across decoder layers)
    iterative_fusion_config = config.get("iterative_semantic_fusion", None)
    if iterative_fusion_config is not None and is_main_process():
        print("Iterative Semantic Fusion config:")
        print(f"  - Enabled: {iterative_fusion_config.get('enabled', False)}")
        print(f"  - Mode: {iterative_fusion_config.get('mode', 'all_layers')}")
        print(f"  - Gate strategy: {iterative_fusion_config.get('gate_strategy', 'increasing')}")
        print(f"  - Deep supervision: {iterative_fusion_config.get('deep_supervision', False)}")

    # Relative Position Encoding config (RPE in cross-attention)
    rpe_config = config.get("rpe", None)
    if rpe_config is not None and is_main_process():
        rpe_type = rpe_config.get("type", "table")
        print("Relative Position Encoding (RPE) config:")
        print(f"  - Enabled: {rpe_config.get('enabled', False)}")
        print(f"  - Type: {rpe_type}")
        if rpe_type == "table":
            print("  [Table-based RPE]")
            print(f"    - grid_size: {rpe_config.get('grid_size', 0.05)}")
            print(f"    - num_buckets: {rpe_config.get('num_buckets', 24)}")
            print(f"    - use_query_bias: {rpe_config.get('use_query_bias', True)}")
            print(f"    - use_key_bias: {rpe_config.get('use_key_bias', True)}")
            print(f"    - lr_scale: {rpe_config.get('lr_scale', 1.0)}x")
        elif rpe_type == "mlp":
            print("  [MLP-based RPE]")
            print(f"    - hidden_dim: {rpe_config.get('mlp_hidden_dim', 64)}")
            print(f"    - num_layers: {rpe_config.get('mlp_num_layers', 2)}")
            print(f"    - use_fourier: {rpe_config.get('mlp_use_fourier', True)}")
            print(f"    - num_frequencies: {rpe_config.get('mlp_num_frequencies', 8)}")

    model = ArticulatedMAFT(
        partfield_dim=model_config.get("partfield_dim", 448),
        vae_dim=model_config.get("vae_dim", 8),
        d_model=model_config.get("d_model", 448),
        d_global=model_config.get("d_global", 64),
        num_queries=model_config.get("num_queries", 100),
        num_decoder_layers=model_config.get("num_decoder_layers", 6),
        nhead=model_config.get("nhead", 8),
        dim_feedforward=model_config.get("dim_feedforward", 1792),
        dropout=model_config.get("dropout", 0.1),
        num_categories=_num_categories,  # Dynamic based on config
        num_motion_types=model_config.get("num_motion_types", 4),  # 4 classes: F, P, R, C
        use_learnable_fallback=model_config.get("use_learnable_fallback", True),
        use_position_refinement=model_config.get("use_position_refinement", True),
        # Category classification: VAE + reasoner-feature fusion
        use_partfield_for_category=model_config.get("use_partfield_for_category", True),
        partfield_proj_dim=model_config.get("partfield_proj_dim", 32),
        # Semantic Fusion (use_motion_prior_head is inside semantic_fusion_config)
        semantic_fusion_config=semantic_fusion_config,
        # Part Geometric Feature (for motion origin prediction)
        part_geometric_config=part_geometric_config,
        # Iterative Semantic Fusion (EASE-style)
        iterative_fusion_config=iterative_fusion_config,
        # Relative Position Encoding (RPE in cross-attention)
        rpe_config=rpe_config,
    )

    if is_main_process():
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def compute_category_class_counts(csv_path: str, num_categories: int = 46) -> list:
    """
    Compute class counts from training CSV for class balancing.

    Args:
        csv_path: Path to training CSV file
        num_categories: Number of categories (7 for singapo, 46 for full)

    Returns:
        List of counts per category index
    """
    import pandas as pd

    from monoart.motion.datasets.motion_parser import get_category_map

    df = pd.read_csv(csv_path)
    counts = [0] * num_categories

    # Get the appropriate category map based on num_categories
    category_map = get_category_map(num_categories)

    # Count samples per category
    for cat_name in df["model_cat"]:
        if cat_name in category_map:
            idx = category_map[cat_name]
            if idx < num_categories:
                counts[idx] += 1

    return counts


def compute_motion_type_class_counts(csv_path: str, json_dir: str) -> list:
    """
    Compute motion type class counts from training data for class balancing.

    Reads JSON files referenced in the CSV and counts parts by motion type.
    This helps balance the loss for minority classes (Prismatic, Continuous).

    Motion Types:
    - 0: Fixed (F) - no motion
    - 1: Prismatic (P) - translation
    - 2: Revolute (R) - rotation with limits
    - 3: Continuous (C) - unlimited rotation

    Args:
        csv_path: Path to training CSV file
        json_dir: Directory containing JSON annotation files

    Returns:
        List of counts [count_F, count_P, count_R, count_C]
    """
    import pandas as pd

    from monoart.motion.datasets.motion_parser import MotionParser

    counts = [0, 0, 0, 0]  # [F, P, R, C]
    parser = MotionParser(use_3class=True, verbose=False, use_world_coordinates=False)
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        anno_id = str(row["anno_id"])
        parts = anno_id.split("_config_", maxsplit=1)
        model_id = parts[0]
        config_name = f"config_{parts[1]}" if len(parts) == 2 else anno_id
        parsed = parser.parse(os.path.join(json_dir, model_id, f"{config_name}.json"))
        if parsed is None:
            continue
        for motion in parsed["motions"]:
            counts[int(motion["motion_type_idx"])] += 1

    return counts


def create_loss_fn(
    config: dict,
    category_class_counts: list = None,
    motion_type_class_counts: list = None,
) -> torch.nn.Module:
    """Create combined loss function."""
    loss_config = config.get("loss", {})
    sf_config = config.get("semantic_fusion", {})

    # Determine if part_class_loss should be enabled
    # Enable if: loss.use_part_class_loss=true OR semantic_fusion.enabled=true
    use_part_class_loss = loss_config.get("use_part_class_loss", False)
    if sf_config.get("enabled", False) and sf_config.get("enable_part_classification", True):
        use_part_class_loss = True

    if is_main_process() and use_part_class_loss:
        print("Part Classification Loss enabled:")
        print(
            f"  - Weight: {loss_config.get('part_class_weight', sf_config.get('part_class_loss_weight', 0.5))}"
        )
        print(
            f"  - Class Balance: {loss_config.get('part_class_balance', sf_config.get('use_part_class_balance', True))}"
        )

    loss_fn = CombinedLoss(
        # Mask loss weights
        mask_focal_weight=loss_config.get("mask_focal_weight", 0.5),
        mask_dice_weight=loss_config.get("mask_dice_weight", 2.0),
        score_weight=loss_config.get("score_weight", 2.0),  # Increased for QFL
        # Motion loss weights
        motion_type_weight=loss_config.get("motion_type_weight", 1.0),
        motion_direction_weight=loss_config.get("motion_direction_weight", 1.0),
        motion_origin_weight=loss_config.get("motion_origin_weight", 0.5),
        motion_anchor_weight=loss_config.get("motion_anchor_weight", 0.5),
        use_anchor_loss=loss_config.get("use_anchor_loss", True),
        # Category loss weight
        category_weight=loss_config.get("category_weight", 0.3),
        # Motion limit loss weight
        motion_limit_weight=loss_config.get("motion_limit_weight", 0.5),
        # Overall task weights
        seg_weight=loss_config.get("seg_weight", 1.0),
        motion_weight=loss_config.get("motion_weight", 1.0),
        motion_warmup_epochs=loss_config.get("motion_warmup_epochs", 5),
        num_categories=config["model"].get("num_categories", 46),  # 46 categories
        # Category class balancing
        category_class_counts=category_class_counts,
        # Motion type class balancing (for imbalanced F/P/R/C distribution)
        motion_type_class_counts=motion_type_class_counts,
        # Rank-DETR improvements
        use_high_order_matching=loss_config.get("use_high_order_matching", True),
        iou_power_alpha=loss_config.get("iou_power_alpha", 3.0),
        use_quality_focal_loss=loss_config.get("use_quality_focal_loss", True),
        quality_focal_beta=loss_config.get("quality_focal_beta", 2.0),
        use_giou_target=loss_config.get("use_giou_target", True),
        # Unmatched Query suppression (no background)
        use_unmatched_loss=loss_config.get("use_unmatched_loss", True),
        unmatched_weight=loss_config.get("unmatched_weight", 0.5),
        # Spatial Affinity Loss
        use_affinity_loss=loss_config.get("use_affinity_loss", True),
        affinity_k=loss_config.get("affinity_k", 16),
        affinity_sigma=loss_config.get("affinity_sigma", 0.1),
        affinity_weight=loss_config.get("affinity_weight", 0.5),
        # Motion Limit Loss
        use_limit_loss=loss_config.get("use_limit_loss", True),
        limit_loss_type=loss_config.get("limit_loss_type", "l1"),
        # Matcher chunked computation (OOM prevention)
        matcher_chunk_size=loss_config.get("matcher_chunk_size", 10),
        # Part Classification Loss (for semantic fusion)
        use_part_class_loss=use_part_class_loss,
        part_class_weight=loss_config.get(
            "part_class_weight", sf_config.get("part_class_loss_weight", 0.5)
        ),
        part_class_balance=loss_config.get(
            "part_class_balance", sf_config.get("use_part_class_balance", True)
        ),
        # Center Loss (for query position supervision)
        use_center_loss=loss_config.get("use_center_loss", False),
        center_weight=loss_config.get("center_weight", 0.5),
        center_loss_type=loss_config.get("center_loss_type", "l1"),
        # Iterative Deep Supervision (for iterative semantic fusion)
        use_iterative_deep_supervision=config.get("iterative_semantic_fusion", {}).get(
            "deep_supervision", False
        ),
        iterative_deep_supervision_weight=config.get("iterative_semantic_fusion", {}).get(
            "deep_supervision_weight", 0.1
        ),
    )

    return loss_fn


def load_pretrained_weights(
    model: torch.nn.Module, pretrained_path: str, device: torch.device
) -> dict:
    """
    Load pretrained weights with strict=False to allow partial loading.

    This enables:
    - Loading weights from a model without SemanticFusion into a model with SemanticFusion
    - New modules (SemanticFusionModule, MotionTypeHeadWithResidual) will be randomly initialized
    - Existing modules (Encoder, Decoder, Category Head, etc.) will load pretrained weights

    Args:
        model: The model to load weights into
        pretrained_path: Path to the pretrained checkpoint
        device: Device to load the checkpoint to

    Returns:
        Dictionary with loading statistics
    """
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

    print(f"Loading pretrained weights from: {pretrained_path}")
    checkpoint = motion_payload(load_torch(pretrained_path, map_location=device))

    # Get the state dict
    if "model_state_dict" in checkpoint:
        pretrained_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        pretrained_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        pretrained_dict = checkpoint["state_dict"]
    else:
        pretrained_dict = checkpoint

    # Get current model state dict
    model_dict = model.state_dict()

    # Find matching and missing keys
    matched_keys = []
    mismatched_keys = []
    missing_keys = []
    unexpected_keys = []

    for key in pretrained_dict.keys():
        if key in model_dict:
            if pretrained_dict[key].shape == model_dict[key].shape:
                matched_keys.append(key)
            else:
                mismatched_keys.append((key, pretrained_dict[key].shape, model_dict[key].shape))
        else:
            unexpected_keys.append(key)

    for key in model_dict.keys():
        if key not in pretrained_dict:
            missing_keys.append(key)

    # Filter pretrained dict to only include matching keys
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in matched_keys}

    # Load the filtered dict
    model.load_state_dict(filtered_dict, strict=False)

    # Print statistics
    print("\nPretrained weights loading statistics:")
    print(f"  - Matched and loaded: {len(matched_keys)} parameters")
    print(
        f"  - Missing in pretrained (will be randomly initialized): {len(missing_keys)} parameters"
    )
    print(f"  - Unexpected in pretrained (ignored): {len(unexpected_keys)} parameters")
    print(f"  - Shape mismatch (ignored): {len(mismatched_keys)} parameters")

    if missing_keys:
        # Group missing keys by module
        missing_modules = {}
        for key in missing_keys:
            module = key.split(".")[0]
            if module not in missing_modules:
                missing_modules[module] = []
            missing_modules[module].append(key)

        print("\n  Missing modules (randomly initialized):")
        for module, keys in missing_modules.items():
            print(f"    - {module}: {len(keys)} parameters")

    if mismatched_keys:
        print("\n  Shape mismatches (ignored):")
        for key, pre_shape, model_shape in mismatched_keys[:5]:  # Show first 5
            print(f"    - {key}: pretrained={list(pre_shape)} vs model={list(model_shape)}")
        if len(mismatched_keys) > 5:
            print(f"    ... and {len(mismatched_keys) - 5} more")

    return {
        "matched": len(matched_keys),
        "missing": len(missing_keys),
        "unexpected": len(unexpected_keys),
        "mismatched": len(mismatched_keys),
        "epoch": checkpoint.get("epoch", 0),
    }


def setup_stage2_training(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
) -> tuple:
    """
    Setup Stage 2 training for parent-child prediction.

    This function:
    1. Loads Stage 1 checkpoint
    2. Creates ParentPredictionHead
    3. Freezes specified modules
    4. Creates ParentPredictionLoss

    Args:
        model: The ArticulatedMAFT model
        config: Full config dict
        device: Target device

    Returns:
        (parent_head, parent_loss_fn, frozen_params, trainable_params)
    """
    pp_config = config.get("parent_prediction", {})

    # 1. Load Stage 1 checkpoint
    stage1_checkpoint = pp_config.get("stage1_checkpoint")
    if stage1_checkpoint is None:
        raise ValueError(
            "parent_prediction.stage1_checkpoint must be specified for Stage 2 training. "
            "Please set it to the path of your Stage 1 trained model."
        )

    if is_main_process():
        print(f"\n{'=' * 60}")
        print("Stage 2 Training: Parent-Child Relation Prediction")
        print(f"{'=' * 60}")
        print(f"Loading Stage 1 checkpoint: {stage1_checkpoint}")

    load_stats = load_pretrained_weights(model, stage1_checkpoint, device)
    if is_main_process():
        print(f"Stage 1 model loaded (epoch {load_stats['epoch']})")

    # 2. Create ParentPredictionHead
    model_config = config.get("model", {})
    head_config = pp_config.get("head", {})

    parent_head = ParentPredictionHead(
        d_model=model_config.get("d_model", 448),
        use_position=head_config.get("use_position", True),
        use_semantic=head_config.get("use_semantic", True),
        lambda_semantic_init=head_config.get("lambda_semantic_init", 0.3),
    ).to(device)

    if is_main_process():
        print("\nParentPredictionHead created:")
        print(f"  - Parameters: {sum(p.numel() for p in parent_head.parameters()):,}")
        print(f"  - use_position: {head_config.get('use_position', True)}")
        print(f"  - use_semantic: {head_config.get('use_semantic', True)}")

    # 3. Freeze specified modules
    freeze_config = pp_config.get("freeze", {})
    frozen_params = []
    trainable_params = []

    # Mapping from config key to model attribute patterns
    freeze_patterns = {
        "backbone": ["feature_proj", "global_module", "pos_embed", "pos_embed_proj"],
        "decoder": ["decoder_layers", "decoder_norm", "pos_refinement"],
        "seg_heads": ["mask_embed", "score_head"],
        "motion_heads": ["motion_head", "part_geometric_module", "motion_prior_head"],
        "semantic_fusion": ["semantic_fusion", "iterative_fusion"],
        "category_head": ["global_module.category_head"],
        "part_class_head": ["semantic_fusion.part_class_head"],
    }

    # Determine which params to freeze
    params_to_freeze = set()
    for freeze_key, should_freeze in freeze_config.items():
        if should_freeze and freeze_key in freeze_patterns:
            params_to_freeze.update(freeze_patterns[freeze_key])

    if is_main_process():
        print("\nFreeze strategy:")

    for name, param in model.named_parameters():
        should_freeze = any(
            name == pattern or name.startswith(f"{pattern}.") for pattern in params_to_freeze
        )
        if should_freeze:
            param.requires_grad = False
            frozen_params.append(name)
        else:
            trainable_params.append(name)

    if is_main_process():
        # Group frozen params by module
        frozen_modules = {}
        for name in frozen_params:
            module = name.split(".")[0]
            if module not in frozen_modules:
                frozen_modules[module] = 0
            frozen_modules[module] += 1

        trainable_modules = {}
        for name in trainable_params:
            module = name.split(".")[0]
            if module not in trainable_modules:
                trainable_modules[module] = 0
            trainable_modules[module] += 1

        print(f"  Frozen modules ({len(frozen_params)} params):")
        for module, count in frozen_modules.items():
            print(f"    - {module}: {count}")

        print(f"  Trainable modules ({len(trainable_params)} params):")
        for module, count in trainable_modules.items():
            print(f"    - {module}: {count}")

    # 4. Create ParentPredictionLoss
    loss_config = pp_config.get("loss", {})
    parent_loss_fn = ParentPredictionLoss(
        weight=loss_config.get("parent_weight", 1.0),
        label_smoothing=loss_config.get("label_smoothing", 0.0),
    )

    if is_main_process():
        print("\nParentPredictionLoss created:")
        print(f"  - weight: {loss_config.get('parent_weight', 1.0)}")
        print(f"  - label_smoothing: {loss_config.get('label_smoothing', 0.0)}")

    return parent_head, parent_loss_fn, frozen_params, trainable_params


class Stage2Trainer:
    """
    Trainer for Stage 2: Parent-Child Relation Prediction.

    This trainer:
    1. Runs the frozen model forward to get query features
    2. Runs ParentPredictionHead to get parent logits
    3. Computes parent loss and backprops only through trainable params
    """

    def __init__(
        self,
        model: torch.nn.Module,
        parent_head: torch.nn.Module,
        loss_fn: torch.nn.Module,
        parent_loss_fn: torch.nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        config: dict,
        device: torch.device,
        distributed: bool = False,
        resume_from: str | None = None,
    ):
        self.model = model
        self.parent_head = parent_head
        self.loss_fn = loss_fn
        self.parent_loss_fn = parent_loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.distributed = distributed

        # Training config
        training_config = config.get("training", {})
        self.max_epochs = training_config.get("max_epochs", 100)
        self.grad_clip = training_config.get("grad_clip", 0.1)
        self.use_amp = bool(training_config.get("use_amp", True) and device.type == "cuda")
        self.log_interval = training_config.get("log_interval", 10)
        self.save_interval = training_config.get("save_interval", 5)
        self.exp_name = training_config.get("exp_name", "articulated_maft_stage2")
        self.checkpoint_dir = training_config.get("checkpoint_dir", "checkpoints")
        self.start_epoch = 0
        self.best_validation_loss = float("inf")

        # Setup AMP
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Setup logging
        self.log_dir = training_config.get("log_dir", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Move to device
        self.model = self.model.to(device)
        self.parent_head = self.parent_head.to(device)

        # Wrap with DDP if distributed
        if distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            # Note: model is mostly frozen, but we still wrap for consistency
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], find_unused_parameters=True
            )
            self.parent_head = torch.nn.parallel.DistributedDataParallel(
                self.parent_head, device_ids=[local_rank]
            )

        warmup_epochs = min(int(training_config.get("warmup_epochs", 0)), self.max_epochs)
        minimum_lr = float(training_config.get("minimum_learning_rate", 1e-6))
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.max_epochs - warmup_epochs),
                eta_min=minimum_lr,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.max_epochs),
                eta_min=minimum_lr,
            )

        if resume_from:
            self.load_checkpoint(resume_from)

    def train(self):
        """Main training loop for Stage 2."""
        if is_main_process():
            print(f"\nStarting Stage 2 training for {self.max_epochs} epochs...")

        for epoch in range(self.start_epoch, self.max_epochs):
            train_metrics = self.train_one_epoch(epoch)

            # Validation (every epoch)
            validation_metrics = self.validate(epoch) if self.val_loader is not None else None
            self.scheduler.step()

            if is_main_process():
                improved = False
                if validation_metrics is not None:
                    validation_loss = validation_metrics["loss"]
                    improved = validation_loss < self.best_validation_loss
                    self.best_validation_loss = min(self.best_validation_loss, validation_loss)
                self.save_checkpoint("last.pth", epoch)
                if (epoch + 1) % self.save_interval == 0:
                    self.save_checkpoint(f"epoch_{epoch + 1:03d}.pth", epoch)
                if improved:
                    self.save_checkpoint("best.pth", epoch)
                print(
                    f"Epoch [{epoch + 1}/{self.max_epochs}] "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"train_acc={train_metrics['accuracy']:.4f}"
                )
            if self.distributed:
                torch.distributed.barrier()

        if is_main_process():
            print("\nStage 2 training complete!")

    def train_one_epoch(self, epoch: int):
        """Train for one epoch."""
        model_module = self.model.module if hasattr(self.model, "module") else self.model
        model_module.train()
        for module in model_module.modules():
            parameters = list(module.parameters())
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                module.eval()
        self.parent_head.train()

        # Set sampler epoch for distributed training
        if self.distributed and hasattr(self.train_loader, "sampler"):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        total_correct = 0.0
        total_valid = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._move_to_device(batch)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with AMP
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # 1. Run main model to get query features and matched indices
                outputs = self.model(
                    partfield_features=batch["partfield_features"],
                    vae_features=batch["vae_features"],
                    points=batch["points"],
                )

                # 2. Get query features
                q_content = outputs.get("query_features")  # [B, Q, D]
                query_positions_3d = outputs.get("query_positions")  # [B, Q, 3]
                class_probs = outputs.get("part_class_probs")  # From semantic fusion

                # 3. Compute position embedding from 3D coordinates
                # Access model's pos_embed modules (handle DDP wrapper)
                model_module = self.model.module if hasattr(self.model, "module") else self.model
                q_position = model_module.pos_embed_proj(
                    model_module.pos_embed(query_positions_3d)
                )  # [B, Q, D]

                # 4. Run parent head
                parent_logits = self.parent_head(q_content, q_position, class_probs)

                # 5. Get matched_indices from Hungarian matcher
                # Need to run matcher to get query-to-GT correspondences
                # Build gt_masks from group_ids (same as CombinedLoss does)
                gt_masks = self.loss_fn._build_gt_masks(batch["group_ids"])
                targets = {
                    "gt_masks": gt_masks,
                }
                matched_indices = self.loss_fn.matcher(outputs, targets)

                gt_parent_info = batch.get("gt_parent_info")
                gt_link_ids = batch.get("gt_link_ids")

                parent_result = self.parent_loss_fn(
                    parent_logits,
                    matched_indices,
                    gt_parent_info,
                    gt_link_ids,
                )

                loss = parent_result["parent_loss"]

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.parent_head.parameters())
                        + [p for p in self.model.parameters() if p.requires_grad],
                        self.grad_clip,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.parent_head.parameters())
                        + [p for p in self.model.parameters() if p.requires_grad],
                        self.grad_clip,
                    )
                self.optimizer.step()

            # Accumulate metrics
            valid = parent_result["parent_num_valid"].item()
            total_loss += loss.item() * valid
            total_correct += parent_result["parent_accuracy"].item() * valid
            total_valid += valid

            # Log
            if is_main_process() and (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / max(total_valid, 1.0)
                avg_acc = total_correct / max(total_valid, 1.0)
                print(
                    f"Epoch [{epoch + 1}/{self.max_epochs}] "
                    f"Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}"
                )

        total_loss, total_correct, total_valid = self._reduce_metrics(
            total_loss, total_correct, total_valid
        )
        if total_valid <= 0:
            raise RuntimeError(
                "No valid parent edges were found in the training epoch; "
                "check gt_parent_info, gt_link_ids, and matcher assignments."
            )
        return {
            "loss": total_loss / max(total_valid, 1.0),
            "accuracy": total_correct / max(total_valid, 1.0),
            "num_valid": total_valid,
        }

    @torch.no_grad()
    def validate(self, epoch: int):
        """Run validation."""
        self.model.eval()
        self.parent_head.eval()

        total_loss = 0.0
        total_correct = 0.0
        total_valid = 0.0

        for batch in self.val_loader:
            batch = self._move_to_device(batch)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(
                    partfield_features=batch["partfield_features"],
                    vae_features=batch["vae_features"],
                    points=batch["points"],
                )
                q_content = outputs.get("query_features")  # [B, Q, D]
                query_positions_3d = outputs.get("query_positions")  # [B, Q, 3]
                class_probs = outputs.get("part_class_probs")

                # Compute position embedding from 3D coordinates
                model_module = self.model.module if hasattr(self.model, "module") else self.model
                q_position = model_module.pos_embed_proj(
                    model_module.pos_embed(query_positions_3d)
                )  # [B, Q, D]

                parent_logits = self.parent_head(q_content, q_position, class_probs)

                # Get matched_indices from Hungarian matcher
                # Build gt_masks from group_ids (same as CombinedLoss does)
                gt_masks = self.loss_fn._build_gt_masks(batch["group_ids"])
                targets = {
                    "gt_masks": gt_masks,
                }
                matched_indices = self.loss_fn.matcher(outputs, targets)

                gt_parent_info = batch.get("gt_parent_info")
                gt_link_ids = batch.get("gt_link_ids")

                parent_result = self.parent_loss_fn(
                    parent_logits,
                    matched_indices,
                    gt_parent_info,
                    gt_link_ids,
                )

            valid = parent_result["parent_num_valid"].item()
            total_loss += parent_result["parent_loss"].item() * valid
            total_correct += parent_result["parent_accuracy"].item() * valid
            total_valid += valid

        total_loss, total_correct, total_valid = self._reduce_metrics(
            total_loss, total_correct, total_valid
        )
        if total_valid <= 0:
            raise RuntimeError(
                "No valid parent edges were found during validation; "
                "check gt_parent_info, gt_link_ids, and matcher assignments."
            )
        metrics = {
            "loss": total_loss / max(total_valid, 1.0),
            "accuracy": total_correct / max(total_valid, 1.0),
            "num_valid": total_valid,
        }
        if is_main_process():
            print(
                f"Validation - Epoch [{epoch + 1}] "
                f"Loss: {metrics['loss']:.4f} Acc: {metrics['accuracy']:.4f}"
            )
        return metrics

    def save_checkpoint(self, filename: str, epoch: int):
        """Save checkpoint."""
        # Get model state dict (handle DDP wrapper)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        parent_head_to_save = (
            self.parent_head.module if hasattr(self.parent_head, "module") else self.parent_head
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "parent_head_state_dict": parent_head_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_validation_loss": self.best_validation_loss,
            "config": self.config,
        }
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        temporary_path = f"{checkpoint_path}.tmp"
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, path: str) -> None:
        """Resume all Stage-2 model and optimization state."""
        checkpoint = load_torch(path, map_location=self.device)
        model = self.model.module if hasattr(self.model, "module") else self.model
        parent_head = (
            self.parent_head.module if hasattr(self.parent_head, "module") else self.parent_head
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        parent_head.load_state_dict(checkpoint["parent_head_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_validation_loss = float(
            checkpoint.get("best_validation_loss", self.best_validation_loss)
        )

    def _reduce_metrics(
        self, loss_sum: float, correct_sum: float, valid_count: float
    ) -> tuple[float, float, float]:
        values = torch.tensor(
            [loss_sum, correct_sum, valid_count], dtype=torch.float64, device=self.device
        )
        if self.distributed:
            torch.distributed.all_reduce(values)
        return tuple(float(value) for value in values.cpu().tolist())

    def _move_to_device(self, batch: dict) -> dict:
        """Move batch tensors to device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, list):
                # Handle list of tensors (e.g., gt_link_ids)
                if len(value) > 0 and isinstance(value[0], torch.Tensor):
                    result[key] = [v.to(self.device) for v in value]
                else:
                    result[key] = value
            else:
                result[key] = value
        return result


def create_stage2_optimizer(
    model: torch.nn.Module,
    parent_head: torch.nn.Module,
    config: dict,
) -> torch.optim.Optimizer:
    """
    Create optimizer for Stage 2 training with different learning rates.

    - ParentPredictionHead: full learning rate
    - Decoder (if not frozen): reduced learning rate

    Args:
        model: The ArticulatedMAFT model
        parent_head: The ParentPredictionHead
        config: Full config dict

    Returns:
        Optimizer
    """
    pp_config = config.get("parent_prediction", {})
    training_config = config.get("training", {})

    base_lr = training_config.get("learning_rate", 1e-4)
    weight_decay = training_config.get("weight_decay", 0.01)
    decoder_lr_scale = pp_config.get("decoder_lr_scale", 0.1)

    param_groups = []

    # Parent head: full learning rate
    parent_params = list(parent_head.parameters())
    if parent_params:
        param_groups.append(
            {
                "params": parent_params,
                "lr": base_lr,
                "name": "parent_head",
            }
        )

    # Model trainable params (e.g., decoder if not frozen)
    model_trainable = [p for p in model.parameters() if p.requires_grad]
    if model_trainable:
        param_groups.append(
            {
                "params": model_trainable,
                "lr": base_lr * decoder_lr_scale,
                "name": "model_finetune",
            }
        )

    if is_main_process():
        print("\nOptimizer param groups:")
        for pg in param_groups:
            print(f"  - {pg['name']}: {len(pg['params'])} params, lr={pg['lr']:.6f}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer


def main():
    parser = argparse.ArgumentParser(description="Train ArticulatedMAFT")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (loads optimizer/scheduler state)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model weights (strict=False, only loads matching params)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument(
        "--exp_name", type=str, default=None, help="Experiment name (overrides config)"
    )
    args = parser.parse_args()

    # Initialize distributed training if launched with torchrun
    distributed = setup_distributed()

    # Load config
    config = load_config(args.config)

    # Override with command line args
    if args.exp_name:
        config["training"]["exp_name"] = args.exp_name

    # Setup device
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        print(f"Using device: {device}")
        print(f"Config: {args.config}")
        if distributed:
            print(f"Distributed training: {get_world_size()} GPUs")

    # Create components
    if is_main_process():
        print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config, distributed=distributed)

    if is_main_process():
        print("\nCreating model...")
    model = create_model(config)

    # Check if Stage 2 training (parent-child prediction) is enabled
    pp_config = config.get("parent_prediction", {})
    is_stage2 = pp_config.get("enabled", False)

    if is_stage2:
        # ============================================================
        # Stage 2 Training: Parent-Child Relation Prediction
        # ============================================================
        if is_main_process():
            print("\n" + "=" * 60)
            print("STAGE 2 MODE: Parent-Child Relation Prediction")
            print("=" * 60)

        # Setup Stage 2 training (load checkpoint, create head, freeze modules)
        parent_head, parent_loss_fn, frozen_params, trainable_params = setup_stage2_training(
            model, config, device
        )

        # Create loss function (still needed for model forward to get matched_indices)
        if is_main_process():
            print("\nCreating loss function...")
        loss_fn = create_loss_fn(config)

        # Create Stage 2 optimizer
        optimizer = create_stage2_optimizer(model, parent_head, config)

        # Create Stage 2 trainer
        trainer = Stage2Trainer(
            model=model,
            parent_head=parent_head,
            loss_fn=loss_fn,
            parent_loss_fn=parent_loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            device=device,
            distributed=distributed,
            resume_from=args.resume,
        )

        # Start Stage 2 training
        if is_main_process():
            print("\nStarting Stage 2 training...")

    else:
        # ============================================================
        # Stage 1 Training: Segmentation + Motion Prediction
        # ============================================================
        if is_main_process():
            print("\n" + "=" * 60)
            print("STAGE 1 MODE: Segmentation + Motion Prediction")
            print("=" * 60)

        # Load pretrained weights if specified (strict=False for partial loading)
        skip_category_warmup = False
        if args.pretrained:
            if is_main_process():
                print("\nLoading pretrained weights...")
            load_stats = load_pretrained_weights(model, args.pretrained, device)
            if is_main_process():
                print(f"\nPretrained weights from epoch {load_stats['epoch']} loaded successfully.")
                print("New modules will be trained from scratch.")
            # Skip category warmup since category head is already pretrained
            skip_category_warmup = True
            if is_main_process():
                print("Category warmup will be SKIPPED (using pretrained category head).")

        # Compute category class counts for class balancing
        category_class_counts = None
        loss_config = config.get("loss", {})
        data_config = config["data"]
        if loss_config.get("use_category_class_balance", True):
            if is_main_process():
                print("\nComputing category class counts for class balancing...")
            category_class_counts = compute_category_class_counts(
                data_config["train_csv"],
                num_categories=config["model"].get("num_categories", 7),
            )
            if is_main_process():
                print(f"  Class counts: {category_class_counts}")

        # Compute motion type class counts for class balancing (F/P/R/C)
        motion_type_class_counts = None
        if loss_config.get("use_motion_type_balance", False):
            if is_main_process():
                print("\nComputing motion type class counts for class balancing...")
            motion_type_class_counts = compute_motion_type_class_counts(
                data_config["train_csv"],
                data_config["json_dir"],
            )
            if is_main_process():
                motion_names = ["Fixed", "Prismatic", "Revolute", "Continuous"]
                print("  Motion type counts:")
                for i, (name, cnt) in enumerate(zip(motion_names, motion_type_class_counts)):
                    print(f"    {name}({i}): {cnt}")

        if is_main_process():
            print("\nCreating loss function...")
        loss_fn = create_loss_fn(
            config,
            category_class_counts=category_class_counts,
            motion_type_class_counts=motion_type_class_counts,
        )

        # Create trainer
        training_config = config["training"]
        logging_config = config.get("logging", {})

        # Determine category warmup epochs
        category_warmup_epochs = training_config.get("category_warmup_epochs", 0)
        if skip_category_warmup:
            category_warmup_epochs = 0  # Skip warmup when using pretrained weights

        # Get RPE learning rate configuration
        # Note: Only table-based RPE needs higher learning rate due to sparse gradients
        # MLP-based RPE has dense gradients and works well with standard learning rate
        rpe_config = config.get("rpe", {})
        rpe_enabled = rpe_config.get("enabled", False)
        rpe_type = rpe_config.get("type", "table")

        # Only apply lr_scale for table-based RPE
        if rpe_enabled and rpe_type == "table":
            rpe_lr_scale = rpe_config.get("lr_scale", 1.0)
            rpe_lr_warmup_epochs = rpe_config.get("lr_warmup_epochs", 0)
        else:
            rpe_lr_scale = 1.0  # MLP-based RPE uses standard learning rate
            rpe_lr_warmup_epochs = 0

        if is_main_process() and rpe_lr_scale > 1.0:
            print("\nRPE Learning Rate Configuration (Table-based):")
            print(f"  - lr_scale: {rpe_lr_scale}x")
            print(f"  - lr_warmup_epochs: {rpe_lr_warmup_epochs}")

        trainer = BaseTrainer(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=training_config.get("learning_rate", 1e-4),
            weight_decay=training_config.get("weight_decay", 0.01),
            max_epochs=training_config.get("max_epochs", 100),
            warmup_epochs=training_config.get("warmup_epochs", 5),
            minimum_learning_rate=training_config.get("minimum_learning_rate", 1e-6),
            # Category warmup (skipped if using pretrained weights)
            category_warmup_epochs=category_warmup_epochs,
            category_warmup_lr=training_config.get("category_warmup_lr", 1e-3),
            # Training options
            grad_clip=training_config.get("grad_clip", 0.1),
            use_amp=training_config.get("use_amp", True),
            log_dir=training_config.get("log_dir", "logs"),
            exp_name=training_config.get("exp_name", "articulated_maft"),
            log_interval=training_config.get("log_interval", 10),
            checkpoint_dir=training_config.get("checkpoint_dir", "checkpoints"),
            save_interval=training_config.get("save_interval", 5),
            resume_from=args.resume,
            device=device,
            # WandB
            use_wandb=logging_config.get("wandb", False),
            wandb_project=logging_config.get("wandb_project", "articulated-maft"),
            wandb_config=config,  # Log full config to WandB
            # Distributed training
            distributed=distributed,
            # RPE learning rate (higher to compensate for sparse gradients)
            rpe_lr_scale=rpe_lr_scale,
            rpe_lr_warmup_epochs=rpe_lr_warmup_epochs,
        )

        # Start Stage 1 training
        if is_main_process():
            print("\nStarting Stage 1 training...")

    try:
        trainer.train()
    finally:
        # Clean up distributed training
        if distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()
