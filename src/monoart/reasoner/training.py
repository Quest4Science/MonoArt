"""Train the Part-Aware Semantic Reasoner with part-contrastive supervision."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as functional
import yaml
from plyfile import PlyData
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from monoart.checkpoints import load_torch

from .builder import (
    PartAwareSemanticReasoner,
    build_reasoner,
    load_component_state,
)


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    point_cloud: Path
    features: Path
    labels: Path | None


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _manifest_rows(path: Path) -> list[ManifestRow]:
    base = path.resolve().parent
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"point_cloud", "features"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = []
        for index, row in enumerate(reader):
            label_value = (row.get("labels") or "").strip()
            rows.append(
                ManifestRow(
                    sample_id=(row.get("sample_id") or str(index)).strip(),
                    point_cloud=_resolve(base, row["point_cloud"]),
                    features=_resolve(base, row["features"]),
                    labels=_resolve(base, label_value) if label_value else None,
                )
            )
    if not rows:
        raise ValueError(f"{path} contains no samples")
    return rows


def _balanced_indices(labels: np.ndarray, count: int) -> np.ndarray:
    valid = np.flatnonzero(labels >= 0)
    unique = np.unique(labels[valid])
    if len(unique) < 2:
        raise ValueError("A reasoner sample must contain at least two valid parts")

    quota = max(2, count // len(unique))
    selected = []
    for label in unique:
        candidates = np.flatnonzero(labels == label)
        take = min(len(candidates), quota)
        selected.extend(np.random.choice(candidates, take, replace=False).tolist())
    remaining = count - len(selected)
    if remaining > 0:
        selected.extend(np.random.choice(valid, remaining, replace=remaining > len(valid)).tolist())
    elif remaining < 0:
        selected = np.random.choice(selected, count, replace=False).tolist()
    np.random.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


class ReasonerDataset(Dataset[dict[str, torch.Tensor]]):
    """Load aligned XYZ, 8D TRELLIS features, and integer part labels."""

    def __init__(self, manifest: str | Path, points_per_object: int = 5_120) -> None:
        self.rows = _manifest_rows(Path(manifest))
        self.points_per_object = points_per_object

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        vertex = PlyData.read(row.point_cloud)["vertex"]
        points = np.column_stack(
            [np.asarray(vertex[axis], dtype=np.float32) for axis in ("x", "y", "z")]
        )
        with np.load(row.features, allow_pickle=False) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)

        if row.labels is not None:
            labels = np.asarray(np.load(row.labels, allow_pickle=False), dtype=np.int64)
        else:
            names = vertex.data.dtype.names or ()
            if "group_id" not in names:
                raise ValueError(f"{row.point_cloud} has no group_id property and no labels path")
            labels = np.asarray(vertex["group_id"], dtype=np.int64)

        count = len(points)
        if features.shape != (count, 8):
            raise ValueError(f"{row.features}: expected {(count, 8)}, got {features.shape}")
        if labels.shape != (count,):
            raise ValueError(f"{row.sample_id}: expected {count} labels, got {labels.shape}")

        indices = _balanced_indices(labels, self.points_per_object)
        return {
            "points": torch.from_numpy(points[indices]),
            "features": torch.from_numpy(features[indices]),
            "labels": torch.from_numpy(labels[indices]),
        }


def hard_part_infonce(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    anchors_per_object: int,
    negatives_per_anchor: int,
    max_candidates: int,
) -> torch.Tensor:
    """Contrast each anchor with one same-part point and hard other-part points."""
    normalized = functional.normalize(features.float(), dim=-1)
    losses = []
    for sample_features, sample_labels in zip(normalized, labels):
        valid = sample_labels >= 0
        sample_features = sample_features[valid]
        sample_labels = sample_labels[valid]
        if len(sample_labels) < 3:
            continue

        unique, counts = sample_labels.unique(return_counts=True)
        eligible_labels = unique[counts >= 2]
        if len(eligible_labels) == 0 or len(unique) < 2:
            continue
        eligible = torch.isin(sample_labels, eligible_labels).nonzero().flatten()
        anchor_count = min(anchors_per_object, len(eligible))
        anchors = eligible[torch.randperm(len(eligible), device=features.device)[:anchor_count]]

        if len(sample_labels) > max_candidates:
            candidates = torch.randperm(len(sample_labels), device=features.device)[:max_candidates]
            required = anchors[~torch.isin(anchors, candidates)]
            candidates = torch.cat([candidates, required]).unique()
        else:
            candidates = torch.arange(len(sample_labels), device=features.device)

        positive_indices = []
        for anchor in anchors.tolist():
            same = (sample_labels == sample_labels[anchor]).nonzero().flatten()
            same = same[same != anchor]
            positive_indices.append(same[torch.randint(len(same), (1,), device=features.device)])
        positives = torch.cat(positive_indices)

        anchor_features = sample_features[anchors]
        similarities = anchor_features @ sample_features[candidates].T
        different = sample_labels[anchors, None] != sample_labels[candidates][None, :]
        available = different.sum(dim=1)
        negative_count = min(negatives_per_anchor, int(available.min().item()))
        if negative_count < 1:
            continue
        similarities = similarities.masked_fill(~different, float("-inf"))
        hard_negatives = similarities.topk(negative_count, dim=1).values
        positive_similarity = (anchor_features * sample_features[positives]).sum(dim=1)
        logits = torch.cat([positive_similarity[:, None], hard_negatives], dim=1)
        targets = torch.zeros(len(anchors), dtype=torch.long, device=features.device)
        losses.append(functional.cross_entropy(logits / temperature, targets))

    if not losses:
        raise RuntimeError("No valid contrastive pairs were found in this batch")
    return torch.stack(losses).mean()


def _distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def _seed(seed: int, rank: int) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_initial_state(model: PartAwareSemanticReasoner, checkpoint: Path) -> None:
    payload = load_torch(checkpoint)
    if payload.get("monoart_format_version") == 1:
        state = payload["reasoner"]["state_dict"]
    else:
        state = payload.get("state_dict", payload)
    load_component_state(model.encoder, state, "encoder")
    load_component_state(model.triplane_transformer, state, "triplane_transformer")
    load_component_state(model.part_decoder, state, "part_decoder", required=False)


def _lr_factor(step: int, total: int, warmup: int, minimum_ratio: float) -> float:
    if warmup > 0 and step < warmup:
        return 0.01 + 0.99 * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    epoch: int,
    best_validation: float,
) -> dict[str, Any]:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    return {
        "epoch": epoch,
        "state_dict": {key: value.detach().cpu() for key, value in module.state_dict().items()},
        "optimizer_states": [optimizer.state_dict()],
        "lr_schedulers": [scheduler.state_dict()],
        "hyper_parameters": {"config": config},
        "best_validation": best_validation,
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_config: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: GradScaler,
    grad_clip: float,
    use_amp: bool,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = torch.zeros(2, device=device)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            points = batch["points"].to(device, non_blocking=True)
            trellis_features = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                output = model(points, trellis_features)
                loss = hard_part_infonce(
                    output,
                    labels,
                    temperature=float(loss_config.get("temperature", 0.07)),
                    anchors_per_object=int(loss_config.get("anchors_per_object", 512)),
                    negatives_per_anchor=int(loss_config.get("negatives_per_anchor", 256)),
                    max_candidates=int(loss_config.get("max_candidates", 5_120)),
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
            total += torch.tensor([loss.detach(), 1.0], device=device)
    if dist.is_initialized():
        dist.all_reduce(total)
    return float((total[0] / total[1].clamp_min(1)).item())


def train(config_path: str | Path, resume: str | Path | None = None) -> None:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rank, local_rank, world_size = _distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("Reasoner training requires CUDA")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    _seed(int(config.get("seed", 42)), rank)

    data_config = config["data"]
    train_dataset = ReasonerDataset(
        data_config["train_manifest"],
        int(data_config.get("points_per_object", 5_120)),
    )
    val_manifest = data_config.get("val_manifest")
    val_dataset = (
        ReasonerDataset(val_manifest, int(data_config.get("points_per_object", 5_120)))
        if val_manifest
        else None
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if world_size > 1 and val_dataset else None
    )
    loader_kwargs = {
        "batch_size": int(data_config.get("batch_size", 1)),
        "num_workers": int(data_config.get("num_workers", 4)),
        "pin_memory": True,
        "persistent_workers": int(data_config.get("num_workers", 4)) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = (
        DataLoader(val_dataset, shuffle=False, sampler=val_sampler, **loader_kwargs)
        if val_dataset
        else None
    )

    model = PartAwareSemanticReasoner(build_reasoner(config, device)).to(device)
    init_checkpoint = config.get("initial_checkpoint")
    if init_checkpoint and not resume:
        _load_initial_state(model, Path(init_checkpoint))
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    training = config["training"]
    learning_rate = float(training.get("learning_rate", 5e-5))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    epochs = int(training.get("max_epochs", 100))
    total_steps = epochs * len(train_loader)
    warmup_steps = int(training.get("warmup_epochs", 10)) * len(train_loader)
    minimum_ratio = float(training.get("minimum_learning_rate", 1e-6)) / learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_factor(step, total_steps, warmup_steps, minimum_ratio),
    )
    use_amp = bool(training.get("use_amp", True))
    scaler = GradScaler("cuda", enabled=use_amp)
    start_epoch = 0
    best_validation = float("inf")
    if resume:
        payload = load_torch(resume)
        module = model.module if isinstance(model, DistributedDataParallel) else model
        module.load_state_dict(payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_states"][0])
        scheduler.load_state_dict(payload["lr_schedulers"][0])
        start_epoch = int(payload["epoch"]) + 1
        best_validation = float(payload.get("best_validation", best_validation))

    output = Path(training.get("output_dir", "outputs/reasoner_training"))
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = _run_epoch(
            model,
            train_loader,
            device,
            config.get("loss", {}),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            grad_clip=float(training.get("grad_clip", 0.1)),
            use_amp=use_amp,
        )
        validation_loss = (
            _run_epoch(
                model,
                val_loader,
                device,
                config.get("loss", {}),
                optimizer=None,
                scheduler=None,
                scaler=scaler,
                grad_clip=0.0,
                use_amp=use_amp,
            )
            if val_loader is not None
            else train_loss
        )
        improved = validation_loss < best_validation
        best_validation = min(best_validation, validation_loss)
        if rank == 0:
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
            payload = _checkpoint_payload(
                model, optimizer, scheduler, config, epoch, best_validation
            )
            _atomic_save(payload, output / "last.ckpt")
            if improved:
                _atomic_save(payload, output / "best.ckpt")
            if (epoch + 1) % int(training.get("save_every", 10)) == 0:
                _atomic_save(payload, output / f"epoch_{epoch + 1:03d}.ckpt")
            print(json.dumps(metrics))

    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    train(args.config, args.resume)


if __name__ == "__main__":
    main()
