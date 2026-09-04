"""
Base Trainer for Articulated Object Part Segmentation and Motion Prediction.

Provides common training infrastructure:
- Training loop with validation
- Checkpoint management
- Logging (TensorBoard/W&B)
- Learning rate scheduling
- Mixed precision training
- Distributed Data Parallel (DDP) multi-GPU training
"""

import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from monoart.checkpoints import load_torch
from monoart.motion.utils.distributed import (
    get_local_rank,
    get_rank,
    get_world_size,
    is_dist_available_and_initialized,
    is_main_process,
    reduce_dict,
)

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class BaseTrainer:
    """
    Base trainer class for ArticulatedMAFT.

    Supports:
    - End-to-end training
    - Checkpoint save/load
    - TensorBoard logging
    - Mixed precision training
    - Gradient clipping
    - Distributed Data Parallel (DDP) multi-GPU training
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        # Optimization
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_epochs: int = 100,
        warmup_epochs: int = 5,
        minimum_learning_rate: float = 1e-6,
        # Category warmup
        category_warmup_epochs: int = 0,
        category_warmup_lr: float = 1e-3,
        # Training options
        grad_clip: float = 0.1,
        use_amp: bool = True,
        # Logging
        log_dir: str = "logs",
        exp_name: str = "articulated_maft",
        log_interval: int = 10,
        # Checkpoint
        checkpoint_dir: str = "checkpoints",
        save_interval: int = 5,
        resume_from: Optional[str] = None,
        # Device
        device: Optional[torch.device] = None,
        # WandB
        use_wandb: bool = False,
        wandb_project: str = "articulated-maft",
        wandb_config: Optional[Dict] = None,
        # Distributed training
        distributed: bool = False,
        # RPE learning rate configuration
        rpe_lr_scale: float = 1.0,
        rpe_lr_warmup_epochs: int = 0,
    ):
        """
        Args:
            model: The model to train
            loss_fn: Combined loss function
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            learning_rate: Initial learning rate
            weight_decay: AdamW weight decay
            max_epochs: Maximum training epochs
            warmup_epochs: Number of warmup epochs for LR
            minimum_learning_rate: Final learning rate after cosine decay
            category_warmup_epochs: Number of epochs to warmup category head (0 to disable)
            category_warmup_lr: Learning rate for category warmup
            grad_clip: Gradient clipping value
            use_amp: Use automatic mixed precision
            log_dir: Directory for logs
            exp_name: Experiment name
            log_interval: Batches between log updates
            checkpoint_dir: Directory for checkpoints
            save_interval: Epochs between checkpoint saves
            resume_from: Checkpoint path to resume from
            device: Device to train on
            use_wandb: Use Weights & Biases for logging
            wandb_project: WandB project name
            wandb_config: Config dict to log to WandB
            distributed: Enable distributed training (DDP)
        """
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.minimum_learning_rate = minimum_learning_rate
        self.category_warmup_epochs = category_warmup_epochs
        self.category_warmup_lr = category_warmup_lr
        self.grad_clip = grad_clip
        self.use_amp = use_amp

        # RPE learning rate configuration
        self.rpe_lr_scale = rpe_lr_scale
        self.rpe_lr_warmup_epochs = rpe_lr_warmup_epochs

        self.log_dir = log_dir
        self.exp_name = exp_name
        self.log_interval = log_interval
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.use_wandb = use_wandb and HAS_WANDB
        self.wandb_project = wandb_project
        self.wandb_config = wandb_config

        # Distributed training setup
        self.distributed = distributed and is_dist_available_and_initialized()
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.is_main_process = is_main_process()

        # Device setup
        if self.distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
        elif device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.use_amp = bool(self.use_amp and self.device.type == "cuda")

        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)

        # Wrap model with DDP if distributed
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,  # In case some parameters are not used
            )
            if self.is_main_process:
                print(f"Model wrapped with DDP (world_size={self.world_size})")

        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()

        # Mixed precision scaler
        self.scaler = GradScaler("cuda") if self.use_amp else None

        # Setup logging (only on main process for distributed)
        self._setup_logging()

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.warmup_completed = False  # Track if category warmup is done

        # Resume if specified
        if resume_from is not None:
            self.load_checkpoint(resume_from)

        # Create directories (only on main process)
        if self.is_main_process:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _setup_optimizer(self):
        """
        Setup AdamW optimizer with optional parameter groups.

        If rpe_lr_scale > 1.0, RPE parameters get a higher learning rate
        to compensate for sparse gradients in the lookup table.
        """
        # Get the actual model (unwrap DDP if needed)
        model = self.model.module if self.distributed else self.model

        # Check if we need separate learning rate for RPE
        if self.rpe_lr_scale > 1.0:
            # Separate RPE parameters from other parameters
            rpe_params = []
            other_params = []
            rpe_param_names = []

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if ".rpe." in name or name.startswith("rpe."):
                    rpe_params.append(param)
                    rpe_param_names.append(name)
                else:
                    other_params.append(param)

            # Create parameter groups
            param_groups = []

            if other_params:
                param_groups.append(
                    {
                        "params": other_params,
                        "lr": self.learning_rate,
                        "name": "base",
                    }
                )

            if rpe_params:
                rpe_lr = self.learning_rate * self.rpe_lr_scale
                param_groups.append(
                    {
                        "params": rpe_params,
                        "lr": rpe_lr,
                        "name": "rpe",
                    }
                )

                if self.is_main_process:
                    print(f"\n{'=' * 60}")
                    print("Optimizer Parameter Groups (RPE with higher LR)")
                    print(f"{'=' * 60}")
                    print(f"  Base params: {len(other_params)} params, lr={self.learning_rate:.6f}")
                    print(
                        f"  RPE params:  {len(rpe_params)} params, lr={rpe_lr:.6f} ({self.rpe_lr_scale}x)"
                    )
                    print(f"  RPE param names: {rpe_param_names[:3]}...")
                    print(f"{'=' * 60}\n")

            self.optimizer = AdamW(
                param_groups,
                weight_decay=self.weight_decay,
            )
        else:
            # Standard optimizer with single learning rate
            self.optimizer = AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup."""
        # Linear warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )

        # Cosine annealing after warmup
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.max_epochs - self.warmup_epochs),
            eta_min=self.minimum_learning_rate,
        )

        # Combine schedulers
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs],
        )

    def _setup_logging(self):
        """Setup logging (console + TensorBoard + WandB + CSV)."""
        # Console logger - always setup but control verbosity based on rank
        self.logger = logging.getLogger(f"{self.exp_name}_rank{self.rank}")
        self.logger.setLevel(logging.INFO if self.is_main_process else logging.WARNING)

        # Create handlers if not already set
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO if self.is_main_process else logging.WARNING)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # TensorBoard - only on main process
        self.writer = None
        if self.is_main_process:
            if HAS_TENSORBOARD:
                log_path = os.path.join(self.log_dir, self.exp_name)
                Path(log_path).mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_path)
            else:
                self.logger.warning("TensorBoard not available")

        # CSV logging - only on main process
        self.csv_log_path = None
        if self.is_main_process:
            log_path = os.path.join(self.log_dir, self.exp_name)
            Path(log_path).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_log_path = os.path.join(log_path, f"training_log_{timestamp}.csv")
            self.csv_header_written = False
            self.logger.info(f"CSV log path: {self.csv_log_path}")

        # WandB - only on main process
        if self.is_main_process and self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                name=self.exp_name,
                config=self.wandb_config,
                resume="allow",
            )
            # Watch model for gradient logging (use underlying model for DDP)
            model_to_watch = self.model.module if self.distributed else self.model
            wandb.watch(model_to_watch, log="gradients", log_freq=100)
            self.logger.info(f"WandB initialized: {wandb.run.url}")
        elif self.is_main_process and HAS_WANDB:
            self.logger.info("WandB available but not enabled (set use_wandb=True)")
        elif self.is_main_process:
            self.logger.warning("WandB not installed")

    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training on {self.device}")
        if self.distributed:
            self.logger.info(f"Distributed training: rank {self.rank}/{self.world_size}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Category warmup phase (skip if already completed or resuming)
        if self.category_warmup_epochs > 0 and not self.warmup_completed:
            self._category_warmup()
            self.warmup_completed = True
            # Save checkpoint after warmup as epoch_0.pth (only main process)
            if self.is_main_process:
                self.save_checkpoint("epoch_0.pth")
                self.logger.info("Saved post-warmup checkpoint: epoch_0.pth")
            # Synchronize all processes
            if self.distributed:
                import torch.distributed as dist

                dist.barrier()
        elif self.warmup_completed:
            self.logger.info("Category warmup already completed, skipping...")

        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch

            # Set epoch for distributed sampler (ensures different shuffling each epoch)
            if self.distributed and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            # Train one epoch
            train_losses = self._train_epoch()

            # Validation
            val_losses = None
            if self.val_loader is not None:
                val_losses = self._validate()

            # Update learning rate
            self.scheduler.step()

            # Log epoch summary (only main process)
            if self.is_main_process:
                self._log_epoch(train_losses, val_losses)

            # Save checkpoint (only main process)
            if self.is_main_process and (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pth")

            # Save best model (only main process)
            if self.is_main_process and val_losses is not None:
                val_total = val_losses.get("total_loss", float("inf"))
                if val_total < self.best_val_loss:
                    self.best_val_loss = val_total
                    self.save_checkpoint("best.pth")
                    self.logger.info(f"New best model (val_loss: {val_total:.4f})")

            # Save last checkpoint every epoch (only main process)
            if self.is_main_process:
                self.save_checkpoint("last.pth")

            # Synchronize all processes at end of epoch
            if self.distributed:
                import torch.distributed as dist

                dist.barrier()

        # Save final checkpoint (only main process)
        if self.is_main_process:
            self.save_checkpoint("final.pth")
        self.logger.info("Training completed!")

        if self.writer:
            self.writer.close()

        if self.is_main_process and self.use_wandb:
            wandb.finish()

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        loss_meters = {}
        data_time = AverageMeter()
        batch_time = AverageMeter()

        # Category accuracy tracking
        cat_correct = 0
        cat_total = 0

        end = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            # Move batch to device
            batch = self._to_device(batch)

            # Forward pass
            with autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self.model(
                    partfield_features=batch["partfield_features"],
                    vae_features=batch["vae_features"],
                    points=batch["points"],
                )

                # Prepare targets
                targets = self._prepare_targets(batch)

                # Compute loss
                losses = self.loss_fn(outputs, targets, epoch=self.epoch)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(losses["total_loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            # Update loss meters
            for key, val in losses.items():
                if key not in loss_meters:
                    loss_meters[key] = AverageMeter()
                if isinstance(val, torch.Tensor):
                    loss_meters[key].update(val.item())

            # Update category accuracy
            with torch.no_grad():
                cat_preds = outputs["category_logits"].argmax(dim=-1)
                cat_targets = targets["categories"]
                if cat_targets is not None:
                    cat_correct += (cat_preds == cat_targets).sum().item()
                    cat_total += cat_targets.size(0)

            batch_time.update(time.time() - end)
            end = time.time()

            # Log batch (only main process)
            if self.is_main_process and batch_idx % self.log_interval == 0:
                cat_acc = cat_correct / cat_total if cat_total > 0 else 0
                self._log_batch(
                    batch_idx, len(self.train_loader), loss_meters, batch_time, data_time, cat_acc
                )

            self.global_step += 1

        # Return average losses + category accuracy
        result = {key: meter.avg for key, meter in loss_meters.items()}
        result["category_accuracy"] = cat_correct / cat_total if cat_total > 0 else 0

        # Reduce losses across all processes for accurate logging
        if self.distributed:
            result = reduce_dict(result, average=True)
        return result

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        loss_meters = {}

        # Category accuracy tracking
        cat_correct = 0
        cat_total = 0

        for batch in self.val_loader:
            batch = self._to_device(batch)

            outputs = self.model(
                partfield_features=batch["partfield_features"],
                vae_features=batch["vae_features"],
                points=batch["points"],
            )

            targets = self._prepare_targets(batch)
            losses = self.loss_fn(outputs, targets, epoch=self.epoch)

            for key, val in losses.items():
                if key not in loss_meters:
                    loss_meters[key] = AverageMeter()
                if isinstance(val, torch.Tensor):
                    loss_meters[key].update(val.item())

            # Update category accuracy
            cat_preds = outputs["category_logits"].argmax(dim=-1)
            cat_targets = targets["categories"]
            if cat_targets is not None:
                cat_correct += (cat_preds == cat_targets).sum().item()
                cat_total += cat_targets.size(0)

        # Return average losses + category accuracy
        result = {key: meter.avg for key, meter in loss_meters.items()}
        result["category_accuracy"] = cat_correct / cat_total if cat_total > 0 else 0

        # Reduce losses across all processes for accurate logging
        if self.distributed:
            result = reduce_dict(result, average=True)
        return result

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        result = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                result[key] = val.to(self.device)
            elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], torch.Tensor):
                result[key] = [v.to(self.device) for v in val]
            else:
                result[key] = val
        return result

    def _prepare_targets(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare targets dictionary for loss function."""
        targets = {
            "group_ids": batch["group_ids"],
            "categories": batch.get("category_idx_tensor", batch.get("categories")),
            "gt_motion_types": batch["gt_motion_types"],
            "gt_axis_directions": batch["gt_axis_directions"],
            "gt_axis_positions": batch["gt_axis_positions"],
            "points": batch["points"],  # For spatial affinity loss
        }

        # Add optional targets if present in batch
        optional_keys = [
            "gt_motion_limits",  # For motion limit loss
            "gt_link_ids",  # For parent prediction
            "gt_is_movable",  # For motion filtering
            "gt_part_class_labels",  # For part classification loss
            "gt_parent_info",  # For parent prediction loss
        ]
        for key in optional_keys:
            if key in batch:
                targets[key] = batch[key]

        return targets

    def _log_batch(
        self,
        batch_idx: int,
        num_batches: int,
        loss_meters: Dict[str, AverageMeter],
        batch_time: AverageMeter,
        data_time: AverageMeter,
        cat_acc: float = None,
    ):
        """Log batch progress."""
        lr = self.optimizer.param_groups[0]["lr"]

        msg = (
            f"Epoch [{self.epoch + 1}/{self.max_epochs}] "
            f"Batch [{batch_idx + 1}/{num_batches}] "
            f"LR: {lr:.2e} "
            f"Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) "
            f"Data: {data_time.val:.3f}s "
        )

        # Add key losses
        for key in ["total_loss", "seg_loss", "motion_loss"]:
            if key in loss_meters:
                msg += f"{key}: {loss_meters[key].val:.4f} ({loss_meters[key].avg:.4f}) "

        # Add category accuracy
        if cat_acc is not None:
            msg += f"cat_acc: {cat_acc:.4f} "

        self.logger.info(msg)

        # TensorBoard
        if self.writer:
            self.writer.add_scalar("train/lr", lr, self.global_step)
            for key, meter in loss_meters.items():
                self.writer.add_scalar(f"train/{key}", meter.val, self.global_step)

        # WandB
        if self.use_wandb:
            log_dict = {"train/lr": lr, "global_step": self.global_step}
            for key, meter in loss_meters.items():
                log_dict[f"train/{key}"] = meter.val
            if cat_acc is not None:
                log_dict["train/category_accuracy"] = cat_acc
            wandb.log(log_dict, step=self.global_step)

    def _log_epoch(
        self,
        train_losses: Dict[str, float],
        val_losses: Optional[Dict[str, float]],
    ):
        """Log epoch summary."""
        lr = self.optimizer.param_groups[0]["lr"]

        # Console log
        msg = f"Epoch [{self.epoch + 1}/{self.max_epochs}] completed - LR: {lr:.2e}"
        msg += f" | Train Loss: {train_losses.get('total_loss', 0):.4f}"
        msg += f" | Train Cat Acc: {train_losses.get('category_accuracy', 0):.4f}"

        if val_losses:
            msg += f" | Val Loss: {val_losses.get('total_loss', 0):.4f}"
            msg += f" | Val Cat Acc: {val_losses.get('category_accuracy', 0):.4f}"

        self.logger.info(msg)

        # Detailed loss breakdown
        self.logger.info("  Train losses:")
        for key, val in sorted(train_losses.items()):
            if "num" not in key and "scale" not in key:
                self.logger.info(f"    {key}: {val:.4f}")

        if val_losses:
            self.logger.info("  Val losses:")
            for key, val in sorted(val_losses.items()):
                if "num" not in key and "scale" not in key:
                    self.logger.info(f"    {key}: {val:.4f}")

        # CSV logging
        self._log_epoch_csv(train_losses, val_losses, lr)

        # TensorBoard
        if self.writer:
            for key, val in train_losses.items():
                self.writer.add_scalar(f"epoch_train/{key}", val, self.epoch)

            if val_losses:
                for key, val in val_losses.items():
                    self.writer.add_scalar(f"epoch_val/{key}", val, self.epoch)

        # WandB
        if self.use_wandb:
            log_dict = {"epoch": self.epoch, "lr": lr}
            for key, val in train_losses.items():
                log_dict[f"epoch_train/{key}"] = val
            if val_losses:
                for key, val in val_losses.items():
                    log_dict[f"epoch_val/{key}"] = val
            wandb.log(log_dict, step=self.global_step)

    def _log_epoch_csv(
        self,
        train_losses: Dict[str, float],
        val_losses: Optional[Dict[str, float]],
        lr: float,
    ):
        """Write epoch metrics to CSV file."""
        if self.csv_log_path is None:
            return

        # Build row data
        row = {
            "epoch": self.epoch + 1,
            "lr": lr,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Add train losses with prefix
        for key, val in train_losses.items():
            if "num" not in key and "scale" not in key:
                row[f"train_{key}"] = val

        # Add val losses with prefix
        if val_losses:
            for key, val in val_losses.items():
                if "num" not in key and "scale" not in key:
                    row[f"val_{key}"] = val

        # Write to CSV
        file_exists = os.path.exists(self.csv_log_path)
        write_header = not file_exists or not self.csv_header_written

        with open(self.csv_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self.csv_header_written = True
            writer.writerow(row)

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)

        # Get the underlying model if using DDP
        model_to_save = self.model.module if self.distributed else self.model

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "warmup_completed": self.warmup_completed,  # Save warmup status
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        temporary_path = f"{path}.tmp"
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        self.logger.info(f"Loading checkpoint: {path}")
        checkpoint = load_torch(path, map_location=self.device)

        # Load model state - handle DDP vs non-DDP
        model_to_load = self.model.module if self.distributed else self.model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        # Load warmup status - if resuming from any checkpoint, warmup is considered done
        # This ensures we don't repeat warmup when resuming
        self.warmup_completed = checkpoint.get("warmup_completed", True)

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.logger.info(
            f"Resumed from epoch {self.epoch}, warmup_completed={self.warmup_completed}"
        )

    # =========================================================================
    # Category Warmup Methods
    # =========================================================================

    def _category_warmup(self):
        """
        Category head warmup training.

        Freezes all parameters except global_encoder and category_head,
        trains only on category classification loss for better initialization.

        Features:
        - Saves checkpoints to a 'warmup' subdirectory
        - Tracks and saves the best model based on validation accuracy
        - Outputs confusion matrix for analysis
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Category Warmup Training")
        self.logger.info(f"  Epochs: {self.category_warmup_epochs}")
        self.logger.info(f"  Learning rate: {self.category_warmup_lr}")
        self.logger.info("=" * 60)

        # Create warmup checkpoint directory
        warmup_checkpoint_dir = os.path.join(self.checkpoint_dir, "warmup")
        os.makedirs(warmup_checkpoint_dir, exist_ok=True)
        self.logger.info(f"  Warmup checkpoints will be saved to: {warmup_checkpoint_dir}")

        # Save original requires_grad state
        original_requires_grad = {
            name: param.requires_grad for name, param in self.model.named_parameters()
        }

        # Freeze all parameters except global_module's encoder and category head
        self._freeze_except_category()

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")

        # Create optimizer for warmup (only for trainable params)
        warmup_optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.category_warmup_lr,
            weight_decay=self.weight_decay,
        )

        # Category loss criterion
        category_criterion = nn.CrossEntropyLoss()

        # Track best model
        best_val_acc = 0.0
        best_epoch = 0

        # Training loop
        for warmup_epoch in range(self.category_warmup_epochs):
            self.model.train()
            train_loss = AverageMeter()
            train_correct = 0
            train_total = 0

            for batch_idx, batch in enumerate(self.train_loader):
                batch = self._to_device(batch)

                # Forward pass (only need category prediction)
                with autocast(device_type="cuda", enabled=self.use_amp):
                    outputs = self.model(
                        partfield_features=batch["partfield_features"],
                        vae_features=batch["vae_features"],
                        points=batch["points"],
                    )

                    # Category loss
                    category_logits = outputs["category_logits"]
                    targets = batch.get("category_idx_tensor", batch.get("categories"))
                    if targets is None:
                        continue
                    targets = targets.to(self.device)

                    loss = category_criterion(category_logits, targets)

                # Backward pass
                warmup_optimizer.zero_grad()
                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(warmup_optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    warmup_optimizer.step()

                # Update metrics
                train_loss.update(loss.item())
                preds = category_logits.argmax(dim=-1)
                train_correct += (preds == targets).sum().item()
                train_total += targets.size(0)

                # Log progress
                if (batch_idx + 1) % self.log_interval == 0:
                    acc = train_correct / train_total if train_total > 0 else 0
                    self.logger.info(
                        f"  Warmup Epoch [{warmup_epoch + 1}/{self.category_warmup_epochs}] "
                        f"Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                        f"Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) "
                        f"Acc: {acc:.4f}"
                    )

            # Calculate training accuracy
            train_acc = train_correct / train_total if train_total > 0 else 0

            # Validation with confusion matrix
            val_acc = 0.0
            val_loss = 0.0
            confusion_matrix = None
            if self.val_loader is not None:
                val_loss, val_acc, per_class_acc, confusion_matrix = self._compute_category_metrics(
                    self.val_loader, return_confusion_matrix=True
                )

            # Log epoch summary
            self.logger.info(
                f"  Warmup Epoch [{warmup_epoch + 1}/{self.category_warmup_epochs}] "
                f"Train Loss: {train_loss.avg:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Log per-class accuracy
            if self.val_loader is not None and per_class_acc:
                self.logger.info("  Per-class Val Accuracy:")
                for cls_name, acc in per_class_acc.items():
                    self.logger.info(f"    {cls_name}: {acc:.4f}")

            # Save confusion matrix (every 5 epochs or last epoch)
            if confusion_matrix is not None and (
                (warmup_epoch + 1) % 5 == 0 or warmup_epoch == self.category_warmup_epochs - 1
            ):
                self._save_confusion_matrix(
                    confusion_matrix, warmup_epoch + 1, warmup_checkpoint_dir
                )

            # TensorBoard
            if self.writer:
                self.writer.add_scalar("warmup/train_loss", train_loss.avg, warmup_epoch)
                self.writer.add_scalar("warmup/train_acc", train_acc, warmup_epoch)
                self.writer.add_scalar("warmup/val_loss", val_loss, warmup_epoch)
                self.writer.add_scalar("warmup/val_acc", val_acc, warmup_epoch)

            # Save checkpoint every epoch during warmup
            if is_main_process():
                checkpoint = {
                    "warmup_epoch": warmup_epoch + 1,
                    "model_state_dict": self.model.module.state_dict()
                    if self.distributed
                    else self.model.state_dict(),
                    "optimizer_state_dict": warmup_optimizer.state_dict(),
                    "train_loss": train_loss.avg,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }

                # Save latest
                latest_path = os.path.join(warmup_checkpoint_dir, "warmup_latest.pth")
                torch.save(checkpoint, latest_path)

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = warmup_epoch + 1
                    best_path = os.path.join(warmup_checkpoint_dir, "warmup_best.pth")
                    torch.save(checkpoint, best_path)
                    self.logger.info(
                        f"  ★ New best model! Val Acc: {val_acc:.4f} (saved to {best_path})"
                    )

                # Save periodic checkpoints (every 5 epochs)
                if (warmup_epoch + 1) % 5 == 0:
                    epoch_path = os.path.join(
                        warmup_checkpoint_dir, f"warmup_epoch{warmup_epoch + 1}.pth"
                    )
                    torch.save(checkpoint, epoch_path)

        # Restore requires_grad state
        for name, param in self.model.named_parameters():
            param.requires_grad = original_requires_grad[name]

        self.logger.info("=" * 60)
        self.logger.info("Category Warmup Completed!")
        self.logger.info(f"  Best Val Accuracy: {best_val_acc:.4f} at Epoch {best_epoch}")
        self.logger.info(
            f"  Best checkpoint: {os.path.join(warmup_checkpoint_dir, 'warmup_best.pth')}"
        )
        self.logger.info("=" * 60)

        # Store best warmup info for later use
        self.best_warmup_acc = best_val_acc
        self.best_warmup_epoch = best_epoch

    def _freeze_except_category(self):
        """
        Freeze all parameters except category-related modules.

        Trainable modules:
        - global_module.global_encoder
        - global_module.category_head
        - global_module.partfield_proj (if using reasoner features for category)
        """
        trainable_prefixes = [
            "global_module.global_encoder",
            "global_module.category_head",
            "global_module.partfield_proj",  # For VAE + reasoner-feature fusion
        ]

        for name, param in self.model.named_parameters():
            # Check if this parameter belongs to a trainable module
            should_train = any(prefix in name for prefix in trainable_prefixes)
            param.requires_grad = should_train

    @torch.no_grad()
    def _compute_category_metrics(
        self, dataloader: DataLoader, return_confusion_matrix: bool = False
    ):
        """
        Compute category classification metrics.

        Args:
            dataloader: DataLoader to evaluate
            return_confusion_matrix: If True, also return confusion matrix

        Returns:
            tuple: (avg_loss, accuracy, per_class_accuracy_dict, [confusion_matrix])
        """
        self.model.eval()

        category_criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        # Per-class tracking - get num_classes from model's category head
        model_to_check = self.model.module if self.distributed else self.model
        num_classes = model_to_check.global_module.category_head[-1].out_features

        # Debug: log num_classes to help diagnose category name issues
        if hasattr(self, "logger"):
            self.logger.info(f"Category classes: {num_classes}")

        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        # Confusion matrix: [num_classes, num_classes]
        # confusion_matrix[i][j] = samples with true label i predicted as j
        confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

        for batch in dataloader:
            batch = self._to_device(batch)

            outputs = self.model(
                partfield_features=batch["partfield_features"],
                vae_features=batch["vae_features"],
                points=batch["points"],
            )

            category_logits = outputs["category_logits"]
            targets = batch.get("category_idx_tensor", batch.get("categories"))
            if targets is None:
                continue
            targets = targets.to(self.device)

            # Loss
            loss = category_criterion(category_logits, targets)
            total_loss += loss.item() * targets.size(0)

            # Accuracy
            preds = category_logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            # Per-class accuracy and confusion matrix
            for i in range(num_classes):
                mask = targets == i
                class_total[i] += mask.sum().item()
                class_correct[i] += ((preds == targets) & mask).sum().item()

            # Update confusion matrix
            for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
                confusion_matrix[t][p] += 1

        avg_loss = total_loss / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0

        # Per-class accuracy dict
        try:
            from monoart.motion.datasets.motion_parser import get_category_names

            category_names = get_category_names(num_classes)
            # Debug: show category names being used
            if hasattr(self, "logger"):
                self.logger.info(f"Category label preview: {list(category_names.values())[:7]}")
            per_class_acc = {}
            for i in range(num_classes):
                if class_total[i] > 0:
                    cls_name = category_names.get(i, f"Class_{i}")
                    per_class_acc[cls_name] = class_correct[i] / class_total[i]
        except ImportError:
            per_class_acc = {
                f"Class_{i}": class_correct[i] / class_total[i]
                for i in range(num_classes)
                if class_total[i] > 0
            }

        if return_confusion_matrix:
            return avg_loss, accuracy, per_class_acc, confusion_matrix
        return avg_loss, accuracy, per_class_acc

    def _save_confusion_matrix(self, confusion_matrix: List[List[int]], epoch: int, save_dir: str):
        """
        Save confusion matrix to CSV file.

        Args:
            confusion_matrix: [num_classes, num_classes] confusion matrix
            epoch: Current epoch number
            save_dir: Directory to save the confusion matrix
        """

        os.makedirs(save_dir, exist_ok=True)

        try:
            from monoart.motion.datasets.motion_parser import get_category_names

            num_classes = len(confusion_matrix)
            category_names = get_category_names(num_classes)
            class_names = [category_names.get(i, f"Class_{i}") for i in range(num_classes)]
        except ImportError:
            class_names = [f"Class_{i}" for i in range(len(confusion_matrix))]

        # Save as CSV
        csv_path = os.path.join(save_dir, f"confusion_matrix_epoch{epoch}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Header: True\Pred, class_0, class_1, ...
            header = ["True\\Pred"] + class_names
            writer.writerow(header)
            # Data rows
            for i, row in enumerate(confusion_matrix):
                writer.writerow([class_names[i]] + row)

        self.logger.info(f"  Confusion matrix saved to: {csv_path}")

        # Also log top confusions
        self._log_top_confusions(confusion_matrix, class_names, top_k=10)

    def _log_top_confusions(
        self, confusion_matrix: List[List[int]], class_names: List[str], top_k: int = 10
    ):
        """Log top K confusion pairs (excluding correct predictions)."""
        confusions = []
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix)):
                if i != j and confusion_matrix[i][j] > 0:
                    confusions.append(
                        {
                            "true": class_names[i],
                            "pred": class_names[j],
                            "count": confusion_matrix[i][j],
                            "total": sum(confusion_matrix[i]),
                            "rate": confusion_matrix[i][j] / max(sum(confusion_matrix[i]), 1),
                        }
                    )

        # Sort by count (descending)
        confusions.sort(key=lambda x: x["count"], reverse=True)

        if confusions:
            self.logger.info(f"  Top {min(top_k, len(confusions))} Confusions:")
            for conf in confusions[:top_k]:
                self.logger.info(
                    f"    {conf['true']:20s} → {conf['pred']:20s}: "
                    f"{conf['count']:3d}/{conf['total']:3d} ({conf['rate'] * 100:.1f}%)"
                )
