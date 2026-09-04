"""
Iterative Semantic Fusion Module.

Implements EASE-style iterative semantic fusion across Decoder layers.
Instead of applying SemanticFusion only once at the final layer,
this module applies it after each decoder layer with configurable gating.

Key Features:
- Configurable gate scales per layer (increasing, decreasing, peak_middle, custom)
- Skip MotionPriorLookup in intermediate layers (only use at final layer)
- Deep supervision support for intermediate layers
- Full backward compatibility (enabled=False equals original architecture)

Reference:
- EASE: Edge-Aware 3D Instance Segmentation Network with Intelligent Semantic Prior (CVPR 2024)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class IterativeSemanticFusionConfig:
    """Configuration for Iterative Semantic Fusion."""

    # Master switch - when False, equals original architecture
    enabled: bool = False

    # Iteration mode
    mode: str = "all_layers"  # "all_layers", "first_n", "skip_k", "custom"
    first_n_layers: int = 4  # For mode="first_n"
    skip_k_interval: int = 2  # For mode="skip_k"

    # Gate strategy
    gate_strategy: str = (
        "increasing"  # "fixed", "decreasing", "increasing", "peak_middle", "custom"
    )
    fixed_gate_scale: float = 1.0  # For gate_strategy="fixed"
    custom_gate_scales: Optional[List[float]] = None  # For gate_strategy="custom"

    # Component sharing
    share_classification_head: bool = True  # All layers share the same PartClassHead

    # Deep supervision
    deep_supervision: bool = False  # Compute part_class_loss at intermediate layers
    deep_supervision_weight: float = 0.1  # Weight for intermediate layer losses

    # Fusion position
    fusion_position: str = "after_layer"  # "after_layer" or "before_layer"

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]]) -> "IterativeSemanticFusionConfig":
        """Create config from dictionary (e.g., from YAML)."""
        if config_dict is None:
            return cls()
        # Filter only valid fields
        valid_fields = {k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


class IterativeSemanticFusion(nn.Module):
    """
    Manages iterative semantic fusion across Decoder layers.

    This module wraps SemanticFusionModule and applies it at each decoder layer
    with configurable gating. The key insight from EASE is that applying semantic
    guidance iteratively creates a "compound interest" effect, where early layers
    get semantic hints that improve cross-attention focus.

    Usage:
        iter_fusion = IterativeSemanticFusion(config, semantic_fusion_module)

        for i, layer in enumerate(decoder_layers):
            query = layer(query, memory, ...)
            query, aux_outputs = iter_fusion(query, query_pos, layer_idx=i)

    Args:
        config: IterativeSemanticFusionConfig
        semantic_fusion: SemanticFusionModule instance to wrap
        num_layers: Number of decoder layers (default: 6)
    """

    def __init__(
        self,
        config: IterativeSemanticFusionConfig,
        semantic_fusion: nn.Module,  # SemanticFusionModule
        num_layers: int = 6,
    ):
        super().__init__()
        self.config = config
        self.semantic_fusion = semantic_fusion
        self.num_layers = num_layers

        # Pre-compute gate scales for each layer
        gate_scales = self._compute_gate_scales()
        self.register_buffer("gate_scales", torch.tensor(gate_scales, dtype=torch.float32))

        # Log configuration
        logger.info(
            "Initialized iterative semantic fusion: mode=%s strategy=%s scales=%s "
            "deep_supervision=%s",
            config.mode,
            config.gate_strategy,
            gate_scales,
            config.deep_supervision,
        )

    def _compute_gate_scales(self) -> List[float]:
        """Compute gate scale for each layer based on configuration."""
        cfg = self.config
        n = self.num_layers

        # First, compute base scales based on strategy
        if cfg.gate_strategy == "fixed":
            scales = [cfg.fixed_gate_scale] * n

        elif cfg.gate_strategy == "decreasing":
            # 1.0 -> 0.2 linearly
            step = 0.8 / (n - 1) if n > 1 else 0
            scales = [1.0 - i * step for i in range(n)]

        elif cfg.gate_strategy == "increasing":
            # 0.2 -> 1.0 linearly
            step = 0.8 / (n - 1) if n > 1 else 0
            scales = [0.2 + i * step for i in range(n)]

        elif cfg.gate_strategy == "peak_middle":
            # Peak at middle layer (Layer 3-4 for 6 layers)
            mid = (n - 1) / 2
            scales = [1.0 - 0.15 * abs(i - mid) for i in range(n)]
            scales = [max(0.3, s) for s in scales]  # Minimum 0.3

        elif cfg.gate_strategy == "custom":
            if cfg.custom_gate_scales is not None:
                scales = list(cfg.custom_gate_scales)
                # Ensure correct length
                while len(scales) < n:
                    scales.append(scales[-1] if scales else 1.0)
                scales = scales[:n]
            else:
                scales = [1.0] * n

        else:
            # Default to all 1.0
            scales = [1.0] * n

        # Apply mode restrictions
        if cfg.mode == "first_n":
            for i in range(cfg.first_n_layers, n):
                scales[i] = 0.0

        elif cfg.mode == "skip_k":
            for i in range(n):
                if (i + 1) % cfg.skip_k_interval != 0:
                    scales[i] = 0.0

        return scales

    def should_apply_at_layer(self, layer_idx: int) -> bool:
        """Check if semantic fusion should be applied at this layer."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return False
        return self.gate_scales[layer_idx].item() > 0

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Apply semantic fusion at specified layer.

        Args:
            query: [B, Q, D] Current layer's output query features
            query_pos: [B, Q, D] Query position embeddings
            layer_idx: Current layer index (0-based, 0 to num_layers-1)

        Returns:
            query: [B, Q, D] Fused query features
            aux_outputs: Dict with auxiliary outputs for deep supervision (or None)
        """
        gate_scale = self.gate_scales[layer_idx].item()

        # Skip fusion if gate is 0
        if gate_scale == 0.0:
            return query, None

        # Call SemanticFusion with skip_motion_prior=True for intermediate layers
        # (MotionPrior is only needed at the final layer for MotionHead)
        fusion_output = self.semantic_fusion(
            query=query,
            query_pos=query_pos,
            skip_motion_prior=True,  # Always skip in iterative mode
        )

        # Get refined query
        query_refined = fusion_output.get("query", query)

        # Apply gated mixing
        if gate_scale < 1.0:
            query_out = query + gate_scale * (query_refined - query)
        else:
            query_out = query_refined

        # Collect auxiliary outputs for deep supervision
        aux_outputs = None
        if self.config.deep_supervision and self.training:
            aux_outputs = {
                "layer_idx": layer_idx,
                "gate_scale": gate_scale,
                "class_logits": fusion_output.get("class_logits", None),
                "class_probs": fusion_output.get("class_probs", None),
            }

        return query_out, aux_outputs

    def get_gate_info(self) -> Dict[str, Any]:
        """Get gate configuration info for logging."""
        return {
            "mode": self.config.mode,
            "gate_strategy": self.config.gate_strategy,
            "gate_scales": self.gate_scales.tolist(),
            "active_layers": [i for i, s in enumerate(self.gate_scales.tolist()) if s > 0],
            "deep_supervision": self.config.deep_supervision,
        }


def compute_iterative_deep_supervision_loss(
    iterative_aux_outputs: List[Optional[Dict[str, Any]]],
    gt_part_classes: torch.Tensor,
    matched_query_indices: torch.Tensor,
    matched_gt_indices: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    intermediate_weight: float = 0.1,
) -> torch.Tensor:
    """
    Compute deep supervision loss for iterative semantic fusion.

    Uses the same matching as the final layer (from Hungarian matcher)
    to supervise intermediate layer classifications.

    Args:
        iterative_aux_outputs: List of auxiliary outputs from each layer
        gt_part_classes: [B, G] Ground truth part class labels
        matched_query_indices: [M] Indices of matched queries (from final layer matching)
        matched_gt_indices: [M] Indices of matched GT parts
        class_weights: [18] Optional class weights for imbalanced data
        intermediate_weight: Weight for intermediate layer losses (final layer = 1.0)

    Returns:
        total_loss: Weighted sum of intermediate layer losses
    """
    import torch.nn.functional as F

    total_loss = torch.tensor(0.0, device=gt_part_classes.device)
    num_valid_layers = 0

    for aux_out in iterative_aux_outputs:
        if aux_out is None:
            continue

        class_logits = aux_out.get("class_logits", None)
        if class_logits is None:
            continue

        # class_logits: [B, Q, 18]
        # We need to extract logits for matched queries and compute CE loss

        B, Q, C = class_logits.shape

        # For simplicity, assume B=1 (batch processing can be added later)
        if B == 1:
            # Extract matched query logits
            matched_logits = class_logits[0, matched_query_indices]  # [M, 18]
            matched_targets = gt_part_classes[0, matched_gt_indices]  # [M]

            # Filter out invalid targets (e.g., -1 for unknown classes)
            valid_mask = matched_targets >= 0
            if valid_mask.sum() == 0:
                continue

            valid_logits = matched_logits[valid_mask]
            valid_targets = matched_targets[valid_mask]

            # Compute cross-entropy loss
            layer_loss = F.cross_entropy(
                valid_logits,
                valid_targets,
                weight=class_weights,
                reduction="mean",
            )

            total_loss = total_loss + intermediate_weight * layer_loss
            num_valid_layers += 1

    return total_loss


# =============================================================================
# Test
# =============================================================================
