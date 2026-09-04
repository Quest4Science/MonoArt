"""
Part Classification Head for Semantic Fusion.

Classifies each query into one of 18 merged part classes.
The classification probabilities are used for:
1. Motion Prior Lookup (soft retrieval)
2. Semantic Embedding Lookup (Gated Fusion)
3. Part Classification Loss (auxiliary supervision)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CLASS_WEIGHTS, NUM_PART_CLASSES


class PartClassificationHead(nn.Module):
    """
    Part Classification Head.

    Takes query features and predicts part class logits.

    Args:
        d_model: Input feature dimension
        num_classes: Number of output classes (default: 18)
        hidden_dim: Hidden layer dimension (optional, for MLP variant)
        use_mlp: Use MLP instead of single linear layer
        dropout: Dropout rate for MLP variant
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int = NUM_PART_CLASSES,
        hidden_dim: Optional[int] = None,
        use_mlp: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_classes = num_classes
        self.use_mlp = use_mlp

        if use_mlp:
            hidden_dim = hidden_dim or d_model // 2
            self.classifier = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            query: [B, N_query, D_model] query features

        Returns:
            class_logits: [B, N_query, num_classes] classification logits
        """
        return self.classifier(query)

    def get_probs(self, query: torch.Tensor) -> torch.Tensor:
        """
        Get classification probabilities.

        Args:
            query: [B, N_query, D_model] query features

        Returns:
            class_probs: [B, N_query, num_classes] softmax probabilities
        """
        logits = self.forward(query)
        return F.softmax(logits, dim=-1)


class PartClassificationLoss(nn.Module):
    """
    Part Classification Loss with optional class balancing.

    Args:
        num_classes: Number of classes
        use_class_weights: Use class weights for imbalanced data
        label_smoothing: Label smoothing factor
    """

    def __init__(
        self,
        num_classes: int = NUM_PART_CLASSES,
        use_class_weights: bool = True,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_class_weights = use_class_weights
        self.label_smoothing = label_smoothing

        if use_class_weights:
            self.register_buffer("class_weights", CLASS_WEIGHTS.clone())
        else:
            self.class_weights = None

    def forward(
        self,
        class_logits: torch.Tensor,
        gt_class: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute classification loss.

        Args:
            class_logits: [B, N_query, num_classes] predicted logits
            gt_class: [B, N_query] ground truth class indices (0 to num_classes-1)
                      or [B, N_query, num_classes] one-hot/soft labels
            mask: [B, N_query] valid query mask (1 for valid, 0 for invalid)

        Returns:
            loss: Scalar loss value
        """
        B, Q, C = class_logits.shape

        # Reshape for cross entropy
        logits_flat = class_logits.view(-1, C)  # [B*Q, C]

        # Handle different gt formats
        if gt_class.dim() == 2:
            # Integer labels [B, Q]
            gt_flat = gt_class.view(-1)  # [B*Q]
            loss = F.cross_entropy(
                logits_flat,
                gt_flat,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
                reduction="none",
            )  # [B*Q]
        else:
            # Soft labels [B, Q, C]
            gt_flat = gt_class.view(-1, C)  # [B*Q, C]
            log_probs = F.log_softmax(logits_flat, dim=-1)
            loss = -torch.sum(gt_flat * log_probs, dim=-1)  # [B*Q]

        # Apply mask
        if mask is not None:
            mask_flat = mask.view(-1)  # [B*Q]
            loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss


# =============================================================================
# Test
# =============================================================================
