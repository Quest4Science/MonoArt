"""
Motion Prior Module for Semantic Fusion.

Provides two key components:
1. MotionPriorLookup: Soft lookup of motion priors from part classification
2. MotionTypeHeadWithResidual: Combines prior lookup with learnable residual

Design Philosophy (from clip4.md):
- Table lookup provides a safe "base" (100% accurate on training data)
- Neural network learns "residual" to handle special cases
- When network outputs 0, degrades to pure lookup (safe fallback)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MOTION_PRIOR_TABLE


class MotionPriorLookup(nn.Module):
    """
    Soft lookup of motion priors using classification probabilities.

    Formula:
        prior_motion = class_probs @ MOTION_PRIOR_TABLE
        # [B, Q, 18] @ [18, 4] -> [B, Q, 4]

    This provides a soft retrieval where if classification is uncertain,
    the motion prior reflects that uncertainty.

    Example:
        - If class_probs = [1, 0, 0, ...] (100% button)
          -> prior_motion = [0, 1, 0, 0] (100% Prismatic)
        - If class_probs = [0.5, 0.5, 0, ...] (50% button, 50% door)
          -> prior_motion = [0, 0.5, 0.5, 0] (mixed P and R)
    """

    def __init__(self):
        super().__init__()
        # Register as buffer (not learnable, but moves with device)
        self.register_buffer("prior_table", MOTION_PRIOR_TABLE.clone())

    def forward(
        self,
        class_logits: torch.Tensor,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """
        Soft lookup of motion priors.

        Args:
            class_logits: [B, N_query, 18] classification logits
            return_probs: If True, also return class_probs

        Returns:
            prior_motion: [B, N_query, 4] motion type prior probabilities
            (optional) class_probs: [B, N_query, 18] classification probabilities
        """
        class_probs = F.softmax(class_logits, dim=-1)  # [B, Q, 18]
        prior_motion = torch.matmul(class_probs, self.prior_table)  # [B, Q, 4]

        if return_probs:
            return prior_motion, class_probs
        return prior_motion

    def forward_with_probs(
        self,
        class_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Lookup using pre-computed class probabilities.

        Args:
            class_probs: [B, N_query, 18] classification probabilities

        Returns:
            prior_motion: [B, N_query, 4] motion type prior probabilities
        """
        return torch.matmul(class_probs, self.prior_table)


class MotionTypeHeadWithResidual(nn.Module):
    """
    Motion Type Prediction with Prior + Residual Learning.

    Formula:
        motion_logits = log(prior_motion + eps) + residual_net(query)

    Benefits:
    - Day 1: Model knows "drawer -> Prismatic" without learning
    - Long-tail friendly: Even with 69 stapler samples, prior is correct
    - Flexible: Can override prior for unusual cases (e.g., rotating drawer)
    - Safe fallback: When residual=0, degrades to pure prior

    Args:
        d_model: Input query dimension
        num_motion_types: Number of motion types (default: 4)
        hidden_dim: Hidden dimension for residual network
        use_adaptive_mix: Use learnable mixing weight (optional)
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        d_model: int,
        num_motion_types: int = 4,
        hidden_dim: Optional[int] = None,
        use_adaptive_mix: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.num_motion_types = num_motion_types
        self.eps = eps
        self.use_adaptive_mix = use_adaptive_mix

        # Prior lookup
        self.prior_lookup = MotionPriorLookup()

        # Residual network: learns correction to prior
        hidden_dim = hidden_dim or d_model // 2
        self.residual_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_motion_types),
        )

        # Initialize residual network to output near-zero (start from prior)
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)

        # Optional: learnable mixing weight
        if use_adaptive_mix:
            self.mix_gate = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        query: torch.Tensor,
        class_logits: torch.Tensor,
        return_prior: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            query: [B, N_query, D_model] query features
            class_logits: [B, N_query, 18] part classification logits
            return_prior: If True, also return prior_motion

        Returns:
            motion_logits: [B, N_query, 4] motion type logits
            (optional) prior_motion: [B, N_query, 4] prior probabilities
        """
        # Get prior from classification
        prior_motion = self.prior_lookup(class_logits)  # [B, Q, 4]

        # Prior as log-space bias (with smoothing to avoid log(0))
        prior_logits = torch.log(prior_motion + self.eps)  # [B, Q, 4]

        # Residual correction from visual features
        residual = self.residual_net(query)  # [B, Q, 4]

        if self.use_adaptive_mix:
            # Adaptive mixing: alpha * prior + (1-alpha) * residual
            alpha = self.mix_gate(query)  # [B, Q, 1]
            motion_logits = alpha * prior_logits + (1 - alpha) * residual
        else:
            # Simple addition: prior provides base, residual allows correction
            motion_logits = prior_logits + residual

        if return_prior:
            return motion_logits, prior_motion
        return motion_logits

    def forward_with_class_probs(
        self,
        query: torch.Tensor,
        class_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass using pre-computed class probabilities.

        Args:
            query: [B, N_query, D_model] query features
            class_probs: [B, N_query, 18] classification probabilities

        Returns:
            motion_logits: [B, N_query, 4] motion type logits
        """
        prior_motion = self.prior_lookup.forward_with_probs(class_probs)
        prior_logits = torch.log(prior_motion + self.eps)
        residual = self.residual_net(query)

        if self.use_adaptive_mix:
            alpha = self.mix_gate(query)
            return alpha * prior_logits + (1 - alpha) * residual
        return prior_logits + residual


# =============================================================================
# Test
# =============================================================================
