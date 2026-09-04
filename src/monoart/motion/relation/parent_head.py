"""
Parent Prediction Head for Kinematic Tree Structure.

Predicts parent-child relationships between parts using:
1. Visual Branch: Bilinear attention between query features
2. Semantic Branch: Learnable affinity based on part classes
3. Root Token: Learnable embedding for root/base connection
"""

from typing import Optional

import torch
import torch.nn as nn

from .config import DEFAULT_AFFINITY, NUM_PART_CLASSES


class ParentPredictionHead(nn.Module):
    """
    Predicts parent node for each query in the kinematic tree.

    Architecture:
        Q_combined = [Q_content; Q_position]  # [B, N, 2D]

        Visual Branch:
            child_feat = child_proj(Q_combined)   # [B, N, D]
            parent_feat = parent_proj(Q_combined) # [B, N, D]
            parent_feat_with_root = [parent_feat; root_token]  # [B, N+1, D]
            visual_score = child_feat @ parent_feat_with_root.T / sqrt(D)

        Semantic Branch (optional):
            affinity = learnable [C, C] matrix
            semantic_score = child_probs @ affinity @ parent_probs.T

        Output:
            parent_logits = visual_score + lambda * semantic_score
            # [B, N, N+1] where N+1 is root token
    """

    def __init__(
        self,
        d_model: int = 448,
        use_position: bool = True,
        use_semantic: bool = True,
        num_classes: int = NUM_PART_CLASSES,
        lambda_semantic_init: float = 0.3,
        init_affinity: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            d_model: Query feature dimension
            use_position: Whether to use Q_position (doubles input dim)
            use_semantic: Whether to use semantic affinity branch
            num_classes: Number of part classes (for semantic branch)
            lambda_semantic_init: Initial weight for semantic branch
            init_affinity: Initial affinity matrix [C, C], uses default if None
        """
        super().__init__()

        self.d_model = d_model
        self.use_position = use_position
        self.use_semantic = use_semantic
        self.num_classes = num_classes

        # Input dimension
        input_dim = 2 * d_model if use_position else d_model

        # Visual branch projections
        self.child_proj = nn.Linear(input_dim, d_model)
        self.parent_proj = nn.Linear(input_dim, d_model)

        # Learnable root token
        self.root_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Semantic branch (optional)
        if use_semantic:
            # Learnable affinity matrix [C, C]
            if init_affinity is not None:
                self.affinity_matrix = nn.Parameter(init_affinity.clone())
            else:
                self.affinity_matrix = nn.Parameter(DEFAULT_AFFINITY.clone())

            # Learnable weight for semantic branch
            self.lambda_semantic = nn.Parameter(torch.tensor(lambda_semantic_init))

            # Root affinity (probability of each class connecting to root)
            self.root_affinity = nn.Parameter(torch.zeros(num_classes))

        # Scale factor for attention
        self.scale = d_model**-0.5

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.child_proj.weight)
        nn.init.xavier_uniform_(self.parent_proj.weight)
        nn.init.zeros_(self.child_proj.bias)
        nn.init.zeros_(self.parent_proj.bias)

    def forward(
        self,
        q_content: torch.Tensor,
        q_position: Optional[torch.Tensor] = None,
        class_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            q_content: Query content features [B, N, D]
            q_position: Query position features [B, N, D] (optional)
            class_probs: Part class probabilities [B, N, C] (optional, for semantic)

        Returns:
            parent_logits: [B, N, N+1] logits for parent prediction
                          [:, :, :N] = other queries as parent
                          [:, :, N] = root token as parent
        """
        B, N, D = q_content.shape

        # Combine content and position features
        if self.use_position and q_position is not None:
            q_combined = torch.cat([q_content, q_position], dim=-1)  # [B, N, 2D]
        else:
            q_combined = q_content  # [B, N, D]

        # Visual branch
        child_feat = self.child_proj(q_combined)  # [B, N, D]
        parent_feat = self.parent_proj(q_combined)  # [B, N, D]

        # Add root token to parent features
        root_expanded = self.root_token.expand(B, 1, D)  # [B, 1, D]
        parent_feat_with_root = torch.cat([parent_feat, root_expanded], dim=1)  # [B, N+1, D]

        # Compute visual attention scores
        visual_score = (
            torch.bmm(child_feat, parent_feat_with_root.transpose(1, 2)) * self.scale
        )  # [B, N, N+1]

        # Semantic branch (optional)
        if self.use_semantic and class_probs is not None:
            semantic_score = self._compute_semantic_score(class_probs)
            # Combine visual and semantic
            lambda_val = torch.sigmoid(self.lambda_semantic)  # Bound to [0, 1]
            parent_logits = visual_score + lambda_val * semantic_score
        else:
            parent_logits = visual_score

        return parent_logits

    def _compute_semantic_score(
        self,
        class_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute semantic affinity scores between queries.

        Args:
            class_probs: [B, N, C] class probabilities

        Returns:
            semantic_score: [B, N, N+1] semantic affinity scores
        """
        B, N, C = class_probs.shape

        # Compute pairwise affinity: [B, N, N]
        # semantic_score[b, i, j] = sum_c,d class_probs[b,i,c] * A[c,d] * class_probs[b,j,d]
        # This simplifies to: child_probs @ A @ parent_probs.T

        affinity = self.affinity_matrix  # [C, C]

        # child @ affinity: [B, N, C]
        child_affinity = torch.einsum("bnc,cd->bnd", class_probs, affinity)

        # @ parent.T: [B, N, N]
        semantic_pairwise = torch.bmm(child_affinity, class_probs.transpose(1, 2))

        # Root affinity: probability of connecting to root based on class
        root_affinity = torch.sigmoid(self.root_affinity)  # [C]
        root_score = torch.einsum("bnc,c->bn", class_probs, root_affinity)  # [B, N]
        root_score = root_score.unsqueeze(-1)  # [B, N, 1]

        # Combine: [B, N, N+1]
        semantic_score = torch.cat([semantic_pairwise, root_score], dim=-1)

        return semantic_score

    def get_parent_prediction(
        self,
        parent_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get parent prediction from logits (simple argmax).

        Args:
            parent_logits: [B, N, N+1]

        Returns:
            parent_pred: [B, N] indices (N = root)
        """
        return parent_logits.argmax(dim=-1)
