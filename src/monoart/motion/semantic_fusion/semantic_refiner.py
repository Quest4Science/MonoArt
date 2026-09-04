"""
Semantic-Gated Query Refiner (EASE-style).

Provides gated injection of CLIP semantic features into query representations.

Key Design Principles (from clip4.md and clip5.md):
1. Only update Q_content, keep Q_position pure
   - CLIP knows "this is a handle" (shape) but NOT "handle at (0.5, 0.5, 0.5)" (coordinate)
   - Injecting into Q_position would corrupt geometric meaning

2. Q_position as Gate reference (position-aware semantic injection)
   - Gate uses Q_pos to learn: "if at bottom + classified as wheel -> inject strongly"

3. Residual connection for training stability
   - Initial gate ~ 0, equivalent to original model
   - Safe to add without breaking pretrained weights

4. CLIP as "Geometric Navigator" for Axis regression
   - Door embedding -> "edge detection mode" (axis at hinge)
   - Knob embedding -> "center detection mode" (axis at center)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEFAULT_CLIP_EMBEDDING_PATH, NUM_PART_CLASSES

logger = logging.getLogger(__name__)


class SemanticGatedQueryRefiner(nn.Module):
    """
    EASE-style gated semantic injection for MAFT architecture.

    Uses CLIP text embeddings to provide semantic guidance for queries.
    The gate mechanism allows the model to learn when and how much
    to inject semantic information.

    Args:
        d_model: Query feature dimension
        num_classes: Number of part classes (default: 18)
        clip_dim: CLIP embedding dimension (default: 512)
        embedding_path: Path to CLIP embeddings file
        use_position_in_gate: Include Q_position in gate computation
        hidden_dim: Hidden dimension for gate/transform networks
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int = NUM_PART_CLASSES,
        clip_dim: int = 512,
        embedding_path: Optional[str] = None,
        use_position_in_gate: bool = True,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_classes = num_classes
        self.clip_dim = clip_dim
        self.use_position_in_gate = use_position_in_gate

        # Load CLIP embeddings
        embedding_path = embedding_path or DEFAULT_CLIP_EMBEDDING_PATH
        self._load_clip_embeddings(embedding_path)

        # Adapter: CLIP dim -> model dim
        self.text_adapter = nn.Linear(clip_dim, d_model)

        # Gate network dimensions
        hidden_dim = hidden_dim or d_model
        gate_input_dim = 3 * d_model if use_position_in_gate else 2 * d_model

        # Gate network: computes how much semantic to inject
        # Uses Q_content + Q_position + Text to compute gate
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )

        # Transform network: computes semantic delta
        # Only uses Q_content + Text (no position)
        self.transform_mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

        # Layer norm after injection
        self.norm = nn.LayerNorm(d_model)

        # Initialize gate network to output small values (minimal initial injection)
        self._init_weights()

    def _load_clip_embeddings(self, embedding_path: str):
        """Load CLIP embeddings from file."""
        try:
            data = torch.load(embedding_path, map_location="cpu", weights_only=True)
            class_embeddings = data["embeddings"]
        except (FileNotFoundError, KeyError, OSError) as exc:
            raise RuntimeError(f"Unable to load CLIP embeddings from {embedding_path}") from exc
        expected = (self.num_classes, self.clip_dim)
        if tuple(class_embeddings.shape) != expected:
            raise ValueError(
                f"Expected CLIP embeddings shaped {expected}, got {tuple(class_embeddings.shape)}"
            )
        logger.info("Loaded CLIP embeddings from %s", embedding_path)

        self.register_buffer("class_embeddings", class_embeddings)

    def _init_weights(self):
        """Initialize weights for minimal initial injection."""
        # Initialize gate output layer to output small values
        # This makes initial gate ~ 0, so model starts from original behavior
        nn.init.zeros_(self.gate_mlp[-2].weight)
        nn.init.constant_(self.gate_mlp[-2].bias, -2.0)  # sigmoid(-2) ~ 0.12

    def forward(
        self,
        q_content: torch.Tensor,
        q_position: torch.Tensor,
        class_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            q_content: [B, N_query, D_model] content query (will be updated)
            q_position: [B, N_query, D_model] position query (read-only reference)
            class_logits: [B, N_query, 18] part classification logits

        Returns:
            q_content_new: [B, N_query, D_model] updated content query
        """
        # 1. Soft embedding lookup
        class_probs = F.softmax(class_logits, dim=-1)  # [B, Q, 18]
        text_emb = self.text_adapter(self.class_embeddings)  # [18, D]
        text_feat = torch.matmul(class_probs, text_emb)  # [B, Q, D]

        # 2. Compute gate (position-aware if enabled)
        if self.use_position_in_gate:
            # Gate uses Q_pos to learn: "if at bottom + classified as wheel -> inject strongly"
            feat_all = torch.cat([q_content, q_position, text_feat], dim=-1)  # [B, Q, 3D]
        else:
            feat_all = torch.cat([q_content, text_feat], dim=-1)  # [B, Q, 2D]

        gate = self.gate_mlp(feat_all)  # [B, Q, D]

        # 3. Compute semantic delta (content + text only)
        feat_sem = torch.cat([q_content, text_feat], dim=-1)  # [B, Q, 2D]
        delta = self.transform_mlp(feat_sem)  # [B, Q, D]

        # 4. Gated residual injection (only update content)
        q_content_new = q_content + gate * delta
        q_content_new = self.norm(q_content_new)

        return q_content_new

    def forward_with_class_probs(
        self,
        q_content: torch.Tensor,
        q_position: torch.Tensor,
        class_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass using pre-computed class probabilities.

        Args:
            q_content: [B, N_query, D_model] content query
            q_position: [B, N_query, D_model] position query
            class_probs: [B, N_query, 18] classification probabilities

        Returns:
            q_content_new: [B, N_query, D_model] updated content query
        """
        text_emb = self.text_adapter(self.class_embeddings)
        text_feat = torch.matmul(class_probs, text_emb)

        if self.use_position_in_gate:
            feat_all = torch.cat([q_content, q_position, text_feat], dim=-1)
        else:
            feat_all = torch.cat([q_content, text_feat], dim=-1)

        gate = self.gate_mlp(feat_all)
        feat_sem = torch.cat([q_content, text_feat], dim=-1)
        delta = self.transform_mlp(feat_sem)

        q_content_new = q_content + gate * delta
        q_content_new = self.norm(q_content_new)

        return q_content_new

    def get_gate_values(
        self,
        q_content: torch.Tensor,
        q_position: torch.Tensor,
        class_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get gate values for analysis/visualization.

        Returns:
            gate: [B, N_query, D_model] gate values in [0, 1]
        """
        class_probs = F.softmax(class_logits, dim=-1)
        text_emb = self.text_adapter(self.class_embeddings)
        text_feat = torch.matmul(class_probs, text_emb)

        if self.use_position_in_gate:
            feat_all = torch.cat([q_content, q_position, text_feat], dim=-1)
        else:
            feat_all = torch.cat([q_content, text_feat], dim=-1)

        return self.gate_mlp(feat_all)


# =============================================================================
# Test
# =============================================================================
