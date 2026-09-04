"""
Motion Prediction Head for Articulated Object Motion Estimation.

Predicts motion parameters for each query:
- Motion type: Fixed (0), Prismatic (1), Revolute (2), Continuous (3)
- Axis direction: Normalized 3D vector
- Axis origin: 3D position (predicted as offset from query position)
- Motion limit: [min, max] range for P and R types (using center-span parameterization)

Input:
    - content_query: [B, Q, D] refined query features from decoder
    - position_query: [B, Q, 3] query positions (part centers)
    - global_emb: [B, d_global] global embedding (optional)
    - part_feat: [B, Q, d_part_feat] part geometric features (optional, new in v2)
        Contains: basic_feat [448] + pos_mean_enc [64] + pos_std_proj [64] = 576

Output:
    - motion_type_logits: [B, Q, 4] classification logits
    - axis_direction: [B, Q, 3] normalized direction vector
    - axis_origin: [B, Q, 3] origin position in world coordinates
    - revolute_limit: [B, Q, 2] (center, span) for revolute motion (normalized to π)
    - prismatic_limit: [B, Q, 2] (center, span) for prismatic motion
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionHead(nn.Module):
    """
    Motion parameter prediction head.

    Predicts:
    - Motion type: 4-class classification (Fixed/Prismatic/Revolute/Continuous)
    - Axis direction: Unit vector (L2 normalized output)
    - Axis origin: World coordinate (query_pos + offset)
    - Motion limit: Center-span parameterization for P and R types
      - Revolute limit: normalized to π (center, span in units of π)
      - Prismatic limit: scene-normalized scale (center, span)
    """

    def __init__(
        self,
        d_model: int = 448,
        d_global: int = 0,
        num_motion_types: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        # Part geometric feature config (new in v2)
        use_part_geometric_feat: bool = False,
        d_part_feat: int = 576,  # 448 + 64 + 64
    ):
        """
        Args:
            d_model: Query feature dimension
            d_global: Global embedding dimension (0 to disable)
            num_motion_types: Number of motion types (default: 4)
            hidden_dim: Hidden layer dimension (default: same as d_model)
            dropout: Dropout rate
            use_part_geometric_feat: Whether to use part geometric features (new in v2)
            d_part_feat: Part geometric feature dimension (default: 576 = 448+64+64)
        """
        super().__init__()

        self.d_model = d_model
        self.d_global = d_global
        self.num_motion_types = num_motion_types
        self.use_part_geometric_feat = use_part_geometric_feat
        self.d_part_feat = d_part_feat

        # Input dimension: content + position + (optional) part_feat + (optional) global
        if use_part_geometric_feat:
            # content_query + part_feat + position_query + global_emb
            input_dim = d_model + d_part_feat + 3 + d_global
        else:
            # Original: content_query + position_query + global_emb
            input_dim = d_model + 3 + d_global

        hidden_dim = hidden_dim or d_model
        self.hidden_dim = hidden_dim

        # Shared feature extractor
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 1. Motion type classification head (4 classes)
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_motion_types),
        )

        # 2. Axis direction prediction head
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
        )

        # 3. Axis origin offset prediction head
        # Predicts offset from query position to axis origin
        self.origin_offset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
        )

        # 4. Revolute limit expert head (for R type)
        # Output: [center, span_raw] where span = softplus(span_raw) > 0
        # Values are normalized to π (i.e., output 1.0 means π radians)
        self.revolute_limit_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
        )

        # 5. Prismatic limit expert head (for P type)
        # Output: [center, span_raw] where span = softplus(span_raw) > 0
        # Values are in scene-normalized scale
        self.prismatic_limit_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
        )

        # Softplus for ensuring span > 0
        self.softplus = nn.Softplus()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize origin offset to zero (start from query position)
        nn.init.zeros_(self.origin_offset_head[-1].weight)
        nn.init.zeros_(self.origin_offset_head[-1].bias)

        # Initialize direction head with small values
        nn.init.xavier_uniform_(self.direction_head[-1].weight, gain=0.1)

        # Initialize limit heads with small values
        # Center starts at 0, span_raw starts at 0 (softplus(0) ≈ 0.69)
        nn.init.xavier_uniform_(self.revolute_limit_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.revolute_limit_head[-1].bias)
        nn.init.xavier_uniform_(self.prismatic_limit_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.prismatic_limit_head[-1].bias)

    def forward(
        self,
        content_query: torch.Tensor,
        position_query: torch.Tensor,
        global_emb: Optional[torch.Tensor] = None,
        part_feat: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            content_query: [B, Q, D] query content features
            position_query: [B, Q, 3] query positions
            global_emb: [B, d_global] global embedding (optional)
            part_feat: [B, Q, d_part_feat] part geometric features (optional, new in v2)
            return_intermediate: Return intermediate values for debugging

        Returns:
            Dictionary containing:
            - motion_type_logits: [B, Q, 4] type classification logits
            - axis_direction: [B, Q, 3] normalized axis direction
            - axis_origin: [B, Q, 3] axis origin in world coordinates
            - axis_origin_offset: [B, Q, 3] predicted offset (for debugging)
            - revolute_limit: [B, Q, 2] (center, span) for revolute motion
            - prismatic_limit: [B, Q, 2] (center, span) for prismatic motion
        """
        B, Q, D = content_query.shape

        # Concatenate inputs
        # Order: content_query, part_feat (if used), position_query, global_emb
        inputs = [content_query]

        if self.use_part_geometric_feat and part_feat is not None:
            inputs.append(part_feat)  # [B, Q, d_part_feat]

        inputs.append(position_query)  # [B, Q, 3]

        if global_emb is not None and self.d_global > 0:
            # Expand global_emb to match query dimension
            global_expanded = global_emb.unsqueeze(1).expand(-1, Q, -1)  # [B, Q, d_global]
            inputs.append(global_expanded)

        h_input = torch.cat(inputs, dim=-1)  # [B, Q, input_dim]

        # Shared features
        h_shared = self.shared_mlp(h_input)  # [B, Q, hidden_dim]

        # Predict motion type (4 classes)
        motion_type_logits = self.type_head(h_shared)  # [B, Q, 4]

        # Predict axis direction (normalized)
        axis_direction_raw = self.direction_head(h_shared)  # [B, Q, 3]
        axis_direction = F.normalize(axis_direction_raw, p=2, dim=-1, eps=1e-6)

        # Predict axis origin offset
        axis_origin_offset = self.origin_offset_head(h_shared)  # [B, Q, 3]
        axis_origin = position_query + axis_origin_offset  # World coordinates

        # Predict revolute limit (center-span parameterization)
        rev_limit_raw = self.revolute_limit_head(h_shared)  # [B, Q, 2]
        rev_center = rev_limit_raw[..., 0]  # [B, Q] - can be any value
        rev_span = self.softplus(rev_limit_raw[..., 1])  # [B, Q] - always > 0
        revolute_limit = torch.stack([rev_center, rev_span], dim=-1)  # [B, Q, 2]

        # Predict prismatic limit (center-span parameterization)
        pri_limit_raw = self.prismatic_limit_head(h_shared)  # [B, Q, 2]
        pri_center = pri_limit_raw[..., 0]  # [B, Q]
        pri_span = self.softplus(pri_limit_raw[..., 1])  # [B, Q] - always > 0
        prismatic_limit = torch.stack([pri_center, pri_span], dim=-1)  # [B, Q, 2]

        result = {
            "motion_type_logits": motion_type_logits,
            "axis_direction": axis_direction,
            "axis_origin": axis_origin,
            "axis_origin_offset": axis_origin_offset,
            "revolute_limit": revolute_limit,
            "prismatic_limit": prismatic_limit,
        }

        if return_intermediate:
            result["axis_direction_raw"] = axis_direction_raw
            result["h_shared"] = h_shared
            result["rev_limit_raw"] = rev_limit_raw
            result["pri_limit_raw"] = pri_limit_raw

        return result

    def get_motion_predictions(
        self,
        outputs: Dict[str, torch.Tensor],
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert raw outputs to final predictions.

        Args:
            outputs: Output dictionary from forward()
            threshold: Score threshold for filtering (not used here, for API consistency)

        Returns:
            Dictionary with:
            - motion_types: [B, Q] predicted motion type indices (0=F, 1=P, 2=R, 3=C)
            - motion_probs: [B, Q, 4] motion type probabilities
            - axis_direction: [B, Q, 3] axis directions
            - axis_origin: [B, Q, 3] axis origins
            - revolute_limit: [B, Q, 2] (center, span) for revolute
            - prismatic_limit: [B, Q, 2] (center, span) for prismatic
        """
        motion_probs = F.softmax(outputs["motion_type_logits"], dim=-1)
        motion_types = motion_probs.argmax(dim=-1)

        return {
            "motion_types": motion_types,
            "motion_probs": motion_probs,
            "axis_direction": outputs["axis_direction"],
            "axis_origin": outputs["axis_origin"],
            "revolute_limit": outputs["revolute_limit"],
            "prismatic_limit": outputs["prismatic_limit"],
        }
