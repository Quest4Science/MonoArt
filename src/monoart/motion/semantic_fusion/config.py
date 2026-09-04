"""
Semantic Fusion Module Configuration.

Contains all constants and configuration for the semantic fusion module:
- 18 merged part classes
- Motion Prior Table [18, 4]
- Class merge mapping (107 -> 18)
- Class weights for balanced training

Motion Code Handling:
- Compound code "CP" (Continuous + Prismatic, 1.54% of data) mapped to "C"
- All probabilities correctly sum to 1.0
"""

from pathlib import Path
from typing import Dict, List

import torch

# =============================================================================
# Class Configuration (107 -> 18 classes)
# =============================================================================

NUM_PART_CLASSES = 18

# Alphabetically sorted class names
CLASS_NAMES: List[str] = [
    "button",  # 0:  19,056 samples, 100% Prismatic
    "door",  # 1:   2,337 samples,  99% Revolute
    "drawer",  # 2:   1,629 samples, 100% Prismatic
    "handle",  # 3:     216 samples,  96% Revolute
    "head",  # 4:     132 samples,  77% Revolute
    "knob",  # 5:     960 samples,  90% Continuous
    "leg",  # 6:     918 samples,  68% Revolute
    "lid",  # 7:   1,317 samples,  43% Revolute (26%P, 43%R, 31%C)
    "revolute_small",  # 8:     135 samples,  91% Revolute
    "rotation_panel",  # 9:     804 samples,  96% Revolute
    "rotor",  # 10:  1,020 samples,  99% Continuous
    "seat",  # 11:    378 samples,  60% Continuous (CP->C)
    "slider",  # 12:    198 samples, 100% Prismatic
    "stapler",  # 13:     69 samples, 100% Revolute
    "static_body",  # 14:  6,735 samples, 100% Fixed
    "switch_lever",  # 15:    933 samples,  91% Revolute
    "translation_part",  # 16:  1,029 samples, 100% Prismatic
    "wheel",  # 17:  4,287 samples, 100% Continuous
]

CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES)}


# =============================================================================
# Motion Prior Table [18, 4]
# Columns: Fixed (0), Prismatic (1), Revolute (2), Continuous (3)
# All rows sum to 1.0 (CP -> C mapping applied)
# =============================================================================

MOTION_PRIOR_TABLE = torch.tensor(
    [
        # [Fixed, Prismatic, Revolute, Continuous]
        [0.0000, 1.0000, 0.0000, 0.0000],  # button        (100% P)
        [0.0000, 0.0000, 0.9884, 0.0116],  # door          (99% R)
        [0.0000, 1.0000, 0.0000, 0.0000],  # drawer        (100% P)
        [0.0000, 0.0000, 0.9583, 0.0417],  # handle        (96% R)
        [0.0000, 0.0000, 0.7727, 0.2273],  # head          (77% R)
        [0.0000, 0.0000, 0.1000, 0.9000],  # knob          (90% C)
        [0.3203, 0.0000, 0.6797, 0.0000],  # leg           (68% R)
        [0.0000, 0.2620, 0.4260, 0.3121],  # lid           (43% R, 31% C, 26% P)
        [0.0000, 0.0222, 0.9111, 0.0667],  # revolute_small(91% R)
        [0.0000, 0.0000, 0.9590, 0.0410],  # rotation_panel(96% R)
        [0.0000, 0.0000, 0.0059, 0.9941],  # rotor         (99% C)
        [0.0000, 0.0000, 0.3968, 0.6032],  # seat          (60% C, CP->C)
        [0.0000, 1.0000, 0.0000, 0.0000],  # slider        (100% P)
        [0.0000, 0.0000, 1.0000, 0.0000],  # stapler       (100% R)
        [1.0000, 0.0000, 0.0000, 0.0000],  # static_body   (100% F)
        [0.0000, 0.0000, 0.9132, 0.0868],  # switch_lever  (91% R)
        [0.0000, 1.0000, 0.0000, 0.0000],  # translation   (100% P)
        [0.0000, 0.0000, 0.0000, 1.0000],  # wheel         (100% C)
    ],
    dtype=torch.float32,
)


# =============================================================================
# Class Certainty (max probability for each class)
# =============================================================================

CLASS_CERTAINTY = torch.tensor(
    [
        1.0000,  # button
        0.9884,  # door
        1.0000,  # drawer
        0.9583,  # handle
        0.7727,  # head
        0.9000,  # knob
        0.6797,  # leg
        0.4260,  # lid (multi-modal: P/R/C)
        0.9111,  # revolute_small
        0.9590,  # rotation_panel
        0.9941,  # rotor
        0.6032,  # seat (60% C after CP->C mapping)
        1.0000,  # slider
        1.0000,  # stapler
        1.0000,  # static_body
        0.9132,  # switch_lever
        1.0000,  # translation_part
        1.0000,  # wheel
    ],
    dtype=torch.float32,
)


# =============================================================================
# Sample Counts per Class
# =============================================================================

CLASS_COUNTS: Dict[str, int] = {
    "button": 19056,
    "door": 2337,
    "drawer": 1629,
    "handle": 216,
    "head": 132,
    "knob": 960,
    "leg": 918,
    "lid": 1317,
    "revolute_small": 135,
    "rotation_panel": 804,
    "rotor": 1020,
    "seat": 378,
    "slider": 198,
    "stapler": 69,
    "static_body": 6735,
    "switch_lever": 933,
    "translation_part": 1029,
    "wheel": 4287,
}


# =============================================================================
# Merge Mapping (Original 107 -> Merged 18)
# =============================================================================

MERGE_RULES: Dict[str, List[str]] = {
    "static_body": [
        # All *_body
        "furniture_body",
        "trashcan_body",
        "toilet_body",
        "camera_body",
        "dishwasher_body",
        "dispenser_body",
        "bottle_body",
        "coffee_machine_body",
        "usb_body",
        "pen_body",
        "knife_body",
        "refrigerator_body",
        "printer_body",
        "oven_body",
        "microwave_body",
        "safe_body",
        "box_body",
        "lighter_body",
        "bucket_body",
        "pliers_body",
        "washing_machine_body",
        "suitcase_body",
        "clock_body",
        "scissors_body",
        "toaster_body",
        "kettle_body",
        "pot_body",
        "mouse_body",
        "cart_body",
        "glasses_body",
        "lamp_body",
        # All *_base
        "faucet_base",
        "keyboard_base",
        "lamp_base",
        "laptop_base",
        "phone_base",
        "remote_base",
        "stapler_base",
        "display_base",
        # All *_frame
        "fan_frame",
        "globe_frame",
        "window_frame",
        "door_frame",
        "switch_frame",
        "frame",
        # Fixed leg
        "chair_leg",
    ],
    "button": ["button", "key"],
    "drawer": ["drawer"],
    "slider": ["slider"],
    "translation_part": [
        "translation_window",
        "translation_door",
        "translation_lid",
        "translation_screen",
        "translation_tray",
        "translation_blade",
        "translation_bar",
        "translation_handle",
        "pump_lid",
        "pressing_lid",
        "cover_lid",
        "cap",
        "stem",
        "slot",
        "shelf",
        "container",
        "alarm_ring",
    ],
    "door": ["door", "rotation_door"],
    "switch_lever": ["switch", "lever", "toggle_button"],
    "handle": ["handle", "rotation_handle"],
    "rotation_panel": [
        "screen",
        "rotation_screen",
        "rotation_blade",
        "rotation_bar",
        "rotation_window",
        "rotation_button",
        "rotation_container",
    ],
    "seat": ["seat"],
    "stapler": ["stapler_body"],
    "revolute_small": [
        "connector",
        "fastener",
        "lock",
        "portafilter",
        "foot_pad",
        "tilt_leg",
        "nose",
        "rotation_slider",
        "board",
        "fastener_connector",
    ],
    "wheel": ["wheel", "caster", "steering_wheel"],
    "knob": ["knob"],
    "rotor": [
        "rotor",
        "sphere",
        "hand",
        "lens",
        "spout",
        "rotation_tray",
        "rotation_body",
        "ball",
        "circle",
        "usb_rotation",
    ],
    "lid": ["lid", "rotation_lid"],
    "leg": ["leg"],
    "head": ["head"],
}

# Build reverse mapping: original_name -> merged_name
REVERSE_MERGE_MAP: Dict[str, str] = {}
for merged_name, original_names in MERGE_RULES.items():
    for orig in original_names:
        REVERSE_MERGE_MAP[orig] = merged_name


# =============================================================================
# Helper Functions
# =============================================================================


def get_merged_class(original_name: str) -> str:
    """Get merged class name from original part name."""
    return REVERSE_MERGE_MAP.get(original_name, original_name)


def get_merged_class_idx(original_name: str) -> int:
    """Get merged class index from original part name. Returns -1 if unknown."""
    merged_name = get_merged_class(original_name)
    return CLASS_TO_IDX.get(merged_name, -1)


def compute_class_weights(smoothing: float = 0.1) -> torch.Tensor:
    """
    Compute class weights for balanced training (inverse frequency).

    Args:
        smoothing: Smoothing factor to prevent extreme weights

    Returns:
        weights: [NUM_PART_CLASSES] tensor, normalized to mean=1
    """
    total = sum(CLASS_COUNTS.values())
    weights = []
    for name in CLASS_NAMES:
        freq = CLASS_COUNTS[name] / total
        weight = 1.0 / (freq + smoothing)
        weights.append(weight)
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * len(weights)  # Normalize to mean=1
    return weights


# Precomputed class weights
CLASS_WEIGHTS = compute_class_weights()


# =============================================================================
# Default Paths
# =============================================================================

DEFAULT_CLIP_EMBEDDING_PATH = str(
    Path(__file__).resolve().parents[1] / "assets" / "class_axis_embeddings.pt"
)
