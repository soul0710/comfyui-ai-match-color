"""Semantic Match: ADE20K segmentation + per-group local colour transfer.

Source and Reference are segmented *independently*.  Each of the 23 groups is
matched locally (LAB Reinhard transfer from the Reference region onto the Source
region) at its own strength.  Pixels whose group has no matching Reference region
fall back to a global Smart Match, exactly like the desktop app.

The segmentation model is loaded lazily and cached; downloading happens on first
use via :mod:`model_manager`.
"""

from __future__ import annotations

import numpy as np

from . import color_core as cc
from . import model_manager as mm

# ---------------------------------------------------------------------------
# 23 groups (order matters — used by the ComfyUI node widgets)
# ---------------------------------------------------------------------------
GROUPS = [
    "sky",
    "water",
    "foliage_trees",
    "flowers",
    "grass_moss",
    "building_architecture",
    "wall_ceiling",
    "window_door",
    "ground_floor_path",
    "wood",
    "stone_rock",
    "furniture",
    "fabric_curtain_rug",
    "glass_reflective",
    "fire_flame",
    "candle_artificial_light",
    "seasonal_decorations",
    "small_objects_tableware",
    "other",
    "person",
    "skin",
    "hair",
    "clothing",
]

GROUP_LABELS = {
    "sky": "Sky",
    "water": "Water",
    "foliage_trees": "Foliage / Trees",
    "flowers": "Flowers",
    "grass_moss": "Grass / Moss",
    "building_architecture": "Building / Architecture",
    "wall_ceiling": "Wall / Ceiling",
    "window_door": "Window / Door",
    "ground_floor_path": "Ground / Floor / Path",
    "wood": "Wood",
    "stone_rock": "Stone / Rock",
    "furniture": "Furniture",
    "fabric_curtain_rug": "Fabric / Curtain / Rug",
    "glass_reflective": "Glass / Reflective Surface",
    "fire_flame": "Fire / Flame",
    "candle_artificial_light": "Candle / Artificial Light",
    "seasonal_decorations": "Seasonal Decorations",
    "small_objects_tableware": "Small Objects / Tableware",
    "other": "Other",
    "person": "Person",
    "skin": "Skin",
    "hair": "Hair",
    "clothing": "Clothing",
}

# Default per-group strengths (0..100) from the app spec.
DEFAULT_STRENGTHS = {
    "sky": 85,
    "water": 85,
    "foliage_trees": 80,
    "flowers": 70,
    "grass_moss": 80,
    "building_architecture": 70,
    "wall_ceiling": 35,
    "window_door": 35,
    "ground_floor_path": 65,
    "wood": 70,
    "stone_rock": 70,
    "furniture": 60,
    "fabric_curtain_rug": 60,
    "glass_reflective": 30,
    "fire_flame": 10,
    "candle_artificial_light": 20,
    "seasonal_decorations": 50,
    "small_objects_tableware": 50,
    "other": 40,
    "person": 40,
    "skin": 25,
    "hair": 35,
    "clothing": 55,
}

# ADE20K 150 class index -> group key.  Classes not listed fall into "other".
# (ADE20K has no dedicated skin / hair / wood-material classes, so those groups
# simply receive no pixels — matching the app's documented behaviour.)
ADE20K_TO_GROUP = {
    2: "sky",
    21: "water", 26: "water", 60: "water", 109: "water", 113: "water", 128: "water",
    4: "foliage_trees", 17: "foliage_trees", 72: "foliage_trees",
    66: "flowers",
    9: "grass_moss", 29: "grass_moss",
    1: "building_architecture", 25: "building_architecture", 48: "building_architecture",
    61: "building_architecture", 79: "building_architecture", 84: "building_architecture",
    0: "wall_ceiling", 5: "wall_ceiling", 42: "wall_ceiling",
    8: "window_door", 14: "window_door", 58: "window_door",
    3: "ground_floor_path", 6: "ground_floor_path", 11: "ground_floor_path",
    13: "ground_floor_path", 46: "ground_floor_path", 52: "ground_floor_path",
    53: "ground_floor_path", 59: "ground_floor_path", 91: "ground_floor_path",
    94: "ground_floor_path", 121: "ground_floor_path",
    16: "stone_rock", 34: "stone_rock", 68: "stone_rock",
    7: "furniture", 10: "furniture", 15: "furniture", 19: "furniture", 23: "furniture",
    24: "furniture", 30: "furniture", 31: "furniture", 33: "furniture", 35: "furniture",
    40: "furniture", 44: "furniture", 45: "furniture", 55: "furniture", 62: "furniture",
    64: "furniture", 69: "furniture", 70: "furniture", 73: "furniture", 75: "furniture",
    97: "furniture", 99: "furniture", 110: "furniture",
    18: "fabric_curtain_rug", 28: "fabric_curtain_rug", 39: "fabric_curtain_rug",
    57: "fabric_curtain_rug", 63: "fabric_curtain_rug", 81: "fabric_curtain_rug",
    115: "fabric_curtain_rug", 131: "fabric_curtain_rug",
    27: "glass_reflective", 147: "glass_reflective",
    49: "fire_flame",
    36: "candle_artificial_light", 82: "candle_artificial_light",
    85: "candle_artificial_light", 87: "candle_artificial_light",
    134: "candle_artificial_light", 136: "candle_artificial_light",
    132: "seasonal_decorations", 149: "seasonal_decorations",
    41: "small_objects_tableware", 67: "small_objects_tableware",
    98: "small_objects_tableware", 112: "small_objects_tableware",
    119: "small_objects_tableware", 120: "small_objects_tableware",
    125: "small_objects_tableware", 135: "small_objects_tableware",
    137: "small_objects_tableware", 138: "small_objects_tableware",
    142: "small_objects_tableware", 148: "small_objects_tableware",
    12: "person",
    92: "clothing",
}


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------
_SEG_CACHE = {}  # model_key -> (model, processor, device)


def _load_segmenter(model_key: str):
    if model_key in _SEG_CACHE:
        return _SEG_CACHE[model_key]

    path = mm.ensure_model(model_key)

    import torch  # noqa: F401  (ComfyUI always ships torch)

    device = "cuda" if _cuda_available() else "cpu"

    if model_key == "segformer_b2":
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        # use_fast=False avoids the torchvision-only "fast" processor path
        # (keeps the dependency footprint to torch + transformers only).
        try:
            processor = AutoImageProcessor.from_pretrained(path, use_fast=False)
        except TypeError:  # very old transformers without use_fast kwarg
            processor = AutoImageProcessor.from_pretrained(path)
        model = SegformerForSemanticSegmentation.from_pretrained(path)
    elif model_key == "oneformer_swin_large":
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

        processor = OneFormerProcessor.from_pretrained(path)
        model = OneFormerForUniversalSegmentation.from_pretrained(path)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported semantic model '{model_key}'")

    model.to(device).eval()
    _SEG_CACHE[model_key] = (model, processor, device)
    return _SEG_CACHE[model_key]


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def segment(rgb: np.ndarray, model_key: str, max_edge: int = 1024) -> np.ndarray:
    """Return an ``(H, W)`` int array of ADE20K class ids for ``rgb`` in [0,1]."""
    import torch
    from PIL import Image

    model, processor, device = _load_segmenter(model_key)

    h, w = rgb.shape[:2]
    scale = min(1.0, max_edge / max(h, w))
    pil = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))

    with torch.inference_mode():
        if model_key == "oneformer_swin_large":
            inputs = processor(images=pil, task_inputs=["semantic"], return_tensors="pt")
        else:
            inputs = processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        if model_key == "oneformer_swin_large":
            seg = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(h, w)]
            )[0]
        else:
            seg = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(h, w)]
            )[0]
    return seg.detach().cpu().numpy().astype(np.int32)


def group_map(seg: np.ndarray) -> np.ndarray:
    """Map an ADE20K class-id map to group indices (index into GROUPS)."""
    other_idx = GROUPS.index("other")
    out = np.full(seg.shape, other_idx, dtype=np.int32)
    for cls_id, group_key in ADE20K_TO_GROUP.items():
        out[seg == cls_id] = GROUPS.index(group_key)
    return out


# ---------------------------------------------------------------------------
# semantic colour transfer
# ---------------------------------------------------------------------------
def semantic_match(
    source: np.ndarray,
    reference: np.ndarray,
    model_key: str,
    group_strengths: dict,
    max_edge: int = 1024,
    src_group_idx: np.ndarray | None = None,
    feather: int = 2,
):
    """Apply per-group local colour transfer.

    ``src_group_idx`` may be supplied to reuse a previously computed Source
    segmentation (first-frame reuse for video / image batches).  Returns
    ``(matched_rgb, src_group_idx)`` so the caller can cache the map.
    """
    if src_group_idx is None:
        src_seg = segment(source, model_key, max_edge)
        src_group_idx = group_map(src_seg)

    ref_seg = segment(reference, model_key, max_edge)
    ref_group_idx = group_map(ref_seg)

    # base: global smart match for anything without a good local match
    base = cc.smart_match(source, reference)
    out = base.copy()

    min_pixels = max(64, int(0.0005 * source.shape[0] * source.shape[1]))

    for gi, key in enumerate(GROUPS):
        strength = float(group_strengths.get(key, 0)) / 100.0
        if strength <= 0.0:
            continue
        s_mask = src_group_idx == gi
        r_mask = ref_group_idx == gi
        if s_mask.sum() < min_pixels or r_mask.sum() < min_pixels:
            # no matching reference region -> keep the smart-match base
            continue

        matched = cc.global_lab_match(
            source, reference, s_mask.astype(np.float32), r_mask.astype(np.float32)
        )
        local = cc.blend(source, matched, strength)

        m = _feather_mask(s_mask, feather)[..., None]
        out = out * (1.0 - m) + local * m

    return np.clip(out, 0.0, 1.0).astype(np.float32), src_group_idx


def _box_blur_1d(a: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Uniform moving average of width ``2*radius+1`` along ``axis`` (edge pad)."""
    k = 2 * radius + 1
    n = a.shape[axis]
    padded = np.pad(a, [(radius, radius) if ax == axis else (0, 0)
                        for ax in range(a.ndim)], mode="edge")
    csum = np.cumsum(padded, axis=axis)
    zero_shape = list(a.shape)
    zero_shape[axis] = 1
    csum = np.concatenate([np.zeros(zero_shape, dtype=csum.dtype), csum], axis=axis)
    hi = np.take(csum, np.arange(k, k + n), axis=axis)
    lo = np.take(csum, np.arange(0, n), axis=axis)
    return (hi - lo) / k


def _feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Soft-edge a boolean mask with a cheap separable box blur."""
    m = mask.astype(np.float32)
    if radius <= 0:
        return m
    m = _box_blur_1d(m, radius, axis=1)
    m = _box_blur_1d(m, radius, axis=0)
    return np.clip(m, 0.0, 1.0)
