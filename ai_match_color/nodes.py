"""ComfyUI nodes for AI Match Color.

Two nodes are exposed:

* **AI Match Color** — Global LAB / Histogram / Smart matching with live
  adjustments and protections.  No model, no download.
* **AI Match Color (Semantic)** — ADE20K per-group matching.  Downloads the
  segmentation model on first run.

``IMAGE`` tensors in ComfyUI are ``float32`` ``(B, H, W, 3)`` in ``[0, 1]``.  A
batch is treated as video frames: the Reference (single image, first frame of
its batch) is applied consistently to every Source frame, and the Semantic node
can freeze the first Source frame's segmentation for temporal stability.
"""

from __future__ import annotations

import numpy as np

from . import color_core as cc
from . import semantic as sem
from . import model_manager as mm


def _to_np(image_tensor):
    """ComfyUI IMAGE tensor -> list of (H,W,3) float32 numpy frames."""
    arr = image_tensor.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    return [np.clip(arr[i][..., :3], 0.0, 1.0) for i in range(arr.shape[0])]


def _to_tensor(frames):
    import torch

    stacked = np.stack(frames, axis=0).astype(np.float32)
    return torch.from_numpy(stacked)


def _postprocess(source, matched, opts):
    """Apply strength, adjustments and protections to a matched frame."""
    out = matched
    if opts["preserve_luminance"]:
        out = cc.preserve_luminance(source, out)
    out = cc.protect_tones(
        source, out, opts["protect_highlights"], opts["protect_shadows"]
    )
    if opts["preserve_neutral"]:
        out = cc.preserve_neutral(source, out)
    if opts["gamut_compression"] > 0:
        out = cc.gamut_compress(out, opts["gamut_compression"])

    out = cc.blend(source, out, opts["strength"])

    out = cc.apply_adjustments(
        out,
        exposure=opts["exposure"],
        contrast=opts["contrast"],
        temperature=opts["temperature"],
        tint=opts["tint"],
        hue=opts["hue"],
        saturation=opts["saturation"],
    )
    return out


_COMMON_ADJ = {
    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
    "exposure": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
    "contrast": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
    "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
    "tint": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
    "hue": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
    "saturation": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
    "preserve_luminance": ("BOOLEAN", {"default": False}),
    "protect_highlights": ("BOOLEAN", {"default": False}),
    "protect_shadows": ("BOOLEAN", {"default": False}),
    "preserve_neutral": ("BOOLEAN", {"default": False}),
    "gamut_compression": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
}


def _collect_opts(kw):
    return {k: kw[k] for k in _COMMON_ADJ}


# ---------------------------------------------------------------------------
# Node 1: global / histogram / smart
# ---------------------------------------------------------------------------
class AIMatchColor:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "source": ("IMAGE",),
            "reference": ("IMAGE",),
            "method": (["smart", "global_lab", "histogram"], {"default": "smart"}),
            "histogram_space": (["rgb", "lab", "luminance"], {"default": "rgb"}),
        }
        required.update(_COMMON_ADJ)
        return {"required": required}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "AI Match Color"

    def run(self, source, reference, method, histogram_space, **kw):
        opts = _collect_opts(kw)
        src_frames = _to_np(source)
        ref = _to_np(reference)[0]

        out_frames = []
        for frame in src_frames:
            if method == "global_lab":
                matched = cc.global_lab_match(frame, ref)
            elif method == "histogram":
                matched = cc.histogram_match(frame, ref, histogram_space)
            else:
                matched = cc.smart_match(frame, ref)
            out_frames.append(_postprocess(frame, matched, opts))

        return (_to_tensor(out_frames),)


# ---------------------------------------------------------------------------
# Node 2: semantic
# ---------------------------------------------------------------------------
class AIMatchColorSemantic:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "source": ("IMAGE",),
            "reference": ("IMAGE",),
            "model": (list(mm.MODELS.keys()), {"default": "segformer_b2"}),
            "semantic_strength": (
                "FLOAT",
                {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
            ),
            "reuse_first_frame": ("BOOLEAN", {"default": True}),
            "proxy_max_edge": (
                "INT",
                {"default": 1024, "min": 256, "max": 2048, "step": 64},
            ),
        }
        required.update(_COMMON_ADJ)
        # 23 group strength widgets (0..100)
        group_widgets = {}
        for key in sem.GROUPS:
            group_widgets[f"grp_{key}"] = (
                "FLOAT",
                {
                    "default": float(sem.DEFAULT_STRENGTHS[key]),
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                },
            )
        required.update(group_widgets)
        return {"required": required}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "AI Match Color"

    def run(
        self,
        source,
        reference,
        model,
        semantic_strength,
        reuse_first_frame,
        proxy_max_edge,
        **kw,
    ):
        opts = _collect_opts(kw)
        group_strengths = {
            key: float(kw[f"grp_{key}"]) * float(semantic_strength) for key in sem.GROUPS
        }

        src_frames = _to_np(source)
        ref = _to_np(reference)[0]

        out_frames = []
        cached_group_idx = None
        for i, frame in enumerate(src_frames):
            reuse_idx = cached_group_idx if reuse_first_frame else None
            # segmentation depends on resolution; only reuse when it matches
            if reuse_idx is not None and reuse_idx.shape != frame.shape[:2]:
                reuse_idx = None
            matched, group_idx = sem.semantic_match(
                frame,
                ref,
                model,
                group_strengths,
                max_edge=int(proxy_max_edge),
                src_group_idx=reuse_idx,
            )
            if reuse_first_frame and cached_group_idx is None:
                cached_group_idx = group_idx
            out_frames.append(_postprocess(frame, matched, opts))

        return (_to_tensor(out_frames),)


NODE_CLASS_MAPPINGS = {
    "AIMatchColor": AIMatchColor,
    "AIMatchColorSemantic": AIMatchColorSemantic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIMatchColor": "AI Match Color",
    "AIMatchColorSemantic": "AI Match Color (Semantic)",
}
