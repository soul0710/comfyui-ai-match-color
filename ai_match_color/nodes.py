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
    """ComfyUI IMAGE tensor -> (rgb_frames, alpha_frames).

    ``rgb_frames`` is a list of ``(H,W,3)`` float32 arrays; ``alpha_frames`` is a
    matching list of ``(H,W,1)`` arrays, or ``None`` when the input has no alpha.
    """
    arr = image_tensor.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    rgb = [np.clip(arr[i][..., :3], 0.0, 1.0) for i in range(arr.shape[0])]
    alpha = None
    if arr.shape[-1] >= 4:
        alpha = [np.clip(arr[i][..., 3:4], 0.0, 1.0) for i in range(arr.shape[0])]
    return rgb, alpha


def _to_tensor(frames, alpha=None):
    import torch

    if alpha is not None:
        frames = [np.concatenate([f, a], axis=-1) for f, a in zip(frames, alpha)]
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
        src_frames, src_alpha = _to_np(source)
        ref = _to_np(reference)[0][0]

        out_frames = []
        for frame in src_frames:
            if method == "global_lab":
                matched = cc.global_lab_match(frame, ref)
            elif method == "histogram":
                matched = cc.histogram_match(frame, ref, histogram_space)
            else:
                matched = cc.smart_match(frame, ref)
            out_frames.append(_postprocess(frame, matched, opts))

        return (_to_tensor(out_frames, src_alpha),)


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

        src_frames, src_alpha = _to_np(source)
        ref = _to_np(reference)[0][0]

        # Fallback chain, mirroring the desktop app:
        #   chosen model -> SegFormer-B2 -> Smart Match (never crashes).
        model_chain = [model]
        if model != "segformer_b2":
            model_chain.append("segformer_b2")

        out_frames = []
        cached_src_idx = None
        cached_ref_idx = None
        active_model = None  # resolved lazily on the first frame

        for frame in src_frames:
            if active_model is None:
                active_model = self._resolve_model(
                    frame, ref, model_chain, group_strengths, int(proxy_max_edge)
                )

            if active_model == "__smart__":
                matched = cc.smart_match(frame, ref)
            else:
                # segmentation depends on resolution; drop stale caches
                s_idx = cached_src_idx if reuse_first_frame else None
                if s_idx is not None and s_idx.shape != frame.shape[:2]:
                    s_idx = None
                try:
                    matched, s_out, r_out = sem.semantic_match(
                        frame, ref, active_model, group_strengths,
                        max_edge=int(proxy_max_edge),
                        src_group_idx=s_idx, ref_group_idx=cached_ref_idx,
                    )
                    if reuse_first_frame:
                        if cached_src_idx is None:
                            cached_src_idx = s_out
                        if cached_ref_idx is None:
                            cached_ref_idx = r_out
                except Exception as exc:  # inference/OOM mid-batch -> degrade
                    print(f"[AI Match Color] Semantic run failed on a frame "
                          f"({active_model}): {exc}. Falling back to Smart Match.")
                    active_model = "__smart__"
                    matched = cc.smart_match(frame, ref)

            out_frames.append(_postprocess(frame, matched, opts))

        return (_to_tensor(out_frames, src_alpha),)

    @staticmethod
    def _resolve_model(frame, ref, model_chain, group_strengths, max_edge):
        """Ensure a model can be downloaded/loaded; return the first that works,
        or the ``"__smart__"`` sentinel if all semantic models fail.

        Only model *loading* is probed here (the download/load failure point).
        Per-frame inference errors (e.g. OOM) are handled by the caller's loop.
        """
        for mkey in model_chain:
            try:
                sem._load_segmenter(mkey)  # triggers first-run download + load
                print(f"[AI Match Color] Semantic model in use: {mkey}")
                return mkey
            except Exception as exc:
                print(f"[AI Match Color] Model '{mkey}' unavailable: {exc}")
        print("[AI Match Color] All semantic models failed; using Smart Match.")
        return "__smart__"


NODE_CLASS_MAPPINGS = {
    "AIMatchColor": AIMatchColor,
    "AIMatchColorSemantic": AIMatchColorSemantic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIMatchColor": "AI Match Color",
    "AIMatchColorSemantic": "AI Match Color (Semantic)",
}
