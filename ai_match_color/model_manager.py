"""Model discovery and first-run download for the Semantic Match backend.

The desktop app ships three semantic backends.  For a portable ComfyUI node the
default is **SegFormer-B2 ADE20K** (small, pure PyTorch/Transformers, works on
CPU or CUDA).  The heavier OneFormer Swin-Large model is offered as an optional
"high quality" choice for users who have it / want to download it.

Models are stored under ``ComfyUI/models/ai_match_color/<key>`` so they survive
across updates of this custom-node folder.  The first time a semantic node runs
it downloads the requested model with ``huggingface_hub`` and caches it there;
subsequent runs load straight from disk.
"""

from __future__ import annotations

import os
import threading

# Registry: key -> (HuggingFace repo id, human name)
MODELS = {
    "segformer_b2": (
        "nvidia/segformer-b2-finetuned-ade-512-512",
        "Fast/Compatible — SegFormer-B2 ADE20K",
    ),
    "oneformer_swin_large": (
        "shi-labs/oneformer_ade20k_swin_large",
        "High Quality — OneFormer ADE20K Swin-Large",
    ),
}

_lock = threading.Lock()


def _models_root() -> str:
    """Return ``ComfyUI/models/ai_match_color`` (falls back to a local dir)."""
    try:
        import folder_paths  # provided by ComfyUI at runtime

        root = os.path.join(folder_paths.models_dir, "ai_match_color")
    except Exception:
        root = os.path.join(os.path.dirname(__file__), "..", "models")
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    return root


def local_path(model_key: str) -> str:
    return os.path.join(_models_root(), model_key)


def is_downloaded(model_key: str) -> bool:
    path = local_path(model_key)
    if not os.path.isdir(path):
        return False
    # a valid HF snapshot has a config plus at least one weights file
    files = os.listdir(path)
    has_cfg = any(f == "config.json" for f in files)
    has_weights = any(
        f.endswith((".bin", ".safetensors")) for f in files
    )
    return has_cfg and has_weights


def ensure_model(model_key: str) -> str:
    """Download the model on first use, then return its local directory.

    Raises a clear error if the model key is unknown or ``huggingface_hub`` is
    unavailable.
    """
    if model_key not in MODELS:
        raise ValueError(
            f"Unknown semantic model '{model_key}'. "
            f"Valid keys: {', '.join(MODELS)}"
        )

    dest = local_path(model_key)
    if is_downloaded(model_key):
        return dest

    with _lock:
        if is_downloaded(model_key):
            return dest

        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "huggingface_hub is required to download the Semantic Match "
                "model. Install it with `pip install huggingface_hub`, or pick "
                "a non-semantic method (global_lab / histogram / smart)."
            ) from exc

        repo_id, name = MODELS[model_key]
        print(f"[AI Match Color] Downloading semantic model: {name} ({repo_id})")
        print(f"[AI Match Color] Destination: {dest}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=dest,
            local_dir_use_symlinks=False,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.bin",
                "*.safetensors",
                "*.model",
                "preprocessor_config.json",
            ],
        )
        print(f"[AI Match Color] Download complete: {name}")

    if not is_downloaded(model_key):
        raise RuntimeError(
            f"Model '{model_key}' did not download correctly to {dest}. "
            "Check your network connection and try again."
        )
    return dest
