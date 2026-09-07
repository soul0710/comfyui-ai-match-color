"""ComfyUI custom node: AI Match Color.

Local color transfer / color matching — maps the palette, white balance and tone
of a Reference image onto a Source image (or a batch of frames), without changing
the Source's geometry, composition or resolution.

The Semantic Match node downloads its ADE20K segmentation model on first run into
``ComfyUI/models/ai_match_color``.
"""

try:  # normal case: loaded by ComfyUI as a package
    from .ai_match_color import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # standalone / test import
    from ai_match_color import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
