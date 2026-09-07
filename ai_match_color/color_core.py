"""Core color-transfer maths for AI Match Color.

Everything here operates on float32 numpy arrays shaped ``(H, W, 3)`` in linear
sRGB display range ``[0, 1]`` (the same range ComfyUI uses for ``IMAGE``
tensors).  No external dependency beyond numpy is required, so these routines run
anywhere ComfyUI runs.

The algorithms mirror the desktop "AI Match Color" app: a Reference image lends
its palette / white-balance / tone to a Source image without changing the
Source's geometry, composition or resolution.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


# ---------------------------------------------------------------------------
# sRGB <-> CIE-Lab (D65) conversions
# ---------------------------------------------------------------------------
def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1 + a)) ** 2.4)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    a = 0.055
    rgb = np.clip(rgb, 0.0, None)
    return np.where(rgb <= 0.0031308, rgb * 12.92, (1 + a) * (rgb ** (1 / 2.4)) - a)


# D65 reference white
_XYZ_FROM_RGB = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_RGB_FROM_XYZ = np.linalg.inv(_XYZ_FROM_RGB)
_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> CIE-Lab. L in [0,100], a/b roughly [-128,127]."""
    lin = _srgb_to_linear(rgb.astype(np.float64))
    xyz = lin @ _XYZ_FROM_RGB.T
    xyz = xyz / _WHITE

    d = 6.0 / 29.0
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4.0 / 29.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIE-Lab -> sRGB [0,1] (clipped)."""
    lab = lab.astype(np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    d = 6.0 / 29.0

    def finv(t):
        return np.where(t > d, t ** 3, 3 * d * d * (t - 4.0 / 29.0))

    xyz = np.stack([finv(fx), finv(fy), finv(fz)], axis=-1) * _WHITE
    lin = xyz @ _RGB_FROM_XYZ.T
    rgb = _linear_to_srgb(lin)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _weighted_stats(values: np.ndarray, weights: np.ndarray | None):
    """Return (mean, std) per channel for ``values`` (..., C)."""
    flat = values.reshape(-1, values.shape[-1])
    if weights is None:
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
    else:
        w = weights.reshape(-1).astype(np.float64)
        wsum = w.sum() + EPS
        mean = (flat * w[:, None]).sum(axis=0) / wsum
        var = (w[:, None] * (flat - mean) ** 2).sum(axis=0) / wsum
        std = np.sqrt(np.maximum(var, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


def blend(source: np.ndarray, matched: np.ndarray, strength: float) -> np.ndarray:
    """Linear blend Source -> matched by ``strength`` in [0,1]."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength >= 1.0:
        return matched
    if strength <= 0.0:
        return source.copy()
    return (source * (1.0 - strength) + matched * strength).astype(np.float32)


# ---------------------------------------------------------------------------
# Global LAB match
# ---------------------------------------------------------------------------
def global_lab_match(
    source: np.ndarray,
    reference: np.ndarray,
    src_mask: np.ndarray | None = None,
    ref_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Match mean/std of Source to Reference in CIE-Lab (Reinhard transfer)."""
    s_lab = rgb_to_lab(source)
    r_lab = rgb_to_lab(reference)

    s_mean, s_std = _weighted_stats(s_lab, src_mask)
    r_mean, r_std = _weighted_stats(r_lab, ref_mask)

    scale = r_std / (s_std + EPS)
    # keep the transfer gentle to avoid blowing out flat images
    scale = np.clip(scale, 0.2, 5.0)
    out = (s_lab - s_mean) * scale + r_mean
    out[..., 0] = np.clip(out[..., 0], 0.0, 100.0)
    return lab_to_rgb(out)


# ---------------------------------------------------------------------------
# Histogram match
# ---------------------------------------------------------------------------
def _match_channel(src: np.ndarray, ref: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Monotone quantile histogram match of one channel onto ref's distribution."""
    n = 512
    qs = np.linspace(0.0, 1.0, n)
    src_q = np.quantile(src, qs)
    ref_q = np.quantile(ref, qs)
    # light smoothing to reduce banding
    k = 5
    kernel = np.ones(k) / k
    ref_q = np.convolve(ref_q, kernel, mode="same")
    ref_q[0], ref_q[-1] = ref_q[0], ref_q[-1]
    mapped = np.interp(src.reshape(-1), src_q, ref_q, left=ref_q[0], right=ref_q[-1])
    return np.clip(mapped.reshape(src.shape), lo, hi)


def histogram_match(
    source: np.ndarray,
    reference: np.ndarray,
    color_space: str = "rgb",
    src_mask: np.ndarray | None = None,
    ref_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Histogram match in 'rgb', 'lab' or 'luminance'."""
    color_space = color_space.lower()

    def sel(img, mask):
        if mask is None:
            return img.reshape(-1, img.shape[-1])
        m = mask.reshape(-1) > 0.5
        picked = img.reshape(-1, img.shape[-1])[m]
        return picked if picked.size else img.reshape(-1, img.shape[-1])

    if color_space == "lab":
        s = rgb_to_lab(source)
        r = rgb_to_lab(reference)
        s_sel, r_sel = sel(s, src_mask), sel(r, ref_mask)
        out = s.copy()
        ranges = [(0.0, 100.0), (-128.0, 127.0), (-128.0, 127.0)]
        for c in range(3):
            lo, hi = ranges[c]
            src_q = np.quantile(s_sel[:, c], np.linspace(0, 1, 512))
            ref_q = np.quantile(r_sel[:, c], np.linspace(0, 1, 512))
            out[..., c] = np.clip(
                np.interp(s[..., c], src_q, ref_q).reshape(s[..., c].shape), lo, hi
            )
        return lab_to_rgb(out)

    if color_space == "luminance":
        s = rgb_to_lab(source)
        r = rgb_to_lab(reference)
        s_sel, r_sel = sel(s, src_mask), sel(r, ref_mask)
        src_q = np.quantile(s_sel[:, 0], np.linspace(0, 1, 512))
        ref_q = np.quantile(r_sel[:, 0], np.linspace(0, 1, 512))
        out = s.copy()
        out[..., 0] = np.clip(
            np.interp(s[..., 0], src_q, ref_q).reshape(s[..., 0].shape), 0.0, 100.0
        )
        return lab_to_rgb(out)

    # rgb
    s_sel, r_sel = sel(source, src_mask), sel(reference, ref_mask)
    out = source.copy()
    for c in range(3):
        src_q = np.quantile(s_sel[:, c], np.linspace(0, 1, 512))
        ref_q = np.quantile(r_sel[:, c], np.linspace(0, 1, 512))
        out[..., c] = np.clip(
            np.interp(source[..., c], src_q, ref_q).reshape(source[..., c].shape),
            0.0,
            1.0,
        )
    return out


# ---------------------------------------------------------------------------
# Smart match — LAB transfer + a gentle luminance histogram + white balance
# ---------------------------------------------------------------------------
def smart_match(
    source: np.ndarray,
    reference: np.ndarray,
    src_mask: np.ndarray | None = None,
    ref_mask: np.ndarray | None = None,
) -> np.ndarray:
    """A balanced combination aimed at natural results."""
    lab = global_lab_match(source, reference, src_mask, ref_mask)
    # nudge the luminance distribution towards the reference
    lum = histogram_match(lab, reference, "luminance", None, ref_mask)
    out = 0.5 * lab + 0.5 * lum
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Live adjustments (LAB based where it matters)
# ---------------------------------------------------------------------------
def apply_adjustments(
    rgb: np.ndarray,
    exposure: float = 0.0,
    contrast: float = 0.0,
    temperature: float = 0.0,
    tint: float = 0.0,
    hue: float = 0.0,
    saturation: float = 0.0,
) -> np.ndarray:
    """Apply exposure/contrast/temperature/tint/hue/saturation.

    Ranges: exposure/-contrast in stops-ish [-1,1]; temperature/tint [-1,1];
    hue in degrees [-180,180]; saturation [-1,1].
    """
    out = rgb.astype(np.float32).copy()

    # exposure in linear light
    if abs(exposure) > EPS:
        lin = _srgb_to_linear(out.astype(np.float64))
        lin *= 2.0 ** (2.0 * exposure)
        out = _linear_to_srgb(lin).astype(np.float32)

    # temperature (warm/cool) and tint (green/magenta)
    if abs(temperature) > EPS or abs(tint) > EPS:
        out[..., 0] = np.clip(out[..., 0] + 0.10 * temperature, 0, 1)
        out[..., 2] = np.clip(out[..., 2] - 0.10 * temperature, 0, 1)
        out[..., 1] = np.clip(out[..., 1] - 0.10 * tint, 0, 1)

    need_lab = abs(contrast) > EPS or abs(hue) > EPS or abs(saturation) > EPS
    if need_lab:
        lab = rgb_to_lab(out)
        if abs(contrast) > EPS:
            lab[..., 0] = np.clip(
                (lab[..., 0] - 50.0) * (1.0 + contrast) + 50.0, 0.0, 100.0
            )
        if abs(saturation) > EPS:
            lab[..., 1] *= 1.0 + saturation
            lab[..., 2] *= 1.0 + saturation
        if abs(hue) > EPS:
            theta = np.deg2rad(hue)
            ct, st = np.cos(theta), np.sin(theta)
            a = lab[..., 1].copy()
            b = lab[..., 2].copy()
            lab[..., 1] = a * ct - b * st
            lab[..., 2] = a * st + b * ct
        out = lab_to_rgb(lab)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Protections & gamut compression
# ---------------------------------------------------------------------------
def preserve_luminance(source: np.ndarray, matched: np.ndarray) -> np.ndarray:
    """Keep Source L, take matched a/b."""
    s = rgb_to_lab(source)
    m = rgb_to_lab(matched)
    m[..., 0] = s[..., 0]
    return lab_to_rgb(m)


def protect_tones(
    source: np.ndarray,
    matched: np.ndarray,
    highlights: bool,
    shadows: bool,
) -> np.ndarray:
    """Reduce the match effect in highlight / shadow regions of the Source."""
    if not highlights and not shadows:
        return matched
    lum = rgb_to_lab(source)[..., 0] / 100.0
    keep = np.zeros_like(lum)
    if highlights:
        keep = np.maximum(keep, np.clip((lum - 0.75) / 0.25, 0, 1))
    if shadows:
        keep = np.maximum(keep, np.clip((0.25 - lum) / 0.25, 0, 1))
    keep = keep[..., None] * 0.85
    return (matched * (1 - keep) + source * keep).astype(np.float32)


def preserve_neutral(source: np.ndarray, matched: np.ndarray) -> np.ndarray:
    """Keep near-neutral Source pixels from picking up a colour cast."""
    lab = rgb_to_lab(source)
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    keep = np.clip((12.0 - chroma) / 12.0, 0.0, 1.0)[..., None] * 0.8
    return (matched * (1 - keep) + source * keep).astype(np.float32)


def gamut_compress(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Soft compression of out-of-range chroma via LAB scaling."""
    if amount <= EPS:
        return np.clip(rgb, 0, 1)
    lab = rgb_to_lab(rgb)
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2) + EPS
    limit = 100.0 * (1.0 - 0.5 * amount)
    scale = np.where(chroma > limit, limit / chroma, 1.0)
    lab[..., 1] *= scale
    lab[..., 2] *= scale
    return lab_to_rgb(lab)
