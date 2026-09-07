"""Fast, dependency-light tests for the colour maths and semantic mapping.

Run from the repo root:  python -m pytest tests/ -q  (needs numpy).
The segmentation model itself is not downloaded here; only the pure-numpy
routines and the ADE20K -> group mapping are exercised.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai_match_color import color_core as cc  # noqa: E402
from ai_match_color import semantic as sem  # noqa: E402


def test_lab_roundtrip():
    rng = np.random.default_rng(0)
    img = rng.random((32, 48, 3)).astype(np.float32)
    back = cc.lab_to_rgb(cc.rgb_to_lab(img))
    assert np.abs(img - back).max() < 1e-3


def test_white_point():
    w = cc.rgb_to_lab(np.ones((1, 1, 3), np.float32))[0, 0]
    assert abs(w[0] - 100) < 0.5 and abs(w[1]) < 1 and abs(w[2]) < 1


def test_global_lab_moves_toward_reference():
    rng = np.random.default_rng(1)
    src = (rng.random((64, 64, 3)) * 0.3).astype(np.float32)
    ref = np.clip(rng.random((64, 64, 3)) * 0.6 + 0.3, 0, 1).astype(np.float32)
    m = cc.global_lab_match(src, ref)
    assert m.mean() > src.mean()
    assert m.min() >= 0 and m.max() <= 1


def test_histogram_spaces():
    rng = np.random.default_rng(2)
    src = (rng.random((48, 48, 3)) * 0.4).astype(np.float32)
    ref = np.clip(rng.random((48, 48, 3)) * 0.7 + 0.2, 0, 1).astype(np.float32)
    for cs in ("rgb", "lab", "luminance"):
        h = cc.histogram_match(src, ref, cs)
        assert h.shape == src.shape and np.isfinite(h).all()


def test_blend_endpoints():
    rng = np.random.default_rng(3)
    src = rng.random((16, 16, 3)).astype(np.float32)
    m = rng.random((16, 16, 3)).astype(np.float32)
    assert np.allclose(cc.blend(src, m, 0.0), src)
    assert np.allclose(cc.blend(src, m, 1.0), m)
    assert np.allclose(cc.blend(src, m, 0.5), 0.5 * src + 0.5 * m, atol=1e-6)


def test_adjustments_identity_and_direction():
    rng = np.random.default_rng(4)
    src = rng.random((16, 16, 3)).astype(np.float32)
    assert np.allclose(cc.apply_adjustments(src, 0, 0, 0, 0, 0, 0), src, atol=1e-4)
    assert cc.apply_adjustments(src, exposure=0.5).mean() > src.mean()


def test_feather_mask():
    mask = np.zeros((20, 20), bool)
    mask[:, :10] = True
    f = sem._feather_mask(mask, 3)
    assert f.shape == (20, 20) and 0 <= f.min() and f.max() <= 1
    assert f[0, 0] > 0.9 and f[0, -1] < 0.1


def test_group_mapping():
    seg = np.zeros((4, 4), np.int32)
    seg[0, 0] = 2   # sky
    seg[1, 1] = 12  # person
    seg[2, 2] = 999  # unknown
    gm = sem.group_map(seg)
    assert gm[0, 0] == sem.GROUPS.index("sky")
    assert gm[1, 1] == sem.GROUPS.index("person")
    assert gm[2, 2] == sem.GROUPS.index("other")
    assert len(sem.GROUPS) == 23 == len(sem.DEFAULT_STRENGTHS)
