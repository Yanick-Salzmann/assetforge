from __future__ import annotations

import numpy as np
import pytest

from assets import flora, silhouette
from assets.foliage_atlas import Rect

OPAQUE = np.ones((4, 4), dtype=np.float32)
FULL = Rect(0.0, 0.0, 1.0, 1.0)


def _card(width: float = 1.0) -> flora.MeshData:
    builder = flora.Builder()
    builder.card((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 1.0, width, FULL, outward=(0.0, -1.0, 0.0))
    return builder.mesh


def test_an_opaque_card_covers_its_square():
    mesh = _card()
    frame = silhouette.Frame(centre=(0.0, 0.0, 0.5), radius=1.0)
    mask = silhouette.coverage(mesh, OPAQUE, (0.0, -1.0, 0.0), 64, frame)
    assert mask.mean() == pytest.approx(0.25, abs=0.02)
    assert not silhouette.coverage(mesh, np.zeros((4, 4), dtype=np.float32), (0.0, -1.0, 0.0), 64, frame).any()


def test_identical_masks_do_not_pop_and_a_wider_card_does():
    frame = silhouette.Frame(centre=(0.0, 0.0, 0.5), radius=1.0)
    base = silhouette.coverage(_card(), OPAQUE, (0.0, -1.0, 0.0), 64, frame)
    wide = silhouette.coverage(_card(1.5), OPAQUE, (0.0, -1.0, 0.0), 64, frame)
    assert silhouette.pop(base, base) == 0.0
    assert silhouette.pop(base, wide) == pytest.approx(0.5, abs=0.05)
    assert silhouette.blurred_pop(base, wide, 8) == pytest.approx(0.5, abs=0.05)
