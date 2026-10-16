from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from assets import blender as blender_discovery
from assets import cloth_shapes
from assets import validate
from terrain import config

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "assets" / "blender_scripts" / "cloth_selftest.py"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _layouts() -> dict[str, cloth_shapes.ClothLayout]:
    return {kind: builder() for kind, builder in cloth_shapes.BUILDERS.items()}


def _signed_volume(panel: cloth_shapes.Panel) -> float:
    volume = 0.0
    for face in panel.faces:
        a = panel.vertices[face[0]]
        for index in range(1, len(face) - 1):
            b = panel.vertices[face[index]]
            c = panel.vertices[face[index + 1]]
            volume += cloth_shapes.dot(a, cloth_shapes.cross(b, c)) / 6.0
    return volume


def _uv_area(corners) -> float:
    area = 0.0
    for index, (u0, v0) in enumerate(corners):
        u1, v1 = corners[(index + 1) % len(corners)]
        area += u0 * v1 - u1 * v0
    return 0.5 * area


def _face_area(panel: cloth_shapes.Panel, face) -> float:
    a = panel.vertices[face[0]]
    total = 0.0
    for index in range(1, len(face) - 1):
        b = panel.vertices[face[index]]
        c = panel.vertices[face[index + 1]]
        n = cloth_shapes.cross(cloth_shapes.sub(b, a), cloth_shapes.sub(c, a))
        total += 0.5 * math.sqrt(cloth_shapes.dot(n, n))
    return total


@pytest.mark.parametrize("kind", sorted(cloth_shapes.BUILDERS))
def test_panel_is_closed_consistently_wound_and_outward(kind):
    panel = _layouts()[kind].panel
    directed = Counter()
    for face in panel.faces:
        for index, a in enumerate(face):
            directed[(a, face[(index + 1) % len(face)])] += 1
    assert all(count == 1 for count in directed.values())
    assert all((b, a) in directed for a, b in directed)
    assert _signed_volume(panel) > 0
    referenced = {index for face in panel.faces for index in face}
    assert referenced == set(range(len(panel.vertices)))


@pytest.mark.parametrize("kind", sorted(cloth_shapes.BUILDERS))
def test_weights_sum_to_one_and_attached_edge_rides_the_first_bone(kind):
    panel = _layouts()[kind].panel
    origin = panel.bone_heads[0]
    for vertex, weights in zip(panel.vertices, panel.weights):
        assert math.isclose(sum(weight for _, weight in weights), 1.0)
        assert all(weight > 0 for _, weight in weights)
        if abs(cloth_shapes.dot(cloth_shapes.sub(vertex, origin), panel.free)) < 1e-9:
            assert weights == [(0, 1.0)]
    for tail, head in zip(panel.bone_tails, panel.bone_heads[1:]):
        assert max(abs(a - b) for a, b in zip(tail, head)) < 1e-9


def test_chain_weights_blend_between_bone_centres():
    assert cloth_shapes.chain_weights(0.0, 4, 1.0) == [(0, 1.0)]
    assert cloth_shapes.chain_weights(1.0, 4, 1.0) == [(3, 1.0)]
    lower, upper = cloth_shapes.chain_weights(0.25, 4, 1.0)
    assert lower[0] == 0 and upper[0] == 1
    assert math.isclose(lower[1], 0.5) and math.isclose(upper[1], 0.5)


@pytest.mark.parametrize("kind", sorted(cloth_shapes.BUILDERS))
def test_front_and_back_islands_are_disjoint_at_uniform_density_and_the_rim_is_collapsed(kind):
    panel = _layouts()[kind].panel
    densities = []
    boxes = {"front": [], "back": []}
    for face, uvs in zip(panel.faces, panel.uvs):
        uv_area = abs(_uv_area(uvs))
        normal_dot = cloth_shapes.dot(
            cloth_shapes.cross(
                cloth_shapes.sub(panel.vertices[face[1]], panel.vertices[face[0]]),
                cloth_shapes.sub(panel.vertices[face[2]], panel.vertices[face[0]]),
            ),
            panel.normal,
        )
        if abs(normal_dot) < 1e-12:
            assert uv_area < 1e-12
            continue
        densities.append(math.sqrt(uv_area / _face_area(panel, face)))
        boxes["front" if normal_dot > 0 else "back"].append(uvs)
        assert all(0.0 <= value <= 1.0 for uv in uvs for value in uv)
    assert max(densities) / min(densities) < 1.0 + 1e-6
    front_top = max(v for uvs in boxes["front"] for _, v in uvs)
    back_bottom = min(v for uvs in boxes["back"] for _, v in uvs)
    assert back_bottom > front_top


def test_cloth_swings_through_its_thickness_and_faces_the_right_way():
    layouts = _layouts()
    assert layouts["flag"].panel.free == pytest.approx((1.0, 0.0, 0.0))
    assert layouts["flag"].panel.normal == pytest.approx((0.0, -1.0, 0.0))
    assert layouts["banner"].panel.normal == pytest.approx((0.0, 1.0, 0.0))
    assert layouts["awning"].panel.normal[2] > 0.8
    turned = cloth_shapes.banner(facing_deg=90.0)
    assert turned.panel.normal == pytest.approx((-1.0, 0.0, 0.0))


@pytest.mark.parametrize("kind", sorted(cloth_shapes.BUILDERS))
def test_cloth_stands_off_its_rod(kind):
    layout = _layouts()[kind]
    rod = layout.rods[0]
    axis = cloth_shapes.normalised(cloth_shapes.sub(rod.end, rod.start))
    nearest = math.inf
    for vertex in layout.panel.vertices:
        offset = cloth_shapes.sub(vertex, rod.start)
        radial = cloth_shapes.sub(offset, cloth_shapes.mul(axis, cloth_shapes.dot(offset, axis)))
        nearest = min(nearest, math.sqrt(cloth_shapes.dot(radial, radial)))
    assert nearest >= rod.radius + cloth_shapes.STANDOFF_M - 1e-9


def test_pennant_tapers_to_a_point():
    panel = cloth_shapes.pennant().panel
    tip_span = [
        cloth_shapes.dot(vertex, panel.cross_axis)
        for vertex in panel.vertices
        if math.isclose(cloth_shapes.dot(cloth_shapes.sub(vertex, panel.bone_heads[0]), panel.free), 1.0)
    ]
    assert max(tip_span) - min(tip_span) == pytest.approx(0.4 * (1.0 - cloth_shapes.PENNANT_TAPER))


def test_invalid_sizes_are_refused():
    with pytest.raises(cloth_shapes.ClothError):
        cloth_shapes.flag(height=3.0, pole_height=3.0)
    with pytest.raises(cloth_shapes.ClothError):
        cloth_shapes.panel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 1.0, 0)
    with pytest.raises(cloth_shapes.ClothError):
        cloth_shapes.panel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 1.0, 4, taper=1.0)


@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.skipif(blender_discovery.find_blender() is None, reason="Blender executable not found")
def test_cloth_selftest_exports_each_type_skinned_and_validates(tmp_path, monkeypatch):
    result_path = tmp_path / "result.json"
    completed = subprocess.run(
        [
            blender_discovery.require_blender(),
            "--background",
            "--factory-startup",
            "--python",
            str(SCRIPT_PATH),
            "--",
            str(REPO_ROOT),
            str(result_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result_path.is_file(), completed.stdout + completed.stderr
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload.get("ok"), payload
    monkeypatch.setattr(config, "ASSET_OUT_DIR", tmp_path / "assets_out")
    for kind in cloth_shapes.BUILDERS:
        report = validate.validate_asset(f"cloth-{kind}", "prop")
        failed = [f"{check.name}: {check.detail}" for check in report.checks if not check.passed]
        assert not failed, failed
