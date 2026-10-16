"""Runs inside Blender's own interpreter via `blender --background --python`.

Builds a small material'd box, runs it through assets/lod.py's generate_lods (so the scene has
three named _LOD0/_LOD1/_LOD2 objects with the baked normal maps LOD1/2 always carry), exports
the group with assets/export.py's export_asset, and reports the written glb/textures/asset.json
as JSON, so the pytest suite gets automated coverage of bpy-only code it cannot import directly.
It then exports a rigid-animated (spinning <part>_ANIM child) and a skinned (bone chain) asset
through the hierarchy-preserving path and checks the glb node tree, root shift, clips and skin.
Invoked as:
    blender --background --factory-startup --python export_selftest.py -- <repo_root> <out_dir> <result.json>
"""

import json
import sys
import traceback
from pathlib import Path

import bmesh
import bpy


def _read_args() -> tuple[Path, Path, Path]:
    argv = sys.argv
    separator = argv.index("--")
    repo_root, out_dir, result_path = argv[separator + 1:separator + 4]
    return Path(repo_root), Path(out_dir), Path(result_path)


def _clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)


def _detailed_box(name: str, material: "bpy.types.Material") -> "bpy.types.Object":
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=4, use_grid_fill=True)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def _uv_unwrap(obj: "bpy.types.Object") -> None:
    for other in bpy.context.selected_objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")


def _box_at(name: str, location, material: "bpy.types.Material") -> "bpy.types.Object":
    from assets import hardsurface as hs

    obj = hs.bevelled_box(name, size=(1.0, 1.0, 1.0), bevel_width=0.0, location=location)
    obj.data.materials.append(material)
    return obj


def _no_drift_across_repeated_export_calls(export_module, out_dir) -> dict:
    """A live MCP session rebuilds individual parts of a multi-object asset and re-exports
    between fixes. export_asset must never permanently move an object: rebuild anchor fresh, then
    export it alone, then add a second object built at different, asymmetric coordinates and
    export both together - anchor's obj.location must come back to exactly where it started
    both times, or a later patch rebuilding just one part would drift out of alignment with parts
    left untouched, which is exactly what happened to the windmill's tower sections."""
    material = bpy.data.materials.new("drift_test_material")
    material.use_nodes = True

    anchor = _box_at("drift_anchor", (2.0, 0.0, 0.5), material)
    location_before_first = tuple(anchor.location)
    export_module.export_asset([anchor], name="drift-test", kind="prop")
    location_after_first = tuple(anchor.location)

    companion = _box_at("drift_companion", (-5.0, 9.0, 0.5), material)
    export_module.export_asset([anchor, companion], name="drift-test", kind="prop")
    location_after_second = tuple(anchor.location)

    return {
        "location_before_first": location_before_first,
        "location_after_first": location_after_first,
        "location_after_second": location_after_second,
        "no_drift": location_before_first == location_after_first == location_after_second,
    }


def _team_mask_export(export_module, objects, out_dir) -> dict:
    from assets import team_colour

    size = 64
    mask = bpy.data.images.new("selftest_team_mask", width=size, height=size, alpha=True)
    pixels = [0.0, 0.0, 0.0, 1.0] * (size * size)
    for row in range(8, 24):
        for column in range(8, 24):
            pixels[(row * size + column) * 4] = 1.0
    mask.pixels.foreach_set(pixels)
    mask.update()

    name = "export-selftest-unit"
    result = export_module.export_asset(objects, name=name, kind="unit", team_mask=mask)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    entry = manifest.get(team_colour.MANIFEST_KEY)
    mask_written = entry is not None and (out_dir / name / entry["mask"]).is_file()

    rejected = False
    try:
        export_module.export_asset(objects, name="export-selftest-prop", kind="prop", team_mask=mask)
    except export_module.ExportError:
        rejected = True

    return {
        "entry": entry,
        "mask_written": mask_written,
        "prop_rejected": rejected,
        "ok": manifest["kind"] == "unit" and mask_written and rejected,
    }


FPS = 24
SPIN_FRAMES = 48
WAVE_FRAMES = 36
STRIP_BONES = 4
STRIP_HEIGHT_M = 2.0


def _gltf_vector(blender_xyz) -> list[float]:
    x, y, z = blender_xyz
    return [x, z, -y]


def _close(a, b, tolerance: float = 1e-4) -> bool:
    return a is not None and len(a) == len(b) and all(abs(p - q) <= tolerance for p, q in zip(a, b))


def _block(name: str, size, offset, material) -> "bpy.types.Object":
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for vert in bm.verts:
        vert.co.x = vert.co.x * size[0] + offset[0]
        vert.co.y = vert.co.y * size[1] + offset[1]
        vert.co.z = vert.co.z * size[2] + offset[2]
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(material)
    bpy.context.collection.objects.link(obj)
    return obj


def _push_to_nla(owner, action_name: str) -> None:
    action = owner.animation_data.action
    action.name = action_name
    track = owner.animation_data.nla_tracks.new()
    track.name = action_name
    strip = track.strips.new(action_name, int(action.frame_range[0]), action)
    strip.name = action_name
    owner.animation_data.action = None


def _clear_animation_scene() -> None:
    _clear_scene()
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    for armature in list(bpy.data.armatures):
        if armature.users == 0:
            bpy.data.armatures.remove(armature)
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = 1 + max(SPIN_FRAMES, WAVE_FRAMES)


def _fx_socket_export(export_module, out_dir) -> dict:
    """A static block with a chimney smoke socket on top and a fire socket parented to the block:
    both must ship as FX_ nodes at their base-centred transforms with asset.json fx[] beside them,
    the parented socket must not be shifted twice, and a socket without an entry is refused."""
    from assets import fx

    _clear_animation_scene()
    material = bpy.data.materials.new("fx_atlas")
    material.use_nodes = True
    block = _block("Kiln", (1.0, 1.0, 0.4), (0.0, 0.0, 0.2), material)
    block.location = (3.0, -2.0, 0.0)
    bpy.context.view_layer.update()
    smoke, smoke_entry = fx.add_fx_socket("smoke", (3.0, -2.0, 0.4), "idle", 8.0, 0.6, (0.4, 0.4, 0.4), label="chimney")
    fire, fire_entry = fx.add_fx_socket("fire", (3.2, -2.0, 0.2), "working", 30.0, 0.3, (1.0, 0.5, 0.1), parent=block)
    sockets_before = (tuple(smoke.matrix_world.translation), tuple(fire.matrix_world.translation))

    name = "export-selftest-fx"
    result = export_module.export_asset([block, smoke, fire], name=name, kind="prop", fx=[smoke_entry, fire_entry])
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    document = export_module.glb_document(result.glb_path)
    nodes = {node.get("name"): node for node in document.get("nodes", [])}
    names = [node.get("name") for node in document.get("nodes", [])]
    block_node = nodes.get("Kiln", {})
    block_children = [names[index] for index in block_node.get("children", [])]
    fire_world = [
        a + b for a, b in zip(block_node.get("translation", [0.0, 0.0, 0.0]), nodes.get("FX_fire", {}).get("translation", [0.0, 0.0, 0.0]))
    ]

    orphan_rejected = False
    try:
        export_module.export_asset([block, smoke, fire], name=name, kind="prop", fx=[smoke_entry])
    except export_module.ExportError:
        orphan_rejected = True

    checks = {
        "manifest_fx": manifest.get("fx") == [smoke_entry, fire_entry],
        "smoke_socket": _close(nodes.get("FX_smoke_chimney", {}).get("translation"), _gltf_vector((0.0, 0.0, 0.4))),
        "smoke_is_empty": "mesh" not in nodes.get("FX_smoke_chimney", {"mesh": 0}),
        "fire_under_block": block_children == ["FX_fire"],
        "fire_shifted_once": _close(fire_world, _gltf_vector((0.2, 0.0, 0.2))),
        "sockets_not_moved": (tuple(smoke.matrix_world.translation), tuple(fire.matrix_world.translation)) == sockets_before,
        "orphan_rejected": orphan_rejected,
    }
    return {"checks": checks, "nodes": names, "ok": all(checks.values())}


def _rigid_export(export_module, out_dir) -> dict:
    """A base offset from the origin with a spinning <part>_ANIM child: the root must carry the
    base-centre shift so the base and rotor keep their own design-space transforms in the glb."""
    import math

    from assets import animation_manifest

    _clear_animation_scene()
    material = bpy.data.materials.new("rigid_atlas")
    material.use_nodes = True
    base = _block("Base", (1.0, 1.0, 0.4), (0.0, 0.0, 0.2), material)
    base.location = (3.0, -2.0, 0.0)
    rotor = _block("Rotor_ANIM", (1.6, 0.15, 0.08), (0.0, 0.0, 0.0), material)
    rotor.parent = base
    rotor.location = (0.0, 0.0, 0.5)
    rotor.rotation_mode = "XYZ"
    rotor.animation_data_create()
    rotor.rotation_euler = (0.0, 0.0, 0.0)
    rotor.keyframe_insert("rotation_euler", index=2, frame=1)
    rotor.rotation_euler = (0.0, 0.0, 2.0 * math.pi)
    rotor.keyframe_insert("rotation_euler", index=2, frame=1 + SPIN_FRAMES)
    _push_to_nla(rotor, "idle_rotor_spin")
    bpy.context.view_layer.update()

    entry = animation_manifest.manifest_entry("idle", "rotor_spin", "rigid", SPIN_FRAMES / FPS, FPS, ["Rotor_ANIM"])
    base_before = tuple(base.matrix_world.translation)
    name = "export-selftest-rigid"
    result = export_module.export_asset([base, rotor], name=name, kind="prop", animations=[entry])
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    document = export_module.glb_document(result.glb_path)
    nodes = {node.get("name"): node for node in document.get("nodes", [])}
    names = [node.get("name") for node in document.get("nodes", [])]
    root = nodes.get(name, {})
    root_children = [names[index] for index in root.get("children", [])]
    base_children = [names[index] for index in nodes.get("Base", {}).get("children", [])]
    clips = [animation.get("name") for animation in document.get("animations", [])]

    missing_rejected = False
    bad = dict(entry, nodes=["Missing_ANIM"])
    try:
        export_module.export_asset([base, rotor], name=name, kind="prop", animations=[bad])
    except export_module.ExportError:
        missing_rejected = True

    checks = {
        "manifest_animations": manifest.get("animations") == [entry],
        "root_shift": _close(root.get("translation"), _gltf_vector((-3.0, 2.0, 0.0))),
        "root_children": root_children == ["Base"],
        "base_translation": _close(nodes.get("Base", {}).get("translation"), _gltf_vector((3.0, -2.0, 0.0))),
        "rotor_under_base": base_children == ["Rotor_ANIM"],
        "rotor_translation": _close(nodes.get("Rotor_ANIM", {}).get("translation"), _gltf_vector((0.0, 0.0, 0.5))),
        "clips": clips == ["idle_rotor_spin"],
        "root_removed": name not in bpy.data.objects,
        "base_unparented": base.parent is None,
        "rotor_parent_kept": rotor.parent == base,
        "base_not_moved": tuple(base.matrix_world.translation) == base_before,
        "bbox_base_centred": _close(result.bbox_min[2:], [0.0]) and _close(
            [result.bbox_min[0] + result.bbox_max[0], result.bbox_min[1] + result.bbox_max[1]], [0.0, 0.0]
        ),
        "missing_node_rejected": missing_rejected,
    }
    return {"checks": checks, "nodes": names, "ok": all(checks.values())}


def _skinned_export(export_module, out_dir) -> dict:
    from assets import animation_manifest

    _clear_animation_scene()
    material = bpy.data.materials.new("skinned_atlas")
    material.use_nodes = True
    strip = _block("Strip", (0.6, 0.1, STRIP_HEIGHT_M), (0.0, 0.0, STRIP_HEIGHT_M / 2), material)
    for other in bpy.context.selected_objects:
        other.select_set(False)
    strip.select_set(True)
    bpy.context.view_layer.objects.active = strip
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.subdivide(number_cuts=STRIP_BONES * 4 - 1)
    bpy.ops.object.mode_set(mode="OBJECT")

    armature_data = bpy.data.armatures.new("StripRig_data")
    rig = bpy.data.objects.new("StripRig", armature_data)
    rig.location = (-1.5, 4.0, 0.0)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    segment = STRIP_HEIGHT_M / STRIP_BONES
    parent = None
    for index in range(STRIP_BONES):
        bone = armature_data.edit_bones.new(f"seg{index}")
        bone.head = (0.0, 0.0, index * segment)
        bone.tail = (0.0, 0.0, (index + 1) * segment)
        if parent is not None:
            bone.parent = parent
            bone.use_connect = True
        parent = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    for index in range(STRIP_BONES):
        group = strip.vertex_groups.new(name=f"seg{index}")
        centre = (index + 0.5) * segment
        for vert in strip.data.vertices:
            weight = max(0.0, 1.0 - abs(vert.co.z - centre) / segment)
            if weight > 0.0:
                group.add([vert.index], weight, "REPLACE")
    strip.parent = rig
    modifier = strip.modifiers.new("Armature", "ARMATURE")
    modifier.object = rig
    rig.animation_data_create()
    for frame, sign in ((1, 1.0), (1 + WAVE_FRAMES // 2, -1.0), (1 + WAVE_FRAMES, 1.0)):
        for index in range(STRIP_BONES):
            pose_bone = rig.pose.bones[f"seg{index}"]
            pose_bone.rotation_mode = "XYZ"
            pose_bone.rotation_euler = (sign * 0.25, 0.0, 0.0)
            pose_bone.keyframe_insert("rotation_euler", frame=frame)
    _push_to_nla(rig, "idle_strip_wave")
    bpy.context.view_layer.update()

    bones = [f"seg{index}" for index in range(STRIP_BONES)]
    entry = animation_manifest.manifest_entry("idle", "strip_wave", "skinned", WAVE_FRAMES / FPS, FPS, bones)
    name = "export-selftest-skinned"
    result = export_module.export_asset([rig, strip], name=name, kind="creature", animations=[entry])
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    document = export_module.glb_document(result.glb_path)
    names = [node.get("name") for node in document.get("nodes", [])]
    root = next((node for node in document.get("nodes", []) if node.get("name") == name), {})
    root_children = [names[index] for index in root.get("children", [])]
    skins = document.get("skins", [])
    joints = sorted(names[joint] for skin in skins for joint in skin["joints"])

    checks = {
        "manifest_animations": manifest.get("animations") == [entry],
        "root_shift": _close(root.get("translation"), _gltf_vector((1.5, -4.0, 0.0))),
        "rig_under_root": root_children == ["StripRig"],
        "one_skin": len(skins) == 1,
        "joints": joints == bones,
        "clips": [animation.get("name") for animation in document.get("animations", [])] == ["idle_strip_wave"],
        "triangles_counted": result.triangle_counts.get("LOD0", 0) > 0,
        "rest_pose_bbox": _close(result.bbox_max[2:], [STRIP_HEIGHT_M]) and rig.data.pose_position == "POSE",
        "root_removed": name not in bpy.data.objects,
        "rig_unparented": rig.parent is None,
        "strip_parent_kept": strip.parent == rig,
    }
    return {"checks": checks, "nodes": names, "ok": all(checks.values())}


def main() -> None:
    repo_root, out_dir, result_path = _read_args()
    sys.path.insert(0, str(repo_root))
    _clear_scene()

    payload: dict = {"ok": False}
    try:
        from terrain import config

        config.ASSET_OUT_DIR = out_dir

        from assets import export, lod

        material = bpy.data.materials.new("scratch_atlas")
        material.use_nodes = True
        lod0 = _detailed_box("hero", material)
        _uv_unwrap(lod0)

        lod_set = lod.generate_lods(lod0, ratios=(1.0, 0.5, 0.15), texture_size=128)

        name = "export-selftest"
        result = export.export_asset(lod_set.objects, name=name, kind="prop")
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        counts = result.triangle_counts
        ok = (
            result.glb_path.is_file()
            and result.manifest_path.is_file()
            and list(counts) == ["LOD0", "LOD1", "LOD2"]
            and counts["LOD0"] > counts["LOD1"] > counts["LOD2"] > 0
            and len(result.materials) == 3
            and len(result.textures) == 6
            and all((out_dir / name / "textures" / t).is_file() for t in result.textures)
            and all(result.bbox_max[i] >= result.bbox_min[i] for i in range(3))
            and manifest["kind"] == "prop"
            and manifest["contact_sheet"] == export.CONTACT_SHEET_NAME
        )

        drift_check = _no_drift_across_repeated_export_calls(export, out_dir)
        ok = ok and drift_check["no_drift"]

        team_check = _team_mask_export(export, lod_set.objects, out_dir)
        ok = ok and team_check["ok"]

        static_document = export.glb_document(result.glb_path)
        static_check = "animations" not in manifest and "animations" not in static_document
        ok = ok and static_check

        rigid_check = _rigid_export(export, out_dir)
        skinned_check = _skinned_export(export, out_dir)
        fx_check = _fx_socket_export(export, out_dir)
        ok = ok and rigid_check["ok"] and skinned_check["ok"] and fx_check["ok"]

        payload = {
            "ok": ok,
            "result": result.as_dict(),
            "manifest": manifest,
            "drift_check": drift_check,
            "team_check": team_check,
            "static_check": static_check,
            "rigid_check": rigid_check,
            "skinned_check": skinned_check,
            "fx_check": fx_check,
        }
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}

    result_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
