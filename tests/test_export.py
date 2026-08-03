"""Requirement 3/4: the rig, the Spine document, the atlas and the preview."""

from __future__ import annotations

import json

from ocs import atlas as atlas_mod
from ocs import cleanup, limbs, player, rig as rig_mod, skeleton, spine_export, taxonomy
from ocs.config import AtlasSettings, RigSettings


def build(decomp):
    reports = cleanup.analyze(decomp)
    kept, _ = cleanup.apply_verdicts(decomp, reports)
    rig = skeleton.guess_rig(decomp, kept)
    parts, _ = limbs.partition(decomp, kept, rig, RigSettings())
    return rig_mod.build_rig(decomp, parts, rig, RigSettings()), parts


# ── bone frame ───────────────────────────────────────────────────────


def test_bone_hierarchy_is_ordered_and_acyclic(figure):
    built, _ = build(figure)
    seen: set[str] = set()
    for bone in built.bones:
        if bone.parent is not None:
            assert bone.parent in seen, f"{bone.name} precedes its parent {bone.parent}"
        assert bone.name not in seen
        seen.add(bone.name)
    assert built.bones[0].name == "root"
    assert built.bones[0].parent is None


def test_required_bones_always_exist(figure):
    built, _ = build(figure)
    names = {b.name for b in built.bones}
    for required in taxonomy.REQUIRED_BONES:
        assert required in names


def test_root_is_the_spine_origin(figure):
    built, _ = build(figure)
    root = built.bones[0]
    assert (root.local_x, root.local_y, root.local_rot) == (0.0, 0.0, 0.0)
    assert abs(root.world_x) < 1e-6 and abs(root.world_y) < 1e-6


# ── meshes and weights ───────────────────────────────────────────────


def test_deformable_parts_get_weighted_meshes(figure):
    built, _ = build(figure)
    meshes = [a for a in built.attachments.values() if a.kind == "mesh"]
    assert meshes, "limb parts must be meshes, not rigid quads"
    for m in meshes:
        assert m.hull >= 3
        assert len(m.uvs) % 2 == 0
        assert len(m.triangles) % 3 == 0
        assert m.triangles and max(m.triangles) < len(m.uvs) // 2
        assert m.hull <= len(m.uvs) // 2


def test_vertex_weights_sum_to_one(figure):
    built, _ = build(figure)
    n_bones = len(built.bones)
    for att in built.attachments.values():
        if att.kind != "mesh":
            continue
        i = counted = 0
        verts = att.vertices
        while i < len(verts):
            bone_count = int(verts[i]); i += 1
            assert 1 <= bone_count <= RigSettings().max_bones_per_vertex
            total = 0.0
            for k in range(bone_count):
                index = int(verts[i + k * 4])
                assert 0 <= index < n_bones
                weight = verts[i + k * 4 + 3]
                assert 0.0 < weight <= 1.0
                total += weight
            assert abs(total - 1.0) < 1e-3, f"{att.name}: weights sum to {total}"
            i += bone_count * 4
            counted += 1
        assert counted == len(att.uvs) // 2


def test_limb_meshes_span_their_joint(figure):
    """An upper arm weighted only to the shoulder cannot bend at the elbow.

    Only asserted for tags that actually reach past the joint. ``bottomwear`` is
    restricted to the upper leg on purpose, so shorts must *not* be weighted to
    the knee -- that restriction is the fix for weights leaking across limbs.
    """
    built, _ = build(figure)
    index = {b.name: i for i, b in enumerate(built.bones)}
    checked = 0
    for slot in built.slots:
        region = taxonomy.part_region(slot.part_name)
        if region not in ("arm_r_upper", "arm_l_upper", "leg_r_upper", "leg_l_upper"):
            continue
        att = built.attachments[slot.attachment]
        if att.kind != "mesh":
            continue

        tag = taxonomy.base_tag(slot.part_name)
        lower = region.replace("_upper", "_lower")
        allowed = taxonomy.allowed_regions(tag, taxonomy.part_side(slot.part_name))
        if allowed is not None and lower not in allowed:
            continue                      # this tag legitimately stops at the joint

        child = next(s for s in taxonomy.SKIN_REGIONS if s.name == region).to_bone
        if child in index:
            assert index[child] in att.bones_used, f"{slot.name} is not weighted to {child}"
            checked += 1
    assert checked, "no limb meshes were checked -- fixture or taxonomy changed"


def test_weights_never_cross_between_arm_and_leg(figure):
    """A hip garment must not follow a hand, however close the hand hangs."""
    built, _ = build(figure)
    names = [b.name for b in built.bones]
    for slot in built.slots:
        att = built.attachments[slot.attachment]
        if att.kind != "mesh":
            continue
        used = {names[i] for i in att.bones_used}
        tag = taxonomy.base_tag(slot.part_name)
        if tag == "bottomwear":
            assert not (used & {"rightElbow", "leftElbow", "rightHand", "leftHand"}), \
                f"{slot.name} weighted to {used}"
        if tag == "handwear":
            assert not (used & {"rightKnee", "leftKnee", "rightFoot", "leftFoot"}), \
                f"{slot.name} weighted to {used}"


# ── Spine document ───────────────────────────────────────────────────


def test_skeleton_json_validates(figure):
    built, _ = build(figure)
    doc = spine_export.build_skeleton(built, name="fixture")
    spine_export.add_animations(doc, built)
    assert spine_export.validate(doc) == []
    assert doc["skeleton"]["spine"].startswith("4.2")
    assert doc["skins"][0]["name"] == "default"


def test_slot_order_is_draw_order_far_to_near(figure):
    built, _ = build(figure)
    depths = [s.depth_median for s in built.slots]
    assert depths == sorted(depths, reverse=True), "slots must run far -> near"


def test_animations_reference_only_existing_bones(figure):
    built, _ = build(figure)
    doc = spine_export.build_skeleton(built)
    spine_export.add_animations(doc, built)
    names = {b["name"] for b in doc["bones"]}
    assert "idle" in doc["animations"]
    for anim in doc["animations"].values():
        assert set(anim["bones"]) <= names


def test_mirrored_limbs_animate_with_opposite_signs(figure):
    """Both limbs aim down their own length, so the same swing needs opposite signs."""
    built, _ = build(figure)
    doc = spine_export.build_skeleton(built)
    spine_export.add_animations(doc, built, ["walk"])
    bones = doc["animations"]["walk"]["bones"]
    left = bones["leftLeg"]["rotate"][0]["value"]
    right = bones["rightLeg"]["rotate"][0]["value"]
    assert left * right < 0, f"left={left} right={right} should oppose"


def test_curve_arity_matches_the_timeline_channel_count(figure):
    """A translate curve needs 8 numbers, a rotate curve 4.

    The runtime reads channel ``n`` at ``curve[n*4 : n*4+4]``, so a translate
    timeline carrying only 4 reads ``undefined`` for y. ``undefined * scale`` is
    NaN, it lands in the bezier table, and every descendant bone's world
    transform becomes NaN -- which surfaces only as the player refusing to start
    with "Animation bounds are invalid". Nothing else catches it: the JSON is
    well-formed and every number in it is finite.
    """
    built, _ = build(figure)
    doc = spine_export.build_skeleton(built)
    spine_export.add_animations(doc, built)

    seen = set()
    for anim_name, anim in doc["animations"].items():
        for bone_name, timelines in anim["bones"].items():
            for timeline_name, keys in timelines.items():
                channels = spine_export._TIMELINE_CHANNELS[timeline_name]
                seen.add(timeline_name)
                for i, key in enumerate(keys):
                    curve = key.get("curve")
                    if curve is None:
                        assert i == len(keys) - 1, (
                            f"{anim_name}/{bone_name}/{timeline_name} key {i} has no curve"
                        )
                        continue
                    assert len(curve) == channels * 4, (
                        f"{anim_name}/{bone_name}/{timeline_name} key {i}: "
                        f"{len(curve)} values for {channels} channel(s)"
                    )
    assert {"rotate", "translate"} <= seen, "fixture exercises both arities"


def test_curve_handles_are_absolute_coordinates(figure):
    """Handles are (time, value) points, not normalised 0..1 fractions.

    ``readCurve`` uses ``curve[i]`` directly as a time and ``curve[i+1]`` as a
    value, so handles must lie between the two keyframes they interpolate.
    """
    built, _ = build(figure)
    doc = spine_export.build_skeleton(built)
    spine_export.add_animations(doc, built, ["idle"])

    checked = 0
    for bone_name, timelines in doc["animations"]["idle"]["bones"].items():
        for timeline_name, keys in timelines.items():
            channels = spine_export._TIMELINE_CHANNELS[timeline_name]
            for i, key in enumerate(keys[:-1]):
                t1 = key.get("time", 0.0)
                t2 = keys[i + 1].get("time", 0.0)
                curve = key["curve"]
                for c in range(channels):
                    cx1, cx2 = curve[c * 4], curve[c * 4 + 2]
                    assert t1 <= cx1 <= t2, f"{bone_name}/{timeline_name}: cx1 {cx1} outside [{t1},{t2}]"
                    assert t1 <= cx2 <= t2, f"{bone_name}/{timeline_name}: cx2 {cx2} outside [{t1},{t2}]"
                    assert cx1 <= cx2
                    checked += 1
    assert checked, "no curves were checked"


def test_validate_rejects_a_short_translate_curve(figure):
    built, _ = build(figure)
    doc = spine_export.build_skeleton(built)
    spine_export.add_animations(doc, built, ["idle"])
    assert spine_export.validate(doc) == []

    doc["animations"]["idle"]["bones"]["torso"]["translate"][0]["curve"] = [0.1, 0, 0.3, 1]
    problems = spine_export.validate(doc)
    assert any("needs 8" in p for p in problems), problems


def test_validate_rejects_non_finite_numbers(figure):
    built, _ = build(figure)
    doc = spine_export.build_skeleton(built)
    doc["bones"][1]["x"] = float("nan")
    assert any("non-finite" in p for p in spine_export.validate(doc))


def test_region_attachments_counter_rotate_their_bone(figure):
    """A region inherits its bone's world rotation, so it must cancel it out.

    Every bone is aimed down its own length, so without this a face rides at 73
    degrees and a shin lies on its side.
    """
    built, _ = build(figure)
    doc = spine_export.build_skeleton(built)
    by_name = {b.name: b for b in built.bones}
    slot_bone = {s.name: s.bone for s in built.slots}

    checked = 0
    for slot_name, entries in doc["skins"][0]["attachments"].items():
        for _att_name, body in entries.items():
            if body.get("type") == "mesh":
                continue
            bone = by_name[slot_bone[slot_name]]
            got = body.get("rotation", 0.0)
            expected = ((-bone.world_rot + 180) % 360) - 180
            assert abs(got - expected) < 0.01, f"{slot_name}: {got} != {expected}"
            checked += 1
    assert checked, "fixture produced no region attachments"


def test_validate_catches_a_broken_document(figure):
    built, _ = build(figure)
    doc = spine_export.build_skeleton(built)
    doc["slots"][0]["bone"] = "nonexistent"
    problems = spine_export.validate(doc)
    assert any("unknown bone" in p for p in problems)


# ── atlas + preview ──────────────────────────────────────────────────


def test_every_attachment_is_packed(figure):
    built, _ = build(figure)
    packed = atlas_mod.pack(built.part_images, AtlasSettings())
    assert {r.name for r in packed.regions} == set(built.attachments)
    assert packed.size[0] > 0 and packed.size[1] > 0


def test_atlas_regions_stay_inside_the_page(figure):
    built, _ = build(figure)
    packed = atlas_mod.pack(built.part_images, AtlasSettings())
    w, h = packed.size
    for r in packed.regions:
        assert 0 <= r.x and r.x + r.width <= w
        assert 0 <= r.y and r.y + r.height <= h


def test_atlas_text_has_a_header_and_every_region(figure):
    built, _ = build(figure)
    packed = atlas_mod.pack(built.part_images, AtlasSettings())
    text = packed.to_text("skeleton.png")
    assert text.splitlines()[0] == "skeleton.png"
    assert f"size: {packed.size[0]}, {packed.size[1]}" in text
    for r in packed.regions:
        assert f"\n{r.name}\n" in text


def test_preview_is_self_contained(figure, tmp_path):
    built, parts = build(figure)
    packed = atlas_mod.pack(built.part_images, AtlasSettings())
    png, atlas_file = packed.write(tmp_path, "skeleton")
    skel = spine_export.export_skeleton(built, tmp_path / "skeleton.json")

    out, _embedded = player.build_preview(skel, atlas_file, png, tmp_path / "preview.html")
    html = out.read_text(encoding="utf-8")
    assert "data:application/json;base64," in html
    assert "data:image/png;base64," in html
    assert "rawDataURIs" in html
    # No relative asset references that would break from file://
    assert 'src="skeleton.png"' not in html


def test_export_round_trips_through_json(figure, tmp_path):
    built, _ = build(figure)
    path = spine_export.export_skeleton(built, tmp_path / "skeleton.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert len(doc["bones"]) == len(built.bones)
    assert len(doc["slots"]) == len(built.slots)
    assert len(doc["skins"][0]["attachments"]) == len(built.slots)
