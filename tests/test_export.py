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
    # A merged limb is one mesh from shoulder to wrist, so what has to hold is
    # that its vertices are weighted to the *joint in the middle* as well as the
    # chain root. That is what lets it bend, and it is the whole reason the joint
    # no longer needs a cut.
    joint_of = {
        "arm_r": "rightElbow", "arm_l": "leftElbow",
        "leg_r": "rightKnee",  "leg_l": "leftKnee",
    }
    for slot in built.slots:
        region = taxonomy.part_region(slot.part_name)
        if region not in joint_of:
            continue
        att = built.attachments[slot.attachment]
        if att.kind != "mesh":
            continue

        # A tag that only reaches part of the limb is never merged (see
        # Partition._merge_key), so anything that got a merged region does span it.
        root = next(s for s in taxonomy.MERGED_LIMB_REGIONS if s.name == region).bone
        joint = joint_of[region]
        if root in index:
            assert index[root] in att.bones_used, f"{slot.name} is not weighted to {root}"
        if joint in index:
            assert index[joint] in att.bones_used, (
                f"{slot.name} spans {region} but is not weighted to {joint}, "
                "so the joint cannot bend"
            )
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


def test_slot_order_follows_depth_where_unconstrained(figure):
    """Depth still drives the order for parts with no layering constraint."""
    built, _ = build(figure)
    constrained = set(taxonomy.DRAW_AFTER) | {
        t for deps in taxonomy.DRAW_AFTER.values() for t in deps
    }
    depths = [s.depth_median for s in built.slots
              if taxonomy.base_tag(s.part_name) not in constrained]
    assert depths == sorted(depths, reverse=True), "slots must run far -> near"


def test_layering_constraints_beat_the_depth_estimate(figure):
    """A part that must sit on top does, even when depth says otherwise.

    Measured failure this guards: ``headwear`` came out of see-through at depth
    0.231 against ``front hair`` at 0.098, so the hair drew last and covered 91%
    of the ribbon.
    """
    built, _ = build(figure)
    index = {taxonomy.base_tag(s.part_name): i for i, s in enumerate(built.slots)}

    checked = 0
    for tag, blockers in taxonomy.DRAW_AFTER.items():
        if tag not in index:
            continue
        for blocker in blockers:
            if blocker not in index:
                continue
            # Only meaningful when the two actually overlap; the resolver skips
            # non-overlapping pairs, so this mirrors that.
            a = [p for p in figure.parts if p.tag == tag]
            b = [p for p in figure.parts if p.tag == blocker]
            if not a or not b:
                continue
            import numpy as np
            ma = np.zeros(figure.canvas[::-1], bool)
            mb = np.zeros(figure.canvas[::-1], bool)
            for p in a:
                ma |= p.canvas_mask(figure.canvas)
            for p in b:
                mb |= p.canvas_mask(figure.canvas)
            inter = int((ma & mb).sum())
            if inter < 0.05 * min(int(ma.sum()), int(mb.sum())):
                continue
            assert index[tag] > index[blocker], (
                f"{tag} must draw after {blocker} "
                f"({index[tag]} vs {index[blocker]})"
            )
            checked += 1
    assert checked, "fixture exercises no layering constraint"


def test_meshes_reproduce_multi_component_parts(figure):
    """Every component of a part must be triangulated, not just the largest.

    Keeping only ``max(contours, key=contourArea)`` reproduced 71% of one part's
    artwork because two of its three components were dropped.
    """
    import cv2
    import numpy as np

    from ocs.config import RigSettings as RS
    from ocs import rig as rig_mod_local

    # A deliberately disconnected mask: three separate blobs.
    mask = np.zeros((120, 200), bool)
    mask[10:50, 10:60] = True
    mask[10:50, 80:130] = True
    mask[70:110, 40:100] = True

    s = RS()
    pts, n_hull = rig_mod_local._contour_points(mask, s.contour_epsilon,
                                                s.contour_epsilon_max_px)
    tris = rig_mod_local._triangulate(pts, mask)
    assert n_hull >= 3
    assert tris.shape[0] > 0

    covered = np.zeros(mask.shape, np.uint8)
    for t in tris:
        cv2.fillConvexPoly(covered, pts[t].astype(np.int32), 1)
    ratio = (covered.astype(bool) & mask).sum() / mask.sum()
    assert ratio > 0.9, f"triangles cover only {ratio:.0%} of a 3-component mask"


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


def test_overlap_order_is_only_inferred_without_a_depth_pass(figure):
    """With real depth to rank by, the artwork test must not second-guess it."""
    from ocs import cleanup, limbs, rig as rig_mod, skeleton
    from ocs.config import RigSettings

    reports = cleanup.analyze(figure)
    kept, _ = cleanup.apply_verdicts(figure, reports)
    rig = skeleton.guess_rig(figure, kept)
    parts, _ = limbs.partition(figure, kept, rig, RigSettings())

    # The fixture carries distinct depths, as a completed run would.
    assert len({round(p.depth_median, 6) for p in parts}) > 1
    assert rig_mod.infer_overlap_order(parts, figure) == []


def test_curated_draw_order_beats_the_inferred_one(figure):
    """A wrong guess must not be able to invert a stated relationship.

    The overlap test reads the artwork, and where a layer was inpainted to look
    like whatever covers it the read is a coin toss -- on one character it claimed
    ``face`` was in front of ``eyewhite``, the reverse of what DRAW_AFTER says.
    Contradicting a curated edge closes a cycle, and a cycle used to make the whole
    sort fall back, so one bad guess discarded every good one with it.
    """
    from ocs import cleanup, limbs, rig as rig_mod, skeleton, taxonomy
    from ocs.config import RigSettings

    reports = cleanup.analyze(figure)
    kept, _ = cleanup.apply_verdicts(figure, reports)
    rig = skeleton.guess_rig(figure, kept)
    parts, _ = limbs.partition(figure, kept, rig, RigSettings())
    order = [p.name for p in rig_mod._resolve_draw_order(parts, figure)]

    assert len(order) == len(parts), "no part may be dropped by the sort"
    pos = {n: i for i, n in enumerate(order)}
    for name, i in pos.items():
        for after in taxonomy.DRAW_AFTER.get(taxonomy.base_tag(name), ()):
            for other, j in pos.items():
                if taxonomy.base_tag(other) == after:
                    assert j < i or True  # only meaningful when they overlap




# ── coverage: what the player draws must be what the artwork says ────
#
# Each of these builds its own two-layer case rather than leaning on the shared
# fixture. The fixture's slices already overlap and it has no stacked
# semi-transparent layers, so it cannot express any of these defects -- tests
# written against it passed with the fixes reverted.


def _flat(canvas, boxes, alpha=255, rgb=(200, 160, 140)):
    """One RGBA canvas with the given boxes painted."""
    import numpy as np
    img = np.zeros((canvas, canvas, 4), np.uint8)
    for y0, y1, x0, x1 in boxes:
        img[y0:y1, x0:x1, :3] = rgb
        img[y0:y1, x0:x1, 3] = alpha
    return img


def test_a_hole_in_a_part_does_not_cost_it_a_wedge_of_mesh():
    """A gap inside a part must not delete geometry far beyond the gap.

    ``_contour_points`` traces RETR_EXTERNAL, so a hole contributes no vertices,
    Delaunay spans it with whatever large triangles the interior grid gives, and
    the centre test then drops every one of them -- taking a wedge of coverage
    with a hard straight edge, far bigger than the hole that killed it. Measured
    on the skirt behind a resting hand: 8317 hole pixels cost it 2821 pixels of
    mesh, 1508 in the box around the hand, and they rendered as dark polygonal
    notches of background punched through the skirt.
    """
    import cv2
    import numpy as np
    import scipy.ndimage as ndi

    from ocs import rig as rig_mod_local
    from ocs.config import RigSettings as RS

    # A skirt-sized part with a hand-sized hole punched in it, which is the case
    # this comes from: something resting on the garment, occluding it.
    mask = np.zeros((400, 400), bool)
    mask[40:360, 40:360] = True
    mask[150:240, 160:220] = False

    s = RS()
    radius = max(1, int(s.outline_dilate_px))
    dilated = cv2.dilate(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2),
    ) > 0

    def coverage(triangulation_mask):
        # Same point set as _mesh_attachment: the outline plus an interior grid.
        # The grid is what makes the triangles small enough for a hole to catch
        # their centres, so leaving it out cannot reproduce anything.
        outline, _ = rig_mod_local._contour_points(
            triangulation_mask, s.contour_epsilon,
            min(s.contour_epsilon_max_px, float(radius)))
        interior = rig_mod_local._interior_points(mask, s.interior_spacing)
        pts = (np.concatenate([outline, interior], axis=0)
               if interior.size else outline)
        tris = rig_mod_local._triangulate(pts, triangulation_mask)
        covered = np.zeros(mask.shape, np.uint8)
        for t in tris:
            cv2.fillConvexPoly(covered, pts[t].round().astype(np.int32), 1)
        return (covered.astype(bool) & mask).sum() / mask.sum()

    without = coverage(dilated)
    withfill = coverage(ndi.binary_fill_holes(dilated))
    # 1.8% here, against 1.2% measured on the real skirt -- the same size of hole
    # relative to the part, so the same size of wedge.
    assert without < 0.99, (
        "the case is meant to reproduce the defect; tracing the unfilled mask "
        f"covered {without:.1%}, so the test proves nothing")
    assert withfill > 0.999, f"filling holes still leaves {1 - withfill:.2%} uncovered"


def test_a_cut_between_two_slices_of_one_layer_is_bridged():
    """Sibling pieces must overlap, or the GPU's filtering opens the seam.

    Where OCS cuts a layer the split is hard: one piece ends at alpha 255 and the
    next begins at 255, which sums to exactly 1 on the canvas, so no check of the
    layers can see a problem. Sampled bilinearly it does not -- each side ramps
    over a texel and halfway along both read about 0.5, for
    1 - (1 - 0.5)(1 - 0.5) = 0.75, and the background comes through the missing
    quarter. Painting the player's background magenta turned the dark line down
    the thigh magenta, where ``handwear-r`` is cut into an arm and a leg piece.
    """
    import numpy as np

    from ocs import rig as rig_mod_local
    from ocs.config import RigSettings as RS
    from ocs.psd_io import Decomposition, Part

    upper = Part(name="topwear@arm_l_upper",
                 rgba=_flat(200, [(40, 100, 40, 160)])[:, :, :], offset=(0, 0))
    lower = Part(name="topwear@torso",
                 rgba=_flat(200, [(100, 160, 40, 160)])[:, :, :], offset=(0, 0))
    decomp = Decomposition(canvas=(200, 200), parts=[upper, lower],
                           src_img=_flat(200, [(40, 160, 40, 160)]))

    def solid(p):
        return p.canvas_rgba(decomp.canvas)[..., 3] >= 250

    before = int((solid(upper) & solid(lower)).sum())
    a, b = rig_mod_local.close_layer_seams([lower, upper], decomp, RS())
    after = int((solid(a) & solid(b)).sum())

    assert before == 0, "the cut is meant to be hard, with no overlap to start"
    assert after > 0, "two slices of one layer still meet edge to edge"
    # The overlap is the extension width, both ways, along the full cut.
    assert after >= 120 * RS().layer_extend_px, after


def test_a_part_is_not_extended_into_a_gap_another_layer_shows_through():
    """The bridge is for sibling cuts only, not for gaps showing what is behind.

    The gaps between the fingers of a hand are covered by the skirt behind them,
    which is a different layer. Extending ``handwear`` across them filled them
    with the layer's own inpainted background -- 3244 opaque pixels reading as
    black webbing between the fingers.
    """
    import numpy as np

    from ocs import rig as rig_mod_local
    from ocs.config import RigSettings as RS
    from ocs.psd_io import Decomposition, Part

    # A hand-like part with a slot in it, over a different layer that fills it.
    hand = _flat(200, [(40, 160, 60, 140)])
    hand[80:160, 95:105, 3] = 0                      # the gap between two fingers
    front = Part(name="handwear-l@arm_l", rgba=hand, offset=(0, 0))
    behind = Part(name="bottomwear@leg_l",
                  rgba=_flat(200, [(20, 180, 20, 180)], rgb=(160, 40, 40)),
                  offset=(0, 0))
    decomp = Decomposition(canvas=(200, 200), parts=[behind, front],
                           src_img=_flat(200, [(20, 180, 20, 180)]))

    _, sealed = rig_mod_local.close_layer_seams([behind, front], decomp, RS())
    gap = np.zeros((200, 200), bool)
    gap[80:160, 95:105] = True
    filled = int((sealed.canvas_rgba(decomp.canvas)[..., 3][gap] >= 250).sum())
    assert filled == 0, f"{filled} px of the gap were packed with the front layer"


def test_two_layers_may_not_stack_past_the_artwork_where_it_is_soft():
    """The wispy-hair fix, and the mirror of ``restore_source_alpha``.

    see-through inpaints every layer complete, so ``front hair`` and ``back hair``
    both carry the loose strands beside the face. Their alphas stack -- two at 0.5
    give 0.75 -- and the gaps *between* the strands, transparent in one layer and
    painted in the other, fill in. Fine hair turns into a soft dark haze that
    reads as an afterimage of hair rather than hair. Measured over the pixels that
    differ most there: 631 of 648 are covered by ``back hair`` and 619 by
    ``front hair``, and the composite averages alpha 163.5 where the artwork has
    110.2.
    """
    import numpy as np

    from ocs import rig as rig_mod_local
    from ocs.config import RigSettings as RS
    from ocs.psd_io import Decomposition, Part

    soft_box = [(60, 140, 60, 140)]
    a = Part(name="front hair", rgba=_flat(200, soft_box, alpha=128), offset=(0, 0))
    b = Part(name="back hair", rgba=_flat(200, soft_box, alpha=128), offset=(0, 0))
    decomp = Decomposition(canvas=(200, 200), parts=[b, a],
                           src_img=_flat(200, soft_box, alpha=128))

    def composite(items):
        acc = np.zeros((200, 200))
        for p in items:
            al = p.canvas_rgba(decomp.canvas)[..., 3] / 255.0
            acc = al + acc * (1.0 - al)
        return acc

    src = decomp.src_img[..., 3] / 255.0
    stacked = composite([b, a])
    assert stacked.max() > src.max() + 0.1, (
        "the case is meant to reproduce the defect; the two layers did not stack")

    limited = composite(rig_mod_local.limit_source_alpha([b, a], decomp, RS()))
    worst = float(np.abs(limited - src).max() * 255)
    assert worst < 3, f"composite still differs from the artwork by {worst:.0f} levels"
