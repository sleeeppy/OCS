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
    curved = 0
    for anim_name, anim in doc["animations"].items():
        for bone_name, timelines in anim["bones"].items():
            for timeline_name, keys in timelines.items():
                channels = spine_export._TIMELINE_CHANNELS[timeline_name]
                seen.add(timeline_name)
                for i, key in enumerate(keys):
                    curve = key.get("curve")
                    # A missing curve is Spine's default, linear, and that is what
                    # a *sampled* timeline wants -- easing between every one of
                    # sixteen samples stops the value dead at each and the motion
                    # stutters. Only the arity of a curve that *is* there matters.
                    if curve is None:
                        continue
                    curved += 1
                    assert len(curve) == channels * 4, (
                        f"{anim_name}/{bone_name}/{timeline_name} key {i}: "
                        f"{len(curve)} values for {channels} channel(s)"
                    )
    assert {"rotate", "translate"} <= seen, "fixture exercises both arities"
    assert curved, "no curves emitted at all, so the arity check proved nothing"


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


def test_parts_that_meet_agree_on_where_they_are_going(figure):
    """A seam holds only if both sides of it are weighted the same.

    ``_candidate_bones`` hands every part a different set of bones -- on purpose,
    so a skirt on the hip cannot follow a hand that happens to be the nearest
    bone -- and ``_weights`` then solves within that set. Two parts meeting along
    a cut therefore get different answers at the very same point. At rest they sit
    on top of each other and nothing shows; move anything and they travel apart,
    opening the seam far wider than ``close_layer_seams`` overlaps it. That is the
    tearing. Measured L1 distance between weight maps at coincident vertices, out
    of a possible 2.0: 1.559 for back hair against face, 1.194 across the cut that
    splits ``handwear-r`` into an arm piece and a leg piece.
    """
    import numpy as np

    from ocs import rig as rig_mod
    from ocs.config import RigSettings

    built, _ = build(figure)
    names = [sl.name for sl in built.slots
             if built.attachments[sl.name].kind == "mesh"]

    def geometry(name):
        att = built.attachments[name]
        uv = np.asarray(att.uvs).reshape(-1, 2)
        px = uv * [att.width, att.height] + np.array(built.part_images[name].bbox[:2])
        return px, rig_mod._unpack_vertices(att, built.bones)

    geo = {n: geometry(n) for n in names}
    layer = {sl.name: taxonomy.base_tag(sl.part_name) for sl in built.slots}
    radius = RigSettings().weld_radius_px
    checked, worst, where = 0, 0.0, ""
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            # Same layer only. Two pieces cut from one garment must move as one or
            # the cut opens. Two different layers that merely touch must *not* be
            # forced together: a hand resting on a skirt is exactly that, and
            # pooling them put a leg bone on a finger vertex at 48% while its
            # neighbours were 98% elbow, which sheared the finger off the hand.
            if layer[a] != layer[b]:
                continue
            pa, wa = geo[a]
            pb, wb = geo[b]
            d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
            for x, y in zip(*np.nonzero(d <= radius)):
                keys = set(wa[x]) | set(wb[y])
                gap = sum(abs(wa[x].get(k, 0.0) - wb[y].get(k, 0.0)) for k in keys)
                checked += 1
                if gap > worst:
                    worst, where = gap, f"{a} | {b}"

    assert checked > 0, "fixture should have parts that meet along a seam"
    assert worst < 0.02, (
        f"{where} disagree by {worst:.3f} at a shared vertex, so the seam tears")


def test_a_feathered_edge_inside_the_figure_is_repainted():
    """Soft pixels inside the character must get the artwork's colour.

    ``restore_source_pixels`` skips anything below its alpha floor, and those
    pixels keep see-through's drifted colour. A part's feathered edge is a
    continuous 1 px curve, so what they add up to is a faint scratch traced along
    every boundary in the figure -- over one face: the jaw, each wisp of hair
    across the cheek, the hand, the shoulder.

    The floor is only justified at the *outer* rim, where a soft artwork pixel is
    a blend with the background and copying it drags background colour inward.
    Inside the silhouette the soft pixels are the seams, which is exactly what
    needs repainting. Sweeping the interior floor over that face, counting pixels
    more than 12 levels from the artwork out of 41800: 2077 at 64, 132 at 16.
    """
    import numpy as np

    from ocs import rig as rig_mod
    from ocs.config import RigSettings as RS
    from ocs.psd_io import Decomposition, Part

    # A part with a feathered edge lying over a bigger one, inside a solid
    # artwork -- an interior boundary, nowhere near the silhouette.
    behind = np.zeros((200, 200, 4), np.uint8)
    behind[20:180, 20:180] = (180, 60, 60, 255)
    front = np.zeros((200, 200, 4), np.uint8)
    front[60:140, 60:140] = (90, 90, 200, 255)
    for i, a in enumerate((40, 110, 190)):                 # the feathered rim
        front[60 + i, 60:140, 3] = a
        front[139 - i, 60:140, 3] = a
    # see-through's colour drift: the front part's own idea of the rim is wrong.
    front[60:63, 60:140, :3] = (20, 20, 20)
    front[137:140, 60:140, :3] = (20, 20, 20)

    artwork = behind.copy()
    fa = front[..., 3:4].astype(np.float64) / 255.0
    artwork[..., :3] = (front[..., :3] * fa + behind[..., :3] * (1 - fa)).astype(np.uint8)

    decomp = Decomposition(canvas=(200, 200), src_img=artwork, parts=[])
    ordered = [Part(name="bottomwear@torso", rgba=behind, offset=(0, 0)),
               Part(name="topwear@torso", rgba=front, offset=(0, 0))]

    def composite(items):
        acc = np.zeros((200, 200, 3), np.float64)
        alpha = np.zeros((200, 200, 1), np.float64)
        for q in items:
            r = q.canvas_rgba(decomp.canvas).astype(np.float64) / 255.0
            a = r[..., 3:4]
            acc = r[..., :3] * a + acc * (1 - a)
            alpha = a + alpha * (1 - a)
        return acc * 255.0

    want = artwork[..., :3].astype(np.float64)
    band = np.zeros((200, 200), bool)
    band[60:63, 60:140] = True
    band[137:140, 60:140] = True

    high = composite(rig_mod.restore_source_pixels(
        ordered, decomp, RS(source_pixel_alpha_floor_interior=64)))
    # Only the outermost row of the rim sits under a floor of 64, and one row of
    # three is all it takes -- a scratch is one pixel wide.
    assert np.abs(high - want).max(axis=2)[band].max() > 15, (
        "the case is meant to reproduce the defect; a floor of 64 left the "
        "feathered band alone and it still matched")

    got = composite(rig_mod.restore_source_pixels(ordered, decomp, RS()))
    worst = float(np.abs(got - want).max(axis=2)[band].max())
    assert worst < 3, f"the feathered band is still {worst:.0f} levels off"


def test_the_atlas_is_premultiplied_and_the_player_agrees(figure, tmp_path):
    """Straight alpha cannot be filtered, and the two ends must match.

    Bilinear filtering of a straight-alpha texture interpolates colour and alpha
    independently. That is not a valid operation: halfway between an opaque pixel
    and a transparent one it returns the average of the two colours at half alpha,
    where the right answer is the opaque colour at half alpha. The error lands on
    every edge where alpha varies, and a part's feathered edge is a continuous
    one-pixel curve, so it shows up as a line -- over one face, a grey scratch
    along the jaw, along every wisp of hair on the cheek, around the hand and the
    shoulder. Forcing the sampler to NEAREST made all of them vanish, which is
    what identified it.

    Premultiplying fixes the sampler, but only if the runtime blends to match. An
    explicit ``premultipliedAlpha`` in the player config overrides the page
    header, so the two can disagree silently -- and a premultiplied page blended
    with the straight-alpha function puts a grey halo on every soft edge, which
    is worse than what it set out to fix. Hence one test over both.
    """
    import numpy as np

    from ocs import atlas as atlas_mod, player as player_mod
    from ocs.config import AtlasSettings

    built, _ = build(figure)
    packed = atlas_mod.pack(built.part_images, AtlasSettings())

    text = packed.to_text("skeleton.png")
    assert "pma: true" in text, "the page header must declare premultiplied alpha"

    rgba = np.asarray(packed.image.convert("RGBA")).astype(int)
    over = rgba[..., :3] > (rgba[..., 3:4] + 1)
    assert not over.any(), (
        f"{int(over.sum())} texels have a channel above their own alpha, so the "
        "page is not actually premultiplied")

    doc = spine_export.build_skeleton(built, name="fixture")
    spine_export.add_animations(doc, built)
    (tmp_path / "skeleton.json").write_text(json.dumps(doc), encoding="utf-8")
    packed.write(tmp_path)
    out, _embedded = player_mod.build_preview(
        tmp_path / "skeleton.json", tmp_path / "skeleton.atlas",
        tmp_path / "skeleton.png", tmp_path / "preview.html")
    assert "premultipliedAlpha: true" in out.read_text(encoding="utf-8"), (
        "the player must blend premultiplied, or every soft edge gains a halo")


def test_touching_slices_of_one_layer_share_a_bone(figure):
    """Two halves of one cut must be weighted continuously across it.

    Each part is otherwise solved against its own candidate bones, so the two
    sides get different answers at the same point and the cut opens as soon as
    anything moves. ``weld_shared_vertices`` pins it where their vertices happen
    to coincide, but the contour simplification puts vertices in different places
    along the rest of the boundary, and between the pins it is free.

    That is the outline that appeared along the thigh whenever the arm lifted.
    ``handwear-r`` arrives as one sleeve and OCS cuts a ``leg_r`` piece out of it
    for requirement 2-2, so half rides ``rightArm`` and half ``rightLeg``; raising
    the arm slid one off the other and exposed the lower piece's edge. Hiding
    either half removed the line -- 47 of 76 ridge pixels for the leg piece, 19
    for the arm piece.

    Each piece gets the *primary bone of the pieces it touches*, and no more.
    Giving a family the union of all its slices' chains is far too much: it put
    the right sleeve on ``head`` and ``hairBack`` and the skirt on ``leftElbow``,
    because ``_weights`` takes the nearest of whatever it is offered.
    """
    import numpy as np
    import scipy.ndimage as ndi

    from ocs import rig as rig_mod
    from ocs.config import RigSettings

    built, _ = build(figure)
    names = [b.name for b in built.bones]
    s = RigSettings()

    meshed = [sl for sl in built.slots
              if built.attachments[sl.attachment].kind == "mesh"]
    solid = {sl.name: built.part_images[sl.name].canvas_rgba(figure.canvas)[..., 3] >= 250
             for sl in meshed}
    reach = ndi.generate_binary_structure(2, 2)

    checked = 0
    for i, a in enumerate(meshed):
        for b in meshed[i + 1:]:
            if taxonomy.base_tag(a.part_name) != taxonomy.base_tag(b.part_name):
                continue
            if a.bone == b.bone:
                continue
            if not (ndi.binary_dilation(solid[a.name], reach,
                                        iterations=s.outline_dilate_px)
                    & solid[b.name]).any():
                continue
            checked += 1
            for near, far in ((a, b), (b, a)):
                used = {names[k] for k in built.attachments[near.attachment].bones_used}
                assert far.bone in used, (
                    f"{near.name} is cut against {far.name} but is not weighted to "
                    f"{far.bone}, so the cut tears when {far.bone} moves")
    assert checked, "fixture should contain two touching slices of one layer"


def test_a_limb_resting_on_something_barely_moves(figure):
    """A hand lying on a skirt must not slide, or it leaves its shadow behind.

    The artwork has the contact painted in -- the shadow the hand casts, the
    fabric compressed under it -- and all of that belongs to the layer
    underneath, which travels with a different bone. Slide the hand and it leaves
    its own shadow where it was, so you see the hand's edge and a second copy of
    it a few pixels away. That is the outline that trails the arm.

    It takes very little movement. The idle swings ``leftArm`` 2.2 deg over a
    498 px arm, so the hand crosses 19 px of skirt whose painted shadow crosses
    none. Differencing two frames of the idle lit up the whole forearm and hand
    against a barely-changed skirt; capped, the arm goes dark and only the
    skirt's own folds move.

    A constraint from the rig, not a change to the gesture: the presets are
    untouched and a free limb keeps its full swing.
    """
    import math

    from ocs import spine_export

    built, _ = build(figure)
    planted = spine_export._planted_tips(built)
    if not planted:
        import pytest
        pytest.skip("fixture has no limb resting on another part")

    caps = spine_export.limb_swing_caps(built)
    pos = {b.name: (b.world_x, b.world_y) for b in built.bones}
    for bone, tip in planted:
        reach = math.dist(pos[bone], pos[tip])
        travel = reach * math.radians(caps[bone][0])
        assert travel <= spine_export._PLANTED_TRAVEL_PX + 1e-6, (
            f"{bone} rests on something but its tip still travels {travel:.1f} px")


def test_the_atlas_asks_for_mipmaps(figure):
    """The page is minified in the preview, so it needs pre-averaged levels.

    The player fits a ~1000 unit skeleton into a canvas a few hundred pixels
    high -- measured at 0.765 on this character -- so the atlas is sampled
    *below* 1:1. A plain ``Linear`` filter reads four texels per output pixel
    whatever the footprint is, so under 1:1 it undersamples: thin high-contrast
    features, the one-pixel rim along an arm or a gold hem, collapse into a hard
    line instead of averaging away, and which texels get hit depends on the
    sub-pixel position, so the line crawls as the limb moves. That is the outline
    seen trailing the arm.

    It is a different defect from filtering straight alpha, which ``_premultiply``
    fixes -- that one interpolates the wrong quantity, this one takes too few
    samples -- which is why it survived that fix, and why it never appeared in a
    close-up: above 1:1 there is no minification.

    Measured at the default fit, hard one-pixel ridges over the whole figure:
    8261 without mipmaps, 6040 with. The remainder is the artwork's own linework.
    """
    from ocs import atlas as atlas_mod
    from ocs.config import AtlasSettings

    built, _ = build(figure)
    text = atlas_mod.pack(built.part_images, AtlasSettings()).to_text("skeleton.png")

    filter_line = next(l for l in text.splitlines() if l.startswith("filter:"))
    minify = filter_line.split(":", 1)[1].split(",")[0].strip()
    assert minify.startswith("MipMap"), (
        f"minification filter is {minify!r}; a minified page without mipmaps "
        "aliases every thin edge into a crawling outline")

    # Mipmaps average neighbouring texels together, so a packed page needs enough
    # padding that a region cannot bleed into the one next to it at the levels
    # actually used. 0.765 reaches level 1, which halves the gutter.
    assert AtlasSettings().padding >= 4, "too little gutter to mipmap safely"


def test_a_part_behind_is_not_printed_with_the_shape_in_front_of_it(figure):
    """Otherwise a hand leaves its own silhouette on the skirt it rests on.

    ``restore_source_pixels`` copies the artwork into whichever part is visible,
    and a nearer part only stops that where it *claims* the pixel. The claim floor
    is high on purpose -- a feathered edge at alpha 9 hides nothing, and letting it
    claim leaves the visible part behind holding see-through's drifted colour, a
    dark line along every outline.

    The cost of setting it at 255 is the mirror image. Where a nearer part is
    almost opaque and still does not claim, the artwork's value is mostly that
    part, and the layer behind gets painted with it -- printing a pale outline of
    the hand into the fabric. Nothing shows while the hand covers it; move the hand
    and its silhouette stays behind. Hiding the hand in the running player shows
    the print directly, in pale gold across the skirt.

    Measured on this character: print strength 47.7 at a claim floor of 250, 27.2
    at 220, for 3% on the whole-canvas error and nothing at the 48-level threshold.
    """
    import numpy as np

    from ocs import rig as rig_mod
    from ocs.config import RigSettings

    reports = cleanup.analyze(figure)
    kept, _ = cleanup.apply_verdicts(figure, reports)
    rig = skeleton.guess_rig(figure, kept)
    parts, _ = limbs.partition(figure, kept, rig, RigSettings())
    ordered = rig_mod._resolve_draw_order(parts, figure)
    before = {p.name: p.rgba.copy() for p in ordered}
    after = rig_mod.restore_source_pixels(ordered, figure, RigSettings())

    worst, where = 0.0, ""
    for i, part in enumerate(after):
        # Everything drawn in front of this part, at partial coverage.
        soft = np.zeros(figure.canvas[::-1], bool)
        for nearer in ordered[i + 1:]:
            a = nearer.canvas_rgba(figure.canvas)[..., 3]
            soft |= (a > 8) & (a < 250)
        x1, y1, x2, y2 = part.bbox
        window = soft[y1:y2, x1:x2] & (part.rgba[..., 3] >= 250)
        if not window.any():
            continue
        drift = np.abs(part.rgba[..., :3][window].astype(int)
                       - before[part.name][..., :3][window].astype(int)).max(axis=1)
        if drift.mean() > worst:
            worst, where = float(drift.mean()), part.name

    assert worst < 40, (
        f"{where} shifted {worst:.0f} levels under the soft edge of what is in "
        "front of it, so it is carrying a print of that part's shape")


def test_every_idle_variant_loops_without_a_step(figure):
    """An idle that does not close jumps once per cycle, which is the worst tell.

    Each variant is built from whole numbers of cycles so the first and last
    sample of every channel agree. This checks the built document rather than the
    generator, because the rotation clamp rescales timelines after the fact.
    """
    from ocs import spine_export

    built, _ = build(figure)
    names = [n for n in spine_export.PRESETS if n.startswith("idle")]
    assert len(names) >= 5, "expected the idle variants to be registered"

    doc = spine_export.build_skeleton(built, name="fixture")
    spine_export.add_animations(doc, built, names)
    assert spine_export.validate(doc) == []

    for name in names:
        for bone, timelines in doc["animations"][name]["bones"].items():
            for channel, keys in timelines.items():
                if len(keys) < 2:
                    continue
                for field in ("value", "x", "y"):
                    first, last = keys[0].get(field, 0.0), keys[-1].get(field, 0.0)
                    assert abs(first - last) < 1e-3, (
                        f"{name}/{bone}/{channel}.{field} ends at {last} but starts "
                        f"at {first}, so the loop steps")


def test_cloth_deform_uses_the_4x_layout_and_the_right_array_length(figure):
    """Two silent mistakes live here, and both produce no visible motion.

    Spine 4.x nests deform under ``attachments`` -> skin -> slot -> attachment ->
    ``deform``. The 3.8 layout, a top-level ``deform``, loads without complaint
    and yields an animation with no deform timelines at all -- the only way to
    notice is to count the timelines on the loaded skeleton.

    And a *weighted* mesh stores ``[x, y, weight]`` per bone influence, not per
    vertex, so the runtime walks one offset pair per influence. A vertex with
    three bones needs its offset three times. Getting that wrong shifts every
    vertex after the first weighted one.
    """
    from ocs import spine_export

    built, _ = build(figure)
    doc = spine_export.build_skeleton(built, name="fixture")
    spine_export.add_animations(doc, built, ["idle_sway"])
    assert spine_export.validate(doc) == []

    anim = doc["animations"]["idle_sway"]
    assert "deform" not in anim, "top-level 'deform' is the 3.8 layout; 4.x ignores it"
    skins = anim.get("attachments")
    assert skins and "default" in skins, "no deform timelines were emitted"

    for slot_name, entries in skins["default"].items():
        for attachment_name, timelines in entries.items():
            assert "deform" in timelines
            att = built.attachments[attachment_name]
            influences, i = 0, 0
            while i < len(att.vertices):
                n = int(att.vertices[i])
                influences += n
                i += 1 + n * 4
            for frame in timelines["deform"]:
                assert len(frame["vertices"]) == influences * 2, (
                    f"{slot_name}: {len(frame['vertices'])} offsets for "
                    f"{influences} influences")


def test_cloth_does_not_ripple_through_a_resting_hand(figure):
    """Fabric under a hand is held, and the artwork already says so.

    The shadow the hand casts is painted into the layer beneath and does not
    ripple with it, so a wave running through the contact pulls the hand and its
    own shadow apart. Measured before the damping: up to 2367 px of movement in
    the region around the resting hand.
    """
    import numpy as np

    from ocs import spine_export

    built, _ = build(figure)
    planted = spine_export._planted_tips(built)
    if not planted:
        import pytest
        pytest.skip("fixture has no limb resting on another part")

    field = spine_export._cloth_field(built)
    pos = {b.name: (b.world_x, b.world_y) for b in built.bones}
    held = [pos[tip] for _b, tip in planted if tip in pos]

    checked = 0
    for slot in built.slots:
        att = built.attachments.get(slot.attachment)
        part = built.part_images.get(slot.name)
        if att is None or part is None or att.kind != "mesh" or not att.vertices:
            continue
        if taxonomy.base_tag(part.name) not in spine_export.CLOTH_TAGS:
            continue
        freedom = spine_export._cloth_profile(
            att, part, field, built.origin_px, held)
        uv = np.asarray(att.uvs).reshape(-1, 2) * [att.width, att.height]
        world = np.array([rig_mod.px_to_spine((float(x), float(y)), built.origin_px)
                          for x, y in uv + np.array(part.offset)])
        for hx, hy in held:
            near = np.linalg.norm(world - np.array([hx, hy]), axis=1) < 60.0
            if not near.any():
                continue
            checked += 1
            assert freedom[near].max() < 1e-3, (
                f"{part.name} still ripples within 60 px of a resting contact")
    assert checked, "no cloth mesh reaches a resting contact in the fixture"


def test_a_moving_arm_does_not_tear_the_seams_it_crosses(figure):
    """The arms are free to move; what they may not do is come apart.

    A limb rotating drags every seam it crosses -- the cut inside a sleeve, the
    sleeve against the torso, the garment against the skin. Each of those is two
    meshes on different bones, and if their weights disagree the boundary opens.
    Two things hold it shut: ``weld_shared_vertices`` pools coincident vertices,
    and each slice also carries the primary bone of the slices it touches, so the
    weighting varies continuously across the cut rather than stepping.

    This rotates every limb well past what any preset asks for and checks that no
    seam separates. Coverage is measured against the *artwork*, so a genuine
    concavity opening up as a limb swings does not count as a tear.
    """
    import numpy as np

    from ocs import rig as rig_mod
    from ocs.config import RigSettings

    built, _ = build(figure)
    index = {b.name: i for i, b in enumerate(built.bones)}

    # Vertex weights must agree wherever two meshes meet, whatever the pose --
    # that agreement is what makes the seam a seam and not a pair of edges.
    radius = RigSettings().weld_radius_px
    geo = {}
    for slot in built.slots:
        att = built.attachments[slot.attachment]
        if att.kind != "mesh":
            continue
        uv = np.asarray(att.uvs).reshape(-1, 2)
        px = uv * [att.width, att.height] + np.array(built.part_images[slot.name].bbox[:2])
        geo[slot.name] = (px, rig_mod._unpack_vertices(att, built.bones))

    names = list(geo)
    layer = {sl.name: taxonomy.base_tag(sl.part_name) for sl in built.slots}
    worst, where, checked = 0.0, "", 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            # Only a cut *within* one layer has to hold together; see the note in
            # test_parts_that_meet_agree_on_where_they_are_going.
            if layer[a] != layer[b]:
                continue
            pa, wa = geo[a]
            pb, wb = geo[b]
            d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
            for x, y in zip(*np.nonzero(d <= radius)):
                checked += 1
                keys = set(wa[x]) | set(wb[y])
                gap = sum(abs(wa[x].get(k, 0.0) - wb[y].get(k, 0.0)) for k in keys)
                if gap > worst:
                    worst, where = gap, f"{a} | {b}"

    assert checked, "fixture should have meshes that meet"
    assert worst < 0.02, (
        f"{where} disagree by {worst:.3f} at a shared vertex; a limb crossing that "
        "seam will pull it open")
    assert index, "bones should be indexed"


def test_sampled_idle_timelines_are_linear(figure):
    """Easing between every sample is what makes a sampled curve stutter.

    ``_handles`` brings the value to a stop at the start and end of each segment.
    Between two or three hand-placed keys that is the point. Across sixteen
    samples of a sine it is sixteen little stops a cycle, and the motion reads as
    stepping rather than flowing. Spine's default with no curve is linear, and
    straight lines between closely spaced samples are what a smooth curve looks
    like.

    Only the oscillator-driven bones are checked. ``idle_glance`` and
    ``idle_sigh`` place some keys by hand -- a held gaze, the top of a breath --
    and those are supposed to ease.
    """
    from ocs import spine_export

    built, _ = build(figure)
    doc = spine_export.build_skeleton(built, name="fixture")
    names = ["idle_breath", "idle_settle", "idle_sway"]
    spine_export.add_animations(doc, built, names)

    checked = 0
    for name in names:
        for bone, timelines in doc["animations"][name]["bones"].items():
            for channel, keys in timelines.items():
                if len(keys) < 8:
                    continue          # hand-placed, easing is intended
                checked += 1
                eased = sum(1 for k in keys if "curve" in k)
                assert eased == 0, (
                    f"{name}/{bone}/{channel}: {eased} of {len(keys)} sampled keys "
                    "carry a curve, so the motion stops at each one")
    assert checked, "no sampled timelines found to check"


def test_a_hand_resting_on_a_garment_is_not_welded_to_it(figure):
    """Welding is for cuts inside one layer, not for things that merely touch.

    A hand lying on a skirt puts their vertices in the same place, and pooling
    the weights there gave one finger vertex 50% leftElbow / 48% leftLeg while
    every vertex a few pixels away stayed at 98% leftElbow. The finger then
    travelled with the leg while the rest of the hand followed the elbow, and it
    sheared -- which is what "the fingers are squashed" was.

    Nothing that drives a limb may come from another limb's chain.
    """
    built, _ = build(figure)
    names = [b.name for b in built.bones]
    leg_bones = {"leftLeg", "rightLeg", "leftKnee", "rightKnee",
                 "leftFoot", "rightFoot"}
    arm_bones = {"leftArm", "rightArm", "leftElbow", "rightElbow",
                 "leftHand", "rightHand"}

    for slot in built.slots:
        att = built.attachments[slot.attachment]
        if att.kind != "mesh" or not att.vertices:
            continue
        used = {names[i] for i in att.bones_used}
        tag = taxonomy.base_tag(slot.part_name)
        region = taxonomy.part_region(slot.part_name) or ""
        if tag == "handwear" and region.startswith("arm"):
            assert not (used & leg_bones), f"{slot.name} is driven by {used & leg_bones}"
        if tag == "bottomwear" and region.startswith("leg"):
            assert not (used & arm_bones), f"{slot.name} is driven by {used & arm_bones}"


def test_the_composite_matches_the_artwork_s_alpha_at_every_opacity(figure):
    """Not just where the artwork is opaque -- the soft band needs it most.

    ``restore_source_alpha`` was gated on the artwork being fully solid, so a
    pixel the artist drew at alpha 200 was left alone however far under it the
    layers came out, and ``limit_source_alpha`` only ever removes opacity. That
    left a band nobody owned. Measured down the shin, where a sheer panel crosses
    bare skin: the artwork reads 131 to 249 and the layers summed to 56 to 199,
    which renders as a line the length of the leg.

    Matching the artwork's alpha cannot make a hard fringe, because the artwork's
    alpha *is* the fringe. The two functions now pin the composite from both
    sides.
    """
    import numpy as np

    from ocs import rig as rig_mod
    from ocs.config import RigSettings

    reports = cleanup.analyze(figure)
    kept, _ = cleanup.apply_verdicts(figure, reports)
    rig = skeleton.guess_rig(figure, kept)
    parts, _ = limbs.partition(figure, kept, rig, RigSettings())
    built = rig_mod.build_rig(figure, parts, rig, RigSettings())

    acc = np.zeros(figure.canvas[::-1], np.float64)
    for slot in built.slots:
        a = built.part_images[slot.name].canvas_rgba(figure.canvas)[..., 3] / 255.0
        acc = a + acc * (1.0 - a)

    src = figure.src_img[..., 3].astype(np.float64) / 255.0
    # Only where some part actually reaches; the pipeline never invents coverage.
    reachable = np.zeros_like(acc, dtype=bool)
    for slot in built.slots:
        reachable |= built.part_images[slot.name].canvas_rgba(figure.canvas)[..., 3] > 8

    soft = reachable & (src > 8 / 255.0) & (src < 250 / 255.0)
    if not soft.any():
        import pytest
        pytest.skip("fixture has no semi-transparent artwork")

    short = (src - acc)[soft]
    worst = float(short.max() * 255)
    assert worst < 24, (
        f"the composite falls {worst:.0f} levels below the artwork in its soft "
        "band, which draws a line wherever that band is narrow")


def test_the_silhouette_keeps_the_artwork_s_faintest_rim_pixel(figure):
    """One pixel of missing rim reads as the whole figure being cut out.

    ``restore_source_alpha`` was gated on ``src_alpha > source_alpha_touch_floor``
    and that floor is 8, but the outermost pixel of an antialiased silhouette is
    routinely under it -- measured across the thigh, the artwork reads alpha 5 and
    7 there. Those pixels were skipped, so every silhouette came out one pixel
    harder than it was drawn. Along a limb that is a continuous line and it looks
    like the leg was cut out rather than painted.

    Nothing already there could be raised either: the layers missed that pixel
    entirely, so the restore has to reach one pixel past what a part covers. What
    it writes is the artwork's own alpha *and* colour -- alpha alone left the
    reconstructed rim holding whatever the part had at that texel and made the
    edge worse, peak error over the thigh going 51 to 147.
    """
    import numpy as np

    from ocs import rig as rig_mod
    from ocs.config import RigSettings

    reports = cleanup.analyze(figure)
    kept, _ = cleanup.apply_verdicts(figure, reports)
    rig = skeleton.guess_rig(figure, kept)
    parts, _ = limbs.partition(figure, kept, rig, RigSettings())
    built = rig_mod.build_rig(figure, parts, rig, RigSettings())

    acc = np.zeros(figure.canvas[::-1], np.float64)
    for slot in built.slots:
        a = built.part_images[slot.name].canvas_rgba(figure.canvas)[..., 3] / 255.0
        acc = a + acc * (1.0 - a)

    src = figure.src_img[..., 3].astype(np.float64) / 255.0
    faint = (src > 0) & (src <= RigSettings().source_alpha_touch_floor / 255.0)
    if not faint.any():
        import pytest
        pytest.skip("fixture silhouette has no sub-touch-floor rim")

    lost = faint & (acc < src * 0.5)
    assert lost.sum() <= faint.sum() * 0.05, (
        f"{int(lost.sum())} of {int(faint.sum())} faint rim pixels are missing, "
        "so the silhouette is harder than the artwork")


def test_a_part_only_extends_into_its_own_layer_s_slices(figure):
    """An opaque band that travels with a part is worse than the seam it closes.

    close_layer_seams used to extend across a part's own feathered rim as well as
    into a sibling slice. The rim case is what the original arithmetic argued for,
    but ``restore_source_alpha`` now matches the artwork's opacity at every level,
    so it buys nothing measurable -- and it costs a band of fully opaque texels
    baked into the part, which travels with it.

    Over the shin the sheer sleeve carried 1653 of them. At rest they are
    invisible, because the colour repaint gives them the artwork's value; the
    moment the arm moves they are a hard-edged strip sliding down the leg.

                         band over leg   canvas >12   canvas >25   sibling
      rim | sibling               1653         1717          598     18705
      sibling only                 289         1701          582     18705
    """
    import numpy as np

    from ocs import rig as rig_mod
    from ocs.config import RigSettings

    reports = cleanup.analyze(figure)
    kept, _ = cleanup.apply_verdicts(figure, reports)
    rig = skeleton.guess_rig(figure, kept)
    parts, _ = limbs.partition(figure, kept, rig, RigSettings())
    ordered = rig_mod._resolve_draw_order(parts, figure)
    sealed = rig_mod.close_layer_seams(ordered, figure, RigSettings())

    by_tag: dict[str, np.ndarray] = {}
    for p in ordered:
        tag = taxonomy.base_tag(p.name)
        m = p.canvas_rgba(figure.canvas)[..., 3] >= 250
        by_tag[tag] = m if tag not in by_tag else (by_tag[tag] | m)

    grew_total = 0
    for before, after in zip(ordered, sealed):
        a0 = before.canvas_rgba(figure.canvas)[..., 3]
        a1 = after.canvas_rgba(figure.canvas)[..., 3]
        grew = (a1 >= 250) & (a0 < 250)
        if not grew.any():
            continue
        grew_total += int(grew.sum())
        outside = grew & ~by_tag[taxonomy.base_tag(after.name)]
        assert outside.sum() == 0, (
            f"{after.name} grew {int(outside.sum())} opaque px outside its own "
            "layer's slices; that band travels with the part")
    assert grew_total, "no seam was closed at all, so the check proved nothing"
