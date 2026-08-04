"""Requirement 2-2: left and right arms/legs separated, unconditionally.

The hard cases are the point of these tests: a single connected silhouette in a
single layer, and a layer covering both legs at once. Neither can be separated by
connected components or by tag names, only by the bone skeleton.
"""

from __future__ import annotations

import numpy as np

from ocs import cleanup, limbs, skeleton, taxonomy
from ocs.config import RigSettings


def _rig_and_parts(decomp):
    reports = cleanup.analyze(decomp)
    kept, _ = cleanup.apply_verdicts(decomp, reports)
    rig = skeleton.guess_rig(decomp, kept)
    return rig, kept


def test_left_and_right_limbs_are_separate_parts(figure):
    """Requirement 2-2. Separation is per side, not per joint segment.

    Cutting at the elbow and knee too is what tore in motion, so bending there is
    the weighted mesh's job now (see RigSettings.slice_limb_spanning).
    """
    rig, kept = _rig_and_parts(figure)
    parts, _report = limbs.partition(figure, kept, rig, RigSettings())
    result = limbs.verify_limb_separation(parts)
    assert result["ok"], result["missing_limbs"]
    assert result["parts_per_side"]["left"] > 0
    assert result["parts_per_side"]["right"] > 0


def test_limbs_are_left_whole_by_default(figure):
    """No upper/lower cut, so nothing can tear at the joint."""
    rig, kept = _rig_and_parts(figure)
    parts, report = limbs.partition(figure, kept, rig, RigSettings())
    assert report["garment_slices"] == {}, "limb layers should not be sliced"
    for p in parts:
        if p.tag in taxonomy.ARM_TAGS + taxonomy.LEG_TAGS:
            assert p.region is None, f"{p.name} was cut into a region"


def test_slicing_is_still_available_when_asked_for(figure):
    """The region cut remains reachable, and still produces all eight regions."""
    s = RigSettings()
    s.slice_limb_spanning = True
    rig, kept = _rig_and_parts(figure)
    parts, report = limbs.partition(figure, kept, rig, s)
    regions = {p.region for p in parts if p.region}
    for required in taxonomy.MANDATORY_LIMB_REGIONS:
        assert required in regions, f"{required} missing; slices={report['garment_slices']}"


def test_uncut_layers_never_overlap_themselves(figure):
    """The ghosting cause: a layer drawn twice over the same pixels.

    Slicing added an 8 px seam allowance to every cut, so each piece overlapped its
    neighbour. Measured on real art the skirt became 7 pieces overlapping itself
    across 31185 px, and in motion the pieces separated and ghosted.
    """
    rig, kept = _rig_and_parts(figure)
    parts, _ = limbs.partition(figure, kept, rig, RigSettings())

    groups: dict[str, list] = {}
    for p in parts:
        src = p.meta.get("sliced_from") or p.meta.get("split_from") or p.name
        groups.setdefault(src, []).append(p)

    for src, pieces in groups.items():
        if len(pieces) < 2:
            continue
        cover = np.zeros(figure.canvas[::-1], np.int16)
        for p in pieces:
            cover += p.canvas_mask(figure.canvas).astype(np.int16)
        doubled = int((cover > 1).sum())
        union = int((cover > 0).sum())
        assert doubled <= 0.01 * max(1, union), (
            f"{src} overlaps itself on {doubled}/{union} px"
        )


def test_single_blob_single_layer_still_separates(blob_figure):
    """Worst case: nothing but the skeleton distinguishes left from right."""
    rig, kept = _rig_and_parts(blob_figure)
    assert len(kept) == 1, "fixture is deliberately a single layer"

    parts, report = limbs.partition(blob_figure, kept, rig, RigSettings())
    result = limbs.verify_limb_separation(parts)
    assert result["ok"], f"missing {result['missing_limbs']}; forced={report['forced']}"
    assert result["parts_per_side"]["left"] > 0
    assert result["parts_per_side"]["right"] > 0


def test_unsplit_legwear_becomes_left_and_right(figure):
    rig, kept = _rig_and_parts(figure)
    parts, report = limbs.partition(figure, kept, rig, RigSettings())
    assert "legwear" in report["lr_split"]
    sides = {p.side for p in parts if p.tag == "legwear"}
    assert sides == {"left", "right"}


def test_sides_are_not_swapped(figure):
    """``-r`` is the character's right, i.e. the *viewer's left* (lower x).

    This matches see-through's own convention in ``label_lr_split``; getting it
    backwards would mirror every animation.
    """
    rig, kept = _rig_and_parts(figure)
    parts, _ = limbs.partition(figure, kept, rig, RigSettings())

    def centre_x(pred):
        xs = [(p.bbox[0] + p.bbox[2]) / 2 for p in parts if pred(p)]
        return sum(xs) / len(xs) if xs else None

    right = centre_x(lambda p: p.side == "right")
    left = centre_x(lambda p: p.side == "left")
    assert right is not None and left is not None
    assert right < left, "character-right parts must sit at lower x than character-left"

    # And the bones agree.
    assert rig.bones["rightArm"][0] < rig.bones["leftArm"][0]
    assert rig.bones["rightLeg"][0] < rig.bones["leftLeg"][0]


def test_limb_layers_survive_intact(figure):
    """An arm stays one attachment; the elbow bend comes from weights, not a cut."""
    rig, kept = _rig_and_parts(figure)
    parts, _report = limbs.partition(figure, kept, rig, RigSettings())
    for source in ("handwear-r", "handwear-l"):
        matching = [p for p in parts if p.name == source]
        assert len(matching) == 1, f"{source} should survive as exactly one part"


def test_semantic_regions_prevent_cross_limb_bleed(figure):
    """A leg layer must never end up in an arm region.

    Pure nearest-segment assignment does exactly that when the arms hang beside
    the thighs, which is why ``taxonomy.allowed_regions`` exists.
    """
    rig, kept = _rig_and_parts(figure)
    parts, _ = limbs.partition(figure, kept, rig, RigSettings())
    for p in parts:
        if p.tag == "legwear":
            assert p.region is None or p.region.startswith("leg_"), p.name
        if p.tag == "handwear":
            assert p.region is None or p.region.startswith("arm_"), p.name


def test_partition_covers_the_silhouette_without_losing_pixels(figure):
    """Slicing must not silently discard part of a layer.

    see-through's own ``part_lr_split`` keeps only the two largest components and
    drops the rest; OCS must not. Accounted per *tag*, because a layer can go
    through two steps (legwear -> legwear-l/-r -> ...@leg_l_upper), so tracking a
    single ``sliced_from`` link would miss the L/R hop.
    """
    rig, kept = _rig_and_parts(figure)
    parts, _ = limbs.partition(figure, kept, rig, RigSettings())

    def union(items):
        acc = np.zeros(figure.canvas[::-1], bool)
        for p in items:
            acc |= p.canvas_mask(figure.canvas)
        return acc

    tags = {p.tag for p in kept if p.tag in taxonomy.LIMB_SPANNING_TAGS}
    assert tags, "fixture should contain limb-spanning layers"
    for tag in tags:
        before = union([p for p in kept if p.tag == tag])
        after = union([p for p in parts if p.tag == tag])
        assert int(after.sum()) >= int(before.sum()) * 0.98, f"{tag} lost pixels"


def test_region_labels_only_inside_the_silhouette(figure):
    rig, kept = _rig_and_parts(figure)
    sil = skeleton.build_silhouette(figure, kept)
    labels, specs = limbs.region_labels(rig, sil)
    assert len(specs) == len(taxonomy.SKIN_REGIONS)
    assert (labels[~sil] == limbs.UNASSIGNED).all()
    assert (labels[sil] >= 0).all()
