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


def test_all_eight_limb_regions_are_produced(figure):
    rig, kept = _rig_and_parts(figure)
    parts, _report = limbs.partition(figure, kept, rig, RigSettings())
    result = limbs.verify_limb_separation(parts)
    assert result["ok"], result["missing_regions"]
    regions = {p.region for p in parts if p.region}
    for required in taxonomy.MANDATORY_LIMB_REGIONS:
        assert required in regions


def test_single_blob_single_layer_still_separates(blob_figure):
    """Worst case: nothing but the skeleton distinguishes left from right."""
    rig, kept = _rig_and_parts(blob_figure)
    assert len(kept) == 1, "fixture is deliberately a single layer"

    parts, report = limbs.partition(blob_figure, kept, rig, RigSettings())
    result = limbs.verify_limb_separation(parts)
    assert result["ok"], f"missing {result['missing_regions']}; forced={report['forced']}"
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


def test_a_limb_is_one_part_not_two(figure):
    """An arm stays whole; it is *not* cut at the elbow.

    This asserts the opposite of what it used to. Cutting at the joint made the
    two halves separate attachments in separate slots, and that boundary is a hard
    edge in the render whatever the weights say -- 61.8% of the pixels that came
    out darker than the source art sat on a part's own edge. A single mesh spanning
    the joint has no such boundary and still bends, which is what
    ``test_export.test_limb_meshes_span_their_joint`` checks.

    Left/right separation is unaffected, and that is what requirement 2-2 asks
    for; a part per joint segment was an implementation choice on top of it.
    """
    rig, kept = _rig_and_parts(figure)
    parts, _report = limbs.partition(figure, kept, rig, RigSettings())

    for tag, region in (("handwear-r", "arm_r"), ("handwear-l", "arm_l")):
        pieces = [p for p in parts if taxonomy.base_tag(p.name) == "handwear"
                  and p.side == ("right" if region.endswith("_r") else "left")]
        assert pieces, f"no part for {tag}"
        regions = {p.region for p in pieces}
        assert regions == {region}, f"{tag} should be one {region} part, got {regions}"


def test_left_and_right_limbs_are_still_separated(figure):
    """The merge must not weaken requirement 2-2 itself."""
    rig, kept = _rig_and_parts(figure)
    parts, _ = limbs.partition(figure, kept, rig, RigSettings())
    v = limbs.verify_limb_separation(parts, RigSettings())
    assert v["ok"], v
    assert not v["missing_regions"]
    assert v["parts_per_side"]["left"] > 0 and v["parts_per_side"]["right"] > 0


def test_joint_split_is_still_available(figure):
    """Turning the merge off restores the per-segment partition and its check."""
    rig, kept = _rig_and_parts(figure)
    s = RigSettings(merge_limb_slices=False)
    parts, report = limbs.partition(figure, kept, rig, s)
    for source in ("handwear-r", "handwear-l"):
        pieces = report["garment_slices"].get(source)
        assert pieces, f"{source} was not cut at the elbow"
        assert len({taxonomy.part_region(p) for p in pieces}) >= 2
    assert limbs.verify_limb_separation(parts, s)["ok"]


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
