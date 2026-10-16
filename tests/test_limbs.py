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


def test_reslicing_replaces_the_region_instead_of_appending():
    """A part can be cut twice, and the second cut must not stack suffixes.

    _slice_by_regions assigns a region, then enforce_limb_coverage may carve a
    mandatory region out of the result. Appending produced
    ``bottomwear@leg_r_upper@leg_r``, and every reader splits on the *first*
    separator, so the region became ``leg_r_upper@leg_r`` -- matching no spec.
    The damage is silent: part_side goes None so the piece drops out of the
    left/right tally, and bone_for_part falls back to torso, binding a piece of
    skirt lying over the shin to the trunk so it slides across the leg whenever
    the torso moves.
    """
    naming = taxonomy.PartNaming()
    once = naming.garment("bottomwear", "leg_r_upper")
    twice = naming.garment(once, "leg_r")

    assert twice.count(taxonomy.REGION_SEP) == 1, twice
    assert taxonomy.part_region(twice) == "leg_r"
    assert taxonomy.part_side(twice) == "right"

    bones = {b.name for b in taxonomy.BONE_TEMPLATE}
    assert taxonomy.bone_for_part(twice, bones) == "rightLeg"


def test_reslicing_keeps_the_left_right_suffix():
    """The LR suffix is what part_side reads for a layer see-through split."""
    naming = taxonomy.PartNaming()
    twice = naming.garment(naming.garment("legwear-l", "leg_l_upper"), "leg_l")
    assert twice == "legwear-l@leg_l"
    assert taxonomy.part_side(twice) == "left"
    assert taxonomy.base_tag(twice) == "legwear"


def test_layer_dir_skips_see_throughs_intermediates(tmp_path):
    """``head.png`` is a crop the second pass runs on, not an output layer.

    It is in BODY_PASS_TAGS but deliberately absent from PART_TAGS, and
    ``further_extr`` never puts it in the PSD. read_layer_dir imported every PNG,
    so the recovery path picked it up -- and with no TAG_TO_BONE entry it bound to
    ``torso``, giving a whole-head rigid quad riding the trunk.
    """
    import numpy as np
    from PIL import Image
    from ocs import psd_io

    for stem in ("src_img", "src_head", "head", "face", "topwear"):
        a = np.zeros((32, 32, 4), np.uint8)
        a[8:24, 8:24] = (200, 180, 170, 255)
        Image.fromarray(a).save(tmp_path / f"{stem}.png")

    names = {q.name for q in psd_io.read_layer_dir(tmp_path).parts}
    assert names == {"face", "topwear"}, names


def test_a_paired_tag_arriving_unsplit_is_still_split():
    """Whether handwear arrives split is a property of the run, not the tag.

    --tblr_split is applied by further_extr, the last step of an inference, so an
    interrupted run yields both sleeves in one layer. Keying the decision off
    OCS_LR_TAGS alone left it whole, and the bone partition then cut it per pixel
    by nearest segment rather than by connected component -- which, on a pose with
    one arm folded to the chin, handed the right sleeve's drape to the *left* arm
    (genuinely nearer that segment, which runs down to the hand) and rendered it
    as a detached strip lying across the leg.
    """
    assert "handwear" in taxonomy.PAIRED_LR_TAGS
    assert "handwear" not in taxonomy.OCS_LR_TAGS, "upstream normally splits it"
    # Everything either side of the pipeline may split is covered.
    for tag in taxonomy.OCS_LR_TAGS + taxonomy.UPSTREAM_LR_TAGS:
        assert tag in taxonomy.PAIRED_LR_TAGS, tag


def test_already_split_layers_are_left_alone(figure):
    """When upstream did split, the side suffix is present and OCS must not redo it."""
    rig, kept = _rig_and_parts(figure)
    assert any(p.tag == "handwear" and p.side for p in kept), "fixture is pre-split"
    _parts, report = limbs.partition(figure, kept, rig, RigSettings())
    assert "handwear" not in report["lr_split"]
