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


def test_a_forced_limb_region_is_carved_from_a_layer_that_belongs_there():
    """Requirement 2-2 is unconditional, but not at any price.

    ``enforce_limb_coverage`` picked its source by overlap alone, and on a seated
    figure with a long sleeve draped across her shin the sleeve covers more of
    ``leg_r`` than the leg covering does. So the right leg was carved out of
    ``handwear``: a panel of sleeve fabric standing in for a leg, drawn over the
    leg region, which then had to be hidden behind the real leg and showed as a
    hard-edged strip down the shin wherever it stuck out. The draw-order guess
    could not save it either -- both layers come back from see-through at alpha
    254, so ``A over B`` reduces to ``A`` and the RGB test simply reports that the
    leg matches the artwork and the sleeve does not.

    ``allowed_regions`` is the same table that stops the ordinary partition
    putting a leg layer on an arm, and it belongs here too. Its entries are
    *segments*, so they have to be compared through ``merged_region_of``:
    ``leg_r`` never equals ``leg_r_upper``, and without that step every candidate
    scores zero and overlap decides again.
    """
    import numpy as np

    from ocs import taxonomy
    from ocs.psd_io import Decomposition, Part

    def block(x0, y0, x1, y1, rgb):
        a = np.zeros((256, 256, 4), np.uint8)
        a[y0:y1, x0:x1] = (*rgb, 255)
        return a

    # A leg down the left, and a sleeve draped across it covering more of it.
    leg = Part(name="legwear", rgba=block(60, 90, 100, 240, (230, 200, 180)),
               offset=(0, 0))
    sleeve = Part(name="handwear-r", rgba=block(40, 80, 150, 250, (240, 235, 230)),
                  offset=(0, 0))
    torso = Part(name="topwear", rgba=block(90, 20, 170, 110, (200, 60, 60)),
                 offset=(0, 0))
    src = np.zeros((256, 256, 4), np.uint8)
    src[20:250, 40:170] = (200, 120, 100, 255)
    decomp = Decomposition(canvas=(256, 256), parts=[leg, sleeve, torso], src_img=src)

    target = np.zeros((256, 256), bool)
    target[90:240, 55:105] = True                       # the leg_r region

    best = None
    for p in (leg, sleeve, torso):
        overlap = int((p.canvas_mask(decomp.canvas) & target).sum())
        if overlap <= 0:
            continue
        allowed = taxonomy.allowed_regions(p.tag, p.side)
        fits = 1 if allowed is None else int(
            "leg_r" in allowed
            or any(taxonomy.merged_region_of(a) == "leg_r" for a in allowed))
        if best is None or (fits, overlap) > (best[0], best[1]):
            best = (fits, overlap, p)

    assert best is not None
    assert int((sleeve.canvas_mask(decomp.canvas) & target).sum()) > \
        int((leg.canvas_mask(decomp.canvas) & target).sum()), \
        "the case is meant to have the wrong layer covering more of the region"
    assert best[2].name == "legwear", (
        f"carved leg_r out of {best[2].name}; overlap alone would pick the sleeve")
