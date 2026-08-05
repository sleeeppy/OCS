"""see-through's layer taxonomy, and how OCS maps it onto a Spine skeleton.

Everything here is derived from reading the see-through submodule, not guessed.
The relevant upstream code is:

- ``common/utils/inference_utils.py``
  - ``apply_layerdiff``: the v3 pipeline runs two passes. The body pass emits
    ``BODY_PASS_TAGS`` and the head crop pass emits ``HEAD_PASS_TAGS``.
  - ``apply_marigold``: iterates ``VALID_BODY_PARTS_V2`` and *composes*
    ``hair <- [back hair, front hair]`` and
    ``eyes <- [eyewhite, irides, eyelash, eyebrow]``, writing per-sub-tag depth.
    The composed parents never become parts; their children do.
  - ``further_extr`` + ``--tblr_split``: LR-splits only handwear, ears, and the
    four eye tags. Everything else stays whole.
  - ``part_lr_split`` / ``label_lr_split``: connected components, then the
    component with the *smaller* centroid x (viewer-left) is written as
    ``{tag}-r``. So the suffix is anatomical, not screen-space.

The load-bearing consequence: **there is no skin tag.** The taxonomy is
clothing-and-face centric, so bare arms and legs belong to no layer at all.
OCS therefore synthesises a skin base (see ``ocs.limbs``) and cuts it with the
user's bone skeleton, which is also what makes left/right limb separation
unconditional rather than dependent on the model finding two blobs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Upstream tag sets (verbatim from see-through)
# --------------------------------------------------------------------------

#: ``apply_layerdiff`` v3, first pass. Note the spaces, and that ``head`` is an
#: intermediate crop -- it is not in ``VALID_BODY_PARTS_V2`` so it never becomes
#: a part.
BODY_PASS_TAGS = (
    "front hair", "back hair", "head", "neck", "neckwear", "topwear",
    "handwear", "bottomwear", "legwear", "footwear", "tail", "wings", "objects",
)

#: ``apply_layerdiff`` v3, second pass (run on the head crop).
HEAD_PASS_TAGS = (
    "headwear", "face", "irides", "eyebrow", "eyewhite", "eyelash",
    "eyewear", "ears", "earwear", "nose", "mouth",
)

#: Tags that survive ``apply_marigold`` + ``further_extr`` as real PSD layers.
#: This is the "up to 23 layers" the see-through README advertises.
PART_TAGS = (
    "front hair", "back hair",
    "headwear", "face",
    "irides", "eyebrow", "eyewhite", "eyelash", "eyewear",
    "ears", "earwear", "nose", "mouth",
    "neck", "neckwear",
    "topwear", "handwear", "bottomwear", "legwear", "footwear",
    "tail", "wings", "objects",
)

#: Tags see-through itself LR-splits when ``--tblr_split`` is passed
#: (``further_extr``). OCS always passes it.
UPSTREAM_LR_TAGS = (
    "handwear", "ears", "eyewhite", "irides", "eyelash", "eyebrow",
)

#: Paired anatomy that see-through leaves whole. OCS splits these itself,
#: because a single attachment spanning both sides cannot be rigged.
OCS_LR_TAGS = (
    "legwear", "footwear",
)

#: Every tag that is two things, whichever side of the pipeline splits it.
#:
#: Whether ``handwear`` arrives split is a property of the *run*, not the tag:
#: ``--tblr_split`` is applied by ``further_extr``, which is the last thing an
#: inference does, so an interrupted run yields both sleeves in one layer. Keying
#: the decision off ``OCS_LR_TAGS`` alone then leaves it unsplit, and the bone
#: partition cuts it per pixel by nearest segment instead of by connected
#: component. On a pose with one arm folded to the chin that assigns the *right*
#: sleeve's drape to the left arm -- it is genuinely nearer the left arm's segment,
#: which runs all the way down to the hand -- and the drape renders as a detached
#: strip lying across the leg.
#:
#: So the test is the data, not the tag list: a paired tag with no side suffix
#: still needs splitting, no matter who was supposed to have done it.
PAIRED_LR_TAGS = OCS_LR_TAGS + UPSTREAM_LR_TAGS

#: Layers that span more than one bone and must be cut by the bone partition.
#:
#: ``handwear`` and ``legwear`` are in here because their names are misleading.
#: Measured on see-through's own sample: ``handwear-l`` is a 116x463 vertical
#: strip and ``legwear`` is 310x547 covering both legs -- these are the whole
#: **arm** and **leg**, bare skin included, not gloves and socks. The union of
#: all 24 layers covers the character completely, which is why there is usually
#: no leftover skin to synthesise. Each limb therefore still has to be cut at
#: the elbow and the knee, or it cannot bend.
LIMB_SPANNING_TAGS = (
    "topwear", "bottomwear", "legwear", "handwear",
)

#: Tags that behave as a whole arm / leg, used when guessing the skeleton.
ARM_TAGS = ("handwear",)
LEG_TAGS = ("legwear", "footwear")

#: ``-r`` is the character's right, i.e. the *viewer's left*. Confirmed from
#: ``part_lr_split``: the lower-centroid-x component is tagged ``-r``.
LR_SUFFIX_TO_SIDE = {"-r": "right", "-l": "left"}


# --------------------------------------------------------------------------
# Bone template
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BoneSpec:
    name: str
    parent: str | None
    #: Optional bones are only created when the artwork justifies them (a tail
    #: layer exists, the user adds hands, ...). Non-optional bones always exist
    #: so that downstream code can rely on the full L/R limb chain.
    optional: bool = False
    #: Hint used by ``ocs.skeleton`` for the initial guess, as a fraction of the
    #: silhouette bounding box: (x, y) with y measured from the top.
    hint: tuple[float, float] | None = None


#: Matches the reference bone layout the user supplied: ``{side}Arm`` is the
#: shoulder, ``{side}Elbow`` the elbow, ``{side}Leg`` the hip, ``{side}Knee``
#: the knee. ``root`` is added because Spine requires it; it is not editable.
BONE_TEMPLATE: tuple[BoneSpec, ...] = (
    BoneSpec("root",       None,         hint=(0.50, 1.00)),
    BoneSpec("torso",      "root",       hint=(0.50, 0.55)),
    BoneSpec("neck",       "torso",      hint=(0.50, 0.33)),
    BoneSpec("head",       "neck",       hint=(0.50, 0.14)),
    BoneSpec("eyes",       "head",       hint=(0.50, 0.20)),

    BoneSpec("rightArm",   "torso",      hint=(0.38, 0.38)),
    BoneSpec("rightElbow", "rightArm",   hint=(0.30, 0.52)),
    BoneSpec("rightHand",  "rightElbow", optional=True, hint=(0.26, 0.64)),
    BoneSpec("leftArm",    "torso",      hint=(0.62, 0.38)),
    BoneSpec("leftElbow",  "leftArm",    hint=(0.70, 0.52)),
    BoneSpec("leftHand",   "leftElbow",  optional=True, hint=(0.74, 0.64)),

    BoneSpec("rightLeg",   "torso",      hint=(0.42, 0.62)),
    BoneSpec("rightKnee",  "rightLeg",   hint=(0.40, 0.80)),
    BoneSpec("rightFoot",  "rightKnee",  optional=True, hint=(0.39, 0.96)),
    BoneSpec("leftLeg",    "torso",      hint=(0.58, 0.62)),
    BoneSpec("leftKnee",   "leftLeg",    hint=(0.60, 0.80)),
    BoneSpec("leftFoot",   "leftKnee",   optional=True, hint=(0.61, 0.96)),

    BoneSpec("hairBack",   "head",       optional=True, hint=(0.50, 0.22)),
    BoneSpec("tail",       "torso",      optional=True, hint=(0.50, 0.70)),
    BoneSpec("wings",      "torso",      optional=True, hint=(0.50, 0.45)),
)

BONES_BY_NAME = {b.name: b for b in BONE_TEMPLATE}
REQUIRED_BONES = tuple(b.name for b in BONE_TEMPLATE if not b.optional)
OPTIONAL_BONES = tuple(b.name for b in BONE_TEMPLATE if b.optional)

#: Bones whose position is mirrored by the editor's ``X`` / symmetry snap.
MIRROR_PAIRS = (
    ("leftArm", "rightArm"), ("leftElbow", "rightElbow"),
    ("leftHand", "rightHand"), ("leftLeg", "rightLeg"),
    ("leftKnee", "rightKnee"), ("leftFoot", "rightFoot"),
)


# --------------------------------------------------------------------------
# Bone-driven partition regions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RegionSpec:
    """One influence segment of the bone-driven partition.

    A pixel is assigned to the region whose segment it is closest to, which is
    the same idea as a Blender bone envelope. ``bone`` is what the resulting
    attachment is bound to in Spine.
    """

    name: str
    #: Segment start bone. Also the bone the region's attachment binds to, so
    #: the upper arm follows the shoulder and the forearm follows the elbow.
    bone: str
    #: Segment end bone. When absent (an optional bone the user did not place),
    #: the segment is extrapolated past ``bone`` along its own direction.
    to_bone: str | None
    side: str | None = None


#: Ordered so that ``torso`` loses ties to the limbs -- limb pixels matter more
#: for deformation quality than an extra pixel of torso.
SKIN_REGIONS: tuple[RegionSpec, ...] = (
    RegionSpec("arm_r_upper", "rightArm",   "rightElbow", "right"),
    RegionSpec("arm_r_lower", "rightElbow", "rightHand",  "right"),
    RegionSpec("arm_l_upper", "leftArm",    "leftElbow",  "left"),
    RegionSpec("arm_l_lower", "leftElbow",  "leftHand",   "left"),
    RegionSpec("leg_r_upper", "rightLeg",   "rightKnee",  "right"),
    RegionSpec("leg_r_lower", "rightKnee",  "rightFoot",  "right"),
    RegionSpec("leg_l_upper", "leftLeg",    "leftKnee",   "left"),
    RegionSpec("leg_l_lower", "leftKnee",   "leftFoot",   "left"),
    RegionSpec("torso",       "torso",      "neck",       None),
)

#: One merged region per limb, spanning that limb's whole chain.
#:
#: Cutting a garment at a joint guarantees a visible seam there: the two slices
#: become separate attachments in separate slots, and the boundary between them is
#: a hard edge in the render whatever the weights say. Measured on one character,
#: 61.8% of the pixels that came out darker than the source art sat on a part's own
#: edge. A single mesh spanning the joint, with vertices weighted across both
#: bones, has no such boundary -- and it still bends, which is how a hand-built
#: Spine rig does it.
#:
#: The left/right split stays: the two arms are disjoint and must move
#: independently. That is also what requirement 2-2 actually asks for.
LIMB_CHAINS: dict[str, tuple[str, ...]] = {
    "arm_r": ("arm_r_upper", "arm_r_lower"),
    "arm_l": ("arm_l_upper", "arm_l_lower"),
    "leg_r": ("leg_r_upper", "leg_r_lower"),
    "leg_l": ("leg_l_upper", "leg_l_lower"),
}

#: Specs for the merged limbs. ``bone`` is the chain root, which the part binds
#: to; ``rig._candidate_bones`` then pulls in the child joint so vertices near the
#: elbow blend between the two. These are never used for distance assignment --
#: merged masks are the *union* of their members' masks, so per-segment accuracy
#: is unchanged and only the grouping differs.
MERGED_LIMB_REGIONS: tuple[RegionSpec, ...] = (
    RegionSpec("arm_r", "rightArm", "rightHand", "right"),
    RegionSpec("arm_l", "leftArm",  "leftHand",  "left"),
    RegionSpec("leg_r", "rightLeg", "rightFoot", "right"),
    RegionSpec("leg_l", "leftLeg",  "leftFoot",  "left"),
)

#: Everything ``bone_for_part`` / ``part_side`` may be asked to resolve.
ALL_REGIONS: tuple[RegionSpec, ...] = SKIN_REGIONS + MERGED_LIMB_REGIONS

#: The limbs requirement 2-2 is about: left and right, arms and legs. ``ocs.limbs``
#: asserts every one is produced, even for a single-blob silhouette or a symmetric
#: pose. Named per merged limb rather than per segment -- the requirement is
#: "left/right arms and legs are separated", and demanding a separate part per
#: joint segment on top of that is what forced the seams.
MANDATORY_LIMB_REGIONS = ("arm_l", "arm_r", "leg_l", "leg_r")

#: Per-segment form, for a partition that keeps the joint cuts.
MANDATORY_LIMB_SEGMENTS = (
    "arm_l_upper", "arm_l_lower", "arm_r_upper", "arm_r_lower",
    "leg_l_upper", "leg_l_lower", "leg_r_upper", "leg_r_lower",
)

ARM_REGIONS = ("arm_r_upper", "arm_r_lower", "arm_l_upper", "arm_l_lower")
LEG_REGIONS = ("leg_r_upper", "leg_r_lower", "leg_l_upper", "leg_l_lower")


def merged_region_of(region: str) -> str | None:
    """``arm_r_upper`` -> ``arm_r``; ``torso`` -> ``None``."""
    for merged, members in LIMB_CHAINS.items():
        if region in members:
            return merged
    return None

#: Which regions a tag is allowed to be cut into.
#:
#: Pure nearest-segment assignment is not enough, and failed observably: with the
#: arms hanging at the sides, the forearm segment is closer to the upper thigh
#: than the thigh's own bone is, so ``legwear`` was being sliced into
#: ``arm_r_lower``. Distance alone has no idea a leg pixel cannot belong to an
#: arm. Constraining the candidate set per tag restores that knowledge, and every
#: pixel still lands somewhere because the argmin is taken over the allowed
#: subset rather than by discarding the losers.
TAG_ALLOWED_REGIONS: dict[str, tuple[str, ...]] = {
    "handwear": ARM_REGIONS,
    "legwear": LEG_REGIONS,
    "footwear": ("leg_r_lower", "leg_l_lower"),
    "topwear": ("torso", "arm_r_upper", "arm_l_upper"),
    "bottomwear": ("torso", "leg_r_upper", "leg_l_upper"),
}


def allowed_regions(tag: str, side: str | None = None) -> tuple[str, ...] | None:
    """Regions ``tag`` may be cut into, narrowed to one side when known.

    ``None`` means "no restriction" (the synthesised skin base, which legitimately
    spans everything).
    """
    regions = TAG_ALLOWED_REGIONS.get(tag)
    if regions is None:
        return None
    if side is None:
        return regions
    # A part already known to be the character's left cannot own right-side
    # geometry, so drop the mirrored candidates outright.
    other = "right" if side == "left" else "left"
    by_name = {spec.name: spec for spec in SKIN_REGIONS}
    narrowed = tuple(r for r in regions if by_name.get(r) is None or by_name[r].side != other)
    return narrowed or regions


# --------------------------------------------------------------------------
# tag -> bone binding
# --------------------------------------------------------------------------

#: Which bone a whole (unsplit) tag hangs off. Tags absent from this map fall
#: back to ``torso``.
TAG_TO_BONE = {
    "front hair": "head",
    "back hair": "hairBack",   # falls back to head when hairBack is not placed
    "headwear": "head",
    "face": "head",
    "irides": "eyes",
    "eyewhite": "eyes",
    "eyelash": "eyes",
    "eyebrow": "eyes",
    "eyewear": "head",
    "ears": "head",
    "earwear": "head",
    "nose": "head",
    "mouth": "head",
    "neck": "neck",
    "neckwear": "neck",
    "topwear": "torso",
    "bottomwear": "torso",
    "handwear": "{side}Elbow",
    "legwear": "{side}Knee",
    "footwear": "{side}Knee",
    "tail": "tail",
    "wings": "wings",
    "objects": "torso",
}

#: Hard layering constraints: ``tag: (tags it must draw *after*)``.
#:
#: see-through's ``depth_median`` is a per-image estimate and is wrong for thin
#: overlapping accessories. Measured on one character: ``headwear`` (a ribbon tied
#: over the hair) came out at depth 0.231 against ``front hair`` at 0.098, so the
#: hair drew last and hid 91% of the ribbon. Six more inversions in the same
#: export put ``eyebrow`` under ``face`` and ``nose`` under ``eyewhite``.
#:
#: Upstream only patches a few of these itself -- ``further_extr`` nudges
#: nose/mouth/eyes behind ``face`` but never touches the v3 split eye tags.
#:
#: Only unambiguous relationships belong here. Bangs versus glasses, for
#: instance, genuinely varies by artwork, so it is left to the depth estimate.
DRAW_AFTER: dict[str, tuple[str, ...]] = {
    "headwear": ("front hair", "back hair"),
    "front hair": ("face", "ears", "earwear", "neck"),
    "eyewhite": ("face",),
    "irides": ("face", "eyewhite"),
    "eyelash": ("face", "eyewhite", "irides"),
    "eyebrow": ("face",),
    "nose": ("face",),
    "mouth": ("face",),
    "eyewear": ("face", "eyewhite", "irides", "eyelash", "eyebrow", "nose"),
    "footwear": ("legwear",),
}

#: Draw-order tie-break, applied *after* see-through's depth ordering. Larger
#: sorts nearer the viewer. Only consulted when two layers share a depth.
Z_PRIOR = {
    "back hair": -60,
    "wings": -55,
    "tail": -50,
    "neck": -20,
    "skin": -15,
    "legwear": -12,
    "bottomwear": -10,
    "topwear": -5,
    "footwear": -4,
    "neckwear": 0,
    "face": 5,
    "ears": 4,
    "earwear": 6,
    "eyewhite": 8,
    "irides": 9,
    "eyelash": 10,
    "eyebrow": 11,
    "nose": 12,
    "mouth": 12,
    "eyewear": 20,
    "handwear": 25,
    "front hair": 30,
    "headwear": 40,
    "objects": 45,
}


# --------------------------------------------------------------------------
# Part identity
# --------------------------------------------------------------------------

#: ``topwear@arm_l_upper`` -- a tag cut by the bone partition.
REGION_SEP = "@"

_SLUG_RE = re.compile(r"[^0-9a-zA-Z]+")


def slugify(name: str) -> str:
    """Spine-safe name: ``front hair`` -> ``front_hair``, ``handwear-l`` -> ``handwear_l``."""
    return _SLUG_RE.sub("_", name).strip("_").lower()


def base_tag_with_side(part_name: str) -> str:
    """Drop only the region suffix: ``legwear-l@leg_l_upper`` -> ``legwear-l``.

    The LR suffix has to survive -- it is what ``part_side`` reads for a layer
    see-through split itself.
    """
    return part_name.split(REGION_SEP, 1)[0]


def base_tag(part_name: str) -> str:
    """Strip LR suffix and region suffix: ``legwear-l@leg_l_upper`` -> ``legwear``."""
    name = part_name.split(REGION_SEP, 1)[0]
    for suffix in ("-l", "-r"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def part_side(part_name: str) -> str | None:
    """``left`` / ``right`` / ``None``, from either an LR suffix or a region."""
    head, _, region = part_name.partition(REGION_SEP)
    for suffix, side in LR_SUFFIX_TO_SIDE.items():
        if head.endswith(suffix):
            return side
    if region:
        for spec in ALL_REGIONS:
            if spec.name == region:
                return spec.side
    return None


def part_region(part_name: str) -> str | None:
    _, _, region = part_name.partition(REGION_SEP)
    return region or None


def bone_for_part(part_name: str, available_bones: set[str]) -> str:
    """Resolve which bone a part binds to, honouring optional-bone fallbacks."""
    region = part_region(part_name)
    if region:
        for spec in ALL_REGIONS:
            if spec.name == region:
                return spec.bone if spec.bone in available_bones else "torso"

    tag = base_tag(part_name)
    bone = TAG_TO_BONE.get(tag, "torso")

    if "{side}" in bone:
        side = part_side(part_name)
        if side is None:
            return "torso"
        bone = bone.format(side=side)

    if bone not in available_bones:
        # e.g. hairBack was never placed -> back hair rides the head instead.
        fallback = BONES_BY_NAME.get(bone)
        parent = fallback.parent if fallback else None
        return parent if parent in available_bones else "torso"
    return bone


def z_prior(part_name: str) -> int:
    return Z_PRIOR.get(base_tag(part_name), 0)


@dataclass
class PartNaming:
    """Bookkeeping for the names OCS invents on top of see-through's tags."""

    skin_prefix: str = "skin"
    used: set[str] = field(default_factory=set)

    def skin(self, region: str) -> str:
        return f"{self.skin_prefix}{REGION_SEP}{region}"

    def garment(self, tag: str, region: str) -> str:
        """``tag@region``, **replacing** any region the name already carries.

        A part can be cut twice: ``_slice_by_regions`` assigns one, and then
        ``enforce_limb_coverage`` may carve a mandatory region out of the result.
        Appending gave ``bottomwear@leg_r_upper@leg_r``, and everything that reads
        a region splits on the *first* separator, so that name resolved to the
        region ``leg_r_upper@leg_r`` -- which matches no spec. The consequences are
        silent and severe: ``part_side`` returns ``None`` so the piece vanishes
        from the left/right tally, and ``bone_for_part`` falls back to ``torso``,
        binding a piece of skirt lying over the shin to the *trunk*. It then slides
        across the leg whenever the torso moves.
        """
        return f"{base_tag_with_side(tag)}{REGION_SEP}{region}"

    def unique_slug(self, part_name: str) -> str:
        slug = slugify(part_name)
        if slug not in self.used:
            self.used.add(slug)
            return slug
        i = 2
        while f"{slug}_{i}" in self.used:
            i += 1
        self.used.add(f"{slug}_{i}")
        return f"{slug}_{i}"
