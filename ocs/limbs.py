"""Requirement 2-2 - left and right arms/legs, separated unconditionally.

see-through cannot deliver this on its own, for two reasons found by reading it:

* Its taxonomy has **no skin tag**. ``BODY_PASS_TAGS`` is clothing and hair;
  bare arms and legs belong to no layer, so there is nothing to split.
* Where it does split (``part_lr_split`` with ``--tblr_split``) it only splits
  when connected components happen to separate the two sides, and it keeps only
  the two largest components - the rest of the layer is silently discarded.

So OCS derives the separation from geometry it fully controls: the bone
skeleton the user placed. Every pixel is assigned to the bone segment it is
closest to (the same idea as a Blender bone envelope), which yields left and
right limb parts even when the silhouette is a single blob, the pose is
symmetric, or the arms cross the torso.

The guarantee is enforced, not hoped for: ``enforce_limb_coverage`` carves the
region out of whichever layer covers it if nothing else produced a part there.
"""

from __future__ import annotations

import numpy as np
import cv2

from . import taxonomy
from .config import RigSettings
from .psd_io import Decomposition, Part
from .skeleton import Rig, build_silhouette

#: Sentinel for "no region" in the label array.
UNASSIGNED = -1

#: Tags that can never belong to a limb, so the coverage safety net skips them.
_HEAD_TAGS = frozenset({
    "face", "irides", "eyewhite", "eyelash", "eyebrow", "nose", "mouth",
    "front hair", "back hair", "headwear", "eyewear", "ears", "earwear",
})


# --------------------------------------------------------------------------
# bone-driven partition
# --------------------------------------------------------------------------


def _region_segments(rig: Rig) -> list[tuple[taxonomy.RegionSpec, np.ndarray, np.ndarray]]:
    """Resolve each region spec to a concrete segment in canvas coordinates.

    ``torso`` is special-cased: its spec segment is torso->neck, which is only
    the upper spine, so the hips would be captured by the leg segments. Spanning
    neck -> hip midpoint instead makes the trunk region cover the whole trunk.
    """
    out = []
    for spec in taxonomy.SKIN_REGIONS:
        if spec.name == "torso":
            neck = rig.bones.get("neck") or rig.bones.get("torso")
            hips = [rig.bones.get("leftLeg"), rig.bones.get("rightLeg")]
            hips = [h for h in hips if h is not None]
            if neck is None:
                continue
            if hips:
                hip_mid = np.mean(np.array(hips, dtype=np.float64), axis=0)
            else:
                hip_mid = np.array(rig.bones.get("torso", neck), dtype=np.float64)
            out.append((spec, np.array(neck, dtype=np.float64), hip_mid))
            continue

        seg = rig.segment(spec)
        if seg is None:
            continue
        a, b = seg
        if np.linalg.norm(b - a) < 1e-6:
            b = a + np.array([0.0, 1.0])
        out.append((spec, a, b))
    return out


def _segment_distance(shape: tuple[int, int], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel Euclidean distance to segment ab."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    px = xs.astype(np.float32) - a[0]
    py = ys.astype(np.float32) - a[1]
    dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
    denom = dx * dx + dy * dy
    t = np.clip((px * dx + py * dy) / denom, 0.0, 1.0) if denom > 1e-9 else np.zeros_like(px)
    return np.hypot(px - t * dx, py - t * dy)


class Partition:
    """Per-pixel distance to every bone segment, reusable across parts.

    The distance stack is computed once (9 regions on a 1280x1280 canvas is
    ~59 MB) and each part then takes an ``argmin`` over just the regions its tag
    is allowed to occupy. Restricting the candidate set per part -- rather than
    labelling once globally and intersecting -- is what keeps a leg pixel out of
    an arm region while still assigning every pixel somewhere.
    """

    def __init__(self, rig: Rig, silhouette: np.ndarray):
        specs_segs = _region_segments(rig)
        self.specs: list[taxonomy.RegionSpec] = [s for s, _a, _b in specs_segs]
        self.index: dict[str, int] = {s.name: i for i, s in enumerate(self.specs)}
        self.silhouette = silhouette
        if specs_segs:
            self.dists = np.stack(
                [_segment_distance(silhouette.shape, a, b) for _s, a, b in specs_segs], axis=0
            ).astype(np.float32)
        else:
            self.dists = np.empty((0, *silhouette.shape), dtype=np.float32)

    def labels(self, allowed: tuple[str, ...] | None = None) -> np.ndarray:
        """Region label per silhouette pixel, ``UNASSIGNED`` outside it.

        ``taxonomy.SKIN_REGIONS`` lists the limbs before the torso and ``argmin``
        keeps the first minimum, so a shoulder pixel equidistant from the arm and
        the trunk goes to the arm -- limb deformation is what the rig is for.
        """
        out = np.full(self.silhouette.shape, UNASSIGNED, dtype=np.int16)
        if self.dists.shape[0] == 0:
            return out
        cols = self._columns(allowed)
        winner = np.argmin(self.dists[cols], axis=0)
        lookup = np.array(cols, dtype=np.int16)
        out[self.silhouette] = lookup[winner[self.silhouette]]
        return out

    def masks(
        self, allowed: tuple[str, ...] | None = None, seam_px: int = 0,
        within: np.ndarray | None = None, merge_limbs: bool = False,
    ) -> dict[str, np.ndarray]:
        """Region name -> mask, optionally grown by a shared seam allowance.

        Neighbouring parts are meant to overlap slightly; without it a bent elbow
        shows a wedge of background between the upper arm and the forearm.

        ``merge_limbs`` unions each limb's segment masks into one entry keyed by
        the merged name (``arm_r_upper | arm_r_lower -> arm_r``). Assignment itself
        is untouched -- pixels are still matched to the nearest *segment*, so
        accuracy is identical; only the grouping changes. What it buys is one
        attachment per limb instead of two, and therefore no cut at the joint to
        show as a seam.
        """
        labels = self.labels(allowed)
        limit = self.silhouette if within is None else (self.silhouette | within)
        out: dict[str, np.ndarray] = {}
        for name in (allowed if allowed is not None else tuple(self.index)):
            i = self.index.get(name)
            if i is None:
                continue
            mask = labels == i
            if seam_px > 0 and mask.any():
                k = 2 * seam_px + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                mask = (cv2.dilate(mask.astype(np.uint8), kernel) > 0) & limit
            key = name
            if merge_limbs:
                key = self._merge_key(name, allowed)
            out[key] = out[key] | mask if key in out else mask
        return out

    def _merge_key(self, name: str, allowed: tuple[str, ...] | None) -> str:
        """Chain name for ``name``, but only if the caller may span the whole chain.

        ``footwear`` is allowed in ``leg_*_lower`` only, so folding its region into
        ``leg_l`` would bind a shoe to the hip -- and ``rig._candidate_bones``, which
        applies the same table, would then strip that bone back out and leave the
        slot bound to a bone its own mesh is not weighted to. Same for ``topwear``,
        which reaches the shoulder but not the elbow.

        So merge only where the tag genuinely covers the limb: ``legwear`` and
        ``handwear``, the layers that *are* whole limbs.
        """
        merged = taxonomy.merged_region_of(name)
        if merged is None:
            return name
        members = taxonomy.LIMB_CHAINS[merged]
        if allowed is not None and not set(members) <= set(allowed):
            return name
        if not all(m in self.index for m in members):
            return name
        return merged

    def _columns(self, allowed: tuple[str, ...] | None) -> list[int]:
        if allowed is None:
            return list(range(len(self.specs)))
        cols = [self.index[n] for n in allowed if n in self.index]
        return cols or list(range(len(self.specs)))


def mandatory_limb_regions(s: RigSettings) -> tuple[str, ...]:
    """Which limb regions must end up covered, given the partition granularity."""
    return (taxonomy.MANDATORY_LIMB_REGIONS if s.merge_limb_slices
            else taxonomy.MANDATORY_LIMB_SEGMENTS)


def region_labels(
    rig: Rig, silhouette: np.ndarray
) -> tuple[np.ndarray, list[taxonomy.RegionSpec]]:
    """Unrestricted global labelling. Kept for visualisation and tests."""
    part = Partition(rig, silhouette)
    return part.labels(), part.specs


# --------------------------------------------------------------------------
# part construction
# --------------------------------------------------------------------------


def _crop_to_part(
    name: str,
    canvas_rgba: np.ndarray,
    mask: np.ndarray,
    depth_median: float,
    synthetic: bool = True,
    meta: dict | None = None,
    min_opaque: int = 1,
) -> Part | None:
    """Build a cropped ``Part`` from a canvas-sized RGBA + boolean mask.

    ``min_opaque`` guards against the mask and the image disagreeing: the mask can
    select thousands of pixels that turn out to be transparent in the source, so
    a bare "any opaque pixel at all" check lets through parts that are nearly
    empty inside a large bounding box.
    """
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    rgba = canvas_rgba[y1:y2, x1:x2].copy()
    sub = mask[y1:y2, x1:x2]
    rgba[..., 3] = np.where(sub, rgba[..., 3], 0)
    if int((rgba[..., 3] > 8).sum()) < max(1, min_opaque):
        return None

    return Part(
        name=name,
        rgba=rgba,
        offset=(x1, y1),
        depth_median=depth_median,
        synthetic=synthetic,
        meta=meta or {},
    )


def _clean_mask(mask: np.ndarray, min_area: int, open_px: int = 1) -> np.ndarray:
    """Drop antialiasing crumbs: open, then keep components above ``min_area``."""
    if not mask.any():
        return mask
    if open_px > 0:
        k = 2 * open_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep |= labels == i
    return keep


# --------------------------------------------------------------------------
# skin base
# --------------------------------------------------------------------------


def skin_base_mask(
    decomp: Decomposition, parts: list[Part], silhouette: np.ndarray, min_area: int
) -> np.ndarray:
    """Silhouette minus every extracted layer: the skin see-through never emits.

    Intersected with the source's own opaque pixels, which is not redundant. The
    silhouette is morphologically closed and hole-filled, so it reaches a few
    pixels beyond the artwork; that margin belongs to no layer and therefore
    reads as "uncovered skin". Measured on a real run it produced four hairline
    parts -- a 23x178 bounding box holding 75 opaque pixels -- whose contours
    then collapsed below three points and failed to mesh.
    """
    covered = np.zeros_like(silhouette)
    for p in parts:
        covered |= p.canvas_mask(decomp.canvas)

    skin = silhouette & ~covered
    if decomp.src_img is not None and decomp.src_img.shape[2] >= 4:
        skin &= decomp.src_img[..., 3] > 8
    return _clean_mask(skin, min_area=min_area, open_px=2)


def _skin_depth(parts: list[Part]) -> float:
    """Place skin behind clothing.

    see-through orders layers by ``depth_median`` descending (further first), so
    "behind" means a *larger* value than every garment.
    """
    garments = [p.depth_median for p in parts
                if p.tag in ("topwear", "bottomwear", "legwear", "footwear", "handwear")]
    if not garments:
        return 0.9
    return min(1.0, max(garments) + 0.01)


# --------------------------------------------------------------------------
# left / right splitting
# --------------------------------------------------------------------------


def split_lr(
    part: Part,
    decomp: Decomposition,
    rig: Rig,
    grid: "Partition",
    min_fraction: float = 0.02,
) -> list[Part]:
    """Split one part into ``-l`` / ``-r``.

    Improves on see-through's ``part_lr_split`` in two ways: every connected
    component is kept (upstream keeps only the two largest and discards the
    rest), and when the layer is a single blob it is cut along the bone
    partition rather than a straight vertical line, so overlapping legs separate
    correctly.
    """
    mask = part.canvas_mask(decomp.canvas)
    total = int(mask.sum())
    if total == 0:
        return [part]

    canvas_rgba = part.canvas_rgba(decomp.canvas)
    specs = grid.specs
    # Restrict to the regions this tag can occupy before deciding sides, so a
    # leg is not handed to an arm region just because the arm hangs alongside it.
    labels = grid.labels(taxonomy.allowed_regions(part.tag, part.side))
    side_of_region = {spec.name: spec.side for spec in specs}
    region_index = {i: specs[i].name for i in range(len(specs))}

    left = np.zeros_like(mask)
    right = np.zeros_like(mask)

    n, cc, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_fraction * total]

    if len(big) >= 2:
        # Assign whole components, so a shoe and its lace stay together.
        axis = _midline(rig, part.tag)
        for i in range(1, n):
            comp = cc == i
            side = _component_side(comp, labels, region_index, side_of_region, centroids[i], axis)
            (left if side == "left" else right)[comp] = True
    else:
        # One blob: cut it with the bone partition, falling back to the midline
        # for pixels the partition left unassigned.
        axis = _midline(rig, part.tag)
        for i, spec in enumerate(specs):
            if spec.side is None:
                continue
            sel = mask & (labels == i)
            (left if spec.side == "left" else right)[sel] = True
        rest = mask & ~(left | right)
        if rest.any():
            ys_r, xs_r = np.nonzero(rest)
            is_left = xs_r > axis   # viewer-right is the character's left
            left[ys_r[is_left], xs_r[is_left]] = True
            right[ys_r[~is_left], xs_r[~is_left]] = True

    out: list[Part] = []
    for suffix, sel in (("-r", right), ("-l", left)):
        if sel.sum() < min_fraction * total:
            continue
        p = _crop_to_part(
            f"{part.name}{suffix}", canvas_rgba, sel, part.depth_median,
            synthetic=True, meta={**part.meta, "split_from": part.name},
        )
        if p is not None:
            out.append(p)

    return out if len(out) == 2 else [part]


def _midline(rig: Rig, tag: str) -> float:
    """x of the character's centre line, preferring the hips for lower-body tags."""
    if tag in ("legwear", "footwear", "bottomwear"):
        hips = [rig.bones.get("leftLeg"), rig.bones.get("rightLeg")]
        hips = [h for h in hips if h is not None]
        if len(hips) == 2:
            return (hips[0][0] + hips[1][0]) / 2
    torso = rig.bones.get("torso")
    return torso[0] if torso else rig.canvas[0] / 2


def _component_side(
    comp: np.ndarray,
    labels: np.ndarray,
    region_index: dict[int, str],
    side_of_region: dict[str, str | None],
    centroid,
    axis: float,
) -> str:
    """Vote the component's side from the regions it overlaps, else the midline."""
    vals, counts = np.unique(labels[comp], return_counts=True)
    tally = {"left": 0, "right": 0}
    for v, c in zip(vals.tolist(), counts.tolist()):
        side = side_of_region.get(region_index.get(int(v), ""), None)
        if side:
            tally[side] += int(c)
    if tally["left"] != tally["right"]:
        return "left" if tally["left"] > tally["right"] else "right"
    return "left" if float(centroid[0]) > axis else "right"


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def partition(
    decomp: Decomposition,
    parts: list[Part],
    rig: Rig,
    settings: RigSettings | None = None,
) -> tuple[list[Part], dict]:
    """Full requirement-2-2 pass. Returns (new part list, report)."""
    s = settings or RigSettings()
    cw, ch = decomp.canvas
    canvas_area = cw * ch
    min_area = max(48, int(0.0002 * canvas_area))

    silhouette = build_silhouette(decomp, parts)
    part_union = np.zeros_like(silhouette)
    for p in parts:
        part_union |= p.canvas_mask(decomp.canvas)
    grid = Partition(rig, silhouette | part_union)

    report: dict = {"skin_regions": {}, "lr_split": [], "garment_slices": {}, "forced": []}
    result: list[Part] = []

    # --- 1. skin -----------------------------------------------------------
    # Usually a no-op: on real see-through output the 24 layers already cover the
    # character, because handwear/legwear are whole limbs rather than garments.
    # It matters when the model does miss skin.
    skin = skin_base_mask(decomp, parts, silhouette, min_area)
    skin_area = int(skin.sum())
    report["skin_area_px"] = skin_area
    report["skin_coverage"] = skin_area / max(1, int(silhouette.sum()))

    if skin_area >= min_area and decomp.src_img is not None:
        depth = _skin_depth(parts)
        naming = taxonomy.PartNaming()
        skin_masks = grid.masks(None, seam_px=s.seam_allowance_px,
                                merge_limbs=s.merge_limb_slices)
        # Iterate the mask keys, not grid.specs: with merging the keys are limb
        # chains (``arm_r``) rather than segments (``arm_r_upper``).
        for region_name, region_mask in skin_masks.items():
            sel = _clean_mask(skin & region_mask, min_area=min_area, open_px=1)
            if not sel.any():
                continue
            p = _crop_to_part(
                naming.skin(region_name), decomp.src_img, sel, depth,
                synthetic=True, meta={"origin": "skin_base", "region": region_name},
                min_opaque=min_area,
            )
            if p is not None:
                result.append(p)
                report["skin_regions"][region_name] = int(sel.sum())

    # --- 2. per-layer handling --------------------------------------------
    already_split = {p.tag for p in parts if p.side is not None}

    for part in parts:
        tag = part.tag

        if tag in taxonomy.PAIRED_LR_TAGS and part.side is None and tag not in already_split:
            pieces = split_lr(part, decomp, rig, grid, s.min_slice_fraction)
            if len(pieces) > 1:
                report["lr_split"].append(part.name)
                # A trouser leg still spans hip and knee; keep cutting it.
                for piece in pieces:
                    result.extend(
                        _slice_by_regions(piece, decomp, grid, s, report,
                                          _union_of_others(part, parts, decomp))
                    )
                continue
            result.append(part)
            continue

        if tag in taxonomy.LIMB_SPANNING_TAGS:
            result.extend(
                _slice_by_regions(part, decomp, grid, s, report,
                                  _union_of_others(part, parts, decomp))
            )
            continue

        result.append(part)

    # --- 3. guarantee ------------------------------------------------------
    result = enforce_limb_coverage(result, decomp, grid, s, report)
    return result, report


def _union_of_others(part: Part, parts: list[Part], decomp: Decomposition) -> np.ndarray:
    """Everything except ``part``, as one mask.

    Used to test whether a large layer is genuinely the only one -- the condition
    the tag-restriction escape in ``_slice_by_regions`` is meant to detect.
    """
    out = np.zeros(decomp.canvas[::-1], dtype=bool)
    for q in parts:
        if q is not part:
            out |= q.canvas_mask(decomp.canvas)
    return out


def _slice_by_regions(
    part: Part,
    decomp: Decomposition,
    grid: Partition,
    s: RigSettings,
    report: dict,
    others: np.ndarray | None = None,
) -> list[Part]:
    """Cut a limb-spanning layer at its joints, within the regions its tag allows.

    A single attachment covering both arms, or spanning shoulder to wrist, cannot
    be rigged. Layers that turn out to sit inside one region are returned
    untouched.
    """
    mask = part.canvas_mask(decomp.canvas)
    total = int(mask.sum())
    if total == 0:
        return [part]

    allowed = taxonomy.allowed_regions(part.tag, part.side)
    # A layer that is most of the character is a whole-body layer, whatever the
    # model called it, and the per-tag restriction becomes counterproductive:
    # restricting a lone ``topwear`` covering head to toe to torso-plus-shoulders
    # leaves the legs uncuttable. The tag restriction exists to stop *one layer
    # among many* from claiming another limb's geometry, which cannot apply when
    # there is no other layer to claim it.
    #
    # Size alone does not establish that, though, and on a seated figure it is
    # actively wrong: a spread skirt covers 73.5% of the silhouette while the other
    # layers still cover 49.8% of it, so ``bottomwear`` was released from its tag
    # restriction and claimed the arm the sleeve was already covering. The
    # condition the comment above actually describes is *no other layer*, so test
    # that directly -- for a genuinely lone layer the others cover nothing.
    silhouette_px = int(grid.silhouette.sum())
    if silhouette_px and total >= 0.55 * silhouette_px:
        others_frac = 0.0 if others is None else float(others.sum()) / silhouette_px
        if others_frac < s.lone_layer_others_max:
            allowed = None

    masks = grid.masks(allowed, seam_px=s.seam_allowance_px, within=mask,
                       merge_limbs=s.merge_limb_slices)

    hits = [(name, mask & m) for name, m in masks.items()]
    hits = [(name, m) for name, m in hits if m.sum() >= s.min_slice_fraction * total]
    if not hits:
        return [part]
    if len(hits) == 1:
        # Merging turns a limb layer -- ``handwear-r`` is a whole arm -- into a
        # single hit, and returning it unchanged would drop the ``@region`` suffix
        # that binds it to its limb bone and that verify_limb_separation counts.
        # So a lone limb hit is still emitted as an annotated slice; a lone torso
        # hit is left alone, because renaming e.g. ``back hair`` to
        # ``back hair@torso`` would rebind it away from TAG_TO_BONE.
        if hits[0][0] not in taxonomy.LIMB_CHAINS:
            return [part]

    canvas_rgba = part.canvas_rgba(decomp.canvas)
    naming = taxonomy.PartNaming()
    out: list[Part] = []
    for name, m in hits:
        p = _crop_to_part(
            naming.garment(part.name, name), canvas_rgba, m, part.depth_median,
            synthetic=True, meta={**part.meta, "sliced_from": part.name, "region": name},
        )
        if p is not None:
            out.append(p)

    if not out:
        return [part]
    report["garment_slices"][part.name] = [p.name for p in out]
    return out


def enforce_limb_coverage(
    parts: list[Part],
    decomp: Decomposition,
    grid: Partition,
    s: RigSettings,
    report: dict,
) -> list[Part]:
    """Make requirement 2-2 unconditional.

    If any mandatory limb region ended up with no part - the model produced no
    skin there and no layer was cut into it - take whichever remaining layer
    covers that region most and carve the region out of it. Better a slightly
    over-eager cut than an arm that cannot bend.
    """
    covered = {p.region for p in parts if p.region}
    out = list(parts)
    masks = grid.masks(None, seam_px=s.seam_allowance_px,
                       merge_limbs=s.merge_limb_slices)

    for region in mandatory_limb_regions(s):
        if region in covered or region not in masks:
            continue
        target_mask = masks[region]
        if not target_mask.any():
            report["forced"].append({"region": region, "reason": "region_empty"})
            continue

        # Already-sliced parts are eligible sources: after slicing, *every* part
        # has a region, so skipping them left nothing to carve from and the net
        # reported "no_source_layer" for six regions at once.
        best: tuple[int, Part] | None = None
        for p in out:
            if p.region == region:
                continue
            if p.tag in _HEAD_TAGS:
                continue  # head parts never belong to a limb
            overlap = int((p.canvas_mask(decomp.canvas) & target_mask).sum())
            if overlap > 0 and (best is None or overlap > best[0]):
                best = (overlap, p)

        if best is None:
            report["forced"].append({"region": region, "reason": "no_source_layer"})
            continue

        overlap, source = best
        src_mask = source.canvas_mask(decomp.canvas)
        carve = src_mask & target_mask
        remainder = src_mask & ~target_mask
        canvas_rgba = source.canvas_rgba(decomp.canvas)

        naming = taxonomy.PartNaming()
        carved = _crop_to_part(
            naming.garment(source.name, region), canvas_rgba, carve, source.depth_median,
            synthetic=True, meta={**source.meta, "sliced_from": source.name,
                                  "region": region, "forced": True},
        )
        if carved is None:
            report["forced"].append({"region": region, "reason": "carve_empty"})
            continue

        out.remove(source)
        out.append(carved)
        rest = _crop_to_part(
            source.name, canvas_rgba, remainder, source.depth_median,
            synthetic=source.synthetic, meta=source.meta,
        )
        if rest is not None:
            out.append(rest)

        report["forced"].append({
            "region": region, "source": source.name,
            "carved": carved.name, "px": int(carve.sum()),
        })
        covered.add(region)

    return out


def verify_limb_separation(
    parts: list[Part], settings: RigSettings | None = None
) -> dict:
    """Post-condition check used by tests and surfaced in the pipeline report.

    Checks what requirement 2-2 actually asks: that each of the four limbs -- left
    and right, arms and legs -- is present as its own part or parts. It used to
    demand all eight *segments* separately, which additionally forced a cut at
    every joint and with it a visible seam. A merged limb satisfies the
    requirement, so a merged region counts for its members.
    """
    s = settings or RigSettings()
    present = {p.region for p in parts if p.region}
    # A merged limb covers its segments; a segment covers its merged limb.
    for region in list(present):
        merged = taxonomy.merged_region_of(region)
        if merged:
            present.add(merged)
        present.update(taxonomy.LIMB_CHAINS.get(region, ()))

    missing = [r for r in mandatory_limb_regions(s) if r not in present]
    sides: dict[str, int] = {"left": 0, "right": 0}
    for p in parts:
        side = p.side
        if side in sides:
            sides[side] += 1
    return {"ok": not missing, "missing_regions": missing, "parts_per_side": sides}
