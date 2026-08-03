"""Requirement 1 (first half) - produce a plausible starting skeleton.

The user drags joints to their final position, so this only has to land close
enough to be worth correcting rather than redoing. The approach is geodesic
rather than proportional, which is what makes it survive non-T-poses (the
reference image the user supplied is a seated figure with one arm raised):

1. Build the silhouette and seed it at the torso.
2. Walk geodesic distance *inside* the silhouette with ``skimage.graph.MCP_Geometric``.
3. The far extremities of that walk are the head crown, hands and feet.
4. Classify them by where their path leaves the torso, not by absolute position -
   arms exit high on the torso and legs exit low, whatever the pose.
5. Read the mid joints (elbow, knee) off the geodesic path itself, so an elbow
   lands on the arm even when the arm is bent double.

Anything that cannot be found this way falls back to ``BoneSpec.hint``, the
proportional T-pose placement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import taxonomy
from .psd_io import Decomposition, Part

_HEAD_TAGS = ("face", "front hair", "headwear")
_EYE_TAGS = ("irides", "eyewhite", "eyelash")
_TORSO_TAGS = ("topwear", "neckwear")


@dataclass
class Rig:
    """Joint positions in canvas pixel coordinates, y down."""

    canvas: tuple[int, int]
    bones: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: Provenance per bone: "geodesic" | "layer" | "hint" | "user".
    source: dict[str, str] = field(default_factory=dict)

    def names(self) -> set[str]:
        return set(self.bones)

    def pos(self, name: str) -> tuple[float, float] | None:
        return self.bones.get(name)

    def parent_of(self, name: str) -> str | None:
        spec = taxonomy.BONES_BY_NAME.get(name)
        return spec.parent if spec else None

    def to_dict(self) -> dict:
        return {
            "canvas": {"width": self.canvas[0], "height": self.canvas[1]},
            "bones": [
                {
                    "name": b.name,
                    "parent": b.parent,
                    "x": round(self.bones[b.name][0], 2),
                    "y": round(self.bones[b.name][1], 2),
                    "optional": b.optional,
                    "source": self.source.get(b.name, "hint"),
                }
                for b in taxonomy.BONE_TEMPLATE
                if b.name in self.bones
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Rig":
        canvas = (int(data["canvas"]["width"]), int(data["canvas"]["height"]))
        rig = cls(canvas=canvas)
        for entry in data.get("bones", []):
            name = entry["name"]
            if name not in taxonomy.BONES_BY_NAME:
                continue
            rig.bones[name] = (float(entry["x"]), float(entry["y"]))
            rig.source[name] = entry.get("source", "user")
        return rig

    def segment(self, region: taxonomy.RegionSpec) -> tuple[np.ndarray, np.ndarray] | None:
        """Influence segment for a partition region, extrapolating open ends.

        When ``to_bone`` is absent (an optional hand/foot the user never placed)
        the segment continues past ``bone`` along the direction it arrived from,
        so a forearm region still covers the hand.
        """
        start = self.bones.get(region.bone)
        if start is None:
            return None
        a = np.array(start, dtype=np.float64)

        end = self.bones.get(region.to_bone) if region.to_bone else None
        if end is not None:
            return a, np.array(end, dtype=np.float64)

        parent = self.parent_of(region.bone)
        prev = self.bones.get(parent) if parent else None
        if prev is None:
            return a, a + np.array([0.0, 1.0])
        direction = a - np.array(prev, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return a, a + np.array([0.0, 1.0])
        return a, a + direction  # same length again, in the same direction


# --------------------------------------------------------------------------
# silhouette helpers
# --------------------------------------------------------------------------


def build_silhouette(decomp: Decomposition, parts: list[Part] | None = None) -> np.ndarray:
    """Character mask on the canvas. See ``ocs.silhouette`` for why this is hard.

    Short version: ``src_img.png``'s alpha is only a padding mask when the input
    was RGB, and the union of layers misses all bare skin, so neither can be
    used directly.
    """
    from .silhouette import character_mask

    return character_mask(decomp, parts).mask


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _tag_mask(decomp: Decomposition, parts: list[Part], tags: tuple[str, ...]) -> np.ndarray:
    cw, ch = decomp.canvas
    out = np.zeros((ch, cw), dtype=bool)
    for p in parts:
        if p.tag in tags:
            out |= p.canvas_mask(decomp.canvas)
    return out


def _nearest_inside(mask: np.ndarray, point: tuple[float, float]) -> tuple[int, int]:
    """Snap a point into the mask (BFS-free: pick the nearest true pixel)."""
    ys, xs = np.nonzero(mask)
    d = (xs - point[0]) ** 2 + (ys - point[1]) ** 2
    i = int(np.argmin(d))
    return int(xs[i]), int(ys[i])


# --------------------------------------------------------------------------
# geodesic extremity search
# --------------------------------------------------------------------------


def _geodesic(mask: np.ndarray, seed: tuple[int, int]):
    """(cumulative cost array, MCP object) for a walk seeded inside ``mask``."""
    from skimage.graph import MCP_Geometric

    cost = np.where(mask, 1.0, np.inf)
    mcp = MCP_Geometric(cost, fully_connected=True)
    costs, _ = mcp.find_costs([(seed[1], seed[0])])   # MCP is (row, col)
    costs = np.asarray(costs, dtype=np.float64)
    costs[~np.isfinite(costs)] = -1.0
    return costs, mcp


def _extremities(costs: np.ndarray, window: float, min_frac: float = 0.30) -> list[tuple[int, int]]:
    """Limb tips as *regional* maxima of geodesic distance.

    Taking the globally farthest N points instead does not work, and failed
    concretely: on a standing figure the legs are so much longer than the arms
    that the first four global maxima are all on the legs, and the hands are
    never proposed. A tip is properly characterised as a point no neighbour
    exceeds, whatever its absolute distance, which ``maximum_filter`` gives
    directly.
    """
    from scipy.ndimage import maximum_filter

    reach = costs.max()
    if reach <= 0:
        return []
    k = max(3, int(window) | 1)
    peaks = (costs >= maximum_filter(costs, size=k)) & (costs > min_frac * reach)
    if not peaks.any():
        return []

    n, labels, _stats, centroids = cv2.connectedComponentsWithStats(peaks.astype(np.uint8), 8)
    out: list[tuple[int, int]] = []
    for i in range(1, n):
        blob = labels == i
        idx = int(np.argmax(np.where(blob, costs, -1.0)))
        y, x = divmod(idx, costs.shape[1])
        out.append((x, y))
    out.sort(key=lambda p: -costs[p[1], p[0]])
    return out


def _trace_limb(
    mask: np.ndarray, anchor: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    """(root, mid, tip) for one limb, walked along the limb's own geodesic.

    Reading the mid joint off the arc length rather than interpolating between
    root and tip is what puts an elbow on a bent arm instead of in mid-air.
    """
    if not mask.any():
        return None
    seed = _nearest_inside(mask, anchor)
    try:
        costs, mcp = _geodesic(mask, seed)
    except Exception:
        return None
    if costs.max() <= 0:
        return None

    tip_idx = int(np.argmax(costs))
    ty, tx = divmod(tip_idx, costs.shape[1])
    path = _path(mcp, (tx, ty))
    if path.shape[0] < 2:
        return None

    root = _along(path, 0.05) or (float(seed[0]), float(seed[1]))
    mid = _along(path, 0.50) or root
    return root, mid, (float(tx), float(ty))


def _limb_masks(
    decomp: Decomposition, parts: list[Part], tags: tuple[str, ...], midline: float
) -> dict[str, np.ndarray]:
    """Per-side masks for a limb pair, using see-through's own ``-l``/``-r`` split.

    Where see-through already split the tag the suffix is authoritative (verified
    against ``label_lr_split``: the lower-centroid-x component is ``-r``, the
    character's right). Otherwise the layer covers both sides and is divided by
    connected components, falling back to the midline for a single blob.
    """
    cw, ch = decomp.canvas
    sides: dict[str, np.ndarray] = {
        "left": np.zeros((ch, cw), bool), "right": np.zeros((ch, cw), bool)
    }
    unsided: list[Part] = []

    for p in parts:
        if p.tag not in tags:
            continue
        if p.side in sides:
            sides[p.side] |= p.canvas_mask(decomp.canvas)
        else:
            unsided.append(p)

    for p in unsided:
        m = p.canvas_mask(decomp.canvas)
        total = int(m.sum())
        if total == 0:
            continue
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.10 * total]
        if len(big) >= 2:
            for i in range(1, n):
                side = "left" if centroids[i][0] > midline else "right"
                sides[side] |= labels == i
        else:
            ys, xs = np.nonzero(m)
            is_left = xs > midline
            sides["left"][ys[is_left], xs[is_left]] = True
            sides["right"][ys[~is_left], xs[~is_left]] = True

    # Close each side so a sandal separated from the shin by antialiasing still
    # sits on the same geodesic walk as the leg.
    k = max(3, (min(cw, ch) // 160) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for side, m in sides.items():
        if m.any():
            sides[side] = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    return sides


def _path(mcp, end: tuple[int, int]) -> np.ndarray:
    """Geodesic path seed -> end as an (N, 2) array of (x, y)."""
    try:
        rc = mcp.traceback((end[1], end[0]))
    except ValueError:
        return np.empty((0, 2), dtype=np.int64)
    arr = np.asarray(rc, dtype=np.int64)
    return arr[:, ::-1] if arr.size else arr


def _along(path: np.ndarray, t: float) -> tuple[float, float] | None:
    """Point at fraction ``t`` of the path's arc length."""
    if path.shape[0] == 0:
        return None
    if path.shape[0] == 1:
        return float(path[0, 0]), float(path[0, 1])
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    target = t * cum[-1]
    i = int(np.searchsorted(cum, target))
    i = min(max(i, 1), len(cum) - 1)
    span = cum[i] - cum[i - 1]
    f = 0.0 if span < 1e-9 else (target - cum[i - 1]) / span
    p = path[i - 1] + (path[i] - path[i - 1]) * f
    return float(p[0]), float(p[1])


def _exit_point(path: np.ndarray, center: np.ndarray, radius: float) -> tuple[float, float] | None:
    """Where the path first leaves a disk around the torso.

    Classifying arms vs legs by this point (rather than by the tip's absolute
    position) is what makes the guess pose-independent: arms leave the torso
    high, legs leave it low, no matter where the hand or foot ends up.
    """
    if path.shape[0] == 0:
        return None
    d = np.linalg.norm(path - center[None, :], axis=1)
    outside = np.nonzero(d > radius)[0]
    if outside.size == 0:
        return float(path[-1, 0]), float(path[-1, 1])
    i = int(outside[0])
    return float(path[i, 0]), float(path[i, 1])


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------


def guess_rig(decomp: Decomposition, parts: list[Part] | None = None) -> Rig:
    """Best-effort initial skeleton. Never raises; always returns a full rig."""
    cw, ch = decomp.canvas
    parts = parts if parts is not None else decomp.parts
    rig = Rig(canvas=(cw, ch))

    silhouette = build_silhouette(decomp, parts)
    if not silhouette.any():
        _fill_from_hints(rig, (0, 0, cw, ch))
        return rig

    ys, xs = np.nonzero(silhouette)
    body_box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    bw, bh = body_box[2] - body_box[0], body_box[3] - body_box[1]

    # --- anchors that come straight off named layers ----------------------
    head_mask = _tag_mask(decomp, parts, _HEAD_TAGS)
    eye_mask = _tag_mask(decomp, parts, _EYE_TAGS)
    torso_mask = _tag_mask(decomp, parts, _TORSO_TAGS)
    neck_mask = _tag_mask(decomp, parts, ("neck",))

    head = _centroid(head_mask)
    if head:
        rig.bones["head"], rig.source["head"] = head, "layer"

    eyes = _centroid(eye_mask)
    if eyes:
        rig.bones["eyes"], rig.source["eyes"] = eyes, "layer"

    torso = _centroid(torso_mask)
    if torso:
        rig.bones["torso"], rig.source["torso"] = torso, "layer"
    else:
        torso = (body_box[0] + bw * 0.5, body_box[1] + bh * 0.55)
        rig.bones["torso"], rig.source["torso"] = torso, "hint"

    neck = _centroid(neck_mask)
    if neck:
        rig.bones["neck"], rig.source["neck"] = neck, "layer"
    elif head:
        rig.bones["neck"] = ((head[0] + torso[0]) / 2, (head[1] + torso[1]) / 2)
        rig.source["neck"] = "hint"

    # --- limbs, from the layers that actually are limbs --------------------
    # ``handwear`` is the whole arm and ``legwear`` the whole leg (see
    # taxonomy.LIMB_SPANNING_TAGS for the measurements). Tracing those masks
    # individually beats searching the whole silhouette, because side and
    # arm-vs-leg identity come from the tag instead of being inferred.
    midline = torso[0]
    torso_box = _bbox(torso_mask) or body_box
    lower_box = _bbox(_tag_mask(decomp, parts, ("bottomwear",))) or torso_box

    arm_anchor = (midline, float(torso_box[1]))    # shoulder line: top of topwear
    leg_anchor = (midline, float(lower_box[1]))    # waist line: top of bottomwear

    traced: set[str] = set()
    for side, mask in _limb_masks(decomp, parts, taxonomy.ARM_TAGS, midline).items():
        found = _trace_limb(mask, arm_anchor)
        if found:
            _place_chain(rig, f"{side}Arm", f"{side}Elbow", f"{side}Hand", found, "layer")
            traced.add(f"{side}Arm")
    for side, mask in _limb_masks(decomp, parts, taxonomy.LEG_TAGS, midline).items():
        found = _trace_limb(mask, leg_anchor)
        if found:
            hip, knee, foot = found
            # The traced root sits at the top of the *visible* leg, i.e. the hem
            # of the shorts or skirt. The joint it should rotate about is the
            # pelvis, a little further up inside the garment.
            hip = (hip[0], hip[1] - 0.25 * max(0.0, hip[1] - leg_anchor[1]))
            _place_chain(rig, f"{side}Leg", f"{side}Knee", f"{side}Foot",
                         (hip, knee, foot), "layer")
            traced.add(f"{side}Leg")

    # --- fall back to a silhouette-wide search for anything still missing --
    if len(traced) < 4:
        _geodesic_fallback(rig, silhouette, torso, head, head_mask, body_box, traced)

    _fill_from_hints(rig, body_box)
    _enable_optional_bones(rig, decomp, parts, body_box)
    return rig


def _extent(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0, 0.0
    return float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _place_chain(
    rig: Rig, root_bone: str, mid_bone: str, tip_bone: str,
    joints: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    source: str,
) -> None:
    root, mid, tip = joints
    for name, pos in ((root_bone, root), (mid_bone, mid), (tip_bone, tip)):
        rig.bones[name] = (float(pos[0]), float(pos[1]))
        rig.source[name] = source


def _geodesic_fallback(
    rig: Rig,
    silhouette: np.ndarray,
    torso: tuple[float, float],
    head: tuple[float, float] | None,
    head_mask: np.ndarray,
    body_box: tuple[int, int, int, int],
    already: set[str],
) -> None:
    """Silhouette-wide search, for art where the limb tags are missing entirely.

    Less reliable than tracing the limb layers, so it only fills in chains the
    layer pass could not produce.
    """
    bw, bh = body_box[2] - body_box[0], body_box[3] - body_box[1]
    seed = _nearest_inside(silhouette, torso)
    torso_center = np.array([float(seed[0]), float(seed[1])], dtype=np.float64)
    try:
        costs, mcp = _geodesic(silhouette, seed)
    except Exception:
        return

    head_pt = np.array(head if head else (torso[0], body_box[1]), dtype=np.float64)
    head_radius = 0.5 * math.hypot(*_extent(head_mask)) if head_mask.any() else 0.12 * bh

    candidates: list[dict] = []
    for tip in _extremities(costs, window=0.09 * max(bw, bh)):
        if np.linalg.norm(np.array(tip, dtype=np.float64) - head_pt) < max(head_radius, 0.10 * bh):
            continue  # the head crown, not a limb
        path = _path(mcp, tip)
        if path.shape[0] < 3:
            continue
        exit_pt = _exit_point(path, torso_center, 0.16 * max(bw, bh))
        if exit_pt is None:
            continue
        candidates.append({"path": path, "tip": tip, "exit": exit_pt})

    if not candidates:
        return

    # Arms leave the torso above the legs. Classifying on the exit point rather
    # than the tip keeps this true for seated and crouching poses, where a hand
    # can end up lower than a foot.
    candidates.sort(key=lambda c: c["exit"][1])
    n_arms = min(2, max(0, len(candidates) - 2)) if len(candidates) >= 4 else 1
    groups = (("Arm", "Elbow", "Hand", candidates[:n_arms], 0.18, 0.60),
              ("Leg", "Knee", "Foot", candidates[n_arms:], 0.14, 0.55))

    axis = rig.bones.get("torso", (rig.canvas[0] / 2, 0))[0]
    for root_suffix, mid_suffix, tip_suffix, group, t_root, t_mid in groups:
        group = sorted(group, key=lambda c: c["tip"][0])
        assigned: dict[str, dict] = {}
        if len(group) >= 2:
            assigned = {"right": group[0], "left": group[-1]}
        elif len(group) == 1:
            side = "right" if group[0]["exit"][0] <= axis else "left"
            assigned = {side: group[0]}

        for side, limb in assigned.items():
            if f"{side}{root_suffix}" in already:
                continue
            root = _along(limb["path"], t_root)
            mid = _along(limb["path"], t_mid)
            if root is None or mid is None:
                continue
            _place_chain(
                rig, f"{side}{root_suffix}", f"{side}{mid_suffix}", f"{side}{tip_suffix}",
                (root, mid, (float(limb["tip"][0]), float(limb["tip"][1]))), "geodesic",
            )


def _fill_from_hints(rig: Rig, body_box: tuple[int, int, int, int]) -> None:
    """Guarantee every required bone exists, so requirement 2-2 can't be skipped."""
    x1, y1, x2, y2 = body_box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    for spec in taxonomy.BONE_TEMPLATE:
        if spec.optional or spec.name in rig.bones or spec.hint is None:
            continue
        fx, fy = spec.hint
        rig.bones[spec.name] = (x1 + bw * fx, y1 + bh * fy)
        rig.source[spec.name] = "hint"
    if "root" not in rig.bones:
        rig.bones["root"] = (x1 + bw * 0.5, float(y2))
        rig.source["root"] = "hint"


def _enable_optional_bones(
    rig: Rig, decomp: Decomposition, parts: list[Part], body_box: tuple[int, int, int, int]
) -> None:
    """Add tail / wings / hairBack only when a layer actually needs them."""
    present = {p.tag for p in parts}
    wanted = {"tail": "tail", "wings": "wings", "back hair": "hairBack"}
    for tag, bone in wanted.items():
        if tag not in present or bone in rig.bones:
            continue
        c = _centroid(_tag_mask(decomp, parts, (tag,)))
        if c:
            rig.bones[bone], rig.source[bone] = c, "layer"


def mirror_bone(rig: Rig, name: str) -> None:
    """Mirror one joint onto its L/R counterpart across the torso's x."""
    pairs = dict(taxonomy.MIRROR_PAIRS)
    pairs.update({v: k for k, v in taxonomy.MIRROR_PAIRS})
    other = pairs.get(name)
    if other is None or name not in rig.bones:
        return
    axis = rig.bones.get("torso", (rig.canvas[0] / 2, 0))[0]
    x, y = rig.bones[name]
    rig.bones[other] = (2 * axis - x, y)
    rig.source[other] = "user"
