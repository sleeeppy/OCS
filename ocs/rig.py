"""Requirement 3 - turn parts + joints into a weighted Spine rig.

Two things here are deliberately more than a minimal port of the prior art:

**Weighted mesh attachments, not region attachments.** A region attachment is a
rigid quad; an arm bound to one is a stiff plank that pivots at the shoulder and
tears at the elbow. Each deformable part instead becomes a triangulated mesh
whose vertices are weighted to the bones spanning the joint, so the elbow bends
the pixels. Small rigid features (irides, nose, mouth, hardware) stay regions -
meshing them would only add vertices.

**A real bone frame.** Bones get a setup rotation aimed at their child, lengths,
and parent-local offsets, so the export opens sensibly in the Spine editor
rather than as a fan of horizontal sticks. That means the world transform has to
be accumulated properly and mesh vertices baked into each bone's local space,
which is what ``_build_bone_transforms`` and ``_to_bone_local`` do.

Coordinate systems, since three are in play:

- **canvas px**: origin top-left, y down. What the PSD and the editor use.
- **spine world**: origin at the ``root`` joint, y up. ``px_to_spine``.
- **bone local**: spine world rotated/translated into a bone's setup frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import scipy.ndimage as ndi

from . import taxonomy
from .config import RigSettings
from .psd_io import Decomposition, Part
from .skeleton import Rig

#: Tags whose attachments deform and therefore need a weighted mesh.
MESH_TAGS = frozenset({
    "topwear", "bottomwear", "legwear", "footwear", "handwear",
    "neck", "back hair", "front hair", "tail", "wings", "face",
})

#: Bones that never aim at a child (no meaningful "down the bone" direction).
_AIM_UP = frozenset({"head", "eyes", "hairBack"})

#: Bones that sit almost on top of their parent, where the parent->self vector is
#: too short to give a stable angle. They inherit the parent's frame instead.
_INHERIT_ROT = frozenset({"eyes", "hairBack"})


# --------------------------------------------------------------------------


@dataclass
class BoneTransform:
    name: str
    parent: str | None
    local_x: float = 0.0
    local_y: float = 0.0
    local_rot: float = 0.0
    length: float = 0.0
    world_x: float = 0.0
    world_y: float = 0.0
    world_rot: float = 0.0

    def tip(self) -> tuple[float, float]:
        r = math.radians(self.world_rot)
        return (self.world_x + math.cos(r) * self.length,
                self.world_y + math.sin(r) * self.length)


@dataclass
class Attachment:
    name: str
    kind: str                       # "mesh" | "region"
    width: float
    height: float
    #: mesh only
    uvs: list[float] = field(default_factory=list)
    triangles: list[int] = field(default_factory=list)
    hull: int = 0
    edges: list[int] = field(default_factory=list)
    #: mesh: flattened [boneCount, boneIdx, x, y, weight, ...]; region: unused
    vertices: list[float] = field(default_factory=list)
    #: region only
    region_x: float = 0.0
    region_y: float = 0.0
    region_rotation: float = 0.0
    #: bone indices referenced by this attachment (mesh) or its parent bone
    bones_used: list[int] = field(default_factory=list)


@dataclass
class Slot:
    name: str
    bone: str
    attachment: str
    part_name: str
    depth_median: float
    z_prior: int


@dataclass
class RigResult:
    canvas: tuple[int, int]
    origin_px: tuple[float, float]
    bones: list[BoneTransform]
    slots: list[Slot]
    attachments: dict[str, Attachment]
    part_images: dict[str, Part]
    warnings: list[str] = field(default_factory=list)

    def bone_index(self) -> dict[str, int]:
        return {b.name: i for i, b in enumerate(self.bones)}


# --------------------------------------------------------------------------
# coordinate helpers
# --------------------------------------------------------------------------


def px_to_spine(p: tuple[float, float], origin: tuple[float, float]) -> tuple[float, float]:
    """canvas px (y down) -> spine world (y up), origin at the root joint."""
    return p[0] - origin[0], origin[1] - p[1]


def _rot(v: tuple[float, float], deg: float) -> tuple[float, float]:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return v[0] * c - v[1] * s, v[0] * s + v[1] * c


def _children_of(name: str, available: set[str]) -> list[str]:
    return [b.name for b in taxonomy.BONE_TEMPLATE
            if b.parent == name and b.name in available]


def _hierarchy_order(available: set[str]) -> list[str]:
    """Template order already lists parents first; filter to what exists."""
    return [b.name for b in taxonomy.BONE_TEMPLATE if b.name in available]


# --------------------------------------------------------------------------
# bones
# --------------------------------------------------------------------------


def _build_bone_transforms(rig: Rig) -> list[BoneTransform]:
    available = rig.names()
    order = _hierarchy_order(available)
    origin = rig.bones.get("root", (rig.canvas[0] / 2, rig.canvas[1]))

    world: dict[str, tuple[float, float]] = {
        n: px_to_spine(rig.bones[n], origin) for n in order
    }

    # --- world rotations: each bone aims at its first child ----------------
    world_rot: dict[str, float] = {}
    for name in order:
        if name == "root":
            world_rot[name] = 0.0
            continue

        spec = taxonomy.BONES_BY_NAME[name]
        if name in _INHERIT_ROT and spec.parent in world_rot:
            world_rot[name] = world_rot[spec.parent]
            continue

        target: tuple[float, float] | None = None
        if name not in _AIM_UP:
            kids = _children_of(name, available)
            if kids:
                target = world[kids[0]]

        here = world[name]
        if target is not None and math.dist(target, here) > 1e-6:
            world_rot[name] = math.degrees(math.atan2(target[1] - here[1], target[0] - here[0]))
        else:
            parent = spec.parent
            if parent in world and math.dist(world[parent], here) > 1e-6:
                # No child to aim at: keep going the way we came in, so a
                # forearm without a hand bone still points down the arm.
                world_rot[name] = math.degrees(
                    math.atan2(here[1] - world[parent][1], here[0] - world[parent][0])
                )
            else:
                world_rot[name] = 90.0

    # --- lengths ----------------------------------------------------------
    length: dict[str, float] = {}
    for name in order:
        kids = _children_of(name, available)
        if kids and name not in _AIM_UP:
            length[name] = math.dist(world[kids[0]], world[name])
        else:
            spec = taxonomy.BONES_BY_NAME[name]
            parent = spec.parent
            length[name] = math.dist(world[parent], world[name]) * 0.6 if parent in world else 0.0

    out: list[BoneTransform] = []
    for name in order:
        spec = taxonomy.BONES_BY_NAME[name]
        parent = spec.parent if spec.parent in available else None
        if parent is None:
            lx, ly, lr = 0.0, 0.0, world_rot[name]
        else:
            dx = world[name][0] - world[parent][0]
            dy = world[name][1] - world[parent][1]
            lx, ly = _rot((dx, dy), -world_rot[parent])
            lr = _norm_deg(world_rot[name] - world_rot[parent])
        out.append(BoneTransform(
            name=name, parent=parent,
            local_x=lx, local_y=ly, local_rot=lr, length=length[name],
            world_x=world[name][0], world_y=world[name][1], world_rot=world_rot[name],
        ))
    return out


def _norm_deg(d: float) -> float:
    while d > 180:
        d -= 360
    while d <= -180:
        d += 360
    return d


def _to_bone_local(pt: tuple[float, float], bone: BoneTransform) -> tuple[float, float]:
    return _rot((pt[0] - bone.world_x, pt[1] - bone.world_y), -bone.world_rot)


# --------------------------------------------------------------------------
# mesh construction
# --------------------------------------------------------------------------


def _contour_points(
    mask: np.ndarray, epsilon_frac: float, epsilon_max_px: float
) -> tuple[np.ndarray, int]:
    """Simplified outlines of **every** component. Returns (points, hull_count).

    Taking only the largest contour loses whole pieces of a part. Measured on a
    synthesised skin region: three components, and keeping the biggest covered
    4438 of 11078 pixels, so the finished mesh reproduced 71% of its own artwork.
    Delaunay will bridge between components, but ``_triangulate`` drops the
    bridging triangles because their centres fall outside the mask.

    ``hull_count`` is the largest outline's point count, and those points come
    first so Spine's ``hull`` field stays meaningful.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.empty((0, 2), np.float32), 0

    ordered = sorted(contours, key=cv2.contourArea, reverse=True)
    total = float(mask.sum())
    groups: list[np.ndarray] = []
    for contour in ordered:
        if cv2.contourArea(contour) < max(12.0, 0.005 * total):
            continue
        peri = cv2.arcLength(contour, True)
        # Cap the tolerance in absolute pixels. A fraction of the perimeter alone
        # erases small or thin parts: a ribbon 200 px around gets an 8 px
        # tolerance, enough to collapse it below three points.
        eps = max(1.0, min(epsilon_frac * peri, epsilon_max_px))
        pts = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2).astype(np.float32)
        if pts.shape[0] >= 3:
            groups.append(pts)

    if not groups:
        return np.empty((0, 2), np.float32), 0
    return np.concatenate(groups, axis=0), int(groups[0].shape[0])


def _interior_points(mask: np.ndarray, spacing: int) -> np.ndarray:
    """Grid samples strictly inside the mask, so bends have vertices to move."""
    if spacing < 2:
        return np.empty((0, 2), np.float32)
    h, w = mask.shape
    # Scale the grid to the part. A fixed 28 px step leaves a 40 px-wide arm with
    # a single interior vertex, which is not enough to bend smoothly.
    spacing = max(6, min(spacing, max(h, w) // 6 or spacing))
    eroded = cv2.erode(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ) > 0
    if not eroded.any():
        return np.empty((0, 2), np.float32)
    ys = np.arange(spacing // 2, h, spacing)
    xs = np.arange(spacing // 2, w, spacing)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    keep = eroded[gy, gx]
    return np.stack([gx[keep], gy[keep]], axis=1).astype(np.float32)


def _triangulate(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Delaunay, minus triangles whose centre falls outside the shape.

    The filter is what keeps concave parts (a bent arm, a skirt with a gap)
    from being bridged by phantom geometry.
    """
    from scipy.spatial import Delaunay, QhullError

    if points.shape[0] < 3:
        return np.empty((0, 3), np.int32)
    try:
        tri = Delaunay(points.astype(np.float64))
    except (QhullError, ValueError):
        return np.empty((0, 3), np.int32)

    simplices = tri.simplices
    if simplices.size == 0:
        return np.empty((0, 3), np.int32)

    centres = points[simplices].mean(axis=1)
    h, w = mask.shape
    cx = np.clip(centres[:, 0].round().astype(int), 0, w - 1)
    cy = np.clip(centres[:, 1].round().astype(int), 0, h - 1)
    inside = mask[cy, cx]

    # Also drop needle triangles; they produce shading artifacts when deformed.
    a, b, c = points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]]
    area = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / 2.0
    return simplices[inside & (area > 1.0)].astype(np.int32)


def _candidate_bones(part: Part, rig: Rig, primary: str) -> list[str]:
    """Bones allowed to influence this part.

    Starts from the local chain -- the part's own bone, its parent (so the
    shoulder end of an upper arm blends into the torso instead of detaching) and
    its children -- then drops any bone belonging to a region the part's tag
    cannot occupy.

    That second filter matters. Without it the shorts came out weighted to
    ``rightArm`` and ``leftArm``, because with the arms hanging at the sides the
    hands are genuinely the nearest bones to the hip, and inverse-distance
    weighting has no notion that a garment on the pelvis should not follow a
    hand. The same ``taxonomy.allowed_regions`` table that constrains the
    partition constrains the weighting.
    """
    available = rig.names()
    chain = [primary]
    spec = taxonomy.BONES_BY_NAME.get(primary)
    parent = spec.parent if spec else None
    if parent in available:
        chain.append(parent)
    chain.extend(_children_of(primary, available))

    allowed = taxonomy.allowed_regions(part.tag, part.side)
    if allowed is not None:
        region_bones = {s.bone for s in taxonomy.SKIN_REGIONS if s.name in allowed}
        all_region_bones = {s.bone for s in taxonomy.SKIN_REGIONS}
        forbidden = (all_region_bones - region_bones) - {parent}
        chain = [b for b in chain if b not in forbidden]

    out: list[str] = []
    for b in chain:
        if b in available and b not in out and b != "root":
            out.append(b)
    return out or [primary]


def _weights(
    world_pts: np.ndarray,
    bones: list[BoneTransform],
    candidates: list[str],
    s: RigSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """(bone-column indices, weights) for each vertex, top-K normalised."""
    by_name = {b.name: b for b in bones}
    cand = [by_name[c] for c in candidates if c in by_name]
    if not cand:
        return np.zeros((world_pts.shape[0], 1), int), np.ones((world_pts.shape[0], 1))

    dists = np.empty((len(cand), world_pts.shape[0]), dtype=np.float64)
    for i, bone in enumerate(cand):
        a = np.array([bone.world_x, bone.world_y])
        b = np.array(bone.tip())
        ab = b - a
        denom = float(ab @ ab)
        if denom < 1e-9:
            dists[i] = np.linalg.norm(world_pts - a[None, :], axis=1)
        else:
            t = np.clip(((world_pts - a[None, :]) @ ab) / denom, 0.0, 1.0)
            proj = a[None, :] + t[:, None] * ab[None, :]
            dists[i] = np.linalg.norm(world_pts - proj, axis=1)

    k = min(s.max_bones_per_vertex, len(cand))
    order = np.argsort(dists, axis=0)[:k]                       # (k, N)
    picked = np.take_along_axis(dists, order, axis=0)           # (k, N)

    w = 1.0 / np.power(np.maximum(picked, 1e-3), s.weight_falloff)
    # A vertex sitting on a bone belongs to that bone alone; blending there
    # smears the joint.
    pinned = picked[0] <= s.weight_pin_px
    w[:, pinned] = 0.0
    w[0, pinned] = 1.0
    w /= np.maximum(w.sum(axis=0, keepdims=True), 1e-12)

    return order.T, w.T                                        # (N, k) each


def _mesh_attachment(
    part: Part,
    slug: str,
    bones: list[BoneTransform],
    bone_index: dict[str, int],
    candidates: list[str],
    origin: tuple[float, float],
    s: RigSettings,
) -> Attachment | None:
    mask = part.mask()
    if not mask.any():
        return None

    # Trace outside the alpha. A polygon through the alpha boundary clips the
    # feathered antialiased rim, and losing a 1 px outline all the way round a
    # large part is most of its total ink loss (front hair: 1372 px).
    # Overshooting is free -- the extra band is transparent in the texture.
    #
    # One pixel is not enough where two slices of the same garment meet, though.
    # _triangulate drops boundary triangles (centre outside the mask, or needle
    # shaped), which costs each part 0.6-4.6% of its own alpha right at its edge
    # -- measured per part, worst topwear@arm_r_upper at 2.31%. Both sides of a
    # cut lose their edge, so the losses add and the seam shows. Dilating further
    # makes each mesh reach past its neighbour's loss instead.
    radius = max(1, int(s.outline_dilate_px))
    outline_mask = cv2.dilate(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2),
    ) > 0

    outline, n_hull = _contour_points(outline_mask, s.contour_epsilon, s.contour_epsilon_max_px)
    if outline.shape[0] < 3:
        return None
    interior = _interior_points(mask, s.interior_spacing)
    points = np.concatenate([outline, interior], axis=0) if interior.size else outline

    triangles = _triangulate(points, outline_mask)
    if triangles.shape[0] == 0:
        return None

    h, w = mask.shape
    ox, oy = part.offset

    uvs: list[float] = []
    for x, y in points:
        uvs.extend([float(x) / w, float(y) / h])

    world = np.array([
        px_to_spine((float(x) + ox, float(y) + oy), origin) for x, y in points
    ], dtype=np.float64)

    cols, weights = _weights(world, bones, candidates, s)
    cand_bones = [b for b in candidates if b in bone_index]
    by_name = {b.name: b for b in bones}

    vertices: list[float] = []
    used: set[int] = set()
    for vi in range(points.shape[0]):
        entries = [
            (cand_bones[int(cols[vi, ki])], float(weights[vi, ki]))
            for ki in range(cols.shape[1])
            if weights[vi, ki] > 1e-4
        ]
        if not entries:
            entries = [(cand_bones[0], 1.0)]
        total = sum(wt for _n, wt in entries)
        vertices.append(float(len(entries)))
        for bone_name, wt in entries:
            bone = by_name[bone_name]
            lx, ly = _to_bone_local((world[vi, 0], world[vi, 1]), bone)
            idx = bone_index[bone_name]
            used.add(idx)
            vertices.extend([float(idx), lx, ly, wt / total])

    edges: list[int] = []
    for i in range(n_hull):
        edges.extend([i * 2, ((i + 1) % n_hull) * 2])

    return Attachment(
        name=slug, kind="mesh", width=float(w), height=float(h),
        uvs=uvs, triangles=triangles.reshape(-1).tolist(),
        hull=n_hull, edges=edges, vertices=vertices,
        bones_used=sorted(used),
    )


def _region_attachment(
    part: Part, slug: str, bone: BoneTransform, origin: tuple[float, float]
) -> Attachment:
    """Rigid quad, centred on the part's own middle, expressed in bone space.

    ``rotation`` counter-rotates the bone's setup angle. A region attachment
    inherits its bone's world rotation, and OCS aims every bone down its own
    length, so without this the sprite is drawn turned by that angle -- a face
    tilted 73 degrees, a shin lying on its side. Meshes do not need it because
    their vertices are already baked into bone-local space.
    """
    w, h = part.size
    ox, oy = part.offset
    centre_px = (ox + w / 2.0, oy + h / 2.0)
    lx, ly = _to_bone_local(px_to_spine(centre_px, origin), bone)
    return Attachment(
        name=slug, kind="region", width=float(w), height=float(h),
        region_x=lx, region_y=ly, region_rotation=_norm_deg(-bone.world_rot),
    )


# --------------------------------------------------------------------------


def _resolve_draw_order(parts: list[Part], decomp: Decomposition) -> list[Part]:
    """Far-to-near order: see-through's depth, corrected where it is provably wrong.

    Depth stays the ranking signal because it is per-image and usually right. But
    it is only an estimate, and for thin overlapping accessories it inverts:
    measured on one character, ``headwear`` (a ribbon over the hair) ranked
    *behind* ``front hair``, so the hair drew last and hid 91% of the ribbon.

    ``taxonomy.DRAW_AFTER`` states the relationships that cannot be otherwise, and
    this is a topological sort over them with depth as the tie-break priority. A
    constraint only applies between parts whose masks actually overlap, so a hat
    and a shoe never constrain each other.
    """
    import heapq

    order_key = {
        p.name: (-p.depth_median, taxonomy.z_prior(p.name), p.name) for p in parts
    }
    fallback = sorted(parts, key=lambda p: order_key[p.name])
    if not parts:
        return fallback

    by_tag: dict[str, list[Part]] = {}
    for p in parts:
        by_tag.setdefault(p.tag, []).append(p)

    masks: dict[str, np.ndarray] = {}

    def mask_of(p: Part) -> np.ndarray:
        if p.name not in masks:
            masks[p.name] = p.canvas_mask(decomp.canvas)
        return masks[p.name]

    # edge blocker -> dependent: blocker must be drawn first
    successors: dict[str, list[str]] = {p.name: [] for p in parts}
    indegree: dict[str, int] = {p.name: 0 for p in parts}

    for p in parts:
        for blocker_tag in taxonomy.DRAW_AFTER.get(p.tag, ()):
            for blocker in by_tag.get(blocker_tag, ()):
                if blocker.name == p.name:
                    continue
                a, b = mask_of(p), mask_of(blocker)
                inter = int((a & b).sum())
                smaller = min(int(a.sum()), int(b.sum()))
                if smaller == 0 or inter < 0.05 * smaller:
                    continue  # they do not really overlap; order is irrelevant
                successors[blocker.name].append(p.name)
                indegree[p.name] += 1

    if not any(indegree.values()):
        return fallback

    part_by_name = {p.name: p for p in parts}
    heap = [(order_key[n], n) for n, d in indegree.items() if d == 0]
    heapq.heapify(heap)
    out: list[Part] = []
    while heap:
        _key, name = heapq.heappop(heap)
        out.append(part_by_name[name])
        for nxt in successors[name]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(heap, (order_key[nxt], nxt))

    if len(out) != len(parts):
        # A cycle in DRAW_AFTER would strand parts. Fall back rather than drop any.
        return fallback
    return out


def restore_source_pixels(
    ordered: list[Part], decomp: Decomposition, s: RigSettings
) -> list[Part]:
    """Repaint each part's visible pixels from the original artwork.

    see-through does not crop layers out of the input, it regenerates each one with
    a diffusion model, so the RGB it returns drifts from the source. Measured over
    one character: 11.8% of shared opaque pixels differ by more than 16 levels
    (mean 8.96), against 1.7% / 2.67% contributed by everything OCS does
    afterwards. That drift is what flattens hair shading and smudges a ribbon.

    Wherever a part is frontmost, the original composite already holds its exact
    pixels, so they can simply be copied back. Where a part is *behind* another,
    the original holds the occluder instead and the inpainted reconstruction is
    the only thing that knows what is under there -- which is precisely what a rig
    needs once the limb moves. So this walks near-to-far and only claims pixels no
    nearer part has taken.

    Upstream applies the same idea to nose and mouth in ``further_extr``.

    Returns new ``Part`` objects; the inputs are untouched.
    """
    if decomp.src_img is None or decomp.src_img.shape[2] < 4:
        return ordered

    cw, ch = decomp.canvas
    src = decomp.src_img
    claimed = np.zeros((ch, cw), dtype=bool)
    out: list[Part] = list(ordered)

    # Two thresholds, because the reason for a high one only holds in one place.
    #
    # A partly transparent pixel in the source is a blend with whatever is behind
    # it. At the character's outer edge that is the *background*, so copying it in
    # drags background colour into the rim. Everywhere else it is another part of
    # the character, which is the colour we want.
    #
    # Using one high floor for both leaves every interior seam un-repainted, and
    # those un-repainted feathered edges are the dark hairlines that read as
    # "visible cuts". Measured on this character, sweeping a single global floor:
    #
    #   floor   outer-rim px / mean    interior px / mean
    #     200        815 / -22.7          4963 / -19.0
    #      64        796 / -25.3          4119 / -17.3
    #
    # -- the interior improves and the rim gets worse, exactly as the original
    # comment predicted. Splitting the threshold takes the interior win without
    # paying for it at the rim.
    silhouette = np.zeros((ch, cw), dtype=bool)
    for part in ordered:
        silhouette |= part.canvas_mask(decomp.canvas)
    rim = silhouette & ~ndi.binary_erosion(
        silhouette, ndi.generate_binary_structure(2, 2), iterations=2
    )
    rim = ndi.binary_dilation(rim, iterations=1)

    # Near to far: the last-drawn part is the one you actually see.
    for i in range(len(ordered) - 1, -1, -1):
        part = ordered[i]
        x1, y1, x2, y2 = part.bbox
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(cw, x2), min(ch, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        sx, sy = cx1 - x1, cy1 - y1
        h, w = cy2 - cy1, cx2 - cx1
        alpha = part.alpha[sy:sy + h, sx:sx + w]

        floor = np.where(
            rim[cy1:cy2, cx1:cx2],
            s.source_pixel_alpha_floor,
            s.source_pixel_alpha_floor_interior,
        )
        solid = alpha >= floor
        visible = solid & ~claimed[cy1:cy2, cx1:cx2]
        # Claim a pixel only where this part actually hides what is behind it.
        #
        # This used to be ``alpha > 8``, which let a part's *feathered edge* --
        # alpha 9 is enough -- block the repaint of the part behind it, even
        # though at 3% opacity the front part contributes almost nothing there and
        # the back part is what you see. The back part then kept see-through's
        # drifted colour, which is darker, along every front part's outline.
        #
        # That is the dark hairline: measured over 7397 dark pixels, 4575 (61.8%)
        # sat on the frontmost part's own edge, while only 293 (4.0%) sat on an
        # edge behind. It is colour, not coverage -- alpha reconstruction is within
        # 0.95 of the source on all but 232 px of 72918.
        claimed[cy1:cy2, cx1:cx2] |= alpha >= s.source_pixel_claim_floor

        if not visible.any():
            continue
        rgba = part.rgba.copy()
        window = rgba[sy:sy + h, sx:sx + w, :3]
        window[visible] = src[cy1:cy2, cx1:cx2, :3][visible]
        out[i] = Part(
            name=part.name, rgba=rgba, offset=part.offset,
            depth_median=part.depth_median, depth=part.depth,
            synthetic=part.synthetic,
            meta={**part.meta, "source_pixels": int(visible.sum())},
        )
    return out


def _wants_mesh(part: Part) -> bool:
    # Anything the bone partition cut spans a joint by construction, and the
    # bulky garments deform. Everything else is a rigid feature.
    return part.region is not None or part.tag in MESH_TAGS


def build_rig(
    decomp: Decomposition,
    parts: list[Part],
    rig: Rig,
    settings: RigSettings | None = None,
) -> RigResult:
    """Bind parts to bones, build meshes and weights, resolve draw order."""
    s = settings or RigSettings()
    bones = _build_bone_transforms(rig)
    bone_index = {b.name: i for i, b in enumerate(bones)}
    origin = rig.bones.get("root", (rig.canvas[0] / 2, rig.canvas[1]))
    available = rig.names()

    naming = taxonomy.PartNaming()
    slots: list[Slot] = []
    attachments: dict[str, Attachment] = {}
    images: dict[str, Part] = {}
    warnings: list[str] = []

    ordered = _resolve_draw_order(parts, decomp)
    if s.restore_source_pixels:
        ordered = restore_source_pixels(ordered, decomp, s)

    for part in ordered:
        slug = naming.unique_slug(part.name)
        primary = taxonomy.bone_for_part(part.name, available)
        if primary not in bone_index:
            primary = "torso" if "torso" in bone_index else bones[0].name

        attachment: Attachment | None = None
        if _wants_mesh(part):
            attachment = _mesh_attachment(
                part, slug, bones, bone_index,
                _candidate_bones(part, rig, primary), origin, s,
            )
            if attachment is None:
                warnings.append(f"{part.name}: mesh build failed, fell back to region")

        if attachment is None:
            attachment = _region_attachment(part, slug, bones[bone_index[primary]], origin)

        attachments[slug] = attachment
        images[slug] = part
        slots.append(Slot(
            name=slug, bone=primary, attachment=slug, part_name=part.name,
            depth_median=part.depth_median, z_prior=taxonomy.z_prior(part.name),
        ))

    return RigResult(
        canvas=rig.canvas, origin_px=origin, bones=bones, slots=slots,
        attachments=attachments, part_images=images, warnings=warnings,
    )
