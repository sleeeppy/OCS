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
    #: How many mesh vertices ``weld_shared_vertices`` pooled across a seam.
    welded_vertices: int = 0

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

    # Holes are filled before triangulating, and this is not cosmetic.
    #
    # _contour_points traces RETR_EXTERNAL, so a hole contributes no vertices and
    # Delaunay spans it with whatever large triangles the surrounding grid gives.
    # _triangulate then drops every one of those whose centre lands in the hole --
    # and a triangle is far bigger than the hole that killed it, so the part loses
    # a wedge of coverage reaching well past the gap, with a hard straight edge.
    #
    # Measured on ``bottomwear@leg_l``: 8317 hole pixels cost it 2821 pixels of
    # mesh, 1508 of them in the small box around the hand resting on the skirt --
    # rendering as dark polygonal notches of background punched through the skirt.
    #
    # Spanning a hole costs nothing, because the texture there is transparent and
    # decides visibility on its own. This does not bridge *concave* boundaries,
    # which is the case ``_triangulate``'s centre test exists for; fill_holes only
    # closes regions already enclosed by the part.
    outline_mask = ndi.binary_fill_holes(outline_mask)

    # The simplification tolerance is what the dilation band can pay for.
    #
    # ``approxPolyDP``'s epsilon is the furthest the simplified polygon may sit
    # from the traced curve, and it cuts *inward* at every convex bump. The curve
    # was traced ``radius`` pixels outside the real alpha boundary, so a tolerance
    # up to ``radius`` is free -- the polygon still lands on or outside the alpha.
    # Beyond that it eats into the part, and those pixels have no geometry, so the
    # background shows through them. The default 4.0 against a 2 px dilation was
    # spending twice what it had.
    #
    # This is the dark line down the thigh. ``handwear-r`` is cut in two there --
    # arm above, leg below -- and both halves lost their edge to this, 357 px
    # between them, which is the seam. Capping at the radius takes it to 219.
    epsilon_max = min(s.contour_epsilon_max_px, float(radius))
    outline, n_hull = _contour_points(outline_mask, s.contour_epsilon, epsilon_max)
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


def infer_overlap_order(
    parts: list[Part], decomp: Decomposition, min_overlap: int = 400,
    min_margin: float = 8.0,
) -> list[tuple[Part, Part]]:
    """Ask the artwork which of two overlapping layers is in front.

    Only used when there is no depth to rank by. see-through's depth comes from
    ``apply_marigold``, which runs after the layer pass, so a run interrupted in
    between yields layers with no depth at all -- every ``depth_median`` is the
    same 1.0 default and the order collapses onto ``taxonomy.Z_PRIOR``, a table
    written for a standing figure. On a seated one that is wrong in a way you
    cannot miss: ``legwear`` sorts to -12 and ``bottomwear`` to -10, so a raised
    bare leg is drawn *behind* the skirt it is plainly in front of.

    The artwork settles it without any model. Where two layers overlap, only the
    front one is visible, so only the front one matches the source there. Measured
    on this character, mean absolute RGB error over the shared opaque pixels:

        legwear / bottomwear    21680 px      8.3  vs  120.2   -> legwear
        footwear / bottomwear    7567 px     10.1  vs   67.2   -> footwear
        bottomwear / handwear   97885 px     85.2  vs   34.0   -> handwear
        topwear / handwear      34086 px     82.2  vs   15.7   -> handwear

    Margins like that are not close calls. ``min_margin`` ignores the ones that
    are, so a genuinely ambiguous pair keeps whatever order it already had.

    Returns ``(front, back)`` pairs for the topological sort to consume.
    """
    if decomp.src_img is None or decomp.src_img.shape[2] < 4:
        return []
    depths = {round(p.depth_median, 6) for p in parts}
    if len(depths) > 1:
        return []          # a real depth pass ran; trust it

    src = decomp.src_img[..., :3].astype(np.float32)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def solid_rgb(p: Part):
        if p.name not in cache:
            blk = p.canvas_rgba(decomp.canvas)
            cache[p.name] = (blk[..., 3] > 200, blk[..., :3].astype(np.float32))
        return cache[p.name]

    out: list[tuple[Part, Part]] = []
    for i, a in enumerate(parts):
        ma, ra = solid_rgb(a)
        for b in parts[i + 1:]:
            mb, rb = solid_rgb(b)
            ov = ma & mb
            n = int(ov.sum())
            if n < min_overlap:
                continue
            ea = float(np.abs(ra[ov] - src[ov]).mean())
            eb = float(np.abs(rb[ov] - src[ov]).mean())
            if abs(ea - eb) < min_margin:
                continue
            out.append((a, b) if ea < eb else (b, a))
    return out


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

    def constrain(front: Part, back: Part) -> bool:
        a, b = mask_of(front), mask_of(back)
        inter = int((a & b).sum())
        smaller = min(int(a.sum()), int(b.sum()))
        if smaller == 0 or inter < 0.05 * smaller:
            return False  # they do not really overlap; order is irrelevant
        successors[back.name].append(front.name)
        indegree[front.name] += 1
        return True

    for p in parts:
        for blocker_tag in taxonomy.DRAW_AFTER.get(p.tag, ()):
            for blocker in by_tag.get(blocker_tag, ()):
                if blocker.name != p.name:
                    constrain(p, blocker)

    # Inferred edges are advisory and go in second, because they are read off the
    # artwork and can be wrong where a layer was inpainted to look like what is on
    # top of it -- measured here, the overlap test claimed ``face`` was in front of
    # ``eyewhite``, the exact reverse of what DRAW_AFTER states. Contradicting a
    # curated edge creates a cycle, and a cycle makes the whole sort fall back, so
    # a single bad guess used to discard every good one with it. Now the curated
    # table wins and anything that would close a loop is dropped.
    def would_cycle(front: str, back: str) -> bool:
        stack, seen = [front], {front}
        while stack:
            n = stack.pop()
            if n == back:
                return True
            for nxt in successors[n]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    for front, back in infer_overlap_order(parts, decomp):
        if front.name in successors[back.name]:
            continue                       # already stated
        if would_cycle(front.name, back.name):
            continue
        constrain(front, back)

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


def close_layer_seams(
    ordered: list[Part], decomp: Decomposition, s: RigSettings
) -> list[Part]:
    """Make every layer opaque a few pixels past its own edge, inside the artwork.

    This is the fix for the visible cuts, and it follows from the arithmetic rather
    than from tuning.

    In the flat artwork a pixel on the boundary between the collar and the sleeve
    is one **opaque** blend of the two. Split into layers, each gets about half
    alpha there, and straight-alpha compositing gives

        1 - (1 - 0.5)(1 - 0.5) = 0.75

    so the artwork's 1.0 comes back as 0.75 and the background shows through the
    missing quarter -- a dark line along every internal layer boundary. Splitting
    an antialiased image into layers is not invertible by compositing it back.

    Raising the *front* layer to opaque closes the alpha gap but destroys the
    blend: the composite then shows the front layer's colour alone instead of
    ``0.5*collar + 0.5*sleeve``. Extending the layer *behind* is what recovers
    both, because

        A(0.5) over B(1.0)  ->  alpha 1.0,  colour 0.5*A + 0.5*B

    which is exactly what the artwork had. see-through inpaints each layer to be
    complete, so the pixels underneath are already there; only their alpha tapers
    off at the layer's own boundary, and this fills that back in.

    Measured against the artwork composited over the same background:

        extend   >8 diff   >32 diff   mean darkness
             0      4942       1206           -9.8
             2      4065        969           -4.5
             4      3944        958           -3.7
             8      3936        958           -3.7

    It plateaus at four pixels, which is the default. Darkness -- the thing that
    reads as a seam -- drops 62%.

    Only where the *artwork* is solid. The character's own outline is legitimately
    semi-transparent and must stay that way, or the silhouette gains a hard fringe
    and the figure grows by the extension width.

    And only into two kinds of place, because "a few pixels past the edge" is right
    at a seam and wrong everywhere else.

    **Across the part's own feathered rim** -- pixels it already has some alpha at.
    The arithmetic above is about a boundary the split turned from one opaque pixel
    into two half-transparent ones, which always leaves a band of partial alpha to
    find.

    **Into a sibling slice of the same layer** -- a piece carrying the same base
    tag. Where OCS itself cut a layer the split is hard, not feathered: one piece
    ends at alpha 255 and the next begins at 255, so there is no rim to find and the
    first test alone declines to extend. On the canvas that is fine, the two alphas
    sum to exactly 1. On the GPU it is not: the texture is sampled bilinearly, so
    across the cut one piece's alpha ramps 255 -> 0 over a texel while the other
    ramps 0 -> 255, and halfway along both read about 0.5, for a composite of
    1 - (1 - 0.5)(1 - 0.5) = 0.75. The background comes through the missing quarter
    as a hard line, invisible in any nearest-sample check of the layers and obvious
    in the player. Overlapping the pieces by a few opaque pixels means every sample
    point has at least one of them at 1.

    Confirmed rather than inferred: painting the player's background magenta turned
    the dark line down the thigh magenta, where ``handwear-r`` is cut into an arm
    piece and a leg piece.

    Anywhere else the part has no business growing. A gap it genuinely does not
    cover has alpha 0 and no sibling behind it, and filling that hides what is
    supposed to show through. Measured: without these tests the fill put 7469 opaque
    pixels into the box around the left hand, 3244 of them in ``handwear`` itself --
    the spaces between the fingers, packed with the layer's own inpainted
    background, which reads as black webbing between them.
    """
    radius = int(s.layer_extend_px)
    if radius <= 0 or decomp.src_img is None or decomp.src_img.shape[2] < 4:
        return ordered

    cw, ch = decomp.canvas
    solid = decomp.src_img[..., 3] >= s.source_alpha_solid_floor
    struct = ndi.generate_binary_structure(2, 2)

    # Where each layer's own slices ended up, so a piece can be told from the
    # sibling it was cut away from. Keyed by base tag: two slices of one garment
    # share it, and two different garments never do.
    siblings: dict[str, np.ndarray] = {}
    for part in ordered:
        tag = taxonomy.base_tag(part.name)
        acc = siblings.get(tag)
        if acc is None:
            acc = siblings[tag] = np.zeros((ch, cw), dtype=bool)
        acc |= part.canvas_rgba(decomp.canvas)[..., 3] >= s.source_alpha_solid_floor

    # Feathering already claimed by something further back, accumulated as the
    # draw order is walked. See the ``feathered_behind`` test below.
    behind = np.zeros((ch, cw), dtype=bool)

    out: list[Part] = []
    for part in ordered:
        alpha = part.rgba[..., 3]
        core = alpha >= s.source_alpha_solid_floor
        canvas_alpha = part.canvas_rgba(decomp.canvas)[..., 3]
        feathered_behind = behind.copy()
        behind |= ((canvas_alpha > s.source_alpha_touch_floor)
                   & (canvas_alpha < s.source_alpha_solid_floor))
        if not core.any():
            out.append(part)
            continue

        # Clip the canvas-sized masks into this part's own frame.
        x1, y1, _x2, _y2 = part.bbox
        h, w = alpha.shape
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(cw, x1 + w), min(ch, y1 + h)
        if cx2 <= cx1 or cy2 <= cy1:
            out.append(part)
            continue

        def _window(canvas_mask: np.ndarray) -> np.ndarray:
            win = np.zeros_like(core)
            win[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1] = canvas_mask[cy1:cy2, cx1:cx2]
            return win

        allowed = _window(solid)
        # The sibling's territory, minus this piece's own contribution to it.
        sibling = _window(siblings[taxonomy.base_tag(part.name)]) & ~core

        # Only into a sibling slice's territory, and never into a hole *enclosed*
        # by the part: the gaps between the fingers of a hand are interior holes a
        # few pixels wide, and an extension reaching in from both sides would meet
        # in the middle and seal them.
        #
        # Extending across the part's own feathered rim as well was tried and
        # removed. It is what the original arithmetic argued for, but with
        # ``restore_source_alpha`` now matching the artwork's opacity at every
        # level it buys nothing measurable -- and it costs a band of fully opaque
        # texels that travels with the part. Over the shin the sheer sleeve
        # carried 1653 such pixels: invisible at rest, because the colour repaint
        # gives them the artwork's value, and a hard-edged strip sliding down the
        # leg the moment the arm moves.
        #
        #                      band over leg   canvas >12   canvas >25   sibling
        #   rim | sibling               1653         1717          598     18705
        #   sibling only                 289         1701          582     18705
        #
        # Better on every count, and the cut-bridging that the sibling clause
        # exists for is untouched.
        enclosed = ndi.binary_fill_holes(core) & ~core

        # And not where something *behind* is feathering across the same pixel.
        #
        # The extension is meant for the layer behind, as the arithmetic above
        # says: raising the front one closes the alpha gap but replaces the blend
        # with the front layer's colour alone. Two parts feathering into each other
        # are not a cut, they are the artwork's own antialiasing between two
        # different objects, and raising both makes the later-drawn one win
        # outright -- which also hands it the pixels, and the bone they ride.
        #
        # That is the chin stuck to the hand. ``handwear-r@arm_r`` and ``face``
        # both taper off along the jaw; sealing raised both, the hand is drawn
        # after the face, and 139 px of jaw went with ``rightArm`` whenever the
        # arm moved. Letting the face -- the part behind -- own the blend puts
        # them back on ``head``.
        #
        # A cut is unaffected: the piece behind is either fully opaque there or
        # not present at all, never mid-taper, so nothing is skipped.
        grown = (ndi.binary_dilation(core, struct, iterations=radius)
                 & ~core & sibling & ~enclosed & ~_window(feathered_behind))

        target = grown & allowed
        if not target.any():
            out.append(part)
            continue

        rgba = part.rgba.copy()
        rgba[..., 3][target] = 255
        out.append(Part(
            name=part.name, rgba=rgba, offset=part.offset,
            depth_median=part.depth_median, depth=part.depth,
            synthetic=part.synthetic,
            meta={**part.meta, "seam_closed": int(target.sum())},
        ))
    return out


def restore_source_alpha(
    ordered: list[Part], decomp: Decomposition, s: RigSettings
) -> list[Part]:
    """Close the opacity the layer split lost, where the artwork was opaque.

    This is the dark hairline, and the cause is arithmetic rather than a bug in
    anything upstream. In the flat artwork, a pixel on the boundary between the
    collar and the sleeve is a *blend of the two* and fully opaque -- it is inside
    the character. Split into two layers, each gets roughly half alpha there.
    Composite them back with straight alpha and you get

        1 - (1 - 0.5)(1 - 0.5) = 0.75

    so a pixel the artwork had at 1.0 comes back at 0.75 and the background shows
    through the missing quarter. Every internal layer boundary becomes a dark line,
    which is exactly what "the cuts are visible" looks like.

    Measured over the 7274 pixels that render darker than the source: mean
    reconstruction alpha 130.6 against the original's 147.0, and on 3336 of them
    (45.9%) the original is more than 8 levels more opaque. Only 723 (9.9%) are
    fully opaque in both and merely the wrong colour.

    ``restore_source_pixels`` fixes colour and leaves alpha alone, so it cannot
    reach this. The repair is to give the frontmost part the alpha the artwork had:
    it is the part you see, the artwork says the pixel is solid, and raising it
    changes nothing anywhere the reconstruction was already opaque.

    Up to the artwork's own alpha, whatever that is -- not only where the artwork
    is fully opaque. Matching it cannot produce a hard fringe, because the
    artwork's alpha *is* the fringe. The reason a high floor exists belongs to the
    colour repaint instead: copying a rim's colour drags the background inward,
    copying a rim's opacity does not.
    """
    if decomp.src_img is None or decomp.src_img.shape[2] < 4:
        return ordered

    cw, ch = decomp.canvas
    src = decomp.src_img
    src_alpha = src[..., 3]

    # Composite alpha of everything, which is order-independent.
    acc = np.zeros((ch, cw), np.float64)
    for part in ordered:
        a = part.canvas_rgba(decomp.canvas)[..., 3].astype(np.float64) / 255.0
        acc = a + acc * (1.0 - a)

    # Anywhere the composite is short of the artwork, not only where the artwork
    # is fully opaque.
    #
    # The old gate was ``solid``, so a pixel the artist drew at alpha 200 was left
    # alone however far under it the layers came out. That is a whole band down
    # the edge of every sheer panel: measured on the leg, the artwork reads 131 to
    # 249 there and the layers sum to 56 to 199, and no stage touched it -- one is
    # gated on 250, the other only ever removes opacity. The result is a line
    # running the length of the shin.
    #
    # Matching the artwork's own alpha cannot produce a hard fringe, because the
    # artwork's alpha *is* the fringe. The reason for a high floor belongs to the
    # colour repaint, which is a separate step: copying a rim colour drags the
    # background in, copying a rim *opacity* does not.
    #
    # With ``limit_source_alpha`` capping from above, the pair now pins the
    # composite to the artwork from both sides.
    # Down to the faintest pixel the artist drew, not just the ones above the
    # touch floor.
    #
    # That floor is 8, and the outermost pixel of an antialiased silhouette is
    # routinely below it -- measured across the thigh, the artwork reads alpha 5
    # and 7 there. Skipping them dropped the whole outer pixel of the figure, so
    # every silhouette came out one pixel harder than it was drawn. Along a leg
    # that is a continuous line, and it reads as the limb being cut out rather
    # than painted.
    #
    # Reconstructing the artwork's own rim cannot make a hard edge, because that
    # rim *is* the soft edge.
    src_norm = src_alpha.astype(np.float64) / 255.0
    short = (acc < src_norm - 0.5 / 255.0) & (src_alpha > 0)
    if not short.any():
        return ordered

    out = list(ordered)
    # Near to far: the frontmost part owns the pixel, so it is the one to fix.
    fixed = np.zeros((ch, cw), dtype=bool)
    for i in range(len(ordered) - 1, -1, -1):
        part = ordered[i]
        x1, y1, x2, y2 = part.bbox
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(cw, x2), min(ch, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        sx, sy = cx1 - x1, cy1 - y1
        h, w = cy2 - cy1, cx2 - cx1

        window_short = short[cy1:cy2, cx1:cx2] & ~fixed[cy1:cy2, cx1:cx2]
        alpha = part.rgba[sy:sy + h, sx:sx + w, 3]
        # A pixel this part reaches, or is within a pixel of reaching. The strict
        # "must already have alpha here" test cannot rebuild the outer rim at all:
        # the pixel the artwork drew at alpha 5 is one the layers missed entirely,
        # so there is nothing there to raise. One pixel of reach is enough to put
        # it back and too little to grow the part anywhere it does not belong --
        # and the value written is the artwork's own, so the silhouette ends up
        # exactly the shape and softness it was drawn with.
        reach = ndi.binary_dilation(alpha > 0, ndi.generate_binary_structure(2, 2))
        target = window_short & reach
        if not target.any():
            continue

        rgba = part.rgba.copy()
        rgba[sy:sy + h, sx:sx + w, 3][target] = src_alpha[cy1:cy2, cx1:cx2][target]
        # Colour too, wherever the part had nothing here before. Alpha alone is
        # half a pixel: a rim texel the layers missed carries whatever colour the
        # part happened to hold there -- bled from somewhere else, or nothing --
        # and giving it the artwork's opacity without the artwork's colour made
        # the outer edge worse, not better (peak error 51 -> 147 over the thigh).
        #
        # The artwork stores straight alpha, so its RGB at a rim pixel is the
        # object's own colour; the background's share is carried by the alpha, not
        # mixed into the channels. Copying it is exact, and only reaches pixels
        # the part did not previously cover, so nothing already solved is touched.
        fresh = target & (part.rgba[sy:sy + h, sx:sx + w, 3] == 0)
        if fresh.any():
            rgba[sy:sy + h, sx:sx + w, :3][fresh] = src[cy1:cy2, cx1:cx2, :3][fresh]
        fixed[cy1:cy2, cx1:cx2] |= target
        out[i] = Part(
            name=part.name, rgba=rgba, offset=part.offset,
            depth_median=part.depth_median, depth=part.depth,
            synthetic=part.synthetic,
            meta={**part.meta, "source_alpha_fixed": int(target.sum())},
        )
    return out


#: Opacity is held just below 1 so ``log1p(-a)`` stays finite in ``limit_source_alpha``.
_ALPHA_CEILING = 1.0 - 1e-6


def limit_source_alpha(
    ordered: list[Part], decomp: Decomposition, s: RigSettings
) -> list[Part]:
    """Hold the composite down to the artwork's opacity where the artwork is soft.

    ``restore_source_alpha`` handles the composite coming out *less* opaque than
    the source. This is the other half, and it is the wispy hair.

    see-through returns each layer inpainted complete, so ``front hair`` and
    ``back hair`` both carry the loose strands beside the face. Neither is wrong
    on its own -- either one alone would let the hair move -- but they stack:

        1 - (1 - a_front)(1 - a_back)

    Two layers at 0.5 give 0.75, and, worse, the gaps *between* the strands are
    transparent in one layer and painted in the other, so they fill in. The fine
    locks the artist drew turn into a soft dark haze that reads as an afterimage
    of the hair rather than hair. Measured over the pixels that differ most in
    that region, 631 of 648 are covered by ``back hair`` and 619 by ``front hair``
    -- nearly all of them by both -- and the composite averages alpha 163.5 where
    the artwork has 110.2.

    The repair is the same principle the rest of this module runs on: the artwork
    is the truth. Where it is soft, the layers may not add up past it.

    Rather than pick a layer to sacrifice, every contributor is scaled by one
    factor, which keeps their relative weights and so keeps the hair's own shape.
    In optical thickness ``t = -ln(1 - a)`` compositing is a plain sum, so with
    ``T`` the artwork's thickness the factor is ``k = T / sum(t)`` and each layer
    becomes ``1 - exp(k * ln(1 - a))``. Closed form, no iteration.

    Only where the artwork is soft. Inside the silhouette it is opaque, so sheer
    fabric over skin -- which is the same arithmetic but *correct*, both layers
    genuinely being there -- is left alone.
    """
    if decomp.src_img is None or decomp.src_img.shape[2] < 4:
        return ordered

    cw, ch = decomp.canvas
    src_alpha = decomp.src_img[..., 3].astype(np.float64) / 255.0
    soft = src_alpha < (s.source_alpha_solid_floor / 255.0)
    if not soft.any():
        return ordered

    def thickness(a: np.ndarray) -> np.ndarray:
        return -np.log1p(-np.minimum(a, _ALPHA_CEILING))

    total = np.zeros((ch, cw), dtype=np.float64)
    for part in ordered:
        total += thickness(part.canvas_rgba(decomp.canvas)[..., 3].astype(np.float64) / 255.0)

    target = thickness(src_alpha)
    over = soft & (total > target + 1e-9)
    if not over.any():
        return ordered

    scale = np.ones((ch, cw), dtype=np.float64)
    scale[over] = target[over] / total[over]

    out = list(ordered)
    for i, part in enumerate(ordered):
        x1, y1, x2, y2 = part.bbox
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(cw, x2), min(ch, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        sx, sy = cx1 - x1, cy1 - y1
        h, w = cy2 - cy1, cx2 - cx1

        window = over[cy1:cy2, cx1:cx2]
        alpha = part.rgba[sy:sy + h, sx:sx + w, 3]
        touched = window & (alpha > 0)
        if not touched.any():
            continue

        a = np.minimum(alpha.astype(np.float64) / 255.0, _ALPHA_CEILING)
        thinned = 1.0 - np.exp(scale[cy1:cy2, cx1:cx2] * np.log1p(-a))
        rgba = part.rgba.copy()
        target_window = rgba[sy:sy + h, sx:sx + w, 3]
        target_window[touched] = np.clip(
            np.rint(thinned[touched] * 255.0), 0, 255
        ).astype(np.uint8)
        out[i] = Part(
            name=part.name, rgba=rgba, offset=part.offset,
            depth_median=part.depth_median, depth=part.depth,
            synthetic=part.synthetic,
            meta={**part.meta, "source_alpha_limited": int(touched.sum())},
        )
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


def _unpack_vertices(
    attachment: Attachment, bones: list[BoneTransform]
) -> list[dict[str, float]]:
    """The weight map per vertex, keyed by bone name."""
    by_index = {i: b.name for i, b in enumerate(bones)}
    out: list[dict[str, float]] = []
    stream, i = attachment.vertices, 0
    while i < len(stream):
        count = int(stream[i])
        i += 1
        entry: dict[str, float] = {}
        for _ in range(count):
            entry[by_index[int(stream[i])]] = float(stream[i + 3])
            i += 4
        out.append(entry)
    return out


def weld_shared_vertices(
    attachments: dict[str, Attachment],
    part_images: dict[str, Part],
    bones: list[BoneTransform],
    bone_index: dict[str, int],
    origin: tuple[float, float],
    s: RigSettings,
) -> int:
    """Give vertices that share a position the same weights, so they move as one.

    This is the tearing, and it is not a weighting bug so much as a consequence of
    weighting each part on its own. ``_candidate_bones`` hands every part a
    different set of bones -- deliberately, so a skirt on the hip cannot follow a
    hand that happens to be the nearest bone -- and ``_weights`` then solves within
    that set. Two parts that meet along a cut therefore get *different* answers at
    the very same point, and the seam is only as strong as the weakest agreement.

    Measured over coincident vertices, as an L1 distance between weight maps out
    of a possible 2.0:

        back hair | face                          1.559
        handwear-r@arm_r | handwear-r@leg_r       1.194
        back hair | front hair                    0.724
        topwear@arm_l_upper | topwear@torso       0.510

    The second is the cut down the thigh. At rest the two halves sit on top of each
    other and nothing shows; rotate the arm and they travel in different
    directions, opening the seam by far more than the few pixels
    ``close_layer_seams`` overlaps them by. That is what tears.

    So the shared boundary is welded: vertices within ``weld_radius_px`` of each
    other, across attachments, are pooled and every one of them is given the mean
    of the pool. Averaging rather than picking a winner keeps each part's own
    solution in the mix, and a vertex only ever moves toward its neighbours, never
    onto a bone none of them use.

    Only across *different* attachments -- welding a part to itself would do
    nothing, and vertices interior to a part are left exactly as they were, so the
    inside of a mesh still deforms the way its own bones say.

    Returns the number of vertices whose weights changed.
    """
    radius = float(s.weld_radius_px)
    if radius <= 0:
        return 0

    names = [n for n, a in attachments.items() if a.kind == "mesh" and a.vertices]
    if len(names) < 2:
        return 0

    # World position of every vertex, and the weight map beside it.
    points: list[np.ndarray] = []
    maps: dict[str, list[dict[str, float]]] = {}
    owner: list[str] = []
    for name in names:
        att = attachments[name]
        ox, oy = part_images[name].offset
        uv = np.asarray(att.uvs, dtype=np.float64).reshape(-1, 2)
        px = uv * np.array([att.width, att.height]) + np.array([ox, oy])
        world = np.array([px_to_spine((float(x), float(y)), origin) for x, y in px])
        points.append(world)
        maps[name] = _unpack_vertices(att, bones)
        owner.extend([name] * world.shape[0])

    # Keyed by slug, so the tag has to come from the part, not the slug -- the
    # slug has already had its separators flattened and base_tag cannot read it.
    layer_of = {n: taxonomy.base_tag(part_images[n].name) for n in names}

    allpts = np.concatenate(points, axis=0)
    index: list[tuple[str, int]] = []
    for name, world in zip(names, points):
        index.extend((name, i) for i in range(world.shape[0]))

    # Pool by position. A grid bucket of one radius, checking the 3x3 around each
    # vertex, is enough -- these are boundary vertices, a handful per pair.
    buckets: dict[tuple[int, int], list[int]] = {}
    keys = np.floor(allpts / radius).astype(int)
    for i, (kx, ky) in enumerate(keys):
        buckets.setdefault((int(kx), int(ky)), []).append(i)

    welded = 0
    updates: dict[str, dict[int, dict[str, float]]] = {n: {} for n in names}
    seen: set[int] = set()
    for i in range(allpts.shape[0]):
        if i in seen:
            continue
        kx, ky = keys[i]
        near: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                near.extend(buckets.get((int(kx) + dx, int(ky) + dy), ()))
        # Only across slices of the *same layer*. Two pieces cut from one garment
        # have to move as one or the cut opens; two different layers that merely
        # touch must not, and a hand resting on a skirt is exactly that. Welding
        # them pooled a finger vertex to 50% leftElbow / 48% leftLeg while every
        # vertex a few pixels away stayed at 98% leftElbow, so the finger sheared
        # away from the rest of the hand as soon as anything moved. That is the
        # squashing.
        tag_i = layer_of[index[i][0]]
        group = [j for j in near
                 if float(np.linalg.norm(allpts[j] - allpts[i])) <= radius
                 and layer_of[index[j][0]] == tag_i]
        if len({owner[j] for j in group}) < 2:
            continue
        pooled: dict[str, float] = {}
        for j in group:
            name, vi = index[j]
            for bone_name, weight in maps[name][vi].items():
                pooled[bone_name] = pooled.get(bone_name, 0.0) + weight
        total = sum(pooled.values())
        if total <= 0:
            continue
        top = sorted(pooled.items(), key=lambda kv: -kv[1])[:s.max_bones_per_vertex]
        norm = sum(w for _b, w in top)
        shared = {b: w / norm for b, w in top}
        for j in group:
            name, vi = index[j]
            updates[name][vi] = shared
            seen.add(j)
        welded += len(group)

    for name in names:
        if not updates[name]:
            continue
        att = attachments[name]
        ox, oy = part_images[name].offset
        uv = np.asarray(att.uvs, dtype=np.float64).reshape(-1, 2)
        px = uv * np.array([att.width, att.height]) + np.array([ox, oy])
        by_name = {b.name: b for b in bones}
        stream: list[float] = []
        used: set[int] = set()
        for vi in range(uv.shape[0]):
            entry = updates[name].get(vi) or maps[name][vi]
            world = px_to_spine((float(px[vi, 0]), float(px[vi, 1])), origin)
            stream.append(float(len(entry)))
            for bone_name, weight in entry.items():
                bone = by_name[bone_name]
                lx, ly = _to_bone_local(world, bone)
                idx = bone_index[bone_name]
                used.add(idx)
                stream.extend([float(idx), lx, ly, weight])
        attachments[name] = Attachment(
            name=att.name, kind=att.kind, width=att.width, height=att.height,
            uvs=att.uvs, triangles=att.triangles, hull=att.hull, edges=att.edges,
            vertices=stream, bones_used=sorted(used),
        )
    return welded


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
    # Before anything else: fill the opacity the layer split lost, so the repaint
    # floors below see the alpha the artwork actually had.
    ordered = close_layer_seams(ordered, decomp, s)
    if s.restore_source_alpha:
        # Before the colour repaint: raising alpha brings pixels above the repaint
        # floors, so they get the right colour as well as the right opacity.
        ordered = restore_source_alpha(ordered, decomp, s)
    if s.limit_source_alpha:
        # After the two raises, so it sees the alpha they leave behind, and before
        # the colour repaint, whose floors should read the final opacity.
        ordered = limit_source_alpha(ordered, decomp, s)
    if s.restore_source_pixels:
        ordered = restore_source_pixels(ordered, decomp, s)

    # Slices of one layer that touch are weighted across their shared cut.
    #
    # Each part is otherwise solved against its own candidate bones, so two
    # slices of the same layer get different answers at the same point and the
    # cut between them opens as soon as anything moves. ``weld_shared_vertices``
    # pins that shut where their vertices happen to coincide, but the contour
    # simplification puts vertices in different places along the rest of the
    # boundary, and between the pins it is still free.
    #
    # It showed up as an outline appearing along the thigh whenever the arm
    # lifted. ``handwear-r`` arrives as one sleeve and OCS cuts a ``leg_r`` piece
    # out of it to satisfy requirement 2-2, so half of it rides ``rightArm`` and
    # half ``rightLeg``; raising the arm slid one off the other and exposed the
    # lower piece's own edge. Hiding either half removed the line -- 47 of 76
    # ridge pixels for the leg piece, 19 for the arm piece.
    #
    # Each piece gets the *primary bone* of the pieces it actually touches, and
    # nothing else. Handing a family the union of all its slices' chains instead
    # is far too much: it put the right sleeve on ``head`` and ``hairBack`` and
    # the skirt on ``leftElbow``, because ``_weights`` picks the nearest bones out
    # of whatever it is offered and a sleeve passes close to the neck. One bone
    # per neighbour is enough for the weighting to vary continuously across the
    # cut, which is all that has to be true for it not to tear.
    neighbour_bones: dict[str, list[str]] = {}
    meshed = [q for q in ordered if _wants_mesh(q)]
    primary_of: dict[str, str] = {}
    solid: dict[str, np.ndarray] = {}
    for part in meshed:
        name = taxonomy.bone_for_part(part.name, available)
        if name not in bone_index:
            name = "torso" if "torso" in bone_index else bones[0].name
        primary_of[part.name] = name
        solid[part.name] = (part.canvas_rgba(decomp.canvas)[..., 3]
                            >= s.source_alpha_solid_floor)
    reach = ndi.generate_binary_structure(2, 2)
    for i, a in enumerate(meshed):
        for b in meshed[i + 1:]:
            if taxonomy.base_tag(a.name) != taxonomy.base_tag(b.name):
                continue
            if primary_of[a.name] == primary_of[b.name]:
                continue
            touching = ndi.binary_dilation(
                solid[a.name], reach, iterations=max(1, int(s.outline_dilate_px))
            ) & solid[b.name]
            if not touching.any():
                continue
            neighbour_bones.setdefault(a.name, []).append(primary_of[b.name])
            neighbour_bones.setdefault(b.name, []).append(primary_of[a.name])

    for part in ordered:
        slug = naming.unique_slug(part.name)
        primary = taxonomy.bone_for_part(part.name, available)
        if primary not in bone_index:
            primary = "torso" if "torso" in bone_index else bones[0].name

        attachment: Attachment | None = None
        if _wants_mesh(part):
            candidates = _candidate_bones(part, rig, primary)
            for bone in neighbour_bones.get(part.name, ()):
                if bone not in candidates:
                    candidates = candidates + [bone]
            attachment = _mesh_attachment(
                part, slug, bones, bone_index, candidates, origin, s,
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

    # Last, over the finished meshes: a seam holds only if both sides of it agree
    # on where they are going.
    welded = weld_shared_vertices(
        attachments, images, bones, bone_index, origin, s)

    return RigResult(
        canvas=rig.canvas, origin_px=origin, bones=bones, slots=slots,
        attachments=attachments, part_images=images, warnings=warnings,
        welded_vertices=welded,
    )
