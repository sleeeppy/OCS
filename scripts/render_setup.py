"""Rasterise an exported Spine skeleton and diff it against the source art.

An independent renderer, deliberately: it reads only ``skeleton.json``,
``skeleton.atlas`` and ``skeleton.png``, so if the output looks wrong here it is
wrong in the file, not in the player. Handy for judging a rig without a browser,
and for diffing a rig against the artwork it came from.

Implements just enough of the Spine transform to place attachments:

* bone world transforms from parent-local (x, y, rotation, scale)
* weighted mesh vertices, blended per bone as the runtime does
* region attachments as a rotated quad
* per-triangle affine texture mapping

    python scripts/render_setup.py --project <id>
    python scripts/render_setup.py --project <id> --animation idle --time 0.45
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocs.config import PROJECTS_DIR


# --------------------------------------------------------------------------
# atlas
# --------------------------------------------------------------------------

@dataclass
class AtlasRegion:
    name: str
    x: int
    y: int
    w: int
    h: int


def parse_atlas(text: str) -> tuple[tuple[int, int], dict[str, AtlasRegion]]:
    size = (0, 0)
    regions: dict[str, AtlasRegion] = {}
    name = None
    fields: dict[str, str] = {}

    def flush():
        if name and "xy" in fields and "size" in fields:
            x, y = (int(v) for v in fields["xy"].split(","))
            w, h = (int(v) for v in fields["size"].split(","))
            regions[name] = AtlasRegion(name, x, y, w, h)

    for raw in text.splitlines():
        if not raw.strip():
            continue
        if raw.startswith("  ") or raw.startswith("\t"):
            k, _, v = raw.strip().partition(":")
            fields[k.strip()] = v.strip()
            continue
        if ":" in raw and not raw.strip().endswith(".png"):
            k, _, v = raw.partition(":")
            if k.strip() == "size":
                size = tuple(int(t) for t in v.split(","))  # type: ignore[assignment]
            continue
        flush()
        name, fields = raw.strip(), {}
    flush()
    return size, regions


# --------------------------------------------------------------------------
# bones
# --------------------------------------------------------------------------

@dataclass
class Bone:
    name: str
    parent: str | None
    x: float
    y: float
    rotation: float
    scale_x: float
    scale_y: float
    # world
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    wx: float = 0.0
    wy: float = 0.0


def build_bones(doc: dict, anim: dict | None, time: float) -> dict[str, Bone]:
    bones: dict[str, Bone] = {}
    order: list[str] = []
    for entry in doc["bones"]:
        bones[entry["name"]] = Bone(
            name=entry["name"], parent=entry.get("parent"),
            x=float(entry.get("x", 0.0)), y=float(entry.get("y", 0.0)),
            rotation=float(entry.get("rotation", 0.0)),
            scale_x=float(entry.get("scaleX", 1.0)), scale_y=float(entry.get("scaleY", 1.0)),
        )
        order.append(entry["name"])

    if anim:
        for bone_name, timelines in anim.get("bones", {}).items():
            b = bones.get(bone_name)
            if not b:
                continue
            for kind, keys in timelines.items():
                if kind == "rotate":
                    b.rotation += _sample(keys, time, ("value",))[0]
                elif kind == "translate":
                    dx, dy = _sample(keys, time, ("x", "y"))
                    b.x += dx
                    b.y += dy

    for name in order:
        b = bones[name]
        rot = math.radians(b.rotation)
        la, lb = math.cos(rot) * b.scale_x, -math.sin(rot) * b.scale_y
        lc, ld = math.sin(rot) * b.scale_x, math.cos(rot) * b.scale_y
        if b.parent is None:
            b.a, b.b, b.c, b.d = la, lb, lc, ld
            b.wx, b.wy = b.x, b.y
        else:
            p = bones[b.parent]
            b.a = p.a * la + p.b * lc
            b.b = p.a * lb + p.b * ld
            b.c = p.c * la + p.d * lc
            b.d = p.c * lb + p.d * ld
            b.wx = p.a * b.x + p.b * b.y + p.wx
            b.wy = p.c * b.x + p.d * b.y + p.wy
    return bones


def _sample(keys: list[dict], time: float, fields: tuple[str, ...]) -> list[float]:
    """Linear sample. Bezier easing only changes timing, never placement."""
    times = [float(k.get("time", 0.0)) for k in keys]
    vals = [[float(k.get(f, 0.0)) for f in fields] for k in keys]
    if time <= times[0]:
        return vals[0]
    if time >= times[-1]:
        return vals[-1]
    for i in range(len(times) - 1):
        if times[i] <= time <= times[i + 1]:
            span = times[i + 1] - times[i]
            t = 0.0 if span <= 0 else (time - times[i]) / span
            return [v1 + (v2 - v1) * t for v1, v2 in zip(vals[i], vals[i + 1])]
    return vals[-1]


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def _render_mesh(dst: np.ndarray, src: np.ndarray,
                 src_pts: np.ndarray, dst_pts: np.ndarray, triangles: list[int]) -> None:
    """Draw a whole triangle mesh in one pass, seam-free.

    Compositing triangle by triangle cannot be made clean: antialiasing each one
    leaves partial coverage that neighbours do not complete, and hard-filling
    leaves the shared edge claimed by neither, so either way a grid of seams
    appears over the artwork -- glaring on translucent fabric. Spine has no such
    problem because the GPU draws the mesh as one triangle list with shared
    vertices.

    So this builds a destination-to-source coordinate map for the entire mesh
    first, then does a single ``remap`` and a single composite. Every destination
    pixel is written exactly once: no gaps, and no double-compositing either.
    """
    h, w = dst.shape[:2]
    map_x = np.full((h, w), -1.0, np.float32)
    map_y = np.full((h, w), -1.0, np.float32)

    for t in range(0, len(triangles), 3):
        idx = triangles[t:t + 3]
        d = dst_pts[idx].astype(np.float32)
        s = src_pts[idx].astype(np.float32)
        x0, y0, bw, bh = cv2.boundingRect(d)
        x0, y0 = max(0, x0 - 2), max(0, y0 - 2)
        x1, y1 = min(w, x0 + bw + 4), min(h, y0 + bh + 4)
        if x1 <= x0 or y1 <= y0:
            continue
        try:
            # Destination -> source, so the map can be filled by scanning dest px.
            M = cv2.getAffineTransform(d, s)
        except cv2.error:
            continue

        local = np.zeros((y1 - y0, x1 - x0), np.uint8)
        poly = np.int32(d - [x0, y0])
        cv2.fillConvexPoly(local, poly, 255, cv2.LINE_8)
        # Grow by one pixel so neighbouring triangles meet rather than leaving the
        # shared edge unclaimed. Overlap is harmless here: both triangles map that
        # edge to almost the same texel, and the map is overwritten, not blended.
        cv2.polylines(local, [poly], True, 255, 2, cv2.LINE_8)

        ys, xs = np.nonzero(local)
        if ys.size == 0:
            continue
        gx = (xs + x0).astype(np.float32)
        gy = (ys + y0).astype(np.float32)
        map_x[ys + y0, xs + x0] = M[0, 0] * gx + M[0, 1] * gy + M[0, 2]
        map_y[ys + y0, xs + x0] = M[1, 0] * gx + M[1, 1] * gy + M[1, 2]

    valid = map_x >= 0
    if not valid.any():
        return
    warped = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = (warped[..., 3].astype(np.float32) / 255.0) * valid
    dst[..., :3] = warped[..., :3] * a[..., None] + dst[..., :3] * (1 - a[..., None])
    dst[..., 3] = np.clip(warped[..., 3] * valid + dst[..., 3] * (1 - a), 0, 255)


def _warp_triangle(dst: np.ndarray, src: np.ndarray,
                   src_tri: np.ndarray, dst_tri: np.ndarray) -> None:
    """Affine-map one textured triangle into dst, alpha-compositing."""
    dr = cv2.boundingRect(dst_tri.astype(np.float32))
    sr = cv2.boundingRect(src_tri.astype(np.float32))
    if dr[2] <= 0 or dr[3] <= 0 or sr[2] <= 0 or sr[3] <= 0:
        return
    sx, sy, sw, sh = sr
    dx, dy, dw, dh = dr
    H, W = dst.shape[:2]
    if dx + dw <= 0 or dy + dh <= 0 or dx >= W or dy >= H:
        return
    sy2, sx2 = min(src.shape[0], sy + sh), min(src.shape[1], sx + sw)
    if sy2 <= sy or sx2 <= sx:
        return

    src_crop = src[sy:sy2, sx:sx2]
    s_local = (src_tri - [sx, sy]).astype(np.float32)
    d_local = (dst_tri - [dx, dy]).astype(np.float32)
    M = cv2.getAffineTransform(s_local[:3], d_local[:3])
    warped = cv2.warpAffine(src_crop, M, (dw, dh), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    tri_mask = np.zeros((dh, dw), np.uint8)
    # LINE_8, not LINE_AA. Antialiasing each triangle separately leaves its edge
    # pixels partly transparent, and two neighbours' partial coverages do not sum
    # to one, so every shared edge becomes a seam -- a triangle grid over anything
    # translucent. Spine draws the whole mesh as one triangle list with shared
    # vertices and no per-triangle mask, so it has no such seams; this renderer
    # must not invent them and then be read as evidence of a rigging fault.
    cv2.fillConvexPoly(tri_mask, np.int32(d_local), 255, cv2.LINE_8)

    x1, y1 = max(0, dx), max(0, dy)
    x2, y2 = min(W, dx + dw), min(H, dy + dh)
    ox, oy = x1 - dx, y1 - dy
    warped = warped[oy:oy + (y2 - y1), ox:ox + (x2 - x1)]
    tri_mask = tri_mask[oy:oy + (y2 - y1), ox:ox + (x2 - x1)]

    alpha = (warped[..., 3].astype(np.float32) / 255.0) * (tri_mask.astype(np.float32) / 255.0)
    region = dst[y1:y2, x1:x2]
    a = alpha[..., None]
    region[..., :3] = warped[..., :3] * a + region[..., :3] * (1 - a)
    region[..., 3] = np.clip(warped[..., 3] * (tri_mask / 255.0) + region[..., 3] * (1 - a[..., 0]), 0, 255)


def render(doc: dict, atlas_img: np.ndarray, atlas_size: tuple[int, int],
           regions: dict[str, AtlasRegion], canvas: tuple[int, int],
           origin: tuple[float, float], bones: dict[str, Bone]) -> np.ndarray:
    cw, ch = canvas
    out = np.zeros((ch, cw, 4), np.float32)
    aw, ah = atlas_size
    bone_list = [b["name"] for b in doc["bones"]]
    attachments = doc["skins"][0]["attachments"]

    def to_px(wx: float, wy: float) -> tuple[float, float]:
        return wx + origin[0], origin[1] - wy

    for slot in doc["slots"]:
        entry = attachments.get(slot["name"])
        if not entry:
            continue
        att_name, body = next(iter(entry.items()))
        reg = regions.get(att_name)
        if reg is None:
            continue

        if body.get("type") == "mesh":
            uvs = body["uvs"]
            verts = body["vertices"]
            tris = body["triangles"]
            n = len(uvs) // 2
            pts = np.zeros((n, 2), np.float32)
            i = idx = 0
            while i < len(verts):
                count = int(verts[i]); i += 1
                px = py = 0.0
                for _ in range(count):
                    bi = int(verts[i]); vx = verts[i + 1]; vy = verts[i + 2]; w = verts[i + 3]
                    i += 4
                    b = bones[bone_list[bi]]
                    px += (b.a * vx + b.b * vy + b.wx) * w
                    py += (b.c * vx + b.d * vy + b.wy) * w
                pts[idx] = to_px(px, py)
                idx += 1
            src_pts = np.stack([
                np.array(uvs[0::2], np.float32) * reg.w + reg.x,
                np.array(uvs[1::2], np.float32) * reg.h + reg.y,
            ], axis=1)
            _render_mesh(out, atlas_img, src_pts, pts, tris)
        else:
            b = bones[slot["bone"]]
            w, h = float(body["width"]), float(body["height"])
            rot = math.radians(float(body.get("rotation", 0.0)))
            ox, oy = float(body.get("x", 0.0)), float(body.get("y", 0.0))
            cr, sr_ = math.cos(rot), math.sin(rot)
            corners = []
            for lx, ly in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
                rx, ry = lx * cr - ly * sr_ + ox, lx * sr_ + ly * cr + oy
                corners.append(to_px(b.a * rx + b.b * ry + b.wx, b.c * rx + b.d * ry + b.wy))
            quad = np.array(corners, np.float32)
            # Atlas rect corners in the same order (v flipped: region v goes down).
            src_quad = np.array([
                [reg.x, reg.y + reg.h], [reg.x + reg.w, reg.y + reg.h],
                [reg.x + reg.w, reg.y], [reg.x, reg.y],
            ], np.float32)
            for tri in ((0, 1, 2), (0, 2, 3)):
                _warp_triangle(out, atlas_img, src_quad[list(tri)], quad[list(tri)])

    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=None)
    ap.add_argument("--animation", default=None)
    ap.add_argument("--time", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pid = args.project
    if pid is None:
        found = sorted(PROJECTS_DIR.glob("*/export/skeleton.json"))
        if not found:
            print("no exported project found", file=sys.stderr)
            return 1
        pid = found[-1].parent.parent.name
    root = PROJECTS_DIR / pid
    exp = root / "export"

    doc = json.loads((exp / "skeleton.json").read_text(encoding="utf-8"))
    atlas_size, regions = parse_atlas((exp / "skeleton.atlas").read_text(encoding="utf-8"))
    atlas_img = np.array(Image.open(exp / "skeleton.png").convert("RGBA"))

    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    canvas = (state["canvas"]["width"], state["canvas"]["height"])
    rig = json.loads((root / "rig.json").read_text(encoding="utf-8"))
    root_bone = next(b for b in rig["bones"] if b["name"] == "root")
    origin = (root_bone["x"], root_bone["y"])

    anim = doc["animations"].get(args.animation) if args.animation else None
    bones = build_bones(doc, anim, args.time)
    img = render(doc, atlas_img, atlas_size, regions, canvas, origin, bones)

    src_path = root / "layers" / "composite.png"
    src = np.array(Image.open(src_path).convert("RGBA")) if src_path.exists() else None

    tile = 520
    tiles = [("render", img)]
    if src is not None:
        tiles.insert(0, ("source", src))
        diff = img.copy()
        # Red where the render has ink the source does not.
        extra = (img[..., 3] > 40) & (src[..., 3] <= 40)
        missing = (img[..., 3] <= 40) & (src[..., 3] > 40)
        diff[extra] = (255, 60, 60, 255)
        diff[missing] = (40, 90, 255, 255)
        tiles.append(("diff: red=extra blue=missing", diff))
        print(f"extra ink   {int(extra.sum()):>8} px")
        print(f"missing ink {int(missing.sum()):>8} px")

    sheet = Image.new("RGB", (tile * len(tiles), tile), (16, 16, 20))
    for i, (_label, arr) in enumerate(tiles):
        flat = Image.new("RGB", arr.shape[1::-1], (16, 16, 20))
        flat.paste(Image.fromarray(arr), (0, 0), Image.fromarray(arr[..., 3]))
        sheet.paste(flat.resize((tile, tile), Image.LANCZOS), (i * tile, 0))

    out = args.out or (root / ("setup.png" if not args.animation else f"{args.animation}.png"))
    sheet.save(out)
    print(f"\nwrote {out}   ({' | '.join(n for n, _ in tiles)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
