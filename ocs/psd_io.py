"""Read see-through's PSD output into OCS's ``Part`` model.

see-through writes, for an input named ``foo.png`` and ``--save_dir D``:

    D/foo.psd            layers, cropped, positioned by (left, top)
    D/foo_depth.psd      same layers as 8-bit L depth
    D/foo.psd.json       {"parts": {tag: {xyxy, depth_median, tag}}, "frame_size": [H, W]}
    D/foo/src_img.png    the square-padded, resized input -- the canvas reference
    D/foo/reconstruction.png  all parts alpha-composited

Two details that matter downstream:

1. The canvas is **not** the user's original image. ``apply_layerdiff`` runs
   ``center_square_pad_resize(input_img, resolution)`` first, so the canvas is a
   square of side ``resolution`` (1280 by default). OCS works entirely in that
   space and uses ``src_img.png`` as the editor backdrop, which keeps every
   coordinate consistent without needing to invert the padding.

2. Layers are cropped to their bounding box, so ``Part.offset`` is load-bearing.
   ``save_psd`` is called with ``pad_to_canvas=False``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from psd_tools import PSDImage

from . import taxonomy


@dataclass
class Part:
    """One decomposed layer, cropped to its own bounding box."""

    name: str
    #: RGBA uint8, shape (h, w, 4). Cropped -- see ``offset``.
    rgba: np.ndarray
    #: (left, top) of ``rgba`` within the canvas.
    offset: tuple[int, int]
    #: see-through's median depth for this layer. Larger = further away.
    depth_median: float = 1.0
    #: Optional 8-bit depth crop aligned with ``rgba``.
    depth: np.ndarray | None = None
    #: Set when OCS synthesised this part rather than see-through emitting it.
    synthetic: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def tag(self) -> str:
        return taxonomy.base_tag(self.name)

    @property
    def side(self) -> str | None:
        return taxonomy.part_side(self.name)

    @property
    def region(self) -> str | None:
        return taxonomy.part_region(self.name)

    @property
    def alpha(self) -> np.ndarray:
        return self.rgba[..., 3]

    @property
    def size(self) -> tuple[int, int]:
        return self.rgba.shape[1], self.rgba.shape[0]

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """(x1, y1, x2, y2) in canvas coordinates."""
        left, top = self.offset
        h, w = self.rgba.shape[:2]
        return left, top, left + w, top + h

    def mask(self, alpha_floor: int = 8) -> np.ndarray:
        return self.alpha > alpha_floor

    def canvas_mask(self, canvas: tuple[int, int], alpha_floor: int = 8) -> np.ndarray:
        """This part's mask placed on a full-canvas boolean array."""
        cw, ch = canvas
        out = np.zeros((ch, cw), dtype=bool)
        x1, y1, x2, y2 = self.bbox
        sx1, sy1 = max(0, -x1), max(0, -y1)
        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(cw, x2), min(ch, y2)
        if x2c <= x1c or y2c <= y1c:
            return out
        sub = self.alpha[sy1:sy1 + (y2c - y1c), sx1:sx1 + (x2c - x1c)]
        out[y1c:y2c, x1c:x2c] = sub > alpha_floor
        return out

    def canvas_rgba(self, canvas: tuple[int, int]) -> np.ndarray:
        cw, ch = canvas
        out = np.zeros((ch, cw, 4), dtype=np.uint8)
        x1, y1, x2, y2 = self.bbox
        sx1, sy1 = max(0, -x1), max(0, -y1)
        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(cw, x2), min(ch, y2)
        if x2c <= x1c or y2c <= y1c:
            return out
        out[y1c:y2c, x1c:x2c] = self.rgba[sy1:sy1 + (y2c - y1c), sx1:sx1 + (x2c - x1c)]
        return out


@dataclass
class Decomposition:
    """Everything OCS needs from one see-through run."""

    canvas: tuple[int, int]              # (width, height) -- square in practice
    parts: list[Part]
    src_img: np.ndarray | None = None    # RGBA, canvas-sized (src_img.png)
    psd_path: Path | None = None

    def by_name(self) -> dict[str, Part]:
        return {p.name: p for p in self.parts}

    def silhouette(self, alpha_floor: int = 8) -> np.ndarray:
        """Union of every part's mask -- what the decomposition actually covers."""
        cw, ch = self.canvas
        out = np.zeros((ch, cw), dtype=bool)
        for p in self.parts:
            out |= p.canvas_mask(self.canvas, alpha_floor)
        return out

    def source_alpha(self, alpha_floor: int = 8) -> np.ndarray | None:
        """The input image's own alpha, when it had one."""
        if self.src_img is None or self.src_img.shape[2] < 4:
            return None
        return self.src_img[..., 3] > alpha_floor


# --------------------------------------------------------------------------


def _layer_rgba(layer) -> np.ndarray | None:
    """psd-tools layer -> uint8 RGBA, or None when the layer has no pixels."""
    arr = layer.numpy()
    if arr is None or arr.size == 0:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[2] == 1:                      # L -> opaque grey
        arr = np.concatenate([arr.repeat(3, axis=2), np.ones_like(arr)], axis=2)
    elif arr.shape[2] == 2:                    # LA
        arr = np.concatenate([arr[..., :1].repeat(3, axis=2), arr[..., 1:2]], axis=2)
    elif arr.shape[2] == 3:                    # RGB -> assume opaque
        arr = np.concatenate([arr, np.ones_like(arr[..., :1])], axis=2)
    return np.clip(arr[..., :4] * 255.0, 0, 255).round().astype(np.uint8)


def _sidecar(psd_path: Path) -> dict:
    """see-through writes ``<psd>.json`` (note: appended, not extension-swapped)."""
    p = Path(str(psd_path) + ".json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _depth_layers(psd_path: Path) -> dict[str, np.ndarray]:
    depth_path = psd_path.with_name(psd_path.stem + "_depth.psd")
    if not depth_path.exists():
        return {}
    out: dict[str, np.ndarray] = {}
    try:
        psd = PSDImage.open(depth_path)
    except Exception:
        return {}
    for layer in psd:
        arr = layer.numpy()
        if arr is None or arr.size == 0:
            continue
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        out[layer.name] = np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)
    return out


def read_decomposition(psd_path: str | Path, load_depth: bool = True) -> Decomposition:
    """Load a see-through PSD (plus its sidecars) into a ``Decomposition``."""
    psd_path = Path(psd_path)
    psd = PSDImage.open(psd_path)
    canvas = (psd.width, psd.height)

    sidecar = _sidecar(psd_path)
    part_meta: dict[str, dict] = sidecar.get("parts", {}) or {}
    depths = _depth_layers(psd_path) if load_depth else {}

    parts: list[Part] = []
    for index, layer in enumerate(psd):
        rgba = _layer_rgba(layer)
        if rgba is None:
            # A pixel-less layer is a dummy by definition; keep it so cleanup can
            # report it rather than silently swallowing it here.
            parts.append(Part(
                name=layer.name, rgba=np.zeros((1, 1, 4), np.uint8), offset=(0, 0),
                depth_median=float(part_meta.get(layer.name, {}).get("depth_median", 1.0)),
                meta={"psd_index": index, "empty_layer": True},
            ))
            continue

        left, top = int(layer.offset[0]), int(layer.offset[1])
        meta = part_meta.get(layer.name, {})
        depth = depths.get(layer.name)
        if depth is not None and depth.shape != rgba.shape[:2]:
            depth = None

        parts.append(Part(
            name=layer.name,
            rgba=rgba,
            offset=(left, top),
            depth_median=float(meta.get("depth_median", 1.0)),
            depth=depth,
            meta={"psd_index": index, "xyxy": meta.get("xyxy")},
        ))

    src_img = None
    src_p = psd_path.with_suffix("") / "src_img.png"
    if src_p.exists():
        src_img = np.array(Image.open(src_p).convert("RGBA"))

    return Decomposition(canvas=canvas, parts=parts, src_img=src_img, psd_path=psd_path)


def read_layer_dir(layer_dir: str | Path) -> Decomposition:
    """Load see-through's per-tag PNG directory, without needing the PSD.

    Recovery path, and not a marginal one. ``apply_layerdiff`` writes each tag's
    PNG as it goes, but the PSD is only assembled at the very end by
    ``further_extr``. On a machine where one image takes an hour, anything that
    interrupts the run after the body pass throws away every layer it already
    produced -- 30 minutes of GPU work sitting on disk that ``read_decomposition``
    cannot open because the container was never written.

    Two things are missing compared with the PSD path, neither fatal:

    - No ``depth_median``, because that comes from the depth pass. Every part gets
      1.0, which makes every pair a tie, which hands ordering entirely to
      ``taxonomy.Z_PRIOR`` -- a full tag-to-order table that exists precisely
      because the depth estimate is unreliable for thin overlapping parts.
    - No per-part depth maps. Nothing downstream requires them.

    ``src_img.png`` and ``src_head.png`` are see-through's own inputs, not output
    layers, so they are read as the source image rather than treated as parts.
    """
    layer_dir = Path(layer_dir)
    if not layer_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {layer_dir}")

    src_img = None
    src_p = layer_dir / "src_img.png"
    if src_p.exists():
        src_img = np.array(Image.open(src_p).convert("RGBA"))

    #: see-through's inputs, which live in the same directory as its outputs.
    inputs = {"src_img", "src_head"}
    files = sorted(p for p in layer_dir.glob("*.png") if p.stem not in inputs)
    if not files:
        raise FileNotFoundError(f"no layer PNGs in {layer_dir}")

    canvas = None
    if src_img is not None:
        canvas = (src_img.shape[1], src_img.shape[0])

    parts: list[Part] = []
    for index, path in enumerate(files):
        full = np.array(Image.open(path).convert("RGBA"))
        if canvas is None:
            canvas = (full.shape[1], full.shape[0])

        # Crop to the alpha box, exactly as psd-tools resolves each PSD layer to
        # its own box plus an offset. Not cosmetic: cleanup's alpha_mean is the
        # mean over the *part*, so a full-canvas image makes any small real layer
        # look near-transparent. Measured here, uncropped, the character's `neck`
        # -- 1173 fully opaque pixels -- scored alpha_mean 0.0018 against a
        # 0.01 auto-drop floor and was deleted without asking. `footwear` came out
        # at 0.0100, right on the line. Cropping also stops every layer's bbox
        # spanning the canvas, which had `front hair`, `head` and `tail` all
        # flagged `contained_in_other`.
        rgba, offset = _crop_to_alpha(full)
        parts.append(Part(
            name=path.stem,
            rgba=rgba,
            offset=offset,
            depth_median=1.0,
            meta={"psd_index": index, "from_layer_dir": True},
        ))

    return Decomposition(canvas=canvas, parts=parts, src_img=src_img)


def _crop_to_alpha(
    rgba: np.ndarray, alpha_floor: int = 8
) -> tuple[np.ndarray, tuple[int, int]]:
    """Trim to the opaque bounding box, returning the crop and its offset.

    A layer with no opaque pixels keeps a 1x1 placeholder at the origin rather
    than an empty array, so cleanup can report it as the dummy it is instead of
    tripping over a zero-sized shape.
    """
    ys, xs = np.nonzero(rgba[..., 3] > alpha_floor)
    if len(xs) == 0:
        return np.zeros((1, 1, 4), np.uint8), (0, 0)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return rgba[y1:y2, x1:x2].copy(), (x1, y1)


def write_parts_png(parts: list[Part], out_dir: str | Path) -> dict[str, Path]:
    """Dump each part as a trimmed PNG plus an offsets manifest, for the atlas."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    naming = taxonomy.PartNaming()
    written: dict[str, Path] = {}
    manifest = {}
    for part in parts:
        slug = naming.unique_slug(part.name)
        path = out_dir / f"{slug}.png"
        Image.fromarray(part.rgba).save(path)
        written[part.name] = path
        manifest[slug] = {
            "part": part.name,
            "offset": list(part.offset),
            "size": list(part.size),
            "depth_median": part.depth_median,
            "synthetic": part.synthetic,
        }
    (out_dir / "parts.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return written
