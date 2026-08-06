"""Pack part images into a texture atlas and write the ``.atlas`` companion.

Shelf packing with rotation disabled. Rotation would win a few percent of area
but every mesh's UVs would then need transposing, which is a pointless risk for
the gain. Parts are sorted tallest-first, which is what makes shelves pack
tightly.

The libgdx ``.atlas`` format Spine consumes:

    skeleton.png
    size: 1024, 2048
    format: RGBA8888
    filter: MipMapLinearLinear, Linear
    repeat: none
    region_name
      rotate: false
      xy: 4, 4
      size: 190, 267
      orig: 190, 267
      offset: 0, 0
      index: -1

``orig``/``offset`` describe whitespace trimmed off the original image. OCS parts
arrive already cropped to their alpha bounds by see-through, so ``orig == size``
and ``offset`` is zero; the mesh UVs carry the placement instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

from .config import AtlasSettings
from .psd_io import Part


@dataclass
class Region:
    name: str
    x: int
    y: int
    width: int
    height: int


@dataclass
class AtlasResult:
    image: Image.Image
    regions: list[Region]
    size: tuple[int, int]
    png_name: str

    def write(self, out_dir: str | Path, base_name: str = "skeleton") -> tuple[Path, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / f"{base_name}.png"
        atlas_path = out_dir / f"{base_name}.atlas"
        self.image.save(png_path)
        atlas_path.write_text(self.to_text(f"{base_name}.png"), encoding="utf-8")
        return png_path, atlas_path

    def to_text(self, png_name: str) -> str:
        lines = [
            png_name,
            f"size: {self.size[0]}, {self.size[1]}",
            "format: RGBA8888",
            # Minified, so the page needs mipmaps. See ``_MIN_FILTER``.
            f"filter: {_MIN_FILTER}, Linear",
            "repeat: none",
            # The page is premultiplied. See ``_premultiply``.
            "pma: true",
        ]
        for r in self.regions:
            lines += [
                r.name,
                "  rotate: false",
                f"  xy: {r.x}, {r.y}",
                f"  size: {r.width}, {r.height}",
                f"  orig: {r.width}, {r.height}",
                "  offset: 0, 0",
                "  index: -1",
            ]
        return "\n".join(lines) + "\n"


#: Minification filter written into the page header.
#:
#: The preview fits a ~1000 unit skeleton into a canvas a few hundred pixels
#: high, so the atlas is *minified* -- measured at 0.765 on this character. A
#: plain ``Linear`` filter takes four texels per output pixel however large the
#: footprint is, so at any scale under 1 it undersamples: thin high-contrast
#: features -- the one-pixel rim along an arm, a gold hem -- collapse into a hard
#: line instead of averaging away, and because which texels get hit depends on
#: the sub-pixel position, that line crawls as the limb moves. It reads as an
#: outline trailing the arm.
#:
#: This is not the same defect as filtering straight alpha, which ``_premultiply``
#: fixes; that one is about interpolating the wrong quantity, this one is about
#: not taking enough samples. Premultiplying does nothing for it, which is why it
#: survived that fix, and it is invisible at any zoom above 1:1, which is why it
#: never showed up in a close-up.
#:
#: Mipmaps give the sampler a correctly pre-averaged level to read. The runtime
#: generates them; the header is what asks for them.
_MIN_FILTER = "MipMapLinearLinear"


def _premultiply(rgba: np.ndarray) -> np.ndarray:
    """Multiply RGB by alpha, so the GPU may filter the page correctly.

    Bilinear filtering of a *straight*-alpha texture interpolates colour and
    alpha independently, and that is not a valid operation: halfway between an
    opaque pixel and a transparent one it returns the average of the two colours
    at half alpha, when the right answer is the opaque pixel's colour at half
    alpha. The error appears at every edge where alpha varies, and because a
    part's feathered edge is a continuous one-pixel curve, it appears as a line.

    Over one face that was a grey scratch traced along the jaw, along every wisp
    of hair across the cheek, around the hand and the shoulder. Forcing the
    sampler to ``NEAREST`` made all of them vanish, which is what identified it:
    nothing about the texture, the mesh or the compositing was wrong, only the
    filtering of a format that cannot be filtered.

    Premultiplying makes interpolation linear in the quantity actually being
    blended, so the sampler is correct at every edge. ``pma: true`` in the page
    header tells the runtime to pair it with the matching blend function
    (``ONE, ONE_MINUS_SRC_ALPHA`` instead of ``SRC_ALPHA, ONE_MINUS_SRC_ALPHA``).

    ``_bleed_rgb`` still runs first. It costs nothing here and keeps the page
    readable in an image editor, where premultiplied transparent black would
    otherwise hide what each region contains.
    """
    out = rgba.astype(np.uint16)
    out[..., :3] = (out[..., :3] * out[..., 3:4] + 127) // 255
    return out.astype(np.uint8)


def _next_pow2(v: int) -> int:
    p = 1
    while p < v:
        p <<= 1
    return p


def _bleed_rgb(rgba: np.ndarray, alpha_floor: int = 8) -> np.ndarray:
    """Fill transparent pixels with the nearest opaque pixel's colour.

    Alpha is untouched, so nothing new is drawn -- but the *colour* under the
    transparent pixels is what bilinear filtering mixes in at a mesh boundary,
    and leaving it at (0, 0, 0) is what puts a dark hairline around every part.

    Measured on this character before the fix: 5776 pixels differed from the
    source art by more than 8 levels, 5012 of them **darker**, mean -19.6; at a
    32-level threshold, 1830 of 1830 darker, mean -39.7. They formed 491 thin
    clusters tracing part outlines -- the "visible cut" this exists to remove.
    Atlas padding was transparent black, so every edge sample pulled toward it.

    Sampling can only reach a pixel or two past the boundary, so an exact nearest
    fill is more than enough; the point is that the neighbour is the same colour
    rather than black.
    """
    alpha = rgba[..., 3]
    solid = alpha > alpha_floor
    if not solid.any() or solid.all():
        return rgba

    # Nearest opaque source pixel for every pixel, via the EDT's index output.
    _dist, (iy, ix) = ndi.distance_transform_edt(~solid, return_indices=True)
    out = rgba.copy()
    fill = ~solid
    out[..., :3][fill] = rgba[..., :3][iy[fill], ix[fill]]
    return out


def _bleed_into_gutter(
    canvas: np.ndarray, bled: np.ndarray, px: int, py: int, pad: int
) -> None:
    """Repeat the part's border colour outward into the padding gutter.

    ``_bleed_rgb`` only reaches the part's own box. A mesh vertex sitting exactly
    on the box edge can still sample just outside it, so the gutter needs the same
    treatment -- edge-replicate the RGB, leave alpha at zero.
    """
    if pad <= 0:
        return
    h, w = bled.shape[:2]
    ch, cw = canvas.shape[:2]
    y1, y2 = max(0, py - pad), min(ch, py + h + pad)
    x1, x2 = max(0, px - pad), min(cw, px + w + pad)
    # Index into the part with clamped coordinates: standard edge replication.
    ys = np.clip(np.arange(y1, y2) - py, 0, h - 1)
    xs = np.clip(np.arange(x1, x2) - px, 0, w - 1)
    patch = bled[np.ix_(ys, xs)]
    target = canvas[y1:y2, x1:x2]
    # Only paint where nothing has been written yet, so neighbours packed into the
    # same gutter are not overwritten.
    empty = target[..., 3] == 0
    rgb = target[..., :3]
    rgb[empty & (rgb.max(axis=-1) == 0)] = patch[..., :3][empty & (rgb.max(axis=-1) == 0)]


def pack(
    parts: dict[str, Part], settings: AtlasSettings | None = None
) -> AtlasResult:
    """Pack ``{region_name: Part}`` into one atlas page."""
    s = settings or AtlasSettings()
    items = [(name, part) for name, part in parts.items()
             if part.rgba.shape[0] > 0 and part.rgba.shape[1] > 0]
    if not items:
        blank = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        return AtlasResult(image=blank, regions=[], size=(4, 4), png_name="skeleton.png")

    pad = s.padding
    items.sort(key=lambda it: (-it[1].rgba.shape[0], -it[1].rgba.shape[1]))

    total_area = sum((p.rgba.shape[1] + pad) * (p.rgba.shape[0] + pad) for _n, p in items)
    widest = max(p.rgba.shape[1] for _n, p in items) + pad * 2
    # 1.15x the square root leaves a little slack for shelf waste without
    # doubling the page.
    est = max(widest, int((total_area ** 0.5) * 1.15))
    width = min(s.max_size, _next_pow2(est) if s.power_of_two else est)
    width = max(width, _next_pow2(widest) if s.power_of_two else widest)

    placed: list[tuple[str, Part, int, int]] = []
    x = y = pad
    shelf_h = 0
    for name, part in items:
        h, w = part.rgba.shape[:2]
        if x + w + pad > width and shelf_h:
            x = pad
            y += shelf_h + pad
            shelf_h = 0
        placed.append((name, part, x, y))
        x += w + pad
        shelf_h = max(shelf_h, h)

    height = y + shelf_h + pad
    if s.power_of_two:
        height = _next_pow2(height)
    height = min(height, s.max_size)

    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    regions: list[Region] = []
    for name, part, px, py in placed:
        h, w = part.rgba.shape[:2]
        if py + h > height or px + w > width:
            # Only reachable if a single part exceeds max_size; skip rather than
            # corrupt the page, and let the caller notice the missing region.
            continue
        bled = _bleed_rgb(part.rgba)
        canvas[py:py + h, px:px + w] = bled
        _bleed_into_gutter(canvas, bled, px, py, pad)
        regions.append(Region(name=name, x=px, y=py, width=w, height=h))

    regions.sort(key=lambda r: r.name)
    return AtlasResult(
        image=Image.fromarray(_premultiply(canvas)), regions=regions,
        size=(width, height), png_name="skeleton.png",
    )
