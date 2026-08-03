"""Pack part images into a texture atlas and write the ``.atlas`` companion.

Shelf packing with rotation disabled. Rotation would win a few percent of area
but every mesh's UVs would then need transposing, which is a pointless risk for
the gain. Parts are sorted tallest-first, which is what makes shelves pack
tightly.

The libgdx ``.atlas`` format Spine consumes:

    skeleton.png
    size: 1024, 2048
    format: RGBA8888
    filter: Linear, Linear
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
            "filter: Linear, Linear",
            "repeat: none",
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


def _next_pow2(v: int) -> int:
    p = 1
    while p < v:
        p <<= 1
    return p


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
        canvas[py:py + h, px:px + w] = part.rgba
        regions.append(Region(name=name, x=px, y=py, width=w, height=h))

    regions.sort(key=lambda r: r.name)
    return AtlasResult(
        image=Image.fromarray(canvas), regions=regions,
        size=(width, height), png_name="skeleton.png",
    )
