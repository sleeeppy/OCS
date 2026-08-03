"""Synthetic decompositions, shaped like real see-through output.

Built to the layout measured from see-through's own sample rather than an
idealised one, so the fixtures exercise the cases that actually occur:

- ``handwear`` / ``legwear`` are whole limbs, not gloves and socks
- ``legwear`` and ``footwear`` arrive as one layer covering both sides
- there is no skin layer at all
- empty, duplicate and speck layers are present and unfiltered
"""

from __future__ import annotations

import numpy as np
import pytest

from ocs.psd_io import Decomposition, Part

CANVAS = 512


def solid(box, colour=(180, 150, 140), alpha=255):
    """RGBA block for a (x1, y1, x2, y2) box, returned cropped with its offset."""
    x1, y1, x2, y2 = box
    rgba = np.zeros((y2 - y1, x2 - x1, 4), np.uint8)
    rgba[..., :3] = colour
    rgba[..., 3] = alpha
    # A little colour variation so the flat-fill heuristic is not tripped by
    # every fixture layer.
    noise = np.random.default_rng(0).integers(-6, 7, rgba[..., :3].shape, dtype=np.int16)
    rgba[..., :3] = np.clip(rgba[..., :3].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return rgba, (x1, y1)


def part(name, box, depth=0.5, colour=(180, 150, 140), alpha=255):
    rgba, offset = solid(box, colour, alpha)
    return Part(name=name, rgba=rgba, offset=offset, depth_median=depth,
                meta={"psd_index": 0})


def two_blobs(name, box_a, box_b, depth=0.5, colour=(180, 150, 140)):
    """One layer spanning two disconnected blobs, as unsplit legwear does."""
    x1 = min(box_a[0], box_b[0]); y1 = min(box_a[1], box_b[1])
    x2 = max(box_a[2], box_b[2]); y2 = max(box_a[3], box_b[3])
    rgba = np.zeros((y2 - y1, x2 - x1, 4), np.uint8)
    for b in (box_a, box_b):
        sub, _ = solid(b, colour)
        rgba[b[1] - y1:b[3] - y1, b[0] - x1:b[2] - x1] = sub
    return Part(name=name, rgba=rgba, offset=(x1, y1), depth_median=depth,
                meta={"psd_index": 0})


def _standing_figure() -> list[Part]:
    """A blocky front-facing figure. Viewer-left is the character's right."""
    return [
        part("back hair",  (215, 40, 300, 150), depth=0.95, colour=(60, 50, 55)),
        # handwear == the whole arm. Two separate layers, already LR-split
        # upstream: -r is the character's right, i.e. the *lower* x.
        part("handwear-r", (170, 175, 205, 330), depth=0.80, colour=(200, 170, 160)),
        part("handwear-l", (310, 175, 345, 330), depth=0.80, colour=(200, 170, 160)),
        # legwear == both legs in one layer, unsplit by see-through.
        two_blobs("legwear", (215, 320, 248, 460), (268, 320, 301, 460),
                  depth=0.78, colour=(205, 175, 165)),
        two_blobs("footwear", (208, 455, 252, 480), (264, 455, 308, 480),
                  depth=0.76, colour=(230, 225, 220)),
        part("topwear",    (200, 165, 315, 285), depth=0.62, colour=(40, 40, 45)),
        part("bottomwear", (205, 280, 310, 340), depth=0.70, colour=(70, 95, 140)),
        part("neck",       (240, 140, 275, 172), depth=0.74, colour=(205, 175, 165)),
        part("face",       (228, 60, 288, 145), depth=0.52, colour=(225, 195, 185)),
        part("irides-r",   (240, 95, 252, 107), depth=0.15, colour=(90, 130, 190)),
        part("irides-l",   (264, 95, 276, 107), depth=0.15, colour=(90, 130, 190)),
        part("front hair", (218, 35, 298, 105), depth=0.25, colour=(55, 45, 50)),
    ]


@pytest.fixture
def figure() -> Decomposition:
    """Clean figure, no junk layers."""
    parts = _standing_figure()
    src = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    for p in parts:
        block = p.canvas_rgba((CANVAS, CANVAS))
        m = block[..., 3] > 8
        src[m] = block[m]
    return Decomposition(canvas=(CANVAS, CANVAS), parts=parts, src_img=src)


@pytest.fixture
def figure_with_junk(figure: Decomposition) -> Decomposition:
    """Same figure plus the dummy layers see-through leaves in unfiltered."""
    junk = [
        # No pixels at all -- the character has no tail or wings.
        Part(name="tail", rgba=np.zeros((1, 1, 4), np.uint8), offset=(0, 0),
             depth_median=0.9, meta={"psd_index": 0, "empty_layer": True}),
        # Allocated but fully transparent.
        part("wings", (100, 100, 160, 160), depth=0.9, alpha=0),
        # A 3x3 speck, far below the auto-drop floor.
        part("objects", (400, 400, 403, 403), depth=0.4, colour=(120, 120, 120)),
        # A near-exact duplicate of topwear.
        part("neckwear", (200, 165, 315, 285), depth=0.61, colour=(40, 40, 45)),
    ]
    # Give the duplicate the identical pixels, not merely a similar block.
    top = next(p for p in figure.parts if p.name == "topwear")
    junk[-1] = Part(name="neckwear", rgba=top.rgba.copy(), offset=top.offset,
                    depth_median=0.61, meta={"psd_index": 0})
    return Decomposition(canvas=figure.canvas, parts=figure.parts + junk,
                         src_img=figure.src_img)


@pytest.fixture
def blob_figure() -> Decomposition:
    """Worst case for requirement 2-2: one connected silhouette, one layer.

    No LR suffixes, no separate limb layers, arms touching the torso. Nothing but
    the bone skeleton can separate left from right here.
    """
    mask = np.zeros((CANVAS, CANVAS), bool)
    mask[60:150, 225:290] = True     # head
    mask[150:300, 195:320] = True    # torso + arms fused
    mask[300:470, 215:250] = True    # right leg
    mask[300:470, 265:300] = True    # left leg
    rgba = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    rgba[..., :3] = (190, 165, 155)
    rgba[..., 3] = np.where(mask, 255, 0)
    ys, xs = np.nonzero(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    crop = rgba[box[1]:box[3], box[0]:box[2]].copy()
    body = Part(name="topwear", rgba=crop, offset=(box[0], box[1]),
                depth_median=0.6, meta={"psd_index": 0})
    src = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    src[..., :3] = (190, 165, 155)
    src[..., 3] = np.where(mask, 255, 0)
    return Decomposition(canvas=(CANVAS, CANVAS), parts=[body], src_img=src)
