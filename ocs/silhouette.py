"""Find the character's outline - the prerequisite for cutting limbs out of it.

This exists because of a measured surprise rather than a guessed one. Running
see-through on its own ``common/assets/test_image.png`` gives:

    src_img alpha px : 1236480  (75.47% of the 1280x1280 canvas)
    layer union px   :  217493  (13.27%)

That 1236480 is exactly 966x1280 - the *un-padded content rectangle*, not a
character. The sample is an RGB image with no alpha, so ``src_img.png``'s alpha
channel only records where ``center_square_pad_resize`` added padding. Meanwhile
the union of all 24 layers covers 13% of the canvas, because the taxonomy has no
skin tag: clothing, hair and face are extracted, bare arms and legs are not.

So neither source is a silhouette on its own:

``source_alpha``
    Correct and cheap *when the input is a cutout PNG* - which is the intended
    input, and what the user's own reference art is (checkerboard backdrop).
``background_estimate``
    Fallback for flat-background RGB input. Estimates the backdrop colour from
    the border ring and keeps what differs from it, unioned with the layer masks
    since those are foreground by construction.
``layer_union``
    Last resort when background estimation returns something degenerate. Loses
    bare skin, so it is reported as a warning rather than silently accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .psd_io import Decomposition, Part

SOURCE_ALPHA = "source_alpha"
BACKGROUND_ESTIMATE = "background_estimate"
LAYER_UNION = "layer_union"

#: Fraction of the content rectangle that must be transparent before we believe
#: the input really is a cutout rather than a flat photo/JPEG-style image.
_CUTOUT_MIN_TRANSPARENT = 0.02


@dataclass
class SilhouetteResult:
    mask: np.ndarray
    method: str
    content_rect: tuple[int, int, int, int]
    coverage: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "content_rect": list(self.content_rect),
            "coverage": round(self.coverage, 4),
            "warnings": list(self.warnings),
        }


def _layer_union(decomp: Decomposition, parts: list[Part]) -> np.ndarray:
    cw, ch = decomp.canvas
    out = np.zeros((ch, cw), dtype=bool)
    for p in parts:
        out |= p.canvas_mask(decomp.canvas)
    return out


def _content_rect(decomp: Decomposition) -> tuple[int, int, int, int]:
    """The un-padded region of the square canvas."""
    cw, ch = decomp.canvas
    if decomp.src_img is None or decomp.src_img.shape[2] < 4:
        return 0, 0, cw, ch
    a = decomp.src_img[..., 3] > 8
    if not a.any():
        return 0, 0, cw, ch
    ys, xs = np.nonzero(a)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _fill_and_clean(mask: np.ndarray, close_px: int) -> np.ndarray:
    """Close gaps, fill interior holes, keep the largest blob."""
    from scipy.ndimage import binary_fill_holes

    if not mask.any():
        return mask
    if close_px > 0:
        k = 2 * close_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    mask = binary_fill_holes(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 2:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def _estimate_background(
    rgb: np.ndarray, rect: tuple[int, int, int, int], seed: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Foreground mask from a flat backdrop, sampled off the border ring."""
    x1, y1, x2, y2 = rect
    warnings: list[str] = []
    sub = rgb[y1:y2, x1:x2].astype(np.float32)
    h, w = sub.shape[:2]

    ring = max(2, int(0.02 * min(h, w)))
    border = np.concatenate([
        sub[:ring].reshape(-1, 3), sub[-ring:].reshape(-1, 3),
        sub[:, :ring].reshape(-1, 3), sub[:, -ring:].reshape(-1, 3),
    ], axis=0)

    bg = np.median(border, axis=0)
    mad = np.median(np.abs(border - bg[None, :]), axis=0).mean()
    if mad > 24:
        warnings.append("background_not_flat")

    dist = np.abs(sub - bg[None, None, :]).mean(axis=2)
    fg = dist > max(16.0, 3.0 * float(mad))

    full = np.zeros(rgb.shape[:2], dtype=bool)
    full[y1:y2, x1:x2] = fg
    # Extracted layers are foreground whatever the colour test decided.
    full |= seed
    return full, warnings


def character_mask(
    decomp: Decomposition, parts: list[Part] | None = None
) -> SilhouetteResult:
    """Best available silhouette, with the method used reported back."""
    cw, ch = decomp.canvas
    parts = parts if parts is not None else decomp.parts
    rect = _content_rect(decomp)
    x1, y1, x2, y2 = rect
    rect_area = max(1, (x2 - x1) * (y2 - y1))
    close_px = max(2, min(cw, ch) // 300)

    seed = _layer_union(decomp, parts)

    # --- 1. a genuine cutout? --------------------------------------------
    if decomp.src_img is not None and decomp.src_img.shape[2] >= 4:
        alpha = decomp.src_img[..., 3] > 8
        inside = alpha[y1:y2, x1:x2]
        transparent_inside = 1.0 - float(inside.mean())
        if transparent_inside >= _CUTOUT_MIN_TRANSPARENT:
            mask = _fill_and_clean(alpha | seed, close_px)
            return SilhouetteResult(
                mask=mask, method=SOURCE_ALPHA, content_rect=rect,
                coverage=float(mask.sum()) / rect_area,
            )

    # --- 2. flat background -> estimate it -------------------------------
    warnings = [
        "no_alpha_channel: the input has no cutout, so the character outline was "
        "estimated from the background colour. A transparent PNG gives a cleaner rig."
    ]
    if decomp.src_img is not None:
        fg, extra = _estimate_background(decomp.src_img[..., :3], rect, seed)
        warnings.extend(extra)
        mask = _fill_and_clean(fg, close_px)
        coverage = float(mask.sum()) / rect_area
        if 0.02 <= coverage <= 0.95:
            return SilhouetteResult(
                mask=mask, method=BACKGROUND_ESTIMATE, content_rect=rect,
                coverage=coverage, warnings=warnings,
            )
        warnings.append(
            f"background_estimate_degenerate: covered {coverage:.1%} of the frame"
        )

    # --- 3. give up and use the layers ----------------------------------
    mask = _fill_and_clean(seed, close_px)
    warnings.append(
        "layer_union_fallback: outline taken from extracted layers only, so bare "
        "skin is missing. Limb parts will be cut from clothing instead."
    )
    return SilhouetteResult(
        mask=mask, method=LAYER_UNION, content_rect=rect,
        coverage=float(mask.sum()) / rect_area, warnings=warnings,
    )
