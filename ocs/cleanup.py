"""Requirement 2 - drop the dummy layers see-through leaves behind.

see-through does no filtering of its own: ``further_extr`` and ``dump_parts_psd``
write every tag the model produced straight into the PSD, with no area or alpha
threshold anywhere in the path. A 23-layer decomposition routinely contains
fully transparent tags (the character has no tail, no wings, no eyewear), near
duplicates, and inpainting specks.

Two tiers, deliberately:

``auto_dropped``
    Only things that cannot be anything but garbage - no pixels, an area below
    a hair of the canvas, or a near-exact duplicate of another layer. Removed
    without asking, but still reported so the review UI can un-drop them.

``suspicious``
    Judgement calls: slivers, speckle, flat inpaint fill, a layer wholly
    contained in another. Flagged with a reason and left for the user to decide,
    which is what requirement 2-1 asks for.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from .config import CleanupSettings
from .psd_io import Decomposition, Part

KEEP = "keep"
AUTO_DROPPED = "auto_dropped"
SUSPICIOUS = "suspicious"


@dataclass
class LayerReport:
    name: str
    tag: str
    side: str | None
    verdict: str
    reasons: list[str] = field(default_factory=list)

    alpha_area_px: int = 0
    coverage: float = 0.0          # alpha_area / canvas area
    alpha_mean: float = 0.0        # 0..1
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    largest_cc_ratio: float = 0.0  # biggest connected component / alpha_area
    #: Mean per-channel std over opaque pixels. ``None`` when there are too few
    #: solid pixels to mean anything -- deliberately not NaN, because ``NaN`` is
    #: not valid JSON and silently breaks every consumer downstream.
    rgb_std: float | None = 0.0
    dup_of: str | None = None
    contained_in: str | None = None
    depth_median: float = 1.0
    psd_index: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d


def _measure(part: Part, canvas_area: int, s: CleanupSettings) -> LayerReport:
    report = LayerReport(
        name=part.name,
        tag=part.tag,
        side=part.side,
        verdict=KEEP,
        depth_median=part.depth_median,
        psd_index=int(part.meta.get("psd_index", 0)),
        bbox=part.bbox,
    )

    if part.meta.get("empty_layer"):
        report.verdict = AUTO_DROPPED
        report.reasons.append("empty_layer")
        return report

    alpha = part.alpha
    mask = alpha > s.alpha_floor
    report.alpha_area_px = int(mask.sum())
    report.coverage = report.alpha_area_px / max(1, canvas_area)
    report.alpha_mean = float(alpha.mean()) / 255.0

    if report.alpha_area_px == 0:
        report.verdict = AUTO_DROPPED
        report.reasons.append("fully_transparent")
        return report

    n_labels, _labels, stats, _cent = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        report.largest_cc_ratio = float(areas.max()) / max(1, report.alpha_area_px)

    opaque = alpha > 128
    if opaque.sum() >= 16:
        report.rgb_std = float(rgb_std) if np.isfinite(
            rgb_std := part.rgba[..., :3][opaque].astype(np.float32).std(axis=0).mean()
        ) else None
    else:
        # Too few solid pixels to say anything about colour variance; leave it
        # unknown so an artificially low std cannot trip the flat-fill flag.
        report.rgb_std = None

    return report


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(1, union)


def _mean_rgb(part: Part, s: CleanupSettings) -> np.ndarray:
    m = part.alpha > s.alpha_floor
    if not m.any():
        return np.zeros(3, np.float32)
    return part.rgba[..., :3][m].astype(np.float32).mean(axis=0)


def analyze(decomp: Decomposition, settings: CleanupSettings | None = None) -> list[LayerReport]:
    """Score every layer and assign a verdict. Pure - mutates nothing."""
    s = settings or CleanupSettings()
    cw, ch = decomp.canvas
    canvas_area = cw * ch

    reports = [_measure(p, canvas_area, s) for p in decomp.parts]
    by_name = {r.name: r for r in reports}

    # --- tier 1: hard drops on size ---------------------------------------
    area_floor = max(s.min_area_px, s.min_coverage_drop * canvas_area)
    for r in reports:
        if r.verdict == AUTO_DROPPED:
            continue
        if r.alpha_area_px < area_floor:
            r.verdict = AUTO_DROPPED
            r.reasons.append("tiny_area")
        elif r.alpha_mean < s.min_alpha_mean_drop:
            r.verdict = AUTO_DROPPED
            r.reasons.append("near_transparent")

    # --- tier 1: duplicates / tier 2: containment -------------------------
    # Compare only pairs whose bounding boxes plausibly overlap; the full-canvas
    # mask AND is the expensive part and most pairs never need it.
    live = [p for p in decomp.parts if by_name[p.name].verdict != AUTO_DROPPED]
    masks: dict[str, np.ndarray] = {}
    means: dict[str, np.ndarray] = {}

    def mask_of(part: Part) -> np.ndarray:
        if part.name not in masks:
            masks[part.name] = part.canvas_mask(decomp.canvas, s.alpha_floor)
        return masks[part.name]

    def mean_of(part: Part) -> np.ndarray:
        if part.name not in means:
            means[part.name] = _mean_rgb(part, s)
        return means[part.name]

    for i, a in enumerate(live):
        ra = by_name[a.name]
        for b in live[i + 1:]:
            rb = by_name[b.name]
            if rb.verdict == AUTO_DROPPED or ra.verdict == AUTO_DROPPED:
                continue
            if _bbox_iou(ra.bbox, rb.bbox) < 0.5 * s.contained_iou_flag:
                continue

            ma, mb = mask_of(a), mask_of(b)
            inter = int((ma & mb).sum())
            if inter == 0:
                continue
            union = ra.alpha_area_px + rb.alpha_area_px - inter
            iou = inter / max(1, union)

            if iou >= s.dup_iou_drop:
                delta = float(np.abs(mean_of(a) - mean_of(b)).mean())
                if delta <= s.dup_rgb_delta_drop:
                    # Same pixels, same colour: keep the more opaque one.
                    loser, winner = (rb, ra) if ra.alpha_area_px >= rb.alpha_area_px else (ra, rb)
                    loser.verdict = AUTO_DROPPED
                    loser.reasons.append("duplicate")
                    loser.dup_of = winner.name
                    continue

            # Containment is only interesting in one direction: the smaller
            # layer being a redundant subset of the larger.
            small, large = (ra, rb) if ra.alpha_area_px <= rb.alpha_area_px else (rb, ra)
            if inter / max(1, small.alpha_area_px) >= s.contained_iou_flag:
                small.contained_in = large.name
                if "contained_in_other" not in small.reasons:
                    small.reasons.append("contained_in_other")

    # --- tier 2: suspicious flags ----------------------------------------
    for r in reports:
        if r.verdict == AUTO_DROPPED:
            continue
        flags: list[str] = []
        if r.coverage < s.min_coverage_flag:
            flags.append("sliver")
        if (r.largest_cc_ratio and r.largest_cc_ratio < s.speckle_cc_ratio_flag
                and r.coverage < s.speckle_coverage_flag):
            flags.append("speckle")
        # Only large uniform areas are inpainting artifacts. Small ones are
        # legitimately flat -- eyewhite and irides are near-single-colour by
        # nature, and flagging them every run would just train the user to
        # ignore the panel.
        if (r.rgb_std is not None and r.rgb_std < s.flat_rgb_std_flag
                and r.coverage >= s.min_coverage_flag):
            flags.append("flat_fill")
        if r.tag in s.junk_tags and r.coverage < s.junk_tag_coverage_flag:
            flags.append("junk_tag")

        r.reasons.extend(f for f in flags if f not in r.reasons)
        if r.reasons:
            r.verdict = SUSPICIOUS

    return reports


def apply_verdicts(
    decomp: Decomposition,
    reports: list[LayerReport],
    excluded: set[str] | None = None,
    revived: set[str] | None = None,
) -> tuple[list[Part], list[str]]:
    """Resolve reports plus the user's review choices into the surviving parts.

    ``excluded`` are names the user chose to drop on top of the automatic pass
    (requirement 2-1). ``revived`` un-drops an ``auto_dropped`` layer, so the
    "과감하게 제외" default stays reversible.
    """
    excluded = excluded or set()
    revived = revived or set()
    verdict = {r.name: r.verdict for r in reports}

    kept: list[Part] = []
    dropped: list[str] = []
    for part in decomp.parts:
        v = verdict.get(part.name, KEEP)
        if part.name in excluded:
            dropped.append(part.name)
            continue
        if v == AUTO_DROPPED and part.name not in revived:
            dropped.append(part.name)
            continue
        if v == AUTO_DROPPED and part.name in revived and part.meta.get("empty_layer"):
            # Nothing to revive - the layer genuinely has no pixels.
            dropped.append(part.name)
            continue
        kept.append(part)
    return kept, dropped


def summarize(reports: list[LayerReport]) -> dict:
    return {
        "total": len(reports),
        "keep": sum(1 for r in reports if r.verdict == KEEP),
        "auto_dropped": sum(1 for r in reports if r.verdict == AUTO_DROPPED),
        "suspicious": sum(1 for r in reports if r.verdict == SUSPICIOUS),
    }
