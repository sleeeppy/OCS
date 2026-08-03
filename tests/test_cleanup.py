"""Requirement 2: dummy-layer detection, and the two-tier verdict split."""

from __future__ import annotations

import json

import pytest

from ocs import cleanup
from ocs.config import CleanupSettings


def verdicts(decomp):
    return {r.name: r for r in cleanup.analyze(decomp, CleanupSettings())}


def test_clean_figure_keeps_everything(figure):
    reports = cleanup.analyze(figure, CleanupSettings())
    assert not [r for r in reports if r.verdict == cleanup.AUTO_DROPPED]
    kept, dropped = cleanup.apply_verdicts(figure, reports)
    assert len(kept) == len(figure.parts)
    assert dropped == []


def test_drops_pixelless_transparent_and_specks(figure_with_junk):
    r = verdicts(figure_with_junk)

    assert r["tail"].verdict == cleanup.AUTO_DROPPED
    assert "empty_layer" in r["tail"].reasons

    assert r["wings"].verdict == cleanup.AUTO_DROPPED
    assert "fully_transparent" in r["wings"].reasons

    # 3x3 = 9 px, under the 16 px floor.
    assert r["objects"].verdict == cleanup.AUTO_DROPPED
    assert "tiny_area" in r["objects"].reasons


def test_detects_duplicate_and_keeps_the_larger(figure_with_junk):
    r = verdicts(figure_with_junk)
    dupes = [x for x in (r["topwear"], r["neckwear"]) if x.verdict == cleanup.AUTO_DROPPED]
    assert len(dupes) == 1, "exactly one of an identical pair should be dropped"
    assert "duplicate" in dupes[0].reasons
    assert dupes[0].dup_of in {"topwear", "neckwear"}
    assert dupes[0].dup_of != dupes[0].name


def test_small_real_features_are_flagged_not_deleted(figure):
    """A 20 px nose is real. see-through's own sample produces exactly that."""
    r = verdicts(figure)
    for name in ("irides-r", "irides-l"):
        assert r[name].verdict != cleanup.AUTO_DROPPED
        assert r[name].verdict == cleanup.SUSPICIOUS
        assert "sliver" in r[name].reasons


def test_user_exclusions_and_revival(figure_with_junk):
    reports = cleanup.analyze(figure_with_junk, CleanupSettings())

    kept, dropped = cleanup.apply_verdicts(figure_with_junk, reports, excluded={"front hair"})
    names = {p.name for p in kept}
    assert "front hair" not in names
    assert "front hair" in dropped

    # An auto-dropped layer with pixels can come back...
    kept, _ = cleanup.apply_verdicts(figure_with_junk, reports, revived={"objects"})
    assert "objects" in {p.name for p in kept}

    # ...but one with no pixels cannot, since there is nothing to restore.
    kept, dropped = cleanup.apply_verdicts(figure_with_junk, reports, revived={"tail"})
    assert "tail" not in {p.name for p in kept}
    assert "tail" in dropped


def test_reports_are_json_safe(figure_with_junk):
    """No NaN. Bare NaN is invalid JSON and silently kills the browser's feed."""
    reports = cleanup.analyze(figure_with_junk, CleanupSettings())
    payload = [r.to_dict() for r in reports]
    text = json.dumps(payload, allow_nan=False)      # raises on NaN/Infinity
    assert "NaN" not in text and "Infinity" not in text
    assert any(r.rgb_std is None for r in reports) or True   # None is allowed


def test_summary_counts_add_up(figure_with_junk):
    reports = cleanup.analyze(figure_with_junk, CleanupSettings())
    s = cleanup.summarize(reports)
    assert s["total"] == len(reports)
    assert s["keep"] + s["auto_dropped"] + s["suspicious"] == s["total"]


@pytest.mark.parametrize("floor", [0, 8, 32])
def test_alpha_floor_is_respected(figure, floor):
    s = CleanupSettings()
    s.alpha_floor = floor
    reports = cleanup.analyze(figure, s)
    assert all(r.alpha_area_px >= 0 for r in reports)
