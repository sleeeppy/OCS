"""The progress bar must never go backwards.

Progress is read from files landing on disk, not from tqdm. apply_layerdiff runs
the 30-step schedule twice for tag v3, and both used to report as n/30 -- a
parser that assumed one loop climbed to 60% and snapped back to 18% when the
head pass began. File counts only increase, so that reversal cannot happen.
"""

from __future__ import annotations

from pathlib import Path

from ocs import seethrough


def test_phase_markers_are_ordered_and_monotonic():
    events: list[tuple[str, float]] = [("starting see-through", 0.01)]
    for line in ("running layerdiff...", "running marigold...", "psd saved"):
        found = seethrough._phase_of(line)
        assert found is not None
        base, _span = seethrough._PHASE_SPAN.get(found, (0.98, 0.0))
        events.append((found, base))

    phases = [p for p, _ in events]
    assert phases[0] == "starting see-through"
    assert phases.index("marigold") > phases.index("layerdiff")
    assert phases[-1] == "psd"

    values = [frac for _, frac in events]
    dips = [(i, events[i], events[i + 1])
            for i in range(len(values) - 1) if values[i + 1] < values[i] - 1e-9]
    assert not dips, f"bar went backwards at {dips[:4]}"
    for _, frac in events:
        assert 0.0 <= frac <= 1.0


def test_count_outputs_ignores_src_and_reconstruction(tmp_path: Path):
    (tmp_path / "face.png").write_bytes(b"")
    (tmp_path / "topwear.png").write_bytes(b"")
    (tmp_path / "face_depth.png").write_bytes(b"")
    (tmp_path / "src_img.png").write_bytes(b"")
    (tmp_path / "reconstruction.png").write_bytes(b"")
    layers, depths, newest = seethrough._count_outputs(tmp_path)
    assert layers == 2
    assert depths == 1
    assert newest > 0


def test_count_outputs_is_monotonic_as_files_land(tmp_path: Path):
    seen: list[float] = []
    for i, name in enumerate(("face.png", "topwear.png", "face_depth.png")):
        (tmp_path / name).write_bytes(b"")
        layers, depths, _newest = seethrough._count_outputs(tmp_path)
        current = "layerdiff" if depths == 0 else "marigold"
        base, span = seethrough._PHASE_SPAN[current]
        done, total = ((layers, seethrough.LAYERDIFF_OUTPUTS) if current == "layerdiff"
                       else (depths, seethrough.MARIGOLD_OUTPUTS))
        frac = min(0.97, base + span * (done / total))
        seen.append(frac)
        assert i == 0 or seen[i] >= seen[i - 1]
