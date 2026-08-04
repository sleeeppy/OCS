"""The progress bar must never go backwards.

Two things make that non-trivial here. apply_layerdiff runs the 30-step schedule
*twice* for tag v3 -- once over the body tags, once over a crop around the head --
and both report as n/30. And _STEP's (\\d+)/(\\d+) also matches unrelated tqdm
bars ("Loading weights: 517/517"), which interleave with the denoise steps.

Measured on a real 아로나_B log: assuming one loop gave 4 reversals with a 0.55
drop -- the bar climbing to 60% and snapping back to 18% when the head pass began.
"""

from __future__ import annotations

from ocs import seethrough


def drive(lines: list[str]) -> list[tuple[str, float]]:
    """Feed lines through run_inference's progress logic via a fake subprocess."""
    seen: list[tuple[str, float]] = []

    class FakeProc:
        def __init__(self, out): self.stdout = iter(out)
        def wait(self): return 0

    import subprocess

    original = subprocess.Popen
    subprocess.Popen = lambda *a, **k: FakeProc([ln + "\n" for ln in lines])
    try:
        try:
            seethrough.run_inference(
                "img.png", "out",
                on_progress=lambda phase, frac: seen.append((phase, frac)),
            )
        except Exception:
            # It ends in "PSD was not written"; the progress stream is the point.
            pass
    finally:
        subprocess.Popen = original
    return seen


def _denoise(n: int, total: int = 30) -> list[str]:
    return [f"  {int(i / total * 100)}%|##  | {i}/{total} [00:10<00:20, 10s/it]"
            for i in range(1, n + 1)]


def test_two_layerdiff_passes_do_not_reset_the_bar():
    lines = (
        ["running layerdiff..."]
        + _denoise(30)              # body pass
        + _denoise(30)              # head-crop pass, restarts at 1/30
        + ["running marigold..."]
        + _denoise(20, 20)
        + ["psd saved"]
    )
    events = drive(lines)
    values = [frac for _, frac in events]
    assert values, "no progress was reported at all"
    dips = [(i, events[i], events[i + 1])
            for i in range(len(values) - 1) if values[i + 1] < values[i] - 1e-9]
    assert not dips, f"bar went backwards at {dips[:4]}"


def test_unrelated_tqdm_bars_cannot_drag_it_backwards():
    """"Loading weights: 517/517" matches _STEP just as well as a denoise step."""
    lines = (
        ["running layerdiff..."]
        + _denoise(20)
        + ["Loading weights: 100%|####| 517/517 [00:00<00:00, 7060.70it/s]"]
        + ["Loading pipeline components...: 100%|####| 5/5 [00:01<00:00, 4.01it/s]"]
        + _denoise(30)
    )
    events = drive(lines)
    values = [frac for _, frac in events]
    dips = [(i, events[i], events[i + 1])
            for i in range(len(values) - 1) if values[i + 1] < values[i] - 1e-9]
    assert not dips, f"bar went backwards at {dips[:4]}"


def test_phase_changes_are_reported_and_ordered():
    lines = ["running layerdiff...", *_denoise(30), *_denoise(30),
             "running marigold...", *_denoise(20, 20), "psd saved"]
    phases = [p for p, _ in drive(lines)]
    assert phases[0] == "starting see-through"
    assert "layerdiff" in phases and "marigold" in phases
    assert phases[-1] == "psd"
    # marigold must not start before layerdiff's two passes are accounted for.
    assert phases.index("marigold") > phases.index("layerdiff")


def test_progress_stays_within_bounds():
    lines = ["running layerdiff...", *_denoise(30), *_denoise(30),
             "running marigold...", *_denoise(20, 20), "psd saved"]
    for _, frac in drive(lines):
        assert 0.0 <= frac <= 1.0
