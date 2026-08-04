"""Drive the see-through submodule as a subprocess.

Run out-of-process on purpose: it imports torch and loads ~12 GB of diffusion
weights onto the GPU, and the web server should not carry that. The two share one
virtualenv (see ``scripts/setup_env.ps1``), so the same interpreter runs both.

``cwd`` must be the submodule root. ``inference_psd.py`` does
``sys.path.append`` of its own parent and imports ``utils.*`` / ``modules.*``
from the editable ``common`` install, and its default paths are relative.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from .config import PYTHON, SEE_THROUGH_DIR, SeeThroughSettings

ProgressFn = Callable[[str, float | None], None]

#: How many layer PNGs ``apply_layerdiff`` writes on the v3 path: 13 from the body
#: pass plus 11 from the head-crop pass.
LAYERDIFF_OUTPUTS = 24
#: Roughly how many ``{tag}_depth.png`` files ``apply_marigold`` writes.
MARIGOLD_OUTPUTS = 23

#: Fraction of the bar each phase owns.
_PHASE_SPAN = {"layerdiff": (0.04, 0.56), "marigold": (0.60, 0.36)}


class SeeThroughError(RuntimeError):
    pass


def psd_path_for(image_path: str | Path, save_dir: str | Path) -> Path:
    """Where ``further_extr`` will write the PSD for this input."""
    stem = Path(image_path).stem
    return Path(save_dir) / f"{stem}.psd"


def _stream(proc: subprocess.Popen) -> Iterator[str]:
    assert proc.stdout is not None
    for raw in proc.stdout:
        yield raw.rstrip("\r\n")


def _phase_of(line: str) -> str | None:
    low = line.lower()
    if "running layerdiff" in low:
        return "layerdiff"
    if "running marigold" in low:
        return "marigold"
    if "psd saved" in low:
        return "psd"
    return None


def _count_outputs(out_dir: Path) -> tuple[int, int, float]:
    """(layer PNGs, depth PNGs, newest mtime) written so far."""
    layers = depths = 0
    newest = 0.0
    try:
        entries = list(out_dir.iterdir())
    except OSError:
        return 0, 0, 0.0
    for p in entries:
        if p.suffix.lower() != ".png":
            continue
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            pass
        name = p.name
        if "_depth" in name:
            depths += 1
        elif not name.startswith("src_") and name != "reconstruction.png":
            layers += 1
    return layers, depths, newest


def _watch_outputs(
    out_dir: Path, phase: dict, stop: threading.Event, on_progress: ProgressFn
) -> None:
    """Report progress from files landing on disk.

    Parsing tqdm does not work here. OCS passes ``--disable_progressbar``, so the
    only percentages the subprocess prints come from the model *loader*, and a
    regex looking for "n%" latches onto "Loading pipeline components 100%" and then
    never changes -- the bar sat at a fixed number for fifteen minutes while the
    job was in fact running fine, which is indistinguishable from a hang.

    File counts are ground truth: ``apply_layerdiff`` writes one PNG per tag and
    ``apply_marigold`` one ``_depth`` PNG per part. Reporting the newest write also
    tells the UI when something last actually happened, which is the question a
    stalled-looking bar really raises.
    """
    last: tuple[int, int] = (-1, -1)
    while not stop.wait(2.0):
        layers, depths, newest = _count_outputs(out_dir)
        current = phase.get("name", "starting")
        if (layers, depths) == last and current != "starting":
            # No new file, but say so with the timestamp rather than going silent.
            idle = time.time() - newest if newest else 0.0
            if idle > 20:
                on_progress(f"{current} {_fmt(layers, depths, current)} "
                            f"(last write {int(idle)}s ago)", None)
            continue
        last = (layers, depths)
        base, span = _PHASE_SPAN.get(current, (0.02, 0.0))
        done, total = ((layers, LAYERDIFF_OUTPUTS) if current == "layerdiff"
                       else (depths, MARIGOLD_OUTPUTS))
        frac = min(1.0, done / total) if total else 0.0
        on_progress(f"{current} {_fmt(layers, depths, current)}",
                    min(0.97, base + span * frac))


def _fmt(layers: int, depths: int, phase: str) -> str:
    if phase == "layerdiff":
        return f"{layers}/{LAYERDIFF_OUTPUTS} layers"
    if phase == "marigold":
        return f"{depths}/{MARIGOLD_OUTPUTS} depth maps"
    return ""


def run_inference(
    image_path: str | Path,
    save_dir: str | Path,
    settings: SeeThroughSettings | None = None,
    on_progress: ProgressFn | None = None,
    log_path: str | Path | None = None,
) -> Path:
    """Decompose one image. Returns the PSD path.

    The first call downloads the LayerDiff 3D, Marigold and SAM weights from
    HuggingFace (~12 GB), so it is slow exactly once.
    """
    s = settings or SeeThroughSettings()
    image_path = Path(image_path).resolve()
    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    script = SEE_THROUGH_DIR / "inference" / "scripts" / "inference_psd.py"
    if not script.exists():
        raise SeeThroughError(
            f"see-through not found at {SEE_THROUGH_DIR}. "
            "Run: git submodule update --init --recursive"
        )

    cmd = [
        PYTHON, str(script),
        "--srcp", str(image_path),
        "--save_dir", str(save_dir),
        *s.to_args(),
        "--disable_progressbar",
    ]

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")

    if on_progress:
        on_progress("starting see-through", 0.01)

    # Where apply_layerdiff / apply_marigold write their per-tag PNGs.
    out_dir = save_dir / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    log = open(log_path, "w", encoding="utf-8") if log_path else None
    phase: dict[str, str] = {"name": "starting"}
    stop = threading.Event()
    watcher: threading.Thread | None = None
    if on_progress:
        watcher = threading.Thread(
            target=_watch_outputs, args=(out_dir, phase, stop, on_progress), daemon=True
        )
        watcher.start()

    try:
        proc = subprocess.Popen(
            cmd, cwd=str(SEE_THROUGH_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        # stdout is read only for the phase markers -- plain prints from
        # inference_psd.py. The fraction comes from the watcher above.
        for line in _stream(proc):
            if log:
                log.write(line + "\n")
                log.flush()
            found = _phase_of(line)
            if found:
                phase["name"] = found
                if on_progress:
                    base, _span = _PHASE_SPAN.get(found, (0.98, 0.0))
                    on_progress(found, base)
        code = proc.wait()
    finally:
        stop.set()
        if watcher:
            watcher.join(timeout=3)
        if log:
            log.close()

    if code != 0:
        tail = ""
        if log_path and Path(log_path).exists():
            tail = "\n".join(Path(log_path).read_text(encoding="utf-8").splitlines()[-25:])
        raise SeeThroughError(f"inference_psd.py exited {code}\n{tail}")

    psd = psd_path_for(image_path, save_dir)
    if not psd.exists():
        raise SeeThroughError(f"inference finished but {psd} was not written")
    if on_progress:
        on_progress("done", 1.0)
    return psd


def run_lr_split(
    psd_path: str | Path, tags: list[str], on_progress: ProgressFn | None = None
) -> Path:
    """Optional extra pass: ``heuristic_partseg.py seg_wlr`` on chosen tags.

    OCS normally does not need this -- it passes ``--tblr_split`` during inference
    and splits the remaining paired tags itself in ``ocs.limbs``, which keeps every
    connected component instead of only the two largest. Exposed for the cases
    where the upstream heuristic does better on a specific tag.
    """
    psd_path = Path(psd_path).resolve()
    script = SEE_THROUGH_DIR / "inference" / "scripts" / "heuristic_partseg.py"
    cmd = [
        PYTHON, str(script), "seg_wlr",
        "--srcp", str(psd_path),
        "--target_tags", ",".join(tags),
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        cmd, cwd=str(SEE_THROUGH_DIR), env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise SeeThroughError(f"seg_wlr exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    if on_progress:
        on_progress("lr_split", 1.0)
    return psd_path.with_name(psd_path.stem + "_lrsplit.psd")


def check_environment() -> dict:
    """Report whether the GPU half of the stack is usable, without importing torch here."""
    probe = (
        "import json,torch;"
        "print(json.dumps({"
        "'torch': torch.__version__,"
        "'cuda': torch.cuda.is_available(),"
        "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
        "'capability': list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,"
        "'vram_gb': round(torch.cuda.get_device_properties(0).total_memory/2**30,1) if torch.cuda.is_available() else None"
        "}))"
    )
    try:
        out = subprocess.run(
            [PYTHON, "-c", probe], capture_output=True, text=True, timeout=120,
        )
        if out.returncode != 0:
            return {"ok": False, "error": (out.stderr or out.stdout)[-400:]}
        import json as _json
        info = _json.loads(out.stdout.strip().splitlines()[-1])
        info["ok"] = bool(info.get("cuda"))
        info["see_through"] = SEE_THROUGH_DIR.exists()
        return info
    except Exception as exc:                              # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    import json
    print(json.dumps(check_environment(), indent=2))
    sys.exit(0)
