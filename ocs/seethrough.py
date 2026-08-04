"""Drive the see-through submodule as a subprocess.

Run out-of-process on purpose: it imports torch and loads ~12 GB of diffusion
weights onto the accelerator, and the web server should not carry that. The two
share one virtualenv (see ``scripts/setup_env.sh``), so the same interpreter runs
both.

``cwd`` must be the submodule root: ``inference_psd.py``'s default paths are
relative to it, and ``utils.*`` / ``modules.*`` come from the editable ``common``
install.

Everything goes through ``scripts/run_seethrough.py`` rather than calling the
target script directly. see-through hardcodes ``cuda``; that wrapper redirects it
to whatever this machine has, so the same code path serves an RTX card and an
Apple Silicon Mac. See :mod:`ocs.torch_device`.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

from .config import PYTHON, REPO_ROOT, SEE_THROUGH_DIR, SeeThroughSettings

#: OCS-owned launcher that sets the device up before importing see-through.
_RUNNER = REPO_ROOT / "scripts" / "run_seethrough.py"

#: Filled by the first check_environment() call. See its docstring.
_ENV_CACHE: dict | None = None

ProgressFn = Callable[[str, float | None], None]

#: tqdm/diffusers write "  35%|###   | 7/20" to stderr; pull the fraction out so
#: the UI can show a real bar instead of a spinner.
_PCT = re.compile(r"(\d{1,3})%\|")
_STEP = re.compile(r"(\d+)/(\d+)")


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


#: tqdm redraw lines. Useful for the progress bar, useless in an error message --
#: and there are thousands of them, so an unfiltered tail is nothing else.
_NOISE = re.compile(r"it/s|s/it|Materializing|^\s*$|\x1b\[|^\s*\[A")


def _describe_exit(code: int) -> str:
    """Explain an exit status in the terms the reader needs.

    A negative code is a signal, not a status, and "exited -15" invites a hunt for
    a bug that is not there: 15 is SIGTERM, i.e. something outside stopped it.
    """
    if code < 0:
        import signal

        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        hint = " (terminated externally, not a see-through failure)" if -code in (
            signal.SIGTERM, signal.SIGINT
        ) else ""
        if -code == signal.SIGKILL:
            hint = " (killed - usually the OS out-of-memory killer)"
        return f"see-through was stopped by {name}{hint}"
    return f"inference_psd.py exited {code}"


def _error_tail(log_path: str | Path | None, lines: int = 20) -> str:
    """Last meaningful lines of the log, with the progress redraws stripped.

    Dropping ``--disable_progressbar`` gave the UI a real progress bar and gave
    this a 4000-line log of tqdm redraws, so the raw tail showed a wall of
    ``Loading weights: 100%|###`` instead of the reason. Filter first.
    """
    if not log_path or not Path(log_path).exists():
        return ""
    raw = Path(log_path).read_text(encoding="utf-8", errors="replace")
    kept = [
        ln.rstrip() for ln in raw.replace("\r", "\n").splitlines()
        if ln.strip() and not _NOISE.search(ln)
    ]
    if not kept:
        return "(no diagnostic output; see the full log)"
    return "\n".join(kept[-lines:])


def _phase_of(line: str) -> tuple[str, float] | None:
    low = line.lower()
    if "running layerdiff" in low:
        return "layerdiff", 0.05
    if "running marigold" in low:
        return "marigold", 0.60
    if "psd saved" in low:
        return "psd", 0.98
    return None


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

    # No --disable_progressbar on purpose. It suppresses exactly the tqdm output
    # _PCT and _STEP exist to parse, so passing it left the UI on the three coarse
    # phase jumps below with nothing in between -- tolerable on a card that
    # finishes in two minutes, not on MPS where a run is far longer. The cost is a
    # noisier seethrough.log, which is a debug log.
    cmd = [
        PYTHON, str(_RUNNER), str(script),
        "--srcp", str(image_path),
        "--save_dir", str(save_dir),
        *s.to_args(device_type=device_type()),
    ]

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")

    if on_progress:
        on_progress("starting see-through", 0.01)

    log = open(log_path, "w", encoding="utf-8") if log_path else None
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(SEE_THROUGH_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        phase, base = "starting", 0.01
        for line in _stream(proc):
            if log:
                log.write(line + "\n")
                log.flush()
            found = _phase_of(line)
            if found:
                phase, base = found
                if on_progress:
                    on_progress(phase, base)
                continue
            if on_progress:
                m = _PCT.search(line) or _STEP.search(line)
                if m:
                    frac = (int(m.group(1)) / 100.0 if "%" in m.group(0)
                            else int(m.group(1)) / max(1, int(m.group(2))))
                    span = 0.55 if phase == "layerdiff" else 0.35
                    on_progress(phase, min(0.97, base + span * frac))
        code = proc.wait()
    finally:
        if log:
            log.close()

    if code != 0:
        raise SeeThroughError(f"{_describe_exit(code)}\n{_error_tail(log_path)}")

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
        PYTHON, str(_RUNNER), str(script), "seg_wlr",
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


def device_type() -> str:
    """``"cuda"`` / ``"mps"`` / ``"cpu"``, or ``"none"`` when torch is absent.

    Cheap enough to call per request: it reads the cached probe below.
    """
    return (check_environment().get("device") or "none:").split(":")[0]


def check_environment(refresh: bool = False) -> dict:
    """Report whether the decomposition stage is usable on this machine.

    Runs out-of-process so the web server never imports torch. Reports whatever
    accelerator exists rather than CUDA specifically -- on an Apple Silicon Mac
    ``ok`` is true with ``device: "mps"``, and only a machine with no accelerator
    at all (or no torch) comes back false.

    Cached, because the probe costs a subprocess that imports torch -- a couple of
    seconds -- and ``/api/health`` is polled. What it reports cannot change while
    the process runs, short of someone installing torch underneath it, so
    ``refresh=True`` exists for that case.
    """
    global _ENV_CACHE
    if _ENV_CACHE is not None and not refresh:
        return _ENV_CACHE

    probe = (
        "import json,sys;"
        "sys.path.insert(0, %r);"
        "from ocs import torch_device;"
        "d = torch_device.select_device();"
        "print(json.dumps(torch_device.describe(d)))" % str(REPO_ROOT)
    )
    try:
        out = subprocess.run(
            [PYTHON, "-c", probe], capture_output=True, text=True, timeout=120,
        )
        if out.returncode != 0:
            info = {"ok": False, "error": (out.stderr or out.stdout)[-400:]}
        else:
            import json as _json
            info = _json.loads(out.stdout.strip().splitlines()[-1])
            info["ok"] = info.get("device") not in (None, "cpu")
            if info["device"] == "cpu":
                info["error"] = (
                    "no GPU found (CUDA or Apple MPS). Decomposition would take "
                    "hours on CPU; import a PSD or build a demo project instead."
                )
    except Exception as exc:                              # noqa: BLE001
        info = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    info["see_through"] = SEE_THROUGH_DIR.exists()
    _ENV_CACHE = info
    return info


if __name__ == "__main__":
    import json
    print(json.dumps(check_environment(), indent=2))
    sys.exit(0)
