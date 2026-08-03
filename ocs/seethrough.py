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
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

from .config import PYTHON, SEE_THROUGH_DIR, SeeThroughSettings

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
