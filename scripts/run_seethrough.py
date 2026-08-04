"""Run see-through's inference script on whatever accelerator this machine has.

A thin wrapper: pick the device, install the CUDA redirection from
:mod:`ocs.torch_device`, then hand off to the real script with argv untouched.
The target is run through ``runpy`` with ``run_name="__main__"`` so its
``if __name__ == '__main__'`` body executes exactly as it would on its own.

Must be launched with cwd set to the submodule root -- see-through's default
paths are relative to it. ``ocs.seethrough`` already does that.

    python scripts/run_seethrough.py inference/scripts/inference_psd.py --srcp a.png ...

Set ``OCS_TORCH_DEVICE=cpu`` to override the choice.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocs import torch_device


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2

    script = Path(sys.argv[1]).resolve()
    if not script.exists():
        print(f"error: {script} not found", file=sys.stderr)
        return 1

    device = torch_device.select_device()
    info = torch_device.describe(device)
    print(
        f"[ocs] device {info['device']} ({info['name']}"
        + (f", {info['memory_gb']} GB" if info["memory_gb"] else "")
        + f") torch {info['torch']}",
        flush=True,
    )

    if device.type == "mps":
        # Must be set before MPS initialises, and the allocator cap in here is
        # load-bearing -- see configure_mps_env for what the default does.
        for key, value in torch_device.configure_mps_env().items():
            print(f"[ocs] {key}={value}", flush=True)

    torch_device.redirect_cuda_to(device)

    # Drop our own argv[1] so the target sees the argv it expects.
    sys.argv = [str(script)] + sys.argv[2:]
    _cap_resolution_args(sys.argv, device)
    runpy.run_path(str(script), run_name="__main__")
    return 0


def _cap_resolution_args(argv: list[str], device) -> None:
    """Lower ``--resolution`` in place when the device cannot hold it.

    Done here rather than in the UI so every entry point is covered -- the server,
    a hand-run command, a stale saved project whose settings predate the cap. The
    override is printed, never silent: a run that quietly produced a smaller
    decomposition than asked for would be worse than one that explained itself.
    """
    for flag in ("--resolution", "--resolution_depth"):
        if flag not in argv:
            continue
        i = argv.index(flag) + 1
        if i >= len(argv):
            continue
        try:
            requested = int(argv[i])
        except ValueError:
            continue
        capped, note = torch_device.cap_resolution(requested, device)
        if note:
            argv[i] = str(capped)
            print(f"[ocs] {flag} {note}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
