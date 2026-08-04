"""Time see-through with and without ``group_offload``, and record peak VRAM.

OCS defaults ``group_offload`` on because see-through's README puts the plain path
at 12-16 GB at 1280, which is uncomfortably close to a 16 GB card. But it is not
free: it moves model blocks between CPU and GPU every step, and a run was observed
sitting at 100% GPU utilisation while drawing only 83 W of a 300 W limit -- busy
waiting on transfers, not computing -- while VRAM still reached 15.2 GB, so it was
paying the transfer cost without buying the headroom.

Whether that trade is worth it is a measurement, not an opinion. This runs the same
image both ways and prints wall clock and peak VRAM for each.

    python scripts/bench_offload.py path/to/character.png
    python scripts/bench_offload.py path/to/character.png --resolution 1024

Each pass loads ~12 GB of weights from the HF cache and runs two diffusion passes
plus depth, so budget a good while. Nothing else should be using the GPU.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocs.config import SeeThroughSettings, WORKSPACE
from ocs.seethrough import run_inference


def _poll_vram(stop: threading.Event, out: list[int]) -> None:
    while not stop.wait(2.0):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=8,
            )
            mem, power = (t.strip() for t in r.stdout.strip().splitlines()[0].split(","))
            out.append((int(float(mem)), float(power)))
        except Exception:                                   # noqa: BLE001
            pass


def one_run(image: Path, offload: bool, resolution: int, steps: int) -> dict:
    save_dir = WORKSPACE / "bench" / ("offload_on" if offload else "offload_off")
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    settings = SeeThroughSettings(
        resolution=resolution, inference_steps=steps, group_offload=offload
    )
    samples: list = []
    stop = threading.Event()
    watcher = threading.Thread(target=_poll_vram, args=(stop, samples), daemon=True)
    watcher.start()

    label = "on " if offload else "off"
    marks: list[tuple[str, float]] = []
    t0 = time.time()

    def progress(phase: str, frac: float | None) -> None:
        name = phase.split()[0]
        if not marks or marks[-1][0] != name:
            marks.append((name, time.time() - t0))
            print(f"  [offload {label}] {name:<10} at {time.time() - t0:6.1f}s", flush=True)

    error = None
    try:
        run_inference(image, save_dir, settings, on_progress=progress,
                      log_path=save_dir / "bench.log")
    except Exception as exc:                                # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()
        watcher.join(timeout=3)

    elapsed = time.time() - t0
    peak_mem = max((m for m, _p in samples), default=0)
    avg_power = sum(p for _m, p in samples) / len(samples) if samples else 0.0
    return {
        "offload": offload, "elapsed_s": elapsed, "peak_vram_mib": peak_mem,
        "avg_power_w": avg_power, "phases": marks, "error": error,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path)
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--only", choices=["on", "off"], default=None,
                    help="run just one side (useful if the other already OOM'd)")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"error: {args.image} not found", file=sys.stderr)
        return 1

    order = [False, True] if args.only is None else [args.only == "on"]
    results = []
    for offload in order:
        print(f"\n=== group_offload {'ON' if offload else 'OFF'} "
              f"(res {args.resolution}, {args.steps} steps) ===", flush=True)
        r = one_run(args.image, offload, args.resolution, args.steps)
        results.append(r)
        if r["error"]:
            print(f"  FAILED after {r['elapsed_s']:.0f}s: {r['error']}")
        else:
            print(f"  done in {r['elapsed_s']/60:.1f} min, "
                  f"peak VRAM {r['peak_vram_mib']} MiB, "
                  f"avg power {r['avg_power_w']:.0f} W")

    print("\n" + "=" * 62)
    print(f"{'group_offload':<15}{'wall clock':>13}{'peak VRAM':>13}{'avg power':>12}")
    for r in results:
        status = "OOM/failed" if r["error"] else f"{r['elapsed_s']/60:.1f} min"
        print(f"{'ON' if r['offload'] else 'OFF':<15}{status:>13}"
              f"{r['peak_vram_mib']:>10} MiB{r['avg_power_w']:>10.0f} W")

    ok = [r for r in results if not r["error"]]
    if len(ok) == 2:
        on = next(r for r in ok if r["offload"])
        off = next(r for r in ok if not r["offload"])
        speed = on["elapsed_s"] / off["elapsed_s"] if off["elapsed_s"] else 0
        print(f"\nOFF is {speed:.2f}x the speed of ON, and needs "
              f"{off['peak_vram_mib'] - on['peak_vram_mib']:+d} MiB more VRAM.")
        print("Set group_offload accordingly in the upload panel's advanced settings,"
              "\nor change the default in ocs/config.py SeeThroughSettings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
