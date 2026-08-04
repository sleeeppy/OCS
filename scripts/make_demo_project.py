"""Create an OCS project from a synthetic decomposition, with no GPU at all.

``scripts/import_psd.py`` skips the see-through pass but still needs a PSD that
see-through produced. On a machine that cannot run see-through at all -- no
NVIDIA GPU -- there is no PSD to import and the editor has nothing to open past
the upload step.

This builds the project directly from :mod:`ocs.demo`, whose layouts are
measured from see-through's own sample: ``handwear`` is a whole arm, ``legwear``
is both legs in one layer, there is no skin tag, and the empty/duplicate/speck
layers are present. So it exercises the same code the GPU path feeds -- layer
cleanup, silhouette, bone guess -- rather than standing in for it.

The stages after this point (limb partition, meshing, weighting, atlas, Spine
export, preview) are pure numpy/opencv/scipy and run identically here.

    python scripts/make_demo_project.py                 # the junk figure
    python scripts/make_demo_project.py --figure blob    # requirement 2-2 worst case
    python scripts/make_demo_project.py --all            # one project per figure

``blob`` is the one worth looking at: a single connected silhouette in a single
layer, arms fused to the torso, no left/right suffixes anywhere. Only the bone
skeleton can separate the sides there.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from ocs import cleanup, demo, pipeline, silhouette, skeleton
from ocs.config import PROJECTS_DIR, ensure_dirs


def build(figure_name: str, suffix: str = "") -> str:
    """Run the pre-editor half of the pipeline and save a project. Returns its id."""
    make = demo.FIGURES[figure_name]
    decomp = make()

    ensure_dirs()
    pid = time.strftime("%Y%m%d-%H%M%S") + f"-demo-{figure_name}{suffix}"
    root = PROJECTS_DIR / pid
    root.mkdir(parents=True, exist_ok=True)

    # The editor backdrop and the silhouette source. demo composites the parts
    # back down, which is what a real illustration would have been.
    Image.fromarray(decomp.src_img).save(root / "input.png")

    # Same sequence as scripts/import_psd.py from here on.
    layer_info = pipeline._write_layer_images(decomp, root / "layers")
    reports = cleanup.analyze(decomp)
    kept, dropped = cleanup.apply_verdicts(decomp, reports)
    sil = silhouette.character_mask(decomp, kept)
    guess = skeleton.guess_rig(decomp, kept)
    (root / "rig.json").write_text(
        json.dumps(guess.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    project = pipeline.Project(id=pid, root=root, state={
        "id": pid,
        "name": f"demo/{figure_name}",
        "created_at": time.time(),
        "stage": pipeline.STAGE_REVIEW,
        "progress": 1.0,
        "message": f"synthetic demo ({figure_name}), no GPU used",
        "settings": {},
        "input": {"filename": f"{figure_name}.png", "width": decomp.canvas[0],
                  "height": decomp.canvas[1], "had_alpha": True},
        "psd": None,
        "canvas": {"width": decomp.canvas[0], "height": decomp.canvas[1]},
        "layers": [{**r.to_dict(), **layer_info.get(r.name, {})} for r in reports],
        "cleanup_summary": cleanup.summarize(reports),
        "auto_dropped": dropped,
        "exclusions": [],
        "revived": [],
        "silhouette": sil.to_dict(),
        "warnings": sil.warnings + ["Synthetic demo input; see-through was not run."],
        "demo": figure_name,
    })
    project.save()

    print(f"[{figure_name}] canvas {decomp.canvas}, {len(decomp.parts)} layers")
    print(f"[{figure_name}] cleanup    {project.state['cleanup_summary']}")
    print(f"[{figure_name}] auto-dropped {dropped or '[]'}")
    print(f"[{figure_name}] silhouette {sil.method} ({sil.coverage:.1%})")
    print(f"[{figure_name}] bones      {len(guess.bones)}")
    print(f"[{figure_name}] -> http://127.0.0.1:8765/?project={pid}\n")
    return pid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", default="junk", choices=sorted(demo.FIGURES),
                    help="which synthetic decomposition to build (default: junk)")
    ap.add_argument("--all", action="store_true", help="build one project per figure")
    args = ap.parse_args()

    names = sorted(demo.FIGURES) if args.all else [args.figure]
    # Distinct suffixes: the id is second-resolution and --all is faster than that.
    ids = [build(n, suffix=f"-{i}" if args.all else "") for i, n in enumerate(names)]

    print("open the editor:  ./scripts/run_ocs.sh")
    print(f"latest project:   http://127.0.0.1:8765/?project={ids[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
