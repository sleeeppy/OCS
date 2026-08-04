"""Create an OCS project from existing see-through output, skipping the GPU pass.

Useful when you already ran see-through by hand, or when iterating on the cleanup
/ partition / rigging stages without paying for inference each time.

Takes either the PSD or the per-tag layer directory.

**PSD** -- the complete form, with see-through's usual sidecars beside it:

    <name>.psd            required
    <name>.psd.json       recommended (depth_median, xyxy)
    <name>_depth.psd      optional
    <name>/src_img.png    recommended (the editor backdrop and silhouette source)

**Layer directory** -- the recovery form, for when the run did not finish:

    <name>/<tag>.png      one per tag, written as inference goes
    <name>/src_img.png

``apply_layerdiff`` writes each tag's PNG as it produces it, but the PSD is only
assembled at the end by ``further_extr``. So an interruption after the body pass
leaves every layer on disk inside a run that OCS would otherwise call a total
loss -- on Apple Silicon, where one image takes about an hour, that is worth
recovering. Ordering falls back to ``taxonomy.Z_PRIOR`` because the depth pass has
not run; see ``psd_io.read_layer_dir``.

Usage:
    python scripts/import_psd.py workspace/seethrough/test_image.psd
    python scripts/import_psd.py workspace/projects/<id>/seethrough/input/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from ocs import cleanup, pipeline, psd_io, silhouette, skeleton
from ocs.config import PROJECTS_DIR, ensure_dirs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path,
                    help="see-through PSD, or its per-tag layer directory")
    ap.add_argument("--name", default=None, help="project name (defaults to the stem)")
    args = ap.parse_args()

    src = args.src.resolve()
    if not src.exists():
        print(f"error: {src} not found", file=sys.stderr)
        return 1

    ensure_dirs()
    name = args.name or src.stem
    pid = time.strftime("%Y%m%d-%H%M%S") + "-import"
    root = PROJECTS_DIR / pid
    (root / "seethrough").mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        # Recovery path: the PSD was never written, so the layer PNGs are it.
        dest = root / "seethrough" / src.name
        shutil.copytree(src, dest, dirs_exist_ok=True)
        decomp = psd_io.read_layer_dir(dest)
        psd_rel = None
        layer_dir_rel = str(dest.relative_to(root))
        print(f"recovered from layer dir: {len(decomp.parts)} tags, no depth pass")
    else:
        # Copy the PSD plus whichever sidecars exist.
        shutil.copy2(src, root / "seethrough" / src.name)
        for extra in (Path(str(src) + ".json"), src.with_name(src.stem + "_depth.psd")):
            if extra.exists():
                shutil.copy2(extra, root / "seethrough" / extra.name)
        src_dir = src.with_suffix("")
        if src_dir.is_dir():
            shutil.copytree(src_dir, root / "seethrough" / src_dir.name, dirs_exist_ok=True)
        decomp = psd_io.read_decomposition(root / "seethrough" / src.name)
        psd_rel = str((root / "seethrough" / src.name).relative_to(root))
        layer_dir_rel = None

    print(f"canvas {decomp.canvas}, {len(decomp.parts)} layers")

    if decomp.src_img is not None:
        Image.fromarray(decomp.src_img).save(root / "input.png")
    else:
        Image.fromarray(decomp.parts[0].rgba).save(root / "input.png")

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
        "name": name,
        "created_at": time.time(),
        "stage": pipeline.STAGE_REVIEW,
        "progress": 1.0,
        "message": "imported",
        "settings": {},
        "input": {"filename": src.name, "width": decomp.canvas[0],
                  "height": decomp.canvas[1], "had_alpha": None},
        "psd": psd_rel,
        "layer_dir": layer_dir_rel,
        "canvas": {"width": decomp.canvas[0], "height": decomp.canvas[1]},
        "layers": [{**r.to_dict(), **layer_info.get(r.name, {})} for r in reports],
        "cleanup_summary": cleanup.summarize(reports),
        "auto_dropped": dropped,
        "exclusions": [],
        "revived": [],
        "silhouette": sil.to_dict(),
        "warnings": sil.warnings,
    })
    project.save()

    print(f"cleanup   {project.state['cleanup_summary']}")
    print(f"silhouette {sil.method} ({sil.coverage:.1%})")
    print(f"bones     {len(guess.bones)}")
    print(f"\nproject {pid}")
    print(f"open    http://127.0.0.1:8765/?project={pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
