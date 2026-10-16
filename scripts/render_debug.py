"""Render a diagnostic sheet for one project: silhouette, bones, bone partition.

Useful for judging a rig without opening the editor, and for filing a bug report
that shows what OCS actually decided.

    python scripts/render_debug.py                        # newest project
    python scripts/render_debug.py --project 20260803-1200-abc
    python scripts/render_debug.py --psd path/to/foo.psd  # straight from a PSD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocs import cleanup, limbs, pipeline, psd_io, silhouette, skeleton, taxonomy
from ocs.config import PROJECTS_DIR, RigSettings

PALETTE = np.array([
    [255, 92, 92], [255, 152, 58], [90, 152, 255], [58, 220, 220],
    [190, 102, 255], [255, 102, 200], [122, 220, 122], [226, 226, 92],
    [150, 150, 166],
], dtype=np.uint8)

TILE = 560
COL_BONE = (0, 210, 235)
COL_JOINT = (255, 205, 20)
COL_JOINT_OPT = (182, 243, 106)


def _draw_bones(img: Image.Image, rig: skeleton.Rig) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    for spec in taxonomy.BONE_TEMPLATE:
        if spec.name not in rig.bones or not spec.parent or spec.parent not in rig.bones:
            continue
        d.line([rig.bones[spec.parent], rig.bones[spec.name]], fill=COL_BONE, width=4)
    for spec in taxonomy.BONE_TEMPLATE:
        if spec.name not in rig.bones:
            continue
        x, y = rig.bones[spec.name]
        colour = COL_JOINT_OPT if spec.optional else COL_JOINT
        d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=colour, outline=(27, 29, 34), width=2)
    for spec in taxonomy.BONE_TEMPLATE:
        if spec.name not in rig.bones:
            continue
        x, y = rig.bones[spec.name]
        d.text((x, y + 20), spec.name, fill=(242, 244, 248), anchor="mm",
               stroke_width=3, stroke_fill=(10, 12, 16))
    return out


def _tint(base: np.ndarray, mask: np.ndarray, colour) -> Image.Image:
    out = base.astype(np.float32).copy()
    out[mask] = out[mask] * 0.4 + np.array(colour, np.float32) * 0.6
    return Image.fromarray(out.astype(np.uint8))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=None)
    ap.add_argument("--psd", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    project = None
    if args.psd:
        psd_path = args.psd.resolve()
        out_path = args.out or psd_path.with_name(psd_path.stem + "_debug.png")
        decomp = psd_io.read_decomposition(psd_path, load_depth=False)
    else:
        pid = args.project
        if pid is None:
            candidates = sorted(p for p in PROJECTS_DIR.glob("*/state.json"))
            if not candidates:
                print("no projects found; pass --psd instead", file=sys.stderr)
                return 1
            pid = candidates[-1].parent.name
        project = pipeline.Project.load(pid)
        out_path = args.out or project.root / "debug.png"
        # Goes through the pipeline so a project with no PSD -- a demo one, on a
        # machine with no CUDA to produce a PSD -- resolves the same way the
        # server does.
        decomp = pipeline.load_decomposition(project)

    reports = cleanup.analyze(decomp)
    kept, dropped = cleanup.apply_verdicts(decomp, reports)
    sil = silhouette.character_mask(decomp, kept)
    # For a project, show the rig that is actually in effect, user edits and all
    # -- the point of this sheet is what OCS decided, not what it first guessed.
    if project is not None and project.rig_path.exists():
        rig = pipeline.load_rig(project)
    else:
        rig = skeleton.guess_rig(decomp, kept)
    parts, report = limbs.partition(decomp, kept, rig, RigSettings())
    verify = limbs.verify_limb_separation(parts, RigSettings())

    print(f"canvas     {decomp.canvas}")
    print(f"cleanup    {cleanup.summarize(reports)}  dropped={dropped}")
    print(f"silhouette {sil.method} ({sil.coverage:.1%})")
    for w in sil.warnings:
        print(f"  ! {w}")
    print(f"parts      {len(kept)} -> {len(parts)}")
    for src, pieces in report["garment_slices"].items():
        print(f"  {src:<18} -> {[taxonomy.part_region(p) for p in pieces]}")
    if report["lr_split"]:
        print(f"  lr split  {report['lr_split']}")
    for f in report["forced"]:
        print(f"  forced    {f}")
    print(f"verify     {verify}")

    base = decomp.src_img[..., :3] if decomp.src_img is not None else np.zeros(
        (*decomp.canvas[::-1], 3), np.uint8)

    labels, specs = limbs.region_labels(rig, sil.mask)
    regions = base.astype(np.float32).copy()
    for i in range(len(specs)):
        m = labels == i
        regions[m] = regions[m] * 0.45 + PALETTE[i % len(PALETTE)] * 0.55
    regions_img = Image.fromarray(regions.astype(np.uint8))

    tiles = [
        ("source", Image.fromarray(base)),
        ("silhouette", _tint(base, sil.mask, (120, 255, 140))),
        ("bones", _draw_bones(Image.fromarray(base), rig)),
        ("bone partition", _draw_bones(regions_img, rig)),
    ]
    sheet = Image.new("RGB", (TILE * len(tiles), TILE), (18, 18, 22))
    for i, (_label, tile) in enumerate(tiles):
        sheet.paste(tile.resize((TILE, TILE), Image.LANCZOS), (i * TILE, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"\nwrote {out_path}   ({' | '.join(n for n, _ in tiles)})")
    print(f"regions order: {[s.name for s in specs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
