"""Run the tail of a see-through inference over an existing layer directory.

``inference_psd.py`` is layerdiff -> marigold -> further_extr, and the expensive
part by far is the first. A run that dies after it leaves every layer on disk but
none of what the last two produce, and the difference matters more than it sounds:

- **marigold** is where ``depth_median`` comes from. Without it every layer carries
  the same 1.0 default and draw order falls back to ``taxonomy.Z_PRIOR``, a table
  written for a standing figure -- so on a seated one a raised bare leg is ordered
  behind the skirt it is plainly in front of.
- **further_extr** is where ``--tblr_split`` is applied, so without it paired tags
  like ``handwear`` arrive with both sleeves in one layer. It is also what writes
  the PSD.

``ocs.psd_io.read_layer_dir`` and the draw-order inference in ``ocs.rig`` exist to
cope when this cannot be run. Running it is better: it supplies the real inputs
rather than estimating around their absence.

    python scripts/finish_seethrough.py workspace/projects/<id>/seethrough/input

Goes through ``run_seethrough.py`` for the device setup, same as a full run.
"""

from __future__ import annotations

import argparse
import os.path as osp
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layer_dir", type=Path,
                    help="see-through's per-image output directory (holds src_img.png)")
    ap.add_argument("--repo-id-depth", default="24yearsold/seethroughv0.0.1_marigold")
    ap.add_argument("--resolution-depth", type=int, default=768)
    ap.add_argument("--inference-steps-depth", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-offload", action="store_true", default=True)
    ap.add_argument("--tblr-split", action="store_true", default=True)
    ap.add_argument("--skip-depth", action="store_true",
                    help="only run further_extr, e.g. when depth already exists")
    args = ap.parse_args()

    layer_dir = args.layer_dir.resolve()
    if not (layer_dir / "src_img.png").exists():
        print(f"error: {layer_dir}/src_img.png not found -- is this a layer dir?",
              file=sys.stderr)
        return 1

    # apply_marigold takes the *parent* plus the stem, the way inference_psd.py
    # calls it: save_dir/<stem>/ is what it reads and writes.
    save_dir, stem = str(layer_dir.parent), layer_dir.name

    from utils.inference_utils import apply_marigold, further_extr

    if not args.skip_depth:
        from ocs import torch_device
        res, note = torch_device.cap_resolution(
            args.resolution_depth, torch_device.select_device())
        if note:
            print(f"[ocs] --resolution_depth {note}", flush=True)
        print("running marigold...", flush=True)
        apply_marigold(
            osp.join(save_dir, stem + ".png"), args.repo_id_depth,
            save_dir=save_dir, seed=args.seed, resolution=res,
            num_inference_steps=args.inference_steps_depth,
            group_offload=args.group_offload,
        )

    print("running further_extr...", flush=True)
    further_extr(str(layer_dir), rotate=False, save_to_psd=True,
                 tblr_split=args.tblr_split)
    print(f"\ndone. PSD should be at {layer_dir.with_suffix('.psd')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
