"""Paths and tunables. Every default here can be overridden per project."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEE_THROUGH_DIR = REPO_ROOT / "external" / "see-through"
WEB_DIR = REPO_ROOT / "web"
VENDOR_DIR = WEB_DIR / "vendor"
WORKSPACE = Path(os.environ.get("OCS_WORKSPACE", REPO_ROOT / "workspace"))
PROJECTS_DIR = WORKSPACE / "projects"
SEETHROUGH_OUT = WORKSPACE / "seethrough"

#: The interpreter running OCS also has see-through's dependencies installed
#: (single shared venv, see scripts/setup_env.ps1), so subprocess calls reuse it.
PYTHON = os.environ.get("OCS_PYTHON", sys.executable)

SPINE_VERSION = "4.2"
SPINE_PLAYER_CDN_JS = (
    "https://unpkg.com/@esotericsoftware/spine-player@4.2.*/dist/iife/spine-player.js"
)
SPINE_PLAYER_CDN_CSS = (
    "https://unpkg.com/@esotericsoftware/spine-player@4.2.*/dist/spine-player.css"
)


@dataclass
class SeeThroughSettings:
    """Mirrors ``inference_psd.py``'s argparse surface.

    ``group_offload`` defaults on: see-through's own README puts the plain path
    at 12-16 GB of VRAM at 1280, which is right at the edge on a 16 GB card,
    while group offload brings the peak to ~10 GB for a modest speed cost. On
    Apple Silicon it is not a nicety -- without it a 1280 run swaps. Only CPU,
    where there is nothing to offload to, drops the flag. See
    ``ocs.torch_device.should_group_offload``.
    """

    resolution: int = 1280
    resolution_depth: int = 768
    inference_steps: int = 30
    seed: int = 42
    group_offload: bool = True
    #: Always on. It is what produces handwear-l/-r, ears-l/-r and the split eye
    #: tags; OCS handles the rest of the L/R work itself (see ocs.limbs).
    tblr_split: bool = True

    def to_args(self, device_type: str = "cuda") -> list[str]:
        args = [
            "--resolution", str(self.resolution),
            "--resolution_depth", str(self.resolution_depth),
            "--inference_steps", str(self.inference_steps),
            "--seed", str(self.seed),
            "--save_to_psd",
        ]
        if self.group_offload and device_type in ("cuda", "mps"):
            args.append("--group_offload")
        if self.tblr_split:
            args.append("--tblr_split")
        return args


@dataclass
class CleanupSettings:
    """Thresholds for dummy-layer detection (requirement 2).

    Split in two tiers on purpose: ``*_drop`` values are conservative enough to
    delete without asking, everything else only raises a flag the user resolves
    in the review step.
    """

    # tier 1 - dropped automatically
    alpha_floor: int = 8               # alpha <= this counts as transparent
    #: Absolute floor. Kept low on purpose: a real decomposition of the
    #: see-through sample has a 20 px ``nose`` and a 37 px ``mouth``, which are
    #: legitimate features. Anything that small still gets flagged as a sliver
    #: for the user to judge -- it just isn't deleted without asking.
    min_area_px: int = 16
    min_coverage_drop: float = 1e-5    # fraction of canvas
    min_alpha_mean_drop: float = 0.01  # mean alpha / 255
    dup_iou_drop: float = 0.97
    dup_rgb_delta_drop: float = 2.0

    # tier 2 - flagged as suspicious, user decides
    min_coverage_flag: float = 1e-3
    speckle_cc_ratio_flag: float = 0.20
    speckle_coverage_flag: float = 0.02
    flat_rgb_std_flag: float = 1.0
    contained_iou_flag: float = 0.95
    junk_tag_coverage_flag: float = 5e-3
    junk_tags: tuple[str, ...] = ("objects",)


@dataclass
class RigSettings:
    """Mesh + weighting quality knobs."""

    #: Contour simplification, as a fraction of the part's perimeter.
    contour_epsilon: float = 0.004
    #: Absolute cap on that tolerance, in pixels. A pure fraction erases small or
    #: thin parts -- a ribbon with a 200 px perimeter would get an 8 px tolerance,
    #: enough to collapse its outline below three points and fail to mesh.
    contour_epsilon_max_px: float = 4.0
    #: Interior sample spacing in pixels. Lower = denser mesh = smoother bends.
    #: Scaled down for small parts (see rig._interior_points).
    interior_spacing: int = 28
    #: Max bones influencing one vertex. Spine allows any number; 4 is plenty
    #: and keeps the JSON small.
    max_bones_per_vertex: int = 3
    #: Weight falloff. 2.0 = inverse distance squared.
    weight_falloff: float = 2.0
    #: Vertices closer than this to a bone are pinned fully to it.
    weight_pin_px: float = 4.0
    #: Keep each limb as one part spanning its whole chain, instead of cutting it
    #: at the elbow / knee.
    #:
    #: Cutting at a joint makes the two halves separate attachments in separate
    #: slots, and that boundary is a hard edge in the render however the weights
    #: are set -- 61.8% of the pixels that came out darker than the source art sat
    #: on a part's own edge. One mesh spanning the joint has no such boundary and
    #: still bends, because ``rig._candidate_bones`` already blends a part's own
    #: bone with its parent and children.
    #:
    #: The left/right split is unaffected. That is what requirement 2-2 asks for;
    #: a part per joint segment was an implementation choice on top of it.
    merge_limb_slices: bool = True
    #: How much of the silhouette the *other* layers may cover before a big layer
    #: stops counting as "the only layer" and keeps its tag restriction.
    #:
    #: The escape exists so a lone head-to-toe ``topwear`` can be cut into legs.
    #: Size alone does not establish loneness: on a seated figure a spread skirt
    #: covers 73.5% of the silhouette while the other layers still cover 49.8%, and
    #: releasing it let ``bottomwear`` claim the arm the sleeve already had. A
    #: genuinely lone layer has others at ~0. See ``limbs._slice_by_regions``.
    lone_layer_others_max: float = 0.25
    #: Overlap grown onto each partition region so neighbouring parts share a
    #: seam instead of showing a gap when the joint bends.
    seam_allowance_px: int = 8
    #: How far outside its alpha each mesh outline is traced, in pixels.
    #:
    #: One pixel covers the antialiased rim. Two is for the seams: ``_triangulate``
    #: drops boundary triangles, costing every part 0.6-4.6% of its alpha at its
    #: own edge, and where two slices of one garment meet both losses land in the
    #: same place. Overshooting costs nothing -- the extra band is transparent in
    #: the texture, so it renders as nothing until a neighbour's loss exposes it.
    outline_dilate_px: int = 2
    #: A partition slice smaller than this fraction of the source layer is
    #: discarded rather than emitted as a sliver attachment.
    min_slice_fraction: float = 0.02
    #: Repaint each part's *visible* pixels from the original artwork.
    #:
    #: see-through regenerates every layer with a diffusion model rather than
    #: cropping it, so the returned RGB drifts from the source: measured over one
    #: character, 11.8% of shared opaque pixels differ by more than 16 levels
    #: (mean 8.96), which reads as flattened hair shading and a smudged ribbon.
    #: Where a part is frontmost, the original already holds its exact pixels.
    #: Upstream does this itself for nose and mouth in ``further_extr``.
    restore_source_pixels: bool = True
    #: Only repaint pixels this opaque, **at the character's outer rim**. A partly
    #: transparent pixel there is a blend with the background, so copying it in
    #: would drag background colour into the outline.
    source_pixel_alpha_floor: int = 200
    #: The same floor away from the outer rim, where a partly transparent pixel is
    #: a blend with another part of the character rather than the background --
    #: which is the colour the seam should have. Leaving these un-repainted is what
    #: puts dark hairlines along every interior cut. See
    #: ``rig.restore_source_pixels`` for the measurements.
    source_pixel_alpha_floor_interior: int = 64
    #: How opaque a nearer part must be before it stops the part behind it being
    #: repainted from the source. Near-255 on purpose: a feathered edge at alpha 9
    #: hides nothing, so letting it claim the pixel leaves the *visible* part
    #: behind it holding see-through's drifted colour -- a dark line along every
    #: outline. See ``rig.restore_source_pixels``.
    source_pixel_claim_floor: int = 250
    #: Restore the opacity the layer split lost, where the artwork was opaque.
    #:
    #: Splitting one antialiased image into layers halves the alpha at every
    #: internal boundary, so ``collar over sleeve`` composites to 0.75 where the
    #: artwork was 1.0 and the background shows through the difference. That is the
    #: dark hairline. See ``rig.restore_source_alpha``.
    #: Extend every layer's opacity this far past its own edge, inside the artwork.
    #:
    #: The fix for visible cuts, and it is arithmetic: splitting one antialiased
    #: image into layers halves the alpha at each internal boundary, so
    #: ``collar over sleeve`` composites to 0.75 where the artwork was 1.0 and the
    #: background shows through. Extending the layer *behind* restores both the
    #: opacity and the original blend -- raising the front layer instead would show
    #: its colour alone. Plateaus at 4 px; darkness drops 62%. See
    #: ``rig.close_layer_seams``.
    layer_extend_px: int = 4
    restore_source_alpha: bool = True
    #: How opaque the *artwork* must be before its alpha is treated as solid and
    #: copied back. High, so the character's own soft outline -- legitimately
    #: semi-transparent -- keeps its feathering instead of gaining a hard fringe.
    source_alpha_solid_floor: int = 250
    #: How much alpha a part must already have at a pixel before it may be raised.
    #: Above zero so a part is never grown into territory it does not cover.
    source_alpha_touch_floor: int = 8
    #: Hold the composite down to the artwork's alpha where the artwork is soft.
    #:
    #: The mirror of ``restore_source_alpha``. Every see-through layer is inpainted
    #: complete, so ``front hair`` and ``back hair`` both carry the loose strands
    #: beside the face; their alphas stack past the artwork's and the gaps between
    #: the strands fill in, turning fine hair into a soft dark haze. Only applies
    #: outside the solid silhouette, so sheer fabric over skin is untouched.
    #: See ``rig.limit_source_alpha``.
    limit_source_alpha: bool = True


    #: Which preset animations to emit, or ``None`` for all of them.
    #:
    #: The presets assume a standing figure. On a seated character ``walk`` and
    #: ``jump`` are not merely unflattering, they are meaningless -- there is no
    #: stride to take -- and ``limb_swing_caps`` correctly shrinks them to almost
    #: nothing, which leaves the player listing animations that do not move.
    #: Naming the useful subset is better than shipping dead entries.
    animations: tuple[str, ...] | None = None


@dataclass
class AtlasSettings:
    padding: int = 4
    max_size: int = 4096
    power_of_two: bool = True


@dataclass
class OcsSettings:
    seethrough: SeeThroughSettings = None
    cleanup: CleanupSettings = None
    rig: RigSettings = None
    atlas: AtlasSettings = None

    def __post_init__(self):
        self.seethrough = self.seethrough or SeeThroughSettings()
        self.cleanup = self.cleanup or CleanupSettings()
        self.rig = self.rig or RigSettings()
        self.atlas = self.atlas or AtlasSettings()

    def to_dict(self) -> dict:
        return {
            "seethrough": asdict(self.seethrough),
            "cleanup": asdict(self.cleanup),
            "rig": asdict(self.rig),
            "atlas": asdict(self.atlas),
        }


def ensure_dirs() -> None:
    for d in (WORKSPACE, PROJECTS_DIR, SEETHROUGH_OUT, VENDOR_DIR):
        d.mkdir(parents=True, exist_ok=True)
