# OCS — OneClickSpine

Turn one character illustration into a rigged, animated **Spine 4.2** skeleton.

Drop in a PNG and OCS decomposes it into semantic layers, throws away the dummy
layers, cuts the left and right arms and legs apart, lets you drag the bones where
they belong, then builds a weighted mesh rig, packs an atlas, and hands back a
self-contained HTML preview.

```
image ──► layer decomposition ──► layer cleanup ──► bone placement ──► limb split ──► rig ──► Spine + preview
            (see-through, GPU)      you review        you drag         automatic     mesh+weights
```

Built on two upstream projects:

| | |
|---|---|
| [see-through](https://github.com/shitagaki-lab/see-through) | layer decomposition, run as a git submodule (Apache-2.0) |
| [spine-animation-ai](https://github.com/GenielabsOpenSource/spine-animation-ai) | prior art for the rigging stage — read, **not** reused (PolyForm NC) |

See [NOTICE.md](NOTICE.md) for why the second one is referenced rather than
vendored, and for the Spine runtime's licensing.

---

## Why this is not just glue code

Reading see-through's source turned up three gaps that a straight pipeline would
have shipped as bugs. Each one is a component here.

**1. see-through never filters empty layers.** Neither `further_extr` nor
`dump_parts_psd` has an area or alpha threshold anywhere in the path, so every tag
the model emitted lands in the PSD — including fully transparent ones for the
tail, wings and eyewear a character does not have. [`ocs/cleanup.py`](ocs/cleanup.py)
scores each layer and splits the verdict in two: things that cannot be anything
but garbage are dropped automatically, and judgement calls are flagged with a
reason for you to resolve. Both directions are reversible in the UI.

Thresholds are calibrated against real output, not guessed. see-through's own
sample produces a **20 px `nose`** and a **37 px `mouth`** — genuinely small,
genuinely real. Those get flagged, never deleted.

**2. The taxonomy has no skin tag, and two tag names are actively misleading.**
The 23 tags are clothing, hair and face. Measured on the sample, `handwear-l` is a
116×463 vertical strip and `legwear` is 310×547 spanning both legs: these are the
whole **arm** and **leg**, bare skin included, not gloves and socks. And
see-through only left/right-splits `handwear`, `ears` and the four eye tags —
`legwear` and `footwear` arrive as one layer covering both sides. Where it does
split, `part_lr_split` keeps only the two largest connected components and
silently discards the rest.

So [`ocs/limbs.py`](ocs/limbs.py) derives the separation from geometry it controls:
every pixel goes to the bone segment it is nearest, the way a Blender bone envelope
works. That produces left and right limbs even for a single-blob silhouette, a
symmetric pose, or arms crossing the torso — and it keeps every component.

Separation is **per side, not per joint**. Cutting at the elbow and knee as well
seemed reasonable and was wrong: each piece then follows one bone, so the moment
anything moves the cut lines tear and ghost. On one character the skirt became
seven pieces overlapping themselves across 31185 px, and the seams were plainly
visible in motion. A weighted mesh does not need cutting — the weights blend the
bend — so limbs and garments stay whole and are weighted across the joints they
span. `RigSettings.slice_limb_spanning` still enables the old behaviour.

**3. Region attachments cannot bend.** [`ocs/rig.py`](ocs/rig.py) builds
triangulated **mesh attachments with bone weights** for every deformable part, so
the elbow bends the pixels instead of pivoting a rigid quad. Small rigid features
(irides, nose, mouth) stay as regions, where meshing would only add vertices.

Two findings that only showed up by running it:

- Nearest-segment assignment alone is wrong. With the arms hanging at the sides,
  the forearm bone is genuinely closer to the upper thigh than the thigh's own
  bone is, so `legwear` was being cut into `arm_r_lower` and the shorts were
  weighted to the hands. `taxonomy.TAG_ALLOWED_REGIONS` constrains which regions
  each tag may occupy, and the same table constrains the weighting.
- "Left and right are mirrored, so flip the sign" is false. A bone's frame is its
  own aim direction, and on a standing figure both legs point *down* rather than
  outward, so they share a frame. Flipping on top of an alternating walk phase
  cancelled it exactly and both legs swung forward together.

---

## Requirements

- Windows (paths and helper scripts are PowerShell; the Python is portable)
- An NVIDIA GPU. 16 GB is comfortable; 8 GB works with the low-VRAM paths.
- [uv](https://github.com/astral-sh/uv), Python 3.12, git
- ~20 GB free for model weights

Verified on: Windows 11, RTX 5070 Ti (16 GB, sm_120), CUDA 12.8, Python 3.12.13,
torch 2.8.0+cu128. Blackwell cards need cu128 wheels — earlier builds have no
sm_120 kernels.

## Setup

```powershell
git clone --recurse-submodules https://github.com/sleeeppy/OCS.git
cd OCS
./scripts/setup_env.ps1
```

One virtualenv holds both OCS and see-through. see-through's `requirements.txt`
already pins the heavy half of what OCS needs (numpy, opencv, pillow, scipy,
scikit-image, psd-tools), so a second environment would only invite version
conflicts. GPU work runs as a subprocess of the same interpreter.

Optional, for previews that work with no network:

```powershell
./scripts/fetch_spine_player.ps1
```

The Spine Web Player is **not** committed — its licence restricts redistribution
and requires your own Spine licence. Without this step previews load the runtime
from unpkg instead.

## Run

```powershell
./scripts/run_ocs.ps1
```

Then open <http://127.0.0.1:8765/>. Drop in an illustration and work through the
four steps.

> **Use a transparent-background PNG.** OCS falls back to estimating the outline
> from the background colour for flat RGB input, and warns you when it does. The
> silhouette is what the limb split is cut from, so a real cutout is the single
> biggest quality lever.

The first run downloads ~12 GB of model weights. Later runs take a couple of
minutes per image.

### The four steps

1. **Upload** — resolution, steps and seed are under *고급 설정*. `group offload`
   is on by default: the plain path peaks at 12–16 GB at 1280, right at the edge
   of a 16 GB card, while offload brings it to ~10 GB.
2. **Layer cleanup** — thumbnails with the reason each layer was dropped or
   flagged. Untick to exclude; auto-dropped layers can be brought back.
3. **Bone placement** — drag the joints. `Shift`+drag moves the subtree, `X`
   mirrors left↔right, `S` snaps symmetric pairs, `0` fits the view. Toggle the
   partition overlay to see exactly which pixels each bone will claim.
4. **Rig & preview** — mesh generation, atlas packing, `skeleton.json`, and a
   preview you can download and open anywhere.

## Output

```
workspace/projects/<id>/export/
  skeleton.json     Spine 4.2 — weighted meshes, 5 preset animations
  skeleton.atlas    libgdx atlas
  skeleton.png      packed texture page
  preview.html      standalone, everything inlined as data URIs
```

Animation: `idle` — a 3.6 s resting breath, and the only preset exported by
default. It is the one animation that suits any pose; the others in
`spine_export.PRESETS` (`walk`, `wave`, `jump`, `turn_head`) assume a standing
figure, and a seated character swinging a leg through a floor-length skirt looks
broken however well the mesh deforms. Pass `animations=[...]` to `export_skeleton`
to get them.

The idle avoids the three things that make an idle read as mechanical: everything
keys on the same frames, the loop is short, and the whole body inflates together.
So the chest leads and the head follows, the neck and head counter-rotate to keep
the head level, hair trails wider and later — and `torso` is never moved at all,
because it is the root of the body and translating it lifts a skirt off the floor
with it (measured: 3 px of chest rise moved the bottom edge of a seated figure by
3 px).

## Bone layout

```
root
└─ torso
   ├─ neck ── head ── eyes
   ├─ rightArm ── rightElbow ── (rightHand)
   ├─ leftArm  ── leftElbow  ── (leftHand)
   ├─ rightLeg ── rightKnee  ── (rightFoot)
   └─ leftLeg  ── leftKnee   ── (leftFoot)
```

`{side}Arm` is the shoulder and `{side}Leg` the hip. Bracketed bones are optional,
as are `hairBack`, `tail` and `wings` — added only when a matching layer exists, or
by hand in the editor.

**`right` means the character's right, i.e. the viewer's left.** This matches
see-through's own convention in `label_lr_split`, where the lower-centroid-x
component is tagged `-r`. Getting it backwards mirrors every animation.

## Without a GPU run

Already have see-through output, or want to iterate on cleanup, partitioning or
rigging without paying for inference each time:

```powershell
.venv/Scripts/python.exe scripts/import_psd.py workspace/seethrough/foo.psd
.venv/Scripts/python.exe scripts/render_debug.py
```

`render_debug.py` writes a sheet showing the source, the silhouette, the bones and
the bone partition — handy for judging a rig at a glance or filing a bug that
shows what OCS actually decided.

## Layout

```
ocs/
  taxonomy.py     see-through's 23 tags, the bone template, tag→region/bone rules
  seethrough.py   subprocess driver for the submodule
  psd_io.py       PSD + sidecars → Part objects
  silhouette.py   character outline (harder than it sounds — see the module docstring)
  cleanup.py      requirement 2: dummy-layer detection, two tiers
  limbs.py        requirement 2-2: bone-driven partition, L/R guaranteed
  skeleton.py     requirement 1: initial bone guess, geodesic not proportional
  rig.py          requirement 3: meshes, weights, bone frames
  spine_export.py Spine 4.2 writer + animation presets + validator
  atlas.py        shelf packer + .atlas writer
  player.py       requirement 4: standalone preview
  pipeline.py     project state machine
  server.py       FastAPI app, SSE progress
web/              editor UI — plain HTML/CSS/JS, no build step
tests/            pytest; fixtures modelled on real see-through output
```

## Tests

```powershell
.venv/Scripts/python.exe -m pytest tests -q
```

The interesting cases are the hard ones: a single connected silhouette in a single
layer (where only the skeleton can tell left from right), a layer covering both
legs, weights that must not leak between an arm and a leg, and that no NaN ever
reaches the JSON — bare `NaN` is not valid JSON and silently kills the browser's
state feed.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
