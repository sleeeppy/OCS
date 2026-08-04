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

- [uv](https://github.com/astral-sh/uv), Python 3.12, git
- For the layer-decomposition stage: an NVIDIA GPU. 16 GB is comfortable; 8 GB
  works with the low-VRAM paths. ~20 GB free for model weights.

Verified on: Windows 11, RTX 5070 Ti (16 GB, sm_120), CUDA 12.8, Python 3.12.13,
torch 2.8.0+cu128. Blackwell cards need cu128 wheels — earlier builds have no
sm_120 kernels.

Also verified on macOS 15 / Apple M5 Pro (24 GB) with **no GPU stage** — see
[Without a GPU](#without-a-gpu) below. Helper scripts come in both flavours:
`.ps1` for Windows, `.sh` for macOS and Linux. The Python is portable either way.

## Setup

```powershell
git clone --recurse-submodules https://github.com/sleeeppy/OCS.git
cd OCS
./scripts/setup_env.ps1
```

macOS / Linux:

```bash
./scripts/setup_env.sh              # OCS only, no GPU stage — under a minute
./scripts/setup_env.sh --with-gpu   # + torch and see-through's requirements
```

The two setup scripts differ in one deliberate way. `setup_env.ps1` exits
non-zero when `torch.cuda.is_available()` is false, because on Windows that means
a broken install. `setup_env.sh` only reports it: the GPU is needed for exactly
one stage, and failing there would block a stack that is otherwise fully usable.

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
./scripts/run_ocs.ps1     # Windows
```

```bash
./scripts/run_ocs.sh      # macOS / Linux
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

Animations: `idle`, `walk`, `wave`, `jump`, `turn_head`.

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

## Without a GPU

Decomposition is the only stage that needs one. Cleanup, silhouette, bone
placement, limb partition, meshing, weighting, atlas packing, Spine export and
the preview are numpy/opencv/scipy and run anywhere — so on a machine with no
NVIDIA card the editor still works end to end, it just needs its layers from
somewhere else.

Already have see-through output, or want to iterate on the later stages without
paying for inference each time:

```bash
.venv/bin/python scripts/import_psd.py workspace/seethrough/foo.psd
```

No GPU anywhere and no PSD to import — there is nothing for the editor to open
past the upload step. Build a project from a synthetic decomposition instead:

```bash
.venv/bin/python scripts/make_demo_project.py --all
```

The layouts in `ocs/demo.py` are measured from see-through's own sample, not
idealised: `handwear` is a whole arm, `legwear` is both legs in one layer, there
is no skin tag, and the empty/duplicate/speck layers are present. `--figure blob`
is the interesting one — a single connected silhouette in a single layer with
arms fused to the torso and no left/right suffixes anywhere, so only the bone
skeleton can separate the sides. `tests/conftest.py` wraps the same builders.

```bash
.venv/bin/python scripts/render_debug.py --project <id>
```

`render_debug.py` writes a sheet showing the source, the silhouette, the bones and
the bone partition — handy for judging a rig at a glance or filing a bug that
shows what OCS actually decided. For a project it draws the rig in effect,
including your edits.

### Known limitation

`skeleton.guess_rig`'s geodesic fallback misplaces the leg chain on the `blob`
figure: it reads the fused torso block's lower corners as extremities and puts
`{side}Knee` and `{side}Foot` at the bottom of the torso instead of down the
legs, leaving the legs unclaimed by any joint. The partition still verifies 4:4
because `Rig.segment` extrapolates open chain ends far enough to sweep them — so
`verify_limb_separation` passing is not on its own evidence that the joints are
where they belong. Drag them in step 3, or start from a decomposition that has
per-limb layers. Only the single-blob input is affected; a real see-through PSD
places these from layers (`source: "layer"`), which is correct.

## Layout

```
ocs/
  taxonomy.py     see-through's 23 tags, the bone template, tag→region/bone rules
  seethrough.py   subprocess driver for the submodule
  demo.py         synthetic decompositions for the no-GPU path and the tests
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

```bash
.venv/bin/python -m pytest tests -q          # macOS / Linux
.venv/Scripts/python.exe -m pytest tests -q  # Windows
```

The interesting cases are the hard ones: a single connected silhouette in a single
layer (where only the skeleton can tell left from right), a layer covering both
legs, weights that must not leak between an arm and a leg, and that no NaN ever
reaches the JSON — bare `NaN` is not valid JSON and silently kills the browser's
state feed.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
