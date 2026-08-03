# NOTICE

OCS (OneClickSpine)
Copyright 2026 sleeeppy
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

---

## Upstream projects

### see-through — bundled as a git submodule

- Source: <https://github.com/shitagaki-lab/see-through>
- License: Apache License 2.0
- Location in this repo: `external/see-through/` (git submodule, not vendored source)

OCS **invokes** see-through as a subprocess to perform layer decomposition. No
see-through source is copied into this repository; the submodule keeps its own
`LICENSE` file. OCS reads see-through's documented output artifacts:

- `workspace/layerdiff_output/<name>.psd` and `<name>_depth.psd`
- the `<name>.psd.json` sidecar (`parts[tag].xyxy`, `depth_median`, `frame_size`)
- `workspace/layerdiff_output/<name>/src_img.png` (the square-padded input)

### spine-animation-ai — design reference only, **no code reused**

- Source: <https://github.com/GenielabsOpenSource/spine-animation-ai>
- License: PolyForm Noncommercial 1.0.0 (**noncommercial use only**)

OCS does **not** copy, vendor, link, or execute any part of spine-animation-ai.
It was read as prior art while designing `ocs/rig.py`, `ocs/spine_export.py`,
`ocs/atlas.py` and `ocs/player.py`, all of which are independent
implementations written for OCS.

This was a deliberate choice. Reusing that project's code would have forced
OCS — a public repository — to inherit the PolyForm Noncommercial license, and
its rigging output is limited to Spine **region attachments** (rigid quads).
OCS needs weighted **mesh attachments** for limb deformation, a trimmed atlas
packer, and a fully offline HTML player, none of which that code provides, so
independent implementation was both the better technical result and the
license-clean one.

### Spine runtime — fetched locally, **not redistributed**

- Esoteric Software Spine Web Player (`@esotericsoftware/spine-player`).
- Licensed under the **Spine Runtimes License Agreement**:
  <http://esotericsoftware.com/spine-runtimes-license>, which restricts
  redistribution and requires the user to hold a valid Spine license.

Because of that, OCS does **not** commit the runtime. `web/vendor/spine-player.js`
and `.css` are gitignored and fetched on demand by
`scripts/fetch_spine_player.ps1`. Exported previews inline whatever is present
locally so the HTML works offline; if the runtime was never fetched, the export
falls back to loading it from unpkg at view time. Either way, using it requires
your own Spine license from Esoteric Software.

The Spine JSON and `.atlas` file formats are documented interchange formats;
OCS writes them from their public specification.
