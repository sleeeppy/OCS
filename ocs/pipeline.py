"""Project state machine tying the stages together.

    upload -> separate -> [review] -> [place bones] -> partition -> rig -> export

``review`` and ``place bones`` are the two human steps (requirements 2-1 and 1).
Everything is persisted under ``workspace/projects/<id>/`` after each stage, so a
project survives a server restart and the expensive see-through pass never has to
be repeated because of a later mistake.

    input.png              the upload, unmodified
    state.json             stage, settings, reports, warnings
    seethrough/            see-through's own output (PSD + sidecars + src_img)
    layers/<slug>.png      one PNG per decomposed layer, for the review panel
    layers/composite.png   square-padded source, the bone editor's backdrop
    rig.json               joint positions (guessed, then user-edited)
    parts/<slug>.png       final parts after cleanup + limb partition
    export/               skeleton.json, skeleton.atlas, skeleton.png, preview.html
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import atlas as atlas_mod
from . import cleanup, limbs, player, psd_io, rig as rig_mod, seethrough, silhouette, skeleton
from . import spine_export, taxonomy
from .config import PROJECTS_DIR, OcsSettings

STAGE_CREATED = "created"
STAGE_SEPARATING = "separating"
STAGE_REVIEW = "review"
STAGE_EXPORTING = "exporting"
STAGE_DONE = "done"
STAGE_FAILED = "failed"


@dataclass
class Project:
    id: str
    root: Path
    state: dict = field(default_factory=dict)

    # ---- paths ----------------------------------------------------------
    @property
    def input_path(self) -> Path:
        return self.root / "input.png"

    @property
    def seethrough_dir(self) -> Path:
        return self.root / "seethrough"

    @property
    def layers_dir(self) -> Path:
        return self.root / "layers"

    @property
    def parts_dir(self) -> Path:
        return self.root / "parts"

    @property
    def export_dir(self) -> Path:
        return self.root / "export"

    @property
    def rig_path(self) -> Path:
        return self.root / "rig.json"

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def log_path(self) -> Path:
        return self.root / "seethrough.log"

    # ---- persistence ----------------------------------------------------
    def save(self) -> None:
        self.state["updated_at"] = time.time()
        # allow_nan=False on purpose. Python happily writes bare ``NaN``/``Infinity``,
        # which are not valid JSON: JSON.parse rejects the whole document, so a
        # single stray NaN in a layer metric silently kills the browser's state
        # feed. Fail here, where the traceback points at the culprit.
        self.state_path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, project_id: str) -> "Project":
        root = PROJECTS_DIR / project_id
        if not (root / "state.json").exists():
            raise FileNotFoundError(f"no such project: {project_id}")
        state = json.loads((root / "state.json").read_text(encoding="utf-8"))
        return cls(id=project_id, root=root, state=state)

    def settings(self) -> OcsSettings:
        s = OcsSettings()
        stored = self.state.get("settings") or {}
        for key, obj in (("seethrough", s.seethrough), ("cleanup", s.cleanup),
                         ("rig", s.rig), ("atlas", s.atlas)):
            for k, v in (stored.get(key) or {}).items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
        return s

    def set_stage(self, stage: str, message: str = "", progress: float | None = None) -> None:
        self.state["stage"] = stage
        self.state["message"] = message
        if progress is not None:
            self.state["progress"] = round(progress, 4)
        self.save()


# --------------------------------------------------------------------------


def list_projects() -> list[dict]:
    out = []
    if not PROJECTS_DIR.exists():
        return out
    for d in sorted(PROJECTS_DIR.iterdir(), reverse=True):
        f = d / "state.json"
        if not f.is_file():
            continue
        try:
            st = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out.append({
            "id": d.name,
            "stage": st.get("stage"),
            "name": st.get("name"),
            "created_at": st.get("created_at"),
            "updated_at": st.get("updated_at"),
        })
    return out


def create_project(image_bytes: bytes, filename: str, settings: dict | None = None) -> Project:
    """Store the upload as RGBA PNG and open a project around it."""
    pid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    root = PROJECTS_DIR / pid
    root.mkdir(parents=True, exist_ok=True)

    import io
    img = Image.open(io.BytesIO(image_bytes))
    had_alpha = img.mode in ("RGBA", "LA") or "transparency" in img.info
    img = img.convert("RGBA")
    img.save(root / "input.png")

    project = Project(id=pid, root=root, state={
        "id": pid,
        "name": Path(filename).stem or "untitled",
        "created_at": time.time(),
        "stage": STAGE_CREATED,
        "progress": 0.0,
        "message": "uploaded",
        "settings": settings or {},
        "input": {
            "filename": filename,
            "width": img.width, "height": img.height,
            "had_alpha": bool(had_alpha),
        },
        "warnings": ([] if had_alpha else [
            "input_has_no_alpha: the character outline will have to be estimated "
            "from the background. A transparent PNG gives a cleaner rig."
        ]),
    })
    project.save()
    return project


# --------------------------------------------------------------------------
# stage 1: separation + cleanup + initial skeleton
# --------------------------------------------------------------------------


def _write_layer_images(decomp: psd_io.Decomposition, out_dir: Path) -> dict[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    naming = taxonomy.PartNaming()
    info: dict[str, dict] = {}
    for part in decomp.parts:
        slug = naming.unique_slug(part.name)
        if part.rgba.shape[0] > 1 and part.rgba.shape[1] > 1:
            Image.fromarray(part.rgba).save(out_dir / f"{slug}.png")
        info[part.name] = {
            "slug": slug,
            "offset": list(part.offset),
            "size": list(part.size),
        }
    if decomp.src_img is not None:
        Image.fromarray(decomp.src_img).save(out_dir / "composite.png")
    return info


def run_separation(project: Project, on_progress=None) -> Project:
    """see-through -> cleanup report -> initial rig. The slow stage."""
    s = project.settings()

    def progress(phase: str, frac: float | None) -> None:
        project.set_stage(STAGE_SEPARATING, f"see-through: {phase}",
                          None if frac is None else 0.05 + 0.75 * frac)
        if on_progress:
            on_progress(project.state)

    project.set_stage(STAGE_SEPARATING, "starting", 0.02)
    if on_progress:
        on_progress(project.state)

    try:
        psd = seethrough.run_inference(
            project.input_path, project.seethrough_dir, s.seethrough,
            on_progress=progress, log_path=project.log_path,
        )
    except Exception as exc:                                # noqa: BLE001
        project.state["error"] = f"{type(exc).__name__}: {exc}"
        project.set_stage(STAGE_FAILED, "see-through failed")
        if on_progress:
            on_progress(project.state)
        raise

    project.set_stage(STAGE_SEPARATING, "reading layers", 0.82)
    if on_progress:
        on_progress(project.state)

    decomp = psd_io.read_decomposition(psd)
    layer_info = _write_layer_images(decomp, project.layers_dir)

    project.set_stage(STAGE_SEPARATING, "scoring layers", 0.88)
    reports = cleanup.analyze(decomp, s.cleanup)
    kept, dropped = cleanup.apply_verdicts(decomp, reports)

    sil = silhouette.character_mask(decomp, kept)

    project.set_stage(STAGE_SEPARATING, "placing bones", 0.94)
    if on_progress:
        on_progress(project.state)
    guess = skeleton.guess_rig(decomp, kept)
    project.rig_path.write_text(
        json.dumps(guess.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    project.state["psd"] = str(psd.relative_to(project.root))
    project.state["canvas"] = {"width": decomp.canvas[0], "height": decomp.canvas[1]}
    project.state["layers"] = [
        {**r.to_dict(), **layer_info.get(r.name, {})} for r in reports
    ]
    project.state["cleanup_summary"] = cleanup.summarize(reports)
    project.state["auto_dropped"] = dropped
    project.state["exclusions"] = []
    project.state["revived"] = []
    project.state["silhouette"] = sil.to_dict()
    project.state["warnings"] = list(dict.fromkeys(
        project.state.get("warnings", []) + sil.warnings
    ))
    project.set_stage(STAGE_REVIEW, "ready for review", 1.0)
    if on_progress:
        on_progress(project.state)
    return project


# --------------------------------------------------------------------------
# stage 2: user review + bone editing
# --------------------------------------------------------------------------


def set_exclusions(project: Project, excluded: list[str], revived: list[str]) -> Project:
    """Requirement 2-1: extra layers the user chose to drop, or to bring back."""
    known = {entry["name"] for entry in project.state.get("layers", [])}
    project.state["exclusions"] = sorted(set(excluded) & known)
    project.state["revived"] = sorted(set(revived) & known)
    project.save()
    return project


def save_rig(project: Project, rig_dict: dict) -> Project:
    """Requirement 1: persist the joint positions the user dragged."""
    parsed = skeleton.Rig.from_dict(rig_dict)
    project.rig_path.write_text(
        json.dumps(parsed.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    project.state["rig_saved_at"] = time.time()
    project.save()
    return project


def load_rig(project: Project) -> skeleton.Rig:
    return skeleton.Rig.from_dict(json.loads(project.rig_path.read_text(encoding="utf-8")))


def resolved_parts(project: Project) -> tuple[psd_io.Decomposition, list[psd_io.Part], skeleton.Rig]:
    """Re-derive the surviving parts from the stored PSD + review decisions."""
    s = project.settings()
    psd = project.root / project.state["psd"]
    decomp = psd_io.read_decomposition(psd)
    reports = cleanup.analyze(decomp, s.cleanup)
    kept, _dropped = cleanup.apply_verdicts(
        decomp, reports,
        excluded=set(project.state.get("exclusions", [])),
        revived=set(project.state.get("revived", [])),
    )
    return decomp, kept, load_rig(project)


def preview_partition(project: Project) -> dict:
    """Region map + limb report for the editor's live overlay, no export."""
    decomp, kept, rig = resolved_parts(project)
    sil = silhouette.character_mask(decomp, kept)
    labels, specs = limbs.region_labels(rig, sil.mask)

    # 8-bit indexed PNG: small, and the browser can colour it however it likes.
    out = np.full(labels.shape, 255, dtype=np.uint8)
    out[labels >= 0] = labels[labels >= 0].astype(np.uint8)
    project.layers_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="L").save(project.layers_dir / "regions.png")

    _new_parts, report = limbs.partition(decomp, kept, rig, project.settings().rig)
    return {
        "regions": [s.name for s in specs],
        "report": report,
        "verify": limbs.verify_limb_separation(_new_parts),
        "part_count": len(_new_parts),
    }


# --------------------------------------------------------------------------
# stage 3: partition + rig + export
# --------------------------------------------------------------------------


def run_export(project: Project, on_progress=None) -> dict:
    s = project.settings()

    def step(msg: str, frac: float) -> None:
        project.set_stage(STAGE_EXPORTING, msg, frac)
        if on_progress:
            on_progress(project.state)

    step("resolving layers", 0.05)
    decomp, kept, rig = resolved_parts(project)

    step("separating limbs", 0.20)
    parts, limb_report = limbs.partition(decomp, kept, rig, s.rig)
    verify = limbs.verify_limb_separation(parts)

    step("building meshes and weights", 0.45)
    built = rig_mod.build_rig(decomp, parts, rig, s.rig)

    step("packing atlas", 0.70)
    if project.parts_dir.exists():
        shutil.rmtree(project.parts_dir)
    psd_io.write_parts_png(parts, project.parts_dir)
    packed = atlas_mod.pack(built.part_images, s.atlas)
    png_path, atlas_path = packed.write(project.export_dir, "skeleton")

    step("writing skeleton.json", 0.85)
    json_path = spine_export.export_skeleton(
        built, project.export_dir / "skeleton.json", name=project.state.get("name", "character")
    )
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    problems = spine_export.validate(doc)

    packed_names = {r.name for r in packed.regions}
    missing = sorted(n for n in built.attachments if n not in packed_names)
    if missing:
        problems.append(f"atlas is missing {len(missing)} region(s): {missing[:5]}")

    step("building preview", 0.94)
    preview_path, embedded = player.build_preview(
        json_path, atlas_path, png_path,
        project.export_dir / "preview.html",
        title=f"OCS - {project.state.get('name', 'character')}",
    )

    result = {
        "parts": len(parts),
        "bones": len(built.bones),
        "slots": len(built.slots),
        "meshes": sum(1 for a in built.attachments.values() if a.kind == "mesh"),
        "regions": sum(1 for a in built.attachments.values() if a.kind == "region"),
        "atlas_size": list(packed.size),
        "animations": sorted(doc.get("animations", {})),
        "limb_report": limb_report,
        "verify": verify,
        "validation": problems,
        "rig_warnings": built.warnings,
        "runtime_embedded": embedded,
        "files": {
            "skeleton_json": str(json_path.relative_to(project.root)),
            "atlas": str(atlas_path.relative_to(project.root)),
            "atlas_png": str(png_path.relative_to(project.root)),
            "preview": str(preview_path.relative_to(project.root)),
        },
    }
    project.state["export"] = result
    project.set_stage(STAGE_DONE, "export complete", 1.0)
    if on_progress:
        on_progress(project.state)
    return result
