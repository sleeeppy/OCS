"""FastAPI app: serves the editor and drives the pipeline.

The see-through pass takes minutes and holds the GPU, so it runs on a background
thread with a single-slot lock and the browser follows along over SSE. Everything
else is fast enough to answer inline.
"""

from __future__ import annotations

import asyncio
import base64
import json
import queue
import threading
import time
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from . import pipeline, seethrough, taxonomy
from .config import PROJECTS_DIR, WEB_DIR, ensure_dirs

app = FastAPI(title="OCS (OneClickSpine)", version="0.1.0")

#: One GPU, one job. A second request is rejected rather than queued so the user
#: gets an immediate answer instead of an unexplained wait.
_gpu_lock = threading.Lock()
_running: dict[str, str] = {}

#: project id -> subscriber queues, for SSE progress.
_subscribers: dict[str, list[queue.Queue]] = {}
_sub_lock = threading.Lock()


def _publish(state: dict) -> None:
    pid = state.get("id")
    if not pid:
        return
    # allow_nan=False: bare NaN is not valid JSON and JSON.parse rejects the
    # entire payload, so the browser would stop receiving updates with no visible
    # cause. Better to raise here than to ship an unparseable frame.
    payload = json.dumps(state, ensure_ascii=False, default=str, allow_nan=False)
    with _sub_lock:
        for q in list(_subscribers.get(pid, [])):
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass


def _project(project_id: str) -> pipeline.Project:
    try:
        return pipeline.Project.load(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# --------------------------------------------------------------------------
# meta
# --------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "environment": seethrough.check_environment(),
        "taxonomy": {
            "part_tags": list(taxonomy.PART_TAGS),
            "upstream_lr_tags": list(taxonomy.UPSTREAM_LR_TAGS),
            "ocs_lr_tags": list(taxonomy.OCS_LR_TAGS),
            "limb_spanning_tags": list(taxonomy.LIMB_SPANNING_TAGS),
            "regions": [s.name for s in taxonomy.SKIN_REGIONS],
            "mandatory_limb_regions": list(taxonomy.MANDATORY_LIMB_REGIONS),
        },
        "bones": [
            {"name": b.name, "parent": b.parent, "optional": b.optional}
            for b in taxonomy.BONE_TEMPLATE
        ],
        "mirror_pairs": [list(p) for p in taxonomy.MIRROR_PAIRS],
    }


@app.get("/api/projects")
def projects() -> list[dict]:
    return pipeline.list_projects()


# --------------------------------------------------------------------------
# stage 1
# --------------------------------------------------------------------------


@app.post("/api/projects")
async def create(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    settings: str = Form("{}"),
    autostart: bool = Form(True),
) -> JSONResponse:
    try:
        parsed = json.loads(settings or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="settings is not valid JSON") from None

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        project = pipeline.create_project(data, file.filename or "upload.png", parsed)
    except Exception as exc:                                # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"cannot read image: {exc}") from exc

    if autostart:
        background.add_task(_separate_worker, project.id)
    return JSONResponse(project.state, status_code=201)


def _separate_worker(project_id: str) -> None:
    if not _gpu_lock.acquire(blocking=False):
        busy = ", ".join(_running) or "another project"
        project = pipeline.Project.load(project_id)
        project.state["error"] = f"GPU busy with {busy}; retry when it finishes"
        project.set_stage(pipeline.STAGE_FAILED, "GPU busy")
        _publish(project.state)
        return
    _running[project_id] = time.strftime("%H:%M:%S")
    try:
        project = pipeline.Project.load(project_id)
        pipeline.run_separation(project, on_progress=_publish)
    except Exception:                                       # noqa: BLE001
        pass  # run_separation already recorded the failure in state.json
    finally:
        _running.pop(project_id, None)
        _gpu_lock.release()


@app.post("/api/projects/{project_id}/separate")
def separate(project_id: str, background: BackgroundTasks) -> dict:
    project = _project(project_id)
    if project.state.get("stage") == pipeline.STAGE_SEPARATING:
        raise HTTPException(status_code=409, detail="already separating")
    background.add_task(_separate_worker, project_id)
    project.set_stage(pipeline.STAGE_SEPARATING, "queued", 0.0)
    return project.state


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    return _project(project_id).state


@app.get("/api/projects/{project_id}/events")
async def events(project_id: str) -> StreamingResponse:
    _project(project_id)  # 404 early
    q: queue.Queue = queue.Queue(maxsize=64)
    with _sub_lock:
        _subscribers.setdefault(project_id, []).append(q)

    async def stream():
        try:
            state = pipeline.Project.load(project_id).state
            yield (
                "data: "
                + json.dumps(state, ensure_ascii=False, default=str, allow_nan=False)
                + "\n\n"
            )
            while True:
                try:
                    payload = await asyncio.get_running_loop().run_in_executor(
                        None, q.get, True, 20.0
                    )
                    yield f"data: {payload}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with _sub_lock:
                subs = _subscribers.get(project_id, [])
                if q in subs:
                    subs.remove(q)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


# --------------------------------------------------------------------------
# stage 2: review + bones
# --------------------------------------------------------------------------


@app.get("/api/projects/{project_id}/layers")
def layers(project_id: str) -> dict:
    project = _project(project_id)
    return {
        "canvas": project.state.get("canvas"),
        "summary": project.state.get("cleanup_summary"),
        "layers": project.state.get("layers", []),
        "exclusions": project.state.get("exclusions", []),
        "revived": project.state.get("revived", []),
        "silhouette": project.state.get("silhouette"),
        "warnings": project.state.get("warnings", []),
    }


@app.patch("/api/projects/{project_id}/layers")
async def patch_layers(project_id: str, body: dict) -> dict:
    project = _project(project_id)
    pipeline.set_exclusions(
        project,
        list(body.get("excluded") or []),
        list(body.get("revived") or []),
    )
    return {
        "exclusions": project.state["exclusions"],
        "revived": project.state["revived"],
    }


@app.get("/api/projects/{project_id}/rig")
def get_rig(project_id: str) -> dict:
    project = _project(project_id)
    if not project.rig_path.exists():
        raise HTTPException(status_code=409, detail="no rig yet; run separation first")
    return json.loads(project.rig_path.read_text(encoding="utf-8"))


@app.put("/api/projects/{project_id}/rig")
async def put_rig(project_id: str, body: dict) -> dict:
    project = _project(project_id)
    if "bones" not in body or "canvas" not in body:
        raise HTTPException(status_code=400, detail="expected {canvas, bones}")
    pipeline.save_rig(project, body)
    return json.loads(project.rig_path.read_text(encoding="utf-8"))


@app.post("/api/projects/{project_id}/partition-preview")
def partition_preview(project_id: str) -> dict:
    """Recompute the bone partition so the editor can show the effect of a drag."""
    project = _project(project_id)
    if not project.rig_path.exists():
        raise HTTPException(status_code=409, detail="no rig yet")
    try:
        return pipeline.preview_partition(project)
    except Exception as exc:                                # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


# --------------------------------------------------------------------------
# stage 3: export
# --------------------------------------------------------------------------


@app.post("/api/projects/{project_id}/export")
def export(project_id: str) -> dict:
    project = _project(project_id)
    if not project.rig_path.exists():
        raise HTTPException(status_code=409, detail="no rig yet; run separation first")
    try:
        return pipeline.run_export(project, on_progress=_publish)
    except Exception as exc:                                # noqa: BLE001
        project.state["error"] = f"{type(exc).__name__}: {exc}"
        project.set_stage(pipeline.STAGE_FAILED, "export failed")
        _publish(project.state)
        raise HTTPException(status_code=500, detail=project.state["error"]) from exc


@app.get("/api/projects/{project_id}/preview")
def preview(project_id: str) -> FileResponse:
    project = _project(project_id)
    path = project.export_dir / "preview.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="not exported yet")
    return FileResponse(path, media_type="text/html")


@app.get("/api/projects/{project_id}/download/{kind}")
def download(project_id: str, kind: str) -> FileResponse:
    project = _project(project_id)
    names = {
        "skeleton": ("skeleton.json", "application/json"),
        "atlas": ("skeleton.atlas", "text/plain"),
        "png": ("skeleton.png", "image/png"),
        "preview": ("preview.html", "text/html"),
    }
    if kind not in names:
        raise HTTPException(status_code=404, detail=f"unknown artifact '{kind}'")
    filename, mime = names[kind]
    path = project.export_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="not exported yet")
    return FileResponse(path, media_type=mime, filename=f"{project.state.get('name')}-{filename}")


# --------------------------------------------------------------------------
# static
# --------------------------------------------------------------------------

ensure_dirs()
app.mount("/files", StaticFiles(directory=str(PROJECTS_DIR)), name="files")
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


#: A 1x1 transparent GIF. Browsers request /favicon.ico unprompted, and a 404 in
#: the log on every page load is noise that hides real problems.
_FAVICON = base64.b64decode(
    b"R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(_FAVICON, media_type="image/gif",
                    headers={"Cache-Control": "public, max-age=86400"})


# HEAD as well as GET: readiness probes (including the one that launches this
# server) send HEAD, and a GET-only route answers 405, which reads like a
# failure in the startup log.
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def index() -> HTMLResponse:
    path = Path(WEB_DIR) / "index.html"
    if not path.exists():
        return HTMLResponse("<h1>OCS</h1><p>web/index.html is missing.</p>", status_code=500)
    return HTMLResponse(path.read_text(encoding="utf-8"))


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")


if __name__ == "__main__":
    main()
