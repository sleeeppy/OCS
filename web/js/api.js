/* Thin fetch wrapper + SSE helper. No framework, no build step. */
"use strict";

const API = (() => {
  async function req(method, url, body, isForm) {
    const opts = { method, headers: {} };
    if (isForm) {
      opts.body = body;
    } else if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(url, opts);
    const text = await res.text();
    let data = null;
    if (text) {
      try { data = JSON.parse(text); } catch { data = { detail: text }; }
    }
    if (!res.ok) {
      const msg = (data && (data.detail || data.message)) || `${res.status} ${res.statusText}`;
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return data;
  }

  return {
    health: () => req("GET", "/api/health"),
    listProjects: () => req("GET", "/api/projects"),
    getProject: (id) => req("GET", `/api/projects/${id}`),

    createProject(file, settings) {
      const fd = new FormData();
      fd.append("file", file, file.name);
      fd.append("settings", JSON.stringify(settings || {}));
      fd.append("autostart", "true");
      return req("POST", "/api/projects", fd, true);
    },
    separate: (id) => req("POST", `/api/projects/${id}/separate`),

    layers: (id) => req("GET", `/api/projects/${id}/layers`),
    patchLayers: (id, excluded, revived) =>
      req("PATCH", `/api/projects/${id}/layers`, { excluded, revived }),

    getRig: (id) => req("GET", `/api/projects/${id}/rig`),
    putRig: (id, rig) => req("PUT", `/api/projects/${id}/rig`, rig),
    partitionPreview: (id) => req("POST", `/api/projects/${id}/partition-preview`),

    exportRig: (id) => req("POST", `/api/projects/${id}/export`),

    fileUrl: (id, rel) => `/files/${id}/${rel.split("\\").join("/")}`,
    previewUrl: (id) => `/api/projects/${id}/preview`,
    downloadUrl: (id, kind) => `/api/projects/${id}/download/${kind}`,

    /** Subscribe to a project's progress. Returns a close() function. */
    events(id, onState) {
      const src = new EventSource(`/api/projects/${id}/events`);
      src.onmessage = (ev) => {
        if (!ev.data) return;
        let state;
        try {
          state = JSON.parse(ev.data);
        } catch (err) {
          console.error("[ocs] unparseable SSE payload", err, ev.data.slice(0, 200));
          return;
        }
        // Report handler failures instead of swallowing them. onState is async,
        // so a bare try/catch here would miss rejections entirely and the UI
        // would just silently never advance.
        try {
          Promise.resolve(onState(state)).catch((err) => {
            console.error("[ocs] state handler failed", err);
          });
        } catch (err) {
          console.error("[ocs] state handler threw", err);
        }
      };
      // The browser retries automatically; a transient drop is not worth surfacing.
      src.onerror = () => {};
      return () => src.close();
    },
  };
})();
