/* Wires the four steps together and owns the project state. */
"use strict";

const Toast = (() => {
  function push(text, cls) {
    const host = document.getElementById("toast");
    const n = document.createElement("div");
    n.className = `toast-item ${cls || ""}`.trim();
    n.textContent = text;
    host.appendChild(n);
    setTimeout(() => n.remove(), cls === "err" ? 7000 : 3200);
  }
  return {
    info: (t) => push(t, ""),
    ok: (t) => push(t, "ok"),
    err: (t) => push(t, "err"),
  };
})();

const App = (() => {
  let projectId = null;
  let stage = null;
  let closeEvents = null;
  let saveTimer = null;
  let meta = null;

  const $ = (id) => document.getElementById(id);

  /* ── step navigation ────────────────────────────────────────── */

  const REACHABLE = { upload: () => true, review: () => !!projectId,
                      bones: () => !!projectId, export: () => !!projectId };

  function show(step) {
    for (const p of document.querySelectorAll(".panel")) {
      p.classList.toggle("active", p.dataset.panel === step);
    }
    for (const b of document.querySelectorAll(".step")) {
      b.classList.toggle("active", b.dataset.step === step);
    }
    if (step === "bones") BoneEditor.refresh();
  }

  function unlock(steps) {
    for (const b of document.querySelectorAll(".step")) {
      if (steps.includes(b.dataset.step)) b.disabled = false;
    }
  }

  /* ── sidebar ────────────────────────────────────────────────── */

  function renderProgress(state) {
    const card = $("progress-card");
    const active = ["separating", "exporting"].includes(state.stage);
    card.classList.toggle("hidden", !active && state.stage !== "failed");
    $("bar-fill").style.width = `${Math.round((state.progress || 0) * 100)}%`;
    const elapsed = state.elapsed_s
      ? ` · ${Math.floor(state.elapsed_s / 60)}분 ${state.elapsed_s % 60}초 경과`
      : "";
    $("progress-msg").textContent = state.stage === "failed"
      ? translateWarning(state.error || "실패")
      : `${state.message || state.stage}${active ? elapsed : ""}`;
    $("progress-msg").style.color = state.stage === "failed" ? "var(--danger)" : "";
    // A failed separation is almost always retryable, so give it a button rather
    // than leaving the user with a dead progress bar.
    $("btn-retry-separate").classList.toggle(
      "hidden", !(state.stage === "failed" && projectId)
    );
  }

  function renderWarnings(list) {
    const card = $("warnings");
    const box = $("warn-list");
    box.textContent = "";
    const items = list || [];
    card.classList.toggle("hidden", items.length === 0);
    for (const w of items) {
      const li = document.createElement("li");
      li.textContent = translateWarning(w);
      box.appendChild(li);
    }
  }

  function translateWarning(w) {
    if (w.startsWith("input_has_no_alpha") || w.startsWith("no_alpha_channel")) {
      return "입력에 알파 채널이 없어 배경색으로 실루엣을 추정했습니다. 투명 배경 PNG를 넣으면 리깅이 깔끔해집니다.";
    }
    if (w.startsWith("background_not_flat")) {
      return "배경이 단색이 아니라 실루엣 추정이 부정확할 수 있습니다.";
    }
    if (w.startsWith("layer_union_fallback")) {
      return "실루엣을 레이어 합집합으로 대체했습니다. 맨살이 빠질 수 있어 옷에서 팔·다리를 잘라냅니다.";
    }
    if (w.startsWith("background_estimate_degenerate")) {
      return "배경 추정 결과가 비정상이라 레이어 기반으로 대체했습니다.";
    }
    if (w.startsWith("The server restarted while layers were being separated")) {
      return "서버가 재시작되면서 레이어 분리 작업이 유실됐습니다. 다시 실행하면 됩니다.";
    }
    if (w.startsWith("GPU busy")) {
      return "GPU가 다른 프로젝트를 처리 중입니다. 끝난 뒤 다시 시도하세요.";
    }
    return w;
  }

  async function loadEnvironment() {
    try {
      meta = await API.health();
      // Stale code is the single most confusing failure mode in this loop: a fix
      // lands on disk, the running server keeps the old module, and the same bug
      // reappears with no explanation. Say so loudly.
      if ((meta.stale_modules || []).length) {
        Toast.err(`서버가 오래된 코드로 실행 중입니다 (${meta.stale_modules.join(", ")}). `
                + "서버를 재시작하세요.");
      }

      const e = meta.environment || {};
      const body = $("env-body");
      if (e.ok) {
        body.innerHTML = "";
        const lines = [
          `${e.device}`,
          `VRAM ${e.vram_gb} GB · sm_${(e.capability || []).join("")}`,
          `torch ${e.torch}`,
          e.see_through ? "see-through ✓" : "see-through 없음 (submodule 초기화 필요)",
        ];
        for (const l of lines) {
          const d = document.createElement("div");
          d.textContent = l;
          body.appendChild(d);
        }
        body.className = "mono";
      } else {
        body.className = "mono";
        body.style.color = "var(--danger)";
        body.textContent = `CUDA 사용 불가: ${e.error || "unknown"}`;
        $("btn-start").disabled = true;
        $("start-note").textContent = "GPU를 사용할 수 없어 레이어 분리를 실행할 수 없습니다.";
      }
      BoneEditor.init(meta.bones, meta.mirror_pairs, onRigChanged);
    } catch (err) {
      $("env-body").textContent = `서버 응답 없음: ${err.message}`;
    }
  }

  async function loadRecent() {
    try {
      const list = await API.listProjects();
      const card = $("recent");
      const box = $("recent-list");
      box.textContent = "";
      card.classList.toggle("hidden", list.length === 0);
      for (const p of list.slice(0, 8)) {
        const li = document.createElement("li");
        const name = document.createElement("b");
        name.textContent = p.name || p.id;
        const st = document.createElement("span");
        st.className = "muted mono";
        st.textContent = p.stage;
        const btn = document.createElement("button");
        btn.textContent = "열기";
        btn.addEventListener("click", () => attach(p.id));
        li.append(name, st, btn);
        box.appendChild(li);
      }
    } catch { /* first run, no workspace yet */ }
  }

  /* ── step 1 ─────────────────────────────────────────────────── */

  function settingsFromForm() {
    return {
      seethrough: {
        resolution: parseInt($("opt-resolution").value, 10),
        inference_steps: parseInt($("opt-steps").value, 10),
        seed: parseInt($("opt-seed").value, 10),
        group_offload: $("opt-offload").checked,
      },
    };
  }

  function bindUpload() {
    const drop = $("drop");
    const input = $("file-input");
    let file = null;

    function accept(f) {
      if (!f || !f.type.startsWith("image/")) { Toast.err("이미지 파일이 아닙니다"); return; }
      file = f;
      $("drop-preview").src = URL.createObjectURL(f);
      drop.classList.add("has-file");
      $("btn-start").disabled = false;
    }

    input.addEventListener("change", () => accept(input.files[0]));
    for (const ev of ["dragenter", "dragover"]) {
      drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
    }
    for (const ev of ["dragleave", "drop"]) {
      drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
    }
    drop.addEventListener("drop", (e) => accept(e.dataTransfer.files[0]));

    $("btn-start").addEventListener("click", async () => {
      if (!file) return;
      $("btn-start").disabled = true;
      $("start-note").textContent = "업로드 중…";
      try {
        const state = await API.createProject(file, settingsFromForm());
        attach(state.id);
        Toast.info("레이어 분리를 시작했습니다. 첫 실행은 모델 다운로드로 오래 걸립니다.");
      } catch (err) {
        Toast.err(`시작 실패: ${err.message}`);
        $("btn-start").disabled = false;
        $("start-note").textContent = "";
      }
    });
  }

  /* ── project lifecycle ──────────────────────────────────────── */

  function attach(id) {
    if (closeEvents) closeEvents();
    projectId = id;
    stage = null;
    unlock(["review", "bones", "export"]);
    closeEvents = API.events(id, onState);
  }

  async function onState(state) {
    renderProgress(state);
    renderWarnings(state.warnings);
    const prev = stage;
    stage = state.stage;

    if (stage === "failed") {
      Toast.err(state.error || "실패했습니다");
      $("btn-start").disabled = false;
      return;
    }

    if (stage === "separating" && prev !== "separating") show("upload");

    // Entering review is the moment the layer report and the guessed rig exist.
    if (stage === "review" && prev !== "review") {
      await enterReview();
      Toast.ok("레이어 분리 완료");
    }
    if (["review", "exporting", "done"].includes(stage) && prev === null) {
      await enterReview();
    }
    if (stage === "done" && state.export) {
      renderExport(state.export);
    }
  }

  async function enterReview() {
    try {
      const payload = await API.layers(projectId);
      LayerPanel.load(projectId, payload);
      renderWarnings(payload.warnings);
      const rig = await API.getRig(projectId);
      BoneEditor.load(rig, API.fileUrl(projectId, "layers/composite.png"));
      for (const b of document.querySelectorAll('.step[data-step="upload"]')) b.classList.add("done");
      show("review");
    } catch (err) {
      Toast.err(`불러오기 실패: ${err.message}`);
    }
  }

  /* ── step 2 → 3 ─────────────────────────────────────────────── */

  function onSelectionChanged(sel) {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
      try { await API.patchLayers(projectId, sel.excluded, sel.revived); }
      catch (err) { Toast.err(`제외 목록 저장 실패: ${err.message}`); }
    }, 350);
  }

  function onRigChanged(rig) {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
      try { await API.putRig(projectId, rig); }
      catch (err) { Toast.err(`본 저장 실패: ${err.message}`); }
    }, 400);
  }

  async function verifyPartition() {
    const btn = $("btn-verify");
    btn.disabled = true;
    btn.textContent = "계산 중…";
    try {
      await API.putRig(projectId, BoneEditor.getRig());
      const res = await API.partitionPreview(projectId);
      renderVerify(res);
      BoneEditor.setRegions(API.fileUrl(projectId, "layers/regions.png"));
    } catch (err) {
      Toast.err(`분할 계산 실패: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "분할 다시 계산";
    }
  }

  function renderVerify(res) {
    const box = $("verify-body");
    box.className = "mono";
    box.innerHTML = "";
    const verify = res.verify || {};
    const ok = verify.ok;
    const LIMB_KO = { "arm-left": "왼팔", "arm-right": "오른팔",
                      "leg-left": "왼다리", "leg-right": "오른다리" };
    const head = document.createElement("div");
    head.textContent = ok
      ? "좌·우 팔/다리 모두 분리됨 ✓"
      : `분리 안 됨: ${(verify.missing_limbs || []).map((m) => LIMB_KO[m] || m).join(", ")}`;
    head.style.color = ok ? "var(--ok)" : "var(--danger)";
    box.appendChild(head);

    const add = (t) => { const d = document.createElement("div"); d.textContent = t; box.appendChild(d); };
    add(`파츠 ${res.part_count}개`);
    const sides = verify.parts_per_side || {};
    add(`좌 ${sides.left || 0} · 우 ${sides.right || 0}`);
    // Slices only exist when RigSettings.slice_limb_spanning is on; by default
    // limbs are left whole and bent by weights instead.
    const slices = (res.report && res.report.garment_slices) || {};
    for (const [src, pieces] of Object.entries(slices)) {
      add(`${src} → ${pieces.length}조각`);
    }
    const forced = (res.report && res.report.forced) || [];
    for (const f of forced) {
      add(`강제 분리: ${f.region}${f.source ? ` ← ${f.source}` : ` (${f.reason})`}`);
    }
  }

  /* ── step 4 ─────────────────────────────────────────────────── */

  async function runExport() {
    const btn = $("btn-export");
    btn.disabled = true;
    btn.textContent = "리깅 중…";
    try {
      await API.putRig(projectId, BoneEditor.getRig());
      const res = await API.exportRig(projectId);
      renderExport(res);
      show("export");
      Toast.ok(`Spine 내보내기 완료 · 메시 ${res.meshes}개`);
    } catch (err) {
      Toast.err(`내보내기 실패: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "리깅 후 Spine 내보내기";
    }
  }

  function renderExport(res) {
    const chips = $("export-stats");
    chips.textContent = "";
    const verifyOk = res.verify && res.verify.ok;
    const items = [
      [`파츠 ${res.parts}`, ""],
      [`본 ${res.bones}`, ""],
      [`메시 ${res.meshes}`, "ok"],
      [`리전 ${res.regions}`, ""],
      [`아틀라스 ${res.atlas_size.join("×")}`, ""],
      [`애니 ${(res.animations || []).length}`, ""],
      [verifyOk ? "좌우 분리 ✓" : "좌우 분리 미완", verifyOk ? "ok" : "bad"],
    ];
    if (!res.runtime_embedded) {
      items.push(["런타임 CDN", "warn"]);
    }
    for (const [text, cls] of items) {
      const c = document.createElement("span");
      c.className = `chip ${cls}`.trim();
      c.textContent = text;
      chips.appendChild(c);
    }

    const issues = [...(res.validation || []), ...(res.rig_warnings || [])];
    if (!res.runtime_embedded) {
      issues.push("Spine Web Player를 로컬에 두지 않아 미리보기가 unpkg CDN을 사용합니다. "
                + "완전 오프라인으로 만들려면 scripts/fetch_spine_player.ps1 를 실행하세요.");
    }
    $("export-issues").classList.toggle("hidden", issues.length === 0);
    const list = $("issue-list");
    list.textContent = "";
    for (const i of issues) {
      const li = document.createElement("li");
      li.textContent = i;
      list.appendChild(li);
    }

    $("preview-frame").src = API.previewUrl(projectId) + "?t=" + Date.now();
    $("open-preview").href = API.previewUrl(projectId);
    for (const [id, kind] of [["dl-skeleton", "skeleton"], ["dl-atlas", "atlas"],
                              ["dl-png", "png"], ["dl-preview", "preview"]]) {
      $(id).href = API.downloadUrl(projectId, kind);
    }
    for (const b of document.querySelectorAll(".step")) {
      if (b.dataset.step !== "export") b.classList.add("done");
    }
  }

  /* ── boot ───────────────────────────────────────────────────── */

  function bind() {
    for (const b of document.querySelectorAll(".step")) {
      b.addEventListener("click", () => {
        if (b.disabled) return;
        if (!REACHABLE[b.dataset.step]()) { Toast.info("먼저 이미지를 분리하세요"); return; }
        show(b.dataset.step);
      });
    }
    LayerPanel.init(onSelectionChanged);
    $("btn-to-bones").addEventListener("click", async () => {
      show("bones");
      await verifyPartition();
    });
    $("btn-verify").addEventListener("click", verifyPartition);
    $("btn-export").addEventListener("click", runExport);
    $("btn-retry-separate").addEventListener("click", async () => {
      const btn = $("btn-retry-separate");
      btn.disabled = true;
      try {
        await API.separate(projectId);
        Toast.info("레이어 분리를 다시 시작했습니다");
      } catch (err) {
        Toast.err(`다시 실행 실패: ${err.message}`);
      } finally {
        btn.disabled = false;
      }
    });
    $("btn-reset-rig").addEventListener("click", async () => {
      try {
        const rig = await API.getRig(projectId);
        BoneEditor.load(rig, API.fileUrl(projectId, "layers/composite.png"));
        Toast.info("저장된 본 위치를 다시 불러왔습니다");
      } catch (err) { Toast.err(err.message); }
    });
    bindUpload();
  }

  return {
    async start() {
      bind();
      show("upload");
      await loadEnvironment();
      await loadRecent();
      const fromUrl = new URLSearchParams(location.search).get("project");
      if (fromUrl) attach(fromUrl);
    },
  };
})();

window.addEventListener("error", (e) =>
  console.error("[ocs] uncaught", e.error || e.message));
window.addEventListener("unhandledrejection", (e) =>
  console.error("[ocs] unhandled rejection", e.reason));

document.addEventListener("DOMContentLoaded", () => {
  App.start().catch((err) => {
    console.error("[ocs] boot failed", err);
    Toast.err(`초기화 실패: ${err.message}`);
  });
});
