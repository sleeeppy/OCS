/* Requirement 1: drag the joints.
 *
 * Canvas 2D, matching the reference layout the user supplied -- cyan parent→child
 * lines, yellow joints, dark label chips, cyan for the selected joint, on a
 * checkerboard so transparent art reads as transparent.
 *
 * Coordinates: the rig is stored in *canvas pixel* space (the 1280² square
 * see-through padded the input into). The view transform only affects drawing
 * and hit-testing, never the stored values, so zooming cannot drift the rig.
 */
"use strict";

const BoneEditor = (() => {
  const JOINT_R = 8.5;          // screen px
  const HIT_R = 13;
  const COL_BONE = "#00d2eb";
  const COL_JOINT = "#ffcd14";
  const COL_JOINT_SEL = "#25e0f5";
  const COL_JOINT_OPT = "#b6f36a";

  /* Overlay palette, same order as taxonomy.SKIN_REGIONS. */
  const REGION_COLORS = [
    [255, 92, 92], [255, 152, 58], [90, 152, 255], [58, 220, 220],
    [190, 102, 255], [255, 102, 200], [122, 220, 122], [226, 226, 92],
    [150, 150, 166],
  ];

  let cv, ctx, host, hud;
  let rig = null;                 // {canvas:{width,height}, bones:[...]}
  let boneByName = new Map();
  let template = [];              // [{name,parent,optional}] from /api/health
  let mirrorPairs = new Map();

  let art = null, regionsTinted = null;
  const view = { scale: 1, x: 0, y: 0 };
  const opts = { art: true, regions: false, labels: true, dim: 1 };

  let selected = null;
  let drag = null;                // {name, dx, dy, subtree:[names]}
  let pan = null;
  let spaceDown = false;
  let onChange = () => {};

  /* ── transforms ─────────────────────────────────────────────── */

  const toScreen = (x, y) => [x * view.scale + view.x, y * view.scale + view.y];
  const toWorld = (sx, sy) => [(sx - view.x) / view.scale, (sy - view.y) / view.scale];

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const r = host.getBoundingClientRect();
    cv.width = Math.max(1, Math.round(r.width * dpr));
    cv.height = Math.max(1, Math.round(r.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function fit() {
    if (!rig) return;
    const r = host.getBoundingClientRect();
    const pad = 30;
    const s = Math.min((r.width - pad * 2) / rig.canvas.width,
                       (r.height - pad * 2) / rig.canvas.height);
    view.scale = s > 0 ? s : 1;
    view.x = (r.width - rig.canvas.width * view.scale) / 2;
    view.y = (r.height - rig.canvas.height * view.scale) / 2;
    draw();
  }

  /* ── drawing ────────────────────────────────────────────────── */

  function checkerboard(w, h) {
    const size = 16;
    ctx.fillStyle = "#22252b";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#2a2e35";
    for (let y = 0; y < h; y += size) {
      for (let x = ((y / size) % 2) * size; x < w; x += size * 2) {
        ctx.fillRect(x, y, size, size);
      }
    }
  }

  function draw() {
    if (!ctx) return;
    const r = host.getBoundingClientRect();
    ctx.clearRect(0, 0, r.width, r.height);
    checkerboard(r.width, r.height);
    if (!rig) return;

    const [ox, oy] = toScreen(0, 0);
    const w = rig.canvas.width * view.scale;
    const h = rig.canvas.height * view.scale;

    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, w, h);
    ctx.clip();

    if (art && opts.art) {
      ctx.globalAlpha = opts.dim;
      ctx.drawImage(art, ox, oy, w, h);
      ctx.globalAlpha = 1;
    }
    if (regionsTinted && opts.regions) {
      ctx.globalAlpha = 0.5;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(regionsTinted, ox, oy, w, h);
      ctx.imageSmoothingEnabled = true;
      ctx.globalAlpha = 1;
    }
    ctx.restore();

    ctx.strokeStyle = "#3a4048";
    ctx.lineWidth = 1;
    ctx.strokeRect(ox + .5, oy + .5, w, h);

    // Bones behind joints so the joints stay clickable-looking.
    ctx.lineCap = "round";
    for (const b of rig.bones) {
      if (!b.parent) continue;
      const p = boneByName.get(b.parent);
      if (!p) continue;
      const [x1, y1] = toScreen(p.x, p.y);
      const [x2, y2] = toScreen(b.x, b.y);
      ctx.strokeStyle = COL_BONE;
      ctx.globalAlpha = (selected === b.name || selected === b.parent) ? 1 : 0.72;
      ctx.lineWidth = (selected === b.name || selected === b.parent) ? 3.2 : 2.2;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Labels *under* the joints. Zoomed out, a neighbour's label chip lands on
    // top of another joint and hides the thing you need to click, so the joints
    // have to win. Selection always gets its label drawn last, above everything.
    ctx.font = "600 12px ui-monospace, Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    const chip = (name, x, y) => {
      const tw = ctx.measureText(name).width;
      ctx.fillStyle = "#121418cc";
      roundRect(x - tw / 2 - 5, y - 8, tw + 10, 16, 4);
      ctx.fill();
      ctx.fillStyle = "#f2f4f8";
      ctx.fillText(name, x, y);
    };

    if (opts.labels) {
      for (const b of rig.bones) {
        if (b.name === selected) continue;
        const [x, y] = toScreen(b.x, b.y);
        chip(b.name, x, y + JOINT_R + 13);
      }
    }

    for (const b of rig.bones) {
      const [x, y] = toScreen(b.x, b.y);
      const isSel = selected === b.name;
      ctx.beginPath();
      ctx.arc(x, y, isSel ? JOINT_R + 1.5 : JOINT_R, 0, Math.PI * 2);
      ctx.fillStyle = isSel ? COL_JOINT_SEL : (b.optional ? COL_JOINT_OPT : COL_JOINT);
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#1b1d22";
      ctx.stroke();
    }

    if (opts.labels && selected) {
      const b = boneByName.get(selected);
      if (b) {
        const [x, y] = toScreen(b.x, b.y);
        chip(b.name, x, y + JOINT_R + 14);
      }
    }

    hud.textContent = `${rig.canvas.width}×${rig.canvas.height} · ${(view.scale * 100).toFixed(0)}%`
      + (selected ? ` · ${selected}` : "");
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  /* ── hit testing ────────────────────────────────────────────── */

  function pick(sx, sy) {
    let best = null, bestD = HIT_R * HIT_R;
    // Reverse order: later bones draw on top, so they should be picked first.
    for (let i = rig.bones.length - 1; i >= 0; i--) {
      const b = rig.bones[i];
      const [x, y] = toScreen(b.x, b.y);
      const d = (x - sx) ** 2 + (y - sy) ** 2;
      if (d <= bestD) { bestD = d; best = b.name; }
    }
    return best;
  }

  function descendants(name) {
    const out = [];
    const stack = [name];
    while (stack.length) {
      const cur = stack.pop();
      for (const b of rig.bones) {
        if (b.parent === cur) { out.push(b.name); stack.push(b.name); }
      }
    }
    return out;
  }

  /* ── mutation ───────────────────────────────────────────────── */

  function markEdited(name) {
    const b = boneByName.get(name);
    if (b) b.source = "user";
  }

  function mirror(name) {
    const other = mirrorPairs.get(name);
    const src = boneByName.get(name);
    const dst = other && boneByName.get(other);
    if (!src || !dst) return false;
    const axis = (boneByName.get("torso") || { x: rig.canvas.width / 2 }).x;
    dst.x = 2 * axis - src.x;
    dst.y = src.y;
    markEdited(other);
    return true;
  }

  function snapSymmetric() {
    const axis = (boneByName.get("torso") || { x: rig.canvas.width / 2 }).x;
    let n = 0;
    for (const [a, b] of mirrorPairs) {
      if (a > b) continue;                       // handle each pair once
      const A = boneByName.get(a), B = boneByName.get(b);
      if (!A || !B) continue;
      const dx = ((axis - A.x) + (B.x - axis)) / 2;
      const y = (A.y + B.y) / 2;
      A.x = axis - dx; B.x = axis + dx;
      A.y = B.y = y;
      markEdited(a); markEdited(b);
      n++;
    }
    return n;
  }

  /* ── region overlay ─────────────────────────────────────────── */

  /** Colourise the 8-bit region index PNG the server writes. */
  function tintRegions(img) {
    const c = document.createElement("canvas");
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    const g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(img, 0, 0);
    const data = g.getImageData(0, 0, c.width, c.height);
    const px = data.data;
    for (let i = 0; i < px.length; i += 4) {
      const idx = px[i];
      if (idx === 255) { px[i + 3] = 0; continue; }   // unassigned
      const col = REGION_COLORS[idx % REGION_COLORS.length];
      px[i] = col[0]; px[i + 1] = col[1]; px[i + 2] = col[2]; px[i + 3] = 255;
    }
    g.putImageData(data, 0, 0);
    return c;
  }

  /* ── events ─────────────────────────────────────────────────── */

  function bindEvents() {
    cv.addEventListener("pointerdown", (e) => {
      // Capture keeps a drag alive when the cursor leaves the canvas. It throws
      // for a pointer id the browser is not currently tracking (synthetic
      // events, some pen drivers), which must not abort the drag.
      try { cv.setPointerCapture(e.pointerId); } catch { /* not capturable */ }
      const [sx, sy] = local(e);
      if (spaceDown || e.button === 1) {
        pan = { sx, sy, ox: view.x, oy: view.y };
        cv.classList.add("panning");
        return;
      }
      const hit = pick(sx, sy);
      selected = hit;
      if (hit) {
        const b = boneByName.get(hit);
        const [bx, by] = toScreen(b.x, b.y);
        drag = {
          name: hit, dx: sx - bx, dy: sy - by,
          subtree: e.shiftKey ? descendants(hit) : [],
          start: new Map((e.shiftKey ? [hit, ...descendants(hit)] : [hit])
            .map((n) => [n, { x: boneByName.get(n).x, y: boneByName.get(n).y }])),
        };
      }
      renderSelection();
      draw();
    });

    cv.addEventListener("pointermove", (e) => {
      const [sx, sy] = local(e);
      if (pan) {
        view.x = pan.ox + (sx - pan.sx);
        view.y = pan.oy + (sy - pan.sy);
        draw();
        return;
      }
      if (!drag) return;
      const [wx, wy] = toWorld(sx - drag.dx, sy - drag.dy);
      const origin = drag.start.get(drag.name);
      const dx = wx - origin.x, dy = wy - origin.y;
      for (const [name, p] of drag.start) {
        const b = boneByName.get(name);
        b.x = clamp(p.x + dx, 0, rig.canvas.width);
        b.y = clamp(p.y + dy, 0, rig.canvas.height);
        markEdited(name);
      }
      renderSelection();
      draw();
    });

    const finish = () => {
      if (pan) { pan = null; cv.classList.remove("panning"); }
      if (drag) { drag = null; onChange(getRig()); }
    };
    cv.addEventListener("pointerup", finish);
    cv.addEventListener("pointercancel", finish);
    cv.addEventListener("lostpointercapture", finish);

    cv.addEventListener("wheel", (e) => {
      e.preventDefault();
      const [sx, sy] = local(e);
      const [wx, wy] = toWorld(sx, sy);
      const f = Math.exp(-e.deltaY * 0.0016);
      view.scale = clamp(view.scale * f, 0.05, 12);
      // Keep the cursor over the same pixel of art while zooming.
      view.x = sx - wx * view.scale;
      view.y = sy - wy * view.scale;
      draw();
    }, { passive: false });

    window.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      if (e.code === "Space") { spaceDown = true; e.preventDefault(); return; }
      if (!rig || !isVisible()) return;
      if (e.key === "0") { fit(); return; }
      if (e.key === "x" || e.key === "X") {
        if (selected && mirror(selected)) { draw(); onChange(getRig()); Toast.ok("좌우 미러 적용"); }
        return;
      }
      if (e.key === "s" || e.key === "S") {
        const n = snapSymmetric();
        if (n) { draw(); onChange(getRig()); Toast.ok(`${n}쌍 대칭 스냅`); }
      }
    });
    window.addEventListener("keyup", (e) => {
      if (e.code === "Space") spaceDown = false;
    });
    window.addEventListener("resize", resize);
  }

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const isVisible = () => !!host.offsetParent;

  function local(e) {
    const r = cv.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  /* ── side panel ─────────────────────────────────────────────── */

  function renderSelection() {
    const box = document.getElementById("sel-body");
    if (!selected) { box.textContent = "조인트를 클릭하세요"; box.className = "mono muted"; return; }
    const b = boneByName.get(selected);
    box.className = "mono";
    box.innerHTML = "";
    const dl = document.createElement("dl");
    const rows = [
      ["이름", b.name],
      ["부모", b.parent || "—"],
      ["x", b.x.toFixed(1)],
      ["y", b.y.toFixed(1)],
      ["추정", b.source === "layer" ? "레이어 기반"
             : b.source === "geodesic" ? "실루엣 추적"
             : b.source === "user" ? "직접 수정" : "비례 배치"],
    ];
    for (const [k, v] of rows) {
      const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = v;
      dl.append(dt, dd);
    }
    box.appendChild(dl);
  }

  function renderOptional() {
    const box = document.getElementById("optional-bones");
    box.textContent = "";
    for (const spec of template) {
      if (!spec.optional) continue;
      const on = boneByName.has(spec.name);
      const btn = document.createElement("button");
      btn.className = on ? "on" : "";
      btn.textContent = spec.name;
      btn.title = on ? "클릭하면 제거" : "클릭하면 추가";
      btn.addEventListener("click", () => {
        if (boneByName.has(spec.name)) {
          for (const n of [spec.name, ...descendants(spec.name)]) {
            rig.bones = rig.bones.filter((b) => b.name !== n);
            boneByName.delete(n);
          }
          if (selected === spec.name) selected = null;
        } else {
          const parent = boneByName.get(spec.parent) || boneByName.get("torso");
          if (!parent) return;
          const b = {
            name: spec.name, parent: spec.parent,
            x: parent.x + 24, y: parent.y + 24, optional: true, source: "user",
          };
          // Keep template order so the parent always precedes the child, which
          // the Spine exporter relies on.
          const order = template.map((t) => t.name);
          const at = rig.bones.findIndex((e) => order.indexOf(e.name) > order.indexOf(spec.name));
          at === -1 ? rig.bones.push(b) : rig.bones.splice(at, 0, b);
          boneByName.set(b.name, b);
          selected = b.name;
        }
        renderOptional();
        renderSelection();
        draw();
        onChange(getRig());
      });
      box.appendChild(btn);
    }
  }

  function getRig() {
    return {
      canvas: rig.canvas,
      bones: rig.bones.map((b) => ({
        name: b.name, parent: b.parent ?? null,
        x: Math.round(b.x * 100) / 100, y: Math.round(b.y * 100) / 100,
        optional: !!b.optional, source: b.source || "user",
      })),
    };
  }

  /* ── public ─────────────────────────────────────────────────── */

  return {
    init(boneTemplate, pairs, handler) {
      host = document.getElementById("canvas-host");
      cv = document.getElementById("bone-canvas");
      ctx = cv.getContext("2d");
      hud = document.getElementById("canvas-hud");
      template = boneTemplate || [];
      mirrorPairs = new Map();
      for (const [a, b] of pairs || []) { mirrorPairs.set(a, b); mirrorPairs.set(b, a); }
      onChange = handler || (() => {});
      bindEvents();

      const bind = (id, key, get) => {
        const node = document.getElementById(id);
        node.addEventListener("input", () => { opts[key] = get(node); draw(); });
      };
      bind("show-art", "art", (n) => n.checked);
      bind("show-labels", "labels", (n) => n.checked);
      bind("art-dim", "dim", (n) => n.value / 100);
      document.getElementById("show-regions").addEventListener("change", (e) => {
        opts.regions = e.target.checked;
        draw();
        if (opts.regions && !regionsTinted) Toast.info("‘분할 다시 계산’을 눌러 오버레이를 생성하세요");
      });
    },

    load(rigData, artUrl) {
      rig = { canvas: rigData.canvas, bones: rigData.bones.map((b) => ({ ...b })) };
      boneByName = new Map(rig.bones.map((b) => [b.name, b]));
      selected = null;
      regionsTinted = null;
      renderOptional();
      renderSelection();
      if (artUrl) {
        const img = new Image();
        img.onload = () => { art = img; fit(); };
        img.onerror = () => { art = null; fit(); };
        img.src = artUrl;
      } else {
        art = null;
      }
      resize();
      fit();
    },

    setRegions(url) {
      const img = new Image();
      img.onload = () => {
        regionsTinted = tintRegions(img);
        const cb = document.getElementById("show-regions");
        cb.checked = true;
        opts.regions = true;
        draw();
      };
      img.onerror = () => Toast.err("분할 오버레이를 불러오지 못했습니다");
      img.src = url + "?t=" + Date.now();      // bust the cache after a re-run
    },

    getRig,
    refresh: () => { resize(); fit(); },
  };
})();
