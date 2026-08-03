/* Requirement 2-1: show what was dropped and why, and let the user drop more.
 *
 * Two sections on purpose. The main grid holds layers that survived, with a
 * warning badge on the ones OCS is unsure about; the collapsed section holds
 * what it deleted on its own, so "과감하게 제외" stays reversible.
 */
"use strict";

const LayerPanel = (() => {
  /** Reason codes from ocs/cleanup.py -> Korean label + tooltip. */
  const REASONS = {
    empty_layer:        ["픽셀 없음", "PSD 레이어에 픽셀 데이터가 아예 없습니다."],
    fully_transparent:  ["완전 투명", "모든 픽셀의 알파가 0입니다."],
    tiny_area:          ["면적 미달", "캔버스 대비 면적이 자동 제외 하한 아래입니다."],
    near_transparent:   ["거의 투명", "평균 알파가 1% 미만입니다."],
    duplicate:          ["중복", "다른 레이어와 마스크가 거의 같고 색도 같습니다."],
    sliver:             ["아주 작음", "캔버스의 0.1% 미만. 눈·코·입처럼 정상일 수도 있습니다."],
    speckle:            ["점박이", "가장 큰 덩어리가 전체의 20% 미만 — 노이즈일 가능성."],
    flat_fill:          ["단색 덩어리", "넓은데 색 변화가 거의 없음 — 인페인트 아티팩트일 가능성."],
    contained_in_other: ["다른 레이어에 포함", "더 큰 레이어 안에 95% 이상 들어 있습니다."],
    junk_tag:           ["잡동사니 태그", "objects 태그이면서 면적이 작습니다."],
  };
  const HARD = new Set(["empty_layer", "fully_transparent", "tiny_area",
                        "near_transparent", "duplicate"]);

  let state = { projectId: null, layers: [], excluded: new Set(), revived: new Set() };
  let onChange = () => {};

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  function thumb(layer) {
    const box = el("div", "thumb");
    if (layer.alpha_area_px === 0 || !layer.slug) {
      box.classList.add("empty");
      box.textContent = "no pixels";
      return box;
    }
    const img = new Image();
    img.loading = "lazy";
    img.alt = layer.name;
    img.src = API.fileUrl(state.projectId, `layers/${layer.slug}.png`);
    img.onerror = () => { box.classList.add("empty"); box.textContent = "no image"; };
    box.appendChild(img);
    return box;
  }

  function card(layer, isDropped) {
    const node = el("div", "layer");
    if (layer.verdict === "suspicious") node.classList.add("suspicious");
    node.appendChild(thumb(layer));

    if (layer.side) node.appendChild(el("span", "side-tag", layer.side === "left" ? "L" : "R"));

    const body = el("div", "body");
    const nameRow = el("div", "name");
    const box = document.createElement("input");
    box.type = "checkbox";
    // In the dropped section the checkbox means "bring this back"; in the main
    // grid it means "keep".
    box.checked = isDropped ? state.revived.has(layer.name) : !state.excluded.has(layer.name);
    box.disabled = isDropped && layer.alpha_area_px === 0;
    box.title = isDropped ? "되살리기" : "리깅에 포함";
    box.addEventListener("change", () => {
      if (isDropped) {
        box.checked ? state.revived.add(layer.name) : state.revived.delete(layer.name);
      } else {
        box.checked ? state.excluded.delete(layer.name) : state.excluded.add(layer.name);
        node.classList.toggle("excluded", !box.checked);
      }
      onChange(exportSelection());
    });
    nameRow.appendChild(box);
    nameRow.appendChild(el("span", null, layer.name));
    body.appendChild(nameRow);

    const pct = (layer.coverage * 100);
    body.appendChild(el("div", "meta",
      `${layer.alpha_area_px.toLocaleString()} px · ${pct < 0.01 ? pct.toFixed(4) : pct.toFixed(2)}%`
      + ` · z ${layer.depth_median.toFixed(2)}`));

    if (layer.reasons && layer.reasons.length) {
      const row = el("div", "reasons");
      for (const code of layer.reasons) {
        const [label, tip] = REASONS[code] || [code, code];
        const b = el("span", HARD.has(code) ? "badge hard" : "badge", label);
        b.title = layer.dup_of ? `${tip}\n중복 대상: ${layer.dup_of}`
                : layer.contained_in ? `${tip}\n포함된 레이어: ${layer.contained_in}` : tip;
        row.appendChild(b);
      }
      body.appendChild(row);
    }

    node.appendChild(body);
    if (!isDropped && state.excluded.has(layer.name)) node.classList.add("excluded");
    return node;
  }

  function exportSelection() {
    return { excluded: [...state.excluded], revived: [...state.revived] };
  }

  function render() {
    const grid = document.getElementById("layer-grid");
    const dropped = document.getElementById("dropped-grid");
    const onlySus = document.getElementById("filter-suspicious").checked;
    grid.textContent = "";
    dropped.textContent = "";

    let shown = 0, hiddenBySus = 0;
    for (const layer of state.layers) {
      if (layer.verdict === "auto_dropped") {
        dropped.appendChild(card(layer, true));
        continue;
      }
      if (onlySus && layer.verdict !== "suspicious") { hiddenBySus++; continue; }
      grid.appendChild(card(layer, false));
      shown++;
    }
    if (!shown) {
      grid.appendChild(el("p", "muted",
        onlySus && hiddenBySus ? "의심으로 표시된 레이어가 없습니다." : "표시할 레이어가 없습니다."));
    }

    const nDropped = state.layers.filter((l) => l.verdict === "auto_dropped").length;
    document.getElementById("dropped-count").textContent = `(${nDropped}개)`;
    document.getElementById("dropped-box").classList.toggle("hidden", nDropped === 0);
    renderStats();
  }

  function renderStats() {
    const total = state.layers.length;
    const auto = state.layers.filter((l) => l.verdict === "auto_dropped").length;
    const sus = state.layers.filter((l) => l.verdict === "suspicious").length;
    const kept = total - auto + state.revived.size - state.excluded.size;
    const box = document.getElementById("review-stats");
    box.textContent = "";
    const chips = [
      [`전체 ${total}`, ""],
      [`유지 ${kept}`, "ok"],
      [`자동 제외 ${auto}`, auto ? "bad" : ""],
      [`의심 ${sus}`, sus ? "warn" : ""],
    ];
    if (state.excluded.size) chips.push([`직접 제외 ${state.excluded.size}`, "bad"]);
    if (state.revived.size) chips.push([`되살림 ${state.revived.size}`, "ok"]);
    for (const [text, cls] of chips) {
      box.appendChild(el("span", `chip ${cls}`.trim(), text));
    }
  }

  return {
    init(handler) {
      onChange = handler || (() => {});
      document.getElementById("filter-suspicious").addEventListener("change", render);
      document.getElementById("btn-drop-suspicious").addEventListener("click", () => {
        for (const l of state.layers) {
          if (l.verdict === "suspicious") state.excluded.add(l.name);
        }
        render();
        onChange(exportSelection());
      });
      document.getElementById("btn-keep-all").addEventListener("click", () => {
        state.excluded.clear();
        render();
        onChange(exportSelection());
      });
    },

    load(projectId, payload) {
      state.projectId = projectId;
      state.layers = (payload.layers || []).slice().sort((a, b) => b.coverage - a.coverage);
      state.excluded = new Set(payload.exclusions || []);
      state.revived = new Set(payload.revived || []);
      render();
    },

    selection: exportSelection,
  };
})();
