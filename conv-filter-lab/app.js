(() => {
  const $ = s => document.querySelector(s);
  const $$ = s => [...document.querySelectorAll(s)];
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  // --- Data: one small "rain radar" grid — two storm cells plus background
  // noise, built the same seeded way every other demo in this set builds data. ---
  function seededRandom(seed) {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  }
  const GRID_SIZE = 10;
  const CELL_CENTERS = [{ cx: 3, cy: 3, amp: 0.9, sigma: 1.3 }, { cx: 7, cy: 6, amp: 0.8, sigma: 1.1 }];
  const CORE_RADIUS = 1.6; // "storm-cell core" label radius, used by the learned-filter task

  function buildGrid(seed) {
    const rand = seededRandom(seed);
    const grid = [];
    for (let y = 0; y < GRID_SIZE; y += 1) {
      const row = [];
      for (let x = 0; x < GRID_SIZE; x += 1) {
        let v = 0.05 + rand() * 0.08;
        CELL_CENTERS.forEach(c => { const d2 = (x - c.cx) ** 2 + (y - c.cy) ** 2; v += c.amp * Math.exp(-d2 / (2 * c.sigma * c.sigma)); });
        row.push(Math.min(1, v));
      }
      grid.push(row);
    }
    return grid;
  }
  const GRID = buildGrid(20260817);
  const MIN_POS = 1, MAX_POS = GRID_SIZE - 2; // interior positions with a full 3x3 patch

  function patchAt(grid, cx, cy) {
    const patch = [];
    for (let dy = -1; dy <= 1; dy += 1) for (let dx = -1; dx <= 1; dx += 1) patch.push(grid[cy + dy][cx + dx]);
    return patch; // row-major, length 9
  }
  function isCore(cx, cy) { return CELL_CENTERS.some(c => Math.hypot(cx - c.cx, cy - c.cy) <= CORE_RADIUS); }
  function response(patch, weights, bias) { return patch.reduce((s, v, i) => s + v * weights[i], 0) + (bias || 0); }
  const sigmoid = z => 1 / (1 + Math.exp(-z));

  // --- Colors: sequential teal for intensity (always >= 0), diverging
  // teal/red for filter weights and response values (can go negative). ---
  const hexToRgb = hex => { const n = parseInt(hex.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };
  const mix = (hexA, hexB, t) => { const a = hexToRgb(hexA), b = hexToRgb(hexB), k = clamp(t, 0, 1); return `rgb(${Math.round(a[0] + (b[0] - a[0]) * k)},${Math.round(a[1] + (b[1] - a[1]) * k)},${Math.round(a[2] + (b[2] - a[2]) * k)})`; };
  const sequentialColor = v => mix("#ffffff", "#148a88", v);
  const divergingColor = (v, maxAbs) => mix("#ffffff", v >= 0 ? "#148a88" : "#c5504e", maxAbs > 1e-9 ? Math.abs(v) / maxAbs : 0);

  // --- Fixed filter: a horizontal Sobel edge detector — hand-designed, never trained. ---
  const FIXED_FILTER = [-1, 0, 1, -2, 0, 2, -1, 0, 1];

  // --- Manual "slide it yourself" section state ---
  const MANUAL_PRESETS = {
    edge: { label: "Edge detector (Sobel)", weights: FIXED_FILTER },
    blur: { label: "Blur / average", weights: [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9] },
  };
  let manualPreset = "edge";
  let manualPos = { cx: MIN_POS, cy: MIN_POS };
  let manualRevealed = false;
  let guessTally = { close: 0, total: 0 };

  function manualWeights() { return MANUAL_PRESETS[manualPreset].weights; }

  function moveManual(dx, dy) {
    manualPos = { cx: clamp(manualPos.cx + dx, MIN_POS, MAX_POS), cy: clamp(manualPos.cy + dy, MIN_POS, MAX_POS) };
    manualRevealed = false;
    $("#manual-guess").value = "";
    renderManual();
  }
  function jumpManual(cx, cy) {
    if (cx < MIN_POS || cx > MAX_POS || cy < MIN_POS || cy > MAX_POS) return;
    manualPos = { cx, cy };
    manualRevealed = false;
    $("#manual-guess").value = "";
    renderManual();
  }
  function revealManual() {
    const guess = Number($("#manual-guess").value);
    const actual = response(patchAt(GRID, manualPos.cx, manualPos.cy), manualWeights(), 0);
    if ($("#manual-guess").value !== "") {
      guessTally.total += 1;
      if (Math.abs(guess - actual) < 0.5) guessTally.close += 1;
    }
    manualRevealed = true;
    renderManual();
  }

  // --- Fixed-vs-learned sweep section state ---
  let stride = 1;
  let pooling = false;
  let lr = 0.4;
  let learned = null; // set by resetLearned()
  let learnedHistory = [];

  function interiorOrder(epoch) {
    const rand = seededRandom(7919 + epoch * 131);
    const cells = [];
    for (let cy = MIN_POS; cy <= MAX_POS; cy += 1) for (let cx = MIN_POS; cx <= MAX_POS; cx += 1) cells.push({ cx, cy });
    for (let i = cells.length - 1; i > 0; i -= 1) { const j = Math.floor(rand() * (i + 1)); [cells[i], cells[j]] = [cells[j], cells[i]]; }
    return cells;
  }
  function accuracyOf(weights, bias) {
    let correct = 0, count = 0;
    for (let cy = MIN_POS; cy <= MAX_POS; cy += 1) for (let cx = MIN_POS; cx <= MAX_POS; cx += 1) {
      const a = sigmoid(response(patchAt(GRID, cx, cy), weights, bias));
      if ((a >= 0.5 ? 1 : 0) === (isCore(cx, cy) ? 1 : 0)) correct += 1;
      count += 1;
    }
    return correct / count;
  }
  function resetLearned() {
    learned = { weights: new Array(9).fill(0), bias: 0, epoch: 1, order: interiorOrder(1), idx: 0, totalUpdates: 0, epochAcc: [{ epoch: 0, acc: accuracyOf(new Array(9).fill(0), 0) }], playing: false, autoStopReason: null };
    learnedHistory = [];
  }
  resetLearned();

  function applyLearnedUpdate() {
    const { cx, cy } = learned.order[learned.idx];
    const patch = patchAt(GRID, cx, cy);
    const y = isCore(cx, cy) ? 1 : 0;
    const z = response(patch, learned.weights, learned.bias);
    const a = sigmoid(z);
    const e = y - a;
    const newWeights = learned.weights.map((w, i) => w + lr * e * patch[i]);
    learned.weights = newWeights;
    learned.bias += lr * e;
    learned.totalUpdates += 1;
    learnedHistory.unshift({ n: learned.totalUpdates, cx, cy, y, z, a, e });
    if (learnedHistory.length > 10) learnedHistory.pop();
  }
  function advanceLearned() {
    learned.idx += 1;
    if (learned.idx >= learned.order.length) {
      learned.epochAcc.push({ epoch: learned.epoch, acc: accuracyOf(learned.weights, learned.bias) });
      learned.epoch += 1; learned.order = interiorOrder(learned.epoch); learned.idx = 0;
    }
  }
  function tickLearned() { applyLearnedUpdate(); advanceLearned(); }
  function runLearnedEpochs(count) {
    const target = learned.epoch + count, guard = count * 64 * 2 + 10;
    let n = 0;
    while (learned.epoch < target && n < guard) { tickLearned(); n += 1; }
  }

  const SPEEDS = [{ m: 1, ms: 260 }, { m: 4, ms: 90 }, { m: 12, ms: 30 }];
  let speedIdx = 0;
  let playTimer = null;
  const MAX_AUTO_EPOCH = 120;
  function hasConverged() {
    if (learned.epochAcc.length < 6) return false;
    const recent = learned.epochAcc.slice(-5).map(e => e.acc);
    return Math.max(...recent) - Math.min(...recent) < 0.003;
  }
  function stopPlaying(reason) { clearInterval(playTimer); playTimer = null; learned.playing = false; learned.autoStopReason = reason || null; }
  function startPlayTimer() {
    playTimer = setInterval(() => {
      tickLearned();
      if (learned.epoch >= MAX_AUTO_EPOCH) stopPlaying(`stopped at epoch ${MAX_AUTO_EPOCH}`);
      else if (hasConverged()) stopPlaying("accuracy stopped changing");
      renderSweep();
    }, SPEEDS[speedIdx].ms);
  }
  function togglePlay() {
    if (learned.playing) stopPlaying();
    else { learned.playing = true; learned.autoStopReason = null; startPlayTimer(); }
    renderSweep();
  }

  // --- Feature maps ---
  function featureMap(weights, bias) {
    const rows = [];
    for (let cy = MIN_POS; cy <= MAX_POS; cy += stride) {
      const row = [];
      for (let cx = MIN_POS; cx <= MAX_POS; cx += stride) row.push(response(patchAt(GRID, cx, cy), weights, bias));
      rows.push(row);
    }
    return rows;
  }
  function maxPool2x2(map) {
    const out = [];
    for (let y = 0; y + 1 < map.length; y += 2) {
      const row = [];
      for (let x = 0; x + 1 < map[0].length; x += 2) row.push(Math.max(map[y][x], map[y][x + 1], map[y + 1][x], map[y + 1][x + 1]));
      out.push(row);
    }
    return out.length ? out : map;
  }

  // --- Rendering: a generic heatmap grid renderer reused for the radar grid,
  // the 3x3 weight grids, and the feature maps. ---
  function heatmapSvg(data, { size = 220, diverging = false, seqDomain = [0, 1], highlight = null, onCellClick = null, idPrefix = "" } = {}) {
    const rows = data.length, cols = data[0].length;
    const cell = size / Math.max(rows, cols);
    let maxAbs = 1e-9;
    if (diverging) data.forEach(row => row.forEach(v => { maxAbs = Math.max(maxAbs, Math.abs(v)); }));
    let markup = "";
    data.forEach((row, y) => row.forEach((v, x) => {
      const color = diverging ? divergingColor(v, maxAbs) : sequentialColor((v - seqDomain[0]) / (seqDomain[1] - seqDomain[0]));
      const isHi = highlight && highlight.x === x && highlight.y === y;
      const cx = x * cell, cy = y * cell;
      markup += `<rect class="hm-cell${isHi ? " hm-hi" : ""}" data-x="${x}" data-y="${y}" x="${cx.toFixed(1)}" y="${cy.toFixed(1)}" width="${cell.toFixed(1)}" height="${cell.toFixed(1)}" fill="${color}"><title>${v.toFixed(2)}</title></rect>`;
    }));
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${cols * cell} ${rows * cell}`);
    svg.setAttribute("class", "heatmap");
    svg.setAttribute("role", "img");
    svg.innerHTML = markup;
    if (onCellClick) svg.addEventListener("click", e => {
      const rect = e.target.closest(".hm-cell");
      if (!rect) return;
      onCellClick(Number(rect.dataset.x), Number(rect.dataset.y));
    });
    return svg;
  }

  function mount(id, svgEl) { const host = $(id); host.innerHTML = ""; host.appendChild(svgEl); }

  // --- Manual section rendering ---
  function renderManual() {
    $$(".manual-preset").forEach(btn => btn.classList.toggle("active", btn.dataset.preset === manualPreset));
    const svg = heatmapSvg(GRID, { size: 300, highlight: { x: manualPos.cx, y: manualPos.cy }, onCellClick: (x, y) => jumpManual(x, y) });
    mount("#manual-grid", svg);

    const weights = manualWeights();
    const patch = patchAt(GRID, manualPos.cx, manualPos.cy);
    const products = patch.map((v, i) => v * weights[i]);
    const actual = products.reduce((s, v) => s + v, 0);

    mount("#manual-patch", heatmapSvg([[patch[0], patch[1], patch[2]], [patch[3], patch[4], patch[5]], [patch[6], patch[7], patch[8]]], { size: 108 }));
    mount("#manual-weights", heatmapSvg([[weights[0], weights[1], weights[2]], [weights[3], weights[4], weights[5]], [weights[6], weights[7], weights[8]]], { size: 108, diverging: true }));

    if (manualRevealed) {
      mount("#manual-products", heatmapSvg([[products[0], products[1], products[2]], [products[3], products[4], products[5]], [products[6], products[7], products[8]]], { size: 108, diverging: true }));
      $("#manual-sum").textContent = actual.toFixed(2);
      const guessVal = $("#manual-guess").value;
      $("#manual-feedback").textContent = guessVal === "" ? "No guess entered — this is the actual response." : `Your guess: ${Number(guessVal).toFixed(2)} · actual: ${actual.toFixed(2)} · ${Math.abs(Number(guessVal) - actual) < 0.5 ? "close!" : "off — try the next cell"}`;
    } else {
      mount("#manual-products", heatmapSvg([[0, 0, 0], [0, 0, 0], [0, 0, 0]], { size: 108, diverging: true }));
      $("#manual-sum").textContent = "?";
      $("#manual-feedback").textContent = "Type a guess (optional), then press Reveal.";
    }
    $("#manual-reveal").disabled = manualRevealed;
    $("#manual-tally").textContent = guessTally.total ? `${guessTally.close}/${guessTally.total} guesses within 0.5` : "No guesses yet";
    $("#manual-position").textContent = `Filter centered at (${manualPos.cx}, ${manualPos.cy})`;
  }

  // --- Sweep section rendering ---
  function renderAccuracyChart() {
    const svg = $("#accuracy-chart");
    const left = 34, right = 290, top = 12, bottom = 130, width = right - left, height = bottom - top;
    const series = learned.epochAcc;
    const maxEpoch = Math.max(1, series[series.length - 1].epoch);
    const px = e => left + (e / maxEpoch) * width;
    const py = a => bottom - a * height;
    const path = series.map((p, i) => `${i ? "L" : "M"}${px(p.epoch).toFixed(1)},${py(p.acc).toFixed(1)}`).join(" ");
    const grid = [0, 0.5, 1].map(v => `<line class="plot-grid" x1="${left}" y1="${py(v).toFixed(1)}" x2="${right}" y2="${py(v).toFixed(1)}"/><text class="axis-label acc-y" x="${left - 6}" y="${(py(v) + 3).toFixed(1)}" text-anchor="end">${(v * 100).toFixed(0)}%</text>`).join("");
    const axes = `<line class="plot-axis" x1="${left}" y1="${bottom}" x2="${right}" y2="${bottom}"/><line class="plot-axis" x1="${left}" y1="${bottom}" x2="${left}" y2="${top}"/><text class="axis-label" x="${(left + right) / 2}" y="${bottom + 27}" text-anchor="middle">epoch</text>`;
    const dot = `<circle class="acc-current-dot" cx="${px(series[series.length - 1].epoch).toFixed(1)}" cy="${py(series[series.length - 1].acc).toFixed(1)}" r="4"/>`;
    svg.innerHTML = `${grid}${axes}<path d="${path}" class="acc-line"/>${dot}`;
  }

  function renderSweep() {
    const fixedMap = featureMap(FIXED_FILTER, 0);
    const learnedMap = featureMap(learned.weights, learned.bias);
    const fixedShown = pooling ? maxPool2x2(fixedMap) : fixedMap;
    const learnedShown = pooling ? maxPool2x2(learnedMap) : learnedMap;

    mount("#fixed-weights", heatmapSvg([[FIXED_FILTER[0], FIXED_FILTER[1], FIXED_FILTER[2]], [FIXED_FILTER[3], FIXED_FILTER[4], FIXED_FILTER[5]], [FIXED_FILTER[6], FIXED_FILTER[7], FIXED_FILTER[8]]], { size: 90, diverging: true }));
    mount("#fixed-feature-map", heatmapSvg(fixedShown, { size: 220, diverging: true }));

    mount("#learned-weights", heatmapSvg([[learned.weights[0], learned.weights[1], learned.weights[2]], [learned.weights[3], learned.weights[4], learned.weights[5]], [learned.weights[6], learned.weights[7], learned.weights[8]]], { size: 90, diverging: true }));
    mount("#learned-feature-map", heatmapSvg(learnedShown, { size: 220, diverging: true }));

    $("#stride-readout").textContent = `stride ${stride} → feature map ${fixedMap[0].length}×${fixedMap.length}${pooling ? ` → pooled ${fixedShown[0].length}×${fixedShown.length}` : ""}`;
    $("#learned-status").textContent = `Epoch ${learned.epoch} · ${learned.totalUpdates} updates${learned.autoStopReason ? ` · auto-paused (${learned.autoStopReason})` : ""}`;
    $("#learned-accuracy").textContent = `${(learned.epochAcc[learned.epochAcc.length - 1].acc * 100).toFixed(0)}%`;
    $("#lr-value").textContent = lr.toFixed(2);
    $("#play-pause").textContent = learned.playing ? "⏸ Pause" : "▶ Auto-train";
    const last = learnedHistory[0];
    $("#last-sample").textContent = last ? `patch (${last.cx},${last.cy}) → y=${last.y}, a=${last.a.toFixed(2)}, e=${last.e.toFixed(2)}` : "No updates yet";
    renderAccuracyChart();
  }

  // --- Events ---
  $$(".manual-preset").forEach(btn => btn.addEventListener("click", () => { manualPreset = btn.dataset.preset; manualRevealed = false; $("#manual-guess").value = ""; renderManual(); }));
  $("#manual-up").addEventListener("click", () => moveManual(0, -1));
  $("#manual-down").addEventListener("click", () => moveManual(0, 1));
  $("#manual-left").addEventListener("click", () => moveManual(-1, 0));
  $("#manual-right").addEventListener("click", () => moveManual(1, 0));
  $("#manual-reveal").addEventListener("click", revealManual);

  $("#stride-toggle").addEventListener("click", () => { stride = stride === 1 ? 2 : 1; renderSweep(); });
  $("#pooling-toggle").addEventListener("change", e => { pooling = e.target.checked; renderSweep(); });
  $("#lr-slider").addEventListener("input", e => { lr = Number(e.target.value); renderSweep(); });
  $("#learned-next").addEventListener("click", () => { tickLearned(); renderSweep(); });
  $("#learned-run").addEventListener("click", () => { const n = Math.max(1, Math.min(120, Math.round(Number($("#learned-epoch-count").value)) || 1)); runLearnedEpochs(n); renderSweep(); });
  $("#speed-toggle").addEventListener("click", () => { speedIdx = (speedIdx + 1) % SPEEDS.length; $("#speed-toggle").textContent = `${SPEEDS[speedIdx].m}×`; if (learned.playing) { clearInterval(playTimer); startPlayTimer(); } });
  $("#play-pause").addEventListener("click", togglePlay);
  $("#learned-reset").addEventListener("click", () => { if (playTimer) { clearInterval(playTimer); playTimer = null; } resetLearned(); renderSweep(); });

  renderManual();
  renderSweep();
})();
