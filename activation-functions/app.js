(() => {
  const $ = s => document.querySelector(s);
  const $$ = s => [...document.querySelectorAll(s)];

  // --- Data: three selectable rain-day datasets. "Overlapping" is the same generator (same
  // seed, same order of operations) as sgd-vs-lda's fixed dataset, so that default matches. ---
  function seededRandom(seed) {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  }
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  const DATASETS = {
    overlapping: { label: "Overlapping", blurb: "The default rain data — LDA tops out at 87.5%.", seed: 20260817, counts: [12, 12], centers: [[34, 38], [66, 68]], spread: [70, 60] },
    separable: { label: "Separable", blurb: "Classes barely overlap — most activations converge cleanly.", seed: 20260820, counts: [12, 12], centers: [[20, 20], [80, 80]], spread: [26, 26] },
    imbalanced: { label: "Imbalanced", blurb: "19 rainy days, 5 dry — skewed class sizes change the pull.", seed: 20260821, counts: [5, 19], centers: [[34, 38], [66, 68]], spread: [70, 60] },
  };
  const DATASET_ORDER = ["overlapping", "separable", "imbalanced"];

  function generateDataset(key) {
    const cfg = DATASETS[key];
    const rand = seededRandom(cfg.seed);
    const rows = [];
    let id = 0;
    cfg.counts.forEach((count, ci) => {
      const [cloudBase, humBase] = cfg.centers[ci];
      const [spreadC, spreadH] = cfg.spread;
      for (let i = 0; i < count; i += 1) {
        const cloud = clamp(Math.round(cloudBase + (rand() - 0.5) * spreadC), 2, 98);
        const humidity = clamp(Math.round(humBase + (rand() - 0.5) * spreadH), 2, 98);
        rows.push({ id: id++, cloud, humidity, y: ci });
      }
    });
    return rows;
  }
  const PREVIEW_DATA = {};
  DATASET_ORDER.forEach(key => { PREVIEW_DATA[key] = generateDataset(key); });

  let DATA = generateDataset("overlapping");
  const scaledX = d => d.cloud / 100, scaledY = d => d.humidity / 100;

  // --- LDA reference error (same closed-form fit as sgd-vs-lda; only the error rate is needed here) ---
  function fitLDA(rows) {
    const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
    const g0 = rows.filter(d => d.y === 0), g1 = rows.filter(d => d.y === 1);
    const m0 = [mean(g0.map(scaledX)), mean(g0.map(scaledY))];
    const m1 = [mean(g1.map(scaledX)), mean(g1.map(scaledY))];
    let sxx = 0, syy = 0, sxy = 0;
    [[g0, m0], [g1, m1]].forEach(([g, m]) => {
      g.forEach(d => { const x = scaledX(d) - m[0], y = scaledY(d) - m[1]; sxx += x * x; syy += y * y; sxy += x * y; });
    });
    const df = rows.length - 2;
    const cxx = sxx / df, cyy = syy / df, cxy = sxy / df;
    const det = cxx * cyy - cxy * cxy;
    const ixx = cyy / det, iyy = cxx / det, ixy = -cxy / det;
    const wOf = m => [ixx * m[0] + ixy * m[1], ixy * m[0] + iyy * m[1]];
    const w1v = wOf(m1), w0v = wOf(m0);
    const w = [w1v[0] - w0v[0], w1v[1] - w0v[1]];
    const prior0 = g0.length / rows.length, prior1 = g1.length / rows.length;
    const b = -0.5 * (m1[0] * w1v[0] + m1[1] * w1v[1]) + 0.5 * (m0[0] * w0v[0] + m0[1] * w0v[1]) + Math.log(prior1 / prior0);
    let correct = 0;
    rows.forEach(d => { const z = w[0] * scaledX(d) + w[1] * scaledY(d) + b; if ((z >= 0 ? 1 : 0) === d.y) correct += 1; });
    return { acc: correct / rows.length };
  }
  let LDA_ERROR = 1 - fitLDA(DATA).acc;

  function shuffledOrder(epoch) {
    const rand = seededRandom(141825719 + epoch * 97);
    const idx = DATA.map(d => d.id);
    for (let i = idx.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rand() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return idx;
  }

  // --- Activation zoo: a(z), its derivative, and each one's own reachable target for label y ---
  const sigmoidRaw = z => (z > 40 ? 1 : z < -40 ? 0 : 1 / (1 + Math.exp(-z)));
  const zeroOne = y => y;
  const plusMinusOne = y => 2 * y - 1;
  const ACT_ORDER = ["sigmoid", "tanh", "arctan", "relu", "step", "linear"];
  // Fixed categorical order, validated for adjacent-pair CVD separation (dataviz skill).
  const COLORS = { sigmoid: "#2a78d6", tanh: "#eb6834", arctan: "#1baf7a", relu: "#eda100", step: "#e87ba4", linear: "#4a3aa7" };
  const ACTIVATIONS = {
    sigmoid: { label: "Sigmoid", f: z => sigmoidRaw(z), fp: (z, a) => a * (1 - a), eq: "a = σ(z) = 1/(1+e^(−z))", target: zeroOne },
    tanh: { label: "Tanh", f: z => Math.tanh(z), fp: (z, a) => 1 - a * a, eq: "a = tanh(z)", target: plusMinusOne },
    arctan: { label: "Arctan", f: z => Math.atan(z), fp: (z) => 1 / (1 + z * z), eq: "a = atan(z)", target: plusMinusOne },
    relu: { label: "ReLU", f: z => Math.max(0, z), fp: (z) => (z >= 0 ? 1 : 0), eq: "a = max(0, z)", target: zeroOne },
    step: { label: "Step", f: z => (z >= 0 ? 1 : 0), fp: () => 0, eq: "a = 1 if z≥0, else 0", target: zeroOne },
    linear: { label: "Linear", f: z => z, fp: () => 1, eq: "a = z", target: plusMinusOne },
  };

  function accuracyFor(w1, w2, b) {
    let correct = 0;
    DATA.forEach(d => { const zraw = w1 * scaledX(d) + w2 * scaledY(d) + b; if ((zraw >= 0 ? 1 : 0) === d.y) correct += 1; });
    return correct / DATA.length;
  }

  const DIVERGE_THRESHOLD = 1e6;
  const isUnsafe = v => !Number.isFinite(v) || Math.abs(v) > DIVERGE_THRESHOLD;

  function updatedWeights(key, w1, w2, b, sample, k, lr) {
    const act = ACTIVATIONS[key];
    const X = scaledX(sample), Y = scaledY(sample);
    const zraw = w1 * X + w2 * Y + b, z = k * zraw, a = act.f(z), t = act.target(sample.y), e = t - a, g = act.fp(z, a);
    const factor = lr * k * e * g;
    return { w1: w1 + factor * X, w2: w2 + factor * Y, b: b + factor };
  }

  function freshRun() {
    return { w1: 0, w2: 0, b: 0, diverged: false, epochErrors: [{ epoch: 0, error: 1 - accuracyFor(0, 0, 0) }] };
  }
  function freshState(datasetKey, k, lr) {
    const runs = {};
    ACT_ORDER.forEach(key => { runs[key] = freshRun(); });
    return { datasetKey, k, lr, epoch: 1, runs, playing: false, autoStopReason: null, visible: new Set(ACT_ORDER) };
  }
  let state = freshState("overlapping", 1, 0.3);
  let playTimer = null;

  function trainEpochFor(key, order) {
    const run = state.runs[key];
    if (!run.diverged) {
      for (const id of order) {
        const sample = DATA[id];
        const upd = updatedWeights(key, run.w1, run.w2, run.b, sample, state.k, state.lr);
        if (isUnsafe(upd.w1) || isUnsafe(upd.w2) || isUnsafe(upd.b)) { run.diverged = true; break; }
        run.w1 = upd.w1; run.w2 = upd.w2; run.b = upd.b;
      }
    }
    run.epochErrors.push({ epoch: state.epoch, error: 1 - accuracyFor(run.w1, run.w2, run.b) });
  }

  function runOneEpoch() {
    const order = shuffledOrder(state.epoch);
    ACT_ORDER.forEach(key => trainEpochFor(key, order));
    state.epoch += 1;
  }

  const MAX_EPOCH = 150;
  function runEpochs(count) {
    const target = Math.min(MAX_EPOCH, state.epoch - 1 + count);
    while (state.epoch <= target) runOneEpoch();
    render();
  }

  function allSettled() {
    return ACT_ORDER.every(key => {
      const run = state.runs[key];
      if (run.diverged) return true;
      if (run.epochErrors.length < 6) return false;
      const recent = run.epochErrors.slice(-5).map(e => e.error);
      return Math.max(...recent) - Math.min(...recent) < 0.001;
    });
  }

  const SPEEDS = [{ m: 1, ms: 320 }, { m: 4, ms: 110 }, { m: 10, ms: 40 }];
  let speedIdx = 0;
  function stopPlaying(reason) { clearInterval(playTimer); playTimer = null; state.playing = false; state.autoStopReason = reason || null; }
  function startPlayTimer() {
    playTimer = setInterval(() => {
      runOneEpoch();
      if (state.epoch > MAX_EPOCH) stopPlaying(`stopped at epoch ${MAX_EPOCH}`);
      else if (allSettled()) stopPlaying("every run stopped changing");
      render();
    }, SPEEDS[speedIdx].ms);
  }
  function togglePlay() {
    if (state.playing) { stopPlaying(); } else { state.playing = true; state.autoStopReason = null; startPlayTimer(); }
    render();
  }
  function cycleSpeed() {
    speedIdx = (speedIdx + 1) % SPEEDS.length;
    $("#speed-toggle").textContent = `${SPEEDS[speedIdx].m}×`;
    if (state.playing) { clearInterval(playTimer); startPlayTimer(); }
  }
  function resetTraining() {
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
    const visible = state.visible;
    state = freshState(state.datasetKey, state.k, state.lr);
    state.visible = visible;
    render();
  }
  function setKLr(k, lr) {
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
    const visible = state.visible;
    state = freshState(state.datasetKey, k, lr);
    state.visible = visible;
    render();
  }
  function setK(k) { setKLr(k, state.lr); }
  function setDataset(key) {
    if (key === state.datasetKey) return;
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
    DATA = generateDataset(key);
    LDA_ERROR = 1 - fitLDA(DATA).acc;
    const visible = state.visible;
    state = freshState(key, state.k, state.lr);
    state.visible = visible;
    render();
  }
  function toggleVisible(key) {
    if (state.visible.has(key)) state.visible.delete(key); else state.visible.add(key);
    render();
  }

  function statusFor(key) {
    const run = state.runs[key];
    if (run.diverged) return { label: "💥 Diverged", cls: "status-oscillating" };
    if (key === "step" && run.epochErrors.length >= 2) return { label: "Frozen (g=0)", cls: "status-oscillating" };
    if (run.epochErrors.length < 4) return { label: "Warming up", cls: "status-warm" };
    const recent = run.epochErrors.slice(-5).map(e => e.error);
    const spread = Math.max(...recent) - Math.min(...recent);
    const meanRecent = recent.reduce((a, b) => a + b, 0) / recent.length;
    if (spread >= 0.12) return { label: "Oscillating ⚠", cls: "status-oscillating" };
    if (meanRecent > LDA_ERROR + 0.15) return { label: "Stuck (saturated)", cls: "status-wobble" };
    if (spread < 0.03) return { label: "Stable ✓", cls: "status-stable" };
    return { label: "Wobbling", cls: "status-wobble" };
  }

  function peakSlopeFor(key, k) {
    const act = ACTIVATIONS[key];
    const N = 60, zMin = -1.5, zMax = 1.5;
    let peak = 0;
    for (let i = 0; i <= N; i += 1) {
      const zraw = zMin + (zMax - zMin) * (i / N);
      const z = k * zraw;
      const a = act.f(z);
      const g = Math.abs(k * act.fp(z, a));
      if (Number.isFinite(g)) peak = Math.max(peak, g);
    }
    return peak;
  }

  // How many of the 24 real training points currently sit where the slope is ~0 for this
  // activation's own weights — the concrete "why" behind a stuck or frozen error line.
  const DEAD_SLOPE_EPS = 0.01;
  function deadPointCountFor(key) {
    const act = ACTIVATIONS[key];
    const run = state.runs[key];
    let count = 0;
    DATA.forEach(d => {
      const zraw = run.w1 * scaledX(d) + run.w2 * scaledY(d) + run.b;
      const z = state.k * zraw;
      const a = act.f(z);
      const g = Math.abs(state.k * act.fp(z, a));
      if (!Number.isFinite(g) || g < DEAD_SLOPE_EPS) count += 1;
    });
    return count;
  }

  // --- Shape & slope viewer: fixed shared y-domains so 6 overlaid curves stay honestly comparable ---
  // Compact but labeled: small titles, a z axis, and just enough y ticks to anchor scale.
  const AV = { left: 26, right: 192, top: 20, bottom: 96, zMin: -1.5, zMax: 1.5 };
  const OUT_MIN = -1.8, OUT_MAX = 1.8, SLOPE_MAX = 3;

  function renderShapeSlope() {
    const { left, right, top, bottom, zMin, zMax } = AV;
    const width = right - left, height = bottom - top;
    const px = z => left + ((z - zMin) / (zMax - zMin)) * width;
    const pyA = a => bottom - ((clamp(a, OUT_MIN, OUT_MAX) - OUT_MIN) / (OUT_MAX - OUT_MIN)) * height;
    const pyS = s => bottom - (Math.min(Math.abs(s), SLOPE_MAX) / SLOPE_MAX) * height;
    const N = 50;

    const zeroLineA = `<line class="av-zero" x1="${px(0)}" y1="${top}" x2="${px(0)}" y2="${bottom}"/><line class="av-zero" x1="${left}" y1="${pyA(0)}" x2="${right}" y2="${pyA(0)}"/>`;
    const zeroLineS = `<line class="av-zero" x1="${px(0)}" y1="${top}" x2="${px(0)}" y2="${bottom}"/><line class="av-zero" x1="${left}" y1="${pyS(0)}" x2="${right}" y2="${pyS(0)}"/>`;

    const titleA = `<text class="mini-av-title" x="${left}" y="12">a(z)</text>`;
    const titleS = `<text class="mini-av-title" x="${left}" y="12">a′(z)</text>`;
    const xAxisLabel = `<text class="mini-av-axis" x="${(left + right) / 2}" y="112" text-anchor="middle">z</text>`;
    const yTicksA = [OUT_MIN, OUT_MAX].map(v => `<text class="mini-av-axis" x="${left - 4}" y="${pyA(v) + 3}" text-anchor="end">${v}</text>`).join("");
    const yTicksS = [0, SLOPE_MAX].map(v => `<text class="mini-av-axis" x="${left - 4}" y="${pyS(v) + 3}" text-anchor="end">${v}</text>`).join("");

    let curvesA = "", curvesS = "";
    // Hidden series draw first (in muted gray, for context) so the visible, colored ones always sit on top.
    const ordered = [...ACT_ORDER.filter(k => !state.visible.has(k)), ...ACT_ORDER.filter(k => state.visible.has(k))];
    ordered.forEach(key => {
      const act = ACTIVATIONS[key];
      const k = state.k;
      const on = state.visible.has(key);
      const ptsA = [], ptsS = [];
      for (let i = 0; i <= N; i += 1) {
        const zraw = zMin + (zMax - zMin) * (i / N);
        const z = k * zraw;
        const a = act.f(z);
        const g = k * act.fp(z, a);
        ptsA.push([px(zraw), pyA(a)]);
        ptsS.push([px(zraw), pyS(g)]);
      }
      const pathA = ptsA.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
      const pathS = ptsS.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
      const cls = on ? "av-curve" : "av-curve av-curve-off";
      const color = on ? COLORS[key] : "";
      curvesA += `<path d="${pathA}" class="${cls}" style="${color ? `stroke:${color}` : ""}"/>`;
      curvesS += `<path d="${pathS}" class="${cls}" style="${color ? `stroke:${color}` : ""}"/>`;
    });

    $("#activation-chart").innerHTML = `${zeroLineA}${curvesA}${titleA}${yTicksA}${xAxisLabel}`;
    $("#slope-chart").innerHTML = `${zeroLineS}${curvesS}${titleS}${yTicksS}${xAxisLabel}`;
  }

  // --- The race: error vs. epoch, every activation on one chart ---
  const RACE = { left: 46, right: 610, top: 14, bottom: 248 };

  function pickEpochTicks(maxEpoch) {
    if (maxEpoch <= 10) return Array.from({ length: maxEpoch + 1 }, (_, i) => i);
    const step = Math.ceil(maxEpoch / 8);
    const ticks = [];
    for (let e = 0; e <= maxEpoch; e += step) ticks.push(e);
    if (ticks[ticks.length - 1] !== maxEpoch) ticks.push(maxEpoch);
    return ticks;
  }

  function renderRaceChart() {
    const svg = $("#race-chart");
    const { left, right, top, bottom } = RACE;
    const width = right - left, height = bottom - top;
    const maxEpoch = Math.max(1, state.epoch - 1);
    let domainMax = LDA_ERROR;
    ACT_ORDER.forEach(key => { state.runs[key].epochErrors.forEach(e => { domainMax = Math.max(domainMax, e.error); }); });
    domainMax = Math.max(0.55, domainMax * 1.12);
    const px = e => left + (e / maxEpoch) * width;
    const py = v => bottom - (v / domainMax) * height;

    const grid = [0, 0.25, 0.5, 0.75, 1].map(f => {
      const v = f * domainMax;
      return `<line class="plot-grid" x1="${left}" y1="${py(v).toFixed(1)}" x2="${right}" y2="${py(v).toFixed(1)}"/><text class="axis-label race-y" x="${left - 8}" y="${(py(v) + 3).toFixed(1)}" text-anchor="end">${(v * 100).toFixed(0)}%</text>`;
    }).join("");
    const ldaY = py(LDA_ERROR);
    const ldaLine = `<line class="err-lda-line" x1="${left}" y1="${ldaY.toFixed(1)}" x2="${right}" y2="${ldaY.toFixed(1)}"/><text class="axis-label race-lda-label" x="${right}" y="${(ldaY - 5).toFixed(1)}" text-anchor="end">LDA ${(LDA_ERROR * 100).toFixed(1)}%</text>`;
    const xTicks = pickEpochTicks(maxEpoch).map(e => `<text class="axis-label" x="${px(e).toFixed(1)}" y="${bottom + 20}" text-anchor="middle">${e}</text>`).join("");
    const xTitle = `<text class="axis-label" x="${(left + right) / 2}" y="${bottom + 38}" text-anchor="middle">epoch</text>`;

    let lines = "";
    // Hidden series draw first (in muted gray, for context) so the visible, colored ones always sit on top.
    const ordered = [...ACT_ORDER.filter(k => !state.visible.has(k)), ...ACT_ORDER.filter(k => state.visible.has(k))];
    ordered.forEach(key => {
      const on = state.visible.has(key);
      const errs = state.runs[key].epochErrors;
      const path = errs.map((p, i) => `${i ? "L" : "M"}${px(p.epoch).toFixed(1)},${py(p.error).toFixed(1)}`).join(" ");
      const last = errs[errs.length - 1];
      const lineCls = on ? "race-line" : "race-line race-line-off";
      const dotCls = on ? "race-dot" : "race-dot race-dot-off";
      const color = on ? COLORS[key] : "";
      lines += `<path d="${path}" class="${lineCls}" style="${color ? `stroke:${color}` : ""}"/><circle class="${dotCls}" cx="${px(last.epoch).toFixed(1)}" cy="${py(last.error).toFixed(1)}" r="4" style="${color ? `fill:${color}` : ""}"/>`;
    });

    svg.innerHTML = `${grid}${ldaLine}${xTicks}${xTitle}${lines}`;
  }

  // --- Legend / toggle row + comparison table + controls ---
  function renderLegend() {
    $("#activation-legend").innerHTML = ACT_ORDER.map(key => {
      const on = state.visible.has(key);
      return `<button type="button" class="legend-toggle${on ? " on" : ""}" data-key="${key}" style="--sw:${COLORS[key]}"><i class="legend-swatch"></i>${ACTIVATIONS[key].label}</button>`;
    }).join("");
    $$(".legend-toggle").forEach(btn => btn.addEventListener("click", () => toggleVisible(btn.dataset.key)));
  }

  function renderTable() {
    $("#compare-body").innerHTML = ACT_ORDER.map(key => {
      const run = state.runs[key];
      const acc = accuracyForRun(run);
      const status = statusFor(key);
      const peak = peakSlopeFor(key, state.k);
      const dead = deadPointCountFor(key);
      const dim = state.visible.has(key) ? "" : " row-dim";
      const deadCls = dead === 0 ? "" : dead >= 18 ? "dead-high" : dead >= 8 ? "dead-mid" : "dead-low";
      return `<tr class="${dim}"><td><i class="legend-swatch" style="--sw:${COLORS[key]}"></i>${ACTIVATIONS[key].label}</td><td>${peak.toFixed(2)}</td><td class="${deadCls}">${dead} / ${DATA.length}</td><td>${(acc * 100).toFixed(1)}%</td><td><span class="status-badge ${status.cls}">${status.label}</span></td></tr>`;
    }).join("");
  }
  function accuracyForRun(run) { return 1 - run.epochErrors[run.epochErrors.length - 1].error; }

  function renderControls() {
    $("#k-slider").value = state.k;
    $("#lr-slider").value = state.lr;
    $("#lr-value").textContent = state.lr.toFixed(2);
    $("#effective-step-value").textContent = (state.k * state.lr).toFixed(2);
    const autoNote = state.autoStopReason ? ` · auto-paused (${state.autoStopReason})` : "";
    $("#status-text").textContent = `${DATASETS[state.datasetKey].label} · k = ${state.k.toFixed(1)} · η = ${state.lr.toFixed(2)} · Epoch ${Math.max(0, state.epoch - 1)}${autoNote}`;
    $("#play-pause").textContent = state.playing ? "⏸ Pause" : "▶ Auto-train";
    $$(".preset-button").forEach(btn => btn.classList.toggle("active", Number(btn.dataset.k) === state.k && Math.abs(Number(btn.dataset.lr) - state.lr) < 1e-9));
    $$(".dataset-button").forEach(btn => btn.classList.toggle("active", btn.dataset.key === state.datasetKey));
  }

  // --- Dataset picker: small static preview scatter per dataset, built once ---
  const PREVIEW_PLOT = { size: 100, pad: 6 };
  function previewSvg(rows) {
    const { size, pad } = PREVIEW_PLOT;
    const px = v => pad + (v / 100) * (size - 2 * pad);
    const dots = rows.map(d => `<circle cx="${px(d.cloud).toFixed(1)}" cy="${(size - px(d.humidity)).toFixed(1)}" r="3" fill="${d.y ? "#148a88" : "#d58b16"}" fill-opacity=".8"/>`).join("");
    return `<svg viewBox="0 0 ${size} ${size}" class="dataset-preview" role="img" aria-hidden="true">${dots}</svg>`;
  }
  function initDatasetPicker() {
    $("#dataset-picker").innerHTML = DATASET_ORDER.map(key => {
      const cfg = DATASETS[key];
      return `<button type="button" class="dataset-button" data-key="${key}">${previewSvg(PREVIEW_DATA[key])}<span class="dataset-name">${cfg.label}</span><span class="dataset-blurb">${cfg.blurb}</span></button>`;
    }).join("");
    $$(".dataset-button").forEach(btn => btn.addEventListener("click", () => setDataset(btn.dataset.key)));
  }

  function render() {
    renderControls();
    renderShapeSlope();
    renderRaceChart();
    renderLegend();
    renderTable();
  }

  // --- Events ---
  $$(".preset-button").forEach(btn => btn.addEventListener("click", () => setKLr(Number(btn.dataset.k), Number(btn.dataset.lr))));
  $("#k-slider").addEventListener("input", e => setK(Number(e.target.value)));
  $("#lr-slider").addEventListener("input", e => { state.lr = Number(e.target.value); render(); });
  $("#run-epoch").addEventListener("click", () => {
    const n = Math.max(1, Math.min(200, Math.round(Number($("#epoch-count").value)) || 1));
    runEpochs(n);
  });
  $("#speed-toggle").addEventListener("click", cycleSpeed);
  $("#play-pause").addEventListener("click", togglePlay);
  $("#reset-btn").addEventListener("click", resetTraining);

  initDatasetPicker();
  render();
})();
