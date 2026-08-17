(() => {
  const $ = s => document.querySelector(s);
  const $$ = s => [...document.querySelectorAll(s)];

  // --- Data: same three rain-day datasets (2 features: cloud %, humidity %) as
  // sgd-vs-lda and activation-functions — same seeds, same generator, so the
  // scatter is pixel-identical across all Chapter 5 demos. ---
  function seededRandom(seed) {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  }
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  const DATASETS = {
    overlapping: { label: "Overlapping", blurb: "The default rain data — LDA tops out at 87.5%.", seed: 20260817, counts: [12, 12], centers: [[34, 38], [66, 68]], spread: [70, 60] },
    separable: { label: "Separable", blurb: "Classes barely overlap — the margin has room to be wide.", seed: 20260820, counts: [12, 12], centers: [[20, 20], [80, 80]], spread: [26, 26] },
    imbalanced: { label: "Imbalanced", blurb: "19 rainy days, 5 dry — skewed class sizes change which points become support vectors.", seed: 20260821, counts: [5, 19], centers: [[34, 38], [66, 68]], spread: [70, 60] },
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

  const CLASS_COLOR = { 0: "#d58b16", 1: "#148a88" };
  const CLASS_LABEL = { 0: "No rain", 1: "Rain" };
  const scaledX = d => d.cloud / 100, scaledY = d => d.humidity / 100;
  const pm1 = d => (d.y === 1 ? 1 : -1); // SVM math wants labels in {-1, +1}, not {0, 1}

  // --- Linear Discriminant Analysis: closed-form, shared-covariance boundary —
  // the same fixed reference line every Chapter 5 demo compares against. ---
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
    return { w, b, acc: correct / rows.length };
  }
  let LDA = fitLDA(DATA);

  function shuffledOrder(epoch) {
    const rand = seededRandom(141825719 + epoch * 97);
    const idx = DATA.map(d => d.id);
    for (let i = idx.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rand() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return idx;
  }

  // --- Training state: Pegasos-style soft-margin SVM, one training point per
  // update. Same shrink-and-nudge shape as the SGD demo's w ← w + η(y−a)x, but
  // the "error" term is now a hinge/margin-violation indicator instead of a
  // smooth residual, and every step also shrinks w toward zero by λ. ---
  const LR = 0.1; // fixed — the slider that matters here is λ, the margin control
  function accuracyFor(w1, w2, b) {
    let correct = 0;
    DATA.forEach(d => { const z = w1 * scaledX(d) + w2 * scaledY(d) + b; if ((z >= 0 ? 1 : 0) === d.y) correct += 1; });
    return correct / DATA.length;
  }
  let LDA_ERROR = 1 - LDA.acc;

  function freshState(datasetKey, lambda) {
    return { datasetKey, lambda, w1: 0, w2: 0, b: 0, epoch: 1, order: shuffledOrder(1), idx: 0, stepInSample: 0, history: [], totalUpdates: 0, playing: false, autoStopReason: null, epochErrors: [{ epoch: 0, error: 1 - accuracyFor(0, 0, 0) }] };
  }
  let state = freshState("overlapping", 0.02);
  let playTimer = null;

  const currentSample = () => DATA[state.order[state.idx]];
  const f3 = n => n.toFixed(3);

  function computeSteps(w1, w2, b, sample, lambda) {
    const X = scaledX(sample), Y = scaledY(sample), y = pm1(sample);
    const z = w1 * X + w2 * Y + b;
    const m = y * z;
    const viol = m < 1 ? 1 : 0;
    const dw1 = LR * (viol * y * X - lambda * w1);
    const dw2 = LR * (viol * y * Y - lambda * w2);
    const db = LR * viol * y;
    const nw1 = w1 + dw1, nw2 = w2 + dw2, nb = b + db;
    return [
      { label: "z", eq: "z = w₁·cloud + w₂·humidity + b", values: `w₁=${f3(w1)}, w₂=${f3(w2)}, b=${f3(b)}, cloud=${X.toFixed(2)}, humidity=${Y.toFixed(2)}`, calc: `${f3(w1)}(${X.toFixed(2)}) + ${f3(w2)}(${Y.toFixed(2)}) + ${f3(b)}`, result: f3(z), meaning: "Signed distance-like score — which side of the boundary, and how far" },
      { label: "margin", eq: "margin = y·z  (y ∈ {−1, +1})", values: `y=${y}, z=${f3(z)}`, calc: `${y} × ${f3(z)}`, result: `${f3(m)}${viol ? "  (< 1 → support vector)" : "  (≥ 1 → outside margin)"}`, meaning: viol ? "Inside the margin or misclassified — this point pushes the boundary" : "Safely past the margin — this point is ignored this update" },
      { label: "Δw₁", eq: "Δw₁ = η·(𝟙[margin<1]·y·cloud − λ·w₁)", values: `η=${LR}, 𝟙=${viol}, y=${y}, cloud=${X.toFixed(2)}, λ=${f3(lambda)}, w₁=${f3(w1)}`, calc: `${LR} × (${viol}×${y}×${X.toFixed(2)} − ${f3(lambda)}×${f3(w1)})`, result: f3(dw1), meaning: "Hinge pull (only if a support vector) minus a constant shrink toward 0" },
      { label: "Δw₂", eq: "Δw₂ = η·(𝟙[margin<1]·y·humidity − λ·w₂)", values: `η=${LR}, 𝟙=${viol}, y=${y}, humidity=${Y.toFixed(2)}, λ=${f3(lambda)}, w₂=${f3(w2)}`, calc: `${LR} × (${viol}×${y}×${Y.toFixed(2)} − ${f3(lambda)}×${f3(w2)})`, result: f3(dw2), meaning: "Same rule, humidity's weight" },
      { label: "Δb", eq: "Δb = η·𝟙[margin<1]·y", values: `η=${LR}, 𝟙=${viol}, y=${y}`, calc: `${LR} × ${viol} × ${y}`, result: f3(db), meaning: "Bias only moves when this point is a support vector — it isn't shrunk by λ" },
      { label: "w₁", eq: "w₁ ← w₁ + Δw₁", values: `w₁=${f3(w1)}, Δw₁=${f3(dw1)}`, calc: `${f3(w1)} + ${f3(dw1)}`, result: f3(nw1), meaning: "Updated cloud-cover weight" },
      { label: "w₂", eq: "w₂ ← w₂ + Δw₂", values: `w₂=${f3(w2)}, Δw₂=${f3(dw2)}`, calc: `${f3(w2)} + ${f3(dw2)}`, result: f3(nw2), meaning: "Updated humidity weight" },
      { label: "b", eq: "b ← b + Δb", values: `b=${f3(b)}, Δb=${f3(db)}`, calc: `${f3(b)} + ${f3(db)}`, result: f3(nb), meaning: "Updated bias" },
    ];
  }

  function updatedWeights(w1, w2, b, sample, lambda) {
    const X = scaledX(sample), Y = scaledY(sample), y = pm1(sample);
    const z = w1 * X + w2 * Y + b, m = y * z, viol = m < 1 ? 1 : 0;
    return { w1: w1 + LR * (viol * y * X - lambda * w1), w2: w2 + LR * (viol * y * Y - lambda * w2), b: b + LR * viol * y, z, m, viol };
  }

  function applyUpdate() {
    const sample = currentSample();
    const { w1, w2, b, z, m, viol } = updatedWeights(state.w1, state.w2, state.b, sample, state.lambda);
    state.history.unshift({ n: state.totalUpdates + 1, sample, z, m, viol, w1, w2, b });
    if (state.history.length > 14) state.history.pop();
    state.w1 = w1; state.w2 = w2; state.b = b;
    state.totalUpdates += 1;
  }

  function advanceToNextSample() {
    state.idx += 1;
    if (state.idx >= state.order.length) {
      state.epochErrors.push({ epoch: state.epoch, error: 1 - accuracyFor(state.w1, state.w2, state.b) });
      state.epoch += 1; state.order = shuffledOrder(state.epoch); state.idx = 0;
    }
    state.stepInSample = 0;
  }

  function tick() {
    if (state.stepInSample < 8) { state.stepInSample = 8; applyUpdate(); }
    else { advanceToNextSample(); }
  }

  function nextStep() {
    if (state.stepInSample < 8) {
      state.stepInSample += 1;
      if (state.stepInSample === 8) applyUpdate();
    } else {
      advanceToNextSample();
    }
    render();
  }
  function finishSample() { tick(); render(); }
  function trainEpochsSilently(count) {
    const targetEpoch = state.epoch + count;
    const maxGuard = count * DATA.length * 2 + 10;
    let guard = 0;
    while (state.epoch < targetEpoch && guard < maxGuard) { tick(); guard += 1; }
  }
  function runEpochs(count) { trainEpochsSilently(count); render(); }

  // A fresh w=0 boundary is invisible (no line, every point "inside" the
  // margin) — confusing the moment you pick a dataset. Pre-train a bit so
  // picking a dataset always shows a real boundary, margin, and a small
  // support-vector set right away. "Reset" still goes back to w=0 for the
  // step-by-step walkthrough.
  const PRETRAIN_EPOCHS = 25;

  const SPEEDS = [{ m: 1, ms: 320 }, { m: 4, ms: 110 }, { m: 10, ms: 40 }];
  let speedIdx = 0;
  const MAX_AUTO_EPOCH = 150;

  function hasConverged() {
    if (state.epochErrors.length < 6) return false;
    const recent = state.epochErrors.slice(-5).map(e => e.error);
    return Math.max(...recent) - Math.min(...recent) < 0.001;
  }
  function stopPlaying(reason) { clearInterval(playTimer); playTimer = null; state.playing = false; state.autoStopReason = reason || null; }
  function startPlayTimer() {
    playTimer = setInterval(() => {
      tick();
      if (state.epoch >= MAX_AUTO_EPOCH) stopPlaying(`stopped at epoch ${MAX_AUTO_EPOCH}`);
      else if (hasConverged()) stopPlaying("error stopped changing");
      render();
    }, SPEEDS[speedIdx].ms);
  }
  function togglePlay() {
    if (state.playing) {
      stopPlaying();
    } else {
      state.playing = true;
      state.autoStopReason = null;
      startPlayTimer();
    }
    render();
  }
  function cycleSpeed() {
    speedIdx = (speedIdx + 1) % SPEEDS.length;
    $("#speed-toggle").textContent = `${SPEEDS[speedIdx].m}×`;
    if (state.playing) { clearInterval(playTimer); startPlayTimer(); }
  }
  function resetTraining() {
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
    state = freshState(state.datasetKey, state.lambda);
    render();
  }
  function setDataset(key) {
    if (key === state.datasetKey) return;
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
    DATA = generateDataset(key);
    LDA = fitLDA(DATA);
    LDA_ERROR = 1 - LDA.acc;
    state = freshState(key, state.lambda);
    trainEpochsSilently(PRETRAIN_EPOCHS);
    render();
  }

  const currentTrainAccuracy = () => accuracyFor(state.w1, state.w2, state.b);
  const supportVectorIds = () => {
    const { w1, w2, b } = state;
    const norm = Math.hypot(w1, w2) || 1;
    return new Set(DATA.filter(d => {
      const z = w1 * scaledX(d) + w2 * scaledY(d) + b;
      return pm1(d) * z < 1 + 1e-6 * norm;
    }).map(d => d.id));
  };

  // --- Chart: scatter + decision regions + boundary/margin lines (SVM vs LDA) ---
  const PLOT = { left: 52, bottom: 300, width: 360, height: 240, top: 56 };

  function boundarySegment(w1, w2, b) {
    const pts = [];
    if (Math.abs(w2) > 1e-9) {
      [0, 1].forEach(X => { const Y = (-b - w1 * X) / w2; if (Y >= -0.02 && Y <= 1.02) pts.push([X, clamp(Y, 0, 1)]); });
    }
    if (Math.abs(w1) > 1e-9) {
      [0, 1].forEach(Y => { const X = (-b - w2 * Y) / w1; if (X >= -0.02 && X <= 1.02) pts.push([clamp(X, 0, 1), Y]); });
    }
    const uniq = [];
    pts.forEach(p => { if (!uniq.some(q => Math.abs(q[0] - p[0]) < 1e-6 && Math.abs(q[1] - p[1]) < 1e-6)) uniq.push(p); });
    if (uniq.length < 2) return null;
    return [[uniq[0][0] * 100, uniq[0][1] * 100], [uniq[1][0] * 100, uniq[1][1] * 100]];
  }

  function renderChart(sample, steps, svIds) {
    const svg = $("#chart");
    const { left, bottom, width, height, top } = PLOT;
    const px = v => left + (v / 100) * width, py = v => bottom - (v / 100) * height;

    let regions = "";
    const cols = 28, rows = 18;
    for (let cx = 0; cx < cols; cx += 1) {
      for (let cy = 0; cy < rows; cy += 1) {
        const fx = (cx + 0.5) / cols * 100, fy = (cy + 0.5) / rows * 100;
        const z = state.w1 * (fx / 100) + state.w2 * (fy / 100) + state.b;
        const pred = z >= 0 ? 1 : 0;
        const x0 = left + (cx / cols) * width, y0 = bottom - ((cy + 1) / rows) * height;
        regions += `<rect x="${x0.toFixed(1)}" y="${y0.toFixed(1)}" width="${(width / cols + 0.6).toFixed(1)}" height="${(height / rows + 0.6).toFixed(1)}" class="decision-region" fill="${CLASS_COLOR[pred]}"/>`;
      }
    }

    let grid = "";
    for (let fr = 0.2; fr < 1; fr += 0.2) {
      const v = fr * 100;
      grid += `<line class="plot-grid" x1="${px(v)}" y1="${top}" x2="${px(v)}" y2="${bottom}"/><line class="plot-grid" x1="${left}" y1="${py(v)}" x2="${left + width}" y2="${py(v)}"/>`;
    }
    const TICKS = [0, 20, 40, 60, 80, 100];
    let ticks = "";
    TICKS.forEach(v => {
      ticks += `<line class="tick-mark" x1="${px(v)}" y1="${bottom}" x2="${px(v)}" y2="${bottom + 5}"/><text class="tick-label" x="${px(v)}" y="${bottom + 17}" text-anchor="middle">${v}</text>`;
      ticks += `<line class="tick-mark" x1="${left - 5}" y1="${py(v)}" x2="${left}" y2="${py(v)}"/><text class="tick-label" x="${left - 9}" y="${py(v) + 4}" text-anchor="end">${v}</text>`;
    });
    const axes = `<line class="plot-axis" x1="${left}" y1="${bottom}" x2="${left + width}" y2="${bottom}"/><line class="plot-axis" x1="${left}" y1="${bottom}" x2="${left}" y2="${top}"/><text class="axis-label" x="${left + width / 2}" y="349">cloud cover, x₁ (%)</text><text class="axis-label" x="16" y="${(top + bottom) / 2}" transform="rotate(-90 16 ${(top + bottom) / 2})">humidity, x₂ (%)</text>`;

    const currentId = sample.id;
    const dots = DATA.map(d => {
      const isCurrent = d.id === currentId;
      const isSV = svIds.has(d.id);
      const ring = isSV ? `<circle class="sv-ring" cx="${px(d.cloud)}" cy="${py(d.humidity)}" r="12"/>` : "";
      return `${ring}<circle class="point${isCurrent ? " point-current" : ""}" cx="${px(d.cloud)}" cy="${py(d.humidity)}" r="${isCurrent ? 9 : 7}" fill="${CLASS_COLOR[d.y]}" stroke="${isCurrent ? "#17212b" : "#fff"}" stroke-width="${isCurrent ? 3.4 : 2}"><title>${CLASS_LABEL[d.y]} · cloud ${d.cloud}%, humidity ${d.humidity}%${isSV ? " · support vector" : ""}</title></circle>`;
    }).join("");

    let lines = "";
    const ldaSeg = boundarySegment(LDA.w[0], LDA.w[1], LDA.b);
    if (ldaSeg) lines += `<line class="boundary-line boundary-lda" x1="${px(ldaSeg[0][0])}" y1="${py(ldaSeg[0][1])}" x2="${px(ldaSeg[1][0])}" y2="${py(ldaSeg[1][1])}"/>`;
    const marginLo = boundarySegment(state.w1, state.w2, state.b - 1);
    const marginHi = boundarySegment(state.w1, state.w2, state.b + 1);
    if (marginLo) lines += `<line class="boundary-line boundary-margin" x1="${px(marginLo[0][0])}" y1="${py(marginLo[0][1])}" x2="${px(marginLo[1][0])}" y2="${py(marginLo[1][1])}"/>`;
    if (marginHi) lines += `<line class="boundary-line boundary-margin" x1="${px(marginHi[0][0])}" y1="${py(marginHi[0][1])}" x2="${px(marginHi[1][0])}" y2="${py(marginHi[1][1])}"/>`;
    const svmSeg = boundarySegment(state.w1, state.w2, state.b);
    if (svmSeg) lines += `<line class="boundary-line boundary-svm" x1="${px(svmSeg[0][0])}" y1="${py(svmSeg[0][1])}" x2="${px(svmSeg[1][0])}" y2="${py(svmSeg[1][1])}"/>`;

    const callout = renderCallout(sample, steps, px(sample.cloud), py(sample.humidity), left, top, width);

    svg.innerHTML = `${regions}${grid}${axes}${ticks}${lines}${dots}${callout}`;
  }

  function escapeXml(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  const truncate = (s, n) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

  function renderCallout(sample, steps, px, py, left, top, width) {
    const lineTexts = state.stepInSample === 0
      ? [`x₁=${sample.cloud}%, x₂=${sample.humidity}%, y=${pm1(sample)} → ${CLASS_LABEL[sample.y]}`]
      : (() => {
          const r = steps[state.stepInSample - 1];
          return [`${r.label} = ${truncate(r.calc, 32)} = ${r.result}`];
        })();
    const ribbonY = 6, ribbonHeight = 26;
    const textLines = lineTexts.map((t, i) => `<text class="callout-text" x="${left + 12}" y="${ribbonY + 17 + i * 15}">${escapeXml(t)}</text>`).join("");
    const dockY = ribbonY + ribbonHeight;
    return `<g class="callout-group"><rect class="callout-box" x="${left}" y="${ribbonY}" width="${width}" height="${ribbonHeight}" rx="7"/>${textLines}<line class="callout-leader" x1="${px}" y1="${dockY}" x2="${px}" y2="${py - 11}"/><circle class="callout-dot" cx="${px}" cy="${dockY}" r="2.6"/></g>`;
  }

  function pickEpochTicks(maxEpoch) {
    if (maxEpoch <= 6) return Array.from({ length: maxEpoch + 1 }, (_, i) => i);
    const step = Math.ceil(maxEpoch / 4);
    const ticks = [];
    for (let e = 0; e <= maxEpoch; e += step) ticks.push(e);
    if (ticks[ticks.length - 1] !== maxEpoch) ticks.push(maxEpoch);
    return ticks;
  }

  function renderErrorChart() {
    const svg = $("#error-chart");
    const left = 30, right = 250, top = 6, bottom = 140, width = right - left, height = bottom - top;
    const errors = state.epochErrors;
    const last = errors[errors.length - 1];
    const maxEpoch = Math.max(1, last.epoch);
    const domainMax = Math.max(0.55, Math.max(...errors.map(e => e.error), LDA_ERROR) * 1.15);
    const px = e => left + (e / maxEpoch) * width;
    const py = v => bottom - (v / domainMax) * height;

    const path = errors.map((p, i) => `${i ? "L" : "M"}${px(p.epoch).toFixed(1)},${py(p.error).toFixed(1)}`).join(" ");
    const area = errors.length > 1 ? `${path} L${px(last.epoch).toFixed(1)},${bottom} L${px(0).toFixed(1)},${bottom} Z` : "";

    const grid = [0, 0.5, 1].map(f => {
      const v = f * domainMax;
      return `<line class="plot-grid" x1="${left}" y1="${py(v).toFixed(1)}" x2="${right}" y2="${py(v).toFixed(1)}"/><text class="axis-label err-y" x="${left - 5}" y="${(py(v) + 3).toFixed(1)}" text-anchor="end">${(v * 100).toFixed(0)}%</text>`;
    }).join("");

    const ldaY = py(LDA_ERROR);
    const ldaLine = `<line class="err-lda-line" x1="${left}" y1="${ldaY.toFixed(1)}" x2="${right}" y2="${ldaY.toFixed(1)}"/>`;
    const currentDot = `<circle class="err-current-dot" cx="${px(last.epoch).toFixed(1)}" cy="${py(last.error).toFixed(1)}" r="3.6"/>`;
    const xTicks = pickEpochTicks(maxEpoch).map(e => `<text class="axis-label" x="${px(e).toFixed(1)}" y="${bottom + 14}" text-anchor="middle">${e}</text>`).join("");
    const xTitle = `<text class="axis-label" x="${(left + right) / 2}" y="${bottom + 27}" text-anchor="middle">epoch</text>`;

    svg.innerHTML = `${grid}${area ? `<path d="${area}" class="err-area"/>` : ""}<path d="${path}" class="err-line"/>${ldaLine}${currentDot}${xTicks}${xTitle}`;

    $("#error-current").textContent = `${(last.error * 100).toFixed(1)}%`;
    $("#error-lda").textContent = `${(LDA_ERROR * 100).toFixed(1)}%`;
  }

  function renderControls() {
    $$(".dataset-button").forEach(btn => btn.classList.toggle("active", btn.dataset.key === state.datasetKey));
    $("#lambda-value").textContent = state.lambda.toFixed(2);
    const autoNote = state.autoStopReason ? ` · auto-paused (${state.autoStopReason})` : "";
    $("#status-text").textContent = `${DATASETS[state.datasetKey].label} · Epoch ${state.epoch} · Sample ${state.idx + 1} / ${DATA.length} · Step ${state.stepInSample} / 8${autoNote}`;
    $("#play-pause").textContent = state.playing ? "⏸ Pause" : "▶ Auto-train";
  }

  const PREVIEW_PLOT = { size: 100, pad: 6 };
  function previewSvg(rows) {
    const { size, pad } = PREVIEW_PLOT;
    const px = v => pad + (v / 100) * (size - 2 * pad);
    const dots = rows.map(d => `<circle cx="${px(d.cloud).toFixed(1)}" cy="${(size - px(d.humidity)).toFixed(1)}" r="3" fill="${CLASS_COLOR[d.y]}" fill-opacity=".8"/>`).join("");
    return `<svg viewBox="0 0 ${size} ${size}" class="dataset-preview" role="img" aria-hidden="true">${dots}</svg>`;
  }
  function initDatasetPicker() {
    $("#dataset-picker").innerHTML = DATASET_ORDER.map(key => {
      const cfg = DATASETS[key];
      return `<button type="button" class="dataset-button" data-key="${key}">${previewSvg(PREVIEW_DATA[key])}<span class="dataset-name">${cfg.label}</span><span class="dataset-blurb">${cfg.blurb}</span></button>`;
    }).join("");
    $$(".dataset-button").forEach(btn => btn.addEventListener("click", () => setDataset(btn.dataset.key)));
  }

  function renderComputation(sample, steps, svIds) {
    $("#sample-x").textContent = `cloud ${sample.cloud}%, humidity ${sample.humidity}%`;
    $("#sample-y").textContent = `actual: ${CLASS_LABEL[sample.y]} (y=${pm1(sample)})`;
    if (state.stepInSample === 0) {
      $("#state-step").textContent = "Step 0 · Press “Next step →” to begin";
      $("#state-equation").textContent = "—";
      ["#state-values", "#state-calculation", "#state-result", "#state-meaning"].forEach(id => { $(id).textContent = "—"; });
    } else {
      const r = steps[state.stepInSample - 1];
      $("#state-step").textContent = `Step ${state.stepInSample} / 8 · ${r.label}`;
      $("#state-equation").textContent = r.eq;
      $("#state-values").textContent = r.values;
      $("#state-calculation").textContent = r.calc;
      $("#state-result").textContent = r.result;
      $("#state-meaning").textContent = r.meaning;
    }
    $("#weights-readout").textContent = `w₁=${state.w1.toFixed(3)} · w₂=${state.w2.toFixed(3)} · b=${state.b.toFixed(3)}`;
    $("#svm-accuracy").textContent = `${(currentTrainAccuracy() * 100).toFixed(1)}%`;
    $("#lda-accuracy").textContent = `${(LDA.acc * 100).toFixed(1)}%`;
    $("#updates-count").textContent = state.totalUpdates;
    const norm = Math.hypot(state.w1, state.w2);
    $("#margin-width").textContent = norm > 1e-6 ? (2 / norm).toFixed(3) : "—";
    $("#sv-count").textContent = `${svIds.size} / ${DATA.length}`;
  }

  function renderTrace() {
    const rows = state.history.map((h, i) => `<tr class="${i === 0 ? "active-row" : ""} revealed"><td>${h.n}</td><td>${h.sample.cloud}%, ${h.sample.humidity}%</td><td>${pm1(h.sample)}</td><td>${h.z.toFixed(3)}</td><td>${h.m.toFixed(3)}</td><td>${h.viol ? "SV" : "—"}</td><td>${h.w1.toFixed(3)}, ${h.w2.toFixed(3)}, ${h.b.toFixed(3)}</td></tr>`).join("");
    $("#trace-body").innerHTML = rows || `<tr><td colspan="7" class="round-empty">No updates yet — press “Next step →” or “Finish sample”.</td></tr>`;
  }

  function render() {
    const sample = currentSample();
    const steps = computeSteps(state.w1, state.w2, state.b, sample, state.lambda);
    const svIds = supportVectorIds();
    renderControls();
    renderComputation(sample, steps, svIds);
    renderChart(sample, steps, svIds);
    renderErrorChart();
    renderTrace();
  }

  // --- Events ---
  // Dragging λ re-trains from scratch at the new value so the margin visibly
  // moves live, instead of only affecting steps taken after the drag.
  $("#lambda-slider").addEventListener("input", e => {
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
    state = freshState(state.datasetKey, Number(e.target.value));
    trainEpochsSilently(PRETRAIN_EPOCHS);
    render();
  });
  $("#next-step").addEventListener("click", nextStep);
  $("#finish-sample").addEventListener("click", finishSample);
  $("#run-epoch").addEventListener("click", () => {
    const n = Math.max(1, Math.min(200, Math.round(Number($("#epoch-count").value)) || 1));
    runEpochs(n);
  });
  $("#speed-toggle").addEventListener("click", cycleSpeed);
  $("#play-pause").addEventListener("click", togglePlay);
  $("#reset-btn").addEventListener("click", resetTraining);

  initDatasetPicker();
  trainEpochsSilently(PRETRAIN_EPOCHS);
  render();
})();
