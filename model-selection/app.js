(() => {
  const $ = selector => document.querySelector(selector);

  // --- Data: the classic Iris dataset (sepal length, sepal width, species index) ---
  // species: 0 = setosa, 1 = versicolor, 2 = virginica
  const RAW = [
    [5.1,3.5,0],[4.9,3,0],[4.7,3.2,0],[4.6,3.1,0],[5,3.6,0],[5.4,3.9,0],[4.6,3.4,0],[5,3.4,0],[4.4,2.9,0],[4.9,3.1,0],
    [5.4,3.7,0],[4.8,3.4,0],[4.8,3,0],[4.3,3,0],[5.8,4,0],[5.7,4.4,0],[5.4,3.9,0],[5.1,3.5,0],[5.7,3.8,0],[5.1,3.8,0],
    [5.4,3.4,0],[5.1,3.7,0],[4.6,3.6,0],[5.1,3.3,0],[4.8,3.4,0],[5,3,0],[5,3.4,0],[5.2,3.5,0],[5.2,3.4,0],[4.7,3.2,0],
    [4.8,3.1,0],[5.4,3.4,0],[5.2,4.1,0],[5.5,4.2,0],[4.9,3.1,0],[5,3.2,0],[5.5,3.5,0],[4.9,3.6,0],[4.4,3,0],[5.1,3.4,0],
    [5,3.5,0],[4.5,2.3,0],[4.4,3.2,0],[5,3.5,0],[5.1,3.8,0],[4.8,3,0],[5.1,3.8,0],[4.6,3.2,0],[5.3,3.7,0],[5,3.3,0],
    [7,3.2,1],[6.4,3.2,1],[6.9,3.1,1],[5.5,2.3,1],[6.5,2.8,1],[5.7,2.8,1],[6.3,3.3,1],[4.9,2.4,1],[6.6,2.9,1],[5.2,2.7,1],
    [5,2,1],[5.9,3,1],[6,2.2,1],[6.1,2.9,1],[5.6,2.9,1],[6.7,3.1,1],[5.6,3,1],[5.8,2.7,1],[6.2,2.2,1],[5.6,2.5,1],
    [5.9,3.2,1],[6.1,2.8,1],[6.3,2.5,1],[6.1,2.8,1],[6.4,2.9,1],[6.6,3,1],[6.8,2.8,1],[6.7,3,1],[6,2.9,1],[5.7,2.6,1],
    [5.5,2.4,1],[5.5,2.4,1],[5.8,2.7,1],[6,2.7,1],[5.4,3,1],[6,3.4,1],[6.7,3.1,1],[6.3,2.3,1],[5.6,3,1],[5.5,2.5,1],
    [5.5,2.6,1],[6.1,3,1],[5.8,2.6,1],[5,2.3,1],[5.6,2.7,1],[5.7,3,1],[5.7,2.9,1],[6.2,2.9,1],[5.1,2.5,1],[5.7,2.8,1],
    [6.3,3.3,2],[5.8,2.7,2],[7.1,3,2],[6.3,2.9,2],[6.5,3,2],[7.6,3,2],[4.9,2.5,2],[7.3,2.9,2],[6.7,2.5,2],[7.2,3.6,2],
    [6.5,3.2,2],[6.4,2.7,2],[6.8,3,2],[5.7,2.5,2],[5.8,2.8,2],[6.4,3.2,2],[6.5,3,2],[7.7,3.8,2],[7.7,2.6,2],[6,2.2,2],
    [6.9,3.2,2],[5.6,2.8,2],[7.7,2.8,2],[6.3,2.7,2],[6.7,3.3,2],[7.2,3.2,2],[6.2,2.8,2],[6.1,3,2],[6.4,2.8,2],[7.2,3,2],
    [7.4,2.8,2],[7.9,3.8,2],[6.4,2.8,2],[6.3,2.8,2],[6.1,2.6,2],[7.7,3,2],[6.3,3.4,2],[6.4,3.1,2],[6,3,2],[6.9,3.1,2],
    [6.7,3.1,2],[6.9,3.1,2],[5.8,2.7,2],[6.8,3.2,2],[6.7,3.3,2],[6.7,3,2],[6.3,2.5,2],[6.5,3,2],[6.2,3.4,2],[5.9,3,2],
  ];

  const CLASS_KEYS = ["setosa", "versicolor", "virginica"];
  const CLASS_INFO = {
    setosa: { label: "Setosa", color: "#d66762" },
    versicolor: { label: "Versicolor", color: "#148a88" },
    virginica: { label: "Virginica", color: "#5267bf" },
  };
  const DATA = RAW.map(([sl, sw, t], id) => ({ x: sl, y: sw, label: CLASS_KEYS[t], id }));
  const xRange = [4, 8], yRange = [1.5, 4.5];
  const xLabel = "sepal length (cm)", yLabel = "sepal width (cm)";

  const classLabel = k => CLASS_INFO[k].label;
  const classColor = k => CLASS_INFO[k].color;

  // --- Stratified train/validation/test split (60/20/20), seeded and reproducible ---
  function seededRandom(seed) {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  }
  function stratifiedSplit(rows, seed, trainFrac, valFrac) {
    const rand = seededRandom(seed);
    const byClass = new Map();
    rows.forEach(r => { if (!byClass.has(r.label)) byClass.set(r.label, []); byClass.get(r.label).push(r); });
    const train = [], val = [], test = [];
    byClass.forEach(list => {
      const shuffled = list.slice();
      for (let i = shuffled.length - 1; i > 0; i -= 1) {
        const j = Math.floor(rand() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }
      const n = shuffled.length;
      const nTrain = Math.round(n * trainFrac);
      const nVal = Math.round(n * valFrac);
      train.push(...shuffled.slice(0, nTrain));
      val.push(...shuffled.slice(nTrain, nTrain + nVal));
      test.push(...shuffled.slice(nTrain + nVal));
    });
    return { train, val, test };
  }
  const SPLIT = stratifiedSplit(DATA, 20260817, 0.6, 0.2);
  const SPLIT_OF = new Map();
  SPLIT.train.forEach(r => SPLIT_OF.set(r.id, "train"));
  SPLIT.val.forEach(r => SPLIT_OF.set(r.id, "val"));
  SPLIT.test.forEach(r => SPLIT_OF.set(r.id, "test"));

  const mean = vals => vals.reduce((a, b) => a + b, 0) / vals.length;
  function classGroups(rows) {
    const g = new Map();
    rows.forEach(r => { if (!g.has(r.label)) g.set(r.label, []); g.get(r.label).push(r); });
    return g;
  }

  // --- Candidate model families ---
  function fitNaiveBayes(rows) {
    const groups = classGroups(rows);
    const n = rows.length;
    const stats = new Map();
    groups.forEach((g, label) => {
      const xs = g.map(r => r.x), ys = g.map(r => r.y);
      const mx = mean(xs), my = mean(ys);
      const vx = mean(xs.map(v => (v - mx) ** 2)) + 1e-6;
      const vy = mean(ys.map(v => (v - my) ** 2)) + 1e-6;
      stats.set(label, { mx, my, vx, vy, prior: g.length / n });
    });
    return stats;
  }
  const gauss = (x, m, v) => Math.exp(-((x - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
  function predictNaiveBayes(stats, point) {
    const raw = [];
    stats.forEach((s, label) => raw.push({ label, score: gauss(point.x, s.mx, s.vx) * gauss(point.y, s.my, s.vy) * s.prior }));
    const total = raw.reduce((a, b) => a + b.score, 0) || 1e-12;
    const ranked = raw.map(r => ({ label: r.label, score: r.score, prob: r.score / total }));
    ranked.sort((a, b) => b.prob - a.prob);
    return ranked;
  }

  function fitLDA(rows) {
    const groups = classGroups(rows);
    const n = rows.length;
    const means = new Map(), priors = new Map();
    groups.forEach((g, label) => {
      means.set(label, { mx: mean(g.map(r => r.x)), my: mean(g.map(r => r.y)) });
      priors.set(label, g.length / n);
    });
    let sxx = 0, syy = 0, sxy = 0, count = 0;
    groups.forEach((g, label) => {
      const m = means.get(label);
      g.forEach(r => { sxx += (r.x - m.mx) ** 2; syy += (r.y - m.my) ** 2; sxy += (r.x - m.mx) * (r.y - m.my); count += 1; });
    });
    const df = count - groups.size;
    const cov = { xx: sxx / df, yy: syy / df, xy: sxy / df };
    const det = cov.xx * cov.yy - cov.xy * cov.xy;
    const inv = { xx: cov.yy / det, yy: cov.xx / det, xy: -cov.xy / det };
    return { means, priors, inv };
  }
  function predictLDA(model, point) {
    const raw = [];
    model.means.forEach((m, label) => {
      const prior = model.priors.get(label);
      const wx = model.inv.xx * m.mx + model.inv.xy * m.my;
      const wy = model.inv.xy * m.mx + model.inv.yy * m.my;
      const c = -0.5 * (m.mx * wx + m.my * wy) + Math.log(prior);
      raw.push({ label, score: point.x * wx + point.y * wy + c });
    });
    const maxScore = Math.max(...raw.map(r => r.score));
    const exps = raw.map(r => Math.exp(r.score - maxScore));
    const total = exps.reduce((a, b) => a + b, 0);
    const ranked = raw.map((r, i) => ({ label: r.label, score: r.score, prob: exps[i] / total }));
    ranked.sort((a, b) => b.score - a.score);
    return ranked;
  }

  function standardize(rows) {
    const xs = rows.map(r => r.x), ys = rows.map(r => r.y);
    const mx = mean(xs), my = mean(ys);
    const sx = Math.sqrt(mean(xs.map(v => (v - mx) ** 2))) || 1;
    const sy = Math.sqrt(mean(ys.map(v => (v - my) ** 2))) || 1;
    return { mx, my, sx, sy };
  }
  function fitLogReg(rows, epochs = 800, lr = 0.5) {
    const scaler = standardize(rows);
    const feats = rows.map(r => [(r.x - scaler.mx) / scaler.sx, (r.y - scaler.my) / scaler.sy]);
    const labels = rows.map(r => r.label);
    const K = CLASS_KEYS.length;
    const W = Array.from({ length: K }, () => [0, 0]);
    const b = Array(K).fill(0);
    const n = rows.length;
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      const gradW = Array.from({ length: K }, () => [0, 0]);
      const gradB = Array(K).fill(0);
      for (let i = 0; i < n; i += 1) {
        const f = feats[i];
        const scores = W.map((w, k) => w[0] * f[0] + w[1] * f[1] + b[k]);
        const maxS = Math.max(...scores);
        const exps = scores.map(s => Math.exp(s - maxS));
        const sum = exps.reduce((a, c) => a + c, 0);
        const probs = exps.map(e => e / sum);
        for (let k = 0; k < K; k += 1) {
          const err = probs[k] - (labels[i] === CLASS_KEYS[k] ? 1 : 0);
          gradW[k][0] += err * f[0]; gradW[k][1] += err * f[1]; gradB[k] += err;
        }
      }
      for (let k = 0; k < K; k += 1) {
        W[k][0] -= lr * gradW[k][0] / n; W[k][1] -= lr * gradW[k][1] / n; b[k] -= lr * gradB[k] / n;
      }
    }
    return { W, b, scaler };
  }
  function predictLogReg(model, point) {
    const f = [(point.x - model.scaler.mx) / model.scaler.sx, (point.y - model.scaler.my) / model.scaler.sy];
    const scores = model.W.map((w, k) => w[0] * f[0] + w[1] * f[1] + model.b[k]);
    const maxS = Math.max(...scores);
    const exps = scores.map(s => Math.exp(s - maxS));
    const sum = exps.reduce((a, c) => a + c, 0);
    const ranked = scores.map((s, k) => ({ label: CLASS_KEYS[k], score: s, prob: exps[k] / sum }));
    ranked.sort((a, b) => b.score - a.score);
    return ranked;
  }

  function evaluate(model, predictFn, rows) {
    let correct = 0, logLossSum = 0;
    const correctIds = new Set();
    rows.forEach(r => {
      const ranked = predictFn(model, r);
      if (ranked[0].label === r.label) { correct += 1; correctIds.add(r.id); }
      const trueProb = ranked.find(x => x.label === r.label).prob;
      logLossSum += -Math.log(Math.max(trueProb, 1e-9));
    });
    return { accuracy: correct / rows.length, logLoss: logLossSum / rows.length, correct, total: rows.length, correctIds };
  }

  const CANDIDATES = [
    { key: "logreg", name: "Logistic Regression", blurb: "a straight-line score turned into class probabilities", fit: fitLogReg, predict: predictLogReg },
    { key: "lda", name: "Linear Discriminant Analysis", blurb: "one shared-shape boundary drawn between class centers", fit: fitLDA, predict: predictLDA },
    { key: "nb", name: "Gaussian Naive Bayes", blurb: "each feature votes for a class on its own, then the votes multiply", fit: fitNaiveBayes, predict: predictNaiveBayes },
  ];
  CANDIDATES.forEach(c => {
    c.model = c.fit(SPLIT.train);
    c.val = evaluate(c.model, c.predict, SPLIT.val);
  });
  CANDIDATES.sort((a, b) => b.val.accuracy - a.val.accuracy || a.val.logLoss - b.val.logLoss);
  const WINNER = CANDIDATES[0];
  WINNER.finalModel = WINNER.fit(SPLIT.train.concat(SPLIT.val));
  const TEST_RESULT = evaluate(WINNER.finalModel, WINNER.predict, SPLIT.test);
  const TEST_ERROR_PAIRS = SPLIT.test.filter(r => !TEST_RESULT.correctIds.has(r.id)).map(r => `${classLabel(r.label)} → ${classLabel(WINNER.predict(WINNER.finalModel, r)[0].label)}`);

  // --- UI state ---
  let stage = 0;
  let focusKey = WINNER.key;
  let testPoint = null;
  let selectedId = null;

  const PLOT = { left: 55, bottom: 310, width: 350, height: 250, top: 60 };

  function classifierAt(stageValue) {
    if (stageValue === 2) {
      const c = CANDIDATES.find(x => x.key === focusKey);
      return p => c.predict(c.model, p)[0].label;
    }
    if (stageValue >= 3) return p => WINNER.predict(WINNER.finalModel, p)[0].label;
    return null;
  }

  function renderChart() {
    const svg = $("#chart");
    const { left, bottom, width, height, top } = PLOT;
    const [minX, maxX] = xRange, [minY, maxY] = yRange;
    const px = v => left + ((v - minX) / (maxX - minX)) * width;
    const py = v => bottom - ((v - minY) / (maxY - minY)) * height;

    const classifier = classifierAt(stage);
    let regions = "";
    if (classifier) {
      const cols = 26, rows = 16;
      for (let cx = 0; cx < cols; cx += 1) {
        for (let cy = 0; cy < rows; cy += 1) {
          const fx = minX + (maxX - minX) * (cx + 0.5) / cols;
          const fy = minY + (maxY - minY) * (cy + 0.5) / rows;
          const label = classifier({ x: fx, y: fy });
          const x0 = left + (cx / cols) * width, y0 = bottom - ((cy + 1) / rows) * height;
          regions += `<rect x="${x0.toFixed(1)}" y="${y0.toFixed(1)}" width="${(width / cols + 0.6).toFixed(1)}" height="${(height / rows + 0.6).toFixed(1)}" class="decision-region" fill="${classColor(label)}"/>`;
        }
      }
    }

    let grid = "";
    for (let fraction = 0.2; fraction < 1; fraction += 0.2) {
      const x = minX + (maxX - minX) * fraction, y = minY + (maxY - minY) * fraction;
      grid += `<line class="plot-grid" x1="${px(x)}" y1="${top}" x2="${px(x)}" y2="${bottom}"/><line class="plot-grid" x1="${left}" y1="${py(y)}" x2="${left + width}" y2="${py(y)}"/>`;
    }
    const axes = `<line class="plot-axis" x1="${left}" y1="${bottom}" x2="${left + width}" y2="${bottom}"/><line class="plot-axis" x1="${left}" y1="${bottom}" x2="${left}" y2="${top}"/><text class="axis-label" x="${left + width / 2}" y="345">${xLabel}</text><text class="axis-label" x="18" y="${(top + bottom) / 2}" transform="rotate(-90 18 ${(top + bottom) / 2})">${yLabel}</text>`;

    const dots = DATA.map(row => {
      const split = SPLIT_OF.get(row.id);
      const isSelected = !testPoint && row.id === selectedId;
      let opacity = 1, stroke = "#fff", strokeWidth = 2, extraClass = "";
      if (stage === 1 || stage === 2) {
        if (split === "test") { opacity = 0.28; }
        else if (split === "val") { stroke = "#17212b"; strokeWidth = 2.6; }
      }
      if (stage >= 3 && split === "test") {
        const correct = TEST_RESULT.correctIds.has(row.id);
        stroke = correct ? "#148a88" : "#c5504e";
        strokeWidth = 3.2;
        extraClass = correct ? "test-correct" : "test-wrong";
      } else if (stage >= 3 && split !== "test") {
        opacity = 0.35;
      }
      if (isSelected) { stroke = "#17212b"; strokeWidth = 3.4; }
      return `<circle class="point ${extraClass}" data-id="${row.id}" cx="${px(row.x)}" cy="${py(row.y)}" r="7.5" fill="${classColor(row.label)}" fill-opacity="${opacity}" stroke="${stroke}" stroke-width="${strokeWidth}"><title>#${row.id + 1} ${classLabel(row.label)} · ${xLabel.split(" ")[0]} ${row.x}, ${yLabel.split(" ")[0]} ${row.y} · ${split}</title></circle>`;
    }).join("");

    let testMarkup = "";
    if (stage === 4 && testPoint) {
      const x = px(testPoint.x), y = py(testPoint.y);
      const predicted = WINNER.predict(WINNER.finalModel, testPoint)[0].label;
      const color = classColor(predicted);
      testMarkup = `<path class="test-point" d="M ${x} ${y - 10} L ${x + 10} ${y} L ${x} ${y + 10} L ${x - 10} ${y} Z" fill="${color}" fill-opacity=".25" stroke="${color}"><title>New flower: ${testPoint.x}, ${testPoint.y} → ${classLabel(predicted)}</title></path>`;
    }

    svg.innerHTML = `${regions}${grid}${axes}${dots}${testMarkup}`;
    svg.querySelectorAll(".point").forEach(circle => circle.addEventListener("click", event => {
      event.stopPropagation();
      selectedId = Number(circle.dataset.id);
      testPoint = null;
      render();
    }));
    svg.onclick = stage === 4 ? event => {
      const pt = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
      const x = minX + ((pt.x - left) / width) * (maxX - minX);
      const y = minY + ((bottom - pt.y) / height) * (maxY - minY);
      testPoint = { x: Number(Math.max(minX, Math.min(maxX, x)).toFixed(1)), y: Number(Math.max(minY, Math.min(maxY, y)).toFixed(1)) };
      selectedId = null;
      render();
    } : null;
  }

  function renderCandidates() {
    const host = $("#candidate-cards");
    host.innerHTML = CANDIDATES.map(c => {
      const isWinner = c.key === WINNER.key;
      const isFocus = stage === 2 && c.key === focusKey;
      const badge = stage >= 3 && isWinner ? `<span class="tag tag-teal">Selected</span>` : "";
      return `<article class="candidate-card ${isFocus ? "candidate-focus" : ""}" data-key="${c.key}"><div class="candidate-head"><h3>${c.name}</h3>${badge}</div><p class="candidate-blurb">${c.blurb}</p><div class="candidate-stats"><div><span>Validation accuracy</span><strong>${(c.val.accuracy * 100).toFixed(1)}%</strong></div><div><span>Validation log loss</span><strong>${c.val.logLoss.toFixed(3)}</strong></div></div></article>`;
    }).join("");
    if (stage === 2) {
      host.querySelectorAll(".candidate-card").forEach(card => card.addEventListener("click", () => {
        focusKey = card.dataset.key;
        render();
      }));
    }
  }

  function renderStory() {
    const kickers = [
      ["Stage 0 · The data", "150 iris flowers, 3 species", "Each flower has 4 measurements, but this demo deliberately keeps only 2 — sepal length and sepal width — so the species overlap and the classifier makes visible, explainable mistakes. Petal length and width exist in the dataset but are set aside here."],
      ["Stage 1 · Split before modeling", "60% train · 20% validation · 20% test", `${SPLIT.train.length} train, ${SPLIT.val.length} validation, ${SPLIT.test.length} test — stratified so each split keeps the same species mix. The hatched, faded points are the test set: locked away until Stage 4.`],
      ["Stage 2 · Fit on train, compare on validation", "Three model families, one scoreboard", "Each candidate is fit on the 90 training flowers only, then scored on the 30 validation flowers. Click a card to preview that model's decision regions. The 30 test flowers are still untouched."],
      ["Stage 3 · Open the test set once", `Selected: ${WINNER.name}`, `${WINNER.name} wins on validation accuracy (ties would go to the lower log loss). It's refit on train + validation (${SPLIT.train.length + SPLIT.val.length} flowers), then evaluated on the 30 test flowers for the first and only time.`],
      ["Stage 4 · Try a new flower", "Predict from measurements alone", "Click anywhere on the chart to place a new flower at that sepal length/width, and see which species the final model predicts, with its confidence for each class."],
    ];
    const [kicker, title, copy] = kickers[stage];
    $("#stage-kicker").textContent = kicker;
    $("#stage-title").textContent = title;
    $("#stage-copy").textContent = copy;

    $("#candidate-section").hidden = stage !== 2;
    $("#test-section").hidden = stage < 3;
    $("#predict-section").hidden = stage !== 4;

    if (stage >= 3) {
      $("#test-accuracy").textContent = `${TEST_RESULT.correct}/${TEST_RESULT.total} (${(TEST_RESULT.accuracy * 100).toFixed(1)}%)`;
      $("#test-logloss").textContent = TEST_RESULT.logLoss.toFixed(3);
      $("#test-errors").textContent = TEST_ERROR_PAIRS.length
        ? `Every miss: ${[...new Set(TEST_ERROR_PAIRS)].map(pair => `${pair} (${TEST_ERROR_PAIRS.filter(p => p === pair).length})`).join(", ")}.`
        : "No misses on the test set.";
    }

    if (stage === 4) {
      const point = testPoint;
      if (!point) {
        $("#predict-result").innerHTML = `<p class="round-empty">Click the chart to place a new flower.</p>`;
      } else {
        const ranked = WINNER.predict(WINNER.finalModel, point);
        $("#predict-result").innerHTML = `<p class="window-summary">New flower: ${xLabel.split(" ")[0]} ${point.x} cm, ${yLabel.split(" ")[0]} ${point.y} cm</p>` +
          ranked.map(r => `<div class="score-row"><span class="prob-dot" style="background:${classColor(r.label)}"></span><span class="prob-name">${classLabel(r.label)}</span><div class="prob-track"><i style="width:${(r.prob * 100).toFixed(0)}%;background:${classColor(r.label)}"></i></div><b>${(r.prob * 100).toFixed(0)}%</b></div>`).join("");
      }
    }

    document.querySelectorAll(".step").forEach(btn => {
      const v = Number(btn.dataset.stage);
      btn.classList.toggle("active", v === stage);
      btn.classList.toggle("done", v < stage);
    });
    $("#previous").disabled = stage === 0;
    const next = $("#next");
    next.disabled = stage === 4;
    next.textContent = stage === 4 ? "End of walkthrough" : "Next →";
  }

  function render() {
    renderCandidates();
    renderStory();
    renderChart();
  }

  function goToStage(next) {
    stage = Math.max(0, Math.min(4, next));
    render();
  }

  $("#previous").addEventListener("click", () => goToStage(stage - 1));
  $("#next").addEventListener("click", () => goToStage(stage + 1));
  document.querySelectorAll(".step").forEach(btn => btn.addEventListener("click", () => goToStage(Number(btn.dataset.stage))));

  render();
})();
