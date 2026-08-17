(() => {
  const $ = selector => document.querySelector(selector);
  const makeDataset = rows => rows.map(([x, y, label], id) => ({ x, y, label, id }));

  const DATASETS = {
    two: {
      xLabel: "practice time (hours/week)", yLabel: "quiz score (/10)", xShort: "practice", yShort: "quiz", xRange: [1, 9], yRange: [1, 9],
      description: "Weekly practice time and quiz score, 2 classes. Click a point to inspect its class scores.",
      classInfo: { coral: { label: "Mastered", color: "#d66762" }, indigo: { label: "Needs review", color: "#5267bf" } },
      selectedId: 16,
      points: makeDataset([
        [1.1, 8.2, "coral"], [1.8, 6.9, "coral"], [2.6, 7.5, "coral"], [3.1, 5.9, "coral"], [3.8, 7.1, "coral"], [4.2, 5.2, "coral"], [4.7, 6.3, "coral"], [5.4, 4.9, "coral"],
        [5.2, 3.8, "indigo"], [5.9, 4.4, "indigo"], [6.3, 2.9, "indigo"], [6.9, 3.5, "indigo"], [7.4, 2.2, "indigo"], [7.7, 4.1, "indigo"], [8.3, 2.8, "indigo"], [8.8, 3.6, "indigo"], [5.6, 4.65, "indigo"],
      ]),
    },
    three: {
      xLabel: "height (cm)", yLabel: "weight (kg)", xShort: "height", yShort: "weight", xRange: [10, 90], yRange: [0, 40],
      description: "Cats, dogs, and birds, 3 classes. Birds span small to large heights, so the ensemble needs both features.",
      classInfo: { coral: { label: "Cat", color: "#d66762" }, teal: { label: "Dog", color: "#148a88" }, indigo: { label: "Bird", color: "#5267bf" } },
      selectedId: 17,
      points: makeDataset([
        [22, 3.2, "coral"], [25, 4.1, "coral"], [27, 3.7, "coral"], [29, 5.0, "coral"], [31, 4.6, "coral"], [34, 5.8, "coral"],
        [32, 9, "teal"], [38, 14, "teal"], [45, 18, "teal"], [52, 22, "teal"], [60, 28, "teal"], [68, 34, "teal"],
        [16, 0.2, "indigo"], [24, 0.5, "indigo"], [34, 1.1, "indigo"], [46, 2.6, "indigo"], [57, 4.5, "indigo"], [68, 7.2, "indigo"], [79, 10.5, "indigo"],
      ]),
    },
  };

  const ETA = 0.5, MAX_DEPTH = 2, ROUNDS = 3;

  const DESCRIPTIONS = [
    ["Round 0 · baseline", "Everyone starts with the same guess", "Before any tree, every class scores 0, so the softmax prediction is a flat tie between classes. The tinted background is that uniform starting guess.", "Start with equal scores for every class."],
    ["Round 1 · first correction", "Round 1 fits one tree per class", "Each class gets its own small tree. It studies the pseudo-residual — how far that class's current softmax probability is from the true label — and fits a correction. All trees in a round are added at once, scaled by η.", "Round 1 adds one small tree per class."],
    ["Round 2 · correct what's still missed", "Round 2 works only on the leftover error", "Round 1's trees are already part of the model. Round 2 never starts over: it fits the smaller residuals Round 1 left behind, again with one tree per class.", "Round 2 sharpens the decision boundary."],
    ["Round 3 · a final gentle correction", "Round 3 refines the boundary", "The final round makes modest refinements. The predicted class is the argmax of the softmax of the summed, η-scaled scores from all three rounds.", "All three rounds combine into the final decision surface."],
  ];

  let activeScenario = DATASETS.two;
  let DATA = activeScenario.points;
  let CLASS_INFO = activeScenario.classInfo;
  let selectedId = activeScenario.selectedId;
  let testPoint = null;
  let stage = 0;
  let MODEL = null;

  const featureLabel = feature => feature === "x" ? activeScenario.xLabel : activeScenario.yLabel;
  const shortFeatureLabel = feature => feature === "x" ? activeScenario.xShort : activeScenario.yShort;
  const classLabel = key => CLASS_INFO[key].label;
  const classColor = key => CLASS_INFO[key].color;
  const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
  const format = value => `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;

  function softmax(scores) {
    const max = Math.max(...scores);
    const exps = scores.map(score => Math.exp(score - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(exp => exp / sum);
  }

  function sse(values) {
    if (!values.length) return 0;
    const m = mean(values);
    return values.reduce((sum, value) => sum + (value - m) ** 2, 0);
  }

  function bestSplit(rows, residuals, features) {
    let best = null;
    features.forEach(feature => {
      const values = [...new Set(rows.map(row => row[feature]))].sort((a, b) => a - b);
      for (let index = 0; index < values.length - 1; index += 1) {
        const threshold = (values[index] + values[index + 1]) / 2;
        const left = rows.filter(row => row[feature] <= threshold);
        const right = rows.filter(row => row[feature] > threshold);
        if (!left.length || !right.length) continue;
        const score = sse(left.map(row => residuals.get(row.id))) + sse(right.map(row => residuals.get(row.id)));
        if (!best || score < best.score) best = { feature, threshold, left, right, score };
      }
    });
    return best;
  }

  function growTree(rows, residuals, depth, maxDepth) {
    const value = mean(rows.map(row => residuals.get(row.id)));
    if (depth >= maxDepth || rows.length < 2) return { value, count: rows.length };
    const split = bestSplit(rows, residuals, ["x", "y"]);
    if (!split) return { value, count: rows.length };
    return { ...split, value, count: rows.length, leftNode: growTree(split.left, residuals, depth + 1, maxDepth), rightNode: growTree(split.right, residuals, depth + 1, maxDepth) };
  }

  function predictTree(node, point) {
    if (!node.leftNode) return node.value;
    return predictTree(point[node.feature] <= node.threshold ? node.leftNode : node.rightNode, point);
  }

  function computeModel() {
    const classes = Object.keys(CLASS_INFO);
    const rounds = [];
    const rawAt = (point, uptoRound) => {
      const scores = classes.map(() => 0);
      for (let round = 0; round < uptoRound; round += 1) {
        rounds[round].trees.forEach((treeInfo, k) => { scores[k] += ETA * predictTree(treeInfo.tree, point); });
      }
      return scores;
    };
    for (let round = 0; round < ROUNDS; round += 1) {
      const probs = DATA.map(row => softmax(rawAt(row, round)));
      const trees = classes.map((cls, k) => {
        const residuals = new Map(DATA.map((row, index) => [row.id, (row.label === cls ? 1 : 0) - probs[index][k]]));
        return { cls, tree: growTree(DATA, residuals, 0, MAX_DEPTH) };
      });
      rounds.push({ trees });
    }
    return { classes, rounds, rawAt };
  }

  function probsAt(point, atStage) {
    return softmax(MODEL.rawAt(point, atStage));
  }

  function predictAt(point, atStage) {
    const probs = probsAt(point, atStage);
    let best = 0;
    probs.forEach((p, index) => { if (p > probs[best]) best = index; });
    return { label: MODEL.classes[best], probs };
  }

  function accuracyAt(atStage) {
    const correct = DATA.filter(row => predictAt(row, atStage).label === row.label).length;
    return correct / DATA.length;
  }

  function loglossAt(atStage) {
    const total = DATA.reduce((sum, row) => {
      const probs = probsAt(row, atStage);
      const k = MODEL.classes.indexOf(row.label);
      return sum - Math.log(Math.max(probs[k], 1e-9));
    }, 0);
    return total / DATA.length;
  }

  function lerpColor(a, b, t) { return [0, 1, 2].map(i => Math.round(a[i] + (b[i] - a[i]) * t)); }
  function hexToRgb(hex) { const n = parseInt(hex.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function classValueColor(v, hex) {
    const t = Math.max(-1, Math.min(1, v));
    const white = [255, 255, 255], away = [163, 173, 181];
    const [r, g, b] = t < 0 ? lerpColor(white, away, -t) : lerpColor(white, hexToRgb(hex), t);
    return `rgb(${r},${g},${b})`;
  }

  function selectedPoint() { return testPoint || DATA[selectedId]; }

  function applyConfiguration() {
    activeScenario = DATASETS[$("#class-scenario").value];
    DATA = activeScenario.points;
    CLASS_INFO = activeScenario.classInfo;
    selectedId = activeScenario.selectedId;
    testPoint = null;
    stage = 0;
    MODEL = computeModel();
  }

  function renderLegend() {
    const classes = MODEL.classes;
    $("#feature-class-count").textContent = `2 features · ${classes.length} classes`;
    $("#training-description").textContent = activeScenario.description;
    $("#chart-legend").innerHTML = classes.map(key => `<span><i class="dot" style="background:${classColor(key)}"></i>${classLabel(key)}</span>`).join("");
  }

  function renderChart() {
    const svg = $("#chart");
    const left = 55, bottom = 310, width = 350, height = 250, top = 60;
    const [minX, maxX] = activeScenario.xRange;
    const [minY, maxY] = activeScenario.yRange;
    const px = value => left + ((value - minX) / (maxX - minX)) * width;
    const py = value => bottom - ((value - minY) / (maxY - minY)) * height;
    const cols = 26, rows = 16;
    let regions = "";
    for (let cx = 0; cx < cols; cx += 1) {
      for (let cy = 0; cy < rows; cy += 1) {
        const fx = minX + (maxX - minX) * (cx + 0.5) / cols;
        const fy = minY + (maxY - minY) * (cy + 0.5) / rows;
        const cls = predictAt({ x: fx, y: fy }, stage).label;
        const x0 = left + (cx / cols) * width, y0 = bottom - ((cy + 1) / rows) * height;
        regions += `<rect x="${x0.toFixed(1)}" y="${y0.toFixed(1)}" width="${(width / cols + 0.6).toFixed(1)}" height="${(height / rows + 0.6).toFixed(1)}" class="decision-region" fill="${classColor(cls)}"/>`;
      }
    }
    let grid = "";
    for (let fraction = 0.2; fraction < 1; fraction += 0.2) {
      const x = minX + (maxX - minX) * fraction, y = minY + (maxY - minY) * fraction;
      grid += `<line class="plot-grid" x1="${px(x)}" y1="${top}" x2="${px(x)}" y2="${bottom}"/><line class="plot-grid" x1="${left}" y1="${py(y)}" x2="${left + width}" y2="${py(y)}"/>`;
    }
    const axes = `<line class="plot-axis" x1="${left}" y1="${bottom}" x2="${left + width}" y2="${bottom}"/><line class="plot-axis" x1="${left}" y1="${bottom}" x2="${left}" y2="${top}"/><text class="axis-label" x="${left + width / 2}" y="345">${activeScenario.xLabel}</text><text class="axis-label" x="18" y="${(top + bottom) / 2}" transform="rotate(-90 18 ${(top + bottom) / 2})">${activeScenario.yLabel}</text>`;
    const dots = DATA.map(point => `<circle class="point" data-id="${point.id}" cx="${px(point.x)}" cy="${py(point.y)}" r="9" fill="${classColor(point.label)}" stroke="${point.id === selectedId ? "#17212b" : "#fff"}" stroke-width="${point.id === selectedId ? 3.5 : 2}"><title>Example ${point.id + 1}: ${activeScenario.xShort} ${point.x}, ${activeScenario.yShort} ${point.y}, ${classLabel(point.label)}</title></circle>`).join("");
    let testMarkup = "";
    if (testPoint) {
      const x = px(testPoint.x), y = py(testPoint.y);
      const prediction = predictAt(testPoint, stage);
      const color = classColor(prediction.label);
      testMarkup = `<path class="test-point" d="M ${x} ${y - 10} L ${x + 10} ${y} L ${x} ${y + 10} L ${x - 10} ${y} Z" fill="${color}" fill-opacity=".25" stroke="${color}"><title>Test data: ${activeScenario.xShort} ${testPoint.x}, ${activeScenario.yShort} ${testPoint.y}; predicts ${classLabel(prediction.label)}</title></path><text class="test-point-label" x="${x + 12}" y="${y - 10}" fill="${color}">test → ${classLabel(prediction.label)}</text>`;
    }
    svg.innerHTML = `${regions}${grid}${axes}${dots}${testMarkup}`;
    svg.querySelectorAll(".point").forEach(circle => circle.addEventListener("click", event => {
      event.stopPropagation();
      selectedId = Number(circle.dataset.id);
      testPoint = null;
      render();
    }));
    svg.onclick = event => {
      const box = svg.getBoundingClientRect();
      const svgX = ((event.clientX - box.left) / box.width) * 470;
      const svgY = ((event.clientY - box.top) / box.height) * 360;
      const x = minX + ((svgX - left) / width) * (maxX - minX);
      const y = minY + ((bottom - svgY) / height) * (maxY - minY);
      testPoint = { x: Number(Math.max(minX, Math.min(maxX, x)).toFixed(1)), y: Number(Math.max(minY, Math.min(maxY, y)).toFixed(1)), isTest: true };
      selectedId = null;
      render();
    };
  }

  function treePlot(node, number, cls) {
    const left = 22, bottom = 150, width = 166, height = 112;
    const [minX, maxX] = activeScenario.xRange;
    const [minY, maxY] = activeScenario.yRange;
    const px = value => left + ((value - minX) / (maxX - minX)) * width;
    const py = value => bottom - ((value - minY) / (maxY - minY)) * height;
    const regionMarkup = [], valueMarkup = [], splitMarkup = [];
    const fillRegions = (n, bounds) => {
      if (!n.leftNode) {
        const x0 = px(bounds.minX), x1 = px(bounds.maxX), y0 = py(bounds.maxY), y1 = py(bounds.minY);
        regionMarkup.push(`<rect class="decision-region value-region" x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}" fill="${classValueColor(ETA * n.value, classColor(cls))}"/>`);
        valueMarkup.push(`<text class="leaf-value" x="${(x0 + x1) / 2}" y="${(y0 + y1) / 2 + 3}" text-anchor="middle">${format(ETA * n.value)}</text>`);
        return;
      }
      if (n.feature === "x") { fillRegions(n.leftNode, { ...bounds, maxX: n.threshold }); fillRegions(n.rightNode, { ...bounds, minX: n.threshold }); }
      else { fillRegions(n.leftNode, { ...bounds, maxY: n.threshold }); fillRegions(n.rightNode, { ...bounds, minY: n.threshold }); }
    };
    const drawSplits = (n, bounds, depth = 0) => {
      if (!n.leftNode) return;
      const color = depth === 0 ? "#2b67c5" : "#148a88";
      if (n.feature === "x") {
        const x = px(n.threshold);
        splitMarkup.push(`<line class="tree-boundary" x1="${x}" y1="${py(bounds.maxY)}" x2="${x}" y2="${py(bounds.minY)}" stroke="${color}"/><text class="tree-rule" x="${x + 3}" y="${py(bounds.maxY) + 10}" fill="${color}">${shortFeatureLabel("x")}≤${n.threshold.toFixed(1)}</text>`);
        drawSplits(n.leftNode, { ...bounds, maxX: n.threshold }, depth + 1); drawSplits(n.rightNode, { ...bounds, minX: n.threshold }, depth + 1);
      } else {
        const y = py(n.threshold);
        splitMarkup.push(`<line class="tree-boundary" x1="${px(bounds.minX)}" y1="${y}" x2="${px(bounds.maxX)}" y2="${y}" stroke="${color}"/><text class="tree-rule" x="${px(bounds.minX) + 3}" y="${y - 3}" fill="${color}">${shortFeatureLabel("y")}≤${n.threshold.toFixed(1)}</text>`);
        drawSplits(n.leftNode, { ...bounds, maxY: n.threshold }, depth + 1); drawSplits(n.rightNode, { ...bounds, minY: n.threshold }, depth + 1);
      }
    };
    fillRegions(node, { minX, maxX, minY, maxY });
    drawSplits(node, { minX, maxX, minY, maxY });
    const points = DATA.map(row => `<circle class="tree-point" cx="${px(row.x)}" cy="${py(row.y)}" r="3.6" fill="${row.label === cls ? classColor(cls) : "#fff"}" stroke="${classColor(row.label)}"/>`).join("");
    return `<svg class="tree-plot" viewBox="0 0 210 170" role="img" aria-label="Leaf value regions for the ${classLabel(cls)} tree in round ${number}"><rect x="${left}" y="${py(maxY)}" width="${width}" height="${height}" class="tree-plot-bg"/>${regionMarkup.join("")}<line class="tree-axis" x1="${left}" y1="${bottom}" x2="${left + width}" y2="${bottom}"/><line class="tree-axis" x1="${left}" y1="${bottom}" x2="${left}" y2="${py(maxY)}"/>${valueMarkup.join("")}${splitMarkup.join("")}${points}<text class="tree-axis-label" x="88" y="166">${shortFeatureLabel("x")}</text><text class="tree-axis-label" x="10" y="95" transform="rotate(-90 10 95)">${shortFeatureLabel("y")}</text></svg>`;
  }

  function treeSignature(node) {
    if (!node.leftNode) return "leaf";
    return `${node.feature}<=${node.threshold.toFixed(3)}|${treeSignature(node.leftNode)}|${treeSignature(node.rightNode)}`;
  }

  function renderTrees() {
    const container = $("#tree-cards");
    const pills = MODEL.rounds.map((round, i) => {
      const roundNumber = i + 1;
      const state = roundNumber > stage ? "future" : roundNumber === stage && stage < ROUNDS ? "active" : "done";
      const label = roundNumber > stage ? "waiting" : roundNumber === stage && stage < ROUNDS ? "learning now" : "added";
      return `<span class="round-pill ${state}">Round ${roundNumber}<small>${label}</small></span>`;
    }).join("");

    if (stage === 0) {
      container.innerHTML = `<div class="round-pills">${pills}</div><p class="round-empty">No trees yet — every class scores 0, so there is nothing to compare. Click “Add Round 1 →” to fit the first tree per class.</p>`;
      return;
    }

    const round = MODEL.rounds[stage - 1];
    const prevRound = stage >= 2 ? MODEL.rounds[stage - 2] : null;
    const cards = round.trees.map(({ cls, tree }, k) => {
      const rootSplit = tree.leftNode ? `${shortFeatureLabel(tree.feature)} ≤ ${tree.threshold.toFixed(1)}? (root)` : "single leaf";
      let changeNote;
      if (!prevRound) changeNote = `First tree fit for ${classLabel(cls)} — nothing to compare yet.`;
      else if (treeSignature(tree) === treeSignature(prevRound.trees[k].tree)) changeNote = `Same split shape as Round ${stage - 1} — only the leaf values shrank.`;
      else changeNote = `Split moved from Round ${stage - 1} to fit the new residuals.`;
      return `<article class="tree-card active"><div class="tree-label"><span style="color:${classColor(cls)}">${classLabel(cls)}</span><small>η × leaf value</small></div><div class="tree-split">${rootSplit}</div>${treePlot(tree, stage, cls)}<p class="change-note">${changeNote}</p></article>`;
    }).join("");
    container.innerHTML = `<div class="round-pills">${pills}</div><div class="tree-cards">${cards}</div>`;
  }

  function renderStory() {
    const copy = DESCRIPTIONS[stage];
    $("#stage-kicker").textContent = copy[0];
    $("#stage-title").textContent = copy[1];
    $("#stage-copy").textContent = copy[2];
    $("#chart-caption").textContent = copy[3];
    $("#formula").textContent = stage === 0 ? "F(x) = 0" : `F(x) = ${Array.from({ length: stage }, (_, i) => `ηR${i + 1}(x)`).join(" + ")}`;
    $("#accuracy").textContent = `${(accuracyAt(stage) * 100).toFixed(0)}%`;
    $("#logloss").textContent = loglossAt(stage).toFixed(3);

    const point = selectedPoint();
    const isTest = Boolean(point.isTest);
    $("#selected-description").textContent = isTest
      ? `◇ Test data: ${activeScenario.xShort} = ${point.x}, ${activeScenario.yShort} = ${point.y} · no training label`
      : `Example ${point.id + 1}: ${activeScenario.xShort} = ${point.x}, ${activeScenario.yShort} = ${point.y} · training class: ${classLabel(point.label)}`;
    const prediction = predictAt(point, stage);
    $("#prediction-tag").textContent = `predicts ${classLabel(prediction.label)}`;
    $("#residual-title").textContent = stage < ROUNDS ? "Class scores for this point" : "Final class scores";
    $("#residual-note").textContent = stage < ROUNDS ? "probability now · residual the next round will fit" : "probability after all rounds";
    $("#residual-list").innerHTML = MODEL.classes.map((cls, k) => {
      const prob = prediction.probs[k];
      const residual = (point.label === cls ? 1 : 0) - prob;
      const residualNote = stage < ROUNDS && !isTest ? `<small>${format(residual)} residual</small>` : "";
      return `<div class="score-row"><span class="prob-dot" style="background:${classColor(cls)}"></span><span class="prob-name">${classLabel(cls)}</span><div class="prob-track"><i style="width:${(prob * 100).toFixed(0)}%;background:${classColor(cls)}"></i></div><b>${(prob * 100).toFixed(0)}%</b>${residualNote}</div>`;
    }).join("");

    document.querySelectorAll(".step").forEach(button => { const value = Number(button.dataset.stage); button.classList.toggle("active", value === stage); button.classList.toggle("done", value < stage); });
    $("#previous").disabled = stage === 0;
    const next = $("#next"); next.disabled = stage === ROUNDS; next.textContent = stage === ROUNDS ? "All rounds added" : `Add Round ${stage + 1} →`;
  }

  function render() {
    renderLegend();
    renderStory();
    renderChart();
    renderTrees();
  }

  function goToStage(newStage) {
    stage = Math.max(0, Math.min(ROUNDS, newStage));
    render();
    const section = $(".tree-section");
    const rect = section.getBoundingClientRect();
    const fullyVisible = rect.top >= 0 && rect.bottom <= window.innerHeight;
    if (!fullyVisible) section.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  $("#previous").addEventListener("click", () => goToStage(stage - 1));
  $("#next").addEventListener("click", () => goToStage(stage + 1));
  document.querySelectorAll(".step").forEach(button => button.addEventListener("click", () => goToStage(Number(button.dataset.stage))));
  $("#class-scenario").addEventListener("change", () => { applyConfiguration(); render(); });

  applyConfiguration();
  render();
})();
