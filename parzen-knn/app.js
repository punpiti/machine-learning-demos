(() => {
  const $ = selector => document.querySelector(selector);
  const makeDataset = rows => rows.map(([x, y, label], id) => ({ x, y, label, id }));

  const DATASETS = {
    two: {
      xLabel: "practice time (hours/week)", yLabel: "quiz score (/10)", xShort: "practice", yShort: "quiz", xRange: [1, 9], yRange: [1, 9],
      description: "Weekly practice time and quiz score, 2 classes.",
      classInfo: { coral: { label: "Mastered", color: "#d66762" }, indigo: { label: "Needs review", color: "#5267bf" } },
      selectedId: 16,
      points: makeDataset([
        [1.1, 8.2, "coral"], [1.8, 6.9, "coral"], [2.6, 7.5, "coral"], [3.1, 5.9, "coral"], [3.8, 7.1, "coral"], [4.2, 5.2, "coral"], [4.7, 6.3, "coral"], [5.4, 4.9, "coral"],
        [5.2, 3.8, "indigo"], [5.9, 4.4, "indigo"], [6.3, 2.9, "indigo"], [6.9, 3.5, "indigo"], [7.4, 2.2, "indigo"], [7.7, 4.1, "indigo"], [8.3, 2.8, "indigo"], [8.8, 3.6, "indigo"], [5.6, 4.65, "indigo"],
      ]),
    },
    three: {
      xLabel: "height (cm)", yLabel: "weight (kg)", xShort: "height", yShort: "weight", xRange: [10, 90], yRange: [0, 40],
      description: "Cats, dogs, and birds, 3 classes. Birds span small to large heights.",
      classInfo: { coral: { label: "Cat", color: "#d66762" }, teal: { label: "Dog", color: "#148a88" }, indigo: { label: "Bird", color: "#5267bf" } },
      selectedId: 17,
      points: makeDataset([
        [22, 3.2, "coral"], [25, 4.1, "coral"], [27, 3.7, "coral"], [29, 5.0, "coral"], [31, 4.6, "coral"], [34, 5.8, "coral"],
        [32, 9, "teal"], [38, 14, "teal"], [45, 18, "teal"], [52, 22, "teal"], [60, 28, "teal"], [68, 34, "teal"],
        [16, 0.2, "indigo"], [24, 0.5, "indigo"], [34, 1.1, "indigo"], [46, 2.6, "indigo"], [57, 4.5, "indigo"], [68, 7.2, "indigo"], [79, 10.5, "indigo"],
      ]),
    },
  };

  const H_MIN = 0.02, H_MAX = 0.9, H_DEFAULT = 0.22;
  const K_MIN = 1, K_MAX = 15, K_DEFAULT = 3;

  let activeScenario = DATASETS.two;
  let DATA = activeScenario.points;
  let CLASS_INFO = activeScenario.classInfo;
  let selectedId = activeScenario.selectedId;
  let testPoint = null;
  let mode = "parzen";
  let h = H_DEFAULT;
  let k = K_DEFAULT;
  let looCurve = [];

  const classLabel = key => CLASS_INFO[key].label;
  const classColor = key => CLASS_INFO[key].color;
  const classes = () => Object.keys(CLASS_INFO);

  function normPoint(p) {
    const [minX, maxX] = activeScenario.xRange, [minY, maxY] = activeScenario.yRange;
    return { nx: (p.x - minX) / (maxX - minX), ny: (p.y - minY) / (maxY - minY) };
  }
  function distNorm(a, b) {
    const na = normPoint(a), nb = normPoint(b);
    return Math.hypot(na.nx - nb.nx, na.ny - nb.ny);
  }

  function emptyCounts() { const c = {}; classes().forEach(cls => c[cls] = 0); return c; }
  function majority(counts) {
    let best = null;
    classes().forEach(cls => { if (best === null || counts[cls] > counts[best]) best = cls; });
    return best;
  }

  function parzenResult(point, excludeId, radius) {
    const counts = emptyCounts();
    const inside = [];
    let total = 0;
    DATA.forEach(row => {
      if (row.id === excludeId) return;
      if (distNorm(point, row) <= radius) { counts[row.label] += 1; total += 1; inside.push(row); }
    });
    return { label: total === 0 ? null : majority(counts), counts, total, inside, radius };
  }

  function knnResult(point, excludeId, neighborCount) {
    const sorted = DATA.filter(row => row.id !== excludeId).map(row => ({ row, d: distNorm(point, row) })).sort((a, b) => a.d - b.d);
    const chosen = sorted.slice(0, neighborCount);
    const counts = emptyCounts();
    chosen.forEach(n => counts[n.row.label] += 1);
    const radius = chosen.length ? chosen[chosen.length - 1].d : 0;
    return { label: majority(counts), counts, total: chosen.length, inside: chosen.map(n => n.row), radius };
  }

  function classify(point, excludeId) {
    return mode === "parzen" ? parzenResult(point, excludeId, h) : knnResult(point, excludeId, k);
  }

  function looAccuracyAt(param) {
    let correct = 0;
    DATA.forEach(row => {
      const result = mode === "parzen" ? parzenResult(row, row.id, param) : knnResult(row, row.id, param);
      if (result.label === row.label) correct += 1;
    });
    return correct / DATA.length;
  }

  function computeLooCurve() {
    if (mode === "parzen") {
      const steps = 30;
      return Array.from({ length: steps }, (_, i) => {
        const p = H_MIN + (H_MAX - H_MIN) * (i / (steps - 1));
        return { param: p, accuracy: looAccuracyAt(p) };
      });
    }
    const maxK = Math.min(K_MAX, DATA.length - 1);
    return Array.from({ length: maxK }, (_, i) => ({ param: i + 1, accuracy: looAccuracyAt(i + 1) }));
  }

  function bestParam() {
    const maxAcc = Math.max(...looCurve.map(p => p.accuracy));
    const tied = looCurve.filter(p => p.accuracy === maxAcc);
    return tied[Math.floor(tied.length / 2)].param;
  }

  function selectedPoint() { return testPoint || DATA[selectedId]; }
  function selectedExcludeId() { return testPoint ? null : selectedId; }

  function applyConfiguration() {
    activeScenario = DATASETS[$("#class-scenario").value];
    DATA = activeScenario.points;
    CLASS_INFO = activeScenario.classInfo;
    selectedId = activeScenario.selectedId;
    testPoint = null;
    looCurve = computeLooCurve();
  }

  function setMode(newMode) {
    mode = newMode;
    document.querySelectorAll(".mode-button").forEach(btn => btn.classList.toggle("active", btn.dataset.mode === mode));
    $("#param-row-parzen").hidden = mode !== "parzen";
    $("#param-row-knn").hidden = mode !== "knn";
    looCurve = computeLooCurve();
    render();
  }

  function renderLegend() {
    $("#feature-class-count").textContent = `2 features · ${classes().length} classes`;
    $("#training-description").textContent = activeScenario.description;
    $("#chart-legend").innerHTML = classes().map(key => `<span><i class="dot" style="background:${classColor(key)}"></i>${classLabel(key)}</span>`).join("");
  }

  function renderFormula() {
    $("#formula-fixed").textContent = mode === "parzen" ? "V fixed (window width h)" : "k fixed (number of neighbors)";
    $("#formula-free").textContent = mode === "parzen" ? "k counted inside" : "V grows to capture k";
  }

  const PLOT = { left: 55, bottom: 310, width: 350, height: 250, top: 60 };

  function renderChart() {
    const svg = $("#chart");
    const { left, bottom, width, height, top } = PLOT;
    const [minX, maxX] = activeScenario.xRange;
    const [minY, maxY] = activeScenario.yRange;
    const px = value => left + ((value - minX) / (maxX - minX)) * width;
    const py = value => bottom - ((value - minY) / (maxY - minY)) * height;

    const point = selectedPoint();
    const excludeId = selectedExcludeId();
    const result = classify(point, excludeId);
    const insideIds = new Set(result.inside.map(row => row.id));

    const cols = 26, rows = 16;
    let regions = "";
    for (let cx = 0; cx < cols; cx += 1) {
      for (let cy = 0; cy < rows; cy += 1) {
        const fx = minX + (maxX - minX) * (cx + 0.5) / cols;
        const fy = minY + (maxY - minY) * (cy + 0.5) / rows;
        const cellResult = classify({ x: fx, y: fy }, null);
        const x0 = left + (cx / cols) * width, y0 = bottom - ((cy + 1) / rows) * height;
        const fill = cellResult.label ? classColor(cellResult.label) : "url(#undef-pattern)";
        regions += `<rect x="${x0.toFixed(1)}" y="${y0.toFixed(1)}" width="${(width / cols + 0.6).toFixed(1)}" height="${(height / rows + 0.6).toFixed(1)}" class="decision-region" fill="${fill}"/>`;
      }
    }

    let grid = "";
    for (let fraction = 0.2; fraction < 1; fraction += 0.2) {
      const x = minX + (maxX - minX) * fraction, y = minY + (maxY - minY) * fraction;
      grid += `<line class="plot-grid" x1="${px(x)}" y1="${top}" x2="${px(x)}" y2="${bottom}"/><line class="plot-grid" x1="${left}" y1="${py(y)}" x2="${left + width}" y2="${py(y)}"/>`;
    }
    const axes = `<line class="plot-axis" x1="${left}" y1="${bottom}" x2="${left + width}" y2="${bottom}"/><line class="plot-axis" x1="${left}" y1="${bottom}" x2="${left}" y2="${top}"/><text class="axis-label" x="${left + width / 2}" y="345">${activeScenario.xLabel}</text><text class="axis-label" x="18" y="${(top + bottom) / 2}" transform="rotate(-90 18 ${(top + bottom) / 2})">${activeScenario.yLabel}</text>`;

    const dots = DATA.map(row => {
      const inWindow = insideIds.has(row.id);
      const isSelected = !testPoint && row.id === selectedId;
      return `<circle class="point" data-id="${row.id}" cx="${px(row.x)}" cy="${py(row.y)}" r="${inWindow ? 9.5 : 8}" fill="${classColor(row.label)}" stroke="${isSelected ? "#17212b" : inWindow ? "#17212b" : "#fff"}" stroke-width="${isSelected ? 3.5 : inWindow ? 2.6 : 2}" stroke-opacity="${inWindow || isSelected ? 1 : 0.9}"><title>Example ${row.id + 1}: ${activeScenario.xShort} ${row.x}, ${activeScenario.yShort} ${row.y}, ${classLabel(row.label)}</title></circle>`;
    }).join("");

    const rx = result.radius * width, ry = result.radius * height;
    const windowMarkup = `<ellipse class="window-ellipse" cx="${px(point.x)}" cy="${py(point.y)}" rx="${rx.toFixed(1)}" ry="${ry.toFixed(1)}"/>`;

    let testMarkup = "";
    if (testPoint) {
      const x = px(testPoint.x), y = py(testPoint.y);
      const color = result.label ? classColor(result.label) : "#8a97a1";
      testMarkup = `<path class="test-point" d="M ${x} ${y - 10} L ${x + 10} ${y} L ${x} ${y + 10} L ${x - 10} ${y} Z" fill="${color}" fill-opacity=".25" stroke="${color}"><title>Test data: ${activeScenario.xShort} ${testPoint.x}, ${activeScenario.yShort} ${testPoint.y}</title></path><text class="test-point-label" x="${x + 12}" y="${y - 10}" fill="${color}">test${result.label ? ` → ${classLabel(result.label)}` : " → undefined"}</text>`;
    }

    svg.innerHTML = `<defs><pattern id="undef-pattern" width="7" height="7" patternTransform="rotate(45)" patternUnits="userSpaceOnUse"><rect width="7" height="7" fill="#eef1f2"/><line x1="0" y1="0" x2="0" y2="7" stroke="#c7ced2" stroke-width="2"/></pattern></defs>${regions}${grid}${axes}${dots}${windowMarkup}${testMarkup}`;
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

  function renderStory() {
    const point = selectedPoint();
    const excludeId = selectedExcludeId();
    const result = classify(point, excludeId);
    const isTest = Boolean(testPoint);

    $("#selected-description").textContent = isTest
      ? `◇ Test data: ${activeScenario.xShort} = ${point.x}, ${activeScenario.yShort} = ${point.y} · no training label`
      : `Example ${point.id + 1}: ${activeScenario.xShort} = ${point.x}, ${activeScenario.yShort} = ${point.y} · training class: ${classLabel(point.label)} (excluded from its own window)`;
    $("#prediction-tag").textContent = result.label ? `predicts ${classLabel(result.label)}` : "no points in window";

    $("#window-summary").textContent = mode === "parzen"
      ? `Window radius h = ${h.toFixed(2)} (normalized) · ${result.total} training point${result.total === 1 ? "" : "s"} found inside`
      : `k = ${k} nearest neighbors · window grew to radius ≈ ${result.radius.toFixed(2)} (normalized) to capture them`;

    if (result.total === 0) {
      $("#score-list").innerHTML = `<p class="round-empty">No training points fall inside this window, so there is nothing to vote on. Try a larger h.</p>`;
    } else {
      $("#score-list").innerHTML = classes().map(cls => {
        const count = result.counts[cls] || 0;
        const pct = (count / result.total) * 100;
        return `<div class="score-row"><span class="prob-dot" style="background:${classColor(cls)}"></span><span class="prob-name">${classLabel(cls)}</span><div class="prob-track"><i style="width:${pct.toFixed(0)}%;background:${classColor(cls)}"></i></div><b>${count}/${result.total}</b></div>`;
      }).join("");
    }

    const loo = looAccuracyAt(mode === "parzen" ? h : k);
    $("#loo-current").textContent = `${(loo * 100).toFixed(0)}%`;
  }

  function renderLooCurve() {
    const svg = $("#loo-chart");
    const left = 46, right = 610, top = 14, bottom = 118, width = right - left, height = bottom - top;
    const params = looCurve.map(p => p.param);
    const minP = params[0], maxP = params[params.length - 1];
    const px = p => left + ((p - minP) / (maxP - minP)) * width;
    const py = acc => bottom - acc * height;

    const path = looCurve.map((p, i) => `${i ? "L" : "M"}${px(p.param).toFixed(1)},${py(p.accuracy).toFixed(1)}`).join(" ");
    const area = `${path} L${px(maxP).toFixed(1)},${bottom} L${px(minP).toFixed(1)},${bottom} Z`;

    const gridLines = [0, 0.25, 0.5, 0.75, 1].map(v => `<line class="plot-grid" x1="${left}" y1="${py(v)}" x2="${right}" y2="${py(v)}"/><text class="axis-label loo-y" x="${left - 6}" y="${py(v) + 3}" text-anchor="end">${(v * 100).toFixed(0)}%</text>`).join("");

    const currentParam = mode === "parzen" ? h : k;
    const currentX = px(currentParam);
    const currentAcc = looAccuracyAt(currentParam);
    const best = bestParam();

    const currentMarker = `<line class="loo-current-line" x1="${currentX.toFixed(1)}" y1="${top}" x2="${currentX.toFixed(1)}" y2="${bottom}"/><circle class="loo-current-dot" cx="${currentX.toFixed(1)}" cy="${py(currentAcc).toFixed(1)}" r="5"/>`;
    const bestMarker = `<circle class="loo-best-dot" cx="${px(best).toFixed(1)}" cy="${py(looAccuracyAt(best)).toFixed(1)}" r="4"/>`;

    const xTicks = mode === "parzen"
      ? [H_MIN, 0.25, 0.5, 0.75, H_MAX].map(v => `<text class="axis-label" x="${px(v)}" y="${bottom + 16}" text-anchor="middle">${v.toFixed(2)}</text>`).join("")
      : looCurve.filter((_, i) => i % 2 === 0).map(p => `<text class="axis-label" x="${px(p.param)}" y="${bottom + 16}" text-anchor="middle">${p.param}</text>`).join("");

    svg.innerHTML = `${gridLines}<path d="${area}" class="loo-area"/><path d="${path}" class="loo-line"/>${bestMarker}${currentMarker}${xTicks}<text class="axis-label" x="${(left + right) / 2}" y="${bottom + 32}" text-anchor="middle">${mode === "parzen" ? "window width h (normalized)" : "k (neighbors)"}</text>`;
  }

  function updateControls() {
    $("#h-slider").value = h;
    $("#h-value").textContent = h.toFixed(2);
    $("#k-slider").value = k;
    $("#k-value").textContent = k;
  }

  function render() {
    renderLegend();
    renderFormula();
    renderChart();
    renderStory();
    renderLooCurve();
    updateControls();
  }

  document.querySelectorAll(".mode-button").forEach(btn => btn.addEventListener("click", () => setMode(btn.dataset.mode)));
  $("#h-slider").addEventListener("input", event => { h = Number(event.target.value); render(); });
  $("#k-slider").addEventListener("input", event => { k = Number(event.target.value); render(); });
  $("#use-best").addEventListener("click", () => {
    if (mode === "parzen") h = bestParam(); else k = bestParam();
    render();
  });
  $("#class-scenario").addEventListener("change", () => { applyConfiguration(); render(); });

  applyConfiguration();
  setMode("parzen");
})();
