const $ = selector => document.querySelector(selector);
const makeDataset = rows => rows.map(([x, y, label], id) => ({ x, y, label, id }));
const DATASETS = {
  two: {
    xLabel: "practice time (hours/week)", yLabel: "quiz score (/10)", xShort: "practice", yShort: "quiz", xRange: [1, 9], yRange: [1, 9],
    description: "Weekly practice time and quiz score. Click a point to inspect the forest’s vote.",
    classInfo: { coral: { label: "Mastered", color: "#d66762" }, indigo: { label: "Needs review", color: "#5267bf" } },
    selectedId: 16,
    points: makeDataset([
      [1.1, 8.2, "coral"], [1.8, 6.9, "coral"], [2.6, 7.5, "coral"], [3.1, 5.9, "coral"], [3.8, 7.1, "coral"], [4.2, 5.2, "coral"], [4.7, 6.3, "coral"], [5.4, 4.9, "coral"],
      [5.2, 3.8, "indigo"], [5.9, 4.4, "indigo"], [6.3, 2.9, "indigo"], [6.9, 3.5, "indigo"], [7.4, 2.2, "indigo"], [7.7, 4.1, "indigo"], [8.3, 2.8, "indigo"], [8.8, 3.6, "indigo"], [5.6, 4.65, "indigo"],
    ]),
  },
  three: {
    xLabel: "height (cm)", yLabel: "weight (kg)", xShort: "height", yShort: "weight", xRange: [10, 90], yRange: [0, 40],
    description: "Cats and dogs form compact groups, while birds span small to large heights and overlap their height ranges. The tree must use both height and weight rules.",
    classInfo: { coral: { label: "Cat", color: "#d66762" }, teal: { label: "Dog", color: "#148a88" }, indigo: { label: "Bird", color: "#5267bf" } },
    selectedId: 17,
    points: makeDataset([
      [22, 3.2, "coral"], [25, 4.1, "coral"], [27, 3.7, "coral"], [29, 5.0, "coral"], [31, 4.6, "coral"], [34, 5.8, "coral"],
      [32, 9, "teal"], [38, 14, "teal"], [45, 18, "teal"], [52, 22, "teal"], [60, 28, "teal"], [68, 34, "teal"],
      [16, 0.2, "indigo"], [24, 0.5, "indigo"], [34, 1.1, "indigo"], [46, 2.6, "indigo"], [57, 4.5, "indigo"], [68, 7.2, "indigo"], [79, 10.5, "indigo"],
    ]),
  },
};

let forest = [];
let activeScenario = DATASETS.two;
let DATA = activeScenario.points;
let CLASS_INFO = activeScenario.classInfo;
let selectedId = DATASETS.two.selectedId;
let testPoint = null;
let seed = 20260817;
let forestGeneration = 0;
let mode = "forest";
const featureLabel = feature => feature === "x" ? activeScenario.xLabel : activeScenario.yLabel;
const shortFeatureLabel = feature => feature === "x" ? activeScenario.xShort : activeScenario.yShort;
const classLabel = label => CLASS_INFO[label].label;
const classColor = label => CLASS_INFO[label].color;
const MODE_INFO = {
  tree: {
    label: "Single Decision Tree",
    bootstrap: "OFF · uses all training rows",
    features: "OFF · considers both features",
    note: "One tree sees every training row and considers both features at every split.",
    takeaway: "This is the baseline: one tree, the full training data, and every feature available at every split. It gives one deterministic set of rules for this dataset.",
  },
  bagging: {
    label: "Bagging",
    bootstrap: "ON · each tree gets a bootstrap sample",
    features: "OFF · every split considers both features",
    note: "Each tree receives a different bootstrap sample, but every split may consider both features.",
    takeaway: "Bagging changes the rows seen by each tree through bootstrap sampling. The trees differ because their training samples differ, then their predictions are combined by voting.",
  },
  forest: {
    label: "Random Forest",
    bootstrap: "ON · each tree gets a bootstrap sample",
    features: "ON · every split considers 1 of 2 features",
    note: "Each tree gets a bootstrap sample and each split considers one randomly selected feature.",
    takeaway: "Random Forest combines both sources of variation: bootstrap rows make the training data differ, and random feature selection makes the split candidates differ. Voting then reduces the instability of an individual tree.",
  },
};

function random() {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 4294967296;
}

function majority(rows) {
  const counts = classCounts(rows.map(row => row.label));
  return Object.keys(counts).sort((left, right) => counts[right] - counts[left] || left.localeCompare(right))[0];
}

function gini(rows) {
  if (!rows.length) return 0;
  return 1 - Object.values(classCounts(rows.map(row => row.label))).reduce((sum, count) => sum + (count / rows.length) ** 2, 0);
}

function classCounts(labels) {
  return labels.reduce((counts, label) => ({ ...counts, [label]: (counts[label] || 0) + 1 }), {});
}

function bestSplit(rows, features) {
  let best = null;
  features.forEach(feature => {
    const values = [...new Set(rows.map(row => row[feature]))].sort((a, b) => a - b);
    for (let index = 0; index < values.length - 1; index++) {
      const threshold = (values[index] + values[index + 1]) / 2;
      const left = rows.filter(row => row[feature] <= threshold);
      const right = rows.filter(row => row[feature] > threshold);
      if (!left.length || !right.length) continue;
      const score = (left.length * gini(left) + right.length * gini(right)) / rows.length;
      if (!best || score < best.score) best = { feature, threshold, left, right, score };
    }
  });
  return best;
}

function grow(rows, depth, maxDepth) {
  const label = majority(rows);
  if (depth >= maxDepth || gini(rows) === 0 || rows.length < 2) return { label, count: rows.length };
  const candidates = mode === "forest" ? [random() < 0.5 ? "x" : "y"] : ["x", "y"];
  const split = bestSplit(rows, candidates);
  if (!split) return { label, count: rows.length };
  return { ...split, candidates, label, count: rows.length, leftNode: grow(split.left, depth + 1, maxDepth), rightNode: grow(split.right, depth + 1, maxDepth) };
}

function predict(tree, point) {
  if (!tree.leftNode) return tree.label;
  return predict(point[tree.feature] <= tree.threshold ? tree.leftNode : tree.rightNode, point);
}

function nodeCount(tree) {
  return tree.leftNode ? 1 + nodeCount(tree.leftNode) + nodeCount(tree.rightNode) : 1;
}

function bootstrap() {
  return Array.from({ length: DATA.length }, () => DATA[Math.floor(random() * DATA.length)]);
}

function applyConfiguration() {
  const scenario = DATASETS[$("#class-scenario").value];
  activeScenario = scenario;
  DATA = scenario.points;
  CLASS_INFO = scenario.classInfo;
  selectedId = scenario.selectedId;
  testPoint = null;
}

function selectedPoint() {
  return testPoint || DATA[selectedId];
}

function configurationReady() {
  $("#forest-status").textContent = "Configuration ready — choose Single Tree, Bagging, or Random Forest.";
}

function growForest(action = "Forest ready") {
  const requestedCount = Number($("#tree-count").value);
  const count = mode === "tree" ? 1 : requestedCount;
  const maxDepth = Number($("#max-depth").value);
  forest = Array.from({ length: count }, () => {
    const sample = mode === "tree" ? DATA : bootstrap();
    return { sample, root: grow(sample, 0, maxDepth), usesBootstrap: mode !== "tree" };
  });
  forestGeneration += 1;
  $("#tree-count").disabled = mode === "tree";
  $("#forest-status").textContent = `${action}: ${MODE_INFO[mode].label} ${forestGeneration} · ${count} tree${count === 1 ? "" : "s"} · depth ${maxDepth}`;
  $("#mechanism-title").textContent = mode === "tree" ? "What does this tree use?" : "Why do the trees differ?";
  $("#mechanism-note").textContent = MODE_INFO[mode].note;
  $("#takeaway-title").textContent = MODE_INFO[mode].label;
  $("#takeaway-copy").textContent = MODE_INFO[mode].takeaway;
  $("#model-difference").innerHTML = `<span><b>Bootstrap rows</b>${MODE_INFO[mode].bootstrap}</span><span><b>Random feature selection</b>${MODE_INFO[mode].features}</span>`;
  document.querySelectorAll(".mode-button").forEach(button => button.classList.toggle("active", button.dataset.mode === mode));
  render();
}

function votesFor(point) {
  const votes = forest.map(tree => predict(tree.root, point));
  const counts = classCounts(votes);
  const prediction = Object.keys(counts).sort((left, right) => counts[right] - counts[left] || left.localeCompare(right))[0];
  return { votes, counts, prediction };
}

function renderScatter() {
  const svg = $("#scatterplot");
  const left = 55, bottom = 310, width = 350, height = 250;
  const [minX, maxX] = activeScenario.xRange;
  const [minY, maxY] = activeScenario.yRange;
  const px = value => left + ((value - minX) / (maxX - minX)) * width;
  const py = value => bottom - ((value - minY) / (maxY - minY)) * height;
  let markup = "";
  for (let fraction = 0.2; fraction < 1; fraction += 0.2) {
    const x = minX + (maxX - minX) * fraction;
    const y = minY + (maxY - minY) * fraction;
    markup += `<line class="plot-grid" x1="${px(x)}" y1="60" x2="${px(x)}" y2="${bottom}"/><line class="plot-grid" x1="${left}" y1="${py(y)}" x2="${left + width}" y2="${py(y)}"/>`;
  }
  markup += `<line class="plot-axis" x1="${left}" y1="${bottom}" x2="${left + width}" y2="${bottom}"/><line class="plot-axis" x1="${left}" y1="${bottom}" x2="${left}" y2="60"/><text class="axis-label" x="225" y="345">${activeScenario.xLabel}</text><text class="axis-label" x="18" y="190" transform="rotate(-90 18 190)">${activeScenario.yLabel}</text>`;
  DATA.forEach(point => {
    markup += `<circle class="point ${point.label}${point.id === selectedId ? " selected" : ""}" data-id="${point.id}" cx="${px(point.x)}" cy="${py(point.y)}" r="9" fill="${classColor(point.label)}"><title>Example ${point.id + 1}: ${activeScenario.xShort} ${point.x}, ${activeScenario.yShort} ${point.y}, ${classLabel(point.label)}</title></circle>`;
  });
  if (testPoint) {
    const x = px(testPoint.x), y = py(testPoint.y);
    const prediction = votesFor(testPoint).prediction;
    const color = classColor(prediction);
    markup += `<path class="test-point" d="M ${x} ${y - 10} L ${x + 10} ${y} L ${x} ${y + 10} L ${x - 10} ${y} Z" fill="${color}" fill-opacity=".25" stroke="${color}"><title>Test data: ${activeScenario.xShort} ${testPoint.x}, ${activeScenario.yShort} ${testPoint.y}; predicts ${classLabel(prediction)}</title></path><text class="test-point-label" x="${x + 12}" y="${y - 10}" fill="${color}">test → ${classLabel(prediction)}</text>`;
  }
  svg.innerHTML = markup;
  svg.querySelectorAll(".point").forEach(circle => circle.addEventListener("click", event => {
    event.stopPropagation();
    selectedId = Number(circle.dataset.id);
    testPoint = null;
    render();
  }));
  svg.addEventListener("click", event => {
    const box = svg.getBoundingClientRect();
    const svgX = ((event.clientX - box.left) / box.width) * 470;
    const svgY = ((event.clientY - box.top) / box.height) * 360;
    const x = minX + ((svgX - left) / width) * (maxX - minX);
    const y = minY + ((bottom - svgY) / height) * (maxY - minY);
    testPoint = { x: Number(Math.max(minX, Math.min(maxX, x)).toFixed(1)), y: Number(Math.max(minY, Math.min(maxY, y)).toFixed(1)), isTest: true };
    selectedId = null;
    render();
  });
}

function renderLegend() {
  const classes = [...new Set(DATA.map(point => point.label))];
  $("#feature-class-count").textContent = `2 features · ${classes.length} classes`;
  $("#training-description").textContent = activeScenario.description;
  $("#class-legend").innerHTML = classes.map(label => `<span><i class="dot" style="background:${classColor(label)}"></i>${classLabel(label)}</span>`).join("");
}

function renderVotes() {
  const point = selectedPoint();
  const isTest = Boolean(point.isTest);
  const result = votesFor(point);
  const total = forest.length;
  $("#selected-description").textContent = isTest
    ? `◇ Test data: ${activeScenario.xShort} = ${point.x}, ${activeScenario.yShort} = ${point.y} · no training label`
    : `Example ${point.id + 1}: ${activeScenario.xShort} = ${point.x}, ${activeScenario.yShort} = ${point.y} · training class: ${classLabel(point.label)}`;
  $("#prediction-tag").textContent = `predicts ${classLabel(result.prediction)}`;
  $("#vote-result").innerHTML = `<div><span>The forest predicts</span><strong>${classLabel(result.prediction)}</strong><span>${isTest ? "Prediction for test data — no known label" : result.prediction === point.label ? "Matches the training label" : "Disagrees with the training label"}</span></div>`;
  const classes = [...new Set(DATA.map(row => row.label))];
  $("#vote-bars").innerHTML = classes.map(label => {
    const count = result.counts[label] || 0;
    return `<div><span>${classLabel(label)}</span><div class="vote-track"><i style="width:${(count / total) * 100}%;background:${classColor(label)}"></i></div><b>${count}/${total}</b></div>`;
  }).join("");
  $("#agreement").textContent = `${result.counts[result.prediction]}/${total} trees agree`;
  $("#forest-version").textContent = `Forest ${forestGeneration}`;
  return result;
}

function renderTrees(result) {
  const point = selectedPoint();
  $("#tree-cards").innerHTML = forest.map((tree, index) => {
    const root = tree.root;
    const vote = result.votes[index];
    const correct = point.isTest ? null : vote === point.label;
    const uniqueRows = new Set(tree.sample.map(row => row.id)).size;
    const draw = tree.sample.map(row => row.id + 1).join(", ");
    const oob = DATA.filter(row => !tree.sample.some(sample => sample.id === row.id)).map(row => row.id + 1);
    const split = root.leftNode
      ? `${featureLabel(root.feature)} ${point[root.feature] <= root.threshold ? "≤" : ">"} ${root.threshold.toFixed(1)} at root`
      : `pure bootstrap sample → ${classLabel(root.label)}`;
    const sampleNote = tree.usesBootstrap
      ? `<div class="bootstrap-draw"><b>1. Bootstrap draw</b><br>sampled rows: [${draw}]<br>OOB rows: [${oob.join(", ") || "none"}]</div>`
      : `<div class="bootstrap-draw"><b>1. Training rows</b><br>all rows: [${DATA.map(row => row.id + 1).join(", ")}]<br>OOB rows: none</div>`;
    const decisionState = correct === null ? "test" : correct ? "correct" : "incorrect";
    return `<article class="tree-card decision-${decisionState}"><h3>Tree ${index + 1}</h3><p>${tree.usesBootstrap ? `Bootstrap: ${uniqueRows} unique rows of ${DATA.length}` : "Full training data"}<br>${nodeCount(root)} nodes grown</p>${treePlot(tree.root, point, index + 1, vote)}${sampleNote}<div class="feature-draw"><b>2. Features considered</b><br>${splitSteps(root).join("<br>")}</div><div class="split">${split}</div><span class="tree-vote ${vote}">votes ${classLabel(vote)} · ${point.isTest ? "test prediction" : correct ? "correct" : "incorrect"}</span></article>`;
  }).join("");
}

function splitSteps(node, steps = []) {
  if (!node.leftNode) return steps;
  steps.push(`node ${steps.length + 1}: [${node.candidates.map(shortFeatureLabel).join(", ")}] → ${shortFeatureLabel(node.feature)} ≤ ${node.threshold.toFixed(1)}`);
  splitSteps(node.leftNode, steps);
  splitSteps(node.rightNode, steps);
  return steps;
}

function treePlot(tree, selected, number, prediction) {
  const left = 22, bottom = 150, width = 166, height = 112;
  const [minX, maxX] = activeScenario.xRange;
  const [minY, maxY] = activeScenario.yRange;
  const px = value => left + ((value - minX) / (maxX - minX)) * width;
  const py = value => bottom - ((value - minY) / (maxY - minY)) * height;
  const splitMarkup = [];
  const regionMarkup = [];
  const fillRegions = (node, bounds) => {
    if (!node.leftNode) {
      regionMarkup.push(`<rect class="decision-region" x="${px(bounds.minX)}" y="${py(bounds.maxY)}" width="${px(bounds.maxX) - px(bounds.minX)}" height="${py(bounds.minY) - py(bounds.maxY)}" fill="${classColor(node.label)}"/>`);
      return;
    }
    if (node.feature === "x") {
      fillRegions(node.leftNode, { ...bounds, maxX: node.threshold });
      fillRegions(node.rightNode, { ...bounds, minX: node.threshold });
    } else {
      fillRegions(node.leftNode, { ...bounds, maxY: node.threshold });
      fillRegions(node.rightNode, { ...bounds, minY: node.threshold });
    }
  };
  const drawSplits = (node, bounds, depth = 0) => {
    if (!node.leftNode) return;
    const color = depth === 0 ? "#2b67c5" : "#148a88";
    if (node.feature === "x") {
      const x = px(node.threshold);
      splitMarkup.push(`<line class="tree-boundary" x1="${x}" y1="${py(bounds.maxY)}" x2="${x}" y2="${py(bounds.minY)}" stroke="${color}"/><text class="tree-rule" x="${x + 3}" y="${py(bounds.maxY) + 10}" fill="${color}">x≤${node.threshold.toFixed(1)}</text>`);
      drawSplits(node.leftNode, { ...bounds, maxX: node.threshold }, depth + 1);
      drawSplits(node.rightNode, { ...bounds, minX: node.threshold }, depth + 1);
    } else {
      const y = py(node.threshold);
      splitMarkup.push(`<line class="tree-boundary" x1="${px(bounds.minX)}" y1="${y}" x2="${px(bounds.maxX)}" y2="${y}" stroke="${color}"/><text class="tree-rule" x="${px(bounds.minX) + 3}" y="${y - 3}" fill="${color}">y≤${node.threshold.toFixed(1)}</text>`);
      drawSplits(node.leftNode, { ...bounds, maxY: node.threshold }, depth + 1);
      drawSplits(node.rightNode, { ...bounds, minY: node.threshold }, depth + 1);
    }
  };
  fillRegions(tree, { minX, maxX, minY, maxY });
  drawSplits(tree, { minX, maxX, minY, maxY });
  const points = DATA.map(row => `<circle class="tree-point" cx="${px(row.x)}" cy="${py(row.y)}" r="3.6" fill="${classColor(row.label)}"/>`).join("");
  const correct = selected.isTest ? null : prediction === selected.label;
  const selectedRing = `<circle class="tree-selected ${correct === null ? "test" : correct ? "correct" : "incorrect"}" cx="${px(selected.x)}" cy="${py(selected.y)}" r="6"><title>${correct === null ? "Test-data" : correct ? "Correct" : "Incorrect"} prediction: ${classLabel(prediction)}</title></circle>`;
  return `<svg class="tree-plot" viewBox="0 0 210 170" role="img" aria-label="Decision boundaries for tree ${number}; blue is the root split and teal is a later split"><rect x="${left}" y="${py(maxY)}" width="${width}" height="${height}" class="tree-plot-bg"/>${regionMarkup.join("")}<line class="tree-axis" x1="${left}" y1="${bottom}" x2="${left + width}" y2="${bottom}"/><line class="tree-axis" x1="${left}" y1="${bottom}" x2="${left}" y2="${py(maxY)}"/>${splitMarkup.join("")}${points}${selectedRing}<text class="tree-axis-label" x="88" y="166">${activeScenario.xShort}</text><text class="tree-axis-label" x="10" y="95" transform="rotate(-90 10 95)">${activeScenario.yShort}</text></svg>`;
}

function render() {
  renderLegend();
  renderScatter();
  const result = renderVotes();
  renderTrees(result);
}

$("#regenerate").addEventListener("click", () => { applyConfiguration(); seed += 97; growForest("Current model regrown"); });
$("#tree-count").addEventListener("change", configurationReady);
$("#max-depth").addEventListener("change", configurationReady);
$("#class-scenario").addEventListener("change", event => {
  applyConfiguration();
  seed += 97;
  growForest(`${event.target.value === "three" ? "3-class" : "2-class"} dataset selected`);
});
document.querySelectorAll(".mode-button").forEach(button => button.addEventListener("click", () => {
  mode = button.dataset.mode;
  applyConfiguration();
  seed += 97;
  growForest(`${MODE_INFO[mode].label} selected`);
}));
growForest();
