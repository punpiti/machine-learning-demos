const INPUT_SIZE = 100;
let hiddenSize = 2;
const $ = selector => document.querySelector(selector);

const SEGMENTS = ["abcedf", "bc", "abdeg", "abcdg", "bcfg", "acdfg", "acdefg", "abc", "abcdefg", "abcdfg"];

function makeDigit(digit) {
  const pixels = Array(INPUT_SIZE).fill(0);
  const enabled = SEGMENTS[digit];
  const set = (row, column) => { pixels[row * 10 + column] = 1; };
  for (let column = 2; column <= 7; column++) {
    if (enabled.includes("a")) set(1, column);
    if (enabled.includes("g")) set(5, column);
    if (enabled.includes("d")) set(8, column);
  }
  for (let row = 2; row <= 4; row++) {
    if (enabled.includes("f")) set(row, 1);
    if (enabled.includes("b")) set(row, 8);
  }
  for (let row = 6; row <= 7; row++) {
    if (enabled.includes("e")) set(row, 1);
    if (enabled.includes("c")) set(row, 8);
  }
  return pixels;
}

const ICONS = [
  { name: "circle", glyph: "○", rows: ["..######..", ".##....##.", "##......##", "##......##", "##......##", "##......##", "##......##", "##......##", ".##....##.", "..######.."] },
  { name: "square", glyph: "□", rows: [".########.", ".#......#.", ".#......#.", ".#......#.", ".#......#.", ".#......#.", ".#......#.", ".#......#.", ".#......#.", ".########."] },
  { name: "triangle", glyph: "△", rows: ["....##....", "...####...", "...#..#...", "..#....#..", "..#....#..", ".#......#.", ".#......#.", "#........#", "#........#", "##########"] },
  { name: "diamond", glyph: "◇", rows: ["....##....", "...#..#...", "..#....#..", ".#......#.", "#........#", "#........#", ".#......#.", "..#....#..", "...#..#...", "....##...."] },
  { name: "plus", glyph: "+", rows: ["....##....", "....##....", "....##....", "....##....", "##########", "##########", "....##....", "....##....", "....##....", "....##...."] },
  { name: "cross", glyph: "×", rows: ["##......##", ".##....##.", "..##..##..", "...####...", "....##....", "....##....", "...####...", "..##..##..", ".##....##.", "##......##"] },
  { name: "up arrow", glyph: "↑", rows: ["....##....", "...####...", "..######..", ".###..###.", "....##....", "....##....", "....##....", "....##....", "....##....", "....##...."] },
  { name: "down arrow", glyph: "↓", rows: ["....##....", "....##....", "....##....", "....##....", "....##....", "....##....", ".###..###.", "..######..", "...####...", "....##...."] },
  { name: "heart", glyph: "♥", rows: ["..........", ".###..###.", "##########", "##########", ".########.", "..######..", "...####...", "....##....", "....##....", ".........."] },
  { name: "star", glyph: "★", rows: ["....##....", "....##....", "##..##..##", ".########.", "..######..", "...####...", "..##..##..", ".##....##.", "##......##", ".........."] },
];

function makeIcon(icon) {
  return icon.rows.flatMap(row => [...row].map(pixel => pixel === "#" ? 1 : 0));
}

const DATASETS = {
  digits: {
    title: "digits 0–9",
    samples: Array.from({ length: 10 }, (_, digit) => makeDigit(digit)),
    names: Array.from({ length: 10 }, (_, digit) => `digit ${digit}`),
    glyphs: Array.from({ length: 10 }, (_, digit) => String(digit)),
  },
  icons: {
    title: "10 bitmap icons",
    samples: ICONS.map(makeIcon),
    names: ICONS.map(icon => icon.name),
    glyphs: ICONS.map(icon => icon.glyph),
  },
};

let datasetKey = "digits";
let trainingSet = DATASETS[datasetKey].samples;
let currentInput = [...trainingSet[3]];
let selectedDigit = 3;
let epochs = 0;
let drawMode = "draw";
let pointerDrawing = false;
let augmentationEnabled = false;
let lastAugmentedInput = null;
let randomState = 29;
let encoderWeights, encoderBias, decoderWeights, decoderBias;

function random() {
  randomState = (randomState * 16807) % 2147483647;
  return (randomState - 1) / 2147483646;
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, value))));
}

function resetModel() {
  randomState = 29;
  encoderWeights = Array.from({ length: INPUT_SIZE }, () =>
    Array.from({ length: hiddenSize }, () => (random() - 0.5) * 0.3));
  encoderBias = Array(hiddenSize).fill(0);
  decoderWeights = Array.from({ length: hiddenSize }, () =>
    Array.from({ length: INPUT_SIZE }, () => (random() - 0.5) * 0.3));
  decoderBias = Array(INPUT_SIZE).fill(0);
  epochs = 0;
  lastAugmentedInput = null;
}

function forward(input) {
  const hiddenPre = Array.from({ length: hiddenSize }, (_, hidden) =>
    encoderBias[hidden] + input.reduce((sum, value, pixel) =>
      sum + value * encoderWeights[pixel][hidden], 0));
  const hidden = hiddenPre.map(sigmoid);
  const outputPre = Array.from({ length: INPUT_SIZE }, (_, pixel) =>
    decoderBias[pixel] + hidden.reduce((sum, value, unit) =>
      sum + value * decoderWeights[unit][pixel], 0));
  return { hiddenPre, hidden, outputPre, output: outputPre.map(sigmoid) };
}

function mse(input, output) {
  return input.reduce((sum, value, index) => sum + (value - output[index]) ** 2, 0) / INPUT_SIZE;
}

function weightedBinaryCrossEntropy(target, output) {
  const epsilon = 1e-7;
  return target.reduce((sum, value, pixel) => {
    const probability = Math.max(epsilon, Math.min(1 - epsilon, output[pixel]));
    const weight = value ? 5 : 1;
    return sum - weight * (value * Math.log(probability) + (1 - value) * Math.log(1 - probability));
  }, 0) / INPUT_SIZE;
}

function rankTemplates(reconstruction) {
  return trainingSet
    .map((template, digit) => ({ digit, distance: weightedBinaryCrossEntropy(template, reconstruction) }))
    .sort((left, right) => left.distance - right.distance);
}

function augment(sample) {
  const shifted = Array(INPUT_SIZE).fill(0);
  const rowShift = Math.floor(random() * 3) - 1;
  const columnShift = Math.floor(random() * 7) - 4;
  for (let row = 0; row < 10; row++) {
    for (let column = 0; column < 10; column++) {
      const newRow = row + rowShift;
      const newColumn = column + columnShift;
      if (newRow >= 0 && newRow < 10 && newColumn >= 0 && newColumn < 10) {
        shifted[newRow * 10 + newColumn] = sample[row * 10 + column];
      }
    }
  }
  return shifted.map(value => random() < 0.015 ? 1 - value : value);
}

function trainSample(input, target = input, learningRate = 1.0) {
  const state = forward(input);
  const outputDelta = state.output.map((value, pixel) => {
    const weight = target[pixel] ? 5 : 1;
    return weight * (value - target[pixel]) / INPUT_SIZE;
  });
  const hiddenDelta = state.hidden.map((value, hidden) =>
    outputDelta.reduce((sum, delta, pixel) =>
      sum + delta * decoderWeights[hidden][pixel], 0) * value * (1 - value));

  for (let hidden = 0; hidden < hiddenSize; hidden++) {
    for (let pixel = 0; pixel < INPUT_SIZE; pixel++) {
      decoderWeights[hidden][pixel] -= learningRate * state.hidden[hidden] * outputDelta[pixel];
    }
  }
  for (let pixel = 0; pixel < INPUT_SIZE; pixel++) decoderBias[pixel] -= learningRate * outputDelta[pixel];
  for (let pixel = 0; pixel < INPUT_SIZE; pixel++) {
    for (let hidden = 0; hidden < hiddenSize; hidden++) {
      encoderWeights[pixel][hidden] -= learningRate * input[pixel] * hiddenDelta[hidden];
    }
  }
  for (let hidden = 0; hidden < hiddenSize; hidden++) encoderBias[hidden] -= learningRate * hiddenDelta[hidden];
}

function train(rounds) {
  $("#train-button").disabled = true;
  $("#app-status").textContent = "Training…";
  setTimeout(() => {
    for (let epoch = 0; epoch < rounds; epoch++) {
      trainingSet.forEach(target => {
        const input = augmentationEnabled ? augment(target) : target;
        lastAugmentedInput = augmentationEnabled ? input : null;
        trainSample(input, target);
      });
    }
    epochs += rounds;
    $("#train-button").disabled = false;
    $("#app-status").textContent = "Training complete";
    render();
  }, 20);
}

function gray(value) {
  const channel = Math.round(255 - Math.max(0, Math.min(1, value)) * 235);
  return `rgb(${channel},${channel},${channel})`;
}

function signedColor(value, scale) {
  const strength = Math.min(1, Math.abs(value) / scale);
  const pale = Math.round(255 - strength * 190);
  return value >= 0 ? `rgb(255,${pale},${pale})` : `rgb(${pale},${pale},255)`;
}

function renderBitmap(host, values, editable = false, signed = false) {
  host.innerHTML = "";
  const scale = Math.max(0.001, ...values.map(Math.abs));
  values.forEach((value, index) => {
    const pixel = document.createElement("button");
    pixel.type = "button";
    pixel.className = "pixel";
    pixel.style.background = signed ? signedColor(value, scale) : gray(value);
    pixel.title = `row ${Math.floor(index / 10) + 1}, column ${index % 10 + 1}: ${value.toFixed(3)}`;
    if (editable) {
      const applyBrush = () => {
        currentInput[index] = drawMode === "draw" ? 1 : 0;
        selectedDigit = null;
        pixel.style.background = gray(currentInput[index]);
      };
      pixel.addEventListener("pointerdown", event => {
        event.preventDefault();
        pointerDrawing = true;
        applyBrush();
      });
      pixel.addEventListener("pointerenter", () => {
        if (pointerDrawing) applyBrush();
      });
    }
    host.append(pixel);
  });
}

function renderGallery() {
  const gallery = $("#training-gallery");
  const dataset = DATASETS[datasetKey];
  gallery.innerHTML = "";
  trainingSet.forEach((sample, digit) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `sample${selectedDigit === digit ? " selected" : ""}`;
    button.title = dataset.names[digit];
    const mini = document.createElement("span");
    mini.className = "mini-grid";
    sample.forEach(value => {
      const pixel = document.createElement("i");
      pixel.className = "mini-pixel";
      pixel.style.background = value ? "#17212b" : "#fff";
      mini.append(pixel);
    });
    const label = document.createElement("b");
    label.textContent = dataset.glyphs[digit];
    button.append(mini, label);
    button.addEventListener("click", () => {
      selectedDigit = digit;
      currentInput = [...sample];
      render();
    });
    gallery.append(button);
  });
}

function renderAugmentationPreview() {
  const panel = $("#augmentation-preview");
  panel.hidden = !augmentationEnabled;
  $("#training-mode").textContent = augmentationEnabled
    ? "Augmented mode: shifted/noisy input → canonical target."
    : "Plain mode: input = target.";
  $("#augmentation-button").textContent = augmentationEnabled ? "3. Augmentation ON" : "3. Augmentation OFF";
  $("#augmentation-button").classList.toggle("active", augmentationEnabled);
  if (!augmentationEnabled) return;
  const sample = lastAugmentedInput || augment(trainingSet[selectedDigit ?? 3]);
  const bitmap = $("#augmentation-bitmap");
  bitmap.innerHTML = "";
  sample.forEach(value => {
    const pixel = document.createElement("i");
    pixel.className = "preview-pixel";
    pixel.style.background = value ? "#17212b" : "#fff";
    bitmap.append(pixel);
  });
}

function renderNetwork(state) {
  const svg = $("#network");
  const inputX = 90, hiddenX = 360, outputX = 630;
  const hiddenY = Array.from({ length: hiddenSize }, (_, hidden) =>
    hiddenSize === 1 ? 215 : 75 + hidden * (280 / (hiddenSize - 1)));
  let markup = `<text x="${inputX}" y="20">100 input pixels</text><text x="${hiddenX}" y="20">${hiddenSize} hidden units</text><text x="${outputX}" y="20">100 output pixels</text>`;
  for (let pixel = 0; pixel < INPUT_SIZE; pixel++) {
    const y = 35 + pixel * 3.75;
    for (let hidden = 0; hidden < hiddenSize; hidden++) {
      markup += `<line x1="${inputX + 8}" y1="${y}" x2="${hiddenX - 34}" y2="${hiddenY[hidden]}"/>`;
      markup += `<line x1="${hiddenX + 34}" y1="${hiddenY[hidden]}" x2="${outputX - 8}" y2="${y}"/>`;
    }
    markup += `<circle cx="${inputX}" cy="${y}" r="2" fill="#5d8ed4"/><circle cx="${outputX}" cy="${y}" r="2" fill="#d79b2b"/>`;
  }
  hiddenY.forEach((y, hidden) => {
    markup += `<circle cx="${hiddenX}" cy="${y}" r="31" fill="#e6f7f4" stroke="#148a88"/>`;
    markup += `<text x="${hiddenX}" y="${y - 3}">h${hidden + 1}</text><text x="${hiddenX}" y="${y + 14}">${state.hidden[hidden].toFixed(3)}</text>`;
  });
  svg.innerHTML = markup;
}

function renderFeatureCards(state) {
  const host = $("#feature-cards");
  host.innerHTML = "";
  for (let hidden = 0; hidden < hiddenSize; hidden++) {
    const card = document.createElement("article");
    card.className = "card feature-card";
    card.innerHTML = `<h2>Hidden feature h${hidden + 1}</h2><div class="bitmap weight-map"></div><div class="calculation">z${hidden + 1} = x·w${hidden + 1} + b${hidden + 1} = ${state.hiddenPre[hidden].toFixed(3)}<br>h${hidden + 1} = sigmoid(z${hidden + 1}) = ${state.hidden[hidden].toFixed(3)}</div>`;
    renderBitmap(card.querySelector(".weight-map"), encoderWeights.map(row => row[hidden]), false, true);
    host.append(card);
  }
}

function render() {
  const state = forward(currentInput);
  renderGallery();
  renderAugmentationPreview();
  renderBitmap($("#input-bitmap"), currentInput, true);
  renderBitmap($("#output-bitmap"), state.output);
  renderNetwork(state);
  renderFeatureCards(state);

  $("#epoch-value").textContent = epochs;
  $("#architecture").textContent = `100–${hiddenSize}–100`;
  $("#training-set-title").textContent = `Training set: ${DATASETS[datasetKey].title}`;
  $("#similarity-title").textContent = `Similarity histogram: ${DATASETS[datasetKey].title}`;
  $("#input-loss").textContent = mse(currentInput, state.output).toFixed(4);
  const classRanking = rankTemplates(state.output);
  const temperature = 0.2;
  const minimumDistance = classRanking[0].distance;
  const similarityScores = classRanking.map((item) => ({
    ...item,
    score: Math.exp(-(item.distance - minimumDistance) / temperature),
  }));
  const scoreTotal = similarityScores.reduce((sum, item) => sum + item.score, 0);
  const probabilities = similarityScores
    .map((item) => ({ ...item, probability: item.score / scoreTotal }))
    .sort((left, right) => left.digit - right.digit);
  const winningDigit = classRanking[0].digit;
  const winningProbability = probabilities.find((item) => item.digit === winningDigit).probability;
  renderBitmap($("#closest-bitmap"), trainingSet[winningDigit]);
  $("#closest-class").textContent = datasetKey === "digits"
    ? DATASETS[datasetKey].glyphs[winningDigit]
    : `${DATASETS[datasetKey].glyphs[winningDigit]} ${DATASETS[datasetKey].names[winningDigit]}`;
  $("#closest-probability").textContent = `${(winningProbability * 100).toFixed(1)}%`;
  $("#class-ranking").innerHTML = probabilities
    .map((item) => `<div class="probability-column${item.digit === winningDigit ? " winner" : ""}" title="${DATASETS[datasetKey].names[item.digit]}: ${(item.probability * 100).toFixed(1)}% normalized similarity">
      <div class="probability-track"><div class="probability-bar" style="height:${Math.max(1, item.probability * 100)}%"></div></div>
      <span class="probability-value">${(item.probability * 100).toFixed(0)}%</span>
      <span class="probability-label">${DATASETS[datasetKey].glyphs[item.digit]}</span>
    </div>`)
    .join("");
  const trainingLoss = trainingSet.reduce((sum, sample) => sum + weightedBinaryCrossEntropy(sample, forward(sample).output), 0) / trainingSet.length;
  $("#training-loss").textContent = trainingLoss.toFixed(4);
  const outputPixel = Number($("#output-pixel").value || 0);
  const row = Math.floor(outputPixel / 10) + 1;
  const column = outputPixel % 10 + 1;
  const decoderTerms = state.hidden.map((_, hidden) => `${decoderWeights[hidden][outputPixel].toFixed(3)}·h${hidden + 1}`).join(" + ");
  $("#output-equation").innerHTML = `pixel (${row},${column})<br>weighted sum = ${decoderTerms} + bias ${decoderBias[outputPixel].toFixed(3)}<br>x̂ = sigmoid(weighted sum) = ${state.output[outputPixel].toFixed(3)}`;
}

for (let pixel = 0; pixel < INPUT_SIZE; pixel++) {
  const option = document.createElement("option");
  option.value = pixel;
  option.textContent = `(${Math.floor(pixel / 10) + 1},${pixel % 10 + 1})`;
  $("#output-pixel").append(option);
}

$("#output-pixel").addEventListener("change", render);
$("#train-button").addEventListener("click", () => train(100));
$("#reset-button").addEventListener("click", () => { resetModel(); render(); });
$("#hidden-size").addEventListener("change", event => {
  hiddenSize = Number(event.target.value);
  resetModel();
  render();
});
$("#dataset-choice").addEventListener("change", event => {
  datasetKey = event.target.value;
  trainingSet = DATASETS[datasetKey].samples;
  selectedDigit = 0;
  currentInput = [...trainingSet[0]];
  resetModel();
  $("#app-status").textContent = "Training set changed; model reset";
  render();
});
$("#augmentation-button").addEventListener("click", () => {
  augmentationEnabled = !augmentationEnabled;
  resetModel();
  $("#app-status").textContent = "Mode changed; model reset";
  render();
});
$("#draw-mode").addEventListener("click", () => {
  drawMode = "draw";
  $("#draw-mode").classList.add("active");
  $("#erase-mode").classList.remove("active");
});
$("#erase-mode").addEventListener("click", () => {
  drawMode = "erase";
  $("#erase-mode").classList.add("active");
  $("#draw-mode").classList.remove("active");
});
$("#clear-input").addEventListener("click", () => {
  currentInput = Array(INPUT_SIZE).fill(0);
  selectedDigit = null;
  render();
});
document.addEventListener("pointerup", () => {
  if (pointerDrawing) {
    pointerDrawing = false;
    render();
  }
});

resetModel();
render();
