(() => {
  const $ = selector => document.querySelector(selector);

  function seededRandom(seed) {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  }

  function generateDays(seed) {
    const rand = seededRandom(seed);
    const N = 48;
    const days = [];
    for (let i = 0; i < N; i += 1) {
      const season = rand() < 0.5 ? "rainy" : "dry";
      const seasonBase = season === "rainy" ? 0.42 : 0.28;
      const cloud = Math.min(100, Math.max(0, (season === "rainy" ? 55 : 38) + (rand() - 0.5) * 80));
      const humidity = Math.min(100, Math.max(0, (season === "rainy" ? 60 : 45) + (rand() - 0.5) * 65));
      const score = seasonBase + (cloud / 100 - 0.5) * 0.7 + (humidity / 100 - 0.5) * 0.55 + (rand() - 0.5) * 0.3;
      days.push({ day: i + 1, season, cloud: Math.round(cloud), humidity: Math.round(humidity), rain: score > 0.45 });
    }
    return days;
  }

  const DAYS = generateDays(4835);
  const CLOUD_SPLIT = 50, HUMIDITY_SPLIT = 55;

  function majority(rows) { return rows.filter(r => r.rain).length > rows.length / 2; }

  function groupPredictor(keyFn) {
    const groups = new Map();
    DAYS.forEach(d => { const k = keyFn(d); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(d); });
    const predictions = new Map();
    groups.forEach((rows, k) => predictions.set(k, majority(rows)));
    return d => predictions.get(keyFn(d));
  }

  const STAGES = [
    {
      kicker: "Stage 0 · No data",
      title: "Nothing to go on",
      copy: "You haven't looked at a single past day. There is no pattern to lean on, so no guess is better than any other — you'd do exactly as well flipping a coin.",
      columns: [],
      predictor: null,
      note: "No feature columns are shown yet — there's nothing to base a prediction on.",
    },
    {
      kicker: "Stage 1 · Historical record only",
      title: "You know the base rate",
      copy: "You've now seen 48 past days: it rained on 17 of them (35%). With nothing else to go on, always guessing “no rain” is your best bet.",
      columns: [],
      predictor: groupPredictor(() => "all"),
      note: "One number from history — the overall rain frequency — already beats guessing.",
    },
    {
      kicker: "Stage 2 · + Season",
      title: "A second signal: the season",
      copy: "Add one feature — which season each day fell in — and the rule can split: guess “rain” in the rainy season, “no rain” otherwise.",
      columns: ["season"],
      predictor: groupPredictor(d => d.season),
      note: "Season alone already separates most of the rainy days from the dry ones.",
    },
    {
      kicker: "Stage 3 · + Cloud cover",
      title: "A third signal: cloud cover",
      copy: "Add today's cloud cover on top of season, and the rule can react to unusually cloudy dry days or unusually clear rainy days.",
      columns: ["season", "cloud"],
      predictor: groupPredictor(d => `${d.season}|${d.cloud >= CLOUD_SPLIT ? "high" : "low"}`),
      note: "Cloud cover catches the exceptions season alone missed.",
    },
    {
      kicker: "Stage 4 · + Humidity",
      title: "A fourth signal: humidity",
      copy: "Add humidity too, and the rule narrows in further — the same idea as Chapter 4's window, just with three features stacked instead of one.",
      columns: ["season", "cloud", "humidity"],
      predictor: groupPredictor(d => `${d.season}|${d.cloud >= CLOUD_SPLIT ? "high" : "low"}|${d.humidity >= HUMIDITY_SPLIT ? "high" : "low"}`),
      note: "More evidence keeps narrowing the guess — but notice the gains get smaller each time.",
    },
  ];

  const ACCURACY = STAGES.map(stage => {
    if (!stage.predictor) return null;
    const correct = DAYS.filter(d => stage.predictor(d) === d.rain).length;
    return { correct, total: DAYS.length, pct: correct / DAYS.length };
  });

  let stage = 0;

  const COLUMN_LABELS = { season: "Season", cloud: "Cloud %", humidity: "Humidity %" };
  const fmtBool = v => v ? "Rain" : "No rain";

  function renderTable() {
    const spec = STAGES[stage];
    const acc = ACCURACY[stage];
    const headCols = spec.columns.map(c => `<th>${COLUMN_LABELS[c]}</th>`).join("");
    const showPrediction = Boolean(spec.predictor);
    const head = `<tr><th>Day</th>${headCols}<th>Actual</th>${showPrediction ? "<th>Predicted</th><th></th>" : ""}</tr>`;
    const rows = DAYS.map(d => {
      const cells = spec.columns.map(c => `<td>${c === "season" ? (d.season === "rainy" ? "Rainy" : "Dry") : d[c]}</td>`).join("");
      let predictedCells = "";
      let rowClass = "";
      if (showPrediction) {
        const predicted = spec.predictor(d);
        const correct = predicted === d.rain;
        rowClass = correct ? "row-correct" : "row-wrong";
        predictedCells = `<td>${fmtBool(predicted)}</td><td class="mark">${correct ? "✓" : "✗"}</td>`;
      }
      return `<tr class="${rowClass}"><td>${d.day}</td>${cells}<td>${fmtBool(d.rain)}</td>${predictedCells}</tr>`;
    }).join("");
    $("#data-table").innerHTML = `<thead>${head}</thead><tbody>${rows}</tbody>`;
  }

  function renderBars() {
    const width = 460, height = 130, left = 34, bottom = 108, top = 10, barGap = 10;
    const barW = (width - left - 10 - barGap * (STAGES.length - 1)) / STAGES.length;
    const py = pct => bottom - pct * (bottom - top);
    const grid = [0, 0.25, 0.5, 0.75, 1].map(v => `<line class="plot-grid" x1="${left}" y1="${py(v)}" x2="${width - 5}" y2="${py(v)}"/>`).join("");
    const bars = STAGES.map((s, i) => {
      const acc = ACCURACY[i];
      const x = left + i * (barW + barGap);
      const pct = acc ? acc.pct : 0.5;
      const y = py(pct);
      const active = i === stage;
      const label = acc ? `${(acc.pct * 100).toFixed(0)}%` : "50%";
      return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${(bottom - y).toFixed(1)}" class="acc-bar ${active ? "acc-bar-active" : ""}"/><text class="axis-label" x="${(x + barW / 2).toFixed(1)}" y="${(y - 6).toFixed(1)}" text-anchor="middle">${label}</text><text class="axis-label" x="${(x + barW / 2).toFixed(1)}" y="${bottom + 16}" text-anchor="middle">${i}</text>`;
    }).join("");
    $("#accuracy-bars").innerHTML = `${grid}${bars}<text class="axis-label" x="${width / 2}" y="${bottom + 30}" text-anchor="middle">stage</text>`;
  }

  function render() {
    const spec = STAGES[stage];
    const acc = ACCURACY[stage];
    $("#stage-kicker").textContent = spec.kicker;
    $("#stage-title").textContent = spec.title;
    $("#stage-copy").textContent = spec.copy;
    $("#stage-note").textContent = spec.note;
    $("#accuracy-value").textContent = acc ? `${acc.correct}/${acc.total} (${(acc.pct * 100).toFixed(0)}%)` : "50% (chance)";
    document.querySelectorAll(".step").forEach(btn => {
      const v = Number(btn.dataset.stage);
      btn.classList.toggle("active", v === stage);
      btn.classList.toggle("done", v < stage);
    });
    $("#previous").disabled = stage === 0;
    const next = $("#next");
    next.disabled = stage === STAGES.length - 1;
    next.textContent = stage === STAGES.length - 1 ? "All evidence added" : "Add more evidence →";
    renderTable();
    renderBars();
  }

  function goToStage(next) {
    stage = Math.max(0, Math.min(STAGES.length - 1, next));
    render();
  }

  $("#previous").addEventListener("click", () => goToStage(stage - 1));
  $("#next").addEventListener("click", () => goToStage(stage + 1));
  document.querySelectorAll(".step").forEach(btn => btn.addEventListener("click", () => goToStage(Number(btn.dataset.stage))));

  render();
})();
