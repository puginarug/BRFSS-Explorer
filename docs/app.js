// BRFSS-Explorer — Mental Health Data Explorer
// Observable Plot + D3 · vanilla ES module · no build step

import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm";
import * as d3   from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

// ── Feature metadata ──────────────────────────────────────────────────────────
// keys must match column names in brfss_test.csv exactly

const FEATURES = {
  physical_health_days: {
    label:       "Physical Health (bad days/month)",
    description: "Number of days in the past 30 where physical health was not good. Ranges 0–30. Strong co-morbidity with mental health burden.",
    bin: v => v === 0 ? 0 : v <= 5 ? 1 : v <= 10 ? 2 : v <= 20 ? 3 : 4,
    levels: { 0: "0 days", 1: "1–5 days", 2: "6–10 days", 3: "11–20 days", 4: "21–30 days" },
  },
  general_health: {
    label:       "General Health (1=Excellent → 5=Poor)",
    description: "Self-rated overall health on a 1–5 scale. Captures perceived wellbeing beyond any single diagnosis.",
    levels: { 1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor" },
  },
  depression_diagnosis: {
    label:       "Depression Diagnosis (0=No, 1=Yes)",
    description: "Whether a doctor ever told the respondent they had a depressive disorder. The single strongest binary predictor in this dataset.",
    levels: { 0: "No depression diagnosis", 1: "Depression diagnosed" },
  },
  life_satisfaction: {
    label:       "Life Satisfaction (1=Very Satisfied → 4=Dissatisfied)",
    description: "Overall life satisfaction. Lower = more satisfied. Can also be used as the Y-axis outcome — switch the Y dropdown above.",
    levels: { 1: "Very Satisfied", 2: "Satisfied", 3: "Dissatisfied", 4: "Very Dissatisfied" },
  },
  emotional_support: {
    label:       "Emotional Support (1=Always → 5=Never)",
    description: "How often needed emotional support is available. Lack of support is a consistent risk factor across all demographic groups.",
    levels: { 1: "Always", 2: "Usually", 3: "Sometimes", 4: "Rarely", 5: "Never" },
  },
  loneliness: {
    label:       "Loneliness (1=Always → 5=Never)",
    description: "How often the respondent feels lonely. Shows the clearest population-level gradient of any social variable in the dataset.",
    levels: { 1: "Always lonely", 2: "Usually", 3: "Sometimes", 4: "Rarely", 5: "Never lonely" },
  },
  bills_difficulty: {
    label:       "Hard to Pay Bills (0=No, 1=Yes)",
    description: "Whether bills are difficult or very difficult to pay. A social determinant of health — financial stress compounds mental burden.",
    levels: { 0: "Bills manageable", 1: "Hard to pay bills" },
  },
  exercises: {
    label:       "Exercises Regularly (0=No, 1=Yes)",
    description: "Any physical activity or exercise outside of regular job in the past 30 days. Consistent protective effect across all groups.",
    levels: { 0: "No exercise", 1: "Exercises" },
  },
  ever_smoked: {
    label:       "Smoked ≥100 Cigarettes (0=No, 1=Yes)",
    description: "Lifetime smoking history (ever smoked ≥100 cigarettes). Proxy for health behaviors; associated with worse mental health outcomes.",
    levels: { 0: "Never heavy smoker", 1: "Smoked ≥100 cigarettes" },
  },
  sex: {
    label:       "Sex (0=Female, 1=Male)",
    description: "Biological sex as recorded in the survey. Women report higher rates of mental health burden in this dataset.",
    levels: { 0: "Female", 1: "Male" },
  },
  employment: {
    label:       "Employment Status (1–8 categories)",
    description: "Current employment situation across 8 categories. Unemployment and inability to work are associated with significantly higher burden.",
    levels: {
      1: "Employed",
      2: "Self-employed",
      3: "Unemployed <1 yr",
      4: "Unemployed ≥1 yr",
      5: "Homemaker",
      6: "Student",
      7: "Retired",
      8: "Unable to work",
    },
  },
  income: {
    label:       "Household Income (1–11 scale)",
    description: "Annual household income across 11 levels from <$10k to ≥$200k. Lower income correlates consistently with more bad mental health days.",
    levels: {
      1: "<$10k",  2: "$10–15k", 3: "$15–20k", 4: "$20–25k",  5: "$25–35k",
      6: "$35–50k", 7: "$50–75k", 8: "$75–100k", 9: "$100–150k", 10: "$150–200k", 11: "≥$200k",
    },
  },
  education: {
    label:       "Education Level (1–6)",
    description: "Highest level of education completed. Higher education is consistently associated with better mental health outcomes.",
    levels: {
      1: "Never attended",
      2: "Grades 1–8",
      3: "Grades 9–11",
      4: "HS graduate",
      5: "Some college",
      6: "College grad",
    },
  },
  age_group: {
    label:       "Age Group (1=18–24 → 13=80+)",
    description: "Age in 5-year bands. Shows a non-linear relationship with mental health — the middle-age cohort often carries the highest burden.",
    levels: {
      1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44",
      6: "45–49", 7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69",
      11: "70–74", 12: "75–79", 13: "80+",
    },
  },
  bmi_category: {
    label:       "BMI Category (1=Underweight → 4=Obese)",
    description: "CDC-computed BMI category from reported height and weight. Obesity is associated with higher mental health burden in this dataset.",
    levels: { 1: "Underweight", 2: "Normal", 3: "Overweight", 4: "Obese" },
  },
};

const Y_OPTIONS = {
  menthlth_days:    { label: "Mean Bad Mental Health Days / Month" },
  life_satisfaction: { label: "Mean Life Satisfaction (1=Best → 4=Worst)" },
};

// ── Spearman correlation ──────────────────────────────────────────────────────

function rankArray(arr) {
  const n = arr.length;
  const indexed = Array.from(arr, (v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const ranks = new Array(n);
  indexed.forEach(([, i], r) => { ranks[i] = r; });
  return ranks;
}

function spearmanCorr(x, y) {
  const rx = rankArray(x);
  const ry = rankArray(y);
  const n  = x.length;
  let d2 = 0;
  for (let i = 0; i < n; i++) d2 += (rx[i] - ry[i]) ** 2;
  return 1 - (6 * d2) / (n * (n * n - 1));
}

// ── DOM refs ──────────────────────────────────────────────────────────────────

// Panel 1 — Distribution
const distSelect   = document.getElementById("dist-select");
const distChartDiv = document.getElementById("dist-chart");
const distDescEl   = document.getElementById("dist-desc");

// Panel 2 — Relationships
const xSelect        = document.getElementById("x-select");
const ySelect        = document.getElementById("y-select");
const chartTypeSelect = document.getElementById("chart-type");
const chartDiv       = document.getElementById("chart");
const corrEl         = document.getElementById("correlation");
const descEl         = document.getElementById("feature-desc");

// ── Populate dropdowns ────────────────────────────────────────────────────────

function populateDistDropdown() {
  distSelect.innerHTML = "";
  Object.entries(FEATURES).forEach(([key, meta]) => {
    const opt = document.createElement("option");
    opt.value       = key;
    opt.textContent = meta.label;
    distSelect.appendChild(opt);
  });
  // Default to loneliness
  distSelect.value = "loneliness";
}

// ── Render distribution chart (Panel 1) ──────────────────────────────────────

function renderDistribution() {
  const xKey = distSelect.value;
  const feat = FEATURES[xKey];
  if (!feat || !window.__data) return;

  const data  = window.__data;
  const total = data.length;
  const hasBin = !!feat.bin;

  // Count respondents per level
  const grouped = d3.rollups(
    data,
    v => v.length,
    d => hasBin ? feat.bin(d[xKey]) : d[xKey]
  )
    .map(([level, count]) => ({
      level,
      count,
      pct:   count / total * 100,
      label: feat.levels?.[level] ?? String(level),
    }))
    .sort((a, b) => a.level - b.level);

  const width = distChartDiv.offsetWidth || 840;

  const plot = Plot.plot({
    width,
    height: 320,
    marginBottom: 64,
    marginLeft: 56,
    style: {
      background:  "transparent",
      color:       "#c8c8c8",
      fontFamily:  "Inter, sans-serif",
      fontSize:    "13px",
    },
    x: {
      label:      null,
      tickRotate: grouped.length > 6 ? -30 : 0,
      tickFormat: d => feat.levels?.[d] ?? String(d),
      padding:    0.25,
    },
    y: {
      label:       "% of respondents",
      grid:        true,
      zero:        true,
      line:        true,
      labelOffset: 40,
    },
    marks: [
      Plot.barY(grouped, {
        x:    "level",
        y:    "pct",
        fill: "#4AABF0",
        rx:   4,
        tip:  true,
        title: d => `${d.label}\n${d.pct.toFixed(1)}%  (n = ${d.count.toLocaleString()})`,
      }),
      Plot.ruleY([0], { stroke: "#333", strokeWidth: 1.5 }),
      Plot.text(grouped, {
        x:          "level",
        y:          "pct",
        text:       d => `${d.pct.toFixed(1)}%`,
        dy:         -10,
        fill:       "#e0e0e0",
        fontSize:   12,
        fontWeight: "600",
      }),
    ],
  });

  // Fade out → swap → fade in
  distChartDiv.style.opacity = "0";
  setTimeout(() => {
    distChartDiv.innerHTML = "";
    distChartDiv.appendChild(plot);
    void distChartDiv.offsetHeight;
    distChartDiv.style.opacity = "1";
  }, 140);

  distDescEl.textContent = feat.description;
}

// ── Populate X dropdown ───────────────────────────────────────────────────────

function populateXDropdown(excludeKey) {
  const prev = xSelect.value;
  xSelect.innerHTML = "";
  let defaultKey = null;

  Object.entries(FEATURES).forEach(([key, meta]) => {
    if (key === excludeKey) return;
    const opt = document.createElement("option");
    opt.value       = key;
    opt.textContent = meta.label;
    xSelect.appendChild(opt);
    if (!defaultKey) defaultKey = key;
  });

  // Restore previous selection if still available; otherwise default to loneliness
  if (prev && prev !== excludeKey && FEATURES[prev]) {
    xSelect.value = prev;
  } else if (FEATURES["loneliness"] && excludeKey !== "loneliness") {
    xSelect.value = "loneliness";
  } else {
    xSelect.value = defaultKey;
  }
}

// ── Render chart ──────────────────────────────────────────────────────────────

function renderChart() {
  const xKey      = xSelect.value;
  const yKey      = ySelect.value;
  const chartType = chartTypeSelect.value;   // "bar" | "box"
  const feat      = FEATURES[xKey];

  if (!feat || !window.__data) return;

  const data   = window.__data;
  const hasBin = !!feat.bin;
  const xOf    = d => hasBin ? feat.bin(d[xKey]) : d[xKey];

  // Unique sorted levels (needed for tickRotate in both chart types)
  const levels = [...new Set(data.map(xOf))].sort((a, b) => a - b);

  // Shared Plot config
  const width = chartDiv.offsetWidth || 840;
  const base = {
    width,
    height: 380,
    marginBottom: 64,
    marginLeft:   56,
    style: {
      background: "transparent",
      color:      "#c8c8c8",
      fontFamily: "Inter, sans-serif",
      fontSize:   "13px",
    },
    x: {
      label:      null,
      tickRotate: levels.length > 6 ? -30 : 0,
      tickFormat: d => feat.levels?.[d] ?? String(d),
      padding:    0.25,
    },
    y: {
      label:        Y_OPTIONS[yKey].label,
      labelAnchor:  "center",
      grid:         true,
      zero:         true,
      line:         true,
      labelOffset:  44,
    },
  };

  let plot;

  if (chartType === "bar") {
    // ── Bar of means ─────────────────────────────────────────────
    const grouped = d3.rollups(
      data,
      v => d3.mean(v, d => d[yKey]),
      xOf
    )
      .map(([level, mean]) => ({
        level,
        mean,
        label: feat.levels?.[level] ?? String(level),
      }))
      .sort((a, b) => a.level - b.level)
      .filter(d => d.mean != null && isFinite(d.mean));

    plot = Plot.plot({
      ...base,
      marks: [
        Plot.barY(grouped, {
          x:     "level",
          y:     "mean",
          fill:  "#F0A500",
          rx:    4,
          tip:   true,
          title: d => `${d.label}\n${d.mean.toFixed(2)}`,
        }),
        Plot.ruleY([0], { stroke: "#333", strokeWidth: 1.5 }),
        Plot.text(grouped, {
          x:          "level",
          y:          "mean",
          text:       d => d.mean.toFixed(2),
          dy:         -10,
          fill:       "#e0e0e0",
          fontSize:   12,
          fontWeight: "600",
        }),
      ],
    });

  } else {
    // ── Box plot (median + IQR + whiskers) ───────────────────────
    // Plot.boxY passes raw rows; Observable Plot computes statistics per x group.
    plot = Plot.plot({
      ...base,
      marks: [
        Plot.boxY(data, {
          x:            xOf,
          y:            d => d[yKey],
          fill:         "#F0A500",
          fillOpacity:  0.15,
          stroke:       "#F0A500",
          strokeWidth:  1.5,
          r:            2,           // outlier dot radius
        }),
        Plot.ruleY([0], { stroke: "#333", strokeWidth: 1.5 }),
      ],
    });
  }

  // Fade out → swap → fade in
  chartDiv.style.opacity = "0";
  setTimeout(() => {
    chartDiv.innerHTML = "";
    chartDiv.appendChild(plot);
    void chartDiv.offsetHeight;
    chartDiv.style.opacity = "1";
  }, 140);

  // Spearman ρ (same regardless of chart type — it's a property of the data)
  const xVals = data.map(xOf);
  const yVals = data.map(d => d[yKey]);
  const rho   = spearmanCorr(xVals, yVals);
  corrEl.textContent = `${rho >= 0 ? "+" : ""}${rho.toFixed(3)}`;
  corrEl.style.color = Math.abs(rho) > 0.2 ? "#F0A500" : "#666";

  descEl.textContent = feat.description;
}

// ── Event listeners ───────────────────────────────────────────────────────────

distSelect.addEventListener("change", renderDistribution);

ySelect.addEventListener("change", () => {
  populateXDropdown(ySelect.value === "life_satisfaction" ? "life_satisfaction" : null);
  renderChart();
});

xSelect.addEventListener("change", renderChart);
chartTypeSelect.addEventListener("change", renderChart);

// ── Load data and init ────────────────────────────────────────────────────────

try {
  const raw = await d3.csv("data/brfss_test.csv", d3.autoType);
  window.__data = raw;
  // Panel 1
  populateDistDropdown();
  renderDistribution();
  // Panel 2
  populateXDropdown(null);
  renderChart();
} catch (err) {
  const errMsg = `
    <p class="loading">
      Could not load data. If running locally, use a web server:<br><br>
      <code>python -m http.server 8080</code><br>
      then open <a href="http://localhost:8080">http://localhost:8080</a>
    </p>`;
  distChartDiv.innerHTML = errMsg;
  chartDiv.innerHTML = `
    <p class="loading">
      Could not load data. If running locally, use a web server:<br><br>
      <code>python -m http.server 8080</code><br>
      then open <a href="http://localhost:8080">http://localhost:8080</a>
    </p>`;
  console.error(err);
}
