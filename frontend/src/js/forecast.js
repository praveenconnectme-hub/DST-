// Forecast screen logic — baseline + sensing (P3-6)

const DEFAULT_SKU   = "SKU_MID_01";
const DEFAULT_STATE = "MH";

async function loadForecast() {
  const statusEl  = document.getElementById("forecast-status");
  const tableBody = document.getElementById("forecast-tbody");
  const chartEl   = document.getElementById("forecast-chart");
  const kpiWeeks  = document.getElementById("kpi-forecast-weeks");
  const kpiModel  = document.getElementById("kpi-model");
  const kpiSku    = document.getElementById("kpi-sku");

  statusEl.textContent = "Loading…";
  statusEl.className   = "alert alert-info";
  statusEl.classList.remove("hidden");
  tableBody.innerHTML  = "";

  try {
    const baseline = await API.getBaselineForecast(DEFAULT_SKU, DEFAULT_STATE);

    if (!baseline || !baseline.forecasts || baseline.forecasts.length === 0) {
      statusEl.textContent = "No baseline forecast found. Run ingestion first, then trigger the worker.";
      statusEl.className   = "alert alert-warn";
      return;
    }

    statusEl.classList.add("hidden");

    // KPIs — baseline
    kpiWeeks.textContent = baseline.forecasts.length;
    kpiModel.textContent = baseline.model_id || "—";
    kpiSku.textContent   = `${baseline.sku_id} × ${baseline.state_code}`;
    document.getElementById("header-baseline-badge").textContent =
      baseline.model_id || "Baseline";

    // Build baseline map  week_index → forecast_qty
    const baselineMap = {};
    baseline.forecasts.forEach(r => { baselineMap[r.week_index] = r.forecast_qty; });

    // Attempt to fetch sensing data (may return null if not yet run)
    let sensingMap = {};
    let sensingModelId = null;
    try {
      const sensing = await API.getSensing(DEFAULT_SKU, DEFAULT_STATE);
      if (sensing && sensing.weeks && sensing.weeks.length > 0) {
        sensing.weeks.forEach(r => { sensingMap[r.week_index] = r.sensing_qty; });
        sensingModelId = sensing.model_id;
        document.getElementById("kpi-sensing-model").textContent = sensing.model_id || "—";
        document.getElementById("kpi-sensing-weeks").textContent = `${sensing.weeks.length} holdout weeks`;
        document.getElementById("header-sensing-badge").classList.remove("hidden");
        document.getElementById("sensing-legend").style.display = "";
      }
    } catch (_) {
      // Sensing not yet available — degrade gracefully, show baseline only
    }

    if (!sensingModelId) {
      document.getElementById("kpi-sensing-model").textContent = "Not yet run";
      document.getElementById("kpi-sensing-weeks").textContent = "run pipeline first";
    }

    // Build merged rows: sensing holdout weeks + baseline future weeks
    // Key: all weeks that appear in either source, sorted
    const allWeeks = new Set([
      ...Object.keys(sensingMap),
      ...baseline.forecasts.map(r => r.week_index),
    ]);
    const sortedWeeks = [...allWeeks].sort();

    // Actuals map from baseline response (sales_history joins)
    const actualsMap = {};
    baseline.forecasts.forEach(r => {
      if (r.actual !== null && r.actual !== undefined) {
        actualsMap[r.week_index] = r.actual;
      }
    });

    // Build table + chart data rows
    const rows = sortedWeeks.map(wk => ({
      week_index:    wk,
      actual:        actualsMap[wk] ?? null,
      baseline_qty:  baselineMap[wk] ?? null,
      sensing_qty:   sensingMap[wk] ?? null,
    }));

    renderTable(tableBody, rows);
    renderChart(chartEl, rows, !!sensingModelId);

  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    statusEl.className   = "alert alert-error";
    statusEl.classList.remove("hidden");
  }
}

function renderTable(tbody, rows) {
  tbody.innerHTML = "";
  rows.forEach(row => {
    const baselineDev = (row.baseline_qty !== null && row.actual !== null && row.actual !== 0)
      ? ((row.baseline_qty - row.actual) / row.actual * 100).toFixed(1) + "%"
      : "—";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.week_index}</td>
      <td class="text-right">${row.actual !== null ? Number(row.actual).toLocaleString() : "—"}</td>
      <td class="text-right" style="color:var(--kpmg-teal);font-weight:600;">
        ${row.baseline_qty !== null ? Math.round(row.baseline_qty).toLocaleString() : "—"}</td>
      <td class="text-right" style="color:var(--kpmg-yellow);font-weight:600;">
        ${row.sensing_qty !== null ? Math.round(row.sensing_qty).toLocaleString() : "—"}</td>
      <td class="text-right text-muted">${baselineDev}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderChart(container, rows, hasSensing) {
  container.innerHTML = "";

  const allValues = rows.flatMap(r => [
    r.actual       ?? 0,
    r.baseline_qty ?? 0,
    r.sensing_qty  ?? 0,
  ]);
  const maxVal = Math.max(...allValues, 1);

  const CHART_H = 220;

  // Identify region boundary: holdout weeks have sensing data; forward weeks do not.
  const holdoutCount = hasSensing ? rows.filter(r => r.sensing_qty !== null).length : 0;
  let dividerInserted = false;

  rows.forEach(row => {
    // Insert a vertical divider at the holdout→forward boundary.
    const isFirstForwardWeek = hasSensing && !dividerInserted
      && row.sensing_qty === null && row.baseline_qty !== null;
    if (isFirstForwardWeek) {
      const div = document.createElement("div");
      div.className = "chart-region-divider";
      container.appendChild(div);
      dividerInserted = true;
    }

    const group = document.createElement("div");
    group.className = "chart-bar-group";
    group.style.justifyContent = "flex-end";
    group.style.height = (CHART_H + 30) + "px";

    // Actual bar
    if (row.actual !== null) {
      const h = Math.max(2, Math.round((row.actual / maxVal) * CHART_H));
      const bar = document.createElement("div");
      bar.className = "chart-bar bar-actual";
      bar.style.height = h + "px";
      bar.title = `Actual: ${Math.round(row.actual).toLocaleString()}`;
      group.appendChild(bar);
    }

    // Baseline bar
    if (row.baseline_qty !== null) {
      const h = Math.max(2, Math.round((row.baseline_qty / maxVal) * CHART_H));
      const bar = document.createElement("div");
      bar.className = "chart-bar bar-forecast";
      bar.style.height = h + "px";
      bar.title = `Baseline: ${Math.round(row.baseline_qty).toLocaleString()}`;
      group.appendChild(bar);
    }

    // Sensing bar
    if (hasSensing && row.sensing_qty !== null) {
      const h = Math.max(2, Math.round((row.sensing_qty / maxVal) * CHART_H));
      const bar = document.createElement("div");
      bar.className = "chart-bar bar-sensing";
      bar.style.height = h + "px";
      bar.title = `Sensing: ${Math.round(row.sensing_qty).toLocaleString()}`;
      group.appendChild(bar);
    }

    const lbl = document.createElement("div");
    lbl.className = "chart-label";
    lbl.textContent = row.week_index.slice(5); // 'WXX' portion
    group.appendChild(lbl);

    container.appendChild(group);
  });

  // Add region-label header overlay when both regions are present.
  if (hasSensing && holdoutCount > 0 && holdoutCount < rows.length) {
    const holdoutPct = ((holdoutCount / rows.length) * 100).toFixed(1);
    const forwardPct = (100 - parseFloat(holdoutPct)).toFixed(1);
    const header = document.createElement("div");
    header.className = "chart-region-header";
    header.innerHTML =
      `<div class="chart-region-band--holdout" style="width:${holdoutPct}%">` +
      `Holdout: Actual vs XGBoost Sensing</div>` +
      `<div class="chart-region-band--forward" style="width:${forwardPct}%">` +
      `Forward: Baseline Forecast</div>`;
    container.prepend(header);
  }
}

async function triggerIngest() {
  const btn = document.getElementById("btn-ingest");
  btn.disabled = true;
  btn.textContent = "Running…";
  try {
    const res = await API.triggerIngest();
    alert(`Ingestion queued: ${res.message || JSON.stringify(res)}`);
  } catch (err) {
    alert(`Ingest error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Ingestion";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadForecast();

  const btn = document.getElementById("btn-ingest");
  if (btn) btn.addEventListener("click", triggerIngest);

  const refreshBtn = document.getElementById("btn-refresh");
  if (refreshBtn) refreshBtn.addEventListener("click", loadForecast);
});
