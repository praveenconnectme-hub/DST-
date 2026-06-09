// Central API client — all fetch calls go through here
const API_BASE = "/api";

async function apiFetch(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  // Redirect to login on 401 (except when on the login page itself)
  if (res.status === 401 && !window.location.pathname.endsWith("login.html")) {
    window.location.href = "/pages/login.html";
    return null;
  }
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  // 204 No Content
  if (res.status === 204) return null;
  return res.json();
}

window.API = {
  // ── Health ────────────────────────────────────────────────────────────────
  health: () => apiFetch("/health"),

  // ── Auth ──────────────────────────────────────────────────────────────────
  login:  (username, password) => apiFetch("/auth/login",  { method: "POST", body: JSON.stringify({ username, password }) }),
  logout: ()                   => apiFetch("/auth/logout", { method: "POST" }),
  me:     ()                   => apiFetch("/auth/me"),

  // ── Forecast / Ingestion (Phase 1/2) ──────────────────────────────────────
  getBaselineForecast: (skuId, stateCode) =>
    apiFetch(`/forecast/baseline?sku_id=${encodeURIComponent(skuId)}&state_code=${encodeURIComponent(stateCode)}`),
  listBaselineForecasts: () => apiFetch("/forecast/baseline/list"),
  triggerIngest:         () => apiFetch("/ingest", { method: "POST" }),
  getIngestionStatus:    () => apiFetch("/ingest/status"),
  getPipelineState: (cycleId) => apiFetch(`/pipeline/state?cycle_id=${encodeURIComponent(cycleId)}`),

  // ── Promotions ────────────────────────────────────────────────────────────
  listPromotions: (cycleId) => {
    const qs = cycleId ? `?cycle_id=${encodeURIComponent(cycleId)}` : "";
    return apiFetch(`/promotions${qs}`);
  },
  createPromotion: (body) => apiFetch("/promotions", { method: "POST", body: JSON.stringify(body) }),
  updatePromotion: (id, body) => apiFetch(`/promotions/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  aiDraftPromotions: (cycleId) =>
    apiFetch(`/promotions/ai-draft?cycle_id=${encodeURIComponent(cycleId)}`, { method: "POST" }),

  // ── Gates ─────────────────────────────────────────────────────────────────
  getGateStatus: (gateId, cycleId) => apiFetch(`/gates/${gateId}/${encodeURIComponent(cycleId)}`),
  approveGate:   (gateId, cycleId) => apiFetch(`/gates/${gateId}/${encodeURIComponent(cycleId)}/approve`, { method: "POST" }),

  // ── Sensing (P3-6) ────────────────────────────────────────────────────────
  getSensing: (skuId, stateCode) =>
    apiFetch(`/sensing?sku_id=${encodeURIComponent(skuId)}&state_code=${encodeURIComponent(stateCode)}`),
  getSensingSummary: () => apiFetch("/sensing/summary"),

  // ── Audit ─────────────────────────────────────────────────────────────────
  getAuditLog: (limit = 200) => apiFetch(`/audit?limit=${limit}`),
};
