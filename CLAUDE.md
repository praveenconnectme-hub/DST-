# Demand Sensing Control Tower — CLAUDE.md

## Project Summary

This is a genuinely functional, containerised, single-tenant agentic demand-sensing
platform for the Indian television market (Samsung India TV Division). It ingests weekly
SKU × state sales history, runs statistical baseline forecasts (Holt-Winters / Auto-ARIMA /
Croston), feeds a machine-learning demand-sensing engine (XGBoost), and orchestrates three
hard-blocking human gates — G1 Promotions Calendar, G2 S&OP Consensus, and G3 Accuracy &
Retraining Oversight — within a deterministic state-machine pipeline. The stack runs as four
Docker containers (gateway · frontend · api · worker) brought up with docker-compose, with
all persistence flowing through an AbstractRepository / SQLiteRepository layer backed by
SQLite; the architecture is designed so that swapping to a cloud SQL provider later requires
touching only the factory line. The authoritative specification is the BRD at
`../Demand sensing BRD V1.1 LATEST/Demand_Sensing_Control_Tower_BRD_v1.1.docx`.

---

## Current Phase: 3  ✅ COMPLETE — Phase 4 not started

Phase 1, 2, and 3 complete. Phase 4 (G2 S&OP Consensus + LLM calendar) not yet started.

### Phase 3 — all increments

| Increment | Status | Key outputs |
|---|---|---|
| P3-1: Auth (session+bcrypt) | ✅ Done | `POST /auth/login`, `GET /auth/me`, `POST /auth/logout`; `get_current_user()` single point; 5 demo users seeded; D-018 |
| P3-2: Promotions Ledger API | ✅ Done | GET/POST/PATCH `/promotions`, `POST /promotions/ai-draft` (static, no LLM — D-019); RBAC commercial_head writes; atomic audit; D-020 |
| P3-3: G1 Gate API | ✅ Done | `GET /gates/G1/{cycle_id}`, `POST /gates/G1/{cycle_id}/approve`; 403 for non-commercial_head; D-021 idempotent double-approve |
| P3-4: Worker G1 gate wiring | ✅ Done | G1 check between LOADING_SIGNALS and SENSING; return-not-sleep; resume skips modules 1-3; safety invariant enforced; D-022 |
| P3-5: Promotions UI | ✅ Done | Login page; Promotions screen with 4 live KPI indicators; G1 Approve button (role-gated); System-suggested drafts; sidebar user/logout |
| P3-6: Sensing endpoint + chart | ✅ Done | `GET /api/sensing`, `GET /api/sensing/summary`; Forecast page baseline-vs-sensing chart; MAPE KPI wired in promotions page |
| P3-7: DoD close + full-tree grep | ✅ Done | 156/156 tests pass; guardian clean; DECISIONS.md D-018–D-022 complete; OD-3 resolved |

### Phase 3 DoD (BRD §10) — ✅ VERIFIED

| DoD Item | Evidence |
|---|---|
| Auth: session+bcrypt; single `get_current_user()` | `test_auth.py` 11 tests; `api/dependencies.py` is sole auth point; all routes use `Depends(get_current_user)` or `Depends(require_role(...))` |
| RBAC enforced server-side (planner → 403 at API) | `test_approve_wrong_role_returns_403`, `test_approve_sop_chair_returns_403`, `test_create_promotion_rbac_denied_for_planner` — all POST directly, no UI |
| Static promotions/event calendar; LLM deferred (D-019) | `test_ai_draft_llm_used_is_always_false`; full-tree grep confirms zero LLM/network imports in api/ |
| G1 hard-block: pipeline pauses at G1_PROMOTIONS_BLOCKED | `test_g1_unapproved_halts_at_blocked_state`, `test_safety_invariant_no_sensing_without_g1` |
| Safety invariant: SENSING unreachable without gate approval | `worker/main.py:90-102` (docstring + code); two gate checks (fresh + resume paths) both in same function; grep proof: `PipelineState.SENSING` appears only once, at line 168, after gate check at line 153 |
| Resume skips modules 1-3 | `test_resume_does_not_re_run_ingestion/baseline/signals`; `test_full_audit_trail_shows_blocked_then_resumed` |
| Approval audited atomically | `test_approve_writes_gate_and_audit_atomically`; outer `with repo.transaction()` wraps `set_gate_status()` + `GATE_APPROVED` audit upsert |
| Sensing endpoint + baseline-vs-sensing chart | `test_sensing_endpoint.py` 13 tests; `GET /api/sensing` + `GET /api/sensing/summary` behind `get_current_user()`; forecast page updated |
| Four live KPI indicators on promotions screen | Pipeline State (`/health`), G1 Status (`/gates/G1/{cycle_id}`), Promo Count (`/promotions?cycle_id=`), Sensing MAPE (`/sensing/summary`) |

### Phase 3 Live Demo Script (Samsung click-path)

| Step | Action | Expected |
|---|---|---|
| 1 | Navigate to `http://localhost:8080/` | Redirects to `/pages/login.html` |
| 2 | Login as `planner_01` / `pl-demo-2024` | Redirected to Dashboard |
| 3 | Open Forecast page | Baseline chart (teal) + XGBoost Sensing overlay (yellow) on holdout; region labels visible |
| 4 | Open Promotions page | 4 live KPI cards; G1 Approve button **disabled** (UI hint only) |
| 5 | Click "Get System Suggestions" | 3 draft cards labelled "System-suggested"; `llm_used: false` |
| 6 | Add a draft to ledger | Row appears in Promotions table; audit row written |
| 7 | **Role enforcement (authorization, not just auth):** still logged in as planner_01 — open browser DevTools Console and run: `fetch('/api/gates/G1/2024-W01/approve', {method:'POST',credentials:'same-origin'}).then(r=>r.status).then(console.log)` | **`403`** — server rejects an authenticated planner; button state irrelevant. Backing test: `test_approve_wrong_role_returns_403` at `tests/test_gates.py:144` |
| 8 | Log out → login as `commercial_head_01` / `ch-demo-2024` | Dashboard |
| 9 | Open Promotions → click **Approve G1** | Status KPI flips to `approved`; `POST /api/gates/G1/{cycle_id}/approve` → 200 |
| 10 | Trigger pipeline (Run Ingestion) → poll Dashboard | Worker resumes from SENSING (skips modules 1-3); `CYCLE_COMPLETE` |

---

### Phase 2 — all increments

| Increment | Status | Key outputs |
|---|---|---|
| P2-1: Multi-series HW baseline (all 108 series) | ✅ Done | `baseline_forecast` 1,296 rows; `model_registry` 108 rows |
| P2-2: Champion-challenger selection (HW vs ARIMA vs Croston) | ✅ Done | 108 series → HW champion; `model_registry` 216 rows; `accuracy_metrics` 1,296 rows; `audit_log` 108 CHAMPION + 108 RETIRED |
| P2-3: Signal ingestion (weather / competitor / search → `signal_data`) | ✅ Done | `signal_data` 5,616 rows (3 signals × 12 states × 156 weeks); D-016 |
| P2-4: Feature assembly + XGBoost sensing | ✅ Done | `demand_sensing_output` 1,296 rows + SHAP; sensing 15.2 % vs baseline 15.0 %; D-014 → Option C |
| P2-5: Worker pipeline wiring + accuracy scoring | ✅ Done | 6-state orchestrator (INGESTING→CYCLE_COMPLETE); `accuracy_metrics` 1,296 xgboost rows; D-017 |
| P2-6: API sensing endpoint + frontend chart | ↪ Deferred to Phase 3 | Blocked on OD-3 auth resolution |
| P2-8: Tests (sensing + signals + pipeline cycle) | ✅ Done | 79/79 pass: `test_sensing.py` (11), `test_signals.py` (8), `test_pipeline_cycle.py` (6), prior suites |

### Phase 2 DoD (BRD §10) — ✅ VERIFIED

> *Full baseline and sensing forecasts persisted for the whole synthetic dataset, each row
> carrying its model_id.*

| DoD Item | Evidence |
|---|---|
| Baseline for all 108 viable series (skips logged) | `baseline_forecast` 1,296 rows; 0 skipped; `BASELINE_COMPLETE` audit row |
| Auto-ARIMA + Croston challengers; champion on holdout MAPE | `model_registry` 216 rows (108 champion + 108 retired); `MODEL_CHAMPION_SELECTED` per series |
| `signal_data` from all 3 sources via repository | 5,616 rows; signals: `temp_deviation`, `competitor_price_index`, `search_trend_index` |
| XGBoost per D-012; `demand_sensing_output` + SHAP; MAPE reported | 1,296 rows; SHAP 10 features; sensing 15.2 % vs baseline 15.0 % |
| Actuals + `accuracy_metrics`; pipeline_state through 6 states atomic | `accuracy_metrics` 1,296 xgboost rows; 6/6 integration tests pass; each state write atomic with audit_log |
| §11 MAPE calibration 8–18 % core; D-014 G3-tail gap documented | Baseline: min 7.8 % / median 15.0 % / p90 20.3 % / max 23.5 %; 0 over 30 % → D-014 Option C before Phase 5 |

### Key Phase 2 decisions

- **D-012** OD-1 resolved: XGBoost pools by product tier (5 models); sku_id + state_code as categoricals.
- **D-013** `accuracy_metrics` populated in P2-1/P2-2 for §11 calibration check (advanced from Phase 5).
- **D-014** ⚠ Zero series breach 30 % MAPE — lognormal σ=0.12 insufficient. Option C (`SKU_LUX_FIXTURE_*`) required before Phase 5 G3.
- **D-015** Auto-ARIMA = statsmodels grid-search (pmdarima unavailable locally; in requirements.txt for Docker).
- **D-016** `synthetic.py` writes `weather_data.csv` alongside JSON so `signals.py` can use `repo.read_csv_raw()`.
- **D-017** Phase 2 step states added; `scoring.py` scores `demand_sensing_output` vs `actuals` only (no actuals exist for `baseline_forecast` horizon weeks).

### Signal files (P2-3 complete)

`synthetic.py` generates and `signals.py` loads via `repo.read_csv_raw()`:

| File | signal_name | Rows |
|---|---|---|
| `weather_data.csv` (+ `.json`) | `temp_deviation` | 1,872 |
| `competitor_scrapes.csv` | `competitor_price_index` | 1,872 |
| `google_trends_export.csv` | `search_trend_index` | 1,872 |

D-016: `weather_data.csv` written alongside JSON by `synthetic.py` so all three sources load via `repo.read_csv_raw()` — no new repo method needed.

---

## STANDING RULES — apply to every action in this project, no exceptions

1. **REPOSITORY ABSTRACTION IS SACRED (BRD §5.0).** Implement `AbstractRepository` and
   `SQLiteRepository` exactly as the BRD specifies — all 15 methods, the transaction
   context manager, gate/state methods, and `import_csv`/`export_excel`. After that,
   NO other file — no agent, API route, ML routine, UI handler, or test fixture —
   may `import sqlite3`, open a file, or write SQL. Every read/write goes through the
   repository. Excel/CSV are touched ONLY inside `import_csv` / `export_excel`. Every
   write that mutates gate or plan state appends an `audit_log` row in the SAME
   transaction. A `RepositoryFactory.create(config)` returns the concrete impl; v1
   config names `'sqlite'`.

2. **PHASE 1 ONLY (until user says "continue").** Build only what BRD §10 Phase 1 lists.
   Do NOT build Phase 2–5 features: no XGBoost/sensing, no signals feature assembly, no
   LLM calendar, no promotions/consensus/field/accuracy logic, no champion/challenger.
   When the Phase 1 DoD is met, STOP and tell me. I will explicitly say "continue".

3. **FULL SCHEMA NOW, SLICE OF IT NOW.** Create ALL §5.1 tables via migration in Phase 1
   so the schema is stable. Populate/exercise only the tables Phase 1 needs (ingestion +
   baseline). Write no business logic for unused tables.

4. **DON'T OVER-ENGINEER.** No Airflow/Prefect/Dagster — orchestrator is a custom Python
   state-machine loop in the worker (BRD §9.1). No Kubernetes, no message broker —
   api↔worker communicate via a job table in SQLite. A stub login is fine in Phase 1;
   real RBAC comes later.

5. **ASK BEFORE RESOLVING OPEN DECISIONS (BRD §13: OD-1, OD-2, OD-3).** None block
   Phase 1. Don't resolve them now; flag if something forces an early call.

6. **TECH DEFAULTS** (use unless you state a strong reason): Python + FastAPI (api),
   Python worker, vanilla HTML/JS/CSS frontend mirroring the prototype IA, Nginx gateway,
   SQLite behind the repository, docker-compose for local bring-up. Phase 1 deps minimal:
   pandas, statsmodels (Holt-Winters), fastapi, uvicorn, a SQLite driver. DEFER xgboost,
   shap, MLflow, and the LLM SDK to later phases.

7. **WORK STYLE.** Build in small runnable increments. For each, tell me the exact command
   to run and what I should see. Test as you go. Record any BRD deviation in DECISIONS.md
   and tell me — never bury it. Where this project and the BRD conflict, ASK — do not
   silently pick one.

---

## Phase 1 — Definition of Done (BRD §10)

All items below must be true before Phase 2 begins:

- [ ] The four containers (gateway, frontend, api, worker) build and `docker-compose up`
      brings the stack up cleanly.
- [ ] The frontend loads and successfully reaches `GET /api/health`.
- [ ] `AbstractRepository` + `SQLiteRepository` are implemented per §5.0, and all §5.1
      tables are created via a migration.
- [ ] The synthetic data generator (BRD §11) exists and loads data through `import_csv`
      (no direct file writes elsewhere).
- [ ] Module 1 (Data Ingestion) runs: parses, validates, quarantines bad rows, persists
      via the repository.
- [ ] A single Holt-Winters baseline forecast is produced for one sku×state series,
      persisted to `baseline_forecast`, and displayed on a Forecast screen in the UI.
- [ ] `grep` confirms no file/DB access exists outside the repository layer.

**✅ Phase 1 DoD verified** — all items were met in the initial build session.
Stack runs at `http://localhost:8080/`. Tests (Phase 1 baseline): 28/28 pass.

---

## Repository Layer (key files)

| File | Role |
|---|---|
| `api/repository/abstract.py` | `AbstractRepository` ABC — 15 methods |
| `api/repository/sqlite_repo.py` | `SQLiteRepository` — **ONLY** file allowed to `import sqlite3` |
| `api/repository/factory.py` | `RepositoryFactory.create(config)` |
| `api/migrations/migration_001.py` | Creates all 19 §5.1 tables — the other permitted sqlite3 user |

---

## Sub-Agents

| Agent | File | Purpose |
|---|---|---|
| `repo-guardian` | `.claude/agents/repo-guardian.md` | Reviews diffs for persistence violations |
| `test-writer` | `.claude/agents/test-writer.md` | Writes pytest tests for repository + ingestion |

---

## Key Decisions

See `DECISIONS.md` for the full log. Summary of Phase 1 calls:

- **D-001** Frontend is static HTML served by Nginx (no build step needed).
- **D-002** `sqlite3` stdlib used synchronously in `SQLiteRepository`; transaction depth
  tracked via `threading.local.txn_depth` so `upsert`/`delete` don't auto-commit inside
  `transaction()` blocks.
- **D-003** Worker polls `job_queue` every 5 seconds.
- **D-004** Phase 1 auth is a stub — all API calls accepted without token.
- **D-005** Phase 1 single-series demo: `SKU_MID_01 × MH` (Maharashtra).
- **Gateway port: 8080** (port 80 was taken by another running project on this machine).
- **DB on named volume (D-023):** SQLite lives on `dst_data` Docker named volume, NOT `./data` bind-mount (macOS osxfs file-locking is broken for SQLite WAL). To reset: `docker compose down -v && docker compose up --build` (not `rm -f data/dst.db`).

---

## Open Decisions (BRD §13)

- **OD-1** ✅ RESOLVED as D-012 — pool by product tier.
- **OD-2** MLflow vs custom registry. Decide before Phase 5.
- **OD-3** Auth depth for v1. Confirmed stub for Phase 1; revisit before Phase 3.
