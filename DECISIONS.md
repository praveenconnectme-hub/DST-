# DECISIONS.md — Demand Sensing Control Tower

## Purpose

This file is the permanent log of every deviation from the BRD and every Open-Decision
flag. **Every session that deviates from the BRD must add a row here and tell the user.**
Never bury a deviation in code comments.

Format: add a row to the appropriate phase table. Columns:

| ID | Decision | Reason | BRD Reference |
|----|----------|--------|---------------|

---

## Phase 1

| ID | Decision | Reason | BRD Reference |
|----|----------|--------|---------------|
| D-001 | Frontend served as static files via Nginx (no Node/Vite dev server) | Vanilla HTML/JS needs no build step; keeps container count at 4 as specified | §9.1 |
| D-002 | stdlib `sqlite3` used synchronously in `SQLiteRepository`; transaction depth tracked via `threading.local.txn_depth` so `upsert`/`delete` don't auto-commit inside `transaction()` blocks | FastAPI is async but SQLite writes are fast and infrequent in a weekly batch system; sync is simpler and correct | §5.0 |
| D-003 | Worker polls `job_queue` every 5 seconds (`POLL_INTERVAL=5`) | BRD says "worker polls for queued cycles" without specifying interval; 5 s is safe for a weekly batch pipeline | §9.1 |
| D-004 | Phase 1 auth is a stub — all API calls accepted without token | BRD §10 and kickoff prompt both explicitly permit stub login in Phase 1 | §10, Kickoff |
| D-005 | Phase 1 single-series demo fixed at `SKU_MID_01 × MH` (Maharashtra) | BRD §10 says "ONE sku×state"; deterministic pick, no BRD preference stated | §10 |
| D-006 | Gateway exposed on host port **8080** (not 80) | Port 80 was already bound by another running project on this machine; no architectural impact | §9.1 |
| D-007 | Frontend container serves on port **80** (nginx), not **5173** (Vite) | BRD §9.1 lists `80 → frontend:5173` which implies a Vite dev server. The kickoff prompt and Standing Rules explicitly require "no build step" for the frontend. Nginx serving static HTML/JS/CSS on port 80 is correct for a plain frontend. The gateway upstream is adjusted to `frontend:80` accordingly. | §9.1 |
| D-008 | `read_csv_raw(path) -> DataFrame` added to `AbstractRepository` beyond the BRD §5.0 method list | Without this method, ingestion.py had to call `pd.read_csv()` directly to read sales history for validation before persisting — a repo-guardian violation. Moving raw CSV reads into the repo layer keeps all file I/O inside the abstraction. The method is read-only and never persists. | §5.0 |
| D-009 | Test files (`tests/*.py`) may write fixture CSV files via `df.to_csv()` when the sole purpose is to provide input to `repo.import_csv()` or `repo.read_csv_raw()` | There is no mechanism to test CSV import without first creating a CSV file. This is fixture setup, not an abstraction bypass. Documented as a repo-guardian permitted exception. | §5.0 |
| D-010 | Gateway nginx.conf switched from `upstream` blocks to `set $var` + `resolver 127.0.0.11 valid=5s` | Static `upstream` blocks resolve DNS once at nginx start; when api/frontend containers are recreated they get new IPs, causing 502. Docker's embedded DNS (127.0.0.11) + variable-based `proxy_pass` re-resolves on each request, fixing stale-IP 502s after container restarts. | §9.1 |
| D-011 | Migration files under `api/migrations/` may import `sqlite3` and call `.execute()` directly, outside the repository layer. **BOUNDARY (does not generalise):** scoped strictly to DDL/migration files whose sole purpose is schema creation. Does NOT extend to route handlers, agents, ML routines, UI handlers, or any runtime read/write path — those remain bound by Rule 1 and must go through the repository. Any future use of `sqlite3` in runtime code justified by analogy to migrations is a Rule 1 violation, not a precedent. | Schema DDL is one-time bootstrap infrastructure that creates the database the repository then owns; it is not application/business logic and cannot depend on the repository it provisions. | §5.0 |

---

## Phase 2

| ID | Decision | Reason | BRD Reference |
|----|----------|--------|-----------------|
| D-012 | OD-1 resolved: XGBoost sensing model pools series by product tier (5 models: entry / mid / upper / premium / luxury); `sku_id` and `state_code` added as categorical features. Logged at start of Phase 2. | BRD OD-1 recommendation is pool-by-tier for v1; BRD §4.3 specifies "grouped by product tier"; per-series option (Option A) yields only ~130 training rows per XGBoost model — insufficient for a boosted tree; tier-pooling yields 1,560–3,120 rows per model (12–24× more). | §4.3, §13 OD-1 |
| D-013 | Phase 2 baseline computes per-week APE/bias via leave-last-12-out backtest and persists results to `accuracy_metrics` (formally Phase 5 scope per BRD §10). | Without this check there is no way to validate BRD §11 synthetic-data calibration (expected 8–18 % MAPE band, some series > 30 %) before building the XGBoost sensing layer on top. This is a measurement-only step; no champion/challenger lifecycle or retraining logic is invoked. | §10, §11, §4.7 |
| D-014 | **DATA-REALISM CONCERN (partially resolved):** Holt-Winters achieves MAPE min 7.8 % / median 15.0 % / p90 20.3 % / max 23.5 % — zero series breach 30 %. XGBoost sensing (P2-4) achieves 15.2 % overall — also zero series over 30 %. Root cause: lognormal σ = 0.12 noise is insufficient to push any series above 30 % MAPE. **Resolution confirmed: Option C required** — add 2–3 `SKU_LUX_FIXTURE_*` structurally-difficult series before Phase 5 G3 gate implementation. Do NOT implement now; flag is recorded here. | §11, §7, §4.10 |
| D-015 | Auto-ARIMA implemented as a statsmodels ARIMA grid-search over 8 candidate orders (instead of pmdarima) because pmdarima was unavailable at local build time (no network). `pmdarima>=2.0.4` added to `worker/requirements.txt`; Docker builds will install it. The grid-search selects the order with lowest AIC on the train window, identical in principle to pmdarima's stepwise mode with a capped search space. The model_type stored in `model_registry` is `'auto_arima'` to match the schema CHECK constraint. | §4.2 |
| D-016 | `synthetic.py` writes `weather_data.csv` alongside `weather_data.json` so that `signals.py` can load all three signal files via `repo.read_csv_raw()` — no new AbstractRepository method required. The JSON file is retained for any future consumer that needs JSON format. This is the only Rule 1-compliant path: `signals.py` cannot call `open()` or `json.load()` directly. | §4.4, §5.0 |
| D-017 | Phase 2 intra-cycle step states added to `PipelineState`: INGESTING → BASELINING → LOADING_SIGNALS → SENSING → SCORING → CYCLE_COMPLETE. `scoring.py` (Module 5) scores `demand_sensing_output` vs `actuals` only; `baseline_forecast` is excluded because those rows cover HORIZON weeks beyond `sales_history` — no actuals exist for them. Baseline holdout accuracy was already persisted to `accuracy_metrics` by `baseline.py`'s leave-last-12-out backtest. | Step states give Phase 3 gate insertion points without restructuring `run_job()`. Scoring scope limited to weeks with available actuals (the last 12 weeks of `sales_history` where sensing holdout and actuals overlap). | §3, §4.7 |

---

## Phase 3  ✅ COMPLETE

| ID | Decision | Reason | BRD Reference |
|----|----------|--------|---------------|
| D-018 | OD-3 resolved → session + bcrypt (Option A) | BRD §12 and §13 both specify session login against the users table; pilot is 4–8 named Samsung users; SSO becomes load-bearing at the second tenant, not the pilot. | §12, §13, OD-3 |
| D-019 | Module 4 LLM event calendar DEFERRED to Phase 4 | Prior standing instruction (no LLM in early phases); BRD §12 static fallback satisfies the G1 DoD; Phase 3 ships G1-complete with a static promotions/event calendar, not LLM-complete. **Conscious BRD deviation.** | §10, §12 |
| D-020 | Promotions write endpoints (POST/PATCH) restricted to `commercial_head` | BRD §2.1 RBAC matrix does not explicitly list which roles may create/edit promotions in the ledger. `commercial_head` owns the Promotions Calendar (G1) per BRD §2 and is the natural write owner. Default chosen per standing instruction: "if unspecified, default to commercial_head and log it." GET and ai-draft are open to all authenticated roles. | §2.1, §12 |
| D-021 | `POST /gates/G1/{cycle_id}/approve` is idempotent — second call returns 200 with current state and writes no new audit row | BRD §3 does not specify behaviour on double-approve. 200 no-op is preferred over 409: the gate is already in the desired state, so re-approving succeeds vacuously. This prevents duplicate `GATE_APPROVED` audit rows that would confuse G1 audit reviewers. The idempotency guard is a pre-flight `get_gate_status` check before calling `set_gate_status`. | §3, §12 |
| D-022 | `run_job()` returns immediately when G1 is unapproved (status='blocked'); poll loop also picks up job_queue rows with status='blocked' on every tick | Worker is single-threaded; sleeping in-process would block ALL cycles. Returning immediately releases the worker so other cycles can proceed. Blocked jobs re-enter the poll queue automatically at the existing POLL_INTERVAL cadence — no additional scheduler needed. On re-pick: if gate still unapproved, the function returns without any state writes (idempotent no-op, no duplicate audit rows). If gate approved, the function resumes from SENSING, skipping modules 1–3. | §3, §9.1 |
| D-023 | SQLite database stored on a **Docker named volume** (`dst_data:` with no `driver_opts`), NOT a macOS bind-mount. Two related fixes shipped together: (a) `PRAGMA busy_timeout=5000` added to every connection so concurrent api+worker access retries for 5 s rather than failing immediately; (b) `_process_job()` wraps the full job-processing sequence in a try/except so any exception (including OS-level I/O errors) marks the job terminal instead of leaving it 'queued' for infinite retry. **Implication for DB reset:** `docker compose down -v && docker compose up --build` (not `rm -f data/dst.db`). | Root cause: `driver_opts: type:none, o:bind, device:./data` is a bind-mount through Docker Desktop's osxfs/VirtioFS layer which does not implement POSIX advisory file locks correctly — SQLite WAL mode's `flock()` calls hit ERRNO 35 (Resource deadlock avoided) and disk I/O errors under concurrent api+worker access. Named volumes use Docker Desktop's internal Linux VM ext4 filesystem where `flock()` works correctly. The `./data/` host directory is no longer used at runtime. | §9.1 |

---

## Phase 4  *(not started)*

*(rows added here when Phase 4 build begins)*

---

## Phase 5  *(not started)*

*(rows added here when Phase 5 build begins)*

---

## Open Decisions (BRD §13 — unresolved)

| ID | Description | Recommended default | Must decide before |
|----|-------------|--------------------|--------------------|
| OD-1 | **Sensing granularity.** Train per individual sku×state (sparse) vs pool by product tier with sku/state as categorical features (more stable) | ~~Pool by tier for v1~~ **RESOLVED as D-012** | ~~Phase 2~~ ✅ |
| OD-2 | **MLflow vs custom registry.** MLflow adds container overhead; custom `model_registry` table is lighter | Start with custom tables; adopt MLflow if challenger volume grows | Phase 5 |
| OD-3 | **Auth depth for v1.** Session + bcrypt against `users` table vs full SSO/SAML | ~~Simple session + hashed password is sufficient for demo/pilot~~ **RESOLVED as D-018** | ~~Phase 3~~ ✅ |
