# CLAUDE.md

Guidance for working in this repo. For the user-facing overview see [README.md](README.md);
this file is for someone (or an agent) editing the code.

## What this is

**OCD — Optimized Cash Dashboard**: a fully-local, personalized credit-card spending analyzer.
Statement PDFs in → extracted, LLM-categorized into *user-defined* categories → reviewed and
corrected by a human → interactive HTML/Markdown spending report. Everything runs on-device
(local Ollama/vLLM via an OpenAI-compatible API); no data is meant to leave the machine.

## The pipeline (three gated steps)

```
Step 1  extract     data/statements/*.pdf ─► data/transactions_raw.csv          [monopoly+poppler; LLM fallback]
Step 2  categorize  raw ─► local LLM (per merchant) ─► review/correct ─► finalize [Ollama + human-in-the-loop]
Step 3  report      finalized CSV ─► Plotly HTML + Markdown                        [gated on finalization]
```

The **finalization gate** is central: Step 3 refuses to run unless `data/categorized_meta.yaml`
has `finalized: true` (set by `review.finalize`, pass `--draft` / "preview draft" to bypass).
Finalizing also (a) snapshots the run to `transactions_categorized_previous.csv` for next-run
conflict detection and (b) folds confirmed merchant→category mappings into
`config/merchant_memory.yaml` so future runs are more deterministic.

## Module map (`src/ocd/`)

| File | Responsibility |
|---|---|
| `paths.py` | Single source of truth for all filesystem paths. Root override via `OCD_HOME`. |
| `config.py` | Pydantic models + YAML I/O for categories, merchant memory, and run metadata (the gate). |
| `models.py` | Role-based model selection (`classifier`/`extractor`/`insights`/`embeddings`/`ocr`). One OpenAI-compatible client. |
| `extract.py` | Step 1. Deterministic monopoly parsing first; LLM fallback (`extractor` role) reads the PDF text when monopoly raises or finds nothing. Purchase filtering. |
| `classify.py` | Step 2 auto pass. Per-unique-merchant LLM classification → `{category, confidence, rationale}`. |
| `review.py` | Step 2 interactive. Attention flags, corrections, finalization. |
| `report.py` | Step 3. Aggregation, Plotly figures (incl. `fig_budget_ratio` = trailing-30-day spend ÷ monthly limit), rule-based insights (+ optional LLM summary), HTML/MD render. |
| `service.py` | **Shared service layer** — the single source of truth both the web app and CLI call. Per-user ops (categories, statements, analyze, review, corrections, finalize+report) each run under `paths.use_root(user_home)`. |
| `server.py` | FastAPI backend + auth for `ocd serve`. Thin HTTP glue over `service`. `/api/analyze/stream` streams per-stage progress (SSE, worker-thread + queue). |
| `auth.py` | Accounts (PBKDF2 password hashing → `data/users/accounts.yaml`) + in-memory sessions. Stdlib only. |
| `web/index.html` | Single-page frontend: login gate + 4 sections (Preferences, Statements, Review&correct, Report). The only UI. |
| `cli.py` | Typer `ocd` command (entry point `ocd = ocd.cli:app`). |
| `samples/` | `make_synthetic` (parseable demo statements) + `download_samples` (real bank samples, reference only). |

## Commands

```bash
# Environment — conda is required because monopoly needs poppler (not a pip wheel):
conda create -n ocd python=3.11 poppler pkg-config pip -c conda-forge
conda activate ocd && pip install -e .

ocd doctor              # verify poppler/monopoly import + model endpoint reachable
ocd setup               # define categories (also writes config/models.yaml)
ocd extract             # Step 1
ocd categorize          # Step 2 auto pass  (--finalize to skip review; --no-memory to ignore learned map)
ocd review --finalize   # Step 2 finalize from terminal
ocd report              # Step 3  (--draft to allow a non-finalized run)
ocd pipeline            # all steps end-to-end (single-user CLI; extract→categorize→finalize→report)
ocd serve               # FastAPI multi-user web app on :8000 — the primary UI (Streamlit is retired)

python -m ocd.samples.make_synthetic     # 2 banks × 4 months of demo statements → data/statements/
```

There is **no test suite** and **no linter config** yet. The fastest end-to-end smoke test is:
generate synthetic statements → `ocd extract` → `ocd categorize --finalize` → `ocd report --draft`.

## Conventions to preserve

- **Model access is always role-based and config-driven.** Never hardcode a provider, base_url, or
  model name in pipeline code — go through `models.get_role_config(role)` / `get_client(role)`.
  Swapping Ollama→vLLM→remote must stay a `config/models.yaml` edit only. Roles speak the
  OpenAI-compatible API; `use: <role>` indirection lets one role reuse another's config.
- **Heavy deps are imported lazily** inside functions (`monopoly` in `extract.parse_pdf`, `openai`
  in `models.get_client`). Keep this so the package imports cheaply and `ocd doctor` can diagnose.
- **"One bad input must not kill the batch."** `parse_pdf` and `classify_merchant` catch broadly and
  return an error/fallback record instead of raising. Preserve that resilience.
- **Merchant identity = `classify.normalize_merchant(description)`.** It strips digits and trailing
  geo tokens so identical merchants share one classification, one cache entry, and one memory key.
  Classification, memory, and cross-run reconciliation all key off `merchant_key`.
- **Purchase detection is sign-agnostic** (`extract._is_purchase`): banks disagree on +/- sign, so
  credits/payments/refunds are identified by polarity marker + description regex, and spend is stored
  as `abs(amount)`. Don't reintroduce sign-based logic.
- **`Uncategorized` is always available** and is intentionally *not* written to merchant memory, so
  uncertain merchants stay re-askable.
- **Flag columns are transient** (`review.FLAG_COLUMNS`): recomputed by `compute_flags` on load and
  dropped before persisting. The on-disk categorized CSV holds only stable columns.
- All persisted artifacts are YAML (config/metadata) or CSV (transactions). Stick to those.
- **One shared service layer; per-user paths via context.** All real work lives in `service.py`,
  called by both `server.py` and (single-user) the CLI — never reimplement pipeline orchestration in
  the UI. Per-user isolation is done **in-process** with `paths.use_root(home)`: `paths.py` resolves
  pipeline paths (`CONFIG_DIR`, `STATEMENTS_DIR`, `CATEGORIZED_CSV`, …) against a `contextvars` root,
  so the same modules read/write `data/users/<user>/…` without any reload or subprocess. **Gotcha:**
  don't reintroduce module-level path constants or `paths.X` *default arguments* (they'd bind once at
  import and ignore the context) — resolve inside the function body. `USERS_DIR`/`ACCOUNTS_YAML`/
  `user_home()` stay on the **base** root, never the context root.
- **Progress streaming.** `service.analyze` takes an `on_event` callback; `/api/analyze/stream` runs it
  on a worker thread, pushing events onto a queue that the SSE generator drains
  (`stage ∈ extract|categorize|categorized|error`). The worker re-enters `paths.use_root(home)` itself
  because contextvars don't propagate to manually-spawned threads.
- **Auth lives entirely in `auth.py`** (stdlib PBKDF2, accounts YAML, in-memory sessions). `data/users/`
  holds password hashes + real statements and is git-ignored — never commit it. The seeded demo account
  is `synthetic` / `synthetic`. Auth is lightweight (HTTP, in-memory sessions): fine for local/LAN,
  not a hardened public service.

## Data contracts

- `transactions_raw.csv` columns: `extract.RAW_COLUMNS`.
- categorized CSV adds: `classify.CATEGORIZED_COLUMNS_EXTRA` (`category, confidence, rationale,
  classified_by, merchant_key, year, month, month_label`). `classified_by ∈ {memory, llm, fallback, user}`.
- `config/categories.yaml` → `{categories: [{name, description, monthly_limit}]}` (description feeds the prompt).
- `config/merchant_memory.yaml` → `{merchants: {<merchant_key>: <category>}}` (grows on finalize).
- `data/categorized_meta.yaml` → `RunMeta` (`finalized, finalized_at, period, n_transactions`) — the gate.

## Gotchas

- **poppler / `libstdc++`**: monopoly needs poppler; on HPC an old system `libstdc++` can shadow the
  env (`GLIBCXX … not found`). Always work inside `conda activate ocd` (its activation hook fixes the
  preload). `ocd doctor` reports this clearly.
- **Privacy vs. git reality** ⚠️: the README says statement PDFs / transactions / reports are
  git-ignored, but they are **not** — `data/transactions_*.csv`, `config/merchant_memory.yaml`, and
  `data/statements/*.pdf` are tracked, and real statements added to `data/statements/` are untracked
  but would be caught by `git add .`. The committed `merchant_memory.yaml` and CSVs already contain
  **real** merchant/transaction data. Before committing, double-check you are not adding real
  financial data; the `.gitignore` needs fixing to actually exclude `data/statements/`,
  `data/transactions_*.csv`, and learned `config/merchant_memory.yaml`. (A stale, wrong-path
  `statements/username/` entry exists in `.gitignore` from an earlier layout — the real path is
  `data/statements/`.)
- **Python is pinned to 3.11** (`requires-python = "==3.11.*"`); poppler/monopoly compatibility.
- **Local model server must be up** for Step 2 (and Step 3's optional summary): `ollama serve` +
  `ollama pull qwen2.5:7b-instruct`, or point `config/models.yaml` at a vLLM endpoint.
- **`wandb` is optional** (`[project.optional-dependencies] tracking`), not a core dependency.
```
