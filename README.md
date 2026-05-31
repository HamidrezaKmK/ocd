# OCD — Optimized Cash Dashboard

A fully-local, **personalized** credit-card spending analyzer. Drop your statement PDFs in a
folder; OCD extracts each purchase, categorizes it into **categories you define** (each with a
description and a monthly limit), lets you review and correct the categorization with a
human-in-the-loop step, and generates an interactive spending report — all on your machine, using
open-source models. No data leaves the device.

## How it works — three steps

```
 Step 1  extract     PDFs (data/statements/) ─► data/transactions_raw.csv     [deterministic monopoly; LLM fallback]
 Step 2  categorize  raw ─► local LLM ─► review/correct ─► finalize           [Ollama + you]
 Step 3  report      finalized data ─► interactive HTML + Markdown report      [Plotly]
```

- **Robust extraction:** Step 1 parses each PDF deterministically with `monopoly` first; if that
  fails or finds nothing (an unrecognized layout), it falls back to reading the PDF text with the
  local LLM and emitting the same transaction CSV. Disable with `ocd extract --no-llm-fallback`.
- **Personalized:** you define the categories, their descriptions, and monthly limits. The
  descriptions are given to the model, so it categorizes the way *you* think — not by generic rules.
- **Human-in-the-loop:** after the automatic pass, OCD flags rows that need attention — low
  confidence, over a category's monthly limit, **conflicts with your previous run** (a merchant you
  categorized differently last time), and brand-new merchants. Your corrections are remembered, so
  future runs get more accurate and more deterministic.
- **Gated:** the report (Step 3) only runs once you've *finalized* the categorization (Step 2).
- **Normalized budget view:** alongside the dollar charts, the report plots each category's trailing
  30-day spend as a **ratio of its monthly limit** (1.0 = at budget), so over-spending stands out on a
  single shared scale regardless of category size.
- **Swappable models:** every model is chosen per role in `config/models.yaml`. Default is Ollama
  (`qwen2.5:7b-instruct`); point it at a bigger model or a local vLLM server with a one-line edit.

## Setup (one time)

This project runs in a conda env that bundles Python 3.11 **and** poppler (which `monopoly` needs):

```bash
conda create -n ocd python=3.11 poppler pkg-config pip -c conda-forge
conda activate ocd
pip install -e .
```

> On HPC nodes an older system `libstdc++` can shadow the env's copy and break `monopoly`
> (`GLIBCXX_3.4.31 not found`). The env ships an activation hook that fixes this by preloading the
> right `libstdc++`, so always work inside `conda activate ocd`.

Install and start a local model server (Ollama):

```bash
# user-space install (no sudo): download the linux tarball from ollama.com, extract, then:
ollama serve &                     # detects local GPUs automatically
ollama pull qwen2.5:7b-instruct
```

Verify everything is ready:

```bash
ocd doctor
```

## Usage

### Web app — upload & report (recommended)

```bash
ocd serve         # starts a local server, then open http://localhost:8000
```

**Multi-user.** Sign up with a username + password and log in; each account gets its own workspace at
`data/users/<user>/`. The single-page app walks the full pipeline as four steps:

1. **⚙️ Preferences** — add/edit/remove categories (name, description, monthly limit). The description
   is sent to the model, so it categorizes the way *you* think.
2. **📄 Statements** — drag in PDFs (they persist in your folder), then **Analyze** with **live
   per-stage progress**: upload %, extract (per file), categorize (per merchant) — streamed over SSE.
3. **📝 Review & correct** — fix any categories the model got wrong (flagged rows highlighted), then
   finalize.
4. **📊 Report** — the interactive dashboard, built on finalize.

Everything runs on this machine (monopoly extract + local Ollama categorize + report); nothing leaves
your computer. A demo account ships ready to try: **`synthetic` / `synthetic`**.

Both the web app and the CLI go through one shared service layer (`ocd.service`) running each user's
pipeline in their own workspace (`data/users/<user>/`) — there is no separate UI logic. The backend
URL is configurable (page settings or `?api=`), so the same frontend can later point at a remote server.

> Auth is lightweight (PBKDF2-hashed passwords in a git-ignored `data/users/accounts.yaml`, in-memory
> sessions) — good for a trusted local/LAN setup. Put it behind HTTPS before exposing it publicly.

### Public demo via GitHub Pages + ngrok

The frontend is hosted at **[hamidrezakmk.github.io/ocd](https://hamidrezakmk.github.io/ocd/)**.
All processing still runs on your machine; the tunnel just makes it reachable publicly.

```bash
# Terminal 1 — model server
ollama serve

# Terminal 2 — app backend (cross-origin mode)
OCD_CORS_ORIGINS=https://hamidrezakmk.github.io ocd serve

# Terminal 3 — public tunnel
ngrok http --url=https://cufflink-wisdom-gnat.ngrok-free.dev 8000
```

Share **`https://hamidrezakmk.github.io/ocd/`** — stop the tunnel to take it offline.

To restart after a shutdown, just re-run the three commands above in order.

### Command line (single-user)

```bash
ocd setup                       # define your categories (name / description / monthly limit)
# put statement PDFs in data/statements/
ocd extract                     # Step 1
ocd categorize                  # Step 2 auto pass (prints rows needing attention)
ocd review --finalize           # Step 2 finalize (or review/correct in `ocd serve`)
ocd report                      # Step 3 ─► reports/report_<period>.html + .md

ocd pipeline                    # run all three steps end-to-end (used by `ocd serve`)
ocd serve                       # local web server (upload PDFs in the browser)
```

## Try it without real statements

Real bank statements are private, and the sample PDFs banks publish online are graphical mockups
that aren't machine-readable. So OCD ships:

```bash
python -m ocd.samples.make_synthetic     # 2 banks × 4 months of realistic, parseable statements
python -m ocd.samples.download_samples    # also fetches real bank-hosted samples (reference only)
```

The synthetic generator gives you a multi-month, two-bank dataset so the spending **trends** and
insights in the report are meaningful end-to-end.

> **Future data source:** [Plaid](https://plaid.com) partners with banks to provide authorized
> financial data and could feed OCD directly down the road. It is intentionally not integrated now —
> OCD stays fully local.

## Configuration

- `config/categories.yaml` — your categories (name, description, monthly_limit).
- `config/models.yaml` — per-role model selection. Example to use vLLM on local GPUs instead of Ollama:
  ```yaml
  classifier: { provider: vllm, base_url: http://localhost:8000/v1,
                model: Qwen/Qwen2.5-14B-Instruct, temperature: 0, api_key: EMPTY }
  ```
- `config/merchant_memory.yaml` — learned merchant→category corrections (grows as you review).

## Privacy

Your statement PDFs, extracted transactions, and reports are git-ignored by default. All parsing and
inference run locally; nothing is sent to any cloud service.

## Layout

```
src/ocd/
  config.py     categories, merchant memory, finalization metadata
  models.py     per-role OpenAI-compatible client factory (Ollama / vLLM / remote)
  extract.py    Step 1 — monopoly PDF parsing
  classify.py   Step 2 auto pass — LLM categorization (structured JSON)
  review.py     Step 2 interactive — flagging, reconciliation, finalization gate
  report.py     Step 3 — Plotly figures, insights, Markdown + standalone HTML
  service.py    shared per-user service layer (the single source of truth both UIs call)
  server.py     FastAPI backend + auth for the multi-user web app
  auth.py       accounts (hashed passwords) + sessions
  web/          single-page frontend served by `ocd serve`
  cli.py        `ocd` command
  samples/      download real samples + synthetic statement generator
```
