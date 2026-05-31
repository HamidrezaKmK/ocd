# OCD — Optimized Cash Dashboard

A fully-local, **personalized** credit-card spending analyzer. Drop your statement PDFs in a
folder; OCD extracts each purchase, categorizes it into **categories you define** (each with a
description and a monthly limit), lets you review and correct the categorization with a
human-in-the-loop step, and generates an interactive spending report — all on your machine, using
open-source models. No data leaves the device.

## How it works — three steps

```
 Step 1  extract     PDFs (data/statements/) ─► data/transactions_raw.csv     [deterministic, monopoly]
 Step 2  categorize  raw ─► local LLM ─► review/correct ─► finalize           [Ollama + you]
 Step 3  report      finalized data ─► interactive HTML + Markdown report      [Plotly]
```

- **Personalized:** you define the categories, their descriptions, and monthly limits. The
  descriptions are given to the model, so it categorizes the way *you* think — not by generic rules.
- **Human-in-the-loop:** after the automatic pass, OCD flags rows that need attention — low
  confidence, over a category's monthly limit, **conflicts with your previous run** (a merchant you
  categorized differently last time), and brand-new merchants. Your corrections are remembered, so
  future runs get more accurate and more deterministic.
- **Gated:** the report (Step 3) only runs once you've *finalized* the categorization (Step 2).
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

### Web UI (recommended)

```bash
ocd app          # opens a local Streamlit app: Setup ─► Extract ─► Categorize & review ─► Report
```

### Command line

```bash
ocd setup                       # define your categories (name / description / monthly limit)
# put statement PDFs in data/statements/
ocd extract                     # Step 1
ocd categorize                  # Step 2 auto pass (prints rows needing attention)
ocd review --finalize           # Step 2 finalize (or review/correct in `ocd app`)
ocd report                      # Step 3 ─► reports/report_<period>.html + .md
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
  app.py        Streamlit UI
  cli.py        `ocd` command
  samples/      download real samples + synthetic statement generator
```
