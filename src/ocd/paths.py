"""Centralized filesystem layout for the project.

All paths are resolved relative to the project root (the directory that contains
``config/`` and ``data/``). The root can be overridden with the ``OCD_HOME`` env var,
which is useful for tests and for running the Streamlit app from another directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("OCD_HOME")
    if env:
        return Path(env).expanduser().resolve()
    # src/ocd/paths.py -> repo root is three parents up
    return Path(__file__).resolve().parents[2]


ROOT = project_root()

CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
STATEMENTS_DIR = DATA_DIR / "statements"
REPORTS_DIR = ROOT / "reports"

# Config files
CATEGORIES_YAML = CONFIG_DIR / "categories.yaml"
MODELS_YAML = CONFIG_DIR / "models.yaml"
MERCHANT_MEMORY_YAML = CONFIG_DIR / "merchant_memory.yaml"

# Data artifacts
RAW_CSV = DATA_DIR / "transactions_raw.csv"
CATEGORIZED_CSV = DATA_DIR / "transactions_categorized.csv"
CATEGORIZED_META = DATA_DIR / "categorized_meta.yaml"
# Snapshot of the last *finalized* run, used for cross-run reconciliation.
PREVIOUS_CATEGORIZED_CSV = DATA_DIR / "transactions_categorized_previous.csv"


def ensure_dirs() -> None:
    """Create the directories the pipeline writes to (idempotent)."""
    for d in (CONFIG_DIR, DATA_DIR, STATEMENTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
