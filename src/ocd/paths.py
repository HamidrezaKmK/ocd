"""Centralized filesystem layout for the project.

There are two roots:

* the **base root** — the project (or ``OCD_HOME``) directory that holds ``config/`` and
  ``data/`` and, for the multi-user server, ``data/users/``;
* a **context root** — a per-user workspace pushed temporarily with :func:`use_root`.

Pipeline paths (``CONFIG_DIR``, ``STATEMENTS_DIR``, ``RAW_CSV`` …) resolve against the
*context* root when one is active, else the base root. This lets the service layer run the
whole pipeline in-process for a given user without any module reload::

    with paths.use_root(paths.user_home("hamid")):
        extract.extract_statements(); classify.run_categorize(); ...

Account/user-registry paths (``USERS_DIR``, ``ACCOUNTS_YAML``, :func:`user_home`) always
resolve against the *base* root — they live above any single user's workspace.
"""
from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from pathlib import Path

# Active per-user workspace root (None → use the base root).
_context_root: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "ocd_context_root", default=None)


def base_root() -> Path:
    """Project root: ``OCD_HOME`` if set, else three parents up from this file."""
    env = os.environ.get("OCD_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """The currently active root (context workspace if one is pushed, else base)."""
    return _context_root.get() or base_root()


@contextmanager
def use_root(root: Path):
    """Temporarily resolve pipeline paths against ``root`` (a per-user workspace)."""
    token = _context_root.set(Path(root).resolve())
    try:
        yield
    finally:
        _context_root.reset(token)


def user_home(user: str) -> Path:
    """Per-user workspace root (used as the context root for that user's pipeline)."""
    return base_root() / "data" / "users" / user


# Names resolved against the *context* root (per-user when active).
def _context_paths() -> dict[str, Path]:
    root = project_root()
    config = root / "config"
    data = root / "data"
    return {
        "ROOT": root,
        "CONFIG_DIR": config,
        "DATA_DIR": data,
        "STATEMENTS_DIR": data / "statements",
        "REPORTS_DIR": root / "reports",
        "CATEGORIES_YAML": config / "categories.yaml",
        "MODELS_YAML": config / "models.yaml",
        "MERCHANT_MEMORY_YAML": config / "merchant_memory.yaml",
        "RAW_CSV": data / "transactions_raw.csv",
        "CATEGORIZED_CSV": data / "transactions_categorized.csv",
        "CATEGORIZED_META": data / "categorized_meta.yaml",
        "PREVIOUS_CATEGORIZED_CSV": data / "transactions_categorized_previous.csv",
    }


# Names resolved against the *base* root (the user registry, above any workspace).
def _base_paths() -> dict[str, Path]:
    users = base_root() / "data" / "users"
    return {"USERS_DIR": users, "ACCOUNTS_YAML": users / "accounts.yaml"}


def __getattr__(name: str) -> Path:  # PEP 562 — dynamic module attributes
    ctx = _context_paths()
    if name in ctx:
        return ctx[name]
    base = _base_paths()
    if name in base:
        return base[name]
    raise AttributeError(f"module 'ocd.paths' has no attribute {name!r}")


def ensure_dirs() -> None:
    """Create the directories the pipeline writes to (idempotent), for the active root."""
    p = _context_paths()
    for key in ("CONFIG_DIR", "DATA_DIR", "STATEMENTS_DIR", "REPORTS_DIR"):
        p[key].mkdir(parents=True, exist_ok=True)
