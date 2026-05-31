"""Shared service layer — the single source of truth both front-ends call into.

Every operation runs against a per-user workspace via ``paths.use_root(home)``, so the same
pipeline modules (``extract``, ``classify``, ``review``, ``report``, ``config``) drive the web
app and the CLI alike. No orchestration logic is duplicated in the UI.

``home`` is a per-user workspace root (``paths.user_home(user)``); for single-user CLI use it
is just the project root.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from . import config as cfg
from . import models, paths
from .config import Category, CategoryConfig

# Seeded into a fresh user workspace so categorization starts from sensible defaults.
# (Merchant memory is intentionally *not* seeded — each user learns their own.)
_SEED_CONFIG = ("categories.yaml", "models.yaml")

Event = dict


def ensure_workspace(home: Path) -> None:
    """Create a user's workspace dirs and seed shared config the first time."""
    (home / "data" / "statements").mkdir(parents=True, exist_ok=True)
    (home / "config").mkdir(parents=True, exist_ok=True)
    base_cfg = paths.base_root() / "config"
    for name in _SEED_CONFIG:
        dst, src = home / "config" / name, base_cfg / name
        if not dst.exists() and src.exists():
            shutil.copy(src, dst)


# --------------------------------------------------------------------------- #
# Categories (the interactive "preferences")
# --------------------------------------------------------------------------- #
def get_categories(home: Path) -> list[dict]:
    ensure_workspace(home)
    with paths.use_root(home):
        cats = cfg.load_categories()
    return [{"name": c.name, "description": c.description, "monthly_limit": c.monthly_limit}
            for c in cats.categories]


def save_categories(home: Path, items: list[dict]) -> list[dict]:
    cats: list[Category] = []
    for it in items:
        name = str(it.get("name", "")).strip()
        if not name:
            continue
        cats.append(Category(name=name, description=str(it.get("description", "") or ""),
                             monthly_limit=float(it.get("monthly_limit", 0) or 0)))
    if not cats:
        raise ValueError("Define at least one category.")
    ensure_workspace(home)
    with paths.use_root(home):
        cfg.save_categories(CategoryConfig(categories=cats))
    return get_categories(home)


# --------------------------------------------------------------------------- #
# Statements
# --------------------------------------------------------------------------- #
def list_statements(home: Path) -> list[str]:
    d = home / "data" / "statements"
    return sorted(p.name for p in d.glob("*.pdf")) if d.exists() else []


def save_statements(home: Path, files: dict[str, bytes]) -> int:
    d = home / "data" / "statements"
    d.mkdir(parents=True, exist_ok=True)
    saved = 0
    for filename, data in files.items():
        if filename.lower().endswith(".pdf"):
            (d / Path(filename).name).write_bytes(data)  # .name strips path traversal
            saved += 1
    return saved


def delete_statement(home: Path, name: str) -> bool:
    target = home / "data" / "statements" / Path(name).name
    if target.exists():
        target.unlink()
        return True
    return False


# --------------------------------------------------------------------------- #
# Analyze: extract + categorize (draft). Emits progress events via ``on_event``.
# --------------------------------------------------------------------------- #
def analyze(home: Path, on_event: Callable[[Event], None] = lambda e: None,
            use_memory: bool = True, allow_llm_fallback: bool = True) -> None:
    ensure_workspace(home)
    from .classify import run_categorize
    from .extract import extract_statements

    with paths.use_root(home):
        on_event({"stage": "extract", "status": "start"})
        res = extract_statements(
            allow_llm_fallback=allow_llm_fallback,
            file_cb=lambda i, n, r: on_event({"stage": "extract", "i": i, "n": n, "file": r["file"],
                                              "method": r["method"], "purchases": r["n"],
                                              "status": r["status"]}),
        )
        if res.transactions.empty:
            raise ValueError("No transactions extracted from the uploaded statements.")
        on_event({"stage": "categorize", "status": "start",
                  "n": int(res.transactions["description"].nunique())})
        run_categorize(use_memory=use_memory,
                       progress_cb=lambda i, n, m: on_event({"stage": "categorize", "i": i,
                                                             "n": n, "merchant": m}))
        on_event({"stage": "categorized"})


# --------------------------------------------------------------------------- #
# Review + corrections
# --------------------------------------------------------------------------- #
def get_review(home: Path) -> dict:
    from .review import compute_flags, load_categorized

    with paths.use_root(home):
        cats = cfg.load_categories()
        meta = cfg.load_meta()
        try:
            df = load_categorized()
        except FileNotFoundError:
            return {"ready": False, "rows": [], "over_limit": [], "categories": cats.all_names,
                    "n": 0, "n_attention": 0, "finalized": meta.finalized}
        rs = compute_flags(df, cats)

    rows = []
    for idx, r in rs.df.iterrows():
        rows.append({
            "row_id": int(idx),
            "date": str(r.get("date", "")),
            "description": str(r.get("description", "")),
            "amount": float(r.get("amount", 0) or 0),
            "category": str(r.get("category", "")),
            "confidence": round(float(r.get("confidence", 1) or 1), 2),
            "needs_attention": bool(r.get("needs_attention", False)),
            "attention_reasons": str(r.get("attention_reasons", "") or ""),
            "source_file": str(r.get("source_file", "")),
        })
    return {"ready": True, "rows": rows, "over_limit": rs.over_limit,
            "categories": cats.all_names, "n": len(rows), "n_attention": rs.n_attention,
            "finalized": meta.finalized}


def apply_corrections(home: Path, corrections: dict) -> dict:
    from .review import apply_corrections as _apply
    from .review import load_categorized, save_draft

    corr = {int(k): str(v) for k, v in corrections.items() if str(v).strip()}
    with paths.use_root(home):
        df2 = _apply(load_categorized(), corr)
        save_draft(df2)
    return get_review(home)


# --------------------------------------------------------------------------- #
# Finalize + report
# --------------------------------------------------------------------------- #
def finalize_and_report(home: Path) -> str:
    """Finalize the (corrected) categorization and return the report HTML."""
    from .report import generate_report
    from .review import finalize as _finalize
    from .review import load_categorized

    with paths.use_root(home):
        cats = cfg.load_categories()
        _finalize(load_categorized(), cats)
        out = generate_report(require_finalized=True)
        return Path(out["html"]).read_text()


def model_health() -> dict:
    ok, msg = models.health_check("classifier")
    return {"ok": ok, "message": msg}
