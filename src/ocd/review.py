"""Step 2 (interactive loop) — human-in-the-loop review, reconciliation, and finalization.

This module is the brain behind the review UI. It:
  * flags rows that need attention (low confidence, over a category's monthly limit,
    conflicts vs the previous finalized run, brand-new merchants),
  * applies the user's category corrections,
  * and finalizes the run — snapshotting the current categorization as the "previous" one,
    folding corrections into the persistent merchant memory, and flipping the finalized gate
    so Step 3 (report) is allowed to run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config as cfg
from . import paths
from .config import UNCATEGORIZED, CategoryConfig

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_THRESHOLD = 0.6

FLAG_COLUMNS = ["flag_low_conf", "flag_over_limit", "flag_conflict", "flag_new",
                "prev_category", "needs_attention", "attention_reasons"]


@dataclass
class ReviewItem:
    index: int
    date: str
    description: str
    amount: float
    category: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    prev_category: Optional[str] = None


@dataclass
class ReviewState:
    df: pd.DataFrame
    items: list[ReviewItem]
    over_limit: list[dict]  # [{category, month_label, spent, limit}]

    @property
    def n_attention(self) -> int:
        return len(self.items)


def load_categorized(path: Optional[Path] = None) -> pd.DataFrame:
    path = Path(path) if path is not None else paths.CATEGORIZED_CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `ocd categorize` first.")
    return pd.read_csv(path)


def _previous_merchant_map(prev_df: Optional[pd.DataFrame]) -> dict[str, str]:
    """Map merchant_key -> most common category in the previous finalized run."""
    if prev_df is None or prev_df.empty or "merchant_key" not in prev_df.columns:
        return {}
    out: dict[str, str] = {}
    for key, grp in prev_df.groupby("merchant_key"):
        out[key] = grp["category"].mode().iloc[0]
    return out


def compute_over_limit(df: pd.DataFrame, categories: CategoryConfig) -> list[dict]:
    """Per (category, month) totals that exceed the category's monthly limit."""
    if df.empty:
        return []
    grouped = df.groupby(["category", "month_label"])["amount"].sum().reset_index()
    out = []
    for _, row in grouped.iterrows():
        limit = categories.limit_for(row["category"])
        if limit and row["amount"] > limit:
            out.append({
                "category": row["category"],
                "month_label": row["month_label"],
                "spent": round(float(row["amount"]), 2),
                "limit": float(limit),
                "over_by": round(float(row["amount"]) - float(limit), 2),
            })
    return out


def compute_flags(
    df: pd.DataFrame,
    categories: Optional[CategoryConfig] = None,
    prev_df: Optional[pd.DataFrame] = None,
    low_conf_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> ReviewState:
    """Annotate ``df`` with attention flags and return a ReviewState."""
    categories = categories or cfg.load_categories()
    if prev_df is None and paths.PREVIOUS_CATEGORIZED_CSV.exists():
        prev_df = pd.read_csv(paths.PREVIOUS_CATEGORIZED_CSV)

    df = df.copy()
    prev_map = _previous_merchant_map(prev_df)

    over_limit = compute_over_limit(df, categories)
    over_limit_keys = {(o["category"], o["month_label"]) for o in over_limit}

    # confidence column may be absent for memory-only runs; default to 1.0
    conf = pd.to_numeric(df.get("confidence", 1.0), errors="coerce").fillna(1.0)
    by = df.get("classified_by", "llm")

    df["flag_low_conf"] = (conf < low_conf_threshold) & (by != "memory")
    df["flag_over_limit"] = [
        (c, m) in over_limit_keys for c, m in zip(df["category"], df["month_label"])
    ]
    df["prev_category"] = df["merchant_key"].map(prev_map)
    have_prev = bool(prev_map)
    df["flag_conflict"] = (
        df["prev_category"].notna() & (df["prev_category"] != df["category"])
    )
    df["flag_new"] = have_prev & df["prev_category"].isna()

    items: list[ReviewItem] = []
    reasons_col: list[str] = []
    attention_col: list[bool] = []
    for idx, row in df.iterrows():
        reasons = []
        if row["flag_low_conf"]:
            reasons.append(f"low confidence ({conf[idx]:.2f})")
        if row["flag_over_limit"]:
            reasons.append("category over monthly limit")
        if row["flag_conflict"]:
            reasons.append(f"was '{row['prev_category']}' last time")
        if row["flag_new"]:
            reasons.append("new merchant")
        if row["category"] == UNCATEGORIZED:
            reasons.append("uncategorized")
        reasons_col.append("; ".join(reasons))
        attention_col.append(bool(reasons))
        if reasons:
            items.append(ReviewItem(
                index=int(idx), date=str(row["date"]), description=str(row["description"]),
                amount=float(row["amount"]), category=str(row["category"]),
                confidence=float(conf[idx]), reasons=reasons,
                prev_category=(None if pd.isna(row["prev_category"]) else str(row["prev_category"])),
            ))
    df["needs_attention"] = attention_col
    df["attention_reasons"] = reasons_col

    # Sort attention items: conflicts and over-limit first, then low confidence, then new.
    def _priority(it: ReviewItem) -> tuple:
        return (
            0 if any("last time" in r for r in it.reasons) else
            1 if any("over monthly" in r for r in it.reasons) else
            2 if any("low confidence" in r or "uncategorized" in r for r in it.reasons) else 3,
            -it.amount,
        )
    items.sort(key=_priority)
    return ReviewState(df=df, items=items, over_limit=over_limit)


def apply_corrections(df: pd.DataFrame, corrections: dict[int, str]) -> pd.DataFrame:
    """Apply ``{row_index: new_category}`` edits, returning the updated DataFrame."""
    df = df.copy()
    for idx, new_cat in corrections.items():
        if idx in df.index and new_cat:
            df.at[idx, "category"] = new_cat
            # A manual edit is authoritative.
            if "confidence" in df.columns:
                df.at[idx, "confidence"] = 1.0
            if "classified_by" in df.columns:
                df.at[idx, "classified_by"] = "user"
    return df


def save_draft(df: pd.DataFrame, path: Optional[Path] = None) -> None:
    """Persist edits without finalizing (keeps the gate closed)."""
    paths.ensure_dirs()
    df.to_csv(path if path is not None else paths.CATEGORIZED_CSV, index=False)
    cfg.mark_draft(n_transactions=len(df))


def finalize(df: pd.DataFrame, categories: Optional[CategoryConfig] = None) -> cfg.RunMeta:
    """Finalize categorization: persist the CSV, snapshot it as the previous run, fold the
    merchant->category map into memory, and flip the finalized gate."""
    categories = categories or cfg.load_categories()
    paths.ensure_dirs()

    # Drop transient flag columns from the persisted file (recomputable on load).
    persist = df.drop(columns=[c for c in FLAG_COLUMNS if c in df.columns], errors="ignore")
    persist.to_csv(paths.CATEGORIZED_CSV, index=False)
    # Snapshot as the previous finalized run for next time's reconciliation.
    persist.to_csv(paths.PREVIOUS_CATEGORIZED_CSV, index=False)

    # Learn every merchant->category mapping (skip Uncategorized so it stays re-askable).
    corrections = {}
    for key, grp in persist.groupby("merchant_key"):
        cat = grp["category"].mode().iloc[0]
        if cat and cat != UNCATEGORIZED:
            corrections[str(key)] = str(cat)
    cfg.remember_merchants(corrections)

    period = None
    if "month_label" in persist.columns and not persist.empty:
        months = sorted(persist["month_label"].dropna().unique())
        if months:
            period = months[0] if len(months) == 1 else f"{months[0]}_to_{months[-1]}"

    meta = cfg.mark_finalized(period=period, n_transactions=len(persist))
    logger.info("Finalized %d transactions (period=%s); learned %d merchant mappings.",
                len(persist), period, len(corrections))
    return meta
