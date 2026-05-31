"""Step 2 (auto pass) — categorize each transaction with a local LLM.

For every *unique* normalized merchant we either reuse a user-confirmed category from the
persistent merchant memory (deterministic, no model call) or ask the classifier model to pick
the best-fitting user category, returning a structured ``{category, confidence, rationale}``.
Results are written as a *draft* categorized CSV (``finalized: false``); the interactive review
step (review.py) is what eventually finalizes it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from . import config as cfg
from . import models, paths
from .config import UNCATEGORIZED, CategoryConfig

logger = logging.getLogger(__name__)

CATEGORIZED_COLUMNS_EXTRA = [
    "category", "confidence", "rationale", "classified_by",
    "merchant_key", "year", "month", "month_label",
]

_DIGIT_RUN = re.compile(r"\d{2,}")
_PUNCT = re.compile(r"[^A-Z0-9&/ ]+")
_WS = re.compile(r"\s+")
# Trailing location noise common on card statements (country/state codes, "SINGAPORE SG").
_TRAILING_GEO = re.compile(
    r"\b(SINGAPORE|SG|USA?|US|GBR?|UK|CANADA|CA|NY|CA|TX|FL|WA|IL|MA|ONLINE|LLC|INC|LTD)\b\s*$",
    re.IGNORECASE,
)


def normalize_merchant(description: str) -> str:
    """Collapse a raw statement description to a stable merchant key so identical merchants
    share one classification (and one cache entry)."""
    s = (description or "").upper().strip()
    s = _DIGIT_RUN.sub(" ", s)          # drop auth/card numbers
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    for _ in range(3):                  # peel a few trailing geo tokens
        new = _TRAILING_GEO.sub("", s).strip()
        if new == s:
            break
        s = new
    return s or (description or "").strip().upper()


@dataclass
class MerchantVerdict:
    category: str
    confidence: float
    rationale: str
    classified_by: str  # 'memory' | 'llm' | 'fallback'


def _classification_schema(names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": names},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string"},
        },
        "required": ["category", "confidence", "rationale"],
        "additionalProperties": False,
    }


def _system_prompt(categories: CategoryConfig,
                   examples: Optional[dict[str, str]] = None) -> str:
    lines = [
        "You are a precise personal-finance assistant that categorizes credit-card purchases.",
        "Assign each merchant to exactly ONE of the user's categories below, based on the",
        "category descriptions. These categories are personal to this user — follow their",
        "descriptions, not generic conventions.",
        "",
        "Categories:",
    ]
    for c in categories.categories:
        lines.append(f"- {c.name}: {c.description}")
    lines.append(f"- {UNCATEGORIZED}: use only if no category plausibly fits.")
    if examples:
        # User-confirmed labels — generalize this intent to similar merchants.
        lines.append("")
        lines.append("The user has CONFIRMED these merchant → category labels. Treat them as ground "
                     "truth and categorize similar merchants consistently with this intent:")
        for merchant, cat in list(examples.items())[:40]:
            lines.append(f"- {merchant} → {cat}")
    lines.append("")
    lines.append("Return strict JSON: {\"category\": <one of the names above>, "
                 "\"confidence\": <0..1>, \"rationale\": <short reason>}. "
                 "confidence reflects how sure you are.")
    return "\n".join(lines)


def _chat_json(client, model: str, system: str, user: str, schema: dict,
               temperature: float) -> dict:
    """Call the chat endpoint asking for JSON. Prefers strict json_schema; falls back to
    json_object for backends that don't support schema-guided decoding."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    # Attempt 1: strict schema (Ollama >=0.5 and vLLM support this).
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "categorization", "schema": schema, "strict": True},
            },
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:  # noqa: BLE001
        logger.debug("json_schema mode failed (%s); falling back to json_object", e)
    # Attempt 2: json_object mode.
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def classify_merchant(client, model: str, temperature: float, merchant: str,
                      categories: CategoryConfig, system: str, schema: dict) -> MerchantVerdict:
    valid = set(categories.all_names)
    try:
        data = _chat_json(client, model, system,
                          f"Merchant description: {merchant!r}\nWhich category?",
                          schema, temperature)
        category = str(data.get("category", "")).strip()
        if category not in valid:
            # match case-insensitively, else fall back
            match = next((n for n in valid if n.lower() == category.lower()), None)
            category = match or UNCATEGORIZED
        conf = float(data.get("confidence", 0.0))
        conf = min(max(conf, 0.0), 1.0)
        rationale = str(data.get("rationale", ""))[:300]
        by = "llm" if category != UNCATEGORIZED or conf > 0 else "fallback"
        return MerchantVerdict(category, conf, rationale, by)
    except Exception as e:  # noqa: BLE001 - never let one merchant break the batch
        logger.warning("Classification failed for %r: %s", merchant, e)
        return MerchantVerdict(UNCATEGORIZED, 0.0, f"classification error: {e}", "fallback")


def _encode_dates(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["month_label"] = dt.dt.strftime("%Y-%m")
    return df


def classify_transactions(
    df: pd.DataFrame,
    categories: Optional[CategoryConfig] = None,
    use_memory: bool = True,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    pinned: Optional[dict[str, str]] = None,
    examples: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Categorize a raw-transactions DataFrame. Returns a new DataFrame with category columns.

    ``pinned`` (merchant_key → category) are user-confirmed labels held fixed without asking the
    model. ``examples`` are merchant_key → category pairs injected into the prompt as few-shot
    guidance so the model generalizes the user's intent. Set ``use_memory=False`` to send every
    non-pinned merchant to the model instead of short-circuiting on the exact-match memory."""
    categories = categories or cfg.load_categories()
    memory = cfg.load_merchant_memory() if use_memory else {}
    pinned = pinned or {}
    valid = set(categories.all_names)

    df = df.copy()
    df["merchant_key"] = df["description"].map(normalize_merchant)
    unique_merchants = sorted(df["merchant_key"].unique())

    system = _system_prompt(categories, examples=examples)
    schema = _classification_schema(categories.all_names)
    rc = models.get_role_config("classifier")
    client = None  # lazily created only if an LLM call is actually needed

    verdicts: dict[str, MerchantVerdict] = {}
    total = len(unique_merchants)
    for i, merchant in enumerate(unique_merchants, 1):
        if merchant in pinned and pinned[merchant] in valid:
            verdicts[merchant] = MerchantVerdict(pinned[merchant], 1.0, "user-confirmed", "user")
        elif use_memory and merchant in memory and memory[merchant] in valid:
            verdicts[merchant] = MerchantVerdict(memory[merchant], 1.0,
                                                 "remembered from a previous correction", "memory")
        else:
            if client is None:
                client = models.get_client("classifier")
            verdicts[merchant] = classify_merchant(client, rc.model, rc.temperature,
                                                    merchant, categories, system, schema)
        if progress_cb:
            progress_cb(i, total, merchant)

    df["category"] = df["merchant_key"].map(lambda m: verdicts[m].category)
    df["confidence"] = df["merchant_key"].map(lambda m: round(verdicts[m].confidence, 3))
    df["rationale"] = df["merchant_key"].map(lambda m: verdicts[m].rationale)
    df["classified_by"] = df["merchant_key"].map(lambda m: verdicts[m].classified_by)
    df = _encode_dates(df)
    return df


def run_categorize(
    raw_csv: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    categories: Optional[CategoryConfig] = None,
    use_memory: bool = True,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Load raw CSV, classify, write a *draft* categorized CSV, and mark the run not-finalized."""
    raw_csv = Path(raw_csv) if raw_csv is not None else paths.RAW_CSV
    output_csv = output_csv if output_csv is not None else paths.CATEGORIZED_CSV
    if not raw_csv.exists():
        raise FileNotFoundError(f"{raw_csv} not found. Run `ocd extract` first.")
    df = pd.read_csv(raw_csv)
    if df.empty:
        raise ValueError(f"{raw_csv} has no transactions.")

    out = classify_transactions(df, categories=categories, use_memory=use_memory,
                                progress_cb=progress_cb)
    paths.ensure_dirs()
    out.to_csv(output_csv, index=False)
    cfg.mark_draft(n_transactions=len(out))
    logger.info("Wrote draft categorized CSV (%d rows) to %s", len(out), output_csv)
    return out
