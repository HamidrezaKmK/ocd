"""Step 1 — extract.

Parse every credit-card statement PDF in ``data/statements/`` into a single
``data/transactions_raw.csv``. All rows start in the ``Uncategorized`` bucket —
categorization happens in Step 2.

Two-tier extraction, per file:
  1. **Deterministic (preferred):** monopoly's ``BankDetector`` auto-detects the bank,
     falling back to monopoly's generic parser for unrecognized layouts.
  2. **LLM fallback:** if the deterministic tier raises *or* finds no transactions at all,
     the statement text is read by the local ``extractor`` model (``config/models.yaml``)
     which returns the transactions as structured JSON. Toggle with ``allow_llm_fallback``.

Optional OCR for scanned PDFs is toggled via ``config/models.yaml`` -> ocr.enabled.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from . import models, paths
from .config import UNCATEGORIZED

logger = logging.getLogger(__name__)

RAW_COLUMNS = [
    "date", "description", "amount", "polarity", "balance",
    "bank", "statement_date", "statement_month", "source_file", "category",
]

# Descriptions that are not real purchases (payments, balances, interest credits, refunds-in).
_NON_PURCHASE_RE = re.compile(
    r"\b(payment|autopay|auto[- ]?pay|thank you|last month'?s balance|previous balance|"
    r"balance brought forward|direct debit received|credit balance|statement credit|"
    r"cash ?rebate|cashback|cash ?back|refund|reversal|rebate|adjustment)\b",
    re.IGNORECASE,
)

# Polarity markers that mean "money in" (credit) rather than a purchase/charge.
_CREDIT_POLARITY = {"CR", "C", "+", "CREDIT"}


@dataclass
class ExtractResult:
    transactions: pd.DataFrame
    per_file: list[dict] = field(default_factory=list)  # {file, bank, n, status, error}

    @property
    def n_files_ok(self) -> int:
        return sum(1 for r in self.per_file if r["status"] == "ok")

    @property
    def banks(self) -> list[str]:
        return sorted({r["bank"] for r in self.per_file if r.get("bank")})


def _is_purchase(description: str, amount: float, polarity: Optional[str]) -> bool:
    """A purchase is a charge (money out), not a payment/credit/rebate.

    Sign conventions differ across banks (monopoly may report charges as negative and
    credits as positive, or vice versa), so we do NOT rely on sign. Instead we identify
    credits via the polarity marker (``CR``) and obvious payment/balance/refund descriptions,
    and treat everything else as a purchase. The spend magnitude is ``abs(amount)``."""
    if amount is None or amount == 0:
        return False
    if polarity and polarity.strip().upper() in _CREDIT_POLARITY:
        return False
    if _NON_PURCHASE_RE.search(description or ""):
        return False
    return True


def _monopoly_extract(file: Path, use_ocr: bool) -> tuple[str, list[dict], int]:
    """Deterministic parse via monopoly. Returns ``(bank_name, purchase_rows, n_raw)``
    where ``n_raw`` is the number of transactions found *before* purchase filtering.
    Raises on any parsing failure."""
    # Imported lazily so importing ocd.extract doesn't require the heavy monopoly stack.
    from monopoly.banks import BankDetector, banks
    from monopoly.generic import GenericBank
    from monopoly.pdf import PdfDocument, PdfParser
    from monopoly.pipeline import Pipeline

    document = PdfDocument(file)
    document.unlock_document()
    if use_ocr:
        document = PdfParser.apply_ocr(document)

    bank = BankDetector(document).detect_bank(banks) or GenericBank
    parser = PdfParser(bank, document)
    pipeline = Pipeline(parser)

    statement = pipeline.extract()
    transactions = list(pipeline.transform(statement))

    bank_name = getattr(statement, "bank_name", None) or bank.__name__
    stmt_date = getattr(statement, "statement_date", None)
    stmt_date_iso = stmt_date.date().isoformat() if hasattr(stmt_date, "date") else (
        str(stmt_date) if stmt_date else "")
    stmt_month = stmt_date.strftime("%Y-%m") if hasattr(stmt_date, "strftime") else ""

    rows = []
    for t in transactions:
        polarity = getattr(t, "polarity", None)
        if not _is_purchase(t.description, t.amount, polarity):
            continue
        rows.append({
            "date": t.date,
            "description": (t.description or "").strip(),
            "amount": abs(float(t.amount)),  # store spend magnitude (positive)
            "polarity": polarity or "",
            "balance": getattr(t, "balance", None),
            "bank": bank_name,
            "statement_date": stmt_date_iso,
            "statement_month": stmt_month,
            "source_file": file.name,
            "category": UNCATEGORIZED,
        })
    return bank_name, rows, len(transactions)


def _extract_pdf_text(file: Path, max_chars: int = 60_000) -> str:
    """Best-effort plain-text extraction for the LLM fallback (poppler via pdftotext —
    the same dependency monopoly already relies on)."""
    import pdftotext

    with open(file, "rb") as fh:
        pages = pdftotext.PDF(fh)
    text = "\n\n".join(pages)
    if len(text) > max_chars:
        logger.warning("Truncating %s from %d to %d chars for LLM extraction",
                       file.name, len(text), max_chars)
        text = text[:max_chars]
    return text


_LLM_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "bank": {"type": "string"},
        "statement_date": {"type": "string"},
        "transactions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "description": {"type": "string"},
                    "amount": {"type": "number"},
                    "direction": {"type": "string", "enum": ["debit", "credit"]},
                },
                "required": ["date", "description", "amount", "direction"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["transactions"],
    "additionalProperties": False,
}

_LLM_EXTRACT_SYSTEM = (
    "You are a precise bank- and credit-card-statement parser. You are given the raw text of one "
    "statement. Extract EVERY transaction line into structured JSON. Rules:\n"
    "- amount: a positive number, no currency symbols or thousands separators.\n"
    "- direction: 'debit' for purchases/charges/withdrawals (money out); 'credit' for payments, "
    "refunds, deposits, interest, or anything that adds money.\n"
    "- date: ISO 'YYYY-MM-DD'; infer the year from the statement period if a row shows only month/day.\n"
    "- description: the merchant/description text, trimmed.\n"
    "- Also return the bank/issuer name and the statement closing date (ISO) when present.\n"
    "- Do NOT invent transactions; include only rows actually present. Return strict JSON."
)


def _llm_extract(file: Path, text: str) -> dict:
    """Read a statement's text with the ``extractor`` model and return ``{bank, rows}``.
    Raises on hard failures so the caller can record an error for this file."""
    from .classify import _chat_json  # shared OpenAI-compatible JSON-chat helper

    rc = models.get_role_config("extractor")
    client = models.get_client("extractor")
    data = _chat_json(
        client, rc.model, _LLM_EXTRACT_SYSTEM,
        f"Statement text:\n\n{text}\n\nReturn the JSON now.",
        _LLM_EXTRACT_SCHEMA, rc.temperature,
    )

    bank_name = (str(data.get("bank") or "").strip() or "Unknown") + " (LLM)"
    stmt_dt = pd.to_datetime(str(data.get("statement_date") or "").strip(), errors="coerce")
    stmt_date_iso = stmt_dt.date().isoformat() if pd.notna(stmt_dt) else ""
    stmt_month = stmt_dt.strftime("%Y-%m") if pd.notna(stmt_dt) else ""

    rows = []
    for t in data.get("transactions", []) or []:
        try:
            amount = abs(float(t.get("amount")))
        except (TypeError, ValueError):
            continue
        desc = str(t.get("description") or "").strip()
        polarity = "CR" if str(t.get("direction") or "").strip().lower() == "credit" else ""
        if not _is_purchase(desc, amount, polarity):
            continue
        date_dt = pd.to_datetime(str(t.get("date") or ""), errors="coerce")
        rows.append({
            "date": date_dt.date().isoformat() if pd.notna(date_dt) else str(t.get("date") or ""),
            "description": desc,
            "amount": amount,
            "polarity": polarity,
            "balance": None,
            "bank": bank_name,
            "statement_date": stmt_date_iso,
            "statement_month": stmt_month,
            "source_file": file.name,
            "category": UNCATEGORIZED,
        })
    return {"bank": bank_name, "rows": rows}


def parse_pdf(file: Path, use_ocr: Optional[bool] = None,
              allow_llm_fallback: bool = True) -> dict:
    """Parse a single statement PDF; never raises. Returns a dict with rows + metadata,
    including ``method`` (``'monopoly'`` | ``'llm'`` | ``None``).

    Deterministic parsing is tried first; the LLM fallback runs only if it raised or found
    no transactions at all (and the ``extractor`` role is enabled)."""
    if use_ocr is None:
        use_ocr = models.get_role_config("ocr").enabled

    bank_name: Optional[str] = None
    rows: list[dict] = []
    n_raw = 0
    method: Optional[str] = None
    error: Optional[str] = None

    try:
        bank_name, rows, n_raw = _monopoly_extract(file, use_ocr)
        method = "monopoly"
    except Exception as err:  # noqa: BLE001 - one bad PDF must not kill the batch
        logger.warning("Deterministic parse failed for %s: %s", file.name, err)
        error = f"{type(err).__name__}: {err}"

    if (error is not None or n_raw == 0) and allow_llm_fallback and models.is_enabled("extractor"):
        try:
            text = _extract_pdf_text(file)
            if text.strip():
                res = _llm_extract(file, text)
                bank_name, rows, method, error = res["bank"], res["rows"], "llm", None
        except Exception as err:  # noqa: BLE001 - fallback failure is just a parse error
            logger.warning("LLM fallback failed for %s: %s", file.name, err)
            error = (f"{error} | " if error else "") + f"llm fallback: {type(err).__name__}: {err}"

    status = "ok" if (method is not None and error is None) else "error"
    return {"file": file.name, "bank": bank_name, "n": len(rows),
            "method": method, "status": status, "error": error, "rows": rows}


def extract_statements(
    statements_dir: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    use_ocr: Optional[bool] = None,
    allow_llm_fallback: bool = True,
    file_cb: Optional[Callable[[int, int, dict], None]] = None,
) -> ExtractResult:
    """Parse all PDFs in ``statements_dir`` and write the combined raw CSV.

    Paths default to the active workspace (see ``paths``). ``file_cb(i, total, result)`` is
    invoked after each file (1-based ``i``) with the per-file result dict — used to stream
    extraction progress to the web UI."""
    statements_dir = Path(statements_dir) if statements_dir is not None else paths.STATEMENTS_DIR
    output_csv = output_csv if output_csv is not None else paths.RAW_CSV
    pdfs = sorted(p for p in statements_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found in %s", statements_dir)

    all_rows: list[dict] = []
    per_file: list[dict] = []
    for i, pdf in enumerate(pdfs, 1):
        res = parse_pdf(pdf, use_ocr=use_ocr, allow_llm_fallback=allow_llm_fallback)
        all_rows.extend(res.pop("rows"))
        per_file.append(res)
        logger.info("%s -> %s via %s (%d purchases, %s)",
                    pdf.name, res["bank"], res["method"], res["n"], res["status"])
        if file_cb:
            file_cb(i, len(pdfs), res)

    df = pd.DataFrame(all_rows, columns=RAW_COLUMNS)
    if not df.empty:
        df = df.sort_values(["date", "source_file"]).reset_index(drop=True)

    if output_csv is not None:
        paths.ensure_dirs()
        df.to_csv(output_csv, index=False)
        logger.info("Wrote %d transactions to %s", len(df), output_csv)

    return ExtractResult(transactions=df, per_file=per_file)
