"""Step 1 — extract.

Parse every credit-card statement PDF in ``data/statements/`` into a single
``data/transactions_raw.csv``. Bank type is auto-detected (monopoly's ``BankDetector``),
falling back to the generic parser for unrecognized layouts. All rows start in the
``Uncategorized`` bucket — categorization happens in Step 2.

This is purely deterministic parsing: no ML model is involved (optional OCR for scanned
PDFs is the only model-ish component, toggled via ``config/models.yaml`` -> ocr.enabled).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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


def parse_pdf(file: Path, use_ocr: Optional[bool] = None) -> dict:
    """Parse a single statement PDF. Returns a dict with rows + metadata; never raises."""
    # Imported lazily so importing ocd.extract doesn't require the heavy monopoly stack.
    from monopoly.banks import BankDetector, banks
    from monopoly.generic import GenericBank
    from monopoly.pdf import PdfDocument, PdfParser
    from monopoly.pipeline import Pipeline

    if use_ocr is None:
        use_ocr = models.get_role_config("ocr").enabled

    try:
        document = PdfDocument(file)
        document.unlock_document()
        if use_ocr:
            document = PdfParser.apply_ocr(document)

        bank = BankDetector(document).detect_bank(banks) or GenericBank
        parser = PdfParser(bank, document)
        pipeline = Pipeline(parser)

        statement = pipeline.extract()
        transactions = pipeline.transform(statement)

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
        return {"file": file.name, "bank": bank_name, "n": len(rows),
                "status": "ok", "error": None, "rows": rows}
    except Exception as err:  # noqa: BLE001 - one bad PDF must not kill the batch
        logger.warning("Failed to parse %s: %s", file.name, err)
        return {"file": file.name, "bank": None, "n": 0,
                "status": "error", "error": f"{type(err).__name__}: {err}", "rows": []}


def extract_statements(
    statements_dir: Path = paths.STATEMENTS_DIR,
    output_csv: Optional[Path] = paths.RAW_CSV,
    use_ocr: Optional[bool] = None,
) -> ExtractResult:
    """Parse all PDFs in ``statements_dir`` and write the combined raw CSV."""
    statements_dir = Path(statements_dir)
    pdfs = sorted(p for p in statements_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found in %s", statements_dir)

    all_rows: list[dict] = []
    per_file: list[dict] = []
    for pdf in pdfs:
        res = parse_pdf(pdf, use_ocr=use_ocr)
        all_rows.extend(res.pop("rows"))
        per_file.append(res)
        logger.info("%s -> %s (%d purchases, %s)", pdf.name, res["bank"], res["n"], res["status"])

    df = pd.DataFrame(all_rows, columns=RAW_COLUMNS)
    if not df.empty:
        df = df.sort_values(["date", "source_file"]).reset_index(drop=True)

    if output_csv is not None:
        paths.ensure_dirs()
        df.to_csv(output_csv, index=False)
        logger.info("Wrote %d transactions to %s", len(df), output_csv)

    return ExtractResult(transactions=df, per_file=per_file)
