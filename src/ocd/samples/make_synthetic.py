"""Generate synthetic multi-month credit-card statement PDFs for two banks.

Real downloadable samples are single statements with fixed dates, which is not enough to show a
per-month spending *trend*. This generator produces, for two distinct issuers over the SAME set of
months, statement PDFs whose text layout mirrors the format monopoly's generic parser handles
(``DD/MM  DESCRIPTION  CITY  CC  AMOUNT`` rows under a ``TRANSACTION DATE / DESCRIPTION /
AMOUNT (SGD)`` header, with a ``LAST MONTH'S BALANCE`` line). The result feeds the full pipeline
(extract -> categorize -> report) so trends and insights are meaningful in a demo.

Deterministic given a seed so runs are reproducible.
"""

from __future__ import annotations

import calendar
import logging
import random
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from .. import paths

logger = logging.getLogger(__name__)

# Merchants grouped so categorization has signal and category mix varies month to month.
MERCHANTS = {
    "Groceries": ["ARCTIC MARKET", "FRESH FIELDS GROCER", "GREEN BASKET MART", "DAILY PANTRY",
                  "HARVEST FOODS", "CORNER GROCERY"],
    "Dining": ["MORNING BITES CAFE", "URBAN EATS", "SUNNY CAFE", "GLOBAL GRUB", "FAST FEAST",
               "GOLDEN TOAST", "DRIVE-THRU DELIGHTS", "NOODLE HOUSE"],
    "Transport": ["URBAN TRANSIT CO.", "SPEEDY FUEL", "CITY METRO CARD", "RIDEFLOW",
                  "PARK & GO", "SHELL STATION"],
    "Shopping": ["SPEEDY DRIVE SHOP", "TRENDWEAR", "GADGET WORLD", "HOME STYLE",
                 "BOOK NOOK", "MEGA MART ONLINE"],
    "Entertainment": ["STREAMFLIX", "CINEMA PLUS", "GAME REALM", "TUNE STREAM", "EVENT TIX"],
    "Health": ["WELLNESS PHARMACY", "FITZONE GYM", "CARE CLINIC", "VITAL DRUGS"],
    "Utilities & Bills": ["TELCO MOBILE", "BRIGHT ENERGY", "AQUA UTILITIES", "NET CONNECT"],
    "Travel": ["SKYWAY AIR", "COZY STAY HOTEL", "RENT-A-RIDE"],
}

# (filename, issuer header text, city, country-code) for two distinct "banks".
BANKS = [
    ("synthetic_cardco_{ym}.pdf", "CARDCO REWARDS CREDIT CARD", "SINGAPORE", "SG"),
    ("synthetic_metrobank_{ym}.pdf", "METROBANK PLATINUM CREDIT CARD", "SINGAPORE", "SG"),
]


def _money(x: float) -> str:
    return f"{x:,.2f}"


def _build_transactions(rng: random.Random, year: int, month: int, intensity: float):
    """Return a list of (day, description, city, cc, amount) for one statement month.

    ``intensity`` scales spending so consecutive months differ and a trend is visible.
    """
    days_in_month = calendar.monthrange(year, month)[1]
    txns = []
    # rough per-category transaction counts and typical amount ranges
    plan = {
        "Groceries": (rng.randint(3, 6), (15, 90)),
        "Dining": (rng.randint(4, 9), (5, 45)),
        "Transport": (rng.randint(2, 5), (2, 40)),
        "Shopping": (rng.randint(1, 4), (20, 160)),
        "Entertainment": (rng.randint(1, 3), (9, 30)),
        "Health": (rng.randint(0, 2), (12, 70)),
        "Utilities & Bills": (rng.randint(1, 2), (30, 120)),
        "Travel": (rng.randint(0, 2), (60, 320)),
    }
    for cat, (count, (lo, hi)) in plan.items():
        for _ in range(count):
            merchant = rng.choice(MERCHANTS[cat])
            day = rng.randint(1, days_in_month)
            amt = round(rng.uniform(lo, hi) * intensity, 2)
            txns.append((day, merchant, amt))
    txns.sort(key=lambda t: t[0])
    return txns


def _render_pdf(out: Path, issuer: str, city: str, cc: str, year: int, month: int,
                prev_balance: float, txns) -> None:
    c = canvas.Canvas(str(out), pagesize=A4)
    width, height = A4
    x0 = 40
    y = height - 50
    mono = "Courier"

    def line(text: str, size: int = 9, dy: int = 13, font: str = mono):
        nonlocal y
        c.setFont(font, size)
        c.drawString(x0, y, text)
        y -= dy

    stmt_date = f"{calendar.monthrange(year, month)[1]:02d}-{month:02d}-{year}"
    due = f"15-{(month % 12) + 1:02d}-{year if month < 12 else year + 1}"

    line("CARD STATEMENT", 13, 20, "Courier-Bold")
    line("ACCOUNT HOLDER: J. DOE    123 EXAMPLE ROAD    SINGAPORE 100000", 8)
    y -= 6
    line("STATEMENT DATE        PAYMENT DUE DATE        TOTAL CREDIT LIMIT", 9, 13, "Courier-Bold")
    line(f"   {stmt_date}           {due}                 S$20,000.00", 9)
    y -= 8
    # transaction header (must contain these keywords for the generic parser)
    line("TRANSACTION DATE          DESCRIPTION                              AMOUNT (SGD)",
         9, 16, "Courier-Bold")
    line(issuer, 9, 13, "Courier-Bold")
    line("J. DOE                    5488-0000-0000-0000", 8)
    y -= 4

    # previous balance + a payment (these are filtered out as non-purchases downstream)
    prev_label = "LAST MONTH'S BALANCE"
    pay_date = f"01/{month:02d}"
    line(f"{'':10}{prev_label:<55}{_money(prev_balance):>12}")
    line(f"{pay_date:<10}{'PAYMENT BY INTERNET':<55}({_money(prev_balance)})")

    total = 0.0
    for day, merchant, amt in txns:
        total += amt
        date = f"{day:02d}/{month:02d}"
        desc = f"{merchant:<{38}} {city:<11} {cc}"
        line(f"{date:<10}{desc:<55}{_money(amt):>12}")
        if y < 70:  # new page if needed
            c.showPage()
            y = height - 50

    y -= 10
    line(f"{'':10}{'SUB-TOTAL (PURCHASES)':<55}{_money(total):>12}", 9, 13, "Courier-Bold")
    line(f"{'':10}{'NEW BALANCE':<55}{_money(total):>12}", 9, 13, "Courier-Bold")
    c.showPage()
    c.save()


def generate(
    dest: Path = paths.STATEMENTS_DIR,
    months: int = 4,
    start_year: int = 2024,
    start_month: int = 1,
    seed: int = 7,
) -> list[Path]:
    """Generate ``months`` statements for each of the two banks over the same period."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written: list[Path] = []

    for bank_i, (fname_tpl, issuer, city, cc) in enumerate(BANKS):
        prev_balance = round(rng.uniform(200, 600), 2)
        for k in range(months):
            month = (start_month - 1 + k) % 12 + 1
            year = start_year + (start_month - 1 + k) // 12
            # gentle trend: spending drifts up then down, different phase per bank
            intensity = 1.0 + 0.18 * (k - months / 2) / max(months, 1) + 0.1 * bank_i
            txns = _build_transactions(rng, year, month, intensity)
            ym = f"{year}-{month:02d}"
            out = dest / fname_tpl.format(ym=ym)
            _render_pdf(out, issuer, city, cc, year, month, prev_balance, txns)
            prev_balance = round(sum(t[2] for t in txns), 2)
            written.append(out)
            logger.info("wrote %s (%d purchases)", out.name, len(txns))
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    written = generate()
    print(f"\nGenerated {len(written)} synthetic statements in {paths.STATEMENTS_DIR}:")
    for p in written:
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
