"""Redact personal data from bank-statement PDFs, keeping the layout intact.

Run it:

    uv run python redact.py                 # statements/hamid -> statements/redacted
    uv run python redact.py IN_DIR OUT_DIR  # custom folders

Replacements are *true* redactions: the original text is removed from the PDF
content stream and synthetic, format-preserving text is drawn back in its
place. Transaction amounts, dates, and merchants are left untouched.

To redact different data, just edit the REPLACEMENTS list below
(each entry is "real string" -> "synthetic replacement").
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz  # PyMuPDF

SRC_DEFAULT = "statements/hamid"
DST_DEFAULT = "statements/redacted"

# Ordered longest-first so overlapping substrings replace correctly.
REPLACEMENTS: list[tuple[str, str]] = [
    # Address
    ("143 ALBANY ST APT 014C", "742 EVERGREEN TER APT 100C"),
    ("143 ALBANY ST APT 014", "742 EVERGREEN TER APT 100"),
    ("CAMBRIDGE, MA  02139-4262", "SPRINGFIELD, MA  01101-0000"),
    ("CAMBRIDGE, MA 02139-4262", "SPRINGFIELD, MA 01101-0000"),
    ("02139-4262", "01101-0000"),
    # Name — token-level so any ordering (header / "INDN:" / reversed) is caught
    ("HAMIDREZA", "JORDAN"),
    ("KAMKARI", "RIVERA"),
    ("Hamidreza", "Jordan"),
    ("Kamkari", "Rivera"),
    # Phone (the account holder's contact number, not merchant numbers)
    ("(617) 206-7442", "(617) 555-0142"),
    # Account / card identifiers
    ("Acct Ending 7036", "Acct Ending 4417"),
    ("ENDING IN 5147", "ENDING IN 8890"),
    ("ID:7036", "ID:4417"),
    ("4660 2470 5491", "5500 1234 9012"),
]


def scrub(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def _rgb(color: int) -> tuple[float, float, float]:
    return ((color >> 16 & 255) / 255, (color >> 8 & 255) / 255, (color & 255) / 255)


def redact_pdf(src: Path, dst: Path) -> int:
    """Write a redacted copy of ``src`` to ``dst``; return spans changed."""
    doc = fitz.open(src)
    changed = 0
    for page in doc:
        edits = []  # (bbox, origin, new_text, fontsize, rgb)
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:  # 0 == text
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    new = scrub(span["text"])
                    if new != span["text"]:
                        edits.append(
                            (fitz.Rect(span["bbox"]), span["origin"], new,
                             span["size"], _rgb(span["color"]))
                        )
        if not edits:
            continue
        for bbox, *_ in edits:
            page.add_redact_annot(bbox, fill=(1, 1, 1))
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        for bbox, origin, new, size, rgb in edits:
            page.insert_text(origin, new, fontsize=size, fontname="helv", color=rgb)
            changed += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    doc.save(dst, garbage=4, deflate=True)
    doc.close()
    return changed


def verify(dst_dir: Path) -> bool:
    """Re-scan outputs for any leftover original PII. Returns True if clean."""
    needles = [old for old, _ in REPLACEMENTS]
    clean = True
    for pdf in sorted(dst_dir.glob("*.pdf")):
        text = "\n".join(p.get_text() for p in fitz.open(pdf))
        leaks = [n for n in needles if n in text]
        if leaks:
            clean = False
            print(f"  ⚠ {pdf.name}: still contains {leaks}")
    return clean


def main(argv: list[str]) -> None:
    src_dir = Path(argv[0]) if len(argv) > 0 else Path(SRC_DEFAULT)
    dst_dir = Path(argv[1]) if len(argv) > 1 else Path(DST_DEFAULT)
    pdfs = sorted(src_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"no PDFs found in {src_dir}")
    for pdf in pdfs:
        n = redact_pdf(pdf, dst_dir / pdf.name)
        print(f"  {pdf.name}: {n} spans redacted")
    print(f"Done. {len(pdfs)} files -> {dst_dir}")
    print("Verifying…", "clean ✓" if verify(dst_dir) else "LEAKS FOUND ✗")


if __name__ == "__main__":
    main(sys.argv[1:])
