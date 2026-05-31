"""Download real, bank-hosted sample credit-card statement PDFs for testing.

These are public sample/marketing PDFs published by the banks themselves. They let you exercise
the multi-bank parsing path (monopoly auto-detects the bank). Note they are single statements with
fixed dates, so for multi-month trend testing use ``make_synthetic`` as well.

Plaid (https://plaid.com) is a plausible future source of real, user-authorized statement data;
it is intentionally NOT integrated here — this project stays fully local.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import paths

logger = logging.getLogger(__name__)

# name -> (url, monopoly-supported?) Real bank-hosted sample statements.
SAMPLES: dict[str, str] = {
    "chase_sample.pdf":
        "https://www.chase.com/content/dam/chase-ux/documents/digital/resources/"
        "paperless_statements_chase_sample.pdf",
    "bank_of_america_sample.pdf":
        "https://secure.bankofamerica.com/content/pdf/en_us/IHL-Statements.pdf",
    "capital_one_sample.pdf":
        "https://ecm.capitalone.com/WCM/bank/pdfs/sample-estatement.pdf",
    "cfpb_sample.pdf":
        "https://files.consumerfinance.gov/f/documents/"
        "cfpb_building_block_activities_sample-credit-card-statement_handout.pdf",
}


def download_samples(dest: Path = paths.STATEMENTS_DIR, overwrite: bool = False) -> list[Path]:
    import requests

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for name, url in SAMPLES.items():
        out = dest / name
        if out.exists() and not overwrite:
            logger.info("skip (exists): %s", name)
            saved.append(out)
            continue
        try:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 (ocd-sample-fetch)"})
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "pdf" not in ctype.lower() and not resp.content[:4] == b"%PDF":
                logger.warning("not a PDF (%s): %s", ctype, name)
                continue
            out.write_bytes(resp.content)
            saved.append(out)
            logger.info("downloaded %s (%d KB)", name, len(resp.content) // 1024)
        except Exception as e:  # noqa: BLE001
            logger.warning("failed to download %s: %s", name, e)
    return saved


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    saved = download_samples()
    print(f"\nSaved {len(saved)} sample(s) to {paths.STATEMENTS_DIR}:")
    for p in saved:
        print(f"  - {p.name}")
    print("\nNote: real samples are single-month. For multi-month trend testing run:")
    print("  python -m ocd.samples.make_synthetic")


if __name__ == "__main__":
    main()
