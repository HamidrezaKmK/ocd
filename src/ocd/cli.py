"""Command-line interface for OCD.

    ocd doctor                 # check Ollama / model / poppler are ready
    ocd setup                  # define categories + write default model config
    ocd extract                # Step 1: PDFs -> data/transactions_raw.csv
    ocd categorize             # Step 2 (auto pass): draft categorization + attention list
    ocd review                 # Step 2 (interactive): inspect flags / finalize from terminal
    ocd report                 # Step 3: build Markdown + HTML report (needs finalized run)
    ocd app                    # launch the Streamlit UI (setup -> review -> report)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="OCD — Optimized Cash Dashboard (local spending analyzer).")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def doctor():
    """Check that the local environment (Ollama, model, poppler) is ready."""
    from . import models
    ok = True

    # poppler / monopoly
    try:
        import pdftotext  # noqa: F401
        import monopoly  # noqa: F401
        typer.secho("✓ monopoly + poppler import OK", fg="green")
    except Exception as e:  # noqa: BLE001
        ok = False
        typer.secho(f"✗ monopoly/poppler: {e}", fg="red")
        typer.echo("  Hint: run inside `conda activate ocd` (sets the libstdc++ LD_PRELOAD).")

    for role in ("classifier", "insights"):
        if not models.is_enabled(role):
            typer.secho(f"• {role}: disabled", fg="yellow")
            continue
        good, msg = models.health_check(role)
        typer.secho(("✓ " if good else "✗ ") + msg, fg="green" if good else "red")
        ok = ok and (good or role == "insights")
    if not ok:
        typer.echo("  Hint: start the model server with `ollama serve` and "
                   "`ollama pull qwen2.5:7b-instruct`.")
    raise typer.Exit(0 if ok else 1)


@app.command()
def setup(reset: bool = typer.Option(False, help="Overwrite existing categories.")):
    """Define spending categories (name, description, monthly limit) interactively."""
    from . import config as cfg
    from . import models
    from .config import Category, CategoryConfig

    models.write_default_models_config()
    if cfg.categories_exist() and not reset:
        current = cfg.load_categories()
        typer.echo("Existing categories:")
        for c in current.categories:
            typer.echo(f"  - {c.name}: {c.description} (limit ${c.monthly_limit:.0f})")
        if not typer.confirm("Redefine them?", default=False):
            raise typer.Exit(0)

    typer.echo("Define each category. Press Enter with an empty name to finish.\n")
    cats: list[Category] = []
    while True:
        name = typer.prompt(f"Category #{len(cats)+1} name", default="", show_default=False)
        if not name.strip():
            break
        desc = typer.prompt("  description", default="")
        limit = typer.prompt("  monthly limit ($)", default=0.0, type=float)
        cats.append(Category(name=name.strip(), description=desc.strip(), monthly_limit=limit))
    if not cats:
        typer.secho("No categories entered; keeping defaults.", fg="yellow")
        cats = list(cfg.DEFAULT_CATEGORIES)
    cfg.save_categories(CategoryConfig(categories=cats))
    typer.secho(f"Saved {len(cats)} categories to {cfg.paths.CATEGORIES_YAML}", fg="green")


@app.command()
def extract(
    statements_dir: Optional[Path] = typer.Option(None, help="Folder of statement PDFs."),
    ocr: bool = typer.Option(False, help="Apply OCR (for scanned/image PDFs)."),
):
    """Step 1 — parse statement PDFs into data/transactions_raw.csv."""
    from . import paths
    from .extract import extract_statements
    sd = statements_dir or paths.STATEMENTS_DIR
    typer.echo(f"Extracting PDFs from {sd} ...")
    res = extract_statements(statements_dir=sd, use_ocr=ocr or None)
    for r in res.per_file:
        mark = "✓" if r["status"] == "ok" else "✗"
        color = "green" if r["status"] == "ok" else "red"
        typer.secho(f"  {mark} {r['file']} -> {r['bank']} ({r['n']} purchases)"
                    + (f" — {r['error']}" if r["error"] else ""), fg=color)
    typer.secho(f"{len(res.transactions)} purchases from {res.n_files_ok} statement(s); "
                f"banks: {', '.join(res.banks) or 'none'}", fg="cyan")
    if res.transactions.empty:
        raise typer.Exit(1)


@app.command()
def categorize(
    finalize: bool = typer.Option(False, help="Finalize immediately (non-interactive)."),
    no_memory: bool = typer.Option(False, help="Ignore learned merchant memory."),
):
    """Step 2 (auto pass) — categorize transactions with the local LLM."""
    from . import config as cfg
    from .classify import run_categorize
    from .review import compute_flags, finalize as do_finalize

    typer.echo("Categorizing (this calls the local model for each new merchant) ...")
    with typer.progressbar(length=100, label="classifying") as bar:
        state = {"last": 0}
        def cb(i, total, merchant):
            pct = int(i / max(total, 1) * 100)
            bar.update(pct - state["last"]); state["last"] = pct
        df = run_categorize(use_memory=not no_memory, progress_cb=cb)

    rs = compute_flags(df)
    typer.secho(f"\n{len(df)} transactions categorized; {rs.n_attention} need attention.", fg="cyan")
    for it in rs.items[:15]:
        typer.echo(f"  • {it.date}  ${it.amount:>8.2f}  {it.description[:34]:34s} "
                   f"-> {it.category:14s} [{'; '.join(it.reasons)}]")
    if rs.n_attention > 15:
        typer.echo(f"  ... and {rs.n_attention - 15} more (use `ocd app` to review all).")

    if finalize:
        meta = do_finalize(df)
        typer.secho(f"Finalized {meta.n_transactions} transactions (period {meta.period}).", fg="green")
    else:
        typer.echo("\nReview & finalize with `ocd app`, or `ocd categorize --finalize` to accept as-is.")


@app.command()
def review(
    finalize: bool = typer.Option(False, help="Finalize the current categorization."),
):
    """Step 2 (interactive) — list rows needing attention, or finalize from the terminal."""
    from .review import compute_flags, finalize as do_finalize, load_categorized
    df = load_categorized()
    rs = compute_flags(df)
    typer.secho(f"{rs.n_attention} of {len(df)} transactions need attention.", fg="cyan")
    if rs.over_limit:
        typer.secho("Over budget:", fg="yellow")
        for o in rs.over_limit:
            typer.echo(f"  {o['category']} {o['month_label']}: ${o['spent']:.0f} / ${o['limit']:.0f}")
    for it in rs.items:
        typer.echo(f"  • {it.date}  ${it.amount:>8.2f}  {it.description[:34]:34s} "
                   f"-> {it.category:14s} [{'; '.join(it.reasons)}]")
    if finalize:
        meta = do_finalize(df)
        typer.secho(f"Finalized {meta.n_transactions} transactions.", fg="green")
    else:
        typer.echo("\nEdit categories in `ocd app`, or pass --finalize to accept as-is.")


@app.command()
def report(draft: bool = typer.Option(False, help="Allow report on a non-finalized draft.")):
    """Step 3 — generate the Markdown + HTML spending report."""
    from .report import NotFinalizedError, generate_report
    try:
        out = generate_report(require_finalized=not draft)
    except NotFinalizedError as e:
        typer.secho(f"✗ {e}", fg="red")
        raise typer.Exit(1)
    typer.secho(f"✓ HTML:     {out['html']}", fg="green")
    typer.secho(f"✓ Markdown: {out['markdown']}", fg="green")
    typer.echo("\nInsights:")
    for i in out["insights"]:
        typer.echo(f"  - {i}")


@app.command(name="app")
def app_ui(
    port: int = typer.Option(8501, help="Port for the Streamlit server."),
):
    """Launch the Streamlit web UI."""
    import subprocess
    app_path = Path(__file__).with_name("app.py")
    typer.echo(f"Launching Streamlit at http://localhost:{port} ...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path),
                    "--server.port", str(port)], check=False)


if __name__ == "__main__":
    app()
