"""Streamlit UI for OCD — the local web interface for the whole pipeline.

Pages:
  Setup              define categories (name / description / monthly limit) + model status
  1 · Extract        parse statement PDFs in data/statements/ -> raw transactions
  2 · Categorize     auto-classify, then review/edit and finalize (the gate for step 3)
  3 · Report         interactive charts + insights once categorization is finalized

Launch with ``ocd app`` (which runs ``streamlit run`` on this file).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ocd import config as cfg
from ocd import models, paths
from ocd.config import Category, CategoryConfig

st.set_page_config(page_title="OCD — Optimized Cash Dashboard", page_icon="💸", layout="wide")
paths.ensure_dirs()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def model_status_badge() -> None:
    good, msg = models.health_check("classifier")
    (st.success if good else st.error)(msg)
    if not good:
        st.caption("Start the model server: `ollama serve` and `ollama pull qwen2.5:7b-instruct`. "
                   "Edit `config/models.yaml` to point at a different model or a vLLM endpoint.")


def categorized_exists() -> bool:
    return paths.CATEGORIZED_CSV.exists()


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_setup() -> None:
    st.header("⚙️ Setup — your categories")
    st.write("Define the spending categories that fit *you*. The description is given to the model, "
             "so it categorizes the way you think. Set a monthly limit per category (0 = no limit).")

    cur = cfg.load_categories()
    df = pd.DataFrame([{"name": c.name, "description": c.description,
                        "monthly_limit": c.monthly_limit} for c in cur.categories])
    edited = st.data_editor(
        df, num_rows="dynamic", use_container_width=True, key="cat_editor",
        column_config={
            "name": st.column_config.TextColumn("Category", required=True),
            "description": st.column_config.TextColumn("Description", width="large"),
            "monthly_limit": st.column_config.NumberColumn("Monthly limit ($)", min_value=0, step=25),
        },
    )
    col1, col2 = st.columns([1, 4])
    if col1.button("💾 Save categories", type="primary"):
        cats = []
        for _, r in edited.iterrows():
            name = str(r["name"]).strip()
            if not name:
                continue
            cats.append(Category(name=name, description=str(r.get("description", "") or ""),
                                 monthly_limit=float(r.get("monthly_limit", 0) or 0)))
        if cats:
            cfg.save_categories(CategoryConfig(categories=cats))
            models.write_default_models_config()
            col2.success(f"Saved {len(cats)} categories to {paths.CATEGORIES_YAML}")
        else:
            col2.warning("Add at least one category.")

    st.divider()
    st.subheader("Model")
    model_status_badge()


def page_extract() -> None:
    st.header("1 · Extract — statements ➜ transactions")
    st.write(f"Drop credit-card statement PDFs into **`{paths.STATEMENTS_DIR}`**, then extract. "
             "Bank type is auto-detected; unknown layouts use a generic parser.")

    pdfs = sorted(paths.STATEMENTS_DIR.glob("*.pdf"))
    up = st.file_uploader("…or upload PDFs here", type="pdf", accept_multiple_files=True)
    if up:
        for f in up:
            (paths.STATEMENTS_DIR / f.name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(up)} file(s) to {paths.STATEMENTS_DIR}")
        pdfs = sorted(paths.STATEMENTS_DIR.glob("*.pdf"))

    st.caption(f"{len(pdfs)} PDF(s) in folder: " + (", ".join(p.name for p in pdfs) or "none"))
    use_ocr = st.checkbox("Apply OCR (scanned/image PDFs)", value=False)

    if st.button("📄 Extract transactions", type="primary", disabled=not pdfs):
        from ocd.extract import extract_statements
        with st.spinner("Parsing PDFs…"):
            res = extract_statements(use_ocr=use_ocr or None)
        st.session_state["extract_files"] = res.per_file
        st.success(f"Extracted {len(res.transactions)} purchases from {res.n_files_ok} statement(s). "
                   f"Banks: {', '.join(res.banks) or 'none'}")

    if "extract_files" in st.session_state:
        st.dataframe(pd.DataFrame(st.session_state["extract_files"])[["file", "bank", "n", "status", "error"]],
                     use_container_width=True, hide_index=True)
    if paths.RAW_CSV.exists():
        raw = pd.read_csv(paths.RAW_CSV)
        st.caption(f"Current raw file: {len(raw)} purchases across "
                   f"{raw['statement_month'].nunique()} month(s).")
        with st.expander("Preview raw transactions"):
            st.dataframe(raw, use_container_width=True, hide_index=True)


def page_categorize() -> None:
    st.header("2 · Categorize & review")
    if not paths.RAW_CSV.exists():
        st.info("Run **Extract** first.")
        return

    c1, c2, c3 = st.columns(3)
    use_memory = c1.toggle("Use learned merchant memory", value=True)
    if c2.button("🤖 Run auto-categorization", type="primary"):
        from ocd.classify import run_categorize
        prog = st.progress(0.0, text="Classifying merchants…")
        def cb(i, total, merchant):
            prog.progress(i / max(total, 1), text=f"Classifying {merchant[:30]}… ({i}/{total})")
        with st.spinner("Calling the local model…"):
            df = run_categorize(use_memory=use_memory, progress_cb=cb)
        prog.empty()
        st.session_state["cat_df"] = df
        st.success(f"Categorized {len(df)} transactions.")

    # Load working df
    if "cat_df" not in st.session_state and categorized_exists():
        st.session_state["cat_df"] = pd.read_csv(paths.CATEGORIZED_CSV)
    if "cat_df" not in st.session_state:
        st.info("Run auto-categorization to begin.")
        return

    from ocd.review import apply_corrections, compute_flags, finalize, save_draft

    df = st.session_state["cat_df"]
    cats = cfg.load_categories()
    rs = compute_flags(df, cats)

    meta = cfg.load_meta()
    a, b, c = st.columns(3)
    a.metric("Transactions", len(rs.df))
    b.metric("Need attention", rs.n_attention)
    c.metric("Status", "✅ finalized" if meta.finalized else "📝 draft")

    if rs.over_limit:
        st.warning("**Over budget:** " + " · ".join(
            f"{o['category']} {o['month_label']} ${o['spent']:.0f}/${o['limit']:.0f}" for o in rs.over_limit))

    st.write("Edit the **category** column for any row. Rows needing attention are flagged with a reason.")
    only_attn = st.toggle("Show only rows needing attention", value=rs.n_attention > 0)
    view = rs.df.copy()
    view["row_id"] = view.index
    if only_attn:
        view = view[view["needs_attention"]]
    cols = ["row_id", "date", "description", "amount", "category", "confidence",
            "attention_reasons", "source_file"]
    view = view[[c for c in cols if c in view.columns]]

    edited = st.data_editor(
        view, use_container_width=True, hide_index=True, key="review_editor",
        column_config={
            "row_id": None,
            "date": st.column_config.TextColumn("Date", disabled=True),
            "description": st.column_config.TextColumn("Description", disabled=True, width="large"),
            "amount": st.column_config.NumberColumn("Amount", disabled=True, format="$%.2f"),
            "category": st.column_config.SelectboxColumn("Category", options=cats.all_names, required=True),
            "confidence": st.column_config.NumberColumn("Conf.", disabled=True, format="%.2f"),
            "attention_reasons": st.column_config.TextColumn("Why flagged", disabled=True, width="medium"),
            "source_file": st.column_config.TextColumn("Source", disabled=True),
        },
    )

    def _collect_corrections() -> dict[int, str]:
        return {int(r["row_id"]): str(r["category"]) for _, r in edited.iterrows()}

    s1, s2, _ = st.columns([1, 1, 3])
    if s1.button("💾 Save edits (draft)"):
        df2 = apply_corrections(rs.df.drop(columns=[c for c in ["needs_attention"] if c in rs.df], errors="ignore"),
                                _collect_corrections())
        save_draft(df2)
        st.session_state["cat_df"] = df2
        st.success("Saved draft. Re-run flags to see updates.")
        st.rerun()

    if s2.button("✅ Finalize", type="primary"):
        df2 = apply_corrections(rs.df, _collect_corrections())
        m = finalize(df2, cats)
        st.session_state["cat_df"] = pd.read_csv(paths.CATEGORIZED_CSV)
        st.success(f"Finalized {m.n_transactions} transactions (period {m.period}). "
                   "Learned merchant corrections for next time. → Go to **Report**.")
        st.balloons()


def page_report() -> None:
    st.header("3 · Report")
    meta = cfg.load_meta()
    if not meta.finalized:
        st.info("Finalize your categorization in **Categorize & review** first.")
        if not categorized_exists():
            return
        if not st.checkbox("Preview a draft report anyway"):
            return
        require_final = False
    else:
        require_final = True

    if st.button("📊 Generate report", type="primary"):
        from ocd.report import generate_report
        with st.spinner("Building charts and insights…"):
            out = generate_report(require_finalized=require_final)
        st.session_state["report"] = {k: out[k] for k in ("html", "markdown", "insights", "summary")}
        st.session_state["report_figs"] = out["figures"]

    if "report_figs" in st.session_state:
        rep = st.session_state["report"]
        figs = st.session_state["report_figs"]
        if rep.get("summary"):
            st.subheader("Summary")
            st.write(rep["summary"])
        st.subheader("Insights")
        for i in rep["insights"]:
            st.markdown(f"- {i}")
        st.plotly_chart(figs["trend"], use_container_width=True)
        cols = st.columns(2)
        cols[0].plotly_chart(figs["budget"], use_container_width=True)
        cols[1].plotly_chart(figs["share"], use_container_width=True)

        html_path, md_path = rep["html"], rep["markdown"]
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download HTML", html_path.read_text(), file_name=html_path.name,
                           mime="text/html")
        d2.download_button("⬇️ Download Markdown", md_path.read_text(), file_name=md_path.name,
                           mime="text/markdown")
        st.caption(f"Saved to {html_path} and {md_path}")


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
PAGES = {
    "⚙️ Setup": page_setup,
    "1 · Extract": page_extract,
    "2 · Categorize & review": page_categorize,
    "3 · Report": page_report,
}

st.sidebar.title("💸 OCD")
st.sidebar.caption("Optimized Cash Dashboard — local, personalized spending analysis.")
choice = st.sidebar.radio("Steps", list(PAGES.keys()),
                          index=0 if not cfg.categories_exist() else 1)
st.sidebar.divider()
meta = cfg.load_meta()
st.sidebar.caption(f"Categories: {len(cfg.load_categories().categories)} · "
                   f"Run: {'✅ finalized' if meta.finalized else '📝 draft' if paths.CATEGORIZED_CSV.exists() else '—'}")

PAGES[choice]()
