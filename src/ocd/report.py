"""Step 3 — report.

Runs ONLY against a finalized categorization (the gate set by review.finalize). Produces:
  * interactive Plotly figures — monthly spending trend per category, spend-vs-budget per
    category, and category share,
  * minimalistic, rule-based insights (with an optional one-paragraph LLM summary),
  * a self-contained HTML report (Plotly inlined, opens offline) and a Markdown report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from jinja2 import Template

from . import config as cfg
from . import models, paths
from .config import CategoryConfig
from .review import compute_over_limit

logger = logging.getLogger(__name__)

PALETTE = pio.templates["plotly_white"].layout.colorway or None


class NotFinalizedError(RuntimeError):
    """Raised when a report is requested before categorization is finalized."""


@dataclass
class Aggregates:
    df: pd.DataFrame
    months: list[str]
    n_transactions: int
    total: float
    by_category: pd.DataFrame          # category, amount, n, share
    by_cat_month: pd.DataFrame         # category x month_label pivot of summed amount
    over_limit: list[dict]
    period: str = ""
    extras: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def compute_aggregates(df: pd.DataFrame, categories: CategoryConfig) -> Aggregates:
    df = df.copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    months = sorted(m for m in df["month_label"].dropna().unique())
    total = float(df["amount"].sum())

    by_category = (
        df.groupby("category")["amount"].agg(["sum", "count"]).reset_index()
        .rename(columns={"sum": "amount", "count": "n"})
        .sort_values("amount", ascending=False).reset_index(drop=True)
    )
    by_category["share"] = (by_category["amount"] / total * 100) if total else 0.0

    by_cat_month = (
        df.pivot_table(index="month_label", columns="category", values="amount",
                       aggfunc="sum", fill_value=0.0)
        .reindex(months).fillna(0.0)
    )

    over_limit = compute_over_limit(df, categories)
    period = months[0] if len(months) == 1 else (f"{months[0]} — {months[-1]}" if months else "")
    return Aggregates(df=df, months=months, n_transactions=len(df), total=total,
                      by_category=by_category, by_cat_month=by_cat_month,
                      over_limit=over_limit, period=period)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_monthly_trend(agg: Aggregates) -> go.Figure:
    fig = go.Figure()
    for cat in agg.by_cat_month.columns:
        fig.add_trace(go.Scatter(
            x=list(agg.by_cat_month.index), y=agg.by_cat_month[cat].round(2),
            mode="lines+markers", name=cat,
            hovertemplate=f"<b>{cat}</b><br>%{{x}}: $%{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Monthly spending trend by category", template="plotly_white",
        xaxis_title="Month", yaxis_title="Spend ($)", hovermode="x unified",
        legend_title="Category", margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


def fig_spend_vs_budget(agg: Aggregates, categories: CategoryConfig) -> go.Figure:
    """Average monthly spend per category vs the monthly limit (apples-to-apples)."""
    n_months = max(len(agg.months), 1)
    rows = []
    for _, r in agg.by_category.iterrows():
        limit = categories.limit_for(r["category"])
        rows.append((r["category"], r["amount"] / n_months, limit))
    rows.sort(key=lambda x: x[1], reverse=True)
    cats = [r[0] for r in rows]
    avg = [round(r[1], 2) for r in rows]
    limits = [r[2] for r in rows]
    colors = ["#d62728" if lim and a > lim else "#1f77b4" for a, lim in zip(avg, limits)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=cats, y=avg, marker_color=colors, name="Avg monthly spend",
                         hovertemplate="<b>%{x}</b><br>avg/mo: $%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=cats, y=limits, mode="markers", name="Monthly limit",
        marker=dict(symbol="line-ew-open", size=22, color="black", line=dict(width=3)),
        hovertemplate="<b>%{x}</b><br>limit: $%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Average monthly spend vs. limit (red = over budget)", template="plotly_white",
        xaxis_title="Category", yaxis_title="$ / month", margin=dict(t=60, l=60, r=30, b=80),
    )
    return fig


def fig_category_share(agg: Aggregates) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=agg.by_category["category"], values=agg.by_category["amount"].round(2),
        hole=0.45, textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>$%{value:.2f} (%{percent})<extra></extra>",
    ))
    fig.update_layout(title="Share of total spending", template="plotly_white",
                      margin=dict(t=60, l=20, r=20, b=20))
    return fig


def fig_budget_ratio(agg: Aggregates, categories: CategoryConfig, window: int = 30) -> go.Figure:
    """Trailing ``window``-day spend per category as a ratio of its monthly limit.

    Normalizes every category onto the same scale: **1.0 means spending exactly at the
    monthly budget**, so anything poking above the dashed 1.0 line is over budget — and
    *how far* above is directly comparable across categories regardless of dollar size.
    Categories without a monthly limit are omitted (nothing to normalize against)."""
    df = agg.df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    fig = go.Figure()
    if not df.empty:
        daily = (df.groupby([df["date"].dt.normalize(), "category"])["amount"].sum()
                   .unstack(fill_value=0.0))
        full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(full, fill_value=0.0)
        trailing = daily.rolling(window=window, min_periods=1).sum()  # trailing N-day spend
        for cat in trailing.columns:
            limit = categories.limit_for(cat)
            if not limit:
                continue
            fig.add_trace(go.Scatter(
                x=list(trailing.index), y=(trailing[cat] / limit).round(3),
                mode="lines", name=cat,
                hovertemplate=f"<b>{cat}</b><br>%{{x|%Y-%m-%d}}: %{{y:.2f}}× limit<extra></extra>",
            ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#d62728",
                  annotation_text="monthly limit (1.0×)", annotation_position="top left")
    fig.update_layout(
        title=f"{window}-day spending vs. limit (1.0× = at budget; above = over)",
        template="plotly_white", xaxis_title="Date",
        yaxis_title=f"{window}-day spend ÷ monthly limit", hovermode="x unified",
        legend_title="Category", margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


def build_figures(agg: Aggregates, categories: CategoryConfig) -> dict[str, go.Figure]:
    return {
        "trend": fig_monthly_trend(agg),
        "ratio": fig_budget_ratio(agg, categories),
        "budget": fig_spend_vs_budget(agg, categories),
        "share": fig_category_share(agg),
    }


# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #
def _pretty_merchant(description: str, maxlen: int = 36) -> str:
    """Make an ALL-CAPS, code-laden statement description nicer to read."""
    s = " ".join(str(description).split()).title()
    return (s[: maxlen - 1] + "…") if len(s) > maxlen else s


def rule_based_insights(agg: Aggregates, categories: CategoryConfig) -> list[str]:
    """Friendly, human-readable bullet insights (Markdown bold + a leading emoji each)."""
    if agg.n_transactions == 0:
        return ["🗂️ Nothing to analyze yet — add some statements and run the pipeline."]

    df = agg.df
    out: list[str] = []
    n_months = max(len(agg.months), 1)
    per_month = agg.total / n_months

    # 1) Headline
    span = "this month" if n_months == 1 else f"over {n_months} months"
    out.append(
        f"💸 You spent **${agg.total:,.0f}** {span} across **{agg.n_transactions}** purchases — "
        f"that's about **${per_month:,.0f} a month**."
    )

    # 2) Where the money went
    top = agg.by_category.head(3)
    if not top.empty:
        parts = [f"**{r['category']}** (${r['amount']:,.0f}, {r['share']:.0f}%)"
                 for _, r in top.iterrows()]
        cat_str = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f", and {parts[-1]}"
        out.append(f"🏆 Most of your money went to {cat_str}.")

    # 3) Largest single purchase
    big = df.loc[df["amount"].idxmax()]
    when = f" on {big['date']}" if "date" in df.columns and pd.notna(big.get("date")) else ""
    out.append(f"🧾 Your biggest single purchase was **${float(big['amount']):,.0f}** at "
               f"{_pretty_merchant(big['description'])}{when}.")

    # 4) Go-to merchant (only if there's a clear repeat)
    if "merchant_key" in df.columns:
        counts = df["merchant_key"].value_counts()
        if len(counts) and int(counts.iloc[0]) >= 3:
            key = counts.index[0]
            spent = float(df.loc[df["merchant_key"] == key, "amount"].sum())
            out.append(f"🔁 Your go-to spot was **{_pretty_merchant(key)}** — "
                       f"{int(counts.iloc[0])} visits totalling **${spent:,.0f}**.")

    # 5) Budget health
    if agg.over_limit:
        worst = sorted(agg.over_limit, key=lambda o: o["over_by"], reverse=True)[:3]
        bullets = [f"**{o['category']}** ({o['month_label']}, ${o['spent']:,.0f} vs ${o['limit']:,.0f})"
                   for o in worst]
        out.append("⚠️ You went over budget in " + "; ".join(bullets) + ".")
    else:
        out.append("🎉 Nice — every category stayed within its monthly limit.")

    # 6) Month-over-month movement of the top category
    if len(agg.months) >= 2 and not agg.by_category.empty:
        top_cat = agg.by_category.iloc[0]["category"]
        series = agg.by_cat_month[top_cat]
        prev, curr = float(series.iloc[-2]), float(series.iloc[-1])
        if prev > 0:
            chg = (curr - prev) / prev * 100
            if chg >= 5:
                out.append(f"📈 {top_cat} spending climbed **{chg:.0f}%** last month "
                           f"(${prev:,.0f} → ${curr:,.0f}) — worth a glance.")
            elif chg <= -5:
                out.append(f"📉 Nice trim: {top_cat} spending fell **{abs(chg):.0f}%** last month "
                           f"(${prev:,.0f} → ${curr:,.0f}).")
            else:
                out.append(f"➡️ {top_cat} spending held steady month-over-month (~${curr:,.0f}).")
    return out


def llm_summary(agg: Aggregates, insights: list[str]) -> Optional[str]:
    """Optional one-paragraph natural-language summary via the 'insights' model role."""
    if not models.is_enabled("insights"):
        return None
    try:
        client = models.get_client("insights")
        model = models.get_model("insights")
        facts = "\n".join(f"- {i}" for i in insights)
        by_cat = "\n".join(f"- {r['category']}: ${r['amount']:.0f} ({r['share']:.0f}%)"
                           for _, r in agg.by_category.iterrows())
        prompt = (
            "Write a concise, friendly 2-3 sentence summary of this month's spending for the "
            "user. Be specific and minimalistic; no preamble, no bullet points.\n\n"
            f"Facts:\n{facts}\n\nBy category:\n{by_cat}"
        )
        resp = client.chat.completions.create(
            model=model, temperature=0.3,
            messages=[{"role": "system", "content": "You are a concise personal-finance assistant."},
                      {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:  # noqa: BLE001 - summary is optional; never block the report
        logger.warning("LLM summary skipped: %s", e)
        return None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
_HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>OCD Spending Report — {{ period }}</title>
<script type="text/javascript">{{ plotlyjs }}</script>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0; background: #f5f6f8; color: #1d2127; }
  .wrap { max-width: 980px; margin: 0 auto; padding: 28px 22px 64px; }
  h1 { margin: 0 0 4px; font-size: 26px; }
  .sub { color: #6b7280; margin-bottom: 22px; }
  .summary { background: #fff; border-radius: 12px; padding: 18px 20px; margin-bottom: 22px;
             box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .insights li { margin: 6px 0; }
  .card { background: #fff; border-radius: 12px; padding: 10px 12px; margin-bottom: 22px;
          box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  table { border-collapse: collapse; width: 100%; font-size: 14px; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #eceef1; }
  th { color: #6b7280; font-weight: 600; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .over { color: #d62728; font-weight: 600; }
  footer { color: #9aa0a6; font-size: 12px; margin-top: 30px; }
</style></head><body><div class="wrap">
  <h1>💸 Spending Report</h1>
  <div class="sub">{{ period }} · generated {{ generated }}</div>

  <div class="summary">
    {% if llm_summary %}<p style="font-size:16px;margin-top:0">{{ llm_summary }}</p>{% endif %}
    <ul class="insights">{% for i in insights_html %}<li>{{ i }}</li>{% endfor %}</ul>
  </div>

  <div class="card">{{ fig_trend }}</div>
  <div class="card">{{ fig_ratio }}</div>
  <div class="card">{{ fig_budget }}</div>
  <div class="card">{{ fig_share }}</div>

  <div class="card">
    <table><thead><tr><th>Category</th><th class="num">Spent</th>
      <th class="num">Share</th><th class="num">#</th><th class="num">Avg/mo</th>
      <th class="num">Limit</th></tr></thead><tbody>
    {% for r in cat_rows %}<tr>
      <td>{{ r.category }}</td>
      <td class="num">${{ '%.2f'|format(r.amount) }}</td>
      <td class="num">{{ '%.0f'|format(r.share) }}%</td>
      <td class="num">{{ r.n }}</td>
      <td class="num {{ 'over' if r.over else '' }}">${{ '%.2f'|format(r.avg) }}</td>
      <td class="num">{{ '$%.0f'|format(r.limit) if r.limit else '—' }}</td>
    </tr>{% endfor %}
    </tbody></table>
  </div>

  <footer>Generated locally by OCD — Optimized Cash Dashboard. All processing on-device.</footer>
</div></body></html>""")


def _cat_rows(agg: Aggregates, categories: CategoryConfig) -> list[dict]:
    n_months = max(len(agg.months), 1)
    rows = []
    for _, r in agg.by_category.iterrows():
        limit = categories.limit_for(r["category"])
        avg = r["amount"] / n_months
        rows.append({"category": r["category"], "amount": float(r["amount"]),
                     "share": float(r["share"]), "n": int(r["n"]), "avg": avg,
                     "limit": float(limit), "over": bool(limit and avg > limit)})
    return rows


def render_html(agg: Aggregates, categories: CategoryConfig, figures: dict[str, go.Figure],
                insights: list[str], summary: Optional[str], generated: str) -> str:
    import re

    def _to_div(fig: go.Figure) -> str:
        return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                           config={"displayModeBar": False, "responsive": True})

    # markdown-ish bold -> html for the insight bullets
    def _md_bold(s: str) -> str:
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)

    return _HTML_TEMPLATE.render(
        period=agg.period or "All transactions",
        generated=generated,
        plotlyjs=get_plotlyjs(),
        insights_html=[_md_bold(i) for i in insights],
        llm_summary=summary,
        fig_trend=_to_div(figures["trend"]),
        fig_ratio=_to_div(figures["ratio"]),
        fig_budget=_to_div(figures["budget"]),
        fig_share=_to_div(figures["share"]),
        cat_rows=_cat_rows(agg, categories),
    )


def render_markdown(agg: Aggregates, categories: CategoryConfig, insights: list[str],
                    summary: Optional[str], generated: str, html_name: str) -> str:
    lines = [f"# 💸 Spending Report — {agg.period or 'All transactions'}",
             f"_Generated {generated} · interactive charts in [{html_name}]({html_name})_", ""]
    if summary:
        lines += ["> " + summary, ""]
    lines.append("## Insights")
    lines += [f"- {i}" for i in insights]
    lines += ["", "## Spending by category", "",
              "| Category | Spent | Share | # | Avg/mo | Limit |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in _cat_rows(agg, categories):
        limit = f"${r['limit']:.0f}" if r["limit"] else "—"
        avg = f"${r['avg']:.2f}" + (" ⚠️" if r["over"] else "")
        lines.append(f"| {r['category']} | ${r['amount']:.2f} | {r['share']:.0f}% | "
                     f"{r['n']} | {avg} | {limit} |")
    lines += ["", f"**Total:** ${agg.total:,.2f} across {agg.n_transactions} purchases."]
    if agg.over_limit:
        lines += ["", "## ⚠️ Over budget"]
        for o in sorted(agg.over_limit, key=lambda x: x["over_by"], reverse=True):
            lines.append(f"- **{o['category']}** in {o['month_label']}: "
                         f"${o['spent']:,.0f} vs ${o['limit']:,.0f} (over by ${o['over_by']:,.0f})")
    lines += ["", "## Monthly trend by category", "",
              "| Month | " + " | ".join(agg.by_cat_month.columns) + " |",
              "|---|" + "---:|" * len(agg.by_cat_month.columns)]
    for month, row in agg.by_cat_month.iterrows():
        lines.append(f"| {month} | " + " | ".join(f"${v:,.0f}" for v in row) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def generate_report(
    categorized_csv: Optional[Path] = None,
    categories: Optional[CategoryConfig] = None,
    require_finalized: bool = True,
    generated: Optional[str] = None,
) -> dict:
    """Generate the Markdown + HTML report. Raises NotFinalizedError unless the run is
    finalized (pass ``require_finalized=False`` to preview a draft)."""
    categories = categories or cfg.load_categories()
    meta = cfg.load_meta()
    if require_finalized and not meta.finalized:
        raise NotFinalizedError(
            "Categorization is not finalized yet. Review and finalize "
            "(in the web app or `ocd categorize --finalize`) before generating the report."
        )

    df = pd.read_csv(categorized_csv if categorized_csv is not None else paths.CATEGORIZED_CSV)
    agg = compute_aggregates(df, categories)
    figures = build_figures(agg, categories)
    insights = rule_based_insights(agg, categories)
    summary = llm_summary(agg, insights)
    generated = generated or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    slug = (agg.period or "all").replace(" — ", "_to_").replace(" ", "")
    html_name = f"report_{slug}.html"
    md_name = f"report_{slug}.md"
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = paths.REPORTS_DIR / html_name
    md_path = paths.REPORTS_DIR / md_name

    html_path.write_text(render_html(agg, categories, figures, insights, summary, generated))
    md_path.write_text(render_markdown(agg, categories, insights, summary, generated, html_name))
    logger.info("Wrote %s and %s", html_path, md_path)
    return {"html": html_path, "markdown": md_path, "aggregates": agg,
            "insights": insights, "summary": summary, "figures": figures}
