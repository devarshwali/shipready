"""Render a GradingReport as a plain-text card, a rich terminal card, or HTML.

Three render targets share one data model (GradingReport / HeadlineMetric):
  - format_report / format_summary: the original plain-text card, always
    available, used for --json-free piping and as the rich fallback.
  - render_rich_report: colorized terminal rendering via `rich`, used by the
    CLI when stdout is a real terminal and --no-color wasn't passed.
  - render_html_report: a self-contained HTML file, no external assets, so
    it opens straight from disk or attaches cleanly to a PR/ticket.
"""

from __future__ import annotations

import html as _html
import textwrap

from .models import ExpertReview, GradingReport, HeadlineMetric, Summary


def _wrap(text: str, indent: str) -> str:
    return textwrap.fill(text, width=78, initial_indent=indent, subsequent_indent=indent)


# ---------------------------------------------------------------------------
# Plain text (unchanged rendering contract from the original tool)
# ---------------------------------------------------------------------------


def format_summary(summary: Summary) -> str:
    lines = ["=" * 62, "SUMMARY", "=" * 62, "What went well:"]
    for bullet in summary.went_well:
        lines.append(_wrap(f"- {bullet}", indent=""))
    lines.append("")
    lines.append("Flags or warnings:")
    for bullet in summary.flags:
        lines.append(_wrap(f"- {bullet}", indent=""))
    if summary.watch:
        lines.append("")
        lines.append("What to watch:")
        for bullet in summary.watch:
            lines.append(_wrap(f"- {bullet}", indent=""))
    lines.append("")
    lines.append(_wrap(f"Verdict: {summary.verdict}", indent=""))
    return "\n".join(lines)


def format_expert_review(review: ExpertReview) -> str:
    lines = ["=" * 62, f"EXPERT REVIEW  |  persona: {review.persona_role}", "=" * 62]
    lines.append(_wrap(review.assessment, indent=""))
    lines.append("")
    if review.strengths:
        lines.append("Strengths:")
        for s in review.strengths:
            lines.append(_wrap(f"- {s}", indent=""))
    if review.concerns:
        lines.append("Concerns:")
        for c in review.concerns:
            lines.append(_wrap(f"- {c}", indent=""))
    lines.append("")
    verdict = "WOULD APPROVE" if review.would_expert_approve else "WOULD NOT APPROVE"
    lines.append(f"{verdict}  (confidence: {review.confidence})")
    return "\n".join(lines)


def format_report(report: GradingReport) -> str:
    card_lines = [
        "=" * 62,
        f"shipready report  |  agent: {report.agent_name}",
        f"case: {report.case_id}  |  provider: {report.provider}  |  model: {report.model}",
        "=" * 62,
    ]

    for g in report.grades:
        mark = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[g.status]
        line = f"[{mark}] {g.criterion_id}  {g.criterion}  ({g.label})"
        if g.status == "fail" and g.severity == "soft":
            line += "  (soft, non-blocking)"
        card_lines.append(line)
        if g.justification:
            card_lines.append(_wrap(g.justification, indent="       "))
        card_lines.append("")

    if not report.ship_ready:
        verdict = "NOT READY"
    elif report.has_warnings:
        verdict = "SHIP-READY (with warnings)"
    else:
        verdict = "SHIP-READY"
    card_lines.append("-" * 62)
    card_lines.append(
        f"{report.passed_count}/{report.total_count} criteria passed  ->  {verdict}"
    )
    card_lines.append("-" * 62)

    card = "\n".join(card_lines)
    blocks = []
    if report.summary is not None:
        blocks.append(format_summary(report.summary))
    blocks.append(card)
    if report.expert_review is not None:
        blocks.append(format_expert_review(report.expert_review))
    return "\n\n".join(blocks)


def format_headline_metric(metric: HeadlineMetric) -> str:
    lines = ["=" * 62, "HEADLINE METRIC", "=" * 62]
    lines.append(f"cases scored:        {metric.cases_scored}")
    lines.append(f"weighted pass rate:  {metric.weighted_pass_rate:.1f}")
    if metric.escalation_rate is not None:
        lines.append(f"escalation rate:     {metric.escalation_rate:.1f}")
    if metric.baseline_similarity is not None:
        lines.append(f"baseline similarity: {metric.baseline_similarity:.1f}")
    lines.append("-" * 62)
    lines.append(f"SCORE: {metric.score:.1f} / 100")
    lines.append("-" * 62)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rich terminal rendering
# ---------------------------------------------------------------------------


def render_rich_report(report: GradingReport, console) -> None:
    """Print a colorized version of the report card via a rich.Console.

    Import of rich is deferred to here so the module (and the whole package)
    still imports fine in an environment without rich installed; the CLI
    falls back to format_report's plain text in that case.
    """
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    if report.summary is not None:
        s = report.summary
        body = Text()
        body.append("What went well:\n", style="bold")
        for b in s.went_well:
            body.append(f"  - {b}\n")
        if s.flags:
            body.append("Flags or warnings:\n", style="bold yellow")
            for b in s.flags:
                body.append(f"  - {b}\n", style="yellow")
        if s.watch:
            body.append("What to watch:\n", style="bold")
            for b in s.watch:
                body.append(f"  - {b}\n")
        body.append(f"\nVerdict: {s.verdict}", style="bold")
        console.print(Panel(body, title="SUMMARY", border_style="cyan"))

    table = Table(title=f"shipready report | {report.agent_name} / {report.case_id}")
    table.add_column("Status")
    table.add_column("Criterion")
    table.add_column("Label")
    table.add_column("Justification", overflow="fold")

    style_by_status = {"pass": "bold green", "warn": "bold yellow", "fail": "bold red"}
    for g in report.grades:
        mark = g.status.upper()
        if g.status == "fail" and g.severity == "soft":
            mark += " (soft)"
        table.add_row(
            Text(mark, style=style_by_status[g.status]),
            f"{g.criterion_id} {g.criterion}",
            g.label,
            g.justification,
        )
    console.print(table)

    if not report.ship_ready:
        verdict, style = "NOT READY", "bold red"
    elif report.has_warnings:
        verdict, style = "SHIP-READY (with warnings)", "bold yellow"
    else:
        verdict, style = "SHIP-READY", "bold green"
    console.print(
        f"[{style}]{report.passed_count}/{report.total_count} criteria passed  ->  {verdict}[/{style}]"
    )

    if report.expert_review is not None:
        r = report.expert_review
        body = Text()
        body.append(f"{r.assessment}\n\n")
        if r.strengths:
            body.append("Strengths:\n", style="bold green")
            for line in r.strengths:
                body.append(f"  - {line}\n")
        if r.concerns:
            body.append("Concerns:\n", style="bold red")
            for c in r.concerns:
                body.append(f"  - {c}\n")
        verdict_style = "bold green" if r.would_expert_approve else "bold red"
        verdict_text = "WOULD APPROVE" if r.would_expert_approve else "WOULD NOT APPROVE"
        body.append(f"\n{verdict_text} (confidence: {r.confidence})", style=verdict_style)
        console.print(Panel(body, title=f"EXPERT REVIEW: {r.persona_role}", border_style="magenta"))


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>shipready report: {agent_name} / {case_id}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
          max-width: 860px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; }}
  .meta {{ color: #666; margin-bottom: 1.5rem; }}
  .verdict {{ font-weight: 700; padding: 0.5rem 1rem; border-radius: 6px; display: inline-block; }}
  .verdict.ready {{ background: #e6f6ec; color: #0a7a33; }}
  .verdict.warn {{ background: #fff8e1; color: #8a6100; }}
  .verdict.fail {{ background: #fdecea; color: #b3261e; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ text-align: left; padding: 0.5rem 0.6rem; border-bottom: 1px solid #eee; vertical-align: top; }}
  th {{ background: #fafafa; }}
  .status {{ font-weight: 700; border-radius: 4px; padding: 0.15rem 0.5rem; font-size: 0.85rem; }}
  .status.pass {{ background: #e6f6ec; color: #0a7a33; }}
  .status.warn {{ background: #fff8e1; color: #8a6100; }}
  .status.fail {{ background: #fdecea; color: #b3261e; }}
  .card {{ border: 1px solid #e5e5e5; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }}
  .card h2 {{ margin-top: 0; font-size: 1.05rem; }}
  ul {{ margin: 0.25rem 0; padding-left: 1.25rem; }}
</style>
</head>
<body>
<h1>shipready report</h1>
<div class="meta">agent: <strong>{agent_name}</strong> &nbsp;|&nbsp; case: <strong>{case_id}</strong>
&nbsp;|&nbsp; provider: {provider} &nbsp;|&nbsp; model: {model}</div>
{summary_block}
<table>
<thead><tr><th>Status</th><th>Criterion</th><th>Label</th><th>Justification</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
<p><span class="verdict {verdict_class}">{passed}/{total} criteria passed &rarr; {verdict}</span></p>
{expert_block}
</body>
</html>
"""


def _esc(text: str) -> str:
    return _html.escape(text, quote=True)


def render_html_report(report: GradingReport) -> str:
    """Render a self-contained HTML report card (no external assets)."""
    rows = []
    for g in report.grades:
        soft_note = " (soft, non-blocking)" if g.status == "fail" and g.severity == "soft" else ""
        rows.append(
            f'<tr><td><span class="status {g.status}">{g.status.upper()}{soft_note}</span></td>'
            f"<td>{_esc(g.criterion_id)} {_esc(g.criterion)}</td>"
            f"<td>{_esc(g.label)}</td><td>{_esc(g.justification)}</td></tr>"
        )

    if not report.ship_ready:
        verdict, verdict_class = "NOT READY", "fail"
    elif report.has_warnings:
        verdict, verdict_class = "SHIP-READY (with warnings)", "warn"
    else:
        verdict, verdict_class = "SHIP-READY", "ready"

    summary_block = ""
    if report.summary is not None:
        s = report.summary
        summary_block = (
            '<div class="card"><h2>Summary</h2>'
            "<p><strong>What went well</strong></p><ul>"
            + "".join(f"<li>{_esc(b)}</li>" for b in s.went_well)
            + "</ul><p><strong>Flags</strong></p><ul>"
            + "".join(f"<li>{_esc(b)}</li>" for b in s.flags)
            + "</ul>"
            + (
                "<p><strong>Watch</strong></p><ul>"
                + "".join(f"<li>{_esc(b)}</li>" for b in s.watch)
                + "</ul>"
                if s.watch
                else ""
            )
            + f"<p><strong>Verdict:</strong> {_esc(s.verdict)}</p></div>"
        )

    expert_block = ""
    if report.expert_review is not None:
        r = report.expert_review
        approve = "Would approve" if r.would_expert_approve else "Would NOT approve"
        expert_block = (
            f'<div class="card"><h2>Expert review — {_esc(r.persona_role)}</h2>'
            f"<p>{_esc(r.assessment)}</p>"
            "<p><strong>Strengths</strong></p><ul>"
            + "".join(f"<li>{_esc(s)}</li>" for s in r.strengths)
            + "</ul><p><strong>Concerns</strong></p><ul>"
            + "".join(f"<li>{_esc(c)}</li>" for c in r.concerns)
            + f"</ul><p><strong>{approve}</strong> (confidence: {r.confidence})</p></div>"
        )

    return _HTML_TEMPLATE.format(
        agent_name=_esc(report.agent_name),
        case_id=_esc(report.case_id),
        provider=_esc(report.provider),
        model=_esc(report.model),
        summary_block=summary_block,
        rows="\n".join(rows),
        passed=report.passed_count,
        total=report.total_count,
        verdict=verdict,
        verdict_class=verdict_class,
        expert_block=expert_block,
    )
