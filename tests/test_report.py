"""Report rendering: plain text, HTML export. (Rich rendering needs a real
terminal and is exercised manually / via the CLI smoke test in CI, not here.)"""

from __future__ import annotations

from shipready import (
    CriterionGrade,
    ExpertReview,
    GradingReport,
    Summary,
)
from shipready.report import (
    format_expert_review,
    format_report,
    format_summary,
    render_html_report,
)


def _sample_report(status="pass", severity="hard"):
    grade = CriterionGrade(
        criterion_id="c1",
        criterion="source_quality",
        severity=severity,
        weight=1.0,
        status=status,
        label="well_sourced" if status != "fail" else "weak_sourcing",
        justification="Every claim is sourced.",
    )
    return GradingReport(agent_name="research_assistant", case_id="t1", model="stub", grades=[grade])


def test_format_report_ship_ready_on_all_pass():
    report = _sample_report("pass")
    text = format_report(report)
    assert "SHIP-READY" in text
    assert "NOT READY" not in text
    assert "1/1 criteria passed" in text


def test_format_report_not_ready_on_hard_fail():
    report = _sample_report("fail", severity="hard")
    text = format_report(report)
    assert "NOT READY" in text


def test_format_report_soft_fail_is_ready_but_flagged():
    report = _sample_report("fail", severity="soft")
    text = format_report(report)
    assert "SHIP-READY" in text
    assert "(soft, non-blocking)" in text


def test_format_summary_includes_all_sections():
    summary = Summary(went_well=["a"], flags=["b"], watch=["c"], verdict="ok overall")
    text = format_summary(summary)
    assert "What went well" in text
    assert "Flags or warnings" in text
    assert "What to watch" in text
    assert "ok overall" in text


def test_format_expert_review_would_approve():
    review = ExpertReview(
        persona_role="a skeptic",
        assessment="Solid.",
        strengths=["sourced"],
        concerns=[],
        would_expert_approve=True,
        confidence="high",
    )
    text = format_expert_review(review)
    assert "WOULD APPROVE" in text
    assert "a skeptic" in text


def test_render_html_report_escapes_and_includes_verdict():
    grade = CriterionGrade(
        criterion_id="c1",
        criterion="<script>alert(1)</script>",
        severity="hard",
        weight=1.0,
        status="pass",
        label="ok",
        justification="fine",
    )
    report = GradingReport(agent_name="a", case_id="t1", model="m", grades=[grade])
    html = render_html_report(report)
    assert "<script>alert(1)</script>" not in html  # must be escaped
    assert "&lt;script&gt;" in html
    assert "SHIP-READY" in html


def test_render_html_report_with_summary_and_expert_review():
    grade = CriterionGrade(
        criterion_id="c1", criterion="x", severity="hard", weight=1.0,
        status="pass", label="ok", justification="fine",
    )
    report = GradingReport(
        agent_name="a", case_id="t1", model="m", grades=[grade],
        summary=Summary(went_well=["good"], flags=[], watch=[], verdict="fine"),
        expert_review=ExpertReview(
            persona_role="reviewer", assessment="ok", strengths=[], concerns=[],
            would_expert_approve=True, confidence="medium",
        ),
    )
    html = render_html_report(report)
    assert "Summary" in html
    assert "Expert review" in html
