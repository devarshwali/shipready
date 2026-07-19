"""Layer 3: headline metric computation."""

from __future__ import annotations

from shipready import CriterionGrade, GradingReport, compute_headline_metric, text_similarity


def _report(case_id, statuses_weights):
    grades = [
        CriterionGrade(
            criterion_id=f"c{i}",
            criterion=f"crit{i}",
            severity="hard",
            weight=w,
            status=s,
            label="ok" if s != "fail" else "not-ok",
            justification="because",
        )
        for i, (s, w) in enumerate(statuses_weights)
    ]
    return GradingReport(agent_name="a", case_id=case_id, model="m", provider="anthropic", grades=grades)


def test_text_similarity_identical_strings_is_100():
    assert text_similarity("hello world", "hello world") == 100.0


def test_text_similarity_empty_vs_nonempty_is_0():
    assert text_similarity("", "something") == 0.0


def test_text_similarity_both_empty_is_100():
    assert text_similarity("  ", "") == 100.0


def test_weighted_pass_rate_all_pass_is_100(research_workbook):
    report = _report("t1", [("pass", 1.0), ("pass", 2.0), ("pass", 1.0)])
    metric = compute_headline_metric(research_workbook, [report])
    assert metric.weighted_pass_rate == 100.0
    assert metric.score == 100.0
    assert metric.cases_scored == 1


def test_weighted_pass_rate_warn_counts_half():
    report = _report("t1", [("warn", 1.0), ("pass", 1.0)])
    from shipready.workbook import load_workbook
    from pathlib import Path

    wb = load_workbook(Path(__file__).resolve().parent.parent / "examples" / "research_assistant.yaml")
    metric = compute_headline_metric(wb, [report])
    assert metric.weighted_pass_rate == 75.0  # (0.5 + 1.0) / 2.0 * 100


def test_weighted_pass_rate_respects_criterion_weight():
    report = _report("t1", [("fail", 3.0), ("pass", 1.0)])
    from shipready.workbook import load_workbook
    from pathlib import Path

    wb = load_workbook(Path(__file__).resolve().parent.parent / "examples" / "research_assistant.yaml")
    metric = compute_headline_metric(wb, [report])
    assert metric.weighted_pass_rate == 25.0  # 1.0 / 4.0 * 100


def test_empty_reports_gives_zero_score(research_workbook):
    metric = compute_headline_metric(research_workbook, [])
    assert metric.cases_scored == 0
    assert metric.score == 0.0


def test_baseline_similarity_blended_into_score_when_present(research_workbook):
    wb = research_workbook.model_copy(deep=True)
    wb.data_set[0].baseline_output = "the exact answer text"
    report = _report("t1", [("pass", 1.0)] * len(wb.framework))
    metric = compute_headline_metric(wb, [report], outputs_by_case={"t1": "the exact answer text"})
    assert metric.baseline_similarity == 100.0
    assert metric.score == 100.0  # 100*0.7 + 100*0.3
