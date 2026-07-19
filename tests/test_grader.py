"""Grading pipeline: prompt building, response parsing, verdict rules."""

from __future__ import annotations

import json

import pytest

from shipready import GradingError, grade, summarize
from shipready.grader import applicable_criteria, build_user_prompt

from conftest import StubBackend


def test_build_user_prompt_contains_every_criterion_and_input(research_workbook):
    case = research_workbook.case("t1")
    prompt = build_user_prompt(research_workbook, case, "some answer")
    for c in research_workbook.framework:
        assert c.id in prompt
        assert c.criterion in prompt
    assert "some answer" in prompt
    assert "research_assistant" in prompt


def test_applicable_criteria_respects_applies_to(research_workbook):
    # No criterion in this workbook restricts applies_to, so all apply.
    case = research_workbook.case("t1")
    crits = applicable_criteria(research_workbook, case)
    assert len(crits) == len(research_workbook.framework)


def test_grade_all_pass(research_workbook, all_pass_response):
    case = research_workbook.case("t1")
    backend = StubBackend(all_pass_response)
    report = grade(research_workbook, case, "answer", backend=backend)
    assert report.ship_ready
    assert report.passed_count == report.total_count == len(research_workbook.framework)
    assert report.provider == "anthropic"


def test_grade_a_hard_fail_blocks_ship_ready(research_workbook):
    case = research_workbook.case("t1")
    grades = [
        {"criterion_id": c.id, "status": "pass", "justification": "ok"}
        for c in research_workbook.framework
    ]
    grades[0]["status"] = "fail"
    backend = StubBackend({"grades": grades})
    report = grade(research_workbook, case, "answer", backend=backend)
    assert not report.ship_ready
    assert report.grades[0].status == "fail"
    assert not report.grades[0].passed


def test_grade_warn_never_blocks_but_flags(research_workbook):
    case = research_workbook.case("t1")
    grades = [
        {"criterion_id": c.id, "status": "pass", "justification": "ok"}
        for c in research_workbook.framework
    ]
    grades[0]["status"] = "warn"
    backend = StubBackend({"grades": grades})
    report = grade(research_workbook, case, "answer", backend=backend)
    assert report.ship_ready
    assert report.has_warnings


def test_soft_fail_surfaces_without_blocking(research_workbook):
    # Make c5 soft for this test by mutating a copy of the workbook.
    wb = research_workbook.model_copy(deep=True)
    wb.criterion("c5").severity = "soft"
    case = wb.case("t1")
    grades = [{"criterion_id": c.id, "status": "pass", "justification": "ok"} for c in wb.framework]
    grades[-1]["status"] = "fail"  # c5, now soft
    backend = StubBackend({"grades": grades})
    report = grade(wb, case, "answer", backend=backend)
    assert report.ship_ready  # soft fail does not block
    assert report.has_warnings  # but it surfaces


def test_process_criterion_with_no_trace_downgrades_pass_to_warn(tool_using_workbook):
    case = tool_using_workbook.case("t1")
    case = case.model_copy(update={"tool_calls": None, "reasoning_trace": None})
    grades = [
        {"criterion_id": c.id, "status": "pass", "justification": "fine"}
        for c in tool_using_workbook.framework
    ]
    backend = StubBackend({"grades": grades})
    report = grade(tool_using_workbook, case, "answer", backend=backend)
    process_grades = [g for g in report.grades if g.criterion_id.startswith("p")]
    assert all(g.status == "warn" for g in process_grades)
    assert all("self-report" in g.justification.lower() or "no trace" in g.justification.lower() for g in process_grades)


def test_missing_criterion_in_response_raises_grading_error(research_workbook):
    case = research_workbook.case("t1")
    backend = StubBackend({"grades": [{"criterion_id": "c1", "status": "pass", "justification": "x"}]})
    with pytest.raises(GradingError, match="missing"):
        grade(research_workbook, case, "answer", backend=backend)


def test_empty_response_raises_grading_error(research_workbook):
    case = research_workbook.case("t1")
    backend = StubBackend("")
    with pytest.raises(GradingError, match="empty"):
        grade(research_workbook, case, "answer", backend=backend)


def test_response_wrapped_in_markdown_fence_is_parsed(research_workbook, all_pass_response):
    case = research_workbook.case("t1")
    fenced = "```json\n" + json.dumps(all_pass_response) + "\n```"
    backend = StubBackend(fenced)
    report = grade(research_workbook, case, "answer", backend=backend)
    assert report.ship_ready


def test_legacy_boolean_passed_field_tolerated(research_workbook):
    case = research_workbook.case("t1")
    grades = [
        {"criterion_id": c.id, "passed": True, "justification": "ok"}
        for c in research_workbook.framework
    ]
    backend = StubBackend({"grades": grades})
    report = grade(research_workbook, case, "answer", backend=backend)
    assert report.ship_ready


def test_summarize_makes_a_second_call_and_parses_summary(research_workbook, all_pass_response):
    case = research_workbook.case("t1")
    backend = StubBackend([all_pass_response])
    report = grade(research_workbook, case, "answer", backend=backend)

    summary_backend = StubBackend(
        {"went_well": ["a"], "flags": [], "watch": [], "verdict": "SHIP-READY: solid"}
    )
    summary = summarize(research_workbook, case, report, backend=summary_backend)
    assert summary.verdict.startswith("SHIP-READY")
    assert summary_backend.calls  # the call actually happened


def test_grading_prompt_tells_model_workbook_content_is_data_not_instructions(research_workbook):
    """Regression guard: the injection-resistance instruction must stay in
    the system prompt, since the t3 example workbook case embeds adversarial
    text on purpose (testing whether the *graded* agent resists it)."""
    from shipready.grader import SYSTEM_PROMPT

    assert "DATA to be graded" in SYSTEM_PROMPT
    assert "never an instruction" in SYSTEM_PROMPT
