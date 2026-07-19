"""Layer 3: a single configurable headline metric over a batch of reports.

Rubric grading (grader.py) tells you pass/warn/fail per criterion per case.
That is legible but doesn't answer "one number I can watch trend over time."
This module computes that number from a batch of GradingReports plus the
workbook and test cases they were graded against, using three deterministic,
non-LLM signals:

    weighted_pass_rate   criterion-weight-adjusted pass rate (warn = half
                          credit), 0-100.
    escalation_rate       fraction of process-eval cases that actually
                          escalated, 0-100, or None if not applicable.
    baseline_similarity   mean text similarity to each case's baseline_output
                          (when supplied), 0-100, or None if no case has one.

score blends whichever of these are available. All of it is deterministic
(difflib, arithmetic) -- no extra model calls, so computing it is free and
reproducible.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List, Optional

from .models import GradingReport, HeadlineMetric, TestCase, Workbook

__all__ = ["text_similarity", "compute_headline_metric"]


def text_similarity(a: str, b: str) -> float:
    """Character-level similarity ratio between two strings, 0-100.

    Uses difflib's SequenceMatcher (no extra dependency, deterministic). This
    is a coarse proxy for "how close is this to the baseline", not a
    semantic-similarity model -- good enough for a directional trend metric,
    not for grading individual cases.
    """
    a, b = a.strip(), b.strip()
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio() * 100.0


def _case_weighted_score(report: GradingReport) -> tuple[float, float]:
    """Return (earned, possible) weight for one report's criteria.

    warn earns half its weight; pass earns full; fail earns none. A soft
    fail still earns none here -- the headline metric reflects rubric
    strictness, ship_ready reflects blocking, and they're allowed to differ.
    """
    earned = 0.0
    possible = 0.0
    for g in report.grades:
        possible += g.weight
        if g.status == "pass":
            earned += g.weight
        elif g.status == "warn":
            earned += g.weight * 0.5
    return earned, possible


def compute_headline_metric(
    workbook: Workbook,
    reports: List[GradingReport],
    outputs_by_case: Optional[Dict[str, str]] = None,
) -> HeadlineMetric:
    """Compute the Layer 3 headline metric for a batch of graded reports.

    outputs_by_case optionally maps case_id -> the candidate agent output
    text actually graded, used to compute baseline_similarity against each
    case's TestCase.baseline_output. When omitted, baseline_similarity is
    skipped even if baselines exist in the workbook.
    """
    if not reports:
        return HeadlineMetric(
            cases_scored=0, weighted_pass_rate=0.0, score=0.0, per_case={}
        )

    per_case: dict = {}
    total_earned = 0.0
    total_possible = 0.0

    escal_applicable = 0
    escal_hit = 0

    similarities: List[float] = []

    for report in reports:
        earned, possible = _case_weighted_score(report)
        total_earned += earned
        total_possible += possible
        case_rate = (earned / possible * 100.0) if possible else 0.0

        case_entry = {"weighted_pass_rate": round(case_rate, 2)}

        try:
            case: TestCase = workbook.case(report.case_id)
        except KeyError:
            case = None  # type: ignore[assignment]

        has_process_criteria = any(
            g.criterion_id and workbook.criterion(g.criterion_id).target == "process"
            for g in report.grades
            if g.criterion_id in {c.id for c in workbook.framework}
        )
        if case is not None and has_process_criteria:
            escal_applicable += 1
            if case.escalation_events:
                escal_hit += 1
            case_entry["escalated"] = bool(case.escalation_events)

        if case is not None and case.baseline_output and outputs_by_case:
            candidate = outputs_by_case.get(report.case_id)
            if candidate is not None:
                sim = text_similarity(candidate, case.baseline_output)
                similarities.append(sim)
                case_entry["baseline_similarity"] = round(sim, 2)

        per_case[report.case_id] = case_entry

    weighted_pass_rate = (
        (total_earned / total_possible * 100.0) if total_possible else 0.0
    )
    escalation_rate = (
        (escal_hit / escal_applicable * 100.0) if escal_applicable else None
    )
    baseline_similarity = (
        (sum(similarities) / len(similarities)) if similarities else None
    )

    # Blend: weighted_pass_rate is always the backbone. When a baseline
    # similarity signal exists, blend it in at 30% so the headline number
    # reflects both rubric compliance and closeness to a known-good answer.
    # Escalation rate is reported alongside but not blended into score,
    # since "should escalate" is workbook-specific (sometimes high is good,
    # sometimes low is good) and blending it in either direction would be a
    # silent assumption.
    if baseline_similarity is not None:
        score = weighted_pass_rate * 0.7 + baseline_similarity * 0.3
    else:
        score = weighted_pass_rate

    return HeadlineMetric(
        cases_scored=len(reports),
        weighted_pass_rate=round(weighted_pass_rate, 2),
        escalation_rate=round(escalation_rate, 2) if escalation_rate is not None else None,
        baseline_similarity=round(baseline_similarity, 2)
        if baseline_similarity is not None
        else None,
        score=round(score, 2),
        per_case=per_case,
    )
