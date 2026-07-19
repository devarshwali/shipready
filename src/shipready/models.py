"""Pydantic models for shipready workbooks, grading reports, expert reviews,
and headline metrics.

A workbook is the rubric for one agent. It has four parts:

    goals       what the agent is for
    boundaries  what the agent must not do
    framework   the grading criteria (each scored pass, warn, or fail)
    data_set    the test cases to grade against

Lineage: this project began as a from-scratch rebuild and extension of a
different, earlier project of the same name, agnitrip/shipready
(https://github.com/agnitrip/shipready) by Agni Tripathi, MIT licensed.
See LICENSE and README for the full naming/lineage note.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, computed_field, model_validator


def _require_unique_ids(items, label):
    """Raise if any two items in a list share an id."""
    seen = set()
    for item in items:
        if item.id in seen:
            raise ValueError(f"duplicate {label} id: {item.id!r}")
        seen.add(item.id)


class Goal(BaseModel):
    """One thing the agent is supposed to accomplish."""

    id: str
    description: str
    sub_goals: List[str] = Field(default_factory=list)


class Boundary(BaseModel):
    """A line the agent must not cross. Informs grading, not scored directly."""

    id: str
    name: str
    what_it_means: str
    example: Optional[str] = None


class Criterion(BaseModel):
    """One graded dimension. Each criterion resolves to pass, warn, or fail.

    target selects which artifact the criterion is graded against. "output"
    grades the agent's final answer. "process" grades the agent's behavior
    through the supplied trace artifacts (tool calls, reasoning, decisions,
    escalations). Defaults to "output".

    severity controls whether a fail blocks ship-readiness. "hard" (default)
    blocks; "soft" surfaces without blocking.

    weight feeds the Layer 3 headline metric: a criterion's contribution to
    the weighted pass rate is proportional to its weight. Defaults to 1.0, so
    an unweighted framework reduces to a plain pass rate.

    applies_to scopes a criterion to specific expected branches (test cases
    whose expected_verdict is in this list). None means it always applies.
    """

    id: str
    criterion: str
    grades_what: str
    pass_label: str
    fail_label: str
    target: Literal["output", "process"] = "output"
    severity: Literal["hard", "soft"] = "hard"
    weight: float = 1.0
    applies_to: Optional[List[str]] = None


class ToolCall(BaseModel):
    """One tool invocation in the agent's trace."""

    tool: str
    args: dict = Field(default_factory=dict)
    returned: Optional[str] = None
    step: Optional[int] = None


class Decision(BaseModel):
    """One decision the agent made during the run."""

    at: str
    decision: str
    rationale: Optional[str] = None


class EscalationEvent(BaseModel):
    """One point where the agent escalated or handed off."""

    at: str
    reason: str
    handed_off_to: Optional[str] = None


class TestCase(BaseModel):
    """One input the agent is expected to handle, plus the target behavior.

    __test__ = False tells pytest not to try collecting this as a test class
    (its name starts with "Test", which pytest's default collector matches).

    baseline_output is optional reference text (a known-good human or gold
    answer) used by the Layer 3 headline metric to compute similarity to a
    baseline. It is never shown to the grading model as part of the rubric
    prompt; it is only used for the separate, deterministic metrics pass.
    """

    id: str
    input: str
    expected_behavior: str
    notes: Optional[str] = None
    expected_verdict: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    reasoning_trace: Optional[str] = None
    decisions_log: Optional[List[Decision]] = None
    escalation_events: Optional[List[EscalationEvent]] = None
    baseline_output: Optional[str] = None

    __test__ = False  # not a pytest test class despite the name


class ExpertPersona(BaseModel):
    """A synthetic domain-expert reviewer persona (Layer 2).

    Used when there is no human baseline to compare against. The persona
    description is folded into the expert-review prompt so the model grades
    the way a specified kind of expert would, rather than as a generic judge.
    """

    role: str
    credentials: Optional[str] = None
    stance: Optional[str] = None


class Workbook(BaseModel):
    """The full per-agent rubric loaded from a YAML file."""

    agent_name: str
    description: str
    goals: List[Goal] = Field(default_factory=list)
    boundaries: List[Boundary] = Field(default_factory=list)
    framework: List[Criterion]
    data_set: List[TestCase]
    expert_persona: Optional[ExpertPersona] = None

    @model_validator(mode="after")
    def _validate(self) -> "Workbook":
        if not self.framework:
            raise ValueError("framework must define at least one criterion")
        if not self.data_set:
            raise ValueError("data_set must define at least one test case")
        _require_unique_ids(self.goals, "goal")
        _require_unique_ids(self.boundaries, "boundary")
        _require_unique_ids(self.framework, "criterion")
        _require_unique_ids(self.data_set, "test case")
        return self

    def case(self, case_id: str) -> TestCase:
        for tc in self.data_set:
            if tc.id == case_id:
                return tc
        known = ", ".join(tc.id for tc in self.data_set)
        raise KeyError(f"no test case {case_id!r} in workbook (have: {known})")

    def criterion(self, criterion_id: str) -> Criterion:
        for c in self.framework:
            if c.id == criterion_id:
                return c
        raise KeyError(f"no criterion {criterion_id!r} in workbook")


class CriterionGrade(BaseModel):
    """The verdict for one criterion on one candidate output."""

    criterion_id: str
    criterion: str
    severity: Literal["hard", "soft"] = "hard"
    weight: float = 1.0
    status: Literal["pass", "warn", "fail"]
    label: str
    justification: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return self.status != "fail"


class Summary(BaseModel):
    """PM-facing synthesis of a grading report from a second grading pass."""

    went_well: List[str] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)
    watch: List[str] = Field(default_factory=list)
    verdict: str


class ExpertReview(BaseModel):
    """Layer 2: a synthetic domain-expert's narrative assessment.

    Distinct from the per-criterion rubric grading: this is a holistic
    qualitative read, the kind of judgment call you'd want from a human
    expert when there is no baseline to diff against.
    """

    persona_role: str
    assessment: str
    strengths: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    would_expert_approve: bool
    confidence: Literal["low", "medium", "high"]


class GradingReport(BaseModel):
    """The full set of criterion verdicts for one graded output."""

    agent_name: str
    case_id: str
    model: str
    provider: str = "anthropic"
    grades: List[CriterionGrade]
    summary: Optional[Summary] = None
    expert_review: Optional[ExpertReview] = None

    @property
    def passed_count(self) -> int:
        return sum(1 for g in self.grades if g.passed)

    @property
    def total_count(self) -> int:
        return len(self.grades)

    @property
    def has_warnings(self) -> bool:
        return any(
            g.status == "warn" or (g.severity == "soft" and g.status == "fail")
            for g in self.grades
        )

    @property
    def ship_ready(self) -> bool:
        return not any(
            g.severity == "hard" and g.status == "fail" for g in self.grades
        )


class HeadlineMetric(BaseModel):
    """Layer 3: a single configurable output-fidelity score for a batch of
    reports.

    weighted_pass_rate: sum of criterion weights for pass/warn(0.5) verdicts
        over total weight, expressed 0-100. warn counts as half credit.
    escalation_rate: fraction of cases carrying at least one escalation event,
        over cases whose workbook criteria include a process target. None
        when not applicable.
    baseline_similarity: mean text-similarity (0-100) between candidate
        output and each case's baseline_output, over cases that have one.
        None when no case in the batch supplies a baseline.
    score: the headline number itself. Defaults to weighted_pass_rate; when
        baseline_similarity is available it is blended in.
    """

    cases_scored: int
    weighted_pass_rate: float
    escalation_rate: Optional[float] = None
    baseline_similarity: Optional[float] = None
    score: float
    per_case: dict = Field(default_factory=dict)
