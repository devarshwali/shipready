"""shipready (devarshwali/shipready): rubric based ship-readiness evals
for AI agents.

Multi-provider, with an expert-review pass and a configurable headline
metric on top of the original outcome/process rubric grading.

Lineage: this project began as a from-scratch rebuild and extension of a
different, earlier project of the same name, agnitrip/shipready
(https://github.com/agnitrip/shipready) by Agni Tripathi, MIT licensed.
See LICENSE and README for the full naming/lineage note and credits.
"""

from .expert import DEFAULT_PERSONA, ExpertReviewError, expert_review
from .grader import DEFAULT_MODEL, DEFAULT_PROVIDER, GradingError, grade, summarize
from .metrics import compute_headline_metric, text_similarity
from .models import (
    Boundary,
    Criterion,
    CriterionGrade,
    Decision,
    EscalationEvent,
    ExpertPersona,
    ExpertReview,
    Goal,
    GradingReport,
    HeadlineMetric,
    Summary,
    TestCase,
    ToolCall,
    Workbook,
)
from .report import format_headline_metric, format_report, format_summary
from .workbook import WorkbookError, load_workbook

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER",
    "GradingError",
    "grade",
    "summarize",
    "expert_review",
    "ExpertReviewError",
    "DEFAULT_PERSONA",
    "compute_headline_metric",
    "text_similarity",
    "Boundary",
    "Criterion",
    "CriterionGrade",
    "Decision",
    "EscalationEvent",
    "ExpertPersona",
    "ExpertReview",
    "Goal",
    "GradingReport",
    "HeadlineMetric",
    "Summary",
    "TestCase",
    "ToolCall",
    "Workbook",
    "format_report",
    "format_summary",
    "format_headline_metric",
    "WorkbookError",
    "load_workbook",
]
