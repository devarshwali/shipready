"""Layer 2: AI-as-expert-evaluator.

For domains with no human baseline to compare against, this module runs a
second model pass that grades the way a specified kind of domain expert
would -- not per-criterion pass/fail, but a holistic narrative judgment:
would this expert actually sign off on it?

This is deliberately a *different* kind of pass than grade()/summarize():
the rubric pass is mechanical and reproducible (same criteria every time);
the expert pass is a synthesized professional opinion, useful when your
rubric might be missing something a real domain expert would catch.
"""

from __future__ import annotations

from typing import Optional

from .models import ExpertPersona, ExpertReview, TestCase, Workbook
from .providers import DEFAULT_PROVIDER, build_backend
from .providers.anthropic_provider import DEFAULT_MODEL
from .providers.base import BackendError, GraderBackend

__all__ = [
    "DEFAULT_PERSONA",
    "ExpertReviewError",
    "build_expert_prompt",
    "expert_review",
]

EXPERT_SYSTEM_PROMPT_TEMPLATE = (
    "You are role-playing a specific expert reviewer: {role}. "
    "{credentials_clause}{stance_clause}"
    "You are not grading against a fixed rubric. You are giving the holistic, "
    "professional judgment that expert would actually give if handed this "
    "agent's output and asked 'would you sign off on this?'. Be candid: real "
    "experts raise concerns rubrics miss, and don't rubber-stamp confident-"
    "sounding but shaky work. Respond with a single JSON object and nothing "
    "else, in this shape: "
    '{{"assessment": "<2-4 sentence overall read>", '
    '"strengths": ["..."], "concerns": ["..."], '
    '"would_expert_approve": true|false, '
    '"confidence": "low|medium|high"}}. '
    "confidence reflects how confident you, this expert, are in your own "
    "verdict given the information available -- lower it when the material "
    "needed to judge properly is thin."
)

DEFAULT_PERSONA = ExpertPersona(
    role="a senior practitioner in the agent's domain with no stake in the outcome",
    credentials=None,
    stance="Judge strictly. A polished answer that is subtly wrong is worse than "
    "an honest 'I don't know'.",
)


class ExpertReviewError(Exception):
    """Raised when the expert pass cannot be parsed into an ExpertReview."""


def _system_prompt(persona: ExpertPersona) -> str:
    credentials_clause = (
        f"Your background: {persona.credentials.strip()}. " if persona.credentials else ""
    )
    stance_clause = f"{persona.stance.strip()} " if persona.stance else ""
    return EXPERT_SYSTEM_PROMPT_TEMPLATE.format(
        role=persona.role.strip(),
        credentials_clause=credentials_clause,
        stance_clause=stance_clause,
    )


def build_expert_prompt(workbook: Workbook, case: TestCase, agent_output: str) -> str:
    """Assemble the user message for the expert-review pass."""
    output_block = agent_output.strip() or "(the agent produced no output)"
    return (
        f'Agent under review: "{workbook.agent_name}" -- '
        f"{workbook.description.strip()}\n\n"
        f"Case input:\n{case.input.strip()}\n\n"
        f"Expected behavior (for context only, not a checklist to tick):\n"
        f"{case.expected_behavior.strip()}\n\n"
        f"AGENT OUTPUT TO REVIEW:\n{output_block}\n\n"
        "Give your expert assessment now."
    )


def expert_review(
    workbook: Workbook,
    case: TestCase,
    agent_output: str,
    model: Optional[str] = None,
    backend: Optional[GraderBackend] = None,
    provider: str = DEFAULT_PROVIDER,
    persona: Optional[ExpertPersona] = None,
    max_tokens: int = 1000,
) -> ExpertReview:
    """Run the Layer 2 synthetic expert-evaluator pass.

    persona precedence: explicit argument, then workbook.expert_persona, then
    DEFAULT_PERSONA. This mirrors how CLI trace flags override the workbook.
    """
    persona = persona or workbook.expert_persona or DEFAULT_PERSONA

    if backend is None:
        backend = build_backend(provider, model or DEFAULT_MODEL)

    from .grader import _extract_json  # reuse the tolerant JSON extractor

    try:
        text = backend.complete(
            _system_prompt(persona),
            build_expert_prompt(workbook, case, agent_output),
            max_tokens,
        )
    except BackendError as exc:
        raise ExpertReviewError(str(exc)) from exc

    if not text.strip():
        raise ExpertReviewError("expert model returned an empty response")

    data = _extract_json(text)
    try:
        return ExpertReview(
            persona_role=persona.role,
            assessment=str(data.get("assessment", "")).strip(),
            strengths=[str(s) for s in data.get("strengths", [])],
            concerns=[str(s) for s in data.get("concerns", [])],
            would_expert_approve=bool(data.get("would_expert_approve", False)),
            confidence=data.get("confidence", "low")
            if data.get("confidence") in ("low", "medium", "high")
            else "low",
        )
    except Exception as exc:
        raise ExpertReviewError(f"could not parse expert review response: {exc}") from exc
