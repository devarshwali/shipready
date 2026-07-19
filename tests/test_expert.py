"""Layer 2: synthetic expert-evaluator pass."""

from __future__ import annotations

import pytest

from shipready import DEFAULT_PERSONA, ExpertPersona, ExpertReviewError, expert_review

from conftest import StubBackend


def test_expert_review_uses_workbook_persona(research_workbook):
    wb = research_workbook.model_copy(
        update={"expert_persona": ExpertPersona(role="a skeptical epidemiologist")}
    )
    case = wb.case("t1")
    backend = StubBackend(
        {
            "assessment": "Solid, sourced answer.",
            "strengths": ["cites UN IGME"],
            "concerns": [],
            "would_expert_approve": True,
            "confidence": "high",
        }
    )
    review = expert_review(wb, case, "some answer", backend=backend)
    assert review.persona_role == "a skeptical epidemiologist"
    assert review.would_expert_approve is True
    assert review.confidence == "high"
    # the persona role must actually reach the model prompt
    assert "skeptical epidemiologist" in backend.calls[0]["system"]


def test_expert_review_falls_back_to_default_persona(research_workbook):
    case = research_workbook.case("t1")
    backend = StubBackend(
        {
            "assessment": "ok",
            "strengths": [],
            "concerns": ["thin sourcing"],
            "would_expert_approve": False,
            "confidence": "low",
        }
    )
    review = expert_review(research_workbook, case, "answer", backend=backend)
    assert review.persona_role == DEFAULT_PERSONA.role
    assert review.would_expert_approve is False


def test_expert_review_invalid_confidence_defaults_to_low(research_workbook):
    case = research_workbook.case("t1")
    backend = StubBackend(
        {
            "assessment": "ok",
            "strengths": [],
            "concerns": [],
            "would_expert_approve": True,
            "confidence": "extremely-sure",  # invalid value
        }
    )
    review = expert_review(research_workbook, case, "answer", backend=backend)
    assert review.confidence == "low"


def test_expert_review_empty_response_raises(research_workbook):
    case = research_workbook.case("t1")
    backend = StubBackend("")
    with pytest.raises(ExpertReviewError, match="empty"):
        expert_review(research_workbook, case, "answer", backend=backend)
