"""Shared fixtures: a stub GraderBackend so tests never hit a real API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shipready import Workbook, load_workbook

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


class StubBackend:
    """A GraderBackend that returns pre-scripted JSON text.

    Construct with either a single response (used for every call) or a list
    of responses (consumed in order, one per .complete() call) so tests can
    script multi-call flows like grade -> summarize -> expert_review.
    """

    def __init__(self, responses, model: str = "stub-model"):
        self.model = model
        self._responses = responses if isinstance(responses, list) else [responses]
        self._i = 0
        self.calls = []

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        response = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return response if isinstance(response, str) else json.dumps(response)


@pytest.fixture
def research_workbook() -> Workbook:
    return load_workbook(EXAMPLES_DIR / "research_assistant.yaml")


@pytest.fixture
def tool_using_workbook() -> Workbook:
    return load_workbook(EXAMPLES_DIR / "tool_using_research_assistant.yaml")


@pytest.fixture
def all_pass_response(research_workbook):
    return {
        "grades": [
            {"criterion_id": c.id, "status": "pass", "justification": "looks solid"}
            for c in research_workbook.framework
        ]
    }
