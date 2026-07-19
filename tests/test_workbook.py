"""Workbook loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pathlib import Path

from shipready import Criterion, TestCase, Workbook, WorkbookError, load_workbook

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_load_research_assistant_workbook():
    wb = load_workbook(EXAMPLES_DIR / "research_assistant.yaml")
    assert wb.agent_name == "research_assistant"
    assert len(wb.framework) == 5
    assert len(wb.data_set) == 3
    assert wb.case("t1").id == "t1"
    assert wb.criterion("c1").criterion == "source_quality"


def test_load_tool_using_workbook_has_process_criteria():
    wb = load_workbook(EXAMPLES_DIR / "tool_using_research_assistant.yaml")
    targets = {c.id: c.target for c in wb.framework}
    assert targets["p1"] == "process"
    assert targets["o1"] == "output"


def test_missing_file_raises_workbook_error(tmp_path):
    with pytest.raises(WorkbookError, match="not found"):
        load_workbook(tmp_path / "nope.yaml")


def test_malformed_yaml_raises_workbook_error(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("agent_name: [unterminated", encoding="utf-8")
    with pytest.raises(WorkbookError, match="invalid YAML"):
        load_workbook(p)


def test_non_mapping_yaml_raises_workbook_error(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(WorkbookError, match="mapping"):
        load_workbook(p)


def test_duplicate_criterion_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate criterion id"):
        Workbook(
            agent_name="a",
            description="d",
            framework=[
                Criterion(id="c1", criterion="x", grades_what="x", pass_label="p", fail_label="f"),
                Criterion(id="c1", criterion="y", grades_what="y", pass_label="p", fail_label="f"),
            ],
            data_set=[TestCase(id="t1", input="i", expected_behavior="e")],
        )


def test_empty_framework_rejected():
    with pytest.raises(ValidationError, match="at least one criterion"):
        Workbook(
            agent_name="a",
            description="d",
            framework=[],
            data_set=[TestCase(id="t1", input="i", expected_behavior="e")],
        )


def test_criterion_defaults():
    c = Criterion(id="c1", criterion="x", grades_what="x", pass_label="p", fail_label="f")
    assert c.target == "output"
    assert c.severity == "hard"
    assert c.weight == 1.0
    assert c.applies_to is None


def test_unknown_case_id_raises_keyerror(research_workbook):
    with pytest.raises(KeyError):
        research_workbook.case("does-not-exist")
