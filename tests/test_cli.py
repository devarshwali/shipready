"""CLI smoke tests via click's CliRunner. No network calls: these only
exercise commands/paths that don't need a live model (validate, cases,
--dry-run, and score against a hand-written reports.json)."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from shipready.cli import cli

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
WORKBOOK = str(EXAMPLES_DIR / "research_assistant.yaml")
OUTPUT_FILE = str(EXAMPLES_DIR / "sample_outputs" / "research_assistant_t1_good.txt")


def test_validate_ok():
    result = CliRunner().invoke(cli, ["validate", "--workbook", WORKBOOK])
    assert result.exit_code == 0
    assert "research_assistant" in result.output


def test_validate_missing_file():
    result = CliRunner().invoke(cli, ["validate", "--workbook", "does/not/exist.yaml"])
    assert result.exit_code != 0


def test_cases_no_color_lists_all_case_ids():
    result = CliRunner().invoke(cli, ["cases", "--workbook", WORKBOOK, "--no-color"])
    assert result.exit_code == 0
    assert "t1" in result.output and "t2" in result.output and "t3" in result.output


def test_grade_dry_run_prints_prompt_without_calling_model():
    result = CliRunner().invoke(
        cli,
        [
            "grade",
            "--workbook", WORKBOOK,
            "--case", "t1",
            "--output-file", OUTPUT_FILE,
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "===== SYSTEM =====" in result.output
    assert "===== USER =====" in result.output


def test_grade_requires_case_or_all():
    result = CliRunner().invoke(
        cli, ["grade", "--workbook", WORKBOOK, "--output-file", OUTPUT_FILE, "--dry-run"]
    )
    assert result.exit_code != 0
    assert "provide --case" in result.output


def test_grade_local_provider_requires_base_url():
    result = CliRunner().invoke(
        cli,
        [
            "grade",
            "--workbook", WORKBOOK,
            "--case", "t1",
            "--output-file", OUTPUT_FILE,
            "--provider", "local",
        ],
    )
    assert result.exit_code != 0
    assert "--base-url" in result.output


def test_score_command_computes_headline_metric(tmp_path):
    from shipready import CriterionGrade, GradingReport, load_workbook

    wb = load_workbook(WORKBOOK)
    grades = [
        CriterionGrade(
            criterion_id=c.id, criterion=c.criterion, severity=c.severity,
            weight=c.weight, status="pass", label=c.pass_label, justification="ok",
        )
        for c in wb.framework
    ]
    report = GradingReport(agent_name=wb.agent_name, case_id="t1", model="stub", grades=grades)
    reports_path = tmp_path / "reports.json"
    reports_path.write_text(json.dumps([json.loads(report.model_dump_json())]), encoding="utf-8")

    result = CliRunner().invoke(
        cli, ["score", "--workbook", WORKBOOK, "--reports", str(reports_path)]
    )
    assert result.exit_code == 0
    assert "SCORE:" in result.output
    assert "100.0" in result.output


def test_score_command_json_output(tmp_path):
    from shipready import CriterionGrade, GradingReport, load_workbook

    wb = load_workbook(WORKBOOK)
    grades = [
        CriterionGrade(
            criterion_id=c.id, criterion=c.criterion, severity=c.severity,
            weight=c.weight, status="pass", label=c.pass_label, justification="ok",
        )
        for c in wb.framework
    ]
    report = GradingReport(agent_name=wb.agent_name, case_id="t1", model="stub", grades=grades)
    reports_path = tmp_path / "reports.json"
    reports_path.write_text(json.dumps([json.loads(report.model_dump_json())]), encoding="utf-8")

    result = CliRunner().invoke(
        cli, ["score", "--workbook", WORKBOOK, "--reports", str(reports_path), "--json"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["score"] == 100.0
