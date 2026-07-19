"""shipready command line interface."""

from __future__ import annotations

import json
import os
import sys

import click
from pydantic import ValidationError

from . import __version__
from .expert import DEFAULT_PERSONA, ExpertReviewError, expert_review
from .grader import DEFAULT_MODEL, DEFAULT_PROVIDER, GradingError, grade, render_prompt, summarize
from .metrics import compute_headline_metric
from .models import GradingReport, TestCase
from .providers import build_backend
from .providers.base import BackendError
from .report import (
    format_headline_metric,
    format_report,
    render_html_report,
)
from .workbook import WorkbookError, load_workbook


def _console(no_color: bool):
    """Return a rich Console if rich is installed and color is wanted, else None.

    None signals the caller to fall back to format_report's plain text, so
    shipready degrades gracefully in a minimal environment (rich is a soft
    dependency) or when piped to a file / CI log where color adds noise.
    """
    if no_color:
        return None
    try:
        from rich.console import Console
    except ImportError:
        return None
    console = Console()
    if not console.is_terminal:
        return None
    return console


def _load_or_exit(workbook_path: str):
    try:
        return load_workbook(workbook_path)
    except WorkbookError as exc:
        raise click.ClickException(str(exc))


def _read_text_file(path: str, label: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise click.ClickException(f"could not read {label} file: {exc}")


def _load_json_list(path: str, label: str) -> list:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"{label} file is not valid JSON: {exc}")
    except OSError as exc:
        raise click.ClickException(f"could not read {label} file: {exc}")
    if not isinstance(data, list):
        raise click.ClickException(f"{label} file must contain a JSON array")
    return data


def _attach_trace(case: TestCase, tool_calls, reasoning_trace, decisions, escalations) -> TestCase:
    overrides: dict[str, object] = {}
    if tool_calls is not None:
        overrides["tool_calls"] = _load_json_list(tool_calls, "--tool-calls")
    if reasoning_trace is not None:
        overrides["reasoning_trace"] = _read_text_file(reasoning_trace, "--reasoning-trace")
    if decisions is not None:
        overrides["decisions_log"] = _load_json_list(decisions, "--decisions")
    if escalations is not None:
        overrides["escalation_events"] = _load_json_list(escalations, "--escalations")

    if not overrides:
        return case

    data = case.model_dump()
    data.update(overrides)
    try:
        return TestCase.model_validate(data)
    except ValidationError as exc:
        raise click.ClickException(f"trace artifact failed validation:\n{exc}")


def _friendly_api_error(exc: Exception) -> str:
    if isinstance(exc, GradingError):
        return f"grading failed: {exc}"
    if isinstance(exc, ExpertReviewError):
        return f"expert review failed: {exc}"
    if isinstance(exc, BackendError):
        return str(exc)
    return f"model call failed: {exc}"


def _build_backend_or_exit(provider, model, base_url, api_key_env):
    api_key = os.environ.get(api_key_env) if api_key_env else None
    try:
        return build_backend(provider, model or DEFAULT_MODEL, base_url=base_url, api_key=api_key)
    except ValueError as exc:
        raise click.ClickException(str(exc))


def _required_key_env(provider: str) -> str:
    return {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "local": "OPENAI_API_KEY"}.get(
        provider, "ANTHROPIC_API_KEY"
    )


def _grade_case(workbook, case, agent_output, backend, provider, summary, do_expert, persona):
    report = grade(workbook, case, agent_output, backend=backend, provider=provider)
    if summary:
        try:
            report.summary = summarize(workbook, case, report, backend=backend, provider=provider)
        except Exception as exc:
            click.echo(f"warning: summary synthesis failed for case {case.id}: {exc}", err=True)
    if do_expert:
        try:
            report.expert_review = expert_review(
                workbook, case, agent_output, backend=backend, provider=provider, persona=persona
            )
        except Exception as exc:
            click.echo(f"warning: expert review failed for case {case.id}: {exc}", err=True)
    return report


def _write_out(path: str, content: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
    except OSError as exc:
        raise click.ClickException(f"could not write --out file: {exc}")
    click.echo(f"wrote {path}", err=True)


def _resolve_output(output: str, output_file, label: str) -> str:
    if output is not None:
        return output
    if output_file is not None:
        return output_file.read()
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            return data
    raise click.ClickException(
        f"no agent output supplied for {label}. Pass --output, --output-file, "
        "or pipe the output on stdin."
    )


def _print_report(report: GradingReport, as_json: bool, no_color: bool, html_out: str | None):
    if html_out:
        with open(html_out, "w", encoding="utf-8") as fh:
            fh.write(render_html_report(report))
        click.echo(f"wrote {html_out}", err=True)

    if as_json:
        click.echo(report.model_dump_json(indent=2, exclude_none=True))
        return

    console = _console(no_color)
    if console is not None:
        from .report import render_rich_report

        render_rich_report(report, console)
    else:
        click.echo(format_report(report))


_PROVIDER_OPTION = click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "local"]),
    default=DEFAULT_PROVIDER,
    show_default=True,
    help="Grading backend. 'local' talks OpenAI-compatible chat completions "
    "to whatever --base-url you point at (Ollama, vLLM, LM Studio, ...).",
)
_MODEL_OPTION = click.option(
    "--model",
    default=None,
    help="Model id for the chosen provider (defaults per-provider).",
)
_BASE_URL_OPTION = click.option(
    "--base-url", default=None, help="Override API base URL (required for --provider local)."
)
_API_KEY_ENV_OPTION = click.option(
    "--api-key-env",
    default=None,
    help="Env var to read the API key from (defaults to the provider's usual var).",
)


@click.group()
@click.version_option(version=__version__, prog_name="shipready")
def cli() -> None:
    """Rubric based ship-readiness evals for AI agents."""


@cli.command(name="grade")
@click.option("--workbook", "workbook_path", required=True, type=click.Path(exists=False, dir_okay=False))
@click.option("--case", "case_id", default=None, help="Test case id to grade (or use --all).")
@click.option("--all", "grade_all", is_flag=True, help="Grade every case in the workbook.")
@click.option("--output", default=None, help="Candidate agent output as an inline string.")
@click.option("--output-file", type=click.File("r", encoding="utf-8"), default=None)
@click.option("--tool-calls", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--reasoning-trace", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--decisions", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--escalations", type=click.Path(exists=True, dir_okay=False), default=None)
@_PROVIDER_OPTION
@_MODEL_OPTION
@_BASE_URL_OPTION
@_API_KEY_ENV_OPTION
@click.option("--verbose", is_flag=True, help="Print the exact prompt sent to the model.")
@click.option("--dry-run", "dry_run", is_flag=True, help="Print the prompt and exit without calling the model.")
@click.option("--summary", is_flag=True, help="Add a PM-facing summary block (second model call).")
@click.option("--expert", "do_expert", is_flag=True, help="Add a Layer 2 synthetic expert-review pass (extra model call).")
@click.option("--json", "as_json", is_flag=True, help="Emit the grading report as JSON.")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None, help="Write JSON report(s) to this file.")
@click.option("--html-out", "html_out", type=click.Path(dir_okay=False), default=None, help="Write a self-contained HTML report card.")
@click.option("--no-color", is_flag=True, help="Disable rich colorized output even on a terminal.")
def grade_cmd(
    workbook_path, case_id, grade_all, output, output_file,
    tool_calls, reasoning_trace, decisions, escalations,
    provider, model, base_url, api_key_env,
    verbose, dry_run, summary, do_expert, as_json, out_path, html_out, no_color,
):
    """Grade one test case, or every case in a workbook with --all."""
    workbook = _load_or_exit(workbook_path)

    if grade_all and case_id:
        raise click.ClickException("use --case or --all, not both.")
    if not grade_all and not case_id:
        raise click.ClickException("provide --case CASE_ID or --all.")

    trace_flags = (tool_calls, reasoning_trace, decisions, escalations)
    if grade_all and any(f is not None for f in trace_flags):
        raise click.ClickException(
            "trace flags apply to a single --case. For --all, embed each case's trace in the workbook."
        )
    if provider == "local" and not base_url:
        raise click.ClickException("--provider local requires --base-url.")

    if grade_all:
        cases = list(workbook.data_set)
    else:
        try:
            cases = [workbook.case(case_id)]
        except KeyError as exc:
            raise click.ClickException(str(exc))
        cases = [_attach_trace(cases[0], tool_calls, reasoning_trace, decisions, escalations)]

    agent_output = _resolve_output(output, output_file, label="grading")

    if dry_run:
        for case in cases:
            click.echo(render_prompt(workbook, case, agent_output))
        sys.exit(0)

    if verbose:
        for case in cases:
            click.echo(render_prompt(workbook, case, agent_output), err=True)

    key_env = api_key_env or _required_key_env(provider)
    if provider != "local" and not os.environ.get(key_env):
        raise click.ClickException(f"{key_env} is not set. See the README install section for setup.")

    backend = _build_backend_or_exit(provider, model, base_url, api_key_env)
    persona = workbook.expert_persona or DEFAULT_PERSONA

    if not grade_all:
        case = cases[0]
        try:
            report = _grade_case(workbook, case, agent_output, backend, provider, summary, do_expert, persona)
        except Exception as exc:
            raise click.ClickException(_friendly_api_error(exc))
        if out_path:
            _write_out(out_path, report.model_dump_json(indent=2, exclude_none=True))
        _print_report(report, as_json, no_color, html_out)
        sys.exit(0 if report.ship_ready else 1)

    reports, failures = [], []
    for case in cases:
        try:
            report = _grade_case(workbook, case, agent_output, backend, provider, summary, do_expert, persona)
        except Exception as exc:
            failures.append((case.id, _friendly_api_error(exc)))
            continue
        reports.append(report)
        _print_report(report, as_json, no_color, None)
        click.echo("")

    for case_id_failed, message in failures:
        click.echo(f"FAILED to grade case {case_id_failed}: {message}", err=True)

    if out_path:
        payload = "[\n" + ",\n".join(r.model_dump_json(indent=2, exclude_none=True) for r in reports) + "\n]"
        _write_out(out_path, payload)

    if html_out:
        # For --all, HTML export writes one file per case beside the given path.
        base, ext = os.path.splitext(html_out)
        for r in reports:
            path = f"{base}.{r.case_id}{ext or '.html'}"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(render_html_report(r))
            click.echo(f"wrote {path}", err=True)

    not_ready = [r.case_id for r in reports if not r.ship_ready]
    click.echo(
        f"graded {len(reports)}/{len(cases)} cases, {len(failures)} failed, {len(not_ready)} not ready",
        err=True,
    )
    sys.exit(1 if (failures or not_ready) else 0)


@cli.command()
@click.option("--workbook", "workbook_path", required=True, type=click.Path(exists=False, dir_okay=False))
def validate(workbook_path):
    """Load a workbook and report whether it is valid."""
    workbook = _load_or_exit(workbook_path)
    click.echo(f"ok: {workbook.agent_name} ({len(workbook.framework)} criteria, {len(workbook.data_set)} cases)")


@cli.command()
@click.option("--workbook", "workbook_path", required=True, type=click.Path(exists=False, dir_okay=False))
@click.option("--no-color", is_flag=True)
def cases(workbook_path, no_color):
    """List the test cases defined in a workbook."""
    workbook = _load_or_exit(workbook_path)
    console = _console(no_color)
    if console is not None:
        from rich.table import Table

        table = Table(title=f"{workbook.agent_name} test cases")
        table.add_column("id")
        table.add_column("input (preview)")
        for tc in workbook.data_set:
            first_line = tc.input.strip().splitlines()[0] if tc.input.strip() else ""
            table.add_row(tc.id, first_line[:70])
        console.print(table)
        return
    for tc in workbook.data_set:
        first_line = tc.input.strip().splitlines()[0] if tc.input.strip() else ""
        click.echo(f"{tc.id}\t{first_line[:70]}")


@cli.command()
@click.option("--workbook", "workbook_path", required=True, type=click.Path(exists=False, dir_okay=False))
@click.option(
    "--reports",
    "reports_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="JSON file of report(s) previously written via `grade --out` (single object or array).",
)
@click.option(
    "--outputs-dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Optional directory of <case_id>.txt files (the graded outputs) used to "
    "compute baseline_similarity against each case's baseline_output.",
)
@click.option("--json", "as_json", is_flag=True)
def score(workbook_path, reports_path, outputs_dir, as_json):
    """Compute the Layer 3 headline metric over a batch of grading reports."""
    workbook = _load_or_exit(workbook_path)
    with open(reports_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    raw_list = raw if isinstance(raw, list) else [raw]
    try:
        reports = [GradingReport.model_validate(r) for r in raw_list]
    except ValidationError as exc:
        raise click.ClickException(f"--reports file failed validation:\n{exc}")

    outputs_by_case = None
    if outputs_dir:
        outputs_by_case = {}
        for r in reports:
            candidate_path = os.path.join(outputs_dir, f"{r.case_id}.txt")
            if os.path.exists(candidate_path):
                with open(candidate_path, "r", encoding="utf-8") as fh:
                    outputs_by_case[r.case_id] = fh.read()

    metric = compute_headline_metric(workbook, reports, outputs_by_case)
    if as_json:
        click.echo(metric.model_dump_json(indent=2, exclude_none=True))
    else:
        click.echo(format_headline_metric(metric))


if __name__ == "__main__":
    cli()
