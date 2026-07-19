# Changelog

## 1.0.0

Initial release of this repository (devarshwali/shipready): a
from-scratch rebuild and extension of the separate, earlier
agnitrip/shipready project by Agni Tripathi (MIT). See README for the
naming/lineage note.

- Rebuilt the workbook layer (Goals, Boundaries, Framework, Data Set),
  outcome/process grading split, and pass/warn/fail + hard/soft verdict
  model from the ground up.
- **New:** pluggable multi-provider grading (`GraderBackend` protocol) with
  Anthropic, OpenAI, and `local` (any OpenAI-compatible endpoint) backends.
- **New:** Layer 2, AI-as-expert-evaluator (`shipready.expert`, `--expert`)
  -- a synthetic domain-expert persona pass, configurable per workbook.
- **New:** Layer 3, headline metric (`shipready.metrics`, `shipready score`)
  -- criterion-weighted pass rate, escalation rate, and baseline-similarity
  blend, computed deterministically with no extra model calls.
- **New:** `weight` field on Criterion (feeds the headline metric) and
  `baseline_output` field on TestCase (feeds baseline similarity).
- **New:** rich colorized terminal output (auto-detected, falls back
  cleanly without the dependency) and self-contained `--html-out` HTML
  report export.
- **New:** an explicit anti-prompt-injection instruction in the grading
  system prompt, with a regression test guarding it.
- **New:** GitHub Actions CI (ruff, mypy, pytest+coverage, CLI smoke test)
  across Python 3.11 and 3.12; project is mypy-clean and ruff-clean.
- 34 unit tests, all provider calls stubbed (zero network calls in the
  suite).
