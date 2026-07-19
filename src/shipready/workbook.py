"""Load and validate workbook YAML files into Workbook models."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml
from pydantic import ValidationError

from .models import Workbook


class WorkbookError(Exception):
    """Raised when a workbook file cannot be loaded or fails validation."""


def load_workbook(path: Union[str, Path]) -> Workbook:
    """Read a YAML workbook from disk and return a validated Workbook.

    Raises WorkbookError with a readable message on any failure: missing file,
    malformed YAML, or a schema violation. Uses yaml.safe_load deliberately:
    a workbook is untrusted-ish input (it may be shared or come from a repo
    you didn't write), so no arbitrary Python object construction is allowed.
    """
    path = Path(path)
    if not path.exists():
        raise WorkbookError(f"workbook not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise WorkbookError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise WorkbookError(
            f"workbook {path} must be a YAML mapping at the top level"
        )

    try:
        return Workbook.model_validate(raw)
    except ValidationError as exc:
        raise WorkbookError(f"workbook {path} failed validation:\n{exc}") from exc
