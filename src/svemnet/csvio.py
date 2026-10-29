"""Shared CSV parsing and report-output safeguards for optional workflows."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

import numpy as np


_NUMERIC_MISSING = frozenset(
    {"NA", "N/A", "n/a", "NaN", "nan", "NULL", "null", "None", "#N/A", "<NA>"}
)


def parse_csv_payload(payload, *, text_columns=()):
    """Parse UTF-8 CSV with blank missing cells and explicit text overrides.

    NA-like tokens are missing only in otherwise numeric columns. Explicit
    text columns retain every nonblank raw field, including leading zeros.
    Returns immutable raw rows, a new DataFrame, and numeric-token counts.
    """
    import pandas as pd

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Could not decode CSV; save as CSV UTF-8 and try again") from exc
    rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    if not rows or not rows[0] or any(not name for name in rows[0]):
        raise ValueError("CSV requires nonempty column names")
    if len(rows[0]) != len(set(rows[0])):
        raise ValueError("duplicate CSV column names are not supported")
    width = len(rows[0])
    rows = [rows[0], *[(row if row else [""] * width) for row in rows[1:]]]
    if any(len(row) != width for row in rows):
        raise ValueError("CSV rows must all have the same number of fields")
    if isinstance(text_columns, str):
        raise ValueError("text_columns must be a sequence of column names")
    text_columns = tuple(dict.fromkeys(text_columns))
    unknown = set(text_columns).difference(rows[0])
    if unknown:
        raise ValueError(f"CSV text columns not found: {sorted(unknown)}")
    data = pd.read_csv(
        io.BytesIO(payload), encoding="utf-8-sig", skip_blank_lines=False,
        keep_default_na=False, na_values=[""],
    )
    if len(data) != len(rows) - 1 or list(data.columns) != rows[0]:
        raise ValueError("CSV parser row/header alignment could not be verified")
    converted = {}
    for index, name in enumerate(rows[0]):
        if name in text_columns:
            data[name] = pd.Series(
                [row[index] if row[index] != "" else np.nan for row in rows[1:]],
                index=data.index, dtype=object,
            )
            continue
        column = data[name]
        tokens = column.isin(_NUMERIC_MISSING)
        if not tokens.any():
            continue
        candidates = column.mask(tokens)
        numeric = pd.to_numeric(candidates, errors="coerce")
        present = candidates.notna()
        if present.any() and numeric[present].notna().all():
            data[name] = numeric
            converted[name] = int(tokens.sum())
    return tuple(tuple(row) for row in rows), data, converted


def csv_parse_summary(converted):
    """Short user-facing description of inferred numeric missing markers."""
    if not converted:
        return ""
    counts = ", ".join(f"{name}: {count}" for name, count in converted.items())
    return (
        f"Numeric missing markers converted ({counts}). "
        "Mark a column categorical to preserve those literal labels."
    )


def assert_no_protected_collision(paths, protected):
    """Refuse aliases of protected input files before any report is written."""
    sources = tuple(Path(source) for source in protected if source is not None)
    for target in map(Path, paths):
        for source in sources:
            same_path = os.path.normcase(os.path.realpath(target)) == os.path.normcase(
                os.path.realpath(source)
            )
            same_file = target.exists() and source.exists() and os.path.samefile(
                target, source
            )
            if same_path or same_file:
                raise ValueError(
                    f"output {target.name} would overwrite source data {source}; "
                    "choose a different output directory"
                )


def _numpy_scalar(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dumps_metadata(metadata):
    """Serialize metadata, including NumPy scalar values, without string coercion."""
    return json.dumps(metadata, indent=2, default=_numpy_scalar)
