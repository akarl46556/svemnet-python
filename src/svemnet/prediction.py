"""Row-aligned predictions and opt-in, recoverable CSV write-back."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .csvio import parse_csv_payload


class PredictionMixin:
    def predict(self, data):
        """Predict with frozen training encodings; missing-factor rows get NaN.

        The response column is not needed. Unseen categories raise an error;
        synthetic null columns, if fitted, must be supplied for new rows.
        """
        X, rows = self.design.transform(data)
        predictions = np.full(len(data), np.nan)
        if len(rows):
            predictions[rows] = self.fit.predict(X)
        return predictions

    def predict_source(self):
        """Predict original rows, including rows with a missing response."""
        source = self.design.source_data.copy()
        for index, name in enumerate(self.design.null_effects):
            source[name] = self.design.null_values[:, index]
        return self.predict(source)

    def save_predictions_csv(self, snapshot, *, column_name=None, text_columns=None):
        """Append only when fitted data matches the pristine, same-policy CSV.

        Pass the same ``text_columns`` used with ``snapshot.frame()`` when
        fitting. Omission uses the policy chosen at ``CSVSnapshot.load()``.
        """
        snapshot._assert_unmodified()
        expected = snapshot.frame(text_columns=text_columns)
        if not _same_frame(self.design.source_data, expected):
            raise ValueError(
                "CSV snapshot is not the data used for this fit "
                "(rows, order, values, columns or types differ)"
            )
        if column_name is None:
            kind = getattr(self, "method", "base model")
            column_name = f"Predicted {self.design.response} ({kind})"
        return snapshot.append_predictions(self.predict_source(), column_name)


def _same_frame(actual, expected):
    return (
        actual.equals(expected)
        and actual.index.identical(expected.index)
        and actual.columns.identical(expected.columns)
    )


@dataclass(frozen=True)
class CSVSnapshot:
    """An immutable byte snapshot plus a convenience DataFrame copy.

    Write-back preserves original field text and row order, creates an exact
    byte backup, and replaces the file atomically. Formatting/quoting may be
    normalized by the CSV writer; the original bytes remain in the backup.
    Export reparses immutable bytes and refuses a mutated convenience frame.
    Mark categorical columns with ``text_columns`` to retain literal NA-like
    labels and leading zeros; blank fields always mean missing.
    """

    path: Path
    data: object
    sha256: str
    _payload: bytes = field(repr=False)
    _rows: tuple[tuple[str, ...], ...] = field(repr=False)

    _text_columns: tuple[str, ...] = field(default=(), repr=False)

    @classmethod
    def load(cls, path, *, text_columns=()):
        path = Path(path).absolute()
        if path.is_symlink() or not path.is_file():
            raise ValueError("source CSV must be a regular file, not a symbolic link")
        payload = path.read_bytes()
        # Materialize once so generators use the same policy on every reparse.
        if isinstance(text_columns, str):
            raise ValueError("text_columns must be a sequence of column names")
        text_columns = tuple(text_columns)
        rows, data, _ = parse_csv_payload(payload, text_columns=text_columns)
        return cls(
            path, data, hashlib.sha256(payload).hexdigest(), payload, rows, text_columns
        )

    def frame(self, *, text_columns=None):
        """Return a fresh frame parsed from immutable bytes, with optional roles."""
        columns = self._text_columns if text_columns is None else text_columns
        return parse_csv_payload(self._payload, text_columns=columns)[1]

    def parse_report(self, *, text_columns=None):
        """Return counts of conventional tokens inferred as numeric missing cells."""
        columns = self._text_columns if text_columns is None else text_columns
        return parse_csv_payload(self._payload, text_columns=columns)[2]

    def _assert_unmodified(self):
        if not _same_frame(self.data, self.frame()):
            raise ValueError(
                "CSV snapshot data changed since loading; reload and refit before saving"
            )

    def append_predictions(self, predictions, column_name):
        values = np.asarray(predictions, dtype=float)
        if values.shape != (len(self._rows) - 1,) or np.isinf(values).any():
            raise ValueError("predictions must be a row-aligned finite/NaN vector")
        self._assert_unmodified()
        if not isinstance(column_name, str) or not column_name.strip():
            raise ValueError("prediction column name must be nonempty")
        if self.path.is_symlink() or self.path.read_bytes() != self._payload:
            raise ValueError(
                "source CSV changed since loading; reload and refit before saving"
            )
        headers = list(self._rows[0])
        name = column_name
        suffix = 2
        while name in headers:
            name = f"{column_name} [{suffix}]"
            suffix += 1
        stream = io.StringIO(newline="")
        newline = "\r\n" if b"\r\n" in self._payload else "\n"
        writer = csv.writer(stream, lineterminator=newline)
        writer.writerow([*headers, name])
        for row, value in zip(self._rows[1:], values):
            writer.writerow([*row, "" if np.isnan(value) else format(value, ".17g")])
        encoding = "utf-8-sig" if self._payload.startswith(b"\xef\xbb\xbf") else "utf-8"
        output = stream.getvalue().encode(encoding)
        backup = self.path.with_name(
            f"{self.path.name}.svemnet-{uuid.uuid4().hex[:12]}.bak"
        )
        # Exclusive creation means an existing backup is never overwritten.
        with backup.open("xb") as handle:
            handle.write(self._payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{self.path.name}.svemnet-",
                suffix=".tmp",
                dir=self.path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(output)
                handle.flush()
                os.fsync(handle.fileno())
            if os.name == "posix":
                shutil.copymode(self.path, temporary)
            if self.path.is_symlink() or self.path.read_bytes() != self._payload:
                raise ValueError(
                    "source CSV changed during export; original was not replaced"
                )
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return {"path": str(self.path), "column": name, "backup": str(backup)}
