"""Row-aligned predictions and opt-in, recoverable CSV write-back."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


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

    def save_predictions_csv(self, snapshot, *, column_name=None):
        """Append predictions only to the unchanged CSV snapshot used for fitting."""
        if not snapshot.data.equals(self.design.source_data):
            raise ValueError("CSV snapshot is not the data used for this fit")
        if column_name is None:
            kind = getattr(self, "method", "base model")
            column_name = f"Predicted {self.design.response} ({kind})"
        return snapshot.append_predictions(self.predict_source(), column_name)


@dataclass(frozen=True)
class CSVSnapshot:
    """An immutable byte snapshot plus the DataFrame parsed from those bytes.

    Write-back preserves original field text and row order, creates an exact
    byte backup, and replaces the file atomically. Formatting/quoting may be
    normalized by the CSV writer; the original bytes remain in the backup.
    """

    path: Path
    data: object
    sha256: str
    _payload: bytes = field(repr=False)
    _rows: tuple[tuple[str, ...], ...] = field(repr=False)

    @classmethod
    def load(cls, path):
        import pandas as pd

        path = Path(path).absolute()
        if path.is_symlink() or not path.is_file():
            raise ValueError("source CSV must be a regular file, not a symbolic link")
        payload = path.read_bytes()
        text = payload.decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
        if not rows or not rows[0] or any(not name for name in rows[0]):
            raise ValueError("CSV requires nonempty column names")
        if len(rows[0]) != len(set(rows[0])):
            raise ValueError("duplicate CSV column names are not supported")
        width = len(rows[0])
        rows = [rows[0], *[(row if row else [""] * width) for row in rows[1:]]]
        if any(len(row) != width for row in rows):
            raise ValueError("CSV rows must all have the same number of fields")
        data = pd.read_csv(
            io.BytesIO(payload), encoding="utf-8-sig", skip_blank_lines=False
        )
        if len(data) != len(rows) - 1 or list(data.columns) != rows[0]:
            raise ValueError("CSV parser row/header alignment could not be verified")
        return cls(
            path,
            data,
            hashlib.sha256(payload).hexdigest(),
            payload,
            tuple(tuple(row) for row in rows),
        )

    def append_predictions(self, predictions, column_name):
        values = np.asarray(predictions, dtype=float)
        if values.shape != (len(self.data),) or np.isinf(values).any():
            raise ValueError("predictions must be a row-aligned finite/NaN vector")
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
            if self.path.is_symlink() or self.path.read_bytes() != self._payload:
                raise ValueError(
                    "source CSV changed during export; original was not replaced"
                )
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return {"path": str(self.path), "column": name, "backup": str(backup)}
