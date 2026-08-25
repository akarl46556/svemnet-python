"""Internal bootstrap-parallelism helpers.

This module deliberately imports joblib only when parallel execution is
requested.  The default ``n_jobs=1`` path therefore creates no executor and
remains suitable for embedded Python interpreters such as JMP's.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Iterable, TypeVar


_T = TypeVar("_T")


@dataclass(frozen=True)
class _ParallelConfig:
    requested: int | None
    effective: int

    @property
    def serial(self) -> bool:
        return self.effective == 1


def _load_joblib():
    """Import joblib lazily through scikit-learn's required dependency."""

    try:
        return importlib.import_module("joblib")
    except ImportError as exc:  # pragma: no cover - broken dependency install
        raise ImportError(
            "Parallel SVEM fitting requires joblib, which is installed with "
            "scikit-learn; reinstall scikit-learn or use n_jobs=1"
        ) from exc


def _resolve_n_jobs(n_jobs: int | None, *, n_tasks: int) -> _ParallelConfig:
    """Validate sklearn-style ``n_jobs`` and cap it at the task count."""

    if n_jobs is None:
        return _ParallelConfig(requested=None, effective=1)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral):
        raise ValueError("n_jobs must be None or a nonzero integer")
    requested = int(n_jobs)
    if requested == 0:
        raise ValueError("n_jobs must be None or a nonzero integer")
    if requested == 1 or n_tasks <= 1:
        return _ParallelConfig(requested=requested, effective=1)

    joblib = _load_joblib()
    effective = int(joblib.effective_n_jobs(requested))
    return _ParallelConfig(
        requested=requested,
        effective=max(1, min(int(n_tasks), effective)),
    )


def _run_parallel(
    worker: Callable[..., _T],
    tasks: Iterable[tuple[Any, ...]],
    *,
    n_jobs: int,
) -> list[_T]:
    """Execute positional worker calls in input order with process preference."""

    joblib = _load_joblib()
    try:
        return joblib.Parallel(n_jobs=n_jobs, prefer="processes")(
            joblib.delayed(worker)(*args) for args in tasks
        )
    except Exception as exc:
        raise RuntimeError(
            "Parallel SVEM bootstrap fitting failed. Retry with n_jobs=1 "
            "when running in JMP or another embedded Python environment."
        ) from exc


__all__ = ["_ParallelConfig", "_resolve_n_jobs", "_run_parallel"]
