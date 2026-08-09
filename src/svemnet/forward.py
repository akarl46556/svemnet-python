"""Forward-selection engines: deterministic IC selection and SVEM-forward.

Two Gaussian fitters over a caller-supplied numeric model matrix ``X``
(shape ``(n, p)``, no intercept column — the structural intercept is always
included internally):

* :func:`forward_select` — deterministic greedy forward selection over
  whole-effect groups of columns, minimizing a least-squares information
  criterion (AICc, AIC, or BIC), with a stop-on-no-improvement rule.
* :func:`fit_svem_forward` — SVEM ensemble whose per-bootstrap base learner
  is greedy forward selection on training-weighted least squares. The forward
  path plays the lambda-path role: every path point is scored on the
  fractional-random-weight validation copy with the same wSSE/wAIC/wBIC
  kernels as :func:`svemnet.core.fit_svem`, and winning coefficient vectors
  are averaged across bootstrap members.

Whole-effect groups: multi-column terms (for example a factor's contrast
block) enter and leave the model together. Pass ``groups`` as an ordered
mapping of group name to column indices; by default every column is its own
group. Candidate ties are broken by group order.

The numerical kernels (QR-update candidate evaluation with exact-SVD
fallbacks, AICc/BIC guards, RSS flooring) follow the reference
implementation used for the SVEMnet R-package port; see the R package
SVEMnet for the corresponding R functions ``forward_aicc()`` and
``svem_forward()``.
"""

from __future__ import annotations

import math
import warnings as _warnings
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .core import (
    _FLOAT_EPS,
    _SUPPORTED_OBJECTIVES,
    _SUPPORTED_WEIGHT_SCHEMES,
    _has_variation,
    _intercept_only_coefficients,
    _linear_calibration,
    _validate_positive_int,
    _validate_weight_uniforms,
    _validate_xy,
    _feature_names,
    kish_effective_n,
    make_svem_weights,
    predict_svem,
    weighted_ic_scores,
)

_SUPPORTED_CRITERIA = ("AICc", "AIC", "BIC")

## ---------------------------------------------------------------------------
## Subset evaluation kernels (exact SVD reference + QR-update fast path)
## ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SubsetFit:
    coefficients: np.ndarray
    residual_sum_of_squares: float
    rank: int
    n_rows: int
    n_columns: int

    @property
    def rank_deficient(self) -> bool:
        return self.rank < self.n_columns


def _evaluate_subset(x: np.ndarray, y: np.ndarray, indices: Sequence[int]) -> _SubsetFit:
    subset = x[:, tuple(indices)]
    # Column equilibration before the SVD solve: lstsq's rank cutoff is
    # relative to the LARGEST singular value, so one badly scaled column
    # (e.g. a raw-timestamp predictor) would truncate the independent
    # directions of well-scaled columns and return a degenerate minimum-norm
    # solution. Unit-norm scaling makes the rank decision column-relative,
    # consistent with the projected-norm gates used on the greedy path.
    scale = np.linalg.norm(subset, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    scaled_coefficients, _, rank, _ = np.linalg.lstsq(
        subset / scale, y, rcond=None
    )
    coefficients = scaled_coefficients / scale
    residuals = y - subset @ coefficients
    return _SubsetFit(
        coefficients=coefficients,
        residual_sum_of_squares=float(np.dot(residuals, residuals)),
        rank=int(rank),
        n_rows=int(subset.shape[0]),
        n_columns=int(subset.shape[1]),
    )


# Relative projected-column-norm ratio above which a candidate is decisively
# full rank. Anything at or below the gate is re-evaluated with exact lstsq so
# rank/skip decisions cannot depend on the fast path.
_QR_INDEPENDENCE_GATE = 1.0e-8


@dataclass(frozen=True)
class _CandidateEvaluation:
    """RSS/rank facts a selection step needs about one candidate model."""

    residual_sum_of_squares: float
    rank: int
    n_rows: int
    n_columns: int

    @property
    def rank_deficient(self) -> bool:
        return self.rank < self.n_columns


class _QRStepEvaluator:
    """Evaluate candidate additions against a fixed current path via QR.

    One thin QR of the current path columns is computed per accepted step.
    Each candidate group is then scored by projecting its columns off the
    current orthonormal basis: the RSS decrease is the squared norm of the
    projected-residual components. Candidates whose projected columns come
    close to linear dependence (relative to their original norms), and
    candidates whose RSS update is cancellation-dominated, are routed to the
    exact SVD reference so rank and value decisions cannot diverge from it.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, path_indices: Sequence[int]):
        self._x = x
        self._y = y
        self._path = tuple(path_indices)
        subset = x[:, self._path]
        q, _ = np.linalg.qr(subset, mode="reduced")
        self._q = q
        self._residual = y - q @ (q.T @ y)
        self._rss_current = float(np.dot(self._residual, self._residual))
        # Below this RSS scale (relative to the response energy) log-RSS
        # comparisons are dominated by rounding, so candidates are evaluated
        # exactly instead.
        self._dust = 1.0e-12 * max(float(np.dot(y, y)), np.finfo(float).tiny)

    def evaluate(self, candidate_indices: Sequence[int]) -> _CandidateEvaluation:
        trial = (*self._path, *candidate_indices)
        n_rows = self._x.shape[0]
        n_columns = len(trial)
        if n_columns > n_rows:
            return _exact_candidate(self._x, self._y, trial)
        if self._rss_current <= self._dust:
            # The current model is already a near-perfect fit; log-RSS
            # comparisons among rounding-scale residuals are meaningless, so
            # defer to the exact reference evaluation.
            return _exact_candidate(self._x, self._y, trial)
        candidate = self._x[:, tuple(candidate_indices)]
        if candidate.shape[1] == 1:
            # Single-column candidates (the common case) avoid the QR call.
            column = candidate[:, 0]
            projected_column = column - self._q @ (self._q.T @ column)
            projected_norm_sq = float(np.dot(projected_column, projected_column))
            original_norm = float(np.linalg.norm(column))
            scale = max(original_norm, np.finfo(float).tiny)
            projected_norm = projected_norm_sq**0.5
            if projected_norm / scale <= _QR_INDEPENDENCE_GATE:
                return _exact_candidate(self._x, self._y, trial)
            gain = float(np.dot(projected_column, self._residual)) / projected_norm
            rss = self._rss_current - gain * gain
        else:
            original_norms = np.linalg.norm(candidate, axis=0)
            projected = candidate - self._q @ (self._q.T @ candidate)
            q_c, r_c = np.linalg.qr(projected, mode="reduced")
            diagonal = np.abs(np.diag(r_c))
            scale = np.maximum(original_norms, np.finfo(float).tiny)
            if np.min(diagonal / scale) <= _QR_INDEPENDENCE_GATE:
                return _exact_candidate(self._x, self._y, trial)
            gains = q_c.T @ self._residual
            rss = self._rss_current - float(np.dot(gains, gains))
        if rss < self._rss_current * 1.0e-12 or rss <= self._dust:
            # Near-perfect candidate fit: the RSS update is
            # cancellation-dominated and the log-RSS criterion is extremely
            # sensitive there, so defer to the exact reference evaluation.
            return _exact_candidate(self._x, self._y, trial)
        return _CandidateEvaluation(
            residual_sum_of_squares=rss,
            rank=n_columns,
            n_rows=n_rows,
            n_columns=n_columns,
        )


def _exact_candidate(
    x: np.ndarray,
    y: np.ndarray,
    indices: Sequence[int],
) -> _CandidateEvaluation:
    fit = _evaluate_subset(x, y, indices)
    return _CandidateEvaluation(
        residual_sum_of_squares=fit.residual_sum_of_squares,
        rank=fit.rank,
        n_rows=fit.n_rows,
        n_columns=fit.n_columns,
    )


## ---------------------------------------------------------------------------
## Information criteria (Gaussian least squares, up to the additive constant)
## ---------------------------------------------------------------------------


def _criterion_value(
    *,
    criterion: str,
    n: int,
    rss: float,
    regression_parameter_count: int,
) -> float | None:
    """AICc/AIC/BIC value, or None when undefined for this (n, p).

    ``regression_parameter_count`` counts the fitted regression coefficients
    including the structural intercept; the criterion parameter count
    ``K = regression_parameter_count + 1`` adds the residual variance.
    """
    K = regression_parameter_count + 1
    rss_for_log = max(float(rss), float(np.finfo(float).tiny))
    if criterion == "AICc":
        denominator = n - K - 1
        if denominator <= 0:
            return None
        aic = n * math.log(rss_for_log / n) + 2.0 * K
        return aic + (2.0 * K * (K + 1.0)) / denominator
    if n - regression_parameter_count <= 0:
        return None
    if criterion == "AIC":
        return n * math.log(rss_for_log / n) + 2.0 * K
    if criterion == "BIC":
        value = n * math.log(rss_for_log / n) + K * math.log(n)
        return value if math.isfinite(value) else None
    raise ValueError(f"criterion must be one of {list(_SUPPORTED_CRITERIA)}")


def _is_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


## ---------------------------------------------------------------------------
## Whole-effect groups
## ---------------------------------------------------------------------------


def _group_index(value: object, name: str) -> int:
    """Coerce one group column index to int, rejecting bools and floats."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"group {name!r} column indices must be integers, not booleans"
        )
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise ValueError(
        f"group {name!r} column index {value!r} is not an integer"
    )


def _normalize_groups(
    groups: Mapping[str, Sequence[int]] | None,
    p: int,
    feature_names: tuple[str, ...],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Validate groups over X columns (0-based, intercept excluded).

    Returns ``((name, column_indices), ...)`` in candidate order. Columns not
    covered by any group can never enter the model. Groups must be disjoint,
    non-empty, and uniquely named (selection frequencies are keyed by name).
    An empty group set (``p == 0`` or an explicit empty mapping) yields an
    intercept-only fit.
    """
    if groups is None:
        out = [(feature_names[j], (j,)) for j in range(p)]
    else:
        out = []
        seen: set[int] = set()
        for name, cols in groups.items():
            idx = tuple(_group_index(c, name) for c in cols)
            if not idx:
                raise ValueError(f"group {name!r} has no columns")
            for c in idx:
                if c < 0 or c >= p:
                    raise ValueError(
                        f"group {name!r} column index {c} is outside [0, {p})"
                    )
                if c in seen:
                    raise ValueError(
                        f"column {c} appears in more than one group"
                    )
                seen.add(c)
            out.append((str(name), idx))
    seen_names: set[str] = set()
    for name, _ in out:
        if name in seen_names:
            raise ValueError(
                f"duplicate group name {name!r}; group (and feature) names "
                "must be unique because selection results are keyed by name"
            )
        seen_names.add(name)
    return tuple(out)


## ---------------------------------------------------------------------------
## Deterministic forward selection by information criterion
## ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForwardICResult:
    """Result of :func:`forward_select`.

    ``coefficients`` holds the intercept in element 0; elements 1..p align to
    the columns of the caller-supplied ``X`` (zeros for unselected columns).
    ``selection_path`` records one dict per accepted step with keys ``step``,
    ``term``, ``k`` (fitted coefficients including the intercept), ``rss``,
    ``criterion_value``, and ``improvement``.
    """

    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    criterion: str
    criterion_value: float | None
    selected_terms: tuple[str, ...]
    selection_path: tuple[dict[str, object], ...]
    residual_sum_of_squares: float
    n_obs: int
    warnings: tuple[str, ...]

    def predict(self, X: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Predict from the selected model."""
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2 or X_arr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X must be two-dimensional with {len(self.feature_names)} columns"
            )
        return X_arr @ self.coefficients[1:] + self.coefficients[0]


def forward_select(
    X: Sequence[Sequence[float]] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    criterion: str = "AICc",
    groups: Mapping[str, Sequence[int]] | None = None,
    feature_names: Sequence[str] | None = None,
    improvement_tolerance: float = 1.0e-12,
) -> ForwardICResult:
    """Deterministic greedy forward selection minimizing AICc, AIC, or BIC.

    Starting from the intercept-only model, each step evaluates every
    remaining whole-effect group, and adds the group whose augmented model
    minimizes the criterion. Selection stops when no candidate improves the
    criterion by more than ``improvement_tolerance``, when every remaining
    candidate is rank deficient given the current model, or when the
    criterion is undefined for every candidate (AICc requires
    ``n - K - 1 > 0``). Rank-deficient candidates are skipped; ties are
    broken by group order. The final selected model is refit exactly.
    """
    if criterion not in _SUPPORTED_CRITERIA:
        raise ValueError(f"criterion must be one of {list(_SUPPORTED_CRITERIA)}")
    if improvement_tolerance < 0 or not math.isfinite(improvement_tolerance):
        raise ValueError("improvement_tolerance must be nonnegative and finite")
    X_arr, y_arr = _validate_xy(X, y)
    n, p = X_arr.shape
    names = _feature_names(feature_names, p)
    group_list = _normalize_groups(groups, p, names)

    x = np.column_stack([np.ones(n), X_arr])
    # groups shifted by 1 for the structural intercept column
    remaining: list[tuple[str, tuple[int, ...]]] = [
        (name, tuple(c + 1 for c in cols)) for name, cols in group_list
    ]
    path_indices: list[int] = [0]
    selected_terms: list[str] = []
    selection_path: list[dict[str, object]] = []
    fit_warnings: list[str] = []

    current = _exact_candidate(x, y_arr, path_indices)
    current_rss = current.residual_sum_of_squares
    current_value = _criterion_value(
        criterion=criterion,
        n=n,
        rss=current_rss,
        regression_parameter_count=1,
    )
    if current_value is None:
        msg = (
            f"{criterion} is undefined for the intercept-only model "
            f"(n = {n} is too small); returning the intercept-only fit."
        )
        fit_warnings.append(msg)
        _warnings.warn(msg, stacklevel=2)

    # Below this RSS scale, criterion differences are rounding noise and
    # selection stops. The anchor is the intercept-only RSS (the signal
    # scale after the mean is absorbed), with a response-energy floor for
    # the degenerate constant-response case; anchoring to raw response
    # energy would silently truncate selection for large-mean responses.
    tiny = float(np.finfo(float).tiny)
    eps = float(np.finfo(float).eps)
    dust = max(
        1.0e-12 * max(current_rss, tiny),
        1.0e-12 * eps * max(float(np.dot(y_arr, y_arr)), tiny),
    )

    step = 1
    while remaining and _is_finite(current_value) and current_rss > dust:
        evaluator = _QRStepEvaluator(x, y_arr, path_indices)
        best_index: int | None = None
        best_value: float | None = None
        best_rss: float | None = None
        for gi, (name, cols) in enumerate(remaining):
            candidate = evaluator.evaluate(cols)
            if candidate.rank_deficient:
                fit_warnings.append(
                    f"Skipped candidate {name!r} at step {step}: rank deficient"
                )
                continue
            value = _criterion_value(
                criterion=criterion,
                n=n,
                rss=candidate.residual_sum_of_squares,
                regression_parameter_count=len(path_indices) + len(cols),
            )
            if not _is_finite(value):
                fit_warnings.append(
                    f"Skipped candidate {name!r} at step {step}: "
                    f"{criterion} undefined or not finite"
                )
                continue
            if best_value is None or value < best_value:
                best_index = gi
                best_value = value
                best_rss = candidate.residual_sum_of_squares
        if best_index is None or best_value is None:
            break
        improvement = float(current_value - best_value)
        if improvement <= improvement_tolerance:
            break
        name, cols = remaining.pop(best_index)
        path_indices.extend(cols)
        selected_terms.append(name)
        current_value = best_value
        if best_rss is not None:
            current_rss = best_rss
        selection_path.append(
            {
                "step": step,
                "term": name,
                "k": len(path_indices),
                "rss": float(best_rss if best_rss is not None else np.nan),
                "criterion_value": best_value,
                "improvement": improvement,
            }
        )
        step += 1

    if remaining and _is_finite(current_value) and current_rss <= dust:
        fit_warnings.append(
            "Selection stopped: residual sum of squares "
            f"{current_rss:.3e} is at the numerical noise floor "
            f"({dust:.3e}); remaining candidate terms were not evaluated."
        )

    # Final diagnostics always come from one exact SVD refit.
    final_fit = _evaluate_subset(x, y_arr, path_indices)
    if final_fit.rank_deficient:
        fit_warnings.append(
            f"Final selected model is rank deficient: rank {final_fit.rank} "
            f"< {final_fit.n_columns}"
        )
    final_value = _criterion_value(
        criterion=criterion,
        n=n,
        rss=final_fit.residual_sum_of_squares,
        regression_parameter_count=len(path_indices),
    )
    coefficients = np.zeros(p + 1, dtype=float)
    coefficients[list(path_indices)] = final_fit.coefficients

    return ForwardICResult(
        coefficients=coefficients,
        feature_names=names,
        criterion=criterion,
        criterion_value=final_value,
        selected_terms=tuple(selected_terms),
        selection_path=tuple(selection_path),
        residual_sum_of_squares=final_fit.residual_sum_of_squares,
        n_obs=n,
        warnings=tuple(fit_warnings),
    )


## ---------------------------------------------------------------------------
## SVEM ensemble with forward-selection base learners
## ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SVEMForwardResult:
    """Result from :func:`fit_svem_forward`.

    Field conventions match :class:`svemnet.core.SVEMGaussianResult` where
    they overlap, so :func:`svemnet.core.predict_svem` (and therefore
    ``.predict(...)`` with ``se_fit``/``interval``) works on either result.
    ``coefficients[0]`` is the intercept; ``coefficients[1:]`` align to the
    columns of the caller-supplied ``X``.
    """

    coefficients: np.ndarray
    coef_matrix: np.ndarray
    predictions: np.ndarray
    diagnostics: Mapping[str, object]
    feature_names: tuple[str, ...]
    raw_coefficients: np.ndarray
    raw_predictions: np.ndarray
    debiased_coefficients: np.ndarray | None
    debiased_predictions: np.ndarray | None
    debias_params: tuple[float, float] | None
    debias_applied: bool
    objective: str
    weight_scheme: str
    seed: int | None
    nBoot_requested: int
    nBoot_used: int
    selected_k: np.ndarray
    n_eff_raw: np.ndarray
    n_eff_adm: np.ndarray
    fallback_mask: np.ndarray
    selection_frequencies: Mapping[str, float]
    pi_sigma: float | None = None
    pi_df: float | None = None

    def predict(
        self,
        X: Sequence[Sequence[float]] | np.ndarray,
        *,
        debias: bool | None = None,
        se_fit: bool = False,
        interval: bool = False,
        level: float = 0.95,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Predict; see :func:`svemnet.core.predict_svem`."""
        return predict_svem(
            self, X, debias=debias, se_fit=se_fit, interval=interval, level=level
        )


def _grow_forward_path(
    x_weighted: np.ndarray,
    y_weighted: np.ndarray,
    *,
    groups: tuple[tuple[str, tuple[int, ...]], ...],
    k_slope_ceiling: float | None,
) -> tuple[list[tuple[int, ...]], list[tuple[str, ...]], bool]:
    """Grow one training-weighted greedy forward path.

    Returns the nested path points (column-index tuples into the
    intercept-augmented matrix; point 0 = intercept-only), the selected group
    names at each point, and whether growth was truncated by the wAIC/wBIC
    admissibility ceiling. ``k_slope_ceiling`` must be None for wSSE.
    """
    path_indices: list[int] = [0]
    selected: list[str] = []
    remaining = list(groups)
    path_points: list[tuple[int, ...]] = [tuple(path_indices)]
    path_point_groups: list[tuple[str, ...]] = [tuple(selected)]
    admissibility_truncated = False

    while remaining:
        if k_slope_ceiling is not None and (len(path_indices) - 1) >= k_slope_ceiling:
            # Every further path point is inadmissible for wAIC/wBIC because
            # k_slope only grows; wSSE must never take this shortcut.
            admissibility_truncated = True
            break
        evaluator = _QRStepEvaluator(x_weighted, y_weighted, path_indices)
        best_index: int | None = None
        best_rss: float | None = None
        for gi, (_, cols) in enumerate(remaining):
            candidate = evaluator.evaluate(cols)
            if candidate.rank_deficient:
                continue
            candidate_rss = float(candidate.residual_sum_of_squares)
            if not math.isfinite(candidate_rss):
                continue
            if best_rss is None or candidate_rss < best_rss:
                best_index = gi
                best_rss = candidate_rss
        if best_index is None:
            break
        name, cols = remaining.pop(best_index)
        path_indices.extend(cols)
        selected.append(name)
        path_points.append(tuple(path_indices))
        path_point_groups.append(tuple(selected))
    return path_points, path_point_groups, admissibility_truncated


@dataclass(frozen=True)
class _BootstrapFit:
    coefficients: np.ndarray
    selected_terms: tuple[str, ...]
    selected_k: int
    fallback: bool
    n_eff_raw: float
    n_eff_adm: float
    path_length: int
    admissibility_truncated: bool


def _fit_one_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    *,
    w_train: np.ndarray,
    w_valid: np.ndarray,
    groups: tuple[tuple[str, tuple[int, ...]], ...],
    objective: str,
) -> _BootstrapFit:
    """Fit one weighted forward path and score it on validation weights."""
    n, n_columns = x.shape
    n_eff_raw, n_eff_adm = kish_effective_n(w_valid, n=n)

    sqrt_w = np.sqrt(w_train)
    x_weighted = x * sqrt_w[:, None]
    y_weighted = y * sqrt_w

    path_points, path_point_groups, admissibility_truncated = _grow_forward_path(
        x_weighted,
        y_weighted,
        groups=groups,
        k_slope_ceiling=(float(n_eff_adm) if objective in ("wAIC", "wBIC") else None),
    )

    # Score path points: training-weighted exact refits; validation weights
    # applied to residuals on the ORIGINAL unweighted rows.
    point_coefficients: list[np.ndarray] = []
    sse_w = np.empty(len(path_points), dtype=float)
    k_values = np.empty(len(path_points), dtype=int)
    for point_index, indices in enumerate(path_points):
        fit = _evaluate_subset(x_weighted, y_weighted, indices)
        point_coefficients.append(fit.coefficients)
        residual = x[:, indices] @ fit.coefficients - y
        sse_w[point_index] = float(np.sum(w_valid * residual * residual))
        k_values[point_index] = len(indices)
    sse_w = np.where(np.isfinite(sse_w), sse_w, np.inf)
    scores = weighted_ic_scores(
        sse_w,
        k_values,
        n_like=float(np.sum(w_valid)),
        n_eff_adm=float(n_eff_adm),
        objective=objective,
    )

    fallback = not np.any(np.isfinite(scores))
    if not fallback:
        winner = int(np.nanargmin(scores))
        coefficients = np.zeros(n_columns, dtype=float)
        coefficients[list(path_points[winner])] = point_coefficients[winner]
        if not np.all(np.isfinite(coefficients)):
            fallback = True
    if fallback:
        coefficients = _intercept_only_coefficients(y, w_train, n_columns - 1)
        selected_terms: tuple[str, ...] = ()
        selected_k = 1
    else:
        selected_terms = path_point_groups[winner]
        selected_k = int(k_values[winner])

    return _BootstrapFit(
        coefficients=coefficients,
        selected_terms=selected_terms,
        selected_k=selected_k,
        fallback=bool(fallback),
        n_eff_raw=float(n_eff_raw),
        n_eff_adm=float(n_eff_adm),
        path_length=len(path_points),
        admissibility_truncated=admissibility_truncated,
    )


def fit_svem_forward(
    X: Sequence[Sequence[float]] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    nBoot: int = 100,
    objective: str = "wAIC",
    weight_scheme: str = "SVEM",
    seed: int | None = None,
    groups: Mapping[str, Sequence[int]] | None = None,
    feature_names: Sequence[str] | None = None,
    debias: bool = False,
    weight_uniforms: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> SVEMForwardResult:
    """Fit a Gaussian SVEM ensemble with forward-selection base learners.

    Per bootstrap, fractional-random-weight train/validation copies are drawn
    exactly as in :func:`svemnet.core.fit_svem`; the forward path is grown on
    training-weighted least squares by greedy training-RSS reduction over
    whole-effect groups; each path point is refit on the weighted rows and
    scored by validation-weighted residuals computed on the original
    (unweighted) rows through the shared wSSE/wAIC/wBIC kernel; the first
    path-point minimum wins. For ``"wAIC"``/``"wBIC"``, path growth stops at
    the Kish admissible-effective-sample-size ceiling (all further points
    would be inadmissible); ``"wSSE"`` paths are never truncated. Bootstraps
    with no admissible finite path point fall back to a weighted
    intercept-only model.

    Parameters mirror :func:`svemnet.core.fit_svem` where they overlap.
    ``groups`` optionally maps group names to column indices of ``X`` so that
    multi-column terms enter together; the default treats every column as its
    own group. ``weight_uniforms`` is the deterministic test/parity hook: a
    ``(nBoot, n)`` matrix of shared uniforms used to construct the FRW
    weights.
    """
    X_arr, y_arr = _validate_xy(X, y)
    n, p = X_arr.shape
    nBoot_int = _validate_positive_int(nBoot, "nBoot")
    if objective not in _SUPPORTED_OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(_SUPPORTED_OBJECTIVES)}")
    if weight_scheme not in _SUPPORTED_WEIGHT_SCHEMES:
        raise ValueError(
            f"weight_scheme must be one of {sorted(_SUPPORTED_WEIGHT_SCHEMES)}"
        )
    names = _feature_names(feature_names, p)
    group_list = _normalize_groups(groups, p, names)
    shifted_groups = tuple(
        (name, tuple(c + 1 for c in cols)) for name, cols in group_list
    )
    uniform_matrix = _validate_weight_uniforms(
        weight_uniforms, nBoot=nBoot_int, n=n
    )

    x = np.column_stack([np.ones(n), X_arr])
    rng = np.random.default_rng(seed)

    boot_fits: list[_BootstrapFit] = []
    valid_sse_values: list[float] = []
    for boot_index in range(nBoot_int):
        uniforms = None if uniform_matrix is None else uniform_matrix[boot_index]
        w_train, w_valid = make_svem_weights(
            n,
            rng,
            scheme=weight_scheme,
            uniforms=uniforms,
        )
        boot_fits.append(
            _fit_one_bootstrap(
                x,
                y_arr,
                w_train=w_train,
                w_valid=w_valid,
                groups=shifted_groups,
                objective=objective,
            )
        )
        # Validation-weighted SSE for the prediction-interval scalars,
        # computed from quantities already in scope (no extra RNG draws).
        coef_b = boot_fits[-1].coefficients
        if np.all(np.isfinite(coef_b)):
            resid_b = y_arr - (X_arr @ coef_b[1:] + coef_b[0])
            valid_sse_values.append(float(np.sum(w_valid * resid_b * resid_b)))
        else:
            valid_sse_values.append(float("nan"))

    coef_matrix = np.vstack([fit.coefficients for fit in boot_fits])
    finite_rows = np.all(np.isfinite(coef_matrix), axis=1)
    if not np.any(finite_rows):
        raise RuntimeError(
            "All bootstrap iterations failed to produce finite coefficients"
        )
    used_fits = [fit for fit, keep in zip(boot_fits, finite_rows) if keep]
    coef_matrix = coef_matrix[finite_rows]
    raw_coefficients = np.mean(coef_matrix, axis=0)
    raw_predictions = X_arr @ raw_coefficients[1:] + raw_coefficients[0]

    n_used = len(used_fits)
    selection_counts: dict[str, int] = {name: 0 for name, _ in group_list}
    for fit in used_fits:
        for term in fit.selected_terms:
            selection_counts[term] += 1
    selection_frequencies = {
        name: count / n_used for name, count in selection_counts.items()
    }

    debias_params: tuple[float, float] | None = None
    debiased_coefficients: np.ndarray | None = None
    debiased_predictions: np.ndarray | None = None
    if coef_matrix.shape[0] >= 10 and _has_variation(raw_predictions):
        a, b = _linear_calibration(raw_predictions, y_arr)
        if np.isfinite(a) and np.isfinite(b):
            debias_params = (float(a), float(b))
            debiased_coefficients = raw_coefficients.copy()
            debiased_coefficients[0] = a + b * debiased_coefficients[0]
            debiased_coefficients[1:] = b * debiased_coefficients[1:]
            debiased_predictions = a + b * raw_predictions

    use_debiased = bool(debias and debiased_coefficients is not None)
    coefficients = (
        debiased_coefficients.copy() if use_debiased else raw_coefficients.copy()
    )
    predictions = (
        debiased_predictions.copy() if use_debiased else raw_predictions.copy()
    )

    selected_k = np.asarray([fit.selected_k for fit in used_fits], dtype=int)
    n_eff_raw_arr = np.asarray([fit.n_eff_raw for fit in used_fits], dtype=float)
    n_eff_adm_arr = np.asarray([fit.n_eff_adm for fit in used_fits], dtype=float)
    fallback_arr = np.asarray([fit.fallback for fit in used_fits], dtype=bool)

    valid_sse_arr = np.asarray(valid_sse_values, dtype=float)[finite_rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        member_df = np.maximum(n - selected_k.astype(float), 1.0)
        member_sig2 = valid_sse_arr / member_df
    finite_sig2 = member_sig2[np.isfinite(member_sig2)]
    pi_sigma = float(np.sqrt(np.median(finite_sig2))) if finite_sig2.size else None
    pi_df = (
        float(max(n - float(np.median(selected_k)), 1.0)) if selected_k.size else None
    )

    diagnostics: dict[str, object] = {
        "k_summary": {
            "k_median": float(np.median(selected_k)),
            "k_iqr": float(
                np.percentile(selected_k, 75) - np.percentile(selected_k, 25)
            ),
            "k_min": int(np.min(selected_k)),
            "k_max": int(np.max(selected_k)),
        },
        "fallback_rate": float(np.mean(fallback_arr)),
        "n_eff_summary": {
            "Min": float(np.min(n_eff_raw_arr)),
            "Median": float(np.median(n_eff_raw_arr)),
            "Mean": float(np.mean(n_eff_raw_arr)),
            "Max": float(np.max(n_eff_raw_arr)),
        },
        "path_length_max": int(max(fit.path_length for fit in used_fits)),
        "admissibility_truncation_rate": float(
            np.mean([fit.admissibility_truncated for fit in used_fits])
        ),
        "selection_frequencies": {
            name: float(freq) for name, freq in sorted(selection_frequencies.items())
        },
        "reproducibility": {
            "deterministic_given_seed": seed is not None,
            "rng": "numpy.random.default_rng",
            "seed": seed,
            "weight_uniforms_supplied": uniform_matrix is not None,
            "serial": True,
            "base_learner": "forward_selection_weighted_least_squares",
            "training_weight_application": "row_scaling_sqrt_w_train",
            "validation_scoring": (
                "w_valid on residuals from ORIGINAL unweighted rows via "
                "weighted_ic_scores"
            ),
            "path_step_rule": "greedy_training_rss_reduction_whole_effect_groups",
            "path_selection": "first_path_point_minimum",
            "candidate_tie_breaking": "group_order",
            "objective": objective,
            "weight_scheme": weight_scheme,
            "nBoot": nBoot_int,
            "family": "gaussian",
        },
        "nBoot_requested": int(nBoot_int),
        "nBoot_used": int(n_used),
        "debias_eligible": debiased_coefficients is not None,
        "debias_applied": use_debiased,
    }

    return SVEMForwardResult(
        coefficients=coefficients,
        coef_matrix=coef_matrix,
        predictions=predictions,
        diagnostics=diagnostics,
        feature_names=names,
        raw_coefficients=raw_coefficients,
        raw_predictions=raw_predictions,
        debiased_coefficients=debiased_coefficients,
        debiased_predictions=debiased_predictions,
        debias_params=debias_params,
        debias_applied=use_debiased,
        objective=objective,
        weight_scheme=weight_scheme,
        seed=seed,
        nBoot_requested=nBoot_int,
        nBoot_used=int(n_used),
        selected_k=selected_k,
        n_eff_raw=n_eff_raw_arr,
        n_eff_adm=n_eff_adm_arr,
        fallback_mask=fallback_arr,
        selection_frequencies=selection_frequencies,
        pi_sigma=pi_sigma,
        pi_df=pi_df,
    )


__all__ = [
    "ForwardICResult",
    "SVEMForwardResult",
    "forward_select",
    "fit_svem_forward",
]
