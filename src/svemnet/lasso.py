"""Single Gaussian lasso/elastic-net models selected by repeated K-fold CV.

This is a scikit-learn counterpart of the non-relaxed R SVEMnet CV workflow,
not a claim of bitwise glmnet parity. Predictor standardization is learned
inside each training fold. All mixing parameters share the same splits.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import enet_path
from sklearn.model_selection import RepeatedKFold

from . import _parallel
from .core import (
    _feature_names,
    _validate_alpha,
    _validate_positive_int,
    _validate_X_new,
    _validate_xy,
)


def _path(X, y, alpha, lambdas, solver_tol, max_iter):
    x_mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    y_mean = float(y.mean())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        _, slopes, _ = enet_path(
            np.asfortranarray((X - x_mean) / scale),
            y - y_mean,
            l1_ratio=float(alpha),
            alphas=lambdas,
            tol=solver_tol,
            max_iter=max_iter,
        )
    slopes = slopes / scale[:, None]
    coefficients = np.vstack((y_mean - x_mean @ slopes, slopes))
    if not np.isfinite(coefficients).all():
        raise RuntimeError("CV path produced non-finite coefficients")
    return coefficients, tuple(str(item.message) for item in caught)


def _score_fold(X, y, train, valid, alphas, grids, solver_tol, max_iter):
    losses = []
    messages = []
    for alpha, grid in zip(alphas, grids):
        coefficients, caught = _path(
            X[train], y[train], alpha, grid, solver_tol, max_iter
        )
        predictions = X[valid] @ coefficients[1:] + coefficients[0]
        losses.append(np.mean((y[valid, None] - predictions) ** 2, axis=0))
        messages.extend(caught)
    return np.asarray(losses), tuple(messages)


@dataclass(frozen=True)
class CVLassoResult:
    coefficients: np.ndarray
    predictions: np.ndarray
    feature_names: tuple[str, ...]
    standardized_coefficients: np.ndarray
    selected_alpha: float
    selected_lambda: float
    alphas: np.ndarray
    lambda_grid: np.ndarray
    cv_mean_mse: np.ndarray
    cv_se_mse: np.ndarray
    fold_mse: np.ndarray
    fold_ids: np.ndarray
    nfolds: int
    repeats: int
    choose_rule: str
    seed: int | None
    workers: int
    warnings: tuple[str, ...]

    def predict(self, X) -> np.ndarray:
        matrix = _validate_X_new(X, len(self.feature_names))
        return matrix @ self.coefficients[1:] + self.coefficients[0]


def fit_lasso_cv(
    X,
    y,
    *,
    alphas: Sequence[float] = (1.0,),
    nfolds: int = 10,
    repeats: int = 5,
    choose_rule: str = "min",
    seed: int | None = 12345,
    nlambda: int = 100,
    lambda_min_ratio: float = 1e-4,
    solver_tol: float = 1e-7,
    max_iter: int = 100000,
    n_jobs: int | None = 1,
    feature_names: Sequence[str] | None = None,
) -> CVLassoResult:
    """Fit one penalized model, defaulting to lasso (mixing alpha=1).

    ``alphas=(0.5, 1)`` searches elastic net and lasso on identical CV splits.
    The lambda grids are derived from the full-data standardized correlations,
    as in path-based CV; centering/scaling for each fold uses training rows
    only. ``min`` minimizes mean fold MSE. ``1se`` chooses the largest lambda
    within one conventional fold-based SE for the winning mixing alpha.
    CV errors are tuning diagnostics, not independent test-error estimates.
    There is no relaxed-lasso or post-selection refit.
    """
    X, y = _validate_xy(X, y)
    n, p = X.shape
    if n < 3:
        raise ValueError("cross-validation requires at least three observations")
    nfolds = _validate_positive_int(nfolds, "nfolds")
    if nfolds < 2:
        raise ValueError("nfolds must be at least two")
    nfolds = min(nfolds, n)
    repeats = _validate_positive_int(repeats, "repeats")
    nlambda = _validate_positive_int(nlambda, "nlambda")
    max_iter = _validate_positive_int(max_iter, "max_iter")
    if choose_rule not in ("min", "1se"):
        raise ValueError("choose_rule must be 'min' or '1se'")
    if not np.isfinite(solver_tol) or solver_tol <= 0:
        raise ValueError("solver_tol must be positive and finite")
    if not np.isfinite(lambda_min_ratio) or not 0 < lambda_min_ratio < 1:
        raise ValueError("lambda_min_ratio must be in (0, 1)")
    alphas = _validate_alpha(alphas)
    names = _feature_names(feature_names, p)
    splits = list(
        RepeatedKFold(n_splits=nfolds, n_repeats=repeats, random_state=seed).split(X)
    )
    config = _parallel._resolve_n_jobs(n_jobs, n_tasks=len(splits))
    fold_ids = np.empty((repeats, n), dtype=int)
    for i, (_, valid) in enumerate(splits):
        fold_ids[i // nfolds, valid] = i % nfolds

    scale = X.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    correlations = np.abs(((X - X.mean(axis=0)) / scale).T @ (y - y.mean())) / n
    max_correlation = float(correlations.max()) if p else 0.0
    grids = np.array(
        [
            np.geomspace(
                max(max_correlation / a, 1e-15),
                max(max_correlation / a, 1e-15) * lambda_min_ratio,
                nlambda,
            )
            for a in alphas
        ]
    )
    messages = []
    if p == 0 or np.std(y) == 0:
        fold_mse = np.array(
            [
                np.full((len(alphas), nlambda), np.mean((y[v] - y[t].mean()) ** 2))
                for t, v in splits
            ]
        ).transpose(1, 2, 0)
        coefficients = np.r_[y.mean(), np.zeros(p)]
        best_alpha, best_lambda = 0, 0
        effective_workers = 1
    else:
        tasks = [(X, y, t, v, alphas, grids, solver_tol, max_iter) for t, v in splits]
        results = (
            [_score_fold(*task) for task in tasks]
            if config.serial
            else _parallel._run_parallel(_score_fold, tasks, n_jobs=config.effective)
        )
        fold_mse = np.stack([item[0] for item in results], axis=2)
        for _, caught in results:
            messages.extend(caught)
        mean_mse = fold_mse.mean(axis=2)
        best_alpha, best_lambda = np.unravel_index(np.argmin(mean_mse), mean_mse.shape)
        if choose_rule == "1se":
            se = fold_mse.std(axis=2, ddof=1) / np.sqrt(len(splits))
            threshold = mean_mse[best_alpha, best_lambda] + se[best_alpha, best_lambda]
            best_lambda = int(np.flatnonzero(mean_mse[best_alpha] <= threshold)[0])
        final_path, caught = _path(
            X, y, alphas[best_alpha], grids[best_alpha], solver_tol, max_iter
        )
        coefficients = final_path[:, best_lambda]
        messages.extend(caught)
        effective_workers = config.effective
    y_sd = float(np.std(y))
    standardized = coefficients[1:] * X.std(axis=0) / (y_sd if y_sd > 0 else 1.0)
    return CVLassoResult(
        coefficients=coefficients,
        predictions=X @ coefficients[1:] + coefficients[0],
        feature_names=names,
        standardized_coefficients=standardized,
        selected_alpha=float(alphas[best_alpha]),
        selected_lambda=float(grids[best_alpha, best_lambda]),
        alphas=alphas,
        lambda_grid=grids,
        cv_mean_mse=fold_mse.mean(axis=2),
        cv_se_mse=fold_mse.std(axis=2, ddof=1) / np.sqrt(len(splits)),
        fold_mse=fold_mse,
        fold_ids=fold_ids,
        nfolds=nfolds,
        repeats=repeats,
        choose_rule=choose_rule,
        seed=seed,
        workers=effective_workers,
        warnings=tuple(dict.fromkeys(messages)),
    )
