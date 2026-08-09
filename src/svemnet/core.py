"""Pure-Python Gaussian SVEM elastic-net core.

This module is intentionally standalone: it depends only on NumPy and
scikit-learn. It expects the caller to supply a numeric expanded model matrix
``X`` with shape ``(n, p)`` and no intercept column.

The implementation targets algorithmic fidelity to the Gaussian, non-relaxed
SVEMnet core:

* fractional random weights (FRW) with anti-correlated validation weights;
* validation-weighted wSSE, wAIC, and wBIC path selection;
* Kish effective-sample-size admissibility guardrail for wAIC/wBIC;
* intercept-only fallback;
* coefficient averaging over bootstrap members;
* optional Gaussian linear debiasing/calibration;
* bootstrap-member prediction standard errors and percentile intervals.

No multiprocessing, threading, global RNG state, project imports, formulas,
contrasts, blocking, binomial response, or relaxed-lasso refits are used here.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import warnings

import numpy as np

# Lazy optional imports are essential in JMP's persistent Python interpreter:
# an initial dependency miss may be repaired in-session, so a failed import must
# never be cached as permanent module state.
sklearn = None
ConvergenceWarning: type[Warning] = Warning
enet_path = None


def _load_sklearn() -> None:
    """Load scikit-learn on demand and retry after any earlier ImportError."""

    global sklearn, ConvergenceWarning, enet_path
    if sklearn is not None and enet_path is not None:
        return
    try:
        package = importlib.import_module("sklearn")
        exceptions = importlib.import_module("sklearn.exceptions")
        linear_model = importlib.import_module("sklearn.linear_model")
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "SVEM elastic-net fitting requires scikit-learn "
            "(a declared dependency of svemnet); reinstall with "
            "'pip install svemnet' or 'pip install scikit-learn'"
        ) from exc
    sklearn = package
    ConvergenceWarning = exceptions.ConvergenceWarning
    enet_path = linear_model.enet_path


_FLOAT_EPS = float(np.finfo(float).eps)
_FLOAT_TINY = float(np.finfo(float).tiny)
_SUPPORTED_OBJECTIVES = {"wSSE", "wAIC", "wBIC"}
_SUPPORTED_WEIGHT_SCHEMES = {"SVEM", "FRW_plain", "Identity"}


@dataclass(frozen=True)
class SVEMGaussianResult:
    """Result from :func:`fit_svem`.

    Attributes
    ----------
    coefficients:
        Coefficients used for ``predictions``. This is ``raw_coefficients``
        unless ``fit_svem(..., debias=True)`` requested calibrated coefficients
        and calibration was possible. Element 0 is the intercept; elements
        1..p align to the columns of the caller-supplied ``X``.
    coef_matrix:
        Per-bootstrap selected coefficients, rows = used bootstrap members,
        columns = intercept then predictors. These are raw member coefficients;
        calibration is applied on-the-fly in prediction summaries when requested.
    predictions:
        Training predictions from ``coefficients`` on the supplied ``X``.
    diagnostics:
        JSON-friendly summaries of selected model sizes, fallback rate,
        effective validation sizes, selected alpha frequencies, and
        reproducibility settings.
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
    selected_alpha: np.ndarray
    selected_lambda: np.ndarray
    selected_k: np.ndarray
    n_eff_raw: np.ndarray
    n_eff_adm: np.ndarray
    fallback_mask: np.ndarray
    coef_tol: float
    pi_sigma: float | None = None
    pi_df: float | None = None

    def predict(
        self,
        X: Sequence[Sequence[float]] | np.ndarray,
        *,
        debias: bool | None = None,
        se_fit: bool = False,
        interval: bool | str = False,
        level: float = 0.95,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Predict from a fitted SVEM Gaussian model.

        When ``se_fit`` or ``interval`` is requested, summaries are computed
        from per-bootstrap member predictions in ``coef_matrix``. With
        ``debias=True``, the stored linear calibration ``a + b*yhat`` is
        applied to each member prediction before computing standard deviations
        and quantiles.

        ``interval`` accepts ``True`` (equivalently ``"confidence"``) for the
        percentile summary of member predictions — an ensemble-spread
        confidence-style summary for the fitted mean — or ``"prediction"``
        for an interval targeting a NEW OBSERVATION at ``X``:
        ``fit +/- t_df * sqrt(sd_member^2 + pi_sigma^2)`` with the
        validation-weighted residual scale ``pi_sigma`` and
        ``df = n - median(member support size)`` stored at fit time.
        """

        return predict_svem(
            self,
            X,
            debias=debias,
            se_fit=se_fit,
            interval=interval,
            level=level,
        )


def fit_svem(
    X: Sequence[Sequence[float]] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    nBoot: int = 100,
    alpha: Sequence[float] = (0.5, 1.0),
    objective: str = "wAIC",
    weight_scheme: str = "SVEM",
    seed: int | None = None,
    nlambda: int = 500,
    lambda_min_ratio: float | None = None,
    solver_tol: float = 1.0e-7,
    max_iter: int = 100000,
    coef_tol: float = 1.0e-7,
    debias: bool = False,
    feature_names: Sequence[str] | None = None,
    weight_uniforms: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> SVEMGaussianResult:
    """Fit a Gaussian SVEM ensemble with elastic-net base learners.

    Parameters
    ----------
    X, y:
        Numeric design matrix and response. ``X`` must not contain an intercept
        column. Missing or non-finite values are rejected.
    nBoot:
        Number of FRW/SVEM bootstrap replicates. The package default is 100
        for iterative small-design workflows; users can raise it for
        production runs.
    alpha:
        Elastic-net mixing parameters, i.e. scikit-learn ``l1_ratio`` values.
        Values must be in ``(0, 1]``. The default ``(0.5, 1.0)`` matches the R
        default used in the requested Gaussian scope.
    objective:
        One of ``"wSSE"``, ``"wAIC"``, or ``"wBIC"``. ``"wAIC"`` is the
        Gaussian default.
    weight_scheme:
        ``"SVEM"`` for anti-correlated train/validation FRW weights,
        ``"FRW_plain"`` for the same FRW vector in both roles as a
        comparison/legacy mode, or ``"Identity"`` for unit weights as a
        diagnostic mode.
    seed:
        Seed passed to ``numpy.random.default_rng``. The module never touches
        NumPy's global RNG state.
    nlambda:
        Number of elastic-net path points. Default 500 follows the R glmnet
        call; the exact lambda grid is scikit-learn's, not glmnet's.
    lambda_min_ratio:
        Minimum lambda divided by maximum lambda for the generated path.
        If omitted, uses ``1e-4`` when ``n > p`` and ``1e-2`` otherwise,
        a glmnet-like convention.
    solver_tol, max_iter:
        Coordinate-descent controls passed to scikit-learn's path solver.
    coef_tol:
        Base tolerance for selected support size. Slopes are counted when
        ``abs(beta_j) > coef_tol * max(1, max_abs_slope_for_path_point)``;
        the intercept is counted structurally as one parameter, matching the
        current R implementation.
    debias:
        If true and eligible, return calibrated coefficients and predictions.
        Raw and calibrated coefficients are both stored in the result.
    feature_names:
        Optional predictor names aligned to columns of ``X``.
    weight_uniforms:
        Optional matrix with shape ``(nBoot, n)`` used as the shared uniforms
        for FRW/SVEM weights. This is a deterministic test/parity hook; default
        behavior draws from a local ``numpy.random.Generator``.
    """

    _load_sklearn()

    X_arr, y_arr = _validate_xy(X, y)
    n, p = X_arr.shape
    nBoot_int = _validate_positive_int(nBoot, "nBoot")
    nlambda_int = _validate_positive_int(nlambda, "nlambda")
    if objective not in _SUPPORTED_OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(_SUPPORTED_OBJECTIVES)}")
    if weight_scheme not in _SUPPORTED_WEIGHT_SCHEMES:
        raise ValueError(
            f"weight_scheme must be one of {sorted(_SUPPORTED_WEIGHT_SCHEMES)}"
        )
    alpha_values = _validate_alpha(alpha)
    if solver_tol <= 0 or not np.isfinite(solver_tol):
        raise ValueError("solver_tol must be positive and finite")
    if max_iter < 1 or not np.isfinite(max_iter):
        raise ValueError("max_iter must be a positive integer")
    max_iter_int = int(max_iter)
    if coef_tol <= 0 or not np.isfinite(coef_tol):
        raise ValueError("coef_tol must be positive and finite")
    if lambda_min_ratio is None:
        lambda_min_ratio = 1.0e-4 if n > p else 1.0e-2
    if not (0 < lambda_min_ratio < 1) or not np.isfinite(lambda_min_ratio):
        raise ValueError("lambda_min_ratio must be in (0, 1)")
    names = _feature_names(feature_names, p)
    uniform_matrix = _validate_weight_uniforms(
        weight_uniforms,
        nBoot=nBoot_int,
        n=n,
    )

    rng = np.random.default_rng(seed)

    coef_rows: list[np.ndarray] = []
    best_alphas: list[float] = []
    best_lambdas: list[float] = []
    k_selected: list[int] = []
    fallback_flags: list[bool] = []
    n_eff_raw_values: list[float] = []
    n_eff_adm_values: list[float] = []
    valid_sse_values: list[float] = []
    warning_messages: list[str] = []

    for boot_index in range(nBoot_int):
        uniforms = None if uniform_matrix is None else uniform_matrix[boot_index]
        w_train, w_valid = make_svem_weights(
            n,
            rng,
            scheme=weight_scheme,
            uniforms=uniforms,
        )
        n_eff_raw, n_eff_adm = kish_effective_n(w_valid, n=n)
        n_eff_raw_values.append(float(n_eff_raw))
        n_eff_adm_values.append(float(n_eff_adm))

        best_score = np.inf
        best_alpha = np.nan
        best_lambda = np.nan
        best_coef: np.ndarray | None = None
        best_k = 1
        best_valid_sse = np.nan

        if p > 0:
            for l1_ratio in alpha_values:
                path = _weighted_enet_path(
                    X_arr,
                    y_arr,
                    w_train,
                    l1_ratio=float(l1_ratio),
                    nlambda=nlambda_int,
                    lambda_min_ratio=float(lambda_min_ratio),
                    solver_tol=float(solver_tol),
                    max_iter=max_iter_int,
                )
                if path is None:
                    continue
                lambdas, coef_path, path_warnings = path
                warning_messages.extend(
                    f"bootstrap {boot_index + 1}, alpha {l1_ratio:g}: {msg}"
                    for msg in path_warnings
                )
                if coef_path.size == 0:
                    continue

                pred_path = X_arr @ coef_path[1:, :] + coef_path[0:1, :]
                residuals = pred_path - y_arr[:, None]
                sse_w = np.sum(w_valid[:, None] * residuals * residuals, axis=0)
                sse_w = np.where(np.isfinite(sse_w), sse_w, np.inf)
                k_path = support_size(coef_path, base_tol=float(coef_tol))
                scores = weighted_ic_scores(
                    sse_w,
                    k_path,
                    n_like=float(np.sum(w_valid)),
                    n_eff_adm=float(n_eff_adm),
                    objective=objective,
                )
                if not np.any(np.isfinite(scores)):
                    continue
                local_idx = int(np.nanargmin(scores))
                local_score = float(scores[local_idx])
                if local_score < best_score:
                    best_score = local_score
                    best_alpha = float(l1_ratio)
                    best_lambda = float(lambdas[local_idx])
                    best_coef = coef_path[:, local_idx].copy()
                    best_k = int(k_path[local_idx])
                    best_valid_sse = float(sse_w[local_idx])

        fallback = best_coef is None or not np.all(np.isfinite(best_coef))
        if fallback:
            best_coef = _intercept_only_coefficients(y_arr, w_train, p)
            best_alpha = np.nan
            best_lambda = np.nan
            best_k = 1
            resid_fb = y_arr - best_coef[0]
            best_valid_sse = float(np.sum(w_valid * resid_fb * resid_fb))

        coef_rows.append(best_coef)
        best_alphas.append(best_alpha)
        best_lambdas.append(best_lambda)
        k_selected.append(best_k)
        fallback_flags.append(bool(fallback))
        valid_sse_values.append(best_valid_sse)

    coef_matrix = np.vstack(coef_rows) if coef_rows else np.empty((0, p + 1))
    finite_rows = np.all(np.isfinite(coef_matrix), axis=1)
    if not np.any(finite_rows):
        raise RuntimeError("All bootstrap iterations failed to produce finite coefficients")

    coef_matrix = coef_matrix[finite_rows]
    best_alpha_arr = np.asarray(best_alphas, dtype=float)[finite_rows]
    best_lambda_arr = np.asarray(best_lambdas, dtype=float)[finite_rows]
    k_arr = np.asarray(k_selected, dtype=int)[finite_rows]
    fallback_arr = np.asarray(fallback_flags, dtype=bool)[finite_rows]
    n_eff_raw_arr = np.asarray(n_eff_raw_values, dtype=float)[finite_rows]
    n_eff_adm_arr = np.asarray(n_eff_adm_values, dtype=float)[finite_rows]
    valid_sse_arr = np.asarray(valid_sse_values, dtype=float)[finite_rows]

    # Prediction-interval scalars: validation-weighted per-member residual
    # variance with per-member support df (median-aggregated, robust to
    # fallback members), and the t degrees of freedom n - median(k).
    # Computed from quantities already in hand - no additional RNG draws,
    # so bootstrap parity with earlier versions is preserved.
    with np.errstate(invalid="ignore", divide="ignore"):
        member_df = np.maximum(n - k_arr.astype(float), 1.0)
        member_sig2 = valid_sse_arr / member_df
    finite_sig2 = member_sig2[np.isfinite(member_sig2)]
    pi_sigma = float(np.sqrt(np.median(finite_sig2))) if finite_sig2.size else None
    pi_df = float(max(n - float(np.median(k_arr)), 1.0)) if k_arr.size else None

    raw_coefficients = np.mean(coef_matrix, axis=0)
    raw_predictions = X_arr @ raw_coefficients[1:] + raw_coefficients[0]

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
    coefficients = debiased_coefficients.copy() if use_debiased else raw_coefficients.copy()
    predictions = debiased_predictions.copy() if use_debiased else raw_predictions.copy()

    diagnostics = _diagnostics(
        k=k_arr,
        fallback=fallback_arr,
        n_eff_raw=n_eff_raw_arr,
        selected_alpha=best_alpha_arr,
        warnings=warning_messages,
        nBoot_requested=nBoot_int,
        nBoot_used=int(coef_matrix.shape[0]),
        objective=objective,
        weight_scheme=weight_scheme,
        alpha_values=alpha_values,
        nlambda=nlambda_int,
        lambda_min_ratio=float(lambda_min_ratio),
        solver_tol=float(solver_tol),
        max_iter=max_iter_int,
        coef_tol=float(coef_tol),
        seed=seed,
        weight_uniforms_supplied=uniform_matrix is not None,
        debias_eligible=debiased_coefficients is not None,
        debias_applied=use_debiased,
    )

    return SVEMGaussianResult(
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
        nBoot_used=int(coef_matrix.shape[0]),
        selected_alpha=best_alpha_arr,
        selected_lambda=best_lambda_arr,
        selected_k=k_arr,
        n_eff_raw=n_eff_raw_arr,
        n_eff_adm=n_eff_adm_arr,
        fallback_mask=fallback_arr,
        coef_tol=float(coef_tol),
        pi_sigma=pi_sigma,
        pi_df=pi_df,
    )


def predict_svem(
    result: SVEMGaussianResult,
    X: Sequence[Sequence[float]] | np.ndarray,
    *,
    debias: bool | None = None,
    se_fit: bool = False,
    interval: bool | str = False,
    level: float = 0.95,
) -> np.ndarray | dict[str, np.ndarray]:
    """Predict from a :class:`SVEMGaussianResult`.

    The returned dictionary, when requested, follows the R predict method's
    component names: ``fit``, optional ``se.fit``, and optional ``lwr``/``upr``.
    ``interval`` may be ``False``, ``True``/``"confidence"`` (percentile
    ensemble-spread summary, as always), or ``"prediction"`` (new-observation
    interval using the fit-time ``pi_sigma``/``pi_df`` scalars).
    """

    X_arr = _validate_X_new(X, len(result.feature_names))
    if not (0.0 < level < 1.0) or not np.isfinite(level):
        raise ValueError("level must be a finite number in (0, 1)")
    if isinstance(interval, str):
        if interval not in ("confidence", "prediction"):
            raise ValueError(
                "interval must be a bool, 'confidence', or 'prediction'"
            )
        interval_kind = interval
        interval = True
    else:
        interval_kind = "confidence" if interval else ""
    use_debias = result.debias_applied if debias is None else bool(debias)
    if use_debias and result.debiased_coefficients is not None:
        point_coef = result.debiased_coefficients
    else:
        point_coef = result.raw_coefficients
    point = X_arr @ point_coef[1:] + point_coef[0]

    if not se_fit and not interval:
        return point

    if result.coef_matrix.size == 0:
        raise ValueError("se_fit/interval require a non-empty coef_matrix")
    member_preds = X_arr @ result.coef_matrix[:, 1:].T + result.coef_matrix[:, 0]
    if use_debias and result.debias_params is not None:
        a, b = result.debias_params
        member_preds = a + b * member_preds
        # Keep point prediction algebraically aligned to calibrated member means.
        point = np.mean(member_preds, axis=1)

    out: dict[str, np.ndarray] = {"fit": point}
    if se_fit:
        ddof = 1 if member_preds.shape[1] > 1 else 0
        out["se.fit"] = np.std(member_preds, axis=1, ddof=ddof)
    if interval and interval_kind == "prediction":
        if result.pi_sigma is None or result.pi_df is None:
            raise ValueError(
                "interval='prediction' requires pi_sigma/pi_df from the fit; "
                "re-fit with this svemnet version to populate them"
            )
        try:  # scipy arrives transitively via scikit-learn (a hard dep)
            from scipy import stats as _sstats  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "interval='prediction' requires scipy (installed alongside "
                "scikit-learn)"
            ) from exc
        tcrit = float(_sstats.t.ppf(1.0 - (1.0 - float(level)) / 2.0, result.pi_df))
        ddof = 1 if member_preds.shape[1] > 1 else 0
        sd_member = np.std(member_preds, axis=1, ddof=ddof)
        half = tcrit * np.sqrt(sd_member**2 + result.pi_sigma**2)
        out["lwr"] = point - half
        out["upr"] = point + half
    elif interval:
        tail = (1.0 - float(level)) / 2.0
        out["lwr"] = np.quantile(member_preds, tail, axis=1)
        out["upr"] = np.quantile(member_preds, 1.0 - tail, axis=1)
    return out


def make_svem_weights(
    n: int,
    rng: np.random.Generator,
    *,
    scheme: str = "SVEM",
    uniforms: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw and mean-1 rescale training/validation weights.

    With ``scheme="SVEM"`` the same uniform draw is transformed into
    anti-correlated train and validation copies:
    ``w_train=-log(U)`` and ``w_valid=-log(1-U)``. Both vectors are rescaled
    to sum to ``n``.
    """

    n_int = _validate_positive_int(n, "n")
    if scheme not in _SUPPORTED_WEIGHT_SCHEMES:
        raise ValueError(f"scheme must be one of {sorted(_SUPPORTED_WEIGHT_SCHEMES)}")
    if scheme == "Identity":
        return np.ones(n_int, dtype=float), np.ones(n_int, dtype=float)

    if uniforms is None:
        u = rng.uniform(0.0, 1.0, size=n_int)
    else:
        u = np.asarray(uniforms, dtype=float)
        if u.shape != (n_int,):
            raise ValueError("uniforms must have shape (n,)")
        if np.any(~np.isfinite(u)) or np.any((u <= 0.0) | (u >= 1.0)):
            raise ValueError("uniforms must be finite values in the open interval (0, 1)")
    u = np.clip(u, _FLOAT_EPS, 1.0 - _FLOAT_EPS)
    w_train = -np.log(u)
    if scheme == "SVEM":
        w_valid = -np.log1p(-u)
    else:  # FRW_plain
        w_valid = w_train.copy()
    w_train = _rescale_mean_one(w_train)
    w_valid = _rescale_mean_one(w_valid)
    return w_train, w_valid


def kish_effective_n(
    weights: Sequence[float] | np.ndarray,
    *,
    n: int | None = None,
    lower: float = 2.0,
) -> tuple[float, float]:
    """Return raw and admissible Kish effective sample sizes.

    ``n_eff_raw = (sum w)^2 / sum(w^2)`` and
    ``n_eff_adm = clip(n_eff_raw, lower, n)`` when ``n`` is supplied.
    """

    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional vector")
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and nonnegative")
    sum_w = float(np.sum(w))
    sum_w2 = float(np.dot(w, w))
    if sum_w2 <= 0.0:
        raw = 0.0
    else:
        raw = (sum_w * sum_w) / sum_w2
    upper = float(w.size if n is None else n)
    adm = min(max(float(raw), float(lower)), upper)
    return float(raw), float(adm)


def support_size(
    coef_path: np.ndarray,
    *,
    base_tol: float = 1.0e-7,
) -> np.ndarray:
    """Count selected coefficients for each path point.

    The intercept is counted structurally as one coefficient. Slope support is
    counted with the relative/absolute tolerance used in the R implementation.
    """

    coefs = np.asarray(coef_path, dtype=float)
    if coefs.ndim == 1:
        coefs = coefs[:, None]
    if coefs.ndim != 2 or coefs.shape[0] < 1:
        raise ValueError("coef_path must have shape (p + 1, n_path)")
    if coefs.shape[0] == 1:
        return np.ones(coefs.shape[1], dtype=int)
    slopes = coefs[1:, :]
    out = np.ones(coefs.shape[1], dtype=int)
    for j in range(coefs.shape[1]):
        col = slopes[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        max_abs = float(np.max(np.abs(finite)))
        tol = float(base_tol) * max(1.0, max_abs)
        out[j] += int(np.sum(np.abs(finite) > tol))
    return out


def weighted_ic_scores(
    sse_w: Sequence[float] | np.ndarray,
    k: Sequence[int] | np.ndarray,
    *,
    n_like: float,
    n_eff_adm: float,
    objective: str = "wAIC",
) -> np.ndarray:
    """Compute Gaussian validation-weighted wSSE/wAIC/wBIC scores."""

    if objective not in _SUPPORTED_OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(_SUPPORTED_OBJECTIVES)}")
    sse = np.asarray(sse_w, dtype=float)
    kval = np.asarray(k, dtype=int)
    if sse.shape != kval.shape:
        raise ValueError("sse_w and k must have the same shape")
    if n_like <= 0 or not np.isfinite(n_like):
        raise ValueError("n_like must be positive and finite")
    sse_adj = np.maximum(sse, _FLOAT_EPS)
    sse_adj = np.where(np.isfinite(sse_adj), sse_adj, np.inf)
    if objective == "wSSE":
        return sse_adj

    k_slope = np.maximum(kval - 1, 0)
    admissible = k_slope < float(n_eff_adm)
    out = np.full(sse_adj.shape, np.inf, dtype=float)
    mse_w = sse_adj / float(n_like)
    penalty_multiplier = 2.0 if objective == "wAIC" else float(np.log(n_eff_adm))
    out[admissible] = (
        float(n_like) * np.log(mse_w[admissible])
        + penalty_multiplier * kval[admissible]
    )
    return out


def _weighted_enet_path(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    l1_ratio: float,
    nlambda: int,
    lambda_min_ratio: float,
    solver_tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    if X.shape[1] == 0:
        return None
    sum_w = float(np.sum(w))
    if sum_w <= 0.0 or not np.isfinite(sum_w):
        return None
    x_mean = (w @ X) / sum_w
    y_mean = float(w @ y / sum_w)
    X_centered = X - x_mean
    y_centered = y - y_mean
    x_var = (w @ (X_centered * X_centered)) / sum_w
    scale = np.sqrt(np.maximum(x_var, 0.0))
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    X_standard = X_centered / scale
    sqrt_w = np.sqrt(w)
    X_weighted = X_standard * sqrt_w[:, None]
    y_weighted = y_centered * sqrt_w

    path_warnings: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            lambdas, coefs_standard, _ = enet_path(
                X_weighted,
                y_weighted,
                l1_ratio=float(l1_ratio),
                eps=float(lambda_min_ratio),
                n_alphas=int(nlambda),
                copy_X=True,
                check_input=True,
                max_iter=int(max_iter),
                tol=float(solver_tol),
            )
        for item in caught:
            if issubclass(item.category, ConvergenceWarning):
                path_warnings.append(str(item.message))
    except Exception as exc:  # sklearn raises ValueError on degenerate paths
        path_warnings.append(f"path fit failed: {exc}")
        return None

    if coefs_standard.ndim != 2 or coefs_standard.shape[0] != X.shape[1]:
        return None
    slopes = coefs_standard / scale[:, None]
    intercepts = y_mean - x_mean @ slopes
    coef_path = np.vstack([intercepts[None, :], slopes])
    return np.asarray(lambdas, dtype=float), coef_path, path_warnings


def _intercept_only_coefficients(y: np.ndarray, weights: np.ndarray, p: int) -> np.ndarray:
    coef = np.zeros(p + 1, dtype=float)
    coef[0] = float(np.sum(weights * y) / np.sum(weights))
    return coef


def _linear_calibration(yhat: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones_like(yhat, dtype=float), yhat])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _has_variation(x: np.ndarray) -> bool:
    if x.size < 2:
        return False
    return bool(np.isfinite(x).all() and np.var(x, ddof=1) > 0.0)


def _diagnostics(
    *,
    k: np.ndarray,
    fallback: np.ndarray,
    n_eff_raw: np.ndarray,
    selected_alpha: np.ndarray,
    warnings: list[str],
    nBoot_requested: int,
    nBoot_used: int,
    objective: str,
    weight_scheme: str,
    alpha_values: np.ndarray,
    nlambda: int,
    lambda_min_ratio: float,
    solver_tol: float,
    max_iter: int,
    coef_tol: float,
    seed: int | None,
    weight_uniforms_supplied: bool,
    debias_eligible: bool,
    debias_applied: bool,
) -> dict[str, object]:
    finite_alpha = selected_alpha[np.isfinite(selected_alpha)]
    alpha_freq: dict[str, float] = {}
    if finite_alpha.size:
        unique, counts = np.unique(finite_alpha, return_counts=True)
        total = float(np.sum(counts))
        alpha_freq = {f"{float(a):g}": float(c / total) for a, c in zip(unique, counts)}
    return {
        "k_summary": {
            "k_median": float(np.median(k)) if k.size else np.nan,
            "k_iqr": float(np.percentile(k, 75) - np.percentile(k, 25)) if k.size else np.nan,
            "k_min": int(np.min(k)) if k.size else None,
            "k_max": int(np.max(k)) if k.size else None,
        },
        "fallback_rate": float(np.mean(fallback)) if fallback.size else np.nan,
        "n_eff_summary": _summary6(n_eff_raw),
        "alpha_freq": alpha_freq,
        "warnings": tuple(warnings),
        "reproducibility": {
            "deterministic_given_seed": seed is not None,
            "rng": "numpy.random.default_rng",
            "seed": seed,
            "weight_uniforms_supplied": bool(weight_uniforms_supplied),
            "serial": True,
            "uses_global_rng": False,
            "uses_multiprocessing_or_threads": False,
            "sklearn_n_jobs": None,
            "objective": objective,
            "weight_scheme": weight_scheme,
            "alpha": [float(a) for a in alpha_values],
            "nlambda": int(nlambda),
            "lambda_min_ratio": float(lambda_min_ratio),
            "solver_tol": float(solver_tol),
            "max_iter": int(max_iter),
            "coef_tol": float(coef_tol),
            "relaxed": False,
            "family": "gaussian",
        },
        "nBoot_requested": int(nBoot_requested),
        "nBoot_used": int(nBoot_used),
        "debias_eligible": bool(debias_eligible),
        "debias_applied": bool(debias_applied),
    }


def _summary6(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {"Min": np.nan, "1st Qu.": np.nan, "Median": np.nan, "Mean": np.nan, "3rd Qu.": np.nan, "Max": np.nan}
    return {
        "Min": float(np.min(x)),
        "1st Qu.": float(np.percentile(x, 25)),
        "Median": float(np.median(x)),
        "Mean": float(np.mean(x)),
        "3rd Qu.": float(np.percentile(x, 75)),
        "Max": float(np.max(x)),
    }


def _rescale_mean_one(w: np.ndarray) -> np.ndarray:
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weight vector has nonpositive or non-finite sum")
    return w * (w.size / total)


def _validate_xy(
    X: Sequence[Sequence[float]] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("X must be a two-dimensional numeric matrix")
    if y_arr.ndim != 1:
        raise ValueError("y must be a one-dimensional numeric vector")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if X_arr.shape[0] < 2:
        raise ValueError(
            "at least two observations are required "
            f"(got n_samples = {X_arr.shape[0]})"
        )
    if np.any(~np.isfinite(X_arr)) or np.any(~np.isfinite(y_arr)):
        raise ValueError("X and y must be finite")
    return X_arr, y_arr


def _validate_X_new(X: Sequence[Sequence[float]] | np.ndarray, p: int) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        if p == 1:
            X_arr = X_arr.reshape(-1, 1)
        elif X_arr.size == p:
            X_arr = X_arr.reshape(1, p)
        else:
            raise ValueError("one-dimensional X is ambiguous for this number of predictors")
    if X_arr.ndim != 2:
        raise ValueError("X must be a two-dimensional numeric matrix")
    if X_arr.shape[1] != p:
        raise ValueError(f"X must have {p} columns; got {X_arr.shape[1]}")
    if np.any(~np.isfinite(X_arr)):
        raise ValueError("X must be finite")
    return X_arr


def _validate_positive_int(value: int, name: str) -> int:
    if not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a positive integer")
    out = int(value)
    if out < 1 or float(out) != float(value):
        raise ValueError(f"{name} must be a positive integer")
    return out


def _validate_alpha(alpha: Sequence[float] | float) -> np.ndarray:
    if np.isscalar(alpha):
        arr = np.asarray([alpha], dtype=float)
    else:
        arr = np.asarray(tuple(alpha), dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("alpha must be a non-empty one-dimensional sequence or scalar")
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0) or np.any(arr > 1.0):
        raise ValueError("alpha values must be finite and in (0, 1]")
    # Preserve first occurrence order.
    unique: list[float] = []
    for value in arr:
        f = float(value)
        if not any(f == old for old in unique):
            unique.append(f)
    return np.asarray(unique, dtype=float)


def _validate_weight_uniforms(
    value: Sequence[Sequence[float]] | np.ndarray | None,
    *,
    nBoot: int,
    n: int,
) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.shape != (nBoot, n):
        raise ValueError("weight_uniforms must have shape (nBoot, n)")
    if np.any(~np.isfinite(arr)) or np.any((arr <= 0.0) | (arr >= 1.0)):
        raise ValueError(
            "weight_uniforms must contain finite values in the open interval (0, 1)"
        )
    return arr


def _feature_names(feature_names: Sequence[str] | None, p: int) -> tuple[str, ...]:
    if feature_names is None:
        return tuple(f"x{j}" for j in range(p))
    names = tuple(str(name) for name in feature_names)
    if len(names) != p:
        raise ValueError("feature_names length must match X columns")
    if any(name == "" for name in names):
        raise ValueError("feature_names must be non-empty strings")
    return names


__all__ = [
    "SVEMGaussianResult",
    "fit_svem",
    "predict_svem",
    "make_svem_weights",
    "kish_effective_n",
    "support_size",
    "weighted_ic_scores",
    "sklearn",
]
