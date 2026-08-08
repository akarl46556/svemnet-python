"""scikit-learn estimator interface for SVEM."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from .core import fit_svem, predict_svem
from .forward import fit_svem_forward

_METHODS = ("elastic_net", "forward")


class SVEMRegressor(RegressorMixin, BaseEstimator):
    """Self-Validated Ensemble Model regressor (Gaussian).

    Fits an SVEM ensemble on a numeric design matrix: per bootstrap,
    anti-correlated fractional-random-weight train/validation copies are
    drawn, a base-learner path is fit on the training copy (an elastic-net
    lambda path, or a greedy forward-selection path), the path point
    minimizing a validation-weighted information criterion wins, and winning
    coefficient vectors are averaged across bootstraps.

    ``X`` must be numeric with no intercept column (expand categoricals
    beforehand, e.g. with ``ColumnTransformer``/``OneHotEncoder`` or the
    optional formula interface in :mod:`svemnet.formula`).

    Parameters
    ----------
    method:
        ``"elastic_net"`` (default) for glmnet-style lasso/elastic-net base
        learners, or ``"forward"`` for forward-selection base learners.
    n_boot:
        Number of FRW/SVEM bootstrap replicates (default 100).
    objective:
        ``"wAIC"`` (default), ``"wBIC"``, or ``"wSSE"``.
    weight_scheme:
        ``"SVEM"`` (default), ``"FRW_plain"``, or ``"Identity"``.
    alphas:
        Elastic-net mixing parameters in (0, 1] (``method="elastic_net"``
        only). Default ``(0.5, 1.0)``.
    debias:
        If True, apply the training-data linear calibration ``a + b * yhat``
        to predictions when eligible (at least 10 bootstrap members and
        varying fitted values).
    groups:
        Optional ordered mapping of whole-effect group name to column
        indices of ``X`` (``method="forward"`` only): multi-column terms
        enter and leave the forward path together. Default: every column is
        its own group.
    random_state:
        Seed (int), NumPy RandomState/Generator, or None for
        non-deterministic fits.

    Attributes
    ----------
    result_:
        The underlying :class:`~svemnet.core.SVEMGaussianResult` or
        :class:`~svemnet.forward.SVEMForwardResult`, with the full bootstrap
        coefficient matrix and diagnostics.
    intercept_, coef_:
        Ensemble coefficients used by :meth:`predict` (debiased when
        ``debias=True`` and calibration was eligible).
    """

    def __init__(
        self,
        method: str = "elastic_net",
        n_boot: int = 100,
        objective: str = "wAIC",
        weight_scheme: str = "SVEM",
        alphas: Sequence[float] = (0.5, 1.0),
        debias: bool = False,
        groups: Mapping[str, Sequence[int]] | None = None,
        random_state: int | None = None,
    ) -> None:
        self.method = method
        self.n_boot = n_boot
        self.objective = objective
        self.weight_scheme = weight_scheme
        self.alphas = alphas
        self.debias = debias
        self.groups = groups
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the SVEM ensemble."""
        X, y = validate_data(self, X, y, y_numeric=True)
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {list(_METHODS)}")
        seed = self._derived_seed()
        names = getattr(self, "feature_names_in_", None)
        feature_names = None if names is None else tuple(str(n) for n in names)
        if self.method == "elastic_net":
            if self.groups is not None:
                raise ValueError(
                    "groups is only supported with method='forward'"
                )
            self.result_ = fit_svem(
                X,
                y,
                nBoot=self.n_boot,
                alpha=self.alphas,
                objective=self.objective,
                weight_scheme=self.weight_scheme,
                seed=seed,
                debias=self.debias,
                feature_names=feature_names,
            )
        else:
            self.result_ = fit_svem_forward(
                X,
                y,
                nBoot=self.n_boot,
                objective=self.objective,
                weight_scheme=self.weight_scheme,
                seed=seed,
                groups=self.groups,
                feature_names=feature_names,
                debias=self.debias,
            )
        self.intercept_ = float(self.result_.coefficients[0])
        self.coef_ = np.asarray(self.result_.coefficients[1:], dtype=float).copy()
        return self

    def predict(self, X):
        """Predict with the ensemble coefficients."""
        check_is_fitted(self, "result_")
        X = validate_data(self, X, reset=False)
        return np.asarray(X @ self.coef_ + self.intercept_, dtype=float)

    def predict_interval(self, X, confidence_level: float = 0.9):
        """Point predictions with bootstrap percentile intervals.

        Returns ``(y_pred, intervals)`` where ``intervals`` has shape
        ``(n_samples, 2)`` holding the lower and upper percentile bounds of
        the per-bootstrap member predictions at ``confidence_level``
        (debiased when the estimator was fit with ``debias=True``).
        """
        check_is_fitted(self, "result_")
        X = validate_data(self, X, reset=False)
        out = predict_svem(
            self.result_,
            X,
            se_fit=False,
            interval=True,
            level=confidence_level,
        )
        return out["fit"], np.column_stack([out["lwr"], out["upr"]])

    def predict_se(self, X):
        """Bootstrap standard errors of the member predictions."""
        check_is_fitted(self, "result_")
        X = validate_data(self, X, reset=False)
        out = predict_svem(self.result_, X, se_fit=True)
        return out["fit"], out["se.fit"]

    def _derived_seed(self) -> int | None:
        if self.random_state is None:
            return None
        if isinstance(self.random_state, (int, np.integer)):
            return int(self.random_state)
        return int(check_random_state(self.random_state).randint(0, 2**31 - 1))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = False
        return tags


__all__ = ["SVEMRegressor"]
