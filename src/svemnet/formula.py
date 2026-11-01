"""Optional R-style formula interface (requires ``svemnet[formula]``).

Mirrors the workflow of the R package SVEMnet for users coming from R or
JMP: fit from a model formula and a DataFrame, with categorical predictors
encoded automatically (pandas ``Categorical`` or string columns become
treatment-coded contrast blocks, exactly like R factors), and predict from
new DataFrames with the training encodings reapplied statefully.

Formulas use `formulaic <https://matthewwardrop.github.io/formulaic/>`_,
whose syntax matches R closely — in particular ``(X1 + X2 + X3)**2``
expands to main effects plus two-way interactions like R's
``(X1 + X2 + X3)^2``, and ``I(X1**2)`` is a literal square.

Example
-------
>>> import svemnet
>>> model = svemnet.svem("y ~ (X1 + X2 + X3)**2 + I(X1**2)", df,
...                      method="forward", seed=1)
>>> preds = model.predict(new_df)
>>> model.coef_table()
"""

from __future__ import annotations

import warnings as _pywarnings
from typing import Mapping, Sequence

import numpy as np

try:
    import pandas as pd
    from formulaic import Formula
    from formulaic.errors import DataMismatchWarning
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "The formula interface requires the optional dependencies "
        "'formulaic' and 'pandas'. Install them with: pip install svemnet[formula]"
    ) from exc

from .core import fit_svem
from .forward import ForwardICResult, _method_objective, fit_svem_forward, forward_select

def _design_from_formula(formula: str, data: "pd.DataFrame", *,
                         prune_zero_columns: bool = True,
                         return_metadata: bool = False):
    """Build (y, X, feature_names, groups, model_spec) from a two-sided formula.

    The structural intercept column is removed from ``X`` (the fitters add
    their own); whole-effect groups map each non-intercept term to its
    contrast-block columns so factors enter forward paths as a unit.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    # Formulaic removes missing rows by index label. Use positional labels so
    # duplicate source labels cannot remove additional, otherwise valid rows.
    positional = data.copy(deep=False)
    positional.index = pd.RangeIndex(len(data))
    dropped_rows: set[int] = set()
    matrices = Formula(formula).get_model_matrix(positional, drop_rows=dropped_rows)
    if not hasattr(matrices, "lhs") or not hasattr(matrices, "rhs"):
        raise ValueError("formula must be two-sided, e.g. 'y ~ X1 + X2'")
    removed_categories = {}
    retained = tuple(int(i) for i in matrices.rhs.index)
    required = {str(variable) for variable in matrices.rhs.model_spec.required_variables}
    for name in required.intersection(positional.columns):
        column = positional[name]
        if not isinstance(column.dtype, pd.CategoricalDtype):
            continue
        observed = column.iloc[list(retained)].dropna().unique()
        absent = tuple(level for level in column.cat.categories if level not in observed)
        if absent:
            removed_categories[name] = absent
            positional[name] = column.cat.remove_categories(absent)
    if removed_categories:
        matrices = Formula(formula).get_model_matrix(positional)
        detail = "; ".join(f"{name}: {list(levels)!r}"
                           for name, levels in sorted(removed_categories.items()))
        _pywarnings.warn(
            f"Removed unobserved categorical level(s) from formula predictors: {detail}.",
            stacklevel=3,
        )
    lhs, rhs = matrices.lhs, matrices.rhs
    y = np.asarray(lhs, dtype=float)
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError("formula must have a single numeric response")
    y = y[:, 0]

    spec = rhs.model_spec
    columns = list(rhs.columns)
    if sum(len(indices) for indices in spec.term_indices.values()) != len(columns):
        raise ValueError(
            "formula terms produced duplicate design-column names; a predictor "
            "named 'Intercept' conflicts with Formulaic's intercept. Rename the "
            "conflicting input column or use the numeric matrix API."
        )
    intercept_cols: list[int] = []
    groups: dict[str, tuple[int, ...]] = {}
    for term, idx in spec.term_indices.items():
        term_str = str(term)
        if all(str(factor) == "1" and factor.eval_method.value == "literal"
               for factor in term.factors):
            intercept_cols.extend(int(i) for i in idx)
        else:
            groups[term_str] = tuple(int(i) for i in idx)
    if len(intercept_cols) != 1:
        raise ValueError(
            "the formula must include the (default) intercept; "
            "'-1'/'+0' formulas are not supported"
        )

    keep = [j for j in range(len(columns)) if j not in set(intercept_cols)]
    remap = {old: new for new, old in enumerate(keep)}
    X = np.asarray(rhs, dtype=float)[:, keep]
    feature_names = tuple(columns[j] for j in keep)
    groups = {
        name: tuple(remap[c] for c in cols) for name, cols in groups.items()
    }

    # A categorical level that is declared but never observed produces an
    # all-zero contrast column, which would make its whole term group rank
    # deficient (and so never selectable by the forward engines). Drop such
    # columns with a warning; the level was never estimable anyway.
    zero_cols = [j for j in range(X.shape[1]) if not np.any(X[:, j])]
    if prune_zero_columns and zero_cols:
        dropped = [feature_names[j] for j in zero_cols]
        _pywarnings.warn(
            "dropping all-zero design column(s) "
            f"{dropped}; a categorical predictor likely declares levels "
            "absent from the data (consider .cat.remove_unused_categories()).",
            stacklevel=3,
        )
        keep2 = [j for j in range(X.shape[1]) if j not in set(zero_cols)]
        remap2 = {old: new for new, old in enumerate(keep2)}
        X = X[:, keep2]
        feature_names = tuple(feature_names[j] for j in keep2)
        groups = {
            name: tuple(remap2[c] for c in cols if c in remap2)
            for name, cols in groups.items()
        }
        groups = {name: cols for name, cols in groups.items() if cols}
    kept_rows = tuple(int(i) for i in rhs.index)
    dropped_rows = set(range(len(data))).difference(kept_rows)
    metadata = {
        "source_row_count": len(data),
        "unused_categorical_levels": removed_categories,
        "training_row_positions": kept_rows,
        "dropped_row_positions": tuple(sorted(dropped_rows)),
        "training_row_labels": tuple(data.index[i] for i in kept_rows),
    }
    if dropped_rows:
        _pywarnings.warn(
            f"Dropped {len(dropped_rows)} training row(s) with missing model values; "
            "training_row_positions_ records the retained source positions.",
            stacklevel=3,
        )
    result = (y, X, feature_names, groups, spec)
    return (*result, metadata) if return_metadata else result


class SVEMFormulaModel:
    """A fitted SVEM model with a formula front end.

    Create with :func:`svem`. Wraps an ensemble result
    (:class:`~svemnet.core.SVEMGaussianResult` or
    :class:`~svemnet.forward.SVEMForwardResult`, available as ``.result_``)
    together with the formulaic model spec used to rebuild design matrices
    from new DataFrames.
    """

    def __init__(self, result, model_spec, feature_names, formula: str,
                 groups: Mapping[str, Sequence[int]], training_metadata=None):
        self.result_ = result
        self.model_spec_ = model_spec
        self.feature_names_ = tuple(feature_names)
        self.formula_ = formula
        self.groups_ = dict(groups)
        metadata = dict(training_metadata or {})
        self.training_row_positions_ = metadata.get("training_row_positions")
        self.dropped_row_positions_ = metadata.get("dropped_row_positions", ())
        self.training_row_labels_ = metadata.get("training_row_labels")
        self.source_row_count_ = metadata.get("source_row_count")
        self.unused_categorical_levels_ = metadata.get("unused_categorical_levels", {})

    def _design(self, data: "pd.DataFrame") -> np.ndarray:
        # formulaic warns and silently encodes unseen factor levels as the
        # reference level, and silently DROPS rows with missing values
        # (shifting the returned predictions); fail fast on both instead.
        with _pywarnings.catch_warnings(record=True) as caught:
            _pywarnings.simplefilter("always")
            try:
                rhs = self.model_spec_.get_model_matrix(data, na_action="raise")
            except Exception as exc:
                raise ValueError(
                    "could not build the design matrix for the new data "
                    f"({exc}). Rows with missing values in model variables "
                    "must be dropped or imputed before predicting."
                ) from exc
        for item in caught:
            if isinstance(item.message, DataMismatchWarning):
                raise ValueError(
                    "new data contains factor levels not seen at fit time "
                    f"({item.message}). Filter those rows or align the "
                    "categories with the training data before predicting."
                )
            _pywarnings.warn_explicit(
                item.message, item.category, item.filename, item.lineno
            )
        columns = list(rhs.columns)
        X = np.asarray(rhs, dtype=float)
        keep = [j for j, c in enumerate(columns) if c in set(self.feature_names_)]
        by_name = {columns[j]: j for j in keep}
        order = [by_name[name] for name in self.feature_names_]
        return X[:, order]

    def predict(
        self,
        data: "pd.DataFrame",
        *,
        debias: bool | None = None,
        se_fit: bool = False,
        interval: bool = False,
        level: float = 0.95,
    ):
        """Predict for new data; see :func:`svemnet.core.predict_svem`."""
        X = self._design(data)
        return self.result_.predict(
            X, debias=debias, se_fit=se_fit, interval=interval, level=level
        )

    def coef_table(self) -> "pd.DataFrame":
        """Ensemble coefficients with bootstrap nonzero rates, as a DataFrame."""
        coefs = np.asarray(self.result_.coefficients, dtype=float)
        names = ("Intercept",) + self.feature_names_
        cm = np.asarray(self.result_.coef_matrix, dtype=float)
        nonzero = (cm != 0.0).mean(axis=0)
        return pd.DataFrame(
            {
                "coefficient": coefs,
                "pct_bootstraps_nonzero": 100.0 * nonzero,
            },
            index=list(names),
        )

    @property
    def selection_frequencies(self) -> dict[str, float]:
        """Per-term bootstrap selection frequencies (forward method only)."""
        freq = getattr(self.result_, "selection_frequencies", None)
        if freq is None:
            raise AttributeError(
                "selection_frequencies is available for method='forward' only"
            )
        return dict(freq)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = type(self.result_).__name__
        n_boot = getattr(self.result_, "nBoot_used", None)
        suffix = f", nBoot={n_boot}" if n_boot is not None else ""
        return f"{type(self).__name__}({self.formula_!r}, {kind}{suffix})"


def svem(
    formula: str,
    data: "pd.DataFrame",
    *,
    method: str = "elastic_net",
    nBoot: int = 100,
    objective: str | None = None,
    weight_scheme: str = "SVEM",
    alphas: Sequence[float] = (1.0,),
    debias: bool = False,
    seed: int | None = None,
    n_jobs: int | None = 1,
) -> SVEMFormulaModel:
    """Fit a Gaussian SVEM model from an R-style formula and a DataFrame.

    The Python counterpart of the R call ``SVEMnet(formula, data, ...)``.
    ``method="elastic_net"`` uses lasso/elastic-net base learners
    (:func:`svemnet.core.fit_svem`); ``method="forward"`` uses
    forward-selection base learners (:func:`svemnet.forward.fit_svem_forward`)
    with whole-effect groups taken from the formula terms, so factor
    contrast blocks enter together. The default objective is Gaussian likelihood
    for forward selection and wAIC for elastic net. ``n_jobs`` controls bootstrap workers;
    the default ``1`` remains serial for JMP and other embedded interpreters.
    """
    y, X, names, groups, spec, metadata = _design_from_formula(
        formula, data, return_metadata=True
    )
    objective = _method_objective(method, objective)
    if method == "elastic_net":
        result = fit_svem(
            X,
            y,
            nBoot=nBoot,
            alpha=alphas,
            objective=objective,
            weight_scheme=weight_scheme,
            seed=seed,
            debias=debias,
            feature_names=names,
            n_jobs=n_jobs,
        )
    elif method == "forward":
        result = fit_svem_forward(
            X,
            y,
            nBoot=nBoot,
            objective=objective,
            weight_scheme=weight_scheme,
            seed=seed,
            groups=groups,
            feature_names=names,
            debias=debias,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError("method must be 'elastic_net' or 'forward'")
    return SVEMFormulaModel(result, spec, names, formula, groups, metadata)


class ForwardAICcModel(SVEMFormulaModel):
    """A fitted deterministic forward-selection model (see :func:`forward_aicc`)."""

    def predict(self, data: "pd.DataFrame", **kwargs):
        """Predict for new data from the selected model."""
        if kwargs:
            raise TypeError(
                "deterministic forward selection has no bootstrap ensemble; "
                "se_fit/interval/debias are not available"
            )
        X = self._design(data)
        return self.result_.predict(X)

    def coef_table(self) -> "pd.DataFrame":
        coefs = np.asarray(self.result_.coefficients, dtype=float)
        names = ("Intercept",) + self.feature_names_
        return pd.DataFrame({"coefficient": coefs}, index=list(names))

    @property
    def selected_terms(self) -> tuple[str, ...]:
        return self.result_.selected_terms

    @property
    def selection_path(self):
        return self.result_.selection_path


def forward_aicc(
    formula: str,
    data: "pd.DataFrame",
    *,
    criterion: str = "AICc",
    improvement_tolerance: float = 1.0e-12,
) -> ForwardAICcModel:
    """Deterministic forward selection by AICc (or AIC/BIC) from a formula.

    The Python counterpart of the R SVEMnet function ``forward_aicc()``: a
    single-model benchmark companion to :func:`svem`. Whole-effect groups
    come from the formula terms, so factor contrast blocks enter together.
    """
    y, X, names, groups, spec, metadata = _design_from_formula(
        formula, data, return_metadata=True
    )
    result: ForwardICResult = forward_select(
        X,
        y,
        criterion=criterion,
        groups=groups,
        feature_names=names,
        improvement_tolerance=improvement_tolerance,
    )
    return ForwardAICcModel(result, spec, names, formula, groups, metadata)


class CVLassoFormulaModel(SVEMFormulaModel):
    """Single cross-validated penalized model with frozen formula encoding."""

    def predict(self, data):
        return self.result_.predict(self._design(data))

    def coef_table(self):
        coefficients = self.result_.coefficients
        return pd.DataFrame({
            "coefficient": coefficients,
            "retained": coefficients != 0.0,
        }, index=["Intercept", *self.feature_names_])


def lasso_cv(formula, data, *, alphas=(1.0,), nfolds=10, repeats=5,
             choose_rule="min", seed=12345, n_jobs=1):
    """Non-relaxed Gaussian CV companion to R's ``glmnet_with_cv``.

    Uses scikit-learn, not glmnet; identical CV paths or selected models across
    solvers are not promised. The default is lasso; ``alphas=(0.5, 1)`` adds
    elastic-net search. Prediction returns point estimates only.
    """
    from .lasso import fit_lasso_cv

    y, X, names, groups, spec, metadata = _design_from_formula(
        formula, data, return_metadata=True
    )
    result = fit_lasso_cv(X, y, alphas=alphas, nfolds=nfolds, repeats=repeats,
                         choose_rule=choose_rule, seed=seed, n_jobs=n_jobs,
                         feature_names=names)
    return CVLassoFormulaModel(result, spec, names, formula, groups, metadata)


__all__ = [
    "CVLassoFormulaModel",
    "SVEMFormulaModel",
    "ForwardAICcModel",
    "svem",
    "forward_aicc",
    "lasso_cv",
]
