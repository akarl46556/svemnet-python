"""DataFrame/CSV variable-screening workflow for Gaussian SVEM models."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SVEM screening requires pandas, formulaic, and matplotlib. "
        "Install them with: pip install 'svemnet[screening]'"
    ) from exc

from .core import fit_svem
from .expansion import response_surface_formula
from .formula import SVEMFormulaModel, _design_from_formula
from .forward import fit_svem_forward
from .prediction import PredictionMixin

MODEL_INTERACTIONS = "Main effects + two-way interactions"
MODEL_RESPONSE_SURFACE = "Response surface"
MODEL_CHOICES = (MODEL_INTERACTIONS, MODEL_RESPONSE_SURFACE)
NULL_EFFECT_PREFIX = "Null"
NULL_EFFECT_COLOR = "#8C8C8C"
MODEL_EFFECT_COLOR = "#2878B5"
METHOD_LASSO = "SVEM lasso"
METHOD_ELASTIC_NET = "SVEM elastic net search (0.5, 1)"
METHOD_FORWARD = "SVEM forward selection"
METHOD_CHOICES = (METHOD_LASSO, METHOD_ELASTIC_NET, METHOD_FORWARD)


@dataclass(frozen=True)
class ScreeningDesign:
    data: pd.DataFrame
    y: np.ndarray
    X: np.ndarray
    feature_names: tuple[str, ...]
    groups: dict[str, tuple[int, ...]]
    formula: str
    response: str
    factors: tuple[str, ...]
    categorical: tuple[str, ...]
    continuous: tuple[str, ...]
    center_polynomials: bool
    polynomial_centers: dict[str, float]
    null_effects: tuple[str, ...]
    null_seed: int | None
    rows_dropped: int
    source_data: pd.DataFrame
    training_rows: tuple[int, ...]
    model_spec: object
    centered_spec: object
    null_values: np.ndarray

    def transform(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Reapply training schema; return complete-row matrix and row positions.

        Missing predictor rows are omitted explicitly, never shifted silently.
        Unseen categorical levels are errors. Synthetic null predictors must
        be supplied explicitly for new data, not randomly regenerated.
        """
        required = (*self.factors, *self.null_effects)
        missing = [name for name in required if name not in data.columns]
        if missing:
            raise ValueError(f"prediction columns were not found: {missing}")
        selected = data.loc[:, list(required)].copy()
        for name in (*self.continuous, *self.null_effects):
            selected[name] = pd.to_numeric(selected[name], errors="raise")
        for name in self.categorical:
            # CSV reloads numeric-coded categories as numeric. Preserve the
            # fitted factor role; the frozen ModelSpec still controls levels.
            selected[name] = selected[name].astype("category")
        rows = np.flatnonzero(selected.notna().all(axis=1).to_numpy())
        selected = selected.iloc[rows].reset_index(drop=True)
        if len(rows) == 0:
            return np.empty((0, len(self.feature_names))), rows
        names = self.feature_names[: len(self.feature_names) - len(self.null_effects)]
        raw_model = SVEMFormulaModel(
            None, self.model_spec, names, self.formula, self.groups
        )
        X = raw_model._design(selected)
        if self.center_polynomials:
            centered = selected.copy()
            for name, center in self.polynomial_centers.items():
                centered[name] = centered[name] - center
            centered_model = SVEMFormulaModel(
                None, self.centered_spec, names, self.formula, self.groups
            )
            centered_X = centered_model._design(centered)
            for name in self.continuous:
                centered_X[:, self.groups[name]] = X[:, self.groups[name]]
            X = centered_X
        if self.null_effects:
            X = np.column_stack(
                (X, selected.loc[:, list(self.null_effects)].to_numpy(float))
            )
        if not np.isfinite(X).all():
            raise ValueError("prediction factors must be finite")
        return X, rows


@dataclass(frozen=True)
class ScreeningResult(PredictionMixin):
    design: ScreeningDesign
    fit: object
    effect_usage: pd.DataFrame
    parameter_usage: pd.DataFrame
    elapsed_seconds: float
    workers: int
    method: str
    alphas: tuple[float, ...]

    def metadata(self) -> dict[str, object]:
        return {
            "response": self.design.response,
            "intercept": float(self.fit.coefficients[0]),
            "method": self.method,
            "alpha_candidates": list(self.alphas),
            "relaxed": False,
            "effect_usage_definition": (
                "selected whole effect"
                if self.method == "forward"
                else "any nonzero contrast coefficient in the effect"
            ),
            "factors": list(self.design.factors),
            "categorical_factors": list(self.design.categorical),
            "continuous_factors": list(self.design.continuous),
            "center_polynomials": self.design.center_polynomials,
            "polynomial_centers": dict(self.design.polynomial_centers),
            "null_vector_count": len(self.design.null_effects),
            "null_effects": list(self.design.null_effects),
            "null_vector_distribution": "independent standard normal",
            "null_vector_seed": self.design.null_seed,
            "formula": self.design.formula,
            "rows_used": int(self.design.X.shape[0]),
            "rows_dropped": self.design.rows_dropped,
            "design_columns": int(self.design.X.shape[1]),
            "bootstrap_members_requested": self.fit.nBoot_requested,
            "bootstrap_members_used": self.fit.nBoot_used,
            "objective": "wAIC",
            "workers": self.workers,
            "elapsed_seconds": self.elapsed_seconds,
            "seed": self.fit.seed,
            "svemnet_parallel_backend": self.fit.diagnostics["reproducibility"].get(
                "parallel_backend_preference"
            ),
        }

    def save(self, output_dir: str | Path) -> Path:
        """Write effect/parameter tables, metadata, and the Pareto plot."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.effect_usage.to_csv(output / "effect_usage.csv", index=False)
        self.parameter_usage.to_csv(output / "parameter_usage.csv", index=False)
        (output / "run_metadata.json").write_text(
            json.dumps(self.metadata(), indent=2), encoding="utf-8"
        )
        figure = make_pareto_figure(
            self.effect_usage, null_effects=self.design.null_effects
        )
        figure.savefig(output / "effect_usage_pareto.png", dpi=180, bbox_inches="tight")
        from matplotlib import pyplot as plt

        plt.close(figure)
        return output


def build_screening_design(
    data: pd.DataFrame,
    *,
    response: str,
    factors: Sequence[str],
    categorical: Sequence[str] = (),
    model: str = MODEL_INTERACTIONS,
    center_polynomials: bool = True,
    n_null_vectors: int = 0,
    null_seed: int | None = 12345,
) -> ScreeningDesign:
    """Construct a formulaic design matrix with whole-effect groups.

    With ``center_polynomials=True``, uncoded continuous factors are centered
    at their complete-case arithmetic means inside interactions and powers,
    while their main-effect columns remain raw. This reproduces JMP Fit
    Model's ``Center Polynomials(1)`` convention. Synthetic null vectors are
    appended afterward as independent standard-normal main effects, so they
    never participate in interactions or powers.
    """
    factors = tuple(dict.fromkeys(factors))
    categorical_requested = tuple(dict.fromkeys(categorical))
    if response not in data.columns:
        raise ValueError(f"response column {response!r} was not found")
    if not factors:
        raise ValueError("select at least one factor")
    if response in factors:
        raise ValueError("the response cannot also be a factor")
    missing = [name for name in factors if name not in data.columns]
    if missing:
        raise ValueError(f"factor columns were not found: {missing}")
    invalid_categorical = [
        name for name in categorical_requested if name not in factors
    ]
    if invalid_categorical:
        raise ValueError(
            f"categorical columns must be selected factors: {invalid_categorical}"
        )
    if model not in MODEL_CHOICES:
        raise ValueError(f"model must be one of {MODEL_CHOICES}")
    if not isinstance(center_polynomials, (bool, np.bool_)):
        raise TypeError("center_polynomials must be True or False")
    if isinstance(n_null_vectors, (bool, np.bool_)) or not isinstance(
        n_null_vectors, Integral
    ):
        raise TypeError("n_null_vectors must be a nonnegative integer")
    if n_null_vectors < 0:
        raise ValueError("n_null_vectors must be a nonnegative integer")

    selected = data.loc[:, [response, *factors]].copy()
    before = len(selected)
    training_rows = np.flatnonzero(selected.notna().all(axis=1).to_numpy())
    selected = selected.iloc[training_rows].reset_index(drop=True)
    if selected.empty:
        raise ValueError("no complete rows remain after dropping missing values")
    selected[response] = pd.to_numeric(selected[response], errors="raise")

    auto_categorical = [
        name for name in factors if not pd.api.types.is_numeric_dtype(selected[name])
    ]
    categorical_set = set(categorical_requested) | set(auto_categorical)
    nominal = tuple(name for name in factors if name in categorical_set)
    continuous = tuple(name for name in factors if name not in categorical_set)
    for name in nominal:
        selected[name] = selected[name].astype("category")
    for name in continuous:
        selected[name] = pd.to_numeric(selected[name], errors="raise")

    numeric = selected.loc[:, [response, *continuous]].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("response and continuous factors must contain finite values")
    constant = [
        name
        for name in continuous
        if float(selected[name].max()) == float(selected[name].min())
    ]
    if constant:
        raise ValueError(f"continuous factors must vary: {constant}")

    polynomial_order = 2 if model == MODEL_RESPONSE_SURFACE else 1
    formula = response_surface_formula(
        response,
        continuous,
        nominal,
        interaction_order=2,
        polynomial_order=polynomial_order,
    )
    y, X_raw, feature_names, groups, model_spec = _design_from_formula(
        formula, selected
    )
    centered_spec = None
    polynomial_centers: dict[str, float] = {}
    X = X_raw
    if center_polynomials:
        polynomial_centers = {name: float(selected[name].mean()) for name in continuous}
        centered = selected.copy()
        for name, center in polynomial_centers.items():
            centered[name] = centered[name] - center
        (
            centered_y,
            centered_X,
            centered_names,
            centered_groups,
            centered_spec,
        ) = _design_from_formula(formula, centered)
        if centered_names != feature_names or centered_groups != groups:
            raise RuntimeError(
                "centering changed the formulaic design structure unexpectedly"
            )
        if not np.array_equal(centered_y, y):
            raise RuntimeError("centering changed the response unexpectedly")
        X = centered_X.copy()
        # JMP leaves uncoded continuous main effects on their original scale;
        # only interaction and power construction uses centered factor values.
        for name in continuous:
            main_columns = groups.get(name)
            if main_columns is None:
                raise RuntimeError(
                    f"could not identify the continuous main effect {name!r}"
                )
            X[:, main_columns] = X_raw[:, main_columns]
    design_groups = dict(groups)
    null_effects: list[str] = []
    null_values = np.empty((before, 0))
    if n_null_vectors:
        rng = np.random.default_rng(null_seed)
        null_values = rng.standard_normal((before, n_null_vectors))
        null_matrix = null_values[training_rows]
        used_names = set(feature_names) | set(design_groups) | set(data.columns)
        for index in range(1, n_null_vectors + 1):
            name = f"{NULL_EFFECT_PREFIX} {index}"
            if name in used_names:
                name = f"{name} [synthetic]"
            suffix = 2
            while name in used_names:
                name = f"{NULL_EFFECT_PREFIX} {index} [synthetic {suffix}]"
                suffix += 1
            used_names.add(name)
            null_effects.append(name)
        first_null_column = X.shape[1]
        X = np.column_stack((X, null_matrix))
        feature_names = (*feature_names, *null_effects)
        for offset, name in enumerate(null_effects):
            design_groups[name] = (first_null_column + offset,)

    return ScreeningDesign(
        data=selected,
        y=y,
        X=X,
        feature_names=feature_names,
        groups=design_groups,
        formula=formula,
        response=response,
        factors=factors,
        categorical=nominal,
        continuous=continuous,
        center_polynomials=bool(center_polynomials),
        polynomial_centers=polynomial_centers,
        null_effects=tuple(null_effects),
        null_seed=null_seed if n_null_vectors else None,
        rows_dropped=before - len(selected),
        source_data=data.copy(deep=True),
        training_rows=tuple(int(i) for i in training_rows),
        model_spec=model_spec,
        centered_spec=centered_spec,
        null_values=null_values,
    )


def run_screening(
    data: pd.DataFrame,
    *,
    response: str,
    factors: Sequence[str],
    categorical: Sequence[str] = (),
    model: str = MODEL_INTERACTIONS,
    center_polynomials: bool = True,
    n_null_vectors: int = 0,
    method: str = "lasso",
    alphas: Sequence[float] = (1.0,),
    n_boot: int = 200,
    seed: int | None = 12345,
    n_jobs: int | None = -1,
    progress: Callable[[int, int], None] | None = None,
) -> ScreeningResult:
    """Run fixed-wAIC SVEM screening, defaulting to lasso and all CPUs.

    ``n_null_vectors`` adds reproducible standard-normal main effects using
    the analysis seed. They provide a noise-selection benchmark without
    changing the requested candidate-model expansion.
    """
    if method not in ("lasso", "elastic_net", "forward"):
        raise ValueError("method must be 'lasso', 'elastic_net', or 'forward'")
    if method == "lasso" and tuple(np.atleast_1d(alphas)) != (1.0,):
        raise ValueError("use method='elastic_net' to search mixing alphas")
    design = build_screening_design(
        data,
        response=response,
        factors=factors,
        categorical=categorical,
        model=model,
        center_polynomials=center_polynomials,
        n_null_vectors=n_null_vectors,
        null_seed=seed,
    )
    if progress is not None:
        progress(0, n_boot)
    started = time.perf_counter()
    fit_options = {
        "nBoot": n_boot,
        "objective": "wAIC",
        "seed": seed,
        "feature_names": design.feature_names,
        "n_jobs": n_jobs,
    }
    if method == "forward":
        fit = fit_svem_forward(design.X, design.y, groups=design.groups, **fit_options)
        frequencies = fit.selection_frequencies
    else:
        fit = fit_svem(design.X, design.y, alpha=alphas, **fit_options)
        nonzero = np.abs(fit.coef_matrix[:, 1:]) > 1e-8
        frequencies = {
            name: np.any(nonzero[:, columns], axis=1).mean()
            for name, columns in design.groups.items()
        }
    elapsed = time.perf_counter() - started
    if progress is not None:
        progress(fit.nBoot_used, n_boot)
    worker_count = int(fit.diagnostics["reproducibility"]["n_jobs_effective"])

    effect_usage = pd.DataFrame(
        [
            {
                "Effect": name,
                "Percent Used": 100.0 * float(frequency),
                "Parameters": len(design.groups[name]),
            }
            for name, frequency in frequencies.items()
        ]
    ).sort_values(
        ["Percent Used", "Effect"], ascending=[False, True], ignore_index=True
    )

    parameter_percent = 100.0 * (np.abs(fit.coef_matrix[:, 1:]) > 1.0e-8).mean(axis=0)
    parameter_usage = pd.DataFrame(
        {
            "Parameter": design.feature_names,
            "Percent Nonzero": parameter_percent,
            "Ensemble Coefficient": fit.coefficients[1:],
        }
    ).sort_values(
        ["Percent Nonzero", "Parameter"],
        ascending=[False, True],
        ignore_index=True,
    )
    return ScreeningResult(
        design=design,
        fit=fit,
        effect_usage=effect_usage,
        parameter_usage=parameter_usage,
        elapsed_seconds=elapsed,
        workers=worker_count,
        method=method,
        alphas=tuple(float(a) for a in np.atleast_1d(alphas))
        if method != "forward"
        else (),
    )


def make_pareto_figure(
    effect_usage: pd.DataFrame,
    *,
    null_effects: Sequence[str] = (),
):
    """Return a descending, top-to-bottom effect-use Pareto figure."""
    from matplotlib.figure import Figure

    table = effect_usage.sort_values("Percent Used", ascending=True)
    # Allocate a full text row per effect. The desktop app places this figure
    # in a vertically scrollable canvas instead of shrinking it to the window.
    height = max(4.8, min(80.0, 0.36 * len(table) + 1.8))
    figure = Figure(figsize=(10.0, height), constrained_layout=True)
    axis = figure.add_subplot(111)
    null_effect_set = set(null_effects)
    colors = [
        NULL_EFFECT_COLOR if effect in null_effect_set else MODEL_EFFECT_COLOR
        for effect in table["Effect"]
    ]
    axis.barh(table["Effect"], table["Percent Used"], color=colors)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Percent of SVEM bootstrap models")
    axis.set_ylabel("")
    axis.set_title("SVEM Variable Screening — Effect Usage")
    axis.grid(axis="x", alpha=0.25)
    axis.tick_params(axis="y", labelsize=9)
    axis.margins(y=0.01)
    if null_effect_set:
        from matplotlib.patches import Patch

        axis.legend(
            handles=[
                Patch(
                    facecolor=NULL_EFFECT_COLOR,
                    label="Synthetic null effect",
                )
            ],
            loc="lower right",
        )
    for patch, value in zip(axis.patches, table["Percent Used"]):
        numeric_value = float(value)
        inside = numeric_value >= 94.0
        axis.text(
            numeric_value - 1.0 if inside else numeric_value + 0.8,
            patch.get_y() + patch.get_height() / 2,
            f"{numeric_value:.1f}%",
            va="center",
            ha="right" if inside else "left",
            color="white" if inside else "#333333",
            fontsize=8,
        )
    return figure


__all__ = [
    "MODEL_CHOICES",
    "MODEL_EFFECT_COLOR",
    "MODEL_INTERACTIONS",
    "MODEL_RESPONSE_SURFACE",
    "NULL_EFFECT_COLOR",
    "ScreeningDesign",
    "ScreeningResult",
    "build_screening_design",
    "make_pareto_figure",
    "run_screening",
]
