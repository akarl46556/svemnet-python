"""DataFrame/CSV variable-screening workflow for SVEM forward selection."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SVEM screening requires pandas, formulaic, and matplotlib. "
        "Install them with: pip install 'svemnet[screening]'"
    ) from exc

from .expansion import response_surface_formula
from .formula import _design_from_formula
from .forward import fit_svem_forward

MODEL_INTERACTIONS = "Main effects + two-way interactions"
MODEL_RESPONSE_SURFACE = "Response surface"
MODEL_CHOICES = (MODEL_INTERACTIONS, MODEL_RESPONSE_SURFACE)


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
    rows_dropped: int


@dataclass(frozen=True)
class ScreeningResult:
    design: ScreeningDesign
    fit: object
    effect_usage: pd.DataFrame
    parameter_usage: pd.DataFrame
    elapsed_seconds: float
    workers: int

    def metadata(self) -> dict[str, object]:
        return {
            "response": self.design.response,
            "factors": list(self.design.factors),
            "categorical_factors": list(self.design.categorical),
            "continuous_factors": list(self.design.continuous),
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
        figure = make_pareto_figure(self.effect_usage)
        figure.savefig(
            output / "effect_usage_pareto.png", dpi=180, bbox_inches="tight"
        )
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
) -> ScreeningDesign:
    """Construct a formulaic design matrix with whole-effect groups."""
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

    selected = data.loc[:, [response, *factors]].copy()
    before = len(selected)
    selected = selected.dropna(axis=0, how="any").reset_index(drop=True)
    if selected.empty:
        raise ValueError("no complete rows remain after dropping missing values")
    selected[response] = pd.to_numeric(selected[response], errors="raise")

    auto_categorical = [
        name
        for name in factors
        if not pd.api.types.is_numeric_dtype(selected[name])
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

    polynomial_order = 2 if model == MODEL_RESPONSE_SURFACE else 1
    formula = response_surface_formula(
        response,
        continuous,
        nominal,
        interaction_order=2,
        polynomial_order=polynomial_order,
    )
    y, X, feature_names, groups, _ = _design_from_formula(formula, selected)
    return ScreeningDesign(
        data=selected,
        y=y,
        X=X,
        feature_names=feature_names,
        groups=dict(groups),
        formula=formula,
        response=response,
        factors=factors,
        categorical=nominal,
        continuous=continuous,
        rows_dropped=before - len(selected),
    )


def run_screening(
    data: pd.DataFrame,
    *,
    response: str,
    factors: Sequence[str],
    categorical: Sequence[str] = (),
    model: str = MODEL_INTERACTIONS,
    n_boot: int = 100,
    seed: int | None = 12345,
    n_jobs: int | None = -1,
    progress: Callable[[int, int], None] | None = None,
) -> ScreeningResult:
    """Run fixed-wAIC forward SVEM screening, using all CPUs by default."""
    design = build_screening_design(
        data,
        response=response,
        factors=factors,
        categorical=categorical,
        model=model,
    )
    if progress is not None:
        progress(0, n_boot)
    started = time.perf_counter()
    fit = fit_svem_forward(
        design.X,
        design.y,
        nBoot=n_boot,
        objective="wAIC",
        seed=seed,
        groups=design.groups,
        feature_names=design.feature_names,
        n_jobs=n_jobs,
    )
    elapsed = time.perf_counter() - started
    if progress is not None:
        progress(fit.nBoot_used, n_boot)
    worker_count = int(
        fit.diagnostics["reproducibility"]["n_jobs_effective"]
    )

    effect_usage = pd.DataFrame(
        [
            {
                "Effect": name,
                "Percent Used": 100.0 * float(frequency),
                "Parameters": len(design.groups[name]),
            }
            for name, frequency in fit.selection_frequencies.items()
        ]
    ).sort_values(
        ["Percent Used", "Effect"], ascending=[False, True], ignore_index=True
    )

    parameter_percent = 100.0 * (
        np.abs(fit.coef_matrix[:, 1:]) > 1.0e-8
    ).mean(axis=0)
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
    )


def make_pareto_figure(effect_usage: pd.DataFrame):
    """Return a descending, top-to-bottom effect-use Pareto figure."""
    from matplotlib.figure import Figure

    table = effect_usage.sort_values("Percent Used", ascending=True)
    height = max(4.0, min(12.0, 0.34 * len(table) + 1.5))
    figure = Figure(figsize=(8.0, height), constrained_layout=True)
    axis = figure.add_subplot(111)
    axis.barh(table["Effect"], table["Percent Used"], color="#2878B5")
    axis.set_xlim(0, 100)
    axis.set_xlabel("Percent of SVEM bootstrap models")
    axis.set_ylabel("")
    axis.set_title("SVEM Variable Screening — Effect Usage")
    axis.grid(axis="x", alpha=0.25)
    for patch, value in zip(axis.patches, table["Percent Used"]):
        axis.text(
            min(float(value) + 1.0, 96.0),
            patch.get_y() + patch.get_height() / 2,
            f"{float(value):.1f}%",
            va="center",
            fontsize=8,
        )
    return figure


__all__ = [
    "MODEL_CHOICES",
    "MODEL_INTERACTIONS",
    "MODEL_RESPONSE_SURFACE",
    "ScreeningDesign",
    "ScreeningResult",
    "build_screening_design",
    "make_pareto_figure",
    "run_screening",
]
