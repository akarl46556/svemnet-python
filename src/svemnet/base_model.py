"""DataFrame workflow and reports for a single cross-validated penalized model."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .lasso import CVLassoResult, fit_lasso_cv
from .prediction import PredictionMixin
from .screening import (
    MODEL_EFFECT_COLOR,
    MODEL_INTERACTIONS,
    NULL_EFFECT_COLOR,
    ScreeningDesign,
    build_screening_design,
)


@dataclass(frozen=True)
class BaseModelResult(PredictionMixin):
    design: ScreeningDesign
    fit: CVLassoResult
    coefficient_table: pd.DataFrame
    elapsed_seconds: float
    workers: int

    def metadata(self):
        return {
            "model": "single cross-validated Gaussian lasso/elastic net",
            "response": self.design.response,
            "intercept": float(self.fit.coefficients[0]),
            "formula": self.design.formula,
            "center_polynomials": self.design.center_polynomials,
            "polynomial_centers": self.design.polynomial_centers,
            "rows_used": len(self.design.y),
            "rows_dropped": self.design.rows_dropped,
            "null_effects": list(self.design.null_effects),
            "null_vector_seed": self.design.null_seed,
            "alpha_candidates": self.fit.alphas.tolist(),
            "selected_alpha": self.fit.selected_alpha,
            "selected_lambda": self.fit.selected_lambda,
            "nfolds": self.fit.nfolds,
            "repeats": self.fit.repeats,
            "choose_rule": self.fit.choose_rule,
            "seed": self.fit.seed,
            "workers": self.workers,
            "elapsed_seconds": self.elapsed_seconds,
            "standardized_coefficient": "beta_j * SD(X_j) / SD(y), training expanded columns",
            "relaxed": False,
            "warnings": list(self.fit.warnings),
            "cv_note": "Tuning scores are not independent test-error estimates.",
        }

    def save(self, output_dir):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.coefficient_table.to_csv(output / "coefficients.csv", index=False)
        rows = [
            {
                "Alpha": float(alpha),
                "Lambda": float(lam),
                "CV Mean MSE": float(mean),
                "CV SE MSE": float(se),
            }
            for i, alpha in enumerate(self.fit.alphas)
            for lam, mean, se in zip(
                self.fit.lambda_grid[i], self.fit.cv_mean_mse[i], self.fit.cv_se_mse[i]
            )
        ]
        pd.DataFrame(rows).to_csv(output / "cv_scores.csv", index=False)
        (output / "run_metadata.json").write_text(
            json.dumps(self.metadata(), indent=2), encoding="utf-8"
        )
        figure = make_coefficient_figure(
            self.coefficient_table, null_effects=self.design.null_effects
        )
        figure.savefig(
            output / "standardized_coefficients.png", dpi=180, bbox_inches="tight"
        )
        from matplotlib import pyplot as plt

        plt.close(figure)
        return output


def run_base_model(
    data,
    *,
    response,
    factors,
    categorical=(),
    model=MODEL_INTERACTIONS,
    center_polynomials=True,
    n_null_vectors=0,
    alphas=(1.0,),
    nfolds=10,
    repeats=5,
    choose_rule="min",
    seed=12345,
    n_jobs=-1,
):
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
    started = time.perf_counter()
    fit = fit_lasso_cv(
        design.X,
        design.y,
        alphas=alphas,
        nfolds=nfolds,
        repeats=repeats,
        choose_rule=choose_rule,
        seed=seed,
        n_jobs=n_jobs,
        feature_names=design.feature_names,
    )
    table = pd.DataFrame(
        {
            "Parameter": design.feature_names,
            "Coefficient": fit.coefficients[1:],
            "Standardized Coefficient": fit.standardized_coefficients,
            "Absolute Standardized Coefficient": np.abs(fit.standardized_coefficients),
            "Retained": fit.coefficients[1:] != 0.0,
        }
    ).sort_values(
        ["Absolute Standardized Coefficient", "Parameter"],
        ascending=[False, True],
        ignore_index=True,
    )
    return BaseModelResult(
        design, fit, table, time.perf_counter() - started, fit.workers
    )


def make_coefficient_figure(table, *, null_effects=()):
    from matplotlib.figure import Figure

    table = table.sort_values("Absolute Standardized Coefficient", ascending=True)
    figure = Figure(
        figsize=(10, max(4.8, min(80, 0.36 * len(table) + 1.8))),
        constrained_layout=True,
    )
    axis = figure.add_subplot(111)
    nulls = set(null_effects)
    colors = [
        NULL_EFFECT_COLOR if name in nulls or not retained else MODEL_EFFECT_COLOR
        for name, retained in zip(table["Parameter"], table["Retained"])
    ]
    axis.barh(
        table["Parameter"], table["Absolute Standardized Coefficient"], color=colors
    )
    axis.set_title("Base Lasso / Elastic Net — Standardized Coefficients")
    axis.set_xlabel("Absolute coefficient × SD(expanded parameter) / SD(response)")
    axis.tick_params(axis="y", labelsize=9)
    axis.grid(axis="x", alpha=0.25)
    axis.margins(x=0.22, y=0.01)
    for patch, value, retained in zip(
        axis.patches, table["Absolute Standardized Coefficient"], table["Retained"]
    ):
        label = f" {value:.3g}" if retained else " not retained"
        axis.text(
            value,
            patch.get_y() + patch.get_height() / 2,
            label,
            va="center",
            fontsize=8,
        )
    return figure
