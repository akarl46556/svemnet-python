"""svemnet: Self-Validated Ensemble Models (SVEM) for Python.

Gaussian SVEM regression for small-sample design-of-experiments workflows:
SVEM lasso / elastic net (:func:`fit_svem`) and SVEM forward selection
(:func:`fit_svem_forward`), tuned per bootstrap by validation-weighted
information criteria, plus a deterministic forward-AICc benchmark
(:func:`forward_select`).

Interfaces
----------
* Matrix API: :func:`fit_svem`, :func:`fit_svem_forward`,
  :func:`forward_select` on numeric arrays.
* scikit-learn API: :class:`SVEMRegressor` for pipelines and
  cross-validation.
* Formula API (optional, ``pip install svemnet[formula]``): :func:`svem` and
  :func:`forward_aicc` on DataFrames with R-style formulas, mirroring the R
  package SVEMnet.

References
----------
Lemkus, Gotwalt, Ramsey, and Weese (2021), "Self-Validated Ensemble Models
for Design of Experiments", Chemometrics and Intelligent Laboratory Systems,
doi:10.1016/j.chemolab.2021.104439. Karl (2024), doi:10.1016/j.chemolab.2024.105122.
Karl (2026), "SVEMnet: ...", Chemometrics and Intelligent Laboratory Systems,
doi:10.1016/j.chemolab.2026.105660. R package: https://CRAN.R-project.org/package=SVEMnet
"""

from __future__ import annotations

from .core import (
    SVEMGaussianResult,
    fit_svem,
    kish_effective_n,
    make_svem_weights,
    predict_svem,
    support_size,
    weighted_ic_scores,
)
from .estimator import SVEMRegressor
from .expansion import response_surface_formula
from .forward import (
    ForwardICResult,
    SVEMForwardResult,
    fit_svem_forward,
    forward_select,
)

__version__ = "0.1.0"

__all__ = [
    "SVEMGaussianResult",
    "SVEMForwardResult",
    "ForwardICResult",
    "SVEMRegressor",
    "fit_svem",
    "fit_svem_forward",
    "forward_select",
    "predict_svem",
    "make_svem_weights",
    "kish_effective_n",
    "support_size",
    "weighted_ic_scores",
    "response_surface_formula",
    # formula extra (lazy):
    "svem",
    "forward_aicc",
    "SVEMFormulaModel",
    "ForwardAICcModel",
    "__version__",
]

_FORMULA_EXPORTS = {"svem", "forward_aicc", "SVEMFormulaModel", "ForwardAICcModel"}


def __getattr__(name: str):
    if name in _FORMULA_EXPORTS:
        from . import formula as _formula

        return getattr(_formula, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
