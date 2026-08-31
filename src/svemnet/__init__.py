"""svemnet: Self-Validated Ensemble Models (SVEM) for Python.

Gaussian SVEM regression for small-sample design-of-experiments workflows:
SVEM lasso / elastic net (:func:`fit_svem`) and SVEM forward selection
(:func:`fit_svem_forward`), tuned per bootstrap by validation-weighted
information criteria, plus a deterministic forward-AICc benchmark
(:func:`forward_select`).

Interfaces
----------
* Matrix API: :func:`fit_svem`, :func:`fit_svem_forward`,
  :func:`forward_select`, :func:`fit_lasso_cv` on numeric arrays.
* scikit-learn API: :class:`SVEMRegressor` for pipelines and
  cross-validation.
* Formula API (optional, ``pip install svemnet[formula]``): :func:`svem`,
  :func:`forward_aicc`, :func:`lasso_cv` on DataFrames with R-style formulas.

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
from .lasso import CVLassoResult, fit_lasso_cv

__version__ = "0.6.1"

# The formula-interface names are provided lazily via __getattr__ and deliberately
# excluded from __all__ so `from svemnet import *` works on installs
# without the [formula] extra.
__all__ = [
    "CVLassoResult",
    "ForwardICResult",
    "SVEMForwardResult",
    "SVEMGaussianResult",
    "SVEMRegressor",
    "__version__",
    "fit_lasso_cv",
    "fit_svem",
    "fit_svem_forward",
    "forward_select",
    "kish_effective_n",
    "make_svem_weights",
    "predict_svem",
    "response_surface_formula",
    "support_size",
    "weighted_ic_scores",
]

_FORMULA_EXPORTS = {
    "svem",
    "forward_aicc",
    "lasso_cv",
    "SVEMFormulaModel",
    "ForwardAICcModel",
    "CVLassoFormulaModel",
}


def __getattr__(name: str):
    if name in _FORMULA_EXPORTS:
        try:
            from . import formula as _formula
        except ImportError as exc:
            raise AttributeError(
                f"svemnet.{name} requires the optional formula dependencies "
                "(formulaic, pandas). Install them with: "
                "pip install svemnet[formula]"
            ) from exc
        return getattr(_formula, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _FORMULA_EXPORTS)
