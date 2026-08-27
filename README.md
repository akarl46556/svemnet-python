# svemnet — Self-Validated Ensemble Models (SVEM) for Python

**SVEM regression for small-sample design of experiments (DOE):** lasso /
elastic-net and forward-selection ensembles tuned by validation-weighted
information criteria, with bootstrap prediction intervals.

`svemnet` is the Python implementation of the Self-Validated Ensemble Model
method (Lemkus, Gotwalt, Ramsey, and Weese 2021) by the author of the
[R package SVEMnet](https://CRAN.R-project.org/package=SVEMnet). SVEM is
designed for the small-*n*, wide-model regime typical of designed
experiments — mixture formulations, process optimization, chemometrics —
where cross-validation is unstable: instead of holding data out, every run
appears in *both* the training and validation roles through anti-correlated
fractional random weights, and predictions are ensembled across bootstrap
replicates.

## Install

```bash
pip install svemnet            # core: numpy + scikit-learn
pip install "svemnet[formula]" # + R-style formula interface (formulaic, pandas)
pip install "svemnet[screening]" # + screening GUI, CLI, tables, and plots
```

## Quickstart (formula interface, closest to R / JMP)

```python
import pandas as pd
import svemnet

df = pd.read_csv("experiment.csv")  # columns: y, X1, X2, X3, ...

# SVEM with lasso / elastic-net base learners (the SVEMnet default)
model = svemnet.svem("y ~ (X1 + X2 + X3)**2 + I(X1**2)", df, seed=1)
preds = model.predict(new_df)

# SVEM with forward-selection base learners
model = svemnet.svem("y ~ (X1 + X2 + X3)**2", df, method="forward", seed=1)
model.selection_frequencies          # per-term bootstrap selection rates
model.coef_table()                   # coefficients + % bootstraps nonzero

# Ensemble-spread percentile intervals (summary of member predictions
# for the fitted mean; not an interval for a new observation)
out = model.predict(new_df, interval=True, level=0.90)
out["fit"], out["lwr"], out["upr"]

# Prediction intervals for a NEW observation at x
out = model.predict(new_df, interval="prediction", level=0.90)

# Deterministic forward selection by AICc (single-model benchmark)
bench = svemnet.forward_aicc("y ~ (X1 + X2 + X3)**2", df)
bench.selected_terms
```

Categorical predictors: mark them as pandas `Categorical` (or leave them as
strings) and they are treatment-coded automatically, like R factors. Their
contrast columns enter and leave forward-selection paths **together** as
whole effects.

`(X1 + X2 + X3)**2` expands to main effects plus two-way interactions,
matching R's `(X1 + X2 + X3)^2`. The helper
`svemnet.response_surface_formula("y", continuous=["X1","X2"], nominal=["F"])`
builds standard response-surface formulas.

## Quickstart (scikit-learn interface)

```python
from svemnet import SVEMRegressor

est = SVEMRegressor(
    method="forward", n_boot=200, random_state=0, n_jobs=-1
)
est.fit(X, y)                        # numeric matrix, no intercept column
y_pred = est.predict(X_new)
y_pred, intervals = est.predict_interval(X_new, confidence_level=0.9)
```

`SVEMRegressor` is a conformant scikit-learn estimator: it works in
`Pipeline`, `cross_val_score`, and `GridSearchCV`, and handles categoricals
the scikit-learn way (`ColumnTransformer` / `OneHotEncoder`).

## Parallel bootstrap fitting

The bootstrap members are independent and can be fit concurrently in every
SVEM interface: pass `n_jobs=-1` to use all available CPUs or a positive
integer to set a worker limit. Process-based execution is preferred, and
results are restored to bootstrap order before aggregation, so a fixed seed
or `weight_uniforms` produces the same member sequence as serial fitting.

The default `n_jobs=1` uses the direct serial loop and does not create a
worker pool. Keep this default in JMP add-ins and other embedded Python
interpreters. For nested workflows such as `GridSearchCV(n_jobs=-1)`, use
`SVEMRegressor(n_jobs=1)` and parallelize at only the outer level. Process
startup can also outweigh computation for small ensembles, which is why
parallel fitting is opt-in.

## Standalone variable-screening application

The optional screening application runs without JMP on Windows or macOS:

```bash
python -m pip install "svemnet[screening]"
svem-screening
```

Choose a CSV file, one numeric response, and one or more factors. Text
columns are treated as categorical automatically; numeric factors can also
be marked categorical in the third selector. The candidate model is either
main effects plus all two-way interactions or a response surface (the same
terms plus squares of continuous factors). The objective is fixed at wAIC,
the default is 200 bootstrap members, and all available CPUs are used.

**Center polynomials like JMP** is enabled by default. For each uncoded
continuous factor, the complete-case arithmetic mean is subtracted when
constructing interactions and powers, while the main-effect column remains on
its original scale. This matches JMP Fit Model's `Center Polynomials(1)`
convention. The saved metadata records the setting and every center used.

The results window contains a vertically scrollable, row-spaced Pareto chart
of whole-effect use, an effect-use table, and a parameter-nonzero table. The
tables, full-height chart, and run metadata can be saved together. Whole
categorical contrast blocks enter forward-selection paths as a unit.

The equivalent command-line run is:

```bash
svem-screening run experiment.csv \
  --response Response \
  --factors Temperature Pressure Catalyst \
  --categorical Catalyst \
  --model response-surface \
  --center-polynomials \
  --bootstraps 200 \
  --jobs -1 \
  --output screening_results
```

`--jobs -1` uses the package's existing deterministic all-CPU bootstrap
implementation. Set `--jobs 1` when process startup would outweigh the fit.
Use `--no-center-polynomials` to reproduce the uncentered polynomial basis.

## Coming from R SVEMnet or JMP

| R SVEMnet | svemnet (Python) |
|---|---|
| `SVEMnet(y ~ ..., data)` | `svemnet.svem("y ~ ...", df)` |
| `svem_forward(y ~ ..., data)` | `svemnet.svem("y ~ ...", df, method="forward")` |
| `forward_aicc(y ~ ..., data)` | `svemnet.forward_aicc("y ~ ...", df)` |
| `predict(fit, newdata, se.fit=, interval=)` | `model.predict(new_df, se_fit=, interval=)` |
| `coef(fit)` / `svem_nonzero(fit)` | `model.coef_table()` |
| `bigexp_terms(...)` | `svemnet.response_surface_formula(...)` |
| `objective = "wAIC"/"wBIC"/"wSSE"` | same |
| `weight_scheme = "SVEM"/"FRW_plain"/"Identity"` | same |
| `debias = TRUE` at predict | `debias=True` at fit or predict |

The Gaussian fitting mathematics match the published SVEM conventions used
by the R package: anti-correlated FRW train/validation weights rescaled to
mean one, validation-weighted wSSE/wAIC/wBIC path selection with the Kish
effective-sample-size admissibility guardrail, intercept-only fallback,
coefficient averaging across the ensemble, and optional linear debiasing.
The forward-selection engines here and the R functions `forward_aicc()` /
`svem_forward()` (SVEMnet ≥ 3.5.0) are ports of the same reference
implementation and validate against each other numerically. The lasso /
elastic-net base learners use scikit-learn's coordinate descent rather than
glmnet, so individual path fits can differ slightly in lambda gridding;
ensemble predictions agree closely in practice.

**Scope.** This package is intentionally minimal: Gaussian responses,
lasso/elastic-net and forward-selection base learners, prediction with
bootstrap uncertainty, and an optional focused variable-screening
application. For binomial responses, whole-model significance
testing, mixture-constrained random-search optimization, and
Thompson-sampling batch design, use the
[R package SVEMnet](https://CRAN.R-project.org/package=SVEMnet); JMP Pro
users have SVEM built into Generalized Regression.

## References

- Lemkus, T., Gotwalt, C., Ramsey, P., & Weese, M. L. (2021). Self-Validated
  Ensemble Models for design of experiments. *Chemometrics and Intelligent
  Laboratory Systems*, 219, 104439.
  [doi:10.1016/j.chemolab.2021.104439](https://doi.org/10.1016/j.chemolab.2021.104439)
- Karl, A. T. (2024). A randomized permutation whole-model test for SVEM.
  *Chemometrics and Intelligent Laboratory Systems*, 249, 105122.
  [doi:10.1016/j.chemolab.2024.105122](https://doi.org/10.1016/j.chemolab.2024.105122)
- Karl, A. T. (2026). SVEMnet: Self-Validated Ensemble Models in R.
  *Chemometrics and Intelligent Laboratory Systems*.
  [doi:10.1016/j.chemolab.2026.105660](https://doi.org/10.1016/j.chemolab.2026.105660)
- Xu, L., Gotwalt, C., Hong, Y., King, C. B., & Meeker, W. Q. (2020).
  Applications of the fractional-random-weight bootstrap. *The American
  Statistician*, 74(4), 345–358.
  [doi:10.1080/00031305.2020.1731599](https://doi.org/10.1080/00031305.2020.1731599)

If you use this package in published work, please cite Lemkus et al. (2021)
for the method and Karl (2026) for the software.

## License

MIT. Copyright (c) 2026 Andrew T. Karl.
