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
The default fitter is **SVEM lasso**. Choose **SVEM elastic net search (0.5, 1)**
to compare the two mixing alphas within each bootstrap, or **SVEM forward
selection** to retain the earlier grouped forward-selection workflow.

**Center polynomials** is enabled by default. For each uncoded
continuous factor, the complete-case arithmetic mean is subtracted when
constructing interactions and powers, while the main-effect column remains on
its original scale. This matches JMP Fit Model's `Center Polynomials(1)`
convention. The saved metadata records the setting and every center used.

The optional **Null vectors** setting adds independent standard-normal noise
columns as main effects only. The default is zero. Null effects use the same
analysis seed for reproducibility, are excluded from interactions and squared
terms, and appear in grey on the Pareto chart. Their use frequencies provide a
direct visual noise benchmark for the selected experimental effects.

The results window contains a vertically scrollable, row-spaced Pareto chart
of whole-effect use, an effect-use table, and a parameter-nonzero table. The
tables, full-height chart, and run metadata can be saved together. Whole
categorical contrast blocks enter forward-selection paths as a unit. Lasso
and elastic net penalize contrasts individually: an effect is counted as
used when at least one of its contrast coefficients is nonzero. This is not
group lasso and neither coefficient sizes nor use rates are significance tests.

The equivalent command-line run is:

```bash
svem-screening run experiment.csv \
  --response Response \
  --factors Temperature Pressure Catalyst \
  --categorical Catalyst \
  --model response-surface \
  --center-polynomials \
  --null-vectors 5 \
  --bootstraps 200 \
  --jobs -1 \
  --output screening_results
```

`--jobs -1` uses the package's existing deterministic all-CPU bootstrap
implementation. Set `--jobs 1` when process startup would outweigh the fit.
Use `--no-center-polynomials` to reproduce the uncentered polynomial basis.
Omit `--null-vectors` (or set it to zero) to run without synthetic null
effects.

### Optional predictions back to the source CSV

**Save predictions back to source CSV (backup created)** is off by default
in both applications. The CLI equivalent is `--save-predictions`. It appends
a uniquely named prediction column rather than replacing an existing column.
Columns beginning with `Predicted ` are not automatically selected as factors
when reopening a CSV, to avoid feeding fitted values back into the analysis.
An exact-byte `.svemnet-<id>.bak` copy is created beside the CSV first, and
replacement is atomic. Saving is refused if the source has changed since it
was loaded. Close Excel/other editors before saving; a locked file is not
forcibly overwritten. Original field text and row order are preserved, though
CSV quoting/line formatting can be normalized; the backup retains all bytes.

Predictions use the fitted training centers and categorical encodings. Rows
with missing responses can be scored when their predictors are complete;
missing-predictor rows get blank predictions. Unseen categorical levels are
errors, never silently treated as the reference level. Synthetic null values
are held fixed for the original rows. For new data, fitted null predictors
must be supplied explicitly, not regenerated. Saved values on training rows
are fitted values, **not out-of-sample validation predictions**.

This release exports numeric predictions, not spreadsheet prediction formulas.
A portable formula workbook would need to preserve the fitted centers,
categorical contrasts, and any synthetic null inputs; it must not recalculate
those definitions from the scoring rows.

### Separate base-lasso application

```bash
svem-lasso gui
# If the launcher is not on PATH:
python -m svemnet.lasso_app gui
```

This fits a **single** Gaussian penalized model, not an SVEM ensemble. It
defaults to lasso, 10-fold CV (capped at the row count), five repetitions,
and the minimum-mean-CV-MSE rule. Elastic-net search over `(0.5, 1)` and the
conventional one-standard-error rule are optional. CV training folds learn
their own predictor scaling; mixing alphas use identical splits. Shared
lambda grids are derived from full-data standardized correlations, as in
path-based CV. CV errors are tuning diagnostics, not independent estimates
of test-set performance, and the fold-based SE is a selection heuristic.

The Pareto chart includes all candidate parameters and shows
`abs(beta_j) * SD(X_j) / SD(y)` using the expanded training design columns.
Zero coefficients are labelled **not retained**. Categorical contrasts are
shown separately; this is coding-dependent coefficient importance, not a
whole-factor significance measure. CSV exports contain signed coefficients,
standardized sizes, retention flags, and the CV score grid; the intercept
and tuning details are in `run_metadata.json`.

```bash
svem-lasso run experiment.csv --response Response \
  --factors Temperature Pressure Catalyst --categorical Catalyst \
  --method elastic_net --alphas 0.5 1 --folds 10 --repeats 5 \
  --output base_model_results
```

The numerical API is `svemnet.fit_lasso_cv(X, y, alphas=(1.0,))`; the optional
formula interface is `svemnet.lasso_cv("y ~ A + B", data)`. All public
lasso/elastic-net entry points now default to lasso (`alpha=1`); use
`alpha=(0.5, 1)` in `fit_svem`, or `alphas=(0.5, 1)` in the formula,
estimator, and CV interfaces for an explicit mixing-parameter search.

### Verification and implementation limits

Python's `pytest` serves the same testing role as R's `testthat`:

```bash
python -m pip install -e ".[test]"
python -m pytest tests -q
```

Tests cover independent sklearn refits, held-out-fold losses and scaling,
serial/parallel reproducibility, frozen categorical/centering predictions,
row alignment, and CSV backup/change/lock safeguards. CI runs on Windows,
Linux, and macOS. The existing R parity harness remains available under
`validation/`; forward fits match numerically, while lasso comparisons are
statistical across solvers.

This package uses **scikit-learn coordinate descent**, not a Python glmnet
binding. It does **not** implement glmnet's relaxed-lasso/gamma search or
post-selection relaxation. Gaussian linear debiasing in the SVEM API is a
different operation, not relaxed lasso. Use the
[R package SVEMnet](https://CRAN.R-project.org/package=SVEMnet) for relaxation
and its glmnet-backed options. The single-model CV workflow mirrors the
non-relaxed Gaussian purpose of R's `glmnet_with_cv`, not exact fold, lambda,
selected-model, or coefficient parity with R.

## Coming from R SVEMnet or JMP

| R SVEMnet | svemnet (Python) |
|---|---|
| `SVEMnet(y ~ ..., data)` | `svemnet.svem("y ~ ...", df)` |
| `svem_forward(y ~ ..., data)` | `svemnet.svem("y ~ ...", df, method="forward")` |
| `forward_aicc(y ~ ..., data)` | `svemnet.forward_aicc("y ~ ...", df)` |
| `glmnet_with_cv(y ~ ..., data)` | `svemnet.lasso_cv("y ~ ...", df)` (non-relaxed Gaussian) |
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
bootstrap uncertainty, and optional focused variable-screening and base-CV
applications with safe prediction CSV export. For relaxed lasso, binomial responses, whole-model significance
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
