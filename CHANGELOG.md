# Changelog

## 0.7.0 (2026-10-29)

- Prevent report output names from overwriting source CSVs. Prediction write-back
  verifies immutable source bytes, fitted row/value/type identity and every
  planned destination before writing. GUI results retain their own source identity.
- Preserve literal categorical values such as `None`, `NA` and leading-zero
  labels. Blank cells are missing; conventional missing markers are interpreted
  only in otherwise numeric columns. Explicit categorical roles override inference.
  Encoding failures now explain how to save UTF-8 CSV.
- Build one final screening basis before pruning, fixing centered OFAT designs.
  Training transforms round-trip exactly; categories, centers and numerical
  center-snapping bounds are frozen. Role inference uses training rows, and
  unseen or ambiguous numeric-coded categories fail clearly.
- Internal support and usage reports use exact solver nonzeros by default
  (`coef_tol=0`). Explicit positive tolerances retain their previous raw-coefficient
  semantics. Gaussian log-loss uses a representational floor; wSSE uses raw loss.
  Forward ensembles now use the same near-perfect-fit path stop as R. Constant
  responses and numerically constant designs use structural intercept-only fits.
- Exclude columns whose variance is within the two-pass floating-point error
  bound, preventing enormous coefficients from roundoff in centered squares.
  Meaningful small-scale and large-offset variation is retained. The mask and
  omitted effects are reported; two-level powers are not removed wholesale.
- Compute affine calibration with centered least squares and reject only
  numerically unresolved prediction spread. Expose solver failure causes; all
  failed paths raise instead of silently returning an intercept-only ensemble.
- Remove unused categorical levels consistently after complete-case filtering.
  Formula fits record retained/dropped row positions, including duplicate index
  labels. A conflicting predictor named `Intercept` now gives actionable guidance.
- Give explicit elastic-net screening its intended `(0.5, 1)` default grid.
  Null controls use a separate deterministic child RNG stream: null values for a
  seed change, while the fitter's seed stream is unchanged.
- Validate integer counts and seeds consistently; support current scikit-learn
  path arguments without changing the path on older supported versions. Saved
  metadata includes versions, warning details and NumPy scalar values safely.
- Keep Kish, wAIC/wBIC penalties and the CV one-SE denominator unchanged, and
  document their heuristic interpretation and interval-coverage limitations.
- Add cross-language constant/exact/tiny-response fixtures, source-safety and
  design-schema regressions, public archive allowlists and release test/version
  gates. Private review/prototype material is excluded from distributions.

## 0.6.1

- Default the screening API, CLI and desktop application to SVEM forward
  selection; retain explicit lasso and elastic-net choices.

## 0.6.0

- Add the separate Gaussian repeated-CV lasso/elastic-net API and desktop/CLI
  application, parameter reports and opt-in recoverable CSV prediction export.
- Preserve numeric categorical roles and frozen prediction schemas when scoring
  original CSV rows.

## 0.5.0

- Add optional synthetic null controls to variable screening, with deterministic
  generation, separate displays and preserved source-row alignment.

## 0.4.0

- Add JMP-style polynomial centering to screening: centered interactions and
  powers with raw main effects and frozen training centers.

## 0.3.0

- Add the standalone variable-screening API, CLI and desktop application with
  effect/parameter usage tables, charts and saved run metadata.

## 0.2.0 (2026-08-25)

- New: deterministic, opt-in bootstrap parallelism through `n_jobs` in the
  matrix, formula, and scikit-learn SVEM interfaces. The default `n_jobs=1`
  remains a direct serial loop for JMP/embedded use; `n_jobs=-1` uses all
  available CPUs. Bootstrap weights are generated on the caller process and
  results are reassembled in bootstrap order, preserving seeded and injected-
  uniform member parity. No new hard dependency is added.

- `interval="prediction"` also works for forward-selection ensembles:
  `fit_svem_forward` now computes and stores the same `pi_sigma`/`pi_df`
  scalars (validation-weighted member residual scale; support-size df)
  from quantities already in its loop, with no change to the fit's
  random-number stream (bootstrap-by-bootstrap parity preserved).

- New: `interval="prediction"` in `predict_svem` / `SVEMGaussianResult.predict`
  and `kind="prediction"` in `SVEMRegressor.predict_interval` — an interval
  for a new observation at x, computed as
  `fit ± t_df · sqrt(sd_member² + pi_sigma²)`, where `pi_sigma` is a
  validation-weighted residual scale and `df = n − median(member support
  size)`, both stored on the fitted result (`pi_sigma`, `pi_df`) at fit time.
  `interval=True` and the new alias `interval="confidence"` keep the existing
  percentile-of-member-predictions behavior, unchanged.
- Documentation: the existing percentile interval is an ensemble-spread
  summary of member predictions for the fitted mean; it is not an interval
  for a new observation. Use `interval="prediction"` for that target.
- Internals: the fit loop now records each member's validation-weighted SSE
  at its selected path point (no additional RNG draws; bootstrap-by-bootstrap
  parity with 0.1.0 fits is preserved and re-verified by the validation
  harness).

## 0.1.0

- Initial release: SVEM lasso/elastic-net (`fit_svem`), SVEM forward
  selection (`fit_svem_forward`), deterministic forward AICc/AIC/BIC
  (`forward_select`), scikit-learn estimator (`SVEMRegressor`), optional
  R-style formula interface.
