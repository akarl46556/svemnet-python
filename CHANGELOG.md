# Changelog

## 0.2.0 (unreleased)

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
