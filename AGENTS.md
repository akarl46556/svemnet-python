# Agent guidance for svemnet-python

Read this before making changes. It explains what this repository is, where
it came from, where it is distributed, and the invariants that must survive
any edit.

## What this is

The **Python implementation of SVEM** (Self-Validated Ensemble Models) for
small-sample design-of-experiments regression, authored and owned by
Andrew T. Karl (akarl@asu.edu), who is also the author/maintainer of the
**R package SVEMnet on CRAN**. License: MIT (the R package is GPL; both are
solely his copyright, so they intentionally differ).

Gaussian only, by design: SVEM lasso/elastic-net (`fit_svem`), SVEM forward
selection (`fit_svem_forward`), deterministic forward AICc/AIC/BIC
(`forward_select`), a scikit-learn estimator (`SVEMRegressor`), and an
optional R-style formula interface (`svemnet.svem`, `svemnet.forward_aicc`).

## Where it is distributed

- **PyPI, package name `svemnet`** (name verified unclaimed 2026-08-08; the
  first publish claims it). Releases are git tags (`v*`) published by
  `.github/workflows/release.yml` via PyPI **trusted publishing** — no API
  tokens. Setup steps and the release process live in `RELEASING.md`.
- **GitHub**: repository to be created by Andrew; `USER` placeholders in
  `pyproject.toml` and `CITATION.cff` must be replaced with the real owner
  before the first publish.
- **conda-forge**: deliberately deferred; see `RELEASING.md`.
- Cross-links: the README, PyPI Project-URLs, and CITATION.cff point to the
  CRAN R package and the Chemometrics and Intelligent Laboratory Systems
  papers (Lemkus et al. 2021 = method; Karl 2026 = software citation).

## Provenance (where the code came from)

| File | Origin |
|---|---|
| `src/svemnet/core.py` | Near-verbatim copy of `26-evaluate/src/design_compare/selection/svemnet_python_core.py` (Andrew's validated research reference). Only edits: module context and the `n_samples` error message. Keep it close to the reference so future syncs stay diffable. |
| `src/svemnet/forward.py` | Lean standalone port of `forward_aicc.py` + `svem_forward.py` from the same repo, with plain `(name, column-indices)` groups replacing the research framework's ModelMatrix/EffectGroup classes. |
| `src/svemnet/estimator.py`, `formula.py`, `expansion.py` | New for this package. |
| R counterparts | `svemnet-update/SVEMnet/R/forward_selection.R` (`forward_aicc()`, `svem_forward()`, SVEMnet ≥ 3.5.0) is a port of the same reference engines. |

Deliberate, documented deviations from the reference implementation (do not
"fix" these back): (1) plain groups instead of the ModelMatrix framework;
(2) an added plain-"AIC" criterion; (3) column equilibration inside
`_evaluate_subset` before `lstsq` (prevents rank truncation with badly
scaled columns, e.g. raw timestamps); (4) a noise-floor guard stopping
`forward_select` when the current RSS falls below
max(1e-12 × intercept-only RSS, 1e-12 × eps × response energy) — anchored to
the intercept-only RSS so large-mean responses are not truncated, with a
recorded warning when it fires (the R package uses the same rule as of
3.5.0); (5) no heredity rules, `path_minimum`, or FitResult metadata.

## The parity contract (most important invariant)

This package is **numerically validated against the R package SVEMnet
(≥ 3.5.0)** by `validation/`:

- Tier 1: deterministic forward selection matches R exactly (terms
  identical; coefficients ~1e-15; criterion values ~1e-14).
- Tier 2: `fit_svem_forward` matches R **bootstrap-by-bootstrap** (~1e-15)
  when R's FRW uniforms are injected via `weight_uniforms`.
- Tier 3: lasso ensembles agree statistically across solvers (glmnet vs
  scikit-learn); exact parity is not expected there.

**Any change touching numerics in `core.py` or `forward.py` must rerun the
harness** (`Rscript validation/run_r_reference.R validation/output` with R +
SVEMnet ≥ 3.5.0 available, then
`python validation/compare_python.py validation/output`) and keep Tiers 1–2
exact. If a numerical change is intentional on both sides, update the R
package in `svemnet-update` in the same effort.

## Scope freeze

Intentionally minimal so maintenance stays near zero. Do **not** add
binomial responses, whole-model significance testing, random-search
optimization, Thompson sampling, plotting, or new hard dependencies
(hard deps: numpy + scikit-learn; `formulaic`/`pandas` only via the
`[formula]` extra). Requests for those features are answered by pointing to
the R package. If Andrew explicitly expands scope, update this file and the
README scope section together.

## Working on this repo

- Dev interpreter: `.venv/Scripts/python` (Windows venv with the package
  installed editable + test extras).
- Tests: `./.venv/Scripts/python -m pytest tests -q` — must stay green,
  including the sklearn `parametrize_with_checks` conformance tests for both
  `method="elastic_net"` and `method="forward"`.
- Build check: `python -m build` then `twine check dist/*`.
- Version lives in **two places**: `pyproject.toml` and
  `src/svemnet/__init__.py` (`__version__`). Bump both.
- Notebooks in `examples/` are execution-verified; if you edit them, rerun
  their code cells.
- `validation/output/` is gitignored ephemeral output; the harness scripts
  themselves are versioned.
