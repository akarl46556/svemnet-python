# Maintainer release checklist

The repository is https://github.com/akarl46556/svemnet-python. Its existing
`release.yml` workflow publishes tags through PyPI trusted publishing using the
`pypi` GitHub environment; normal releases do not need a stored PyPI API token.
CI tests Python 3.10–3.13 on Linux and Python 3.13 on Windows/macOS. The monthly
canary also checks current dependencies.

## Prepare and verify

Use the declared project interpreter and machine-profile shared caches. Bump
both `pyproject.toml` and `src/svemnet/__init__.py`, update `CHANGELOG.md`, then:

```text
python -m pytest tests -q
python -m build --outdir ABSOLUTE_DIST_DIR
python -m twine check ABSOLUTE_DIST_DIR/*
python tools/check_distribution.py ABSOLUTE_DIST_DIR
```

Use a fresh profile-workspace scratch directory per candidate. The `[test,dev]`
extras provide development dependencies; no separate environment per worktree
is needed. Confirm source/version metadata, installation from the built wheel,
and the sdist/wheel file boundaries. Private reviews, diagnostics and the sole
unfinished JMP prototype are excluded by the sdist allowlist.

For numerical changes, generate references with the corresponding checked R
package and run the Python comparison using one explicit output directory:

```text
Rscript validation/run_r_reference.R ABSOLUTE_PARITY_DIR
python validation/compare_python.py ABSOLUTE_PARITY_DIR
```

The current harness requires R SVEMnet 3.6.3 or later. Tier 1 deterministic
forward and Tier 2 shared-weight forward compare numerically; Tier 3 lasso is a
statistical cross-solver comparison. Constant, exact-linear and tiny-response
fixtures exercise safeguards that ordinary noisy examples do not.

## Publish when authorized

Commit the reviewed source and wait for CI, then create and push the matching
`vX.Y.Z` tag. The release workflow repeats tests, verifies the tag against both
source version declarations, builds, runs twine and audits archive boundaries
before its publish job. Verify the workflow result and the actual PyPI files
and hashes. Local validation alone is not authorization to publish.

The owner submits the separate R package manually to CRAN.

## Optional future work

Conda-forge remains deferred. The Python package stays focused on Gaussian
SVEM and base CV workflows; the R package supplies binomial fitting, relaxation,
whole-model testing and multi-response optimization.
