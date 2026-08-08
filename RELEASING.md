# Release checklist (for Andrew)

One-time setup, then each release is just a git tag.

## One-time setup

1. **Create the GitHub repository** (suggested name: `svemnet-python`, owner:
   your GitHub account). Then the URLs below already point at `akarl46556`:
   - `pyproject.toml` (`[project.urls]`)
   - `CITATION.cff` (`repository-code`)
2. **Push this repo**:
   ```bash
   git remote add origin https://github.com/akarl46556/svemnet-python.git
   git push -u origin main
   ```
   CI (`.github/workflows/ci.yml`) runs the test suite on Python 3.10–3.13
   (Linux) and 3.13 (Windows) on every push to `main` and on every pull
   request, plus a monthly cron canary against the latest
   numpy/scikit-learn.
3. **PyPI trusted publisher** (no API tokens needed):
   - Create/log into your account on https://pypi.org.
   - Go to *Your projects → Publishing → Add a new pending publisher* and
     register: project name `svemnet`, owner `akarl46556`, repository
     `svemnet-python`, workflow `release.yml`, environment `pypi`.
   - In the GitHub repo: *Settings → Environments → New environment* named
     `pypi` (optionally require your review before deployments).
4. **Cross-link from the R package** (optional but recommended): add a line
   to the CRAN SVEMnet DESCRIPTION/README pointing to
   `https://pypi.org/project/svemnet/`.

## Each release

```bash
# bump version in pyproject.toml and src/svemnet/__init__.py, commit, then:
git tag v0.1.0
git push origin v0.1.0
```

The `release.yml` workflow builds the sdist/wheel and publishes to PyPI via
trusted publishing.

## Re-running the R cross-validation

With the SVEMnet R package (>= 3.5.0) installed:

```bash
Rscript validation/run_r_reference.R validation/output
python validation/compare_python.py validation/output
```

Tier 1 (deterministic forward selection) and Tier 2 (SVEM forward with
injected FRW uniforms) must match R exactly; Tier 3 (lasso, cross-solver)
is statistical. See `validation/output/VALIDATION_REPORT.md`.

## Later milestones (optional, low urgency)

- **conda-forge** (once the API feels stable): submit a grayskull-generated
  recipe to conda-forge/staged-recipes; afterwards maintenance is merging
  autotick-bot PRs.
- **JOSS paper** (6–12 months out): JOSS requires ~6 months of public
  development history and demonstrated reuse; a JOSS DOI gives Python users
  a natural citation target.
- **Scope guard**: the package is deliberately minimal (Gaussian; lasso/enet
  + forward SVEM; prediction with bootstrap uncertainty). Point requests for
  binomial, whole-model testing, optimization, etc. at the R package rather
  than growing this one.
