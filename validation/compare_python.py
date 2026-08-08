"""Cross-validate the Python svemnet package against R SVEMnet reference output.

Run after ``run_r_reference.R``:

    python validation/compare_python.py [output_dir]

Three tiers:

1. Deterministic forward selection (``forward_aicc`` in R vs
   ``forward_select`` here) on identical design matrices — selected terms,
   coefficients, and criterion values must match exactly (1e-8).
2. SVEM forward selection — the Identity/nBoot=1 fit must match exactly, and
   the seeded SVEM fit must match bootstrap-by-bootstrap when R's uniform
   draws are injected through ``weight_uniforms``.
3. SVEM lasso — different solvers (glmnet vs scikit-learn coordinate
   descent), so ensemble training predictions are compared statistically.

Writes VALIDATION_REPORT.md into the output directory and exits nonzero on
any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from svemnet.core import fit_svem
from svemnet.forward import fit_svem_forward, forward_select

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("validation/output")

report: list[str] = []
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"- **{status}** {label}" + (f" — {detail}" if detail else "")
    report.append(line)
    print(("PASS " if ok else "FAIL ") + label + ("  " + detail if detail else ""))
    if not ok:
        failures.append(label)


def load_matrix(path: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path)
    return df.to_numpy(dtype=float), list(df.columns)


def load_vector(path: Path) -> np.ndarray:
    return pd.read_csv(path).iloc[:, 0].to_numpy(dtype=float)


def load_groups(path: Path) -> dict[str, tuple[int, ...]]:
    df = pd.read_csv(path)
    groups: dict[str, list[int]] = {}
    for term, col in zip(df["term"], df["column"]):
        groups.setdefault(str(term), []).append(int(col))
    return {k: tuple(v) for k, v in groups.items()}


def compare_deterministic(tag: str) -> None:
    X, names = load_matrix(OUT / f"{tag}_X.csv")
    y = load_vector(OUT / f"{tag}_y.csv")
    groups = load_groups(OUT / f"{tag}_groups.csv")
    for crit in ("AICc", "AIC", "BIC"):
        r_coef = pd.read_csv(OUT / f"{tag}_faicc_{crit}_coef.csv")
        r_terms_df = pd.read_csv(OUT / f"{tag}_faicc_{crit}_terms.csv")
        r_terms = [str(t) for t in r_terms_df["selected"]] if len(r_terms_df) else []
        r_value = load_vector(OUT / f"{tag}_faicc_{crit}_value.csv")[0]

        py = forward_select(
            X, y, criterion=crit, groups=groups, feature_names=tuple(names)
        )
        check(
            f"{tag}/{crit}: selected terms match R",
            list(py.selected_terms) == r_terms,
            f"py={list(py.selected_terms)} r={r_terms}",
        )
        r_beta = r_coef["value"].to_numpy(dtype=float)
        max_diff = float(np.max(np.abs(py.coefficients - r_beta)))
        check(
            f"{tag}/{crit}: coefficients match R",
            np.allclose(py.coefficients, r_beta, atol=1e-8, rtol=1e-7),
            f"max abs diff = {max_diff:.3e}",
        )
        crit_diff = abs(float(py.criterion_value) - float(r_value))
        check(
            f"{tag}/{crit}: criterion value matches R",
            crit_diff < 1e-8,
            f"|diff| = {crit_diff:.3e}",
        )


def compare_forward_identity(tag: str) -> None:
    X, names = load_matrix(OUT / f"{tag}_X.csv")
    y = load_vector(OUT / f"{tag}_y.csv")
    groups = load_groups(OUT / f"{tag}_groups.csv")
    r_coef, _ = load_matrix(OUT / f"{tag}_sf_identity_coef.csv")
    py = fit_svem_forward(
        X, y, nBoot=1, weight_scheme="Identity",
        groups=groups, feature_names=tuple(names),
    )
    max_diff = float(np.max(np.abs(py.coef_matrix[0] - r_coef[0])))
    check(
        f"{tag}: Identity nBoot=1 forward SVEM matches R exactly",
        np.allclose(py.coef_matrix[0], r_coef[0], atol=1e-8, rtol=1e-6),
        f"max abs diff = {max_diff:.3e}",
    )


def compare_forward_seeded(tag: str) -> None:
    X, names = load_matrix(OUT / f"{tag}_X.csv")
    y = load_vector(OUT / f"{tag}_y.csv")
    groups = load_groups(OUT / f"{tag}_groups.csv")
    U, _ = load_matrix(OUT / f"{tag}_sf_seeded_U.csv")
    r_coef, _ = load_matrix(OUT / f"{tag}_sf_seeded_coef.csv")
    py = fit_svem_forward(
        X, y, nBoot=U.shape[0], weight_scheme="SVEM", objective="wAIC",
        groups=groups, feature_names=tuple(names), weight_uniforms=U,
    )
    diffs = np.max(np.abs(py.coef_matrix - r_coef), axis=1)
    n_match = int(np.sum(diffs < 1e-6))
    check(
        f"{tag}: seeded SVEM forward matches R bootstrap-by-bootstrap "
        f"({n_match}/{len(diffs)} replicates)",
        n_match == len(diffs),
        f"max row diff = {float(diffs.max()):.3e}",
    )


def compare_lasso(tag: str) -> None:
    X, names = load_matrix(OUT / f"{tag}_X.csv")
    y = load_vector(OUT / f"{tag}_y.csv")
    r_pred = load_vector(OUT / f"{tag}_lasso_pred.csv")
    py = fit_svem(X, y, nBoot=100, alpha=(1.0,), objective="wAIC", seed=42,
                  feature_names=tuple(names))
    corr = float(np.corrcoef(py.raw_predictions, r_pred)[0, 1])
    rmse_ratio = float(
        np.sqrt(np.mean((py.raw_predictions - y) ** 2))
        / np.sqrt(np.mean((r_pred - y) ** 2))
    )
    check(
        f"{tag}: lasso SVEM training predictions agree with R (statistical)",
        corr > 0.99 and 0.8 < rmse_ratio < 1.25,
        f"corr = {corr:.5f}, RMSE ratio (py/R) = {rmse_ratio:.3f}",
    )


def main() -> int:
    report.append("# svemnet (Python) vs SVEMnet (R) validation report\n")
    report.append(
        "Tiers: (1) deterministic forward selection — exact parity; "
        "(2) SVEM forward selection — exact parity given identical FRW "
        "uniforms; (3) SVEM lasso — statistical agreement across solvers "
        "(glmnet vs scikit-learn).\n"
    )

    report.append("\n## Tier 1: deterministic forward selection (exact)\n")
    for tag in ("d1", "d2", "d3"):
        compare_deterministic(tag)

    report.append("\n## Tier 2: SVEM forward selection (exact given weights)\n")
    for tag in ("d1", "d2", "d3"):
        compare_forward_identity(tag)
        compare_forward_seeded(tag)

    report.append("\n## Tier 3: SVEM lasso (statistical)\n")
    compare_lasso("d1")

    report.append(
        f"\n**Result: {'ALL PASS' if not failures else f'{len(failures)} FAILURES'}**\n"
    )
    (OUT / "VALIDATION_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"\nReport: {OUT / 'VALIDATION_REPORT.md'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
