"""Formula interface tests (requires the [formula] extra)."""

import numpy as np
import pandas as pd
import pytest

import svemnet
from svemnet.forward import fit_svem_forward, forward_select


def _toy_df(n=50, seed=7, factor=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "X1": rng.normal(size=n),
            "X2": rng.normal(size=n),
            "X3": rng.normal(size=n),
        }
    )
    y = 1 + 2 * df.X1 - 1.5 * df.X2 + 1.2 * df.X1 * df.X2
    if factor:
        df["F"] = pd.Categorical(rng.choice(["a", "b", "c"], n))
        y = y + 2.0 * (df.F == "b") - 1.5 * (df.F == "c")
    df["y"] = y + rng.normal(0, 0.25, n)
    return df


class TestFormulaSvem:
    def test_n_jobs_is_forwarded(self):
        df = _toy_df(n=30)
        model = svemnet.svem(
            "y ~ X1 + X2", df, method="forward", nBoot=4, seed=3,
            n_jobs=2,
        )
        repro = model.result_.diagnostics["reproducibility"]
        assert repro["n_jobs_requested"] == 2
        assert repro["n_jobs_effective"] == 2

    def test_matches_matrix_api(self):
        df = _toy_df()
        model = svemnet.svem(
            "y ~ (X1 + X2 + X3)**2", df, method="forward", nBoot=8, seed=5
        )
        X = np.column_stack(
            [
                df.X1, df.X2, df.X3,
                df.X1 * df.X2, df.X1 * df.X3, df.X2 * df.X3,
            ]
        )
        direct = fit_svem_forward(
            X, df.y.to_numpy(), nBoot=8, seed=5,
            feature_names=("X1", "X2", "X3", "X1:X2", "X1:X3", "X2:X3"),
        )
        # Align coefficients term-by-term (order-sensitive, sign-sensitive)
        by_name = dict(
            zip(("Intercept",) + model.feature_names_,
                model.result_.coefficients)
        )
        aligned = np.array(
            [by_name["Intercept"]]
            + [by_name[name] for name in direct.feature_names]
        )
        np.testing.assert_allclose(aligned, direct.coefficients, atol=1e-8)

    def test_factor_blocks_group_in_forward(self):
        df = _toy_df(factor=True)
        model = svemnet.svem("y ~ X1 + X2 + F", df, method="forward",
                             nBoot=10, seed=3)
        f_cols = [
            j + 1
            for j, name in enumerate(model.feature_names_)
            if name.startswith("F[")
        ]
        assert len(f_cols) == 2
        nz = (np.abs(model.result_.coef_matrix[:, f_cols]) > 0).sum(axis=1)
        assert set(nz.tolist()) <= {0, 2}
        assert model.selection_frequencies["F"] > 0.8

    def test_predict_reapplies_encodings(self):
        df = _toy_df(factor=True)
        model = svemnet.svem("y ~ X1 + F", df, nBoot=8, seed=2)
        preds_train = model.predict(df)
        assert len(preds_train) == len(df)
        new = df.iloc[:5][["X1", "F"]].copy()
        preds_new = model.predict(new)
        np.testing.assert_allclose(preds_new, preds_train[:5])

    def test_unseen_factor_level_raises(self):
        df = _toy_df(factor=True)
        model = svemnet.svem("y ~ X1 + F", df, nBoot=6, seed=2)
        new = pd.DataFrame(
            {
                "X1": [0.1, 0.2],
                "F": pd.Categorical(["a", "zz"],
                                    categories=["a", "b", "c", "zz"]),
            }
        )
        with pytest.raises(ValueError, match="not seen at fit time"):
            model.predict(new)

    def test_interval_and_coef_table(self):
        df = _toy_df()
        model = svemnet.svem("y ~ X1 + X2 + X3", df, nBoot=12, seed=1)
        out = model.predict(df, interval=True, level=0.9)
        assert set(out) == {"fit", "lwr", "upr"}
        table = model.coef_table()
        assert list(table.index)[0] == "Intercept"
        assert "pct_bootstraps_nonzero" in table.columns

    def test_selection_frequencies_requires_forward(self):
        df = _toy_df()
        model = svemnet.svem("y ~ X1 + X2", df, nBoot=5, seed=1)
        with pytest.raises(AttributeError, match="forward"):
            model.selection_frequencies

    def test_intercept_required(self):
        df = _toy_df()
        with pytest.raises(ValueError, match="intercept"):
            svemnet.svem("y ~ X1 - 1", df, nBoot=3)

    def test_predict_rejects_missing_values(self):
        df = _toy_df()
        model = svemnet.svem("y ~ X1 + X2", df, nBoot=6, seed=2)
        new = df.iloc[:4][["X1", "X2"]].copy()
        new.loc[new.index[1], "X1"] = np.nan
        with pytest.raises(ValueError, match="[Mm]issing"):
            model.predict(new)
        bench = svemnet.forward_aicc("y ~ X1 + X2", df)
        with pytest.raises(ValueError, match="[Mm]issing"):
            bench.predict(new)

    def test_unobserved_category_level_dropped_with_warning(self):
        df = _toy_df(factor=True)
        df["F"] = pd.Categorical(df["F"], categories=["a", "b", "c", "zz"])
        with pytest.warns(UserWarning, match="all-zero"):
            model = svemnet.svem("y ~ X1 + X2 + F", df, method="forward",
                                 nBoot=10, seed=3)
        # The observed levels remain selectable as a group
        assert model.selection_frequencies["F"] > 0.8
        assert not any("zz" in name for name in model.feature_names_)

    def test_intercept_only_formula(self):
        df = _toy_df()
        model = svemnet.svem("y ~ 1", df, method="forward", nBoot=2,
                             weight_scheme="Identity", seed=1)
        np.testing.assert_allclose(
            model.result_.coef_matrix[:, 0], df.y.mean(), atol=1e-8
        )
        preds = model.predict(df)
        np.testing.assert_allclose(preds, df.y.mean(), atol=1e-8)

    def test_one_sided_formula_rejected(self):
        df = _toy_df()
        with pytest.raises(ValueError, match="two-sided"):
            svemnet.svem("~ X1 + X2", df, nBoot=3)

    def test_backticked_names_fit(self):
        df = _toy_df().rename(columns={"X1": "flow-rate", "X2": "temp C"})
        formula = svemnet.response_surface_formula(
            "y", ["flow-rate", "temp C"], polynomial_order=1
        )
        assert "`flow-rate`" in formula and "`temp C`" in formula
        model = svemnet.forward_aicc(formula, df)
        assert "`flow-rate`" in model.selected_terms or "flow-rate" in str(
            model.selected_terms
        )
        coefs = model.coef_table()
        # slope on the renamed X1 main effect is still ~2
        row = [i for i in coefs.index if i.replace("`", "") == "flow-rate"]
        assert len(row) == 1
        assert abs(coefs.loc[row[0], "coefficient"] - 2.0) < 0.5

    def test_invalid_method(self):
        df = _toy_df()
        with pytest.raises(ValueError, match="method"):
            svemnet.svem("y ~ X1", df, method="ridge")


class TestFormulaForwardAicc:
    def test_matches_matrix_api_and_names(self):
        df = _toy_df()
        model = svemnet.forward_aicc("y ~ (X1 + X2 + X3)**2", df)
        assert {"X1", "X2", "X1:X2"} <= set(model.selected_terms)
        X = np.column_stack(
            [
                df.X1, df.X2, df.X3,
                df.X1 * df.X2, df.X1 * df.X3, df.X2 * df.X3,
            ]
        )
        direct = forward_select(
            X, df.y.to_numpy(),
            feature_names=("X1", "X2", "X3", "X1:X2", "X1:X3", "X2:X3"),
        )
        assert set(model.selected_terms) == set(direct.selected_terms)
        np.testing.assert_allclose(
            model.result_.criterion_value, direct.criterion_value, rtol=1e-10
        )

    def test_no_uncertainty_options(self):
        df = _toy_df()
        model = svemnet.forward_aicc("y ~ X1 + X2", df)
        with pytest.raises(TypeError, match="no bootstrap ensemble"):
            model.predict(df, interval=True)

    def test_criteria(self):
        df = _toy_df()
        for criterion in ("AICc", "AIC", "BIC"):
            model = svemnet.forward_aicc("y ~ X1 + X2 + X3", df,
                                         criterion=criterion)
            assert model.result_.criterion == criterion


class TestExpansionRoundTrip:
    def test_response_surface_formula_fits(self):
        df = _toy_df(factor=True)
        formula = svemnet.response_surface_formula(
            "y", ["X1", "X2"], ["F"], polynomial_order=2
        )
        assert formula == "y ~ (X1 + X2 + F)**2 + I(X1**2) + I(X2**2)"
        model = svemnet.forward_aicc(formula, df)
        # X1, X2, F[b], F[c], X1:X2, X1:F[b], X1:F[c], X2:F[b], X2:F[c],
        # I(X1^2), I(X2^2)
        assert len(model.feature_names_) == 11
