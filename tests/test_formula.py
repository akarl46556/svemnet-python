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
        np.testing.assert_allclose(
            np.sort(np.abs(model.result_.coefficients)),
            np.sort(np.abs(direct.coefficients)),
            atol=1e-8,
        )

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
