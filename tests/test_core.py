"""Kernel and elastic-net core tests."""

import numpy as np
import pytest

from svemnet.core import (
    _FLOAT_EPS,
    fit_svem,
    kish_effective_n,
    make_svem_weights,
    predict_svem,
    support_size,
    weighted_ic_scores,
)


def _toy(n=40, seed=1, sd=0.3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 1 + 2 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, sd, n)
    return X, y


class TestWeights:
    def test_svem_weights_are_anticorrelated_and_mean_one(self):
        rng = np.random.default_rng(0)
        n = 200
        w_train, w_valid = make_svem_weights(n, rng, scheme="SVEM")
        assert w_train.shape == (n,) and w_valid.shape == (n,)
        assert np.isclose(w_train.sum(), n)
        assert np.isclose(w_valid.sum(), n)
        assert np.all(w_train > 0) and np.all(w_valid > 0)
        assert np.corrcoef(w_train, w_valid)[0, 1] < -0.5

    def test_frw_plain_uses_same_vector(self):
        rng = np.random.default_rng(0)
        w_train, w_valid = make_svem_weights(50, rng, scheme="FRW_plain")
        np.testing.assert_allclose(w_train, w_valid)

    def test_identity_weights(self):
        rng = np.random.default_rng(0)
        w_train, w_valid = make_svem_weights(10, rng, scheme="Identity")
        np.testing.assert_array_equal(w_train, np.ones(10))
        np.testing.assert_array_equal(w_valid, np.ones(10))

    def test_uniform_injection_reproduces_formulas(self):
        n = 6
        u = np.linspace(0.1, 0.9, n)
        rng = np.random.default_rng(0)
        w_train, w_valid = make_svem_weights(n, rng, scheme="SVEM", uniforms=u)
        raw_train = -np.log(u)
        raw_valid = -np.log1p(-u)
        np.testing.assert_allclose(w_train, raw_train * n / raw_train.sum())
        np.testing.assert_allclose(w_valid, raw_valid * n / raw_valid.sum())

    def test_invalid_uniforms_rejected(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            make_svem_weights(3, rng, scheme="SVEM", uniforms=[0.0, 0.5, 0.9])


class TestKish:
    def test_formula_and_clipping(self):
        w = np.array([1.0, 1.0, 4.0])
        raw, adm = kish_effective_n(w, n=3)
        assert np.isclose(raw, 36.0 / 18.0)
        assert adm == 2.0  # clipped up to the lower bound
        raw2, adm2 = kish_effective_n(np.ones(10), n=10)
        assert np.isclose(raw2, 10.0) and adm2 == 10.0


class TestScores:
    def test_waic_closed_form(self):
        sse = np.array([4.0, 2.0, 1.0])
        k = np.array([1, 2, 3])
        n_like = 10.0
        out = weighted_ic_scores(sse, k, n_like=n_like, n_eff_adm=10.0, objective="wAIC")
        expected = n_like * np.log(sse / n_like) + 2.0 * k
        np.testing.assert_allclose(out, expected)

    def test_wbic_penalty_uses_log_n_eff_adm(self):
        sse = np.array([4.0, 2.0])
        k = np.array([1, 2])
        out = weighted_ic_scores(sse, k, n_like=8.0, n_eff_adm=5.0, objective="wBIC")
        expected = 8.0 * np.log(sse / 8.0) + np.log(5.0) * k
        np.testing.assert_allclose(out, expected)

    def test_admissibility_masks_large_models(self):
        sse = np.array([1.0, 0.5])
        k = np.array([2, 6])  # k_slope = 1, 5
        out = weighted_ic_scores(sse, k, n_like=8.0, n_eff_adm=3.0, objective="wAIC")
        assert np.isfinite(out[0]) and np.isinf(out[1])

    def test_wsse_is_floored_sse(self):
        sse = np.array([0.0, 3.0])
        out = weighted_ic_scores(sse, np.array([1, 2]), n_like=5.0, n_eff_adm=5.0,
                                 objective="wSSE")
        assert out[0] == _FLOAT_EPS and out[1] == 3.0


class TestSupportSize:
    def test_counts_intercept_structurally(self):
        path = np.array([[1.0, 1.0], [0.0, 2.0], [1e-12, -3.0]])
        np.testing.assert_array_equal(support_size(path), [1, 3])


class TestFitSvem:
    def test_reproducible_given_seed(self):
        X, y = _toy()
        r1 = fit_svem(X, y, nBoot=12, seed=5)
        r2 = fit_svem(X, y, nBoot=12, seed=5)
        np.testing.assert_array_equal(r1.coef_matrix, r2.coef_matrix)
        np.testing.assert_array_equal(r1.coefficients, r2.coefficients)

    def test_recovers_strong_signal(self):
        X, y = _toy(n=60)
        result = fit_svem(X, y, nBoot=30, seed=2)
        preds = result.predict(X)
        assert np.corrcoef(preds, y)[0, 1] ** 2 > 0.9
        assert abs(result.coefficients[1] - 2.0) < 0.5
        assert abs(result.coefficients[2] + 1.5) < 0.5

    def test_predict_interval_structure(self):
        X, y = _toy()
        result = fit_svem(X, y, nBoot=15, seed=3)
        out = predict_svem(result, X, se_fit=True, interval=True, level=0.9)
        assert set(out) == {"fit", "se.fit", "lwr", "upr"}
        assert np.all(out["lwr"] <= out["upr"] + 1e-12)
        assert np.all(out["se.fit"] >= 0)

    def test_debias_eligibility_and_application(self):
        X, y = _toy()
        result = fit_svem(X, y, nBoot=15, seed=4, debias=True)
        assert result.debias_params is not None
        assert result.debias_applied
        a, b = result.debias_params
        np.testing.assert_allclose(
            result.coefficients[0], a + b * result.raw_coefficients[0]
        )
        small = fit_svem(X, y, nBoot=5, seed=4, debias=True)
        assert not small.debias_applied  # fewer than 10 members

    def test_constant_response_falls_back_to_intercept(self):
        X, _ = _toy(n=25)
        y = np.full(25, 3.0)
        result = fit_svem(X, y, nBoot=6, seed=1)
        np.testing.assert_allclose(result.coefficients[1:], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.coefficients[0], 3.0, atol=1e-8)

    def test_rejects_nonfinite(self):
        X, y = _toy()
        X[0, 0] = np.nan
        with pytest.raises(ValueError):
            fit_svem(X, y, nBoot=3)

    def test_single_bootstrap_weight_roles(self):
        """Pin the train/validation weight roles of the elastic-net core:
        the lambda path must be fit with w_train and scored with w_valid
        (plus Kish n_eff from w_valid). A role swap fails this test."""
        from svemnet.core import _weighted_enet_path

        X, y = _toy(n=30)
        n = X.shape[0]
        rng = np.random.default_rng(3)
        u = rng.uniform(0.5, 0.999, size=n)  # sharply asymmetric roles
        res = fit_svem(X, y, nBoot=1, seed=0, alpha=(1.0,),
                       weight_uniforms=u[None, :])

        w_train = -np.log(u)
        w_train *= n / w_train.sum()
        w_valid = -np.log1p(-u)
        w_valid *= n / w_valid.sum()

        np.testing.assert_allclose(
            res.n_eff_raw[0],
            w_valid.sum() ** 2 / np.dot(w_valid, w_valid),
            rtol=1e-12,
        )

        lambdas, coef_path, _ = _weighted_enet_path(
            X, y, w_train, l1_ratio=1.0, nlambda=500,
            lambda_min_ratio=1e-4, solver_tol=1e-7, max_iter=100000,
        )
        pred = X @ coef_path[1:, :] + coef_path[0:1, :]
        resid = pred - y[:, None]
        sse = np.sum(w_valid[:, None] * resid**2, axis=0)
        k = support_size(coef_path, base_tol=1e-7)
        _, n_eff_adm = kish_effective_n(w_valid, n=n)
        scores = weighted_ic_scores(
            sse, k, n_like=float(w_valid.sum()), n_eff_adm=n_eff_adm,
            objective="wAIC",
        )
        winner = int(np.nanargmin(scores))
        np.testing.assert_allclose(
            res.coef_matrix[0], coef_path[:, winner], atol=1e-10
        )

    def test_interval_level_semantics(self):
        """A level-0.5 interval must be the 25th/75th member percentiles."""
        X, y = _toy()
        result = fit_svem(X, y, nBoot=20, seed=9)
        out = predict_svem(result, X, interval=True, level=0.5)
        member = X @ result.coef_matrix[:, 1:].T + result.coef_matrix[:, 0]
        np.testing.assert_allclose(out["lwr"], np.quantile(member, 0.25, axis=1))
        np.testing.assert_allclose(out["upr"], np.quantile(member, 0.75, axis=1))

    def test_debias_uncertainty_uses_calibrated_members(self):
        """se.fit/lwr/upr under debias=True must come from calibrated
        member predictions a + b * yhat, not the raw members."""
        X, y = _toy()
        result = fit_svem(X, y, nBoot=15, seed=4, debias=True)
        assert result.debias_params is not None
        a, b = result.debias_params
        out = predict_svem(result, X, debias=True, se_fit=True,
                           interval=True, level=0.9)
        member = X @ result.coef_matrix[:, 1:].T + result.coef_matrix[:, 0]
        member_cal = a + b * member
        np.testing.assert_allclose(out["fit"], member_cal.mean(axis=1))
        np.testing.assert_allclose(out["se.fit"], member_cal.std(axis=1, ddof=1))
        np.testing.assert_allclose(out["lwr"], np.quantile(member_cal, 0.05, axis=1))
        np.testing.assert_allclose(out["upr"], np.quantile(member_cal, 0.95, axis=1))
