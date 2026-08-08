"""SVEM forward-selection ensemble tests."""

import numpy as np
import pytest

from svemnet.core import _FLOAT_EPS, predict_svem
from svemnet.forward import fit_svem_forward


def _toy(n=40, seed=7, sd=0.25):
    rng = np.random.default_rng(seed)
    X1, X2, X3 = rng.normal(size=(3, n))
    y = 1 + 2 * X1 - 1.5 * X2 + 1.2 * X1 * X2 + rng.normal(0, sd, n)
    X = np.column_stack([X1, X2, X3, X1 * X2, X1 * X3, X2 * X3])
    names = ("X1", "X2", "X3", "X1:X2", "X1:X3", "X2:X3")
    return X, y, names


class TestFitSvemForward:
    def test_reproducible_given_seed(self):
        X, y, names = _toy()
        r1 = fit_svem_forward(X, y, nBoot=8, seed=5, feature_names=names)
        r2 = fit_svem_forward(X, y, nBoot=8, seed=5, feature_names=names)
        np.testing.assert_array_equal(r1.coef_matrix, r2.coef_matrix)

    def test_strong_terms_selected_frequently(self):
        X, y, names = _toy(n=60)
        result = fit_svem_forward(X, y, nBoot=20, seed=2, feature_names=names)
        freq = result.selection_frequencies
        assert freq["X1"] > 0.8 and freq["X2"] > 0.8
        preds = result.predict(X)
        assert np.corrcoef(preds, y)[0, 1] ** 2 > 0.9

    def test_single_bootstrap_oracle(self):
        """One injected-uniforms replicate must match an independent
        lstsq-based reimplementation of the grow/score/select contract."""
        X, y, names = _toy(n=30)
        n, p = X.shape
        rng = np.random.default_rng(11)
        U = rng.uniform(0.05, 0.95, size=(1, n))
        result = fit_svem_forward(
            X, y, nBoot=1, seed=0, feature_names=names, weight_uniforms=U
        )

        # Oracle: weights from the shared uniforms
        u = U[0]
        w_train = -np.log(u)
        w_valid = -np.log1p(-u)
        w_train *= n / w_train.sum()
        w_valid *= n / w_valid.sum()
        n_eff_raw = w_valid.sum() ** 2 / np.dot(w_valid, w_valid)
        n_eff_adm = min(max(n_eff_raw, 2.0), float(n))

        Xint = np.column_stack([np.ones(n), X])
        sw = np.sqrt(w_train)
        Xw, yw = Xint * sw[:, None], y * sw

        def wrss(cols):
            beta, *_ = np.linalg.lstsq(Xw[:, cols], yw, rcond=None)
            r = yw - Xw[:, cols] @ beta
            return float(r @ r), beta

        cols = [0]
        remaining = list(range(1, p + 1))
        points = [tuple(cols)]
        while remaining and (len(cols) - 1) < n_eff_adm:
            rss_list = [wrss(cols + [c])[0] for c in remaining]
            best = remaining[int(np.argmin(rss_list))]
            cols.append(best)
            remaining.remove(best)
            points.append(tuple(cols))

        scores = []
        betas = []
        for pt in points:
            _, beta = wrss(list(pt))
            betas.append(beta)
            resid = Xint[:, pt] @ beta - y
            sse = max(float(np.sum(w_valid * resid**2)), _FLOAT_EPS)
            k = len(pt)
            if (k - 1) >= n_eff_adm:
                scores.append(np.inf)
            else:
                n_like = float(w_valid.sum())
                scores.append(n_like * np.log(sse / n_like) + 2.0 * k)
        winner = int(np.argmin(scores))
        expected = np.zeros(p + 1)
        expected[list(points[winner])] = betas[winner]

        np.testing.assert_allclose(result.coef_matrix[0], expected, atol=1e-7)

    def test_waic_ceiling_binds_but_never_for_wsse(self):
        rng = np.random.default_rng(99)
        n = 15
        X = rng.normal(size=(n, 21))
        y = 1 + 2 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 0.3, n)
        r_aic = fit_svem_forward(X, y, nBoot=6, seed=3, objective="wAIC")
        r_sse = fit_svem_forward(X, y, nBoot=6, seed=3, objective="wSSE")
        assert r_aic.diagnostics["admissibility_truncation_rate"] > 0
        assert r_sse.diagnostics["admissibility_truncation_rate"] == 0
        assert (
            r_sse.diagnostics["path_length_max"]
            >= r_aic.diagnostics["path_length_max"]
        )

    def test_groups_enter_jointly_per_bootstrap(self):
        rng = np.random.default_rng(3)
        n = 60
        Fb = (rng.random(n) < 0.5).astype(float)
        Fc = ((rng.random(n) < 0.5) & (Fb == 0)).astype(float)
        X1 = rng.normal(size=n)
        y = 1 + 2 * X1 + 2 * Fb - 1.5 * Fc + rng.normal(0, 0.3, n)
        X = np.column_stack([X1, Fb, Fc])
        result = fit_svem_forward(
            X, y, nBoot=10, seed=4,
            groups={"X1": [0], "F": [1, 2]},
            feature_names=("X1", "F[b]", "F[c]"),
        )
        nz = (result.coef_matrix[:, 2:4] != 0).sum(axis=1)
        assert set(nz.tolist()) <= {0, 2}

    def test_identity_weights_single_boot_deterministic(self):
        X, y, names = _toy()
        r1 = fit_svem_forward(X, y, nBoot=1, weight_scheme="Identity",
                              feature_names=names)
        r2 = fit_svem_forward(X, y, nBoot=1, weight_scheme="Identity",
                              feature_names=names)
        np.testing.assert_array_equal(r1.coefficients, r2.coefficients)
        assert not r1.fallback_mask[0]

    def test_debias_gate(self):
        X, y, names = _toy()
        big = fit_svem_forward(X, y, nBoot=12, seed=1, debias=True)
        assert big.debias_applied and big.debias_params is not None
        small = fit_svem_forward(X, y, nBoot=5, seed=1, debias=True)
        assert not small.debias_applied

    def test_predict_uncertainty_via_shared_kernel(self):
        X, y, names = _toy()
        result = fit_svem_forward(X, y, nBoot=12, seed=6)
        out = predict_svem(result, X, se_fit=True, interval=True, level=0.9)
        assert set(out) == {"fit", "se.fit", "lwr", "upr"}
        assert np.all(out["lwr"] <= out["upr"] + 1e-12)

    def test_constant_response_intercept_only(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(25, 3))
        y = np.full(25, 5.0)
        result = fit_svem_forward(X, y, nBoot=4, seed=1)
        np.testing.assert_allclose(result.coef_matrix[:, 1:], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.coef_matrix[:, 0], 5.0, atol=1e-8)

    def test_input_validation(self):
        X, y, _ = _toy()
        with pytest.raises(ValueError):
            fit_svem_forward(X, y, objective="AICc")
        with pytest.raises(ValueError):
            fit_svem_forward(X, y, weight_scheme="bootstrap")
        with pytest.raises(ValueError):
            fit_svem_forward(X, y, nBoot=0)
