"""Tests for interval="prediction" (new-observation intervals) and the
fit-time pi_sigma/pi_df scalars. The confidence path must be bit-identical
to prior behavior."""

import numpy as np
import pytest

from svemnet.core import fit_svem, predict_svem


def _toy(seed=11, n=24, p=6, n_boot=40):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, p))
    beta = np.zeros(p)
    beta[:2] = [2.0, -1.0]
    y = 5.0 + X @ beta + rng.standard_normal(n)
    return X, y


def test_pi_scalars_populated_and_sane():
    X, y = _toy()
    res = fit_svem(X, y, nBoot=40, seed=3)
    assert res.pi_sigma is not None and res.pi_sigma > 0.0
    assert res.pi_df is not None and 1.0 <= res.pi_df <= X.shape[0]


def test_confidence_path_unchanged_by_string_alias():
    X, y = _toy()
    res = fit_svem(X, y, nBoot=30, seed=5)
    a = predict_svem(res, X[:4], interval=True, level=0.9)
    b = predict_svem(res, X[:4], interval="confidence", level=0.9)
    np.testing.assert_array_equal(a["lwr"], b["lwr"])
    np.testing.assert_array_equal(a["upr"], b["upr"])


def test_prediction_interval_wider_and_symmetric():
    X, y = _toy()
    res = fit_svem(X, y, nBoot=40, seed=7)
    c = predict_svem(res, X[:5], interval="confidence", level=0.9)
    p = predict_svem(res, X[:5], interval="prediction", level=0.9)
    assert np.all((p["upr"] - p["lwr"]) > (c["upr"] - c["lwr"]))
    np.testing.assert_allclose(
        p["upr"] - p["fit"], p["fit"] - p["lwr"], rtol=1e-10
    )


def test_prediction_interval_deterministic_given_seed():
    X, y = _toy()
    r1 = fit_svem(X, y, nBoot=25, seed=9)
    r2 = fit_svem(X, y, nBoot=25, seed=9)
    p1 = predict_svem(r1, X[:3], interval="prediction", level=0.95)
    p2 = predict_svem(r2, X[:3], interval="prediction", level=0.95)
    np.testing.assert_array_equal(p1["lwr"], p2["lwr"])
    assert r1.pi_sigma == r2.pi_sigma and r1.pi_df == r2.pi_df


def test_invalid_interval_string_rejected():
    X, y = _toy()
    res = fit_svem(X, y, nBoot=10, seed=1)
    with pytest.raises(ValueError, match="confidence"):
        predict_svem(res, X[:2], interval="banana")


def test_estimator_kind_prediction():
    from svemnet.estimator import SVEMRegressor

    X, y = _toy()
    est = SVEMRegressor(n_boot=30, random_state=2).fit(X, y)
    fit_c, iv_c = est.predict_interval(X[:4], confidence_level=0.9)
    fit_p, iv_p = est.predict_interval(X[:4], confidence_level=0.9, kind="prediction")
    assert np.all((iv_p[:, 1] - iv_p[:, 0]) > (iv_c[:, 1] - iv_c[:, 0]))
    with pytest.raises(ValueError, match="kind"):
        est.predict_interval(X[:2], kind="wrong")


def test_forward_pi_scalars_and_prediction_interval():
    from svemnet.forward import fit_svem_forward

    X, y = _toy()
    res = fit_svem_forward(X, y, nBoot=30, seed=13)
    assert res.pi_sigma is not None and res.pi_sigma > 0.0
    assert res.pi_df is not None and 1.0 <= res.pi_df <= X.shape[0]
    c = predict_svem(res, X[:4], interval="confidence", level=0.9)
    p = predict_svem(res, X[:4], interval="prediction", level=0.9)
    assert np.all((p["upr"] - p["lwr"]) > (c["upr"] - c["lwr"]))
    # determinism given seed
    res2 = fit_svem_forward(X, y, nBoot=30, seed=13)
    assert res.pi_sigma == res2.pi_sigma and res.pi_df == res2.pi_df
    np.testing.assert_array_equal(res.coef_matrix, res2.coef_matrix)
