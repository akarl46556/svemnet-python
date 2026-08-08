"""scikit-learn estimator interface tests, including API conformance."""

import numpy as np
import pytest
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from svemnet import SVEMRegressor


def _toy(n=50, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 1 + 2 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 0.3, n)
    return X, y


@parametrize_with_checks(
    [
        SVEMRegressor(n_boot=8, random_state=0),
        SVEMRegressor(method="forward", n_boot=8, random_state=0),
    ]
)
def test_sklearn_conformance(estimator, check):
    check(estimator)


class TestSVEMRegressor:
    def test_fit_predict_and_score(self):
        X, y = _toy()
        est = SVEMRegressor(n_boot=15, random_state=0).fit(X, y)
        assert est.score(X, y) > 0.8
        assert est.coef_.shape == (3,)
        assert np.isfinite(est.intercept_)

    def test_forward_method(self):
        X, y = _toy()
        est = SVEMRegressor(method="forward", n_boot=15, random_state=0).fit(X, y)
        assert est.score(X, y) > 0.8
        assert "selection_frequencies" in est.result_.diagnostics

    def test_reproducible_with_random_state(self):
        X, y = _toy()
        p1 = SVEMRegressor(n_boot=10, random_state=7).fit(X, y).predict(X)
        p2 = SVEMRegressor(n_boot=10, random_state=7).fit(X, y).predict(X)
        np.testing.assert_array_equal(p1, p2)

    def test_predict_interval_and_se(self):
        X, y = _toy()
        est = SVEMRegressor(n_boot=15, random_state=0).fit(X, y)
        y_pred, intervals = est.predict_interval(X, confidence_level=0.9)
        assert intervals.shape == (len(y), 2)
        assert np.all(intervals[:, 0] <= intervals[:, 1] + 1e-12)
        y_pred2, se = est.predict_se(X)
        assert np.all(se >= 0)
        np.testing.assert_allclose(y_pred, y_pred2)

    def test_pipeline_and_cross_val(self):
        X, y = _toy(n=60)
        pipe = make_pipeline(
            StandardScaler(), SVEMRegressor(n_boot=8, random_state=0)
        )
        scores = cross_val_score(pipe, X, y, cv=3)
        assert np.all(np.isfinite(scores))

    def test_groups_only_for_forward(self):
        X, y = _toy()
        est = SVEMRegressor(groups={"a": [0, 1]}, n_boot=5, random_state=0)
        with pytest.raises(ValueError, match="forward"):
            est.fit(X, y)
        ok = SVEMRegressor(
            method="forward", groups={"a": [0, 1], "b": [2]}, n_boot=5,
            random_state=0,
        ).fit(X, y)
        nz = (ok.result_.coef_matrix[:, 1:3] != 0).sum(axis=1)
        assert set(nz.tolist()) <= {0, 2}

    def test_invalid_method(self):
        X, y = _toy()
        with pytest.raises(ValueError, match="method"):
            SVEMRegressor(method="ridge", n_boot=3).fit(X, y)

    def test_debias_changes_coefficients_when_eligible(self):
        X, y = _toy()
        raw = SVEMRegressor(n_boot=15, random_state=3).fit(X, y)
        db = SVEMRegressor(n_boot=15, random_state=3, debias=True).fit(X, y)
        assert db.result_.debias_applied
        assert not np.allclose(raw.coef_, db.coef_)

    def test_generator_random_state(self):
        X, y = _toy()
        p1 = SVEMRegressor(
            n_boot=6, random_state=np.random.default_rng(11)
        ).fit(X, y).predict(X)
        p2 = SVEMRegressor(
            n_boot=6, random_state=np.random.default_rng(11)
        ).fit(X, y).predict(X)
        np.testing.assert_array_equal(p1, p2)

    def test_uncertainty_requires_two_members(self):
        X, y = _toy()
        est = SVEMRegressor(n_boot=1, random_state=0).fit(X, y)
        with pytest.raises(ValueError, match="n_boot"):
            est.predict_interval(X)
        with pytest.raises(ValueError, match="n_boot"):
            est.predict_se(X)

    def test_sklearn_named_parameter_errors(self):
        X, y = _toy()
        with pytest.raises(ValueError, match="n_boot"):
            SVEMRegressor(n_boot=1.5).fit(X, y)
        with pytest.raises(ValueError, match="alphas"):
            SVEMRegressor(alphas=(0.5, 2.0)).fit(X, y)
        with pytest.raises(ValueError, match="objective"):
            SVEMRegressor(objective="AICc").fit(X, y)

    def test_star_import_and_dir(self):
        import svemnet

        namespace = {}
        exec("from svemnet import *", namespace)
        assert "fit_svem" in namespace and "SVEMRegressor" in namespace
        assert "svem" in dir(svemnet)
        assert "forward_aicc" in dir(svemnet)
