import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNet

from svemnet import fit_lasso_cv
from svemnet.base_model import run_base_model


def data():
    rng = np.random.default_rng(40)
    X = rng.normal(size=(36, 5)) * np.array([1, 10, 0.1, 2, 3]) + 3
    y = 7 + X[:, 0] * 2 - X[:, 1] * 0.3 + rng.normal(size=36) * 0.3
    return X, y


def test_cv_final_fit_matches_independent_sklearn_refit():
    X, y = data()
    fit = fit_lasso_cv(X, y, alphas=(0.5, 1), nfolds=3, repeats=2, nlambda=25, seed=9)
    mean, scale = X.mean(axis=0), X.std(axis=0)
    oracle = ElasticNet(
        alpha=fit.selected_lambda,
        l1_ratio=fit.selected_alpha,
        tol=1e-10,
        max_iter=100000,
    ).fit((X - mean) / scale, y)
    np.testing.assert_allclose(fit.coefficients[1:], oracle.coef_ / scale, atol=2e-6)
    np.testing.assert_allclose(
        fit.predict(X), oracle.predict((X - mean) / scale), atol=3e-6
    )
    best = np.unravel_index(np.argmin(fit.cv_mean_mse), fit.cv_mean_mse.shape)
    assert fit.selected_alpha == fit.alphas[best[0]]
    assert fit.selected_lambda == fit.lambda_grid[best]


def test_cv_fold_scaling_and_heldout_mse_match_independent_refit():
    X, y = data()
    fit = fit_lasso_cv(X, y, nfolds=3, repeats=1, nlambda=8, seed=9)
    valid = fit.fold_ids[0] == 0
    train = ~valid
    mean, scale = X[train].mean(axis=0), X[train].std(axis=0)
    oracle = ElasticNet(
        alpha=fit.lambda_grid[0, 4], l1_ratio=1, tol=1e-10, max_iter=100000
    ).fit((X[train] - mean) / scale, y[train])
    loss = np.mean((y[valid] - oracle.predict((X[valid] - mean) / scale)) ** 2)
    np.testing.assert_allclose(fit.fold_mse[0, 4, 0], loss, atol=1e-6)


def test_cv_parallel_equals_serial_and_default_is_lasso():
    X, y = data()
    a = fit_lasso_cv(X, y, nfolds=3, repeats=1, nlambda=12, n_jobs=1)
    b = fit_lasso_cv(X, y, nfolds=3, repeats=1, nlambda=12, n_jobs=2)
    np.testing.assert_array_equal(a.fold_ids, b.fold_ids)
    np.testing.assert_array_equal(a.fold_mse, b.fold_mse)
    np.testing.assert_array_equal(a.coefficients, b.coefficients)
    assert a.selected_alpha == b.selected_alpha == 1
    assert b.workers == 2


def test_constant_response_and_one_se_rule():
    X, y = data()
    constant = fit_lasso_cv(X, np.ones(len(y)) * 3, nfolds=3, repeats=1)
    np.testing.assert_array_equal(constant.predict(X), np.full(len(y), 3))
    minimum = fit_lasso_cv(X, y, nfolds=3, repeats=1, nlambda=12)
    sparse = fit_lasso_cv(X, y, nfolds=3, repeats=1, nlambda=12, choose_rule="1se")
    assert sparse.selected_lambda >= minimum.selected_lambda
    with pytest.raises(ValueError, match="at least three"):
        fit_lasso_cv(X[:2], y[:2])


def test_base_workflow_standardized_coefficients_and_exports(tmp_path):
    X, y = data()
    frame = pd.DataFrame(X[:, :2], columns=["A", "B"])
    frame["Y"] = y
    result = run_base_model(
        frame, response="Y", factors=["A", "B"], nfolds=3, repeats=1, n_jobs=1
    )
    expected = result.fit.coefficients[1:] * result.design.X.std(axis=0) / y.std()
    np.testing.assert_allclose(result.fit.standardized_coefficients, expected)
    assert len(result.coefficient_table) == result.design.X.shape[1]
    np.testing.assert_allclose(result.predict_source(), result.fit.predictions)
    result.save(tmp_path)
    assert (tmp_path / "coefficients.csv").exists()
    assert (tmp_path / "cv_scores.csv").exists()


def test_formula_cv_and_public_lasso_defaults():
    import inspect

    import svemnet

    X, y = data()
    frame = pd.DataFrame(X[:, :2], columns=["A", "B"])
    frame["Y"] = y
    fit = svemnet.lasso_cv("Y ~ A + B", frame, nfolds=3, repeats=1)
    np.testing.assert_allclose(fit.predict(frame), fit.result_.predictions)
    assert "Intercept" in fit.coef_table().index
    assert inspect.signature(svemnet.fit_svem).parameters["alpha"].default == (1.0,)
    assert inspect.signature(svemnet.svem).parameters["alphas"].default == (1.0,)
    assert svemnet.SVEMRegressor().alphas == (1.0,)


def test_standardized_sizes_and_retention_survive_factor_unit_changes():
    X, y = data()
    frame = pd.DataFrame({"A": X[:, 0], "B": X[:, 1], "Y": y})
    kwargs = {
        "response": "Y",
        "factors": ["A", "B"],
        "nfolds": 3,
        "repeats": 1,
        "n_jobs": 1,
    }
    original = run_base_model(frame, **kwargs)
    frame["A"] *= 1e12
    rescaled = run_base_model(frame, **kwargs)
    np.testing.assert_allclose(
        original.predict_source(), rescaled.predict_source(), atol=1e-9
    )
    original_table = original.coefficient_table.set_index("Parameter").sort_index()
    rescaled_table = rescaled.coefficient_table.set_index("Parameter").sort_index()
    np.testing.assert_allclose(
        original_table["Standardized Coefficient"],
        rescaled_table["Standardized Coefficient"],
        atol=1e-9,
    )
    np.testing.assert_array_equal(
        original_table["Retained"], rescaled_table["Retained"]
    )
    assert rescaled_table.loc["A", "Retained"]
    assert abs(rescaled_table.loc["A", "Coefficient"]) < 1e-8
