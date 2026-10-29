"""Executed Opus r16/r17 regression cases and numerical-bound invariants."""
import numpy as np
import pandas as pd
import pytest

from svemnet import fit_lasso_cv, fit_svem, fit_svem_forward
from svemnet.core import _calibration_resolvable, _numerically_constant_columns, support_size, weighted_ic_scores
from svemnet.screening import MODEL_RESPONSE_SURFACE, build_screening_design, run_screening
from svemnet.base_model import run_base_model


def inputs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    signal = 2 + 1.5*X[:, 0] - .5*X[:, 1]
    return X, signal, signal+rng.normal(0, .5, 20)


@pytest.mark.parametrize("engine", [fit_svem, fit_svem_forward])
@pytest.mark.parametrize("objective", ["wAIC", "wSSE"])
def test_non_dyadic_constant_response_is_structural(engine, objective):
    X, _, _ = inputs()
    fit = engine(X, np.full(20, 12345.678), nBoot=10, seed=1, objective=objective)
    np.testing.assert_array_equal(fit.coef_matrix[:, 1:], 0.)
    np.testing.assert_array_equal(fit.selected_k, 1)
    assert not fit.fallback_mask.any()
    assert fit.diagnostics["structural_intercept_only"]
    assert not fit.debias_applied


@pytest.mark.parametrize("objective", ["wAIC", "wSSE"])
def test_exact_linear_forward_does_not_fit_rounding_residue(objective):
    X, signal, _ = inputs()
    fit = fit_svem_forward(X, signal, nBoot=20, seed=1, objective=objective)
    np.testing.assert_array_equal(fit.selected_k, 3)
    np.testing.assert_array_equal(fit.coef_matrix[:, 3:], 0.)


@pytest.mark.parametrize("engine", [fit_svem, fit_svem_forward])
@pytest.mark.parametrize("objective", ["wAIC", "wSSE"])
def test_noisy_tiny_response_keeps_selection_and_rescaled_predictions(engine, objective):
    X, _, y = inputs()
    ordinary = engine(X, y, nBoot=12, seed=1, objective=objective)
    tiny = engine(X, y*1e-10, nBoot=12, seed=1, objective=objective)
    np.testing.assert_array_equal(tiny.selected_k, ordinary.selected_k)
    np.testing.assert_allclose(tiny.coef_matrix/1e-10, ordinary.coef_matrix, rtol=1e-6, atol=1e-7)
    assert np.max(tiny.selected_k) > 1


def test_explicit_positive_support_tolerance_keeps_legacy_raw_semantics():
    path = np.array([[1, 1], [1e-12, 1e6], [0., .01]])
    np.testing.assert_array_equal(support_size(path), [2, 3])
    np.testing.assert_array_equal(support_size(path, base_tol=1e-7), [1, 2])
    X, _, y = inputs()
    fit = fit_svem(X, y, nBoot=2, seed=1, coef_tol=1e-7)
    assert fit.coef_tol == 1e-7
    assert fit.diagnostics["support_rule"] == "historical_raw_coefficient_tolerance"
    for value in (-1, float("inf"), True):
        with pytest.raises(ValueError, match="nonnegative"):
            support_size(path, base_tol=value)


def test_log_floor_and_raw_sse_do_not_create_absolute_scale_ties():
    sse = np.array([0., 1e-25, 2e-25])
    raw = weighted_ic_scores(sse, [1, 1, 1], n_like=20, n_eff_adm=10, objective="wSSE")
    np.testing.assert_array_equal(raw, sse)
    scores = weighted_ic_scores(sse, [1, 1, 1], n_like=20, n_eff_adm=10)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores[2]-scores[1], 20*np.log(2))


def test_variance_bound_is_scale_equivariant_and_keeps_timestamp_signal():
    timestamp = 1e12 + np.arange(20)
    dust = np.resize([.3, .30000000000000004], 20)
    X = np.column_stack((timestamp, dust, np.full(20, .1)))
    for multiplier in (1e-6, 1., 1e6):
        np.testing.assert_array_equal(_numerically_constant_columns(X*multiplier), [False, True, True])


@pytest.mark.parametrize("engine", ["svem", "cv"])
def test_engine_masks_rounding_constant_columns_without_losing_schema(engine):
    rng = np.random.default_rng(21)
    a = np.repeat([.1, .3], 9)
    b = np.tile([.5, 1., 1.5], 6)
    X = np.column_stack((a, b, (a-.2)**2))
    y = 10+20*(a-.2)+3*(b-1)+rng.normal(0,.3,18)
    names = ("a", "b", "rounding_square")
    if engine == "svem":
        fit = fit_svem(X, y, nBoot=12, seed=1, alpha=(.5,1), feature_names=names)
        np.testing.assert_array_equal(fit.coef_matrix[:, -1], 0.)
        assert fit.diagnostics["numerically_constant_columns"] == ("rounding_square",)
    else:
        fit = fit_lasso_cv(X, y, alphas=(.5,1), nfolds=3, repeats=1, nlambda=20, feature_names=names)
        assert fit.numerically_constant_columns == ("rounding_square",)
    assert fit.feature_names == names
    assert fit.coefficients[-1] == 0.
    assert np.max(np.abs(fit.coefficients)) < 1000
    prediction = fit.predict(np.array([[.2, 1., 0.]]))[0]
    assert y.min() < prediction < y.max()


@pytest.mark.parametrize("engine", [fit_svem, fit_svem_forward])
def test_all_masked_design_is_structural_and_keeps_prediction_alignment(engine):
    X = np.column_stack((np.resize([.3,.30000000000000004],20),np.ones(20)))
    y = np.arange(20.)
    fit = engine(X, y, nBoot=3, seed=1, feature_names=("dust", "constant"))
    np.testing.assert_array_equal(fit.coef_matrix[:,1:],0.)
    assert fit.diagnostics["numerically_constant_columns"] == ("dust","constant")
    assert fit.diagnostics["structural_intercept_only"]
    assert fit.predict(np.array([[100.,-10.]])).shape == (1,)


def test_cv_constant_response_and_all_masked_predictors_skip_solver():
    X, _, y = inputs()
    constant = fit_lasso_cv(X,np.full(20,.1),nfolds=3,repeats=1,nlambda=10)
    np.testing.assert_array_equal(constant.coefficients[1:],0.)
    np.testing.assert_array_equal(constant.predictions,.1)
    masked = fit_lasso_cv(np.ones((20,2)),y,nfolds=3,repeats=1,nlambda=10)
    np.testing.assert_array_equal(masked.coefficients[1:],0.)


def test_timestamp_column_matches_shifted_column_predictions():
    x = np.arange(24.)
    y = 3+2*x+np.sin(x)*.1
    fit = fit_svem((1e12+x)[:,None],y,nBoot=4,seed=1)
    centered = fit_svem(x[:,None],y,nBoot=4,seed=1)
    assert not fit.diagnostics["numerically_constant_columns"]
    np.testing.assert_allclose(fit.predictions,centered.predictions,atol=.002,rtol=1e-5)


def test_screening_two_level_square_only_pruned_when_final_basis_is_constant():
    a = np.repeat([.1,.3],9)
    b = np.tile([.5,1.,1.5],6)
    data = pd.DataFrame({"y":10+20*(a-.2)+3*(b-1),"a":a,"b":b})
    with pytest.warns(UserWarning,match="numerically constant"):
        centered = build_screening_design(data,response="y",factors=["a","b"],model=MODEL_RESPONSE_SURFACE)
    assert "I(a ** 2)" not in centered.feature_names
    uncentered = build_screening_design(data,response="y",factors=["a","b"],model=MODEL_RESPONSE_SURFACE,center_polynomials=False)
    assert "I(a ** 2)" in uncentered.feature_names
    for fitfun, kwargs in ((run_screening,dict(method="elastic_net",n_boot=10)),(run_base_model,dict(alphas=(.5,1),nfolds=3,repeats=1))):
        with pytest.warns(UserWarning,match="numerically constant"):
            result=fitfun(data,response="y",factors=["a","b"],model=MODEL_RESPONSE_SURFACE,n_jobs=1,**kwargs)
        prediction=result.predict(pd.DataFrame({"a":[.2],"b":[1.]}))[0]
        assert data.y.min() <= prediction <= data.y.max()


def test_star_center_snapping_is_frozen_for_single_row_prediction():
    low, mid, high = .15,.35,.55
    points=[(low,mid),(high,mid),(mid,low),(mid,high)]*3+[(mid,mid)]*4
    a,b=np.array(points).T
    data=pd.DataFrame({"y":5+10*(a-mid)+10*(b-mid),"a":a,"b":b})
    with pytest.warns(UserWarning,match="all-zero final"):
        design=build_screening_design(data,response="y",factors=["a","b"],model=MODEL_RESPONSE_SURFACE)
    assert "a:b" not in design.feature_names
    actual,_=design.transform(data)
    np.testing.assert_array_equal(actual,design.X)
    one,_=design.transform(data.iloc[[0]])
    np.testing.assert_array_equal(one,design.X[[0]])
    assert all(value>0 for value in design.polynomial_snap_bounds.values())


def test_calibration_uses_numerical_origin_bound_not_response_scale():
    X=np.arange(-5.,6.)[:,None]
    tiny_coef=np.array([0.,1e-20])
    assert _calibration_resolvable(X[:,0]*1e-20,X,tiny_coef)
    huge_coef=np.array([1e10,1e-6])
    assert not _calibration_resolvable(huge_coef[0]+X[:,0]*huge_coef[1],X,huge_coef)
    resolvable_coef=np.array([1e10,3.])
    assert _calibration_resolvable(resolvable_coef[0]+X[:,0]*3,X,resolvable_coef)
