"""Regression checks for reviewed numerical/API stabilization."""
from functools import wraps

import numpy as np
import pandas as pd
import pytest

import svemnet.core as core
from svemnet import fit_svem, fit_svem_forward, svem
from svemnet.formula import _design_from_formula


def toy():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(30, 2))
    return x, 3 + 2 * x[:, 0] + rng.normal(0, .2, 30)


@pytest.mark.parametrize("bad", [True, np.bool_(False), 1.5, np.nan, np.inf, "3", 1j])
def test_integer_controls_reject_nonintegers(bad):
    x, y = toy()
    for fitter in (fit_svem, fit_svem_forward):
        with pytest.raises(ValueError, match="positive integer"):
            fitter(x, y, nBoot=bad)
    with pytest.raises(ValueError, match="max_iter"):
        fit_svem(x, y, nBoot=1, max_iter=bad)


def test_integer_float_and_numpy_seed_are_normalized():
    x, y = toy()
    fit = fit_svem(x, y, nBoot=2.0, max_iter=1e5, seed=np.int64(3))
    assert fit.nBoot_used == 2
    assert type(fit.seed) is int
    for seed in (True, np.bool_(True), -1, 1.0):
        with pytest.raises(ValueError, match="seed"):
            fit_svem(x, y, nBoot=1, seed=seed)


def test_large_origin_calibration_is_an_affine_regression():
    pred = 1e10 + np.arange(20)
    y = 5 + 2 * np.arange(20)
    a, b = core._linear_calibration(pred, y)
    assert b == pytest.approx(2, abs=1e-12)
    np.testing.assert_allclose(a + b * pred, y, atol=1e-10, rtol=0)
    x, y = toy()
    for fitter in (fit_svem, fit_svem_forward):
        fit = fitter(x, y + 1e10, nBoot=12, seed=2, debias=True)
        raw = fit.raw_predictions
        dx = raw - raw.mean()
        expected_b = dx @ ((y + 1e10) - (y + 1e10).mean()) / (dx @ dx)
        assert fit.debias_params[1] == pytest.approx(expected_b, abs=1e-10)
        expected = (y + 1e10).mean() + expected_b * dx
        np.testing.assert_allclose(fit.predictions, expected, atol=5e-6, rtol=0)


def test_unexpected_solver_errors_propagate(monkeypatch):
    x, y = toy()
    core._load_sklearn()
    def broken(*args, **kwargs):
        raise TypeError("simulated unsupported solver API")
    monkeypatch.setattr(core, "enet_path", broken)
    with pytest.raises(TypeError, match="unsupported solver API"):
        fit_svem(x, y, nBoot=2, seed=2)


def test_all_expected_solver_failures_raise_and_partial_failures_are_counted(monkeypatch):
    x, y = toy()
    core._load_sklearn()
    original = core.enet_path
    # Capability routing must still inspect the installed solver's signature.
    @wraps(original)
    def broken(*args, **kwargs):
        raise ValueError("simulated numerical failure")
    monkeypatch.setattr(core, "enet_path", broken)
    with pytest.raises(RuntimeError, match="Every bootstrap.*simulated numerical"):
        fit_svem(x, y, nBoot=3, seed=2)
    calls = 0
    @wraps(original)
    def partial(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("one member failed")
        return original(*args, **kwargs)
    monkeypatch.setattr(core, "enet_path", partial)
    with pytest.warns(RuntimeWarning, match="1 elastic-net path"):
        fit = fit_svem(x, y, nBoot=3, seed=2)
    assert fit.diagnostics["path_failure_count"] == 1
    assert fit.diagnostics["path_failure_member_count"] == 1
    assert any("one member failed" in item for item in fit.diagnostics["warnings"])


def test_formula_missing_rows_preserve_source_positions_and_labels():
    data = pd.DataFrame({"y": [1, np.nan, 3, 4, 5, 6], "x": [1, 2, 3, np.nan, 5, 6]},
                        index=["same"] * 6)
    with pytest.warns(UserWarning, match="Dropped 2 training row"):
        fit = svem("y ~ x", data, nBoot=2, seed=1)
    assert fit.training_row_positions_ == (0, 2, 4, 5)
    assert fit.dropped_row_positions_ == (1, 3)
    assert fit.training_row_labels_ == ("same",) * 4
    assert fit.source_row_count_ == 6
    assert len(fit.result_.predictions) == 4
    assert list(data.index) == ["same"] * 6


def test_formula_reserved_name_error_and_unpruned_design_option():
    data = pd.DataFrame({"y": np.arange(10), "Intercept": np.arange(10), "x": 0})
    with pytest.raises(ValueError, match="Rename.*numeric matrix API"):
        svem("y ~ Intercept", data, nBoot=2)
    _, design, names, _, _ = _design_from_formula("y ~ x", data, prune_zero_columns=False)
    assert design.shape == (10, 1)
    assert names == ("x",)


def test_formula_nonzero_report_uses_solver_support():
    x, y = toy()
    data = pd.DataFrame({"x": x[:, 0] * 1e9, "y": y})
    fit = svem("y ~ x", data, nBoot=3, seed=1)
    expected = np.mean(fit.result_.coef_matrix != 0, axis=0) * 100
    np.testing.assert_array_equal(fit.coef_table().pct_bootstraps_nonzero, expected)


def test_solver_grid_routing_handles_legacy_and_new_signatures():
    def legacy(X, y, *, n_alphas=100, alphas=None):
        pass
    def modern(X, y, *, n_alphas="deprecated", alphas="warn"):
        pass
    def future(X, y, *, alphas=100):
        pass
    assert core._enet_path_grid_args(legacy, 17) == {"n_alphas": 17}
    assert core._enet_path_grid_args(modern, 17) == {"alphas": 17}
    assert core._enet_path_grid_args(future, 17) == {"alphas": 17}


def test_solver_grid_compatibility_keeps_actual_path_identical():
    import inspect
    import warnings
    x, y = toy()
    core._load_sklearn()
    if "n_alphas" not in inspect.signature(core.enet_path).parameters:
        pytest.skip("legacy path-length parameter has been removed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        old = core.enet_path(x, y, l1_ratio=1, n_alphas=17)
        current = core.enet_path(x, y, l1_ratio=1,
                                **core._enet_path_grid_args(core.enet_path, 17))
    for before, after in zip(old, current):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("missing_response", [False, True])
def test_unused_reference_category_is_removed_after_complete_cases(missing_response):
    from svemnet import forward_aicc
    data = pd.DataFrame({"F": pd.Categorical(["b"] * 10 + ["c"] * 10,
                                            categories=["a", "b", "c"]),
                         "y": np.r_[np.zeros(10), np.ones(10)]})
    if missing_response:
        data.loc[20] = ["a", np.nan]
        data["F"] = pd.Categorical(data["F"], categories=["a", "b", "c"])
        data["y"] = pd.to_numeric(data["y"])
    for fitter in (lambda d: svem("y ~ F", d, method="forward", nBoot=2, seed=1),
                   lambda d: forward_aicc("y ~ F", d)):
        with pytest.warns(UserWarning) as caught:
            fit = fitter(data)
        assert any("Removed unobserved" in str(item.message) for item in caught)
        assert fit.unused_categorical_levels_ == {"F": ("a",)}
        assert fit.feature_names_ == ("F[T.c]",)
        if hasattr(fit.result_, "selection_frequencies"):
            assert fit.result_.selection_frequencies["F"] == 1.0
        else:
            assert "F" in fit.selected_terms
        with pytest.raises(ValueError, match="levels not seen"):
            fit.predict(pd.DataFrame({"F": ["a"]}))
    assert list(data["F"].cat.categories) == ["a", "b", "c"]
