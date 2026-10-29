"""Frozen design and reporting regressions from the October 2026 audit."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from svemnet import screening
from svemnet.screening import MODEL_RESPONSE_SURFACE, build_screening_design, run_screening


@pytest.mark.parametrize("shift", [False, True])
def test_ofat_combines_bases_before_pruning_and_reproduces_training(shift):
    if shift:
        x = np.array([-1., 0., 1., 0., 0.]) + 10
        z = np.array([0., -1., 0., 1., 0.]) + 20
    else:
        x = np.array([0., 1., 2., 0., 0.])
        z = np.array([0., 0., 0., 1., 2.])
    data = pd.DataFrame({"y": np.arange(5.), "x": x, "z": z})
    if shift:
        with pytest.warns(UserWarning, match="all-zero final"):
            design = build_screening_design(data, response="y", factors=["x", "z"], n_null_vectors=1)
    else:
        design = build_screening_design(data, response="y", factors=["x", "z"], n_null_vectors=1)
    assert ("x:z" in design.groups) is not shift
    for name in ("x", "z"):
        np.testing.assert_array_equal(design.X[:, design.groups[name][0]], data[name])
    if not shift:
        np.testing.assert_array_equal(design.X[:, design.groups["x:z"][0]], (x-x.mean())*(z-z.mean()))
    scoring = data.assign(**{design.null_effects[0]: design.null_values[:, 0]})
    actual, rows = design.transform(scoring)
    np.testing.assert_array_equal(rows, np.arange(len(data)))
    np.testing.assert_array_equal(actual, design.X)
    assert sorted(j for group in design.groups.values() for j in group) == list(range(design.X.shape[1]))


def test_missing_response_rows_do_not_determine_predictor_role():
    data = pd.DataFrame({"y": [1., 2., 4., np.nan], "x": ["1", "2", "3", "bad"]})
    design = build_screening_design(data, response="y", factors=["x"])
    assert design.continuous == ("x",)
    assert design.training_rows == (0, 1, 2)
    with pytest.warns(UserWarning, match=r"row positions \[3\]"):
        actual, rows = design.transform(data)
    np.testing.assert_array_equal(actual, design.X)
    np.testing.assert_array_equal(rows, [0, 1, 2])
    pd.testing.assert_frame_equal(design.source_data, data)


def test_numeric_missing_tokens_are_decided_on_training_rows():
    data = pd.DataFrame({"y": [1., 2., 3., 4., np.nan], "x": ["1", "2", "NA", "3", "bad"]})
    design = build_screening_design(data, response="y", factors=["x"])
    assert design.continuous == ("x",)
    assert design.training_rows == (0, 1, 3)


def test_numeric_category_reload_uses_unambiguous_frozen_labels():
    data = pd.DataFrame({"y": [1., 2., 3., 4.], "code": ["10", "20", "10", "20"]})
    design = build_screening_design(data, response="y", factors=["code"], categorical=["code"])
    actual, rows = design.transform(pd.DataFrame({"code": [10., 20., 10., 20.]}))
    np.testing.assert_array_equal(actual, design.X)
    assert design.categorical_levels == {"code": ("10", "20")}
    with pytest.raises(ValueError, match="not seen at fit time"):
        design.transform(pd.DataFrame({"code": [30]}))
    ambiguous = build_screening_design(pd.DataFrame({"y": [1., 2., 3.], "code": ["10", "10.0", "20"]}), response="y", factors=["code"], categorical=["code"])
    with pytest.raises(ValueError, match="ambiguous"):
        ambiguous.transform(pd.DataFrame({"code": [10.]}))


def test_unused_reference_category_dropped_before_encoding():
    data = pd.DataFrame({"y": [1., 2., 3., np.nan], "code": pd.Categorical(["a", "b", "a", "unused"], categories=["unused", "a", "b"])})
    with pytest.warns(UserWarning, match="unused categories"):
        design = build_screening_design(data, response="y", factors=["code"])
    assert design.categorical_levels == {"code": ("a", "b")}
    assert design.X.shape == (3, 1)
    assert design.warnings
    with pytest.raises(ValueError, match="unused"):
        design.transform(pd.DataFrame({"code": ["unused"]}))


def test_null_vectors_use_independent_child_stream_and_preserve_source_rows():
    data = pd.DataFrame({"y": [1., 2., np.nan, 4.], "x": [1., 2., 3., 4.]})
    design = build_screening_design(data, response="y", factors=["x"], n_null_vectors=2, null_seed=np.int64(21))
    expected = np.random.default_rng(np.random.SeedSequence(21).spawn(1)[0]).standard_normal((4, 2))
    np.testing.assert_array_equal(design.null_values, expected)
    np.testing.assert_array_equal(design.X[:, -2:], expected[[0, 1, 3]])
    assert not np.array_equal(expected, np.random.default_rng(21).standard_normal((4, 2)))
    assert type(design.null_seed) is int


def test_method_defaults_and_usage_keep_exact_solver_coefficients(monkeypatch):
    calls = []
    def fake_fit(X, y, **kwargs):
        calls.append(kwargs)
        coefficients = np.r_[1., np.full(X.shape[1], 1e-12)]
        return SimpleNamespace(coef_matrix=np.tile(coefficients, (2, 1)), coefficients=coefficients,
            nBoot_used=2, nBoot_requested=2, seed=kwargs["seed"], diagnostics={"warnings": ("diagnostic",), "reproducibility": {"n_jobs_effective": 1}})
    monkeypatch.setattr(screening, "fit_svem", fake_fit)
    data = pd.DataFrame({"y": [1., 2., 4.], "x": [1., 2., 3.]})
    elastic = run_screening(data, response="y", factors=["x"], method="elastic_net", n_boot=2, seed=np.int64(7))
    lasso = run_screening(data, response="y", factors=["x"], method="lasso", n_boot=2, seed=7)
    assert tuple(calls[0]["alpha"]) == (.5, 1.)
    assert tuple(calls[1]["alpha"]) == (1.,)
    assert calls[0]["seed"] == 7 and type(calls[0]["seed"]) is int
    assert elastic.parameter_usage["Percent Nonzero"].tolist() == [100.]
    assert elastic.effect_usage["Percent Used"].tolist() == [100.]
    assert elastic.metadata()["warnings"] == ["diagnostic"]
    assert "spawn(1)" in elastic.metadata()["null_rng"]
    assert lasso.alphas == (1.,)


def test_report_preflight_blocks_before_any_output_and_normalizes_numpy(tmp_path):
    data = pd.DataFrame({"y": [1., 2., 3., 4.], "x": [1., 2., 3., 4.]})
    result = run_screening(data, response="y", factors=["x"], n_boot=2, n_jobs=1, seed=np.int64(2))
    source = tmp_path / "run_metadata.json"
    source.write_bytes(b"source bytes")
    with pytest.raises(ValueError, match="overwrite"):
        result.save(tmp_path, protected=(source,))
    assert source.read_bytes() == b"source bytes"
    assert list(tmp_path.iterdir()) == [source]
    output = tmp_path / "reports"
    result.save(output)
    import json
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["seed"] == 2
    assert metadata["svemnet_version"]


def test_quoted_factor_names_and_centered_surface_remain_predictable():
    data = pd.DataFrame({"response value": [1., 2., 3., 4.], "factor value": [1., 2., 3., 4.]})
    design = build_screening_design(data, response="response value", factors=["factor value"], model=MODEL_RESPONSE_SURFACE)
    actual, _ = design.transform(data)
    np.testing.assert_array_equal(actual, design.X)


@pytest.mark.parametrize("method", ["forward", "lasso"])
def test_constant_category_has_valid_empty_effect_tables(method):
    data = pd.DataFrame({"y": [1., 2., 3., 4.], "cat": ["a"] * 4})
    with pytest.warns(UserWarning, match="no effect was tested"):
        result = run_screening(data, response="y", factors=["cat"], n_boot=2, n_jobs=1, method=method)
    assert result.design.X.shape == (4, 0)
    assert result.design.groups == {}
    assert result.effect_usage.empty and result.parameter_usage.empty
    assert result.effect_usage.columns.tolist() == ["Effect", "Percent Used", "Parameters"]
    assert result.parameter_usage.columns.tolist() == ["Parameter", "Percent Nonzero", "Ensemble Coefficient"]
    assert any("no effect was tested" in message for message in result.metadata()["warnings"])
    assert np.isfinite(result.predict_source()).all()


def test_newly_excluded_numeric_missing_row_cannot_determine_another_role():
    data = pd.DataFrame({"y": [1., 2., 3., 4.], "x": ["1", "2", "bad", "3"], "z": ["10", "20", "NA", "30"]})
    design = build_screening_design(data, response="y", factors=["x", "z"])
    assert design.continuous == ("x", "z")
    assert design.training_rows == (0, 1, 3)
    actual, rows = design.transform(data.iloc[[0, 1, 3]])
    np.testing.assert_array_equal(actual, design.X)
