from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from svemnet.screening import (
    MODEL_EFFECT_COLOR,
    MODEL_INTERACTIONS,
    MODEL_RESPONSE_SURFACE,
    NULL_EFFECT_COLOR,
    build_screening_design,
    make_pareto_figure,
    run_screening,
)
from svemnet.screening_app import build_parser


def test_screening_defaults_are_200_members_centered_no_nulls_and_all_cpus():
    signature = inspect.signature(run_screening)
    assert signature.parameters["n_boot"].default == 200
    assert signature.parameters["center_polynomials"].default is True
    assert signature.parameters["n_null_vectors"].default == 0
    assert signature.parameters["n_jobs"].default == -1
    assert signature.parameters["method"].default == "forward"
    assert signature.parameters["alphas"].default == (1.0,)
    args = build_parser().parse_args(
        ["run", "data.csv", "--response", "y", "--factors", "x"]
    )
    assert args.bootstraps == 200
    assert args.center_polynomials is True
    assert args.null_vectors == 0
    assert args.method == "forward"
    args = build_parser().parse_args(
        [
            "run",
            "data.csv",
            "--response",
            "y",
            "--factors",
            "x",
            "--no-center-polynomials",
            "--null-vectors",
            "3",
        ]
    )
    assert args.center_polynomials is False
    assert args.null_vectors == 3


def test_null_vectors_are_reproducible_main_effects_only():
    data = pd.DataFrame(
        {
            "Y": [1.0, 2.0, 4.0, 8.0, 15.0, 16.0],
            "X": [0.0, 1.0, 4.0, 10.0, 20.0, 22.0],
            "Z": [2.0, 3.0, 7.0, 11.0, 17.0, 19.0],
        }
    )
    base = build_screening_design(
        data,
        response="Y",
        factors=["X", "Z"],
        model=MODEL_RESPONSE_SURFACE,
    )
    with_nulls = build_screening_design(
        data,
        response="Y",
        factors=["X", "Z"],
        model=MODEL_RESPONSE_SURFACE,
        n_null_vectors=3,
        null_seed=91,
    )
    repeated = build_screening_design(
        data,
        response="Y",
        factors=["X", "Z"],
        model=MODEL_RESPONSE_SURFACE,
        n_null_vectors=3,
        null_seed=91,
    )
    different_seed = build_screening_design(
        data,
        response="Y",
        factors=["X", "Z"],
        model=MODEL_RESPONSE_SURFACE,
        n_null_vectors=3,
        null_seed=92,
    )

    assert with_nulls.null_effects == ("Null 1", "Null 2", "Null 3")
    assert with_nulls.feature_names[-3:] == with_nulls.null_effects
    np.testing.assert_array_equal(
        with_nulls.X[:, : base.X.shape[1]], base.X
    )
    np.testing.assert_array_equal(with_nulls.X, repeated.X)
    assert not np.array_equal(with_nulls.X[:, -3:], different_seed.X[:, -3:])
    first_null = base.X.shape[1]
    assert with_nulls.groups["Null 1"] == (first_null,)
    assert with_nulls.groups["Null 2"] == (first_null + 1,)
    assert with_nulls.groups["Null 3"] == (first_null + 2,)
    assert "Null" not in with_nulls.formula
    assert all(
        ":" not in name and "I(" not in name
        for name in with_nulls.null_effects
    )


def test_null_vector_count_must_be_a_nonnegative_integer():
    data = pd.DataFrame({"Y": [1.0, 2.0], "X": [0.0, 1.0]})
    for value in (-1,):
        with np.testing.assert_raises(ValueError):
            build_screening_design(
                data, response="Y", factors=["X"], n_null_vectors=value
            )
    for value in (True, 1.5):
        with np.testing.assert_raises(TypeError):
            build_screening_design(
                data, response="Y", factors=["X"], n_null_vectors=value
            )


def test_design_expansion_handles_categorical_groups_and_response_surface():
    data = pd.DataFrame(
        {
            "Response": [1.0, 2.0, 1.5, 3.0, 2.2, 3.4],
            "Temp": [-1, 0, 1, -1, 0, 1],
            "Catalyst": ["A", "A", "A", "B", "B", "B"],
        }
    )
    interactions = build_screening_design(
        data,
        response="Response",
        factors=["Temp", "Catalyst"],
        model=MODEL_INTERACTIONS,
    )
    surface = build_screening_design(
        data,
        response="Response",
        factors=["Temp", "Catalyst"],
        model=MODEL_RESPONSE_SURFACE,
    )
    assert interactions.categorical == ("Catalyst",)
    assert interactions.center_polynomials is True
    assert any("Temp:Catalyst" in name for name in interactions.groups)
    assert not any("I(Temp ** 2)" in name for name in interactions.groups)
    assert any("I(Temp ** 2)" in name for name in surface.groups)
    assert surface.X.shape[1] == interactions.X.shape[1] + 1


def test_polynomial_centering_matches_jmp_fit_model_convention():
    data = pd.DataFrame(
        {
            "Y": [1.0, 2.0, 4.0, 8.0, 15.0],
            "X": [0.0, 1.0, 4.0, 10.0, 20.0],
            "Z": [2.0, 3.0, 7.0, 11.0, 17.0],
        }
    )
    centered = build_screening_design(
        data,
        response="Y",
        factors=["X", "Z"],
        model=MODEL_RESPONSE_SURFACE,
        center_polynomials=True,
    )
    uncentered = build_screening_design(
        data,
        response="Y",
        factors=["X", "Z"],
        model=MODEL_RESPONSE_SURFACE,
        center_polynomials=False,
    )
    columns = {name: i for i, name in enumerate(centered.feature_names)}
    np.testing.assert_array_equal(centered.X[:, columns["X"]], data["X"])
    np.testing.assert_array_equal(centered.X[:, columns["Z"]], data["Z"])
    assert centered.X[0, columns["X:Z"]] == 42.0
    assert centered.X[0, columns["I(X ** 2)"]] == 49.0
    assert centered.polynomial_centers == {"X": 7.0, "Z": 8.0}
    assert uncentered.X[0, columns["X:Z"]] == 0.0
    assert uncentered.X[0, columns["I(X ** 2)"]] == 0.0
    assert uncentered.polynomial_centers == {}


def test_pareto_height_allocates_a_readable_row_per_effect():
    usage = pd.DataFrame(
        {
            "Effect": [f"Effect {i}" for i in range(60)],
            "Percent Used": np.linspace(100.0, 0.0, 60),
            "Parameters": np.ones(60, dtype=int),
        }
    )
    figure = make_pareto_figure(usage)
    assert figure.get_figheight() >= 23.0


def test_pareto_colors_null_effects_grey():
    usage = pd.DataFrame(
        {
            "Effect": ["Temperature", "Null 1"],
            "Percent Used": [60.0, 20.0],
            "Parameters": [1, 1],
        }
    )
    figure = make_pareto_figure(usage, null_effects=("Null 1",))
    axis = figure.axes[0]
    colors = {
        tick.get_text(): patch.get_facecolor()
        for tick, patch in zip(axis.get_yticklabels(), axis.patches)
    }
    np.testing.assert_allclose(colors["Null 1"], to_rgba(NULL_EFFECT_COLOR))
    np.testing.assert_allclose(
        colors["Temperature"], to_rgba(MODEL_EFFECT_COLOR)
    )
    assert axis.get_legend().get_texts()[0].get_text() == "Synthetic null effect"


def test_screening_uses_existing_parallel_engine_and_writes_outputs(tmp_path):
    rng = np.random.default_rng(3)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=14),
            "x1": rng.normal(size=14),
            "x2": rng.normal(size=14),
            "batch": ["A", "B"] * 7,
        }
    )
    result = run_screening(
        data,
        response="y",
        factors=["x1", "x2", "batch"],
        n_null_vectors=2,
        n_boot=4,
        seed=4,
        n_jobs=2,
        method="forward",
    )
    output = result.save(tmp_path)
    reproducibility = result.fit.diagnostics["reproducibility"]
    assert reproducibility["n_jobs_requested"] == 2
    assert reproducibility["n_jobs_effective"] == 2
    assert reproducibility["parallel_backend_preference"] == "processes"
    assert list(result.effect_usage.columns) == [
        "Effect",
        "Percent Used",
        "Parameters",
    ]
    assert result.effect_usage["Percent Used"].between(0, 100).all()
    assert set(result.design.null_effects) == {"Null 1", "Null 2"}
    assert set(result.design.null_effects).issubset(result.effect_usage["Effect"])
    assert result.parameter_usage["Percent Nonzero"].between(0, 100).all()
    for name in (
        "effect_usage.csv",
        "parameter_usage.csv",
        "run_metadata.json",
        "effect_usage_pareto.png",
    ):
        assert (output / name).stat().st_size > 0
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["objective"] == "wAIC"
    assert metadata["bootstrap_members_requested"] == 4
    assert metadata["svemnet_parallel_backend"] == "processes"
    assert metadata["center_polynomials"] is True
    assert set(metadata["polynomial_centers"]) == {"x1", "x2"}
    assert metadata["null_vector_count"] == 2
    assert metadata["null_effects"] == ["Null 1", "Null 2"]
    assert metadata["null_vector_distribution"] == "independent standard normal"
    assert metadata["null_vector_seed"] == 4
