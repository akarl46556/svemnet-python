from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd

from svemnet.screening import (
    MODEL_INTERACTIONS,
    MODEL_RESPONSE_SURFACE,
    build_screening_design,
    run_screening,
)


def test_screening_defaults_are_100_members_and_all_cpus():
    signature = inspect.signature(run_screening)
    assert signature.parameters["n_boot"].default == 100
    assert signature.parameters["n_jobs"].default == -1


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
    assert any("Temp:Catalyst" in name for name in interactions.groups)
    assert not any("I(Temp ** 2)" in name for name in interactions.groups)
    assert any("I(Temp ** 2)" in name for name in surface.groups)
    assert surface.X.shape[1] == interactions.X.shape[1] + 1


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
        n_boot=4,
        seed=4,
        n_jobs=2,
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
