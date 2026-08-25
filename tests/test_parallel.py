"""Bootstrap-parallelism parity and execution-contract tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from svemnet import fit_svem, fit_svem_forward
from svemnet import _parallel


def _toy(n=28, p=5, seed=41):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 1.5 + X @ np.array([2.0, -1.0, 0.5, 0.0, 0.0])
    y += rng.normal(scale=0.25, size=n)
    return X, y


def _assert_member_parity(serial, parallel):
    numeric_arrays = (
        "coef_matrix",
        "coefficients",
        "raw_coefficients",
        "predictions",
        "raw_predictions",
        "selected_k",
        "n_eff_raw",
        "n_eff_adm",
        "fallback_mask",
    )
    for name in numeric_arrays:
        np.testing.assert_allclose(
            getattr(serial, name), getattr(parallel, name),
            rtol=1e-13, atol=1e-13, equal_nan=True,
        )
    for name in ("pi_sigma", "pi_df"):
        assert getattr(parallel, name) == pytest.approx(
            getattr(serial, name), rel=1e-13, abs=1e-13
        )

    if hasattr(serial, "selected_alpha"):
        np.testing.assert_allclose(
            serial.selected_alpha, parallel.selected_alpha,
            rtol=1e-13, atol=1e-13, equal_nan=True,
        )
        np.testing.assert_allclose(
            serial.selected_lambda, parallel.selected_lambda,
            rtol=1e-13, atol=1e-13, equal_nan=True,
        )
    else:
        assert serial.selection_frequencies == parallel.selection_frequencies

    serial_repro = dict(serial.diagnostics["reproducibility"])
    parallel_repro = dict(parallel.diagnostics["reproducibility"])
    execution_keys = {
        "serial",
        "uses_multiprocessing_or_threads",
        "n_jobs_requested",
        "n_jobs_effective",
    }
    for key in execution_keys:
        serial_repro.pop(key)
        parallel_repro.pop(key)
    assert serial_repro == parallel_repro
    assert serial.diagnostics.get("warnings", ()) == parallel.diagnostics.get(
        "warnings", ()
    )


@pytest.mark.parametrize("fitter", [fit_svem, fit_svem_forward])
@pytest.mark.parametrize("use_uniforms", [False, True])
def test_parallel_members_match_serial(fitter, use_uniforms):
    X, y = _toy()
    kwargs = {"nBoot": 5, "seed": 19}
    if use_uniforms:
        kwargs["weight_uniforms"] = np.random.default_rng(73).uniform(
            size=(5, X.shape[0])
        )
    serial = fitter(X, y, n_jobs=1, **kwargs)
    parallel = fitter(X, y, n_jobs=2, **kwargs)
    _assert_member_parity(serial, parallel)
    assert serial.diagnostics["reproducibility"]["serial"] is True
    assert parallel.diagnostics["reproducibility"]["serial"] is False
    assert parallel.diagnostics["reproducibility"]["n_jobs_effective"] == 2


def test_elastic_warning_order_matches_serial():
    X, y = _toy(n=20)
    kwargs = {
        "nBoot": 2,
        "seed": 4,
        "nlambda": 20,
        "max_iter": 1,
        "solver_tol": 1e-15,
    }
    serial = fit_svem(X, y, n_jobs=1, **kwargs)
    parallel = fit_svem(X, y, n_jobs=2, **kwargs)
    warnings_serial = serial.diagnostics["warnings"]
    assert warnings_serial
    assert warnings_serial == parallel.diagnostics["warnings"]


@pytest.mark.parametrize("fitter", [fit_svem, fit_svem_forward])
@pytest.mark.parametrize("bad", [0, True, 1.5, "2"])
def test_invalid_n_jobs_rejected(fitter, bad):
    X, y = _toy()
    with pytest.raises(ValueError, match="n_jobs"):
        fitter(X, y, nBoot=2, n_jobs=bad)


@pytest.mark.parametrize("n_jobs", [None, 1])
def test_serial_path_never_invokes_parallel_runner(monkeypatch, n_jobs):
    X, y = _toy()

    def fail(*args, **kwargs):
        raise AssertionError("parallel runner must not be called")

    monkeypatch.setattr(_parallel, "_run_parallel", fail)
    elastic = fit_svem(X, y, nBoot=2, seed=2, n_jobs=n_jobs)
    forward = fit_svem_forward(X, y, nBoot=2, seed=2, n_jobs=n_jobs)
    assert elastic.diagnostics["reproducibility"]["n_jobs_effective"] == 1
    assert forward.diagnostics["reproducibility"]["n_jobs_effective"] == 1


def test_single_task_and_worker_count_capping(monkeypatch):
    def fail():
        raise AssertionError("joblib must not be imported for one task")

    with monkeypatch.context() as context:
        context.setattr(_parallel, "_load_joblib", fail)
        single = _parallel._resolve_n_jobs(-1, n_tasks=1)
        assert single.serial and single.effective == 1

    capped = _parallel._resolve_n_jobs(99, n_tasks=3)
    assert capped.effective == 3
    all_available = _parallel._resolve_n_jobs(-1, n_tasks=3)
    assert 1 <= all_available.effective <= 3
    all_but_one = _parallel._resolve_n_jobs(-2, n_tasks=3)
    assert 1 <= all_but_one.effective <= 3


def test_parallel_failure_has_serial_guidance(monkeypatch):
    class BrokenParallel:
        def __init__(self, **kwargs):
            pass

        def __call__(self, tasks):
            raise OSError("executor unavailable")

    fake_joblib = SimpleNamespace(
        Parallel=BrokenParallel,
        delayed=lambda worker: worker,
    )
    monkeypatch.setattr(_parallel, "_load_joblib", lambda: fake_joblib)
    with pytest.raises(RuntimeError, match="n_jobs=1") as caught:
        _parallel._run_parallel(lambda: None, [()], n_jobs=2)
    assert isinstance(caught.value.__cause__, OSError)
