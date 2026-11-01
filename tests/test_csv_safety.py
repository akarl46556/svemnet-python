"""Regression checks for CSV parsing, provenance, and source-safe reports."""

import builtins
import csv
import io
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from svemnet.csvio import assert_no_protected_collision, dumps_metadata
from svemnet.prediction import CSVSnapshot, PredictionMixin
from svemnet.screening_app import ScreeningApp, build_parser, launch_gui, run_cli


SCREENING_OUTPUTS = (
    "effect_usage.csv", "parameter_usage.csv", "run_metadata.json", "effect_usage_pareto.png"
)
BASE_OUTPUTS = (
    "coefficients.csv", "cv_scores.csv", "run_metadata.json", "standardized_coefficients.png"
)
REPORT_CASES = [(False, name) for name in SCREENING_OUTPUTS] + [
    (True, name) for name in BASE_OUTPUTS
]


class CapturedFit(PredictionMixin):
    """An already fitted result; no numerical engine needed for provenance checks."""

    def __init__(self, frame):
        self.design = SimpleNamespace(source_data=frame.copy(deep=True), response="Y")

    def predict_source(self):
        return np.arange(len(self.design.source_data), dtype=float)


def _mutate(frame, mutation):
    if mutation == "drop":
        frame.drop(index=1, inplace=True)
    elif mutation == "add":
        frame.loc[3] = [4, 7]
    elif mutation == "reorder":
        frame.sort_values("A", ascending=False, inplace=True)
    elif mutation == "reorder_reset":
        frame.sort_values("A", ascending=False, inplace=True)
        frame.reset_index(drop=True, inplace=True)
    elif mutation == "value":
        frame.loc[0, "A"] = 100
    elif mutation == "rename":
        frame.rename(columns={"A": "B"}, inplace=True)
    elif mutation == "dtype":
        frame["A"] = frame["A"].astype(float)
    elif mutation == "index_metadata":
        frame.index.name = "run"


@pytest.mark.parametrize("mutation", [
    "drop", "add", "reorder", "reorder_reset", "value", "rename", "dtype", "index_metadata"
])
@pytest.mark.parametrize("before_fit", [False, True])
def test_mutated_snapshot_refused_before_any_write(tmp_path, mutation, before_fit):
    path = tmp_path / "data.csv"
    payload = b"Y,A\n1,2\n2,4\n3,6\n"
    path.write_bytes(payload)
    snapshot = CSVSnapshot.load(path)
    if before_fit:
        _mutate(snapshot.data, mutation)
    fitted = CapturedFit(snapshot.data)
    if not before_fit:
        _mutate(snapshot.data, mutation)
    with pytest.raises(ValueError, match="changed"):
        fitted.save_predictions_csv(snapshot)
    assert path.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [path]


def test_fitted_frame_compared_against_bytes_not_mutable_convenience(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("Y,A\n1,2\n2,4\n3,6\n", encoding="utf-8")
    snapshot = CSVSnapshot.load(path)
    fitted = CapturedFit(snapshot.data)
    fitted.design.source_data.loc[0, "A"] = 100
    with pytest.raises(ValueError, match="not the data"):
        fitted.save_predictions_csv(snapshot)
    assert list(tmp_path.iterdir()) == [path]


def test_direct_append_uses_immutable_raw_row_count(tmp_path):
    path = tmp_path / "data.csv"
    payload = b"Y,A\n1,2\n2,4\n3,6\n"
    path.write_bytes(payload)
    snapshot = CSVSnapshot.load(path)
    snapshot.data.drop(index=2, inplace=True)
    with pytest.raises(ValueError, match="row-aligned"):
        snapshot.append_predictions([1, 2], "Prediction")
    with pytest.raises(ValueError, match="changed"):
        snapshot.append_predictions([1, 2, 3], "Prediction")
    assert path.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [path]


def test_csv_na_policy_and_explicit_raw_categorical_roles(tmp_path):
    path = tmp_path / "roles.csv"
    path.write_text(
        "Y,Additive,Code,AllTokens\n1,None,001,NA\n2,Salt,1,None\nNA,NA,NA,NULL\n4,,,\n",
        encoding="utf-8",
    )
    snapshot = CSVSnapshot.load(path)
    assert snapshot.data["Additive"].iloc[:3].tolist() == ["None", "Salt", "NA"]
    assert snapshot.data["AllTokens"].iloc[:3].tolist() == ["NA", "None", "NULL"]
    assert np.isnan(snapshot.data["Y"].iloc[2])
    assert snapshot.data["Code"].iloc[:2].tolist() == [1, 1]
    assert snapshot.parse_report() == {"Y": 1, "Code": 1}
    literal = snapshot.frame(text_columns=("Code",))
    assert literal["Code"].iloc[:3].tolist() == ["001", "1", "NA"]
    assert pd.isna(literal["Code"].iloc[3])
    fitted = CapturedFit(literal)
    with pytest.raises(ValueError, match="not the data"):
        fitted.save_predictions_csv(snapshot)
    payload = path.read_bytes()
    saved = fitted.save_predictions_csv(snapshot, text_columns=("Code",))
    assert Path(saved["backup"]).read_bytes() == payload
    original_rows = list(csv.reader(io.StringIO(payload.decode())))
    saved_rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
    assert [row[:-1] for row in saved_rows] == original_rows


def test_load_text_policy_is_reused_for_export(tmp_path):
    path = tmp_path / "roles.csv"
    path.write_text("Y,Code\n1,001\n2,1\n3,NA\n", encoding="utf-8")
    snapshot = CSVSnapshot.load(path, text_columns=("Code",))
    assert snapshot.data["Code"].tolist() == ["001", "1", "NA"]
    CapturedFit(snapshot.data).save_predictions_csv(snapshot)
    assert len(list(tmp_path.glob("*.bak"))) == 1


def test_non_utf8_csv_has_actionable_error(tmp_path):
    path = tmp_path / "latin.csv"
    path.write_bytes("Y,Factor\n1,caf\xe9\n".encode("cp1252"))
    with pytest.raises(ValueError, match="save as CSV UTF-8"):
        CSVSnapshot.load(path)


@pytest.fixture(scope="module")
def report_results():
    from svemnet.base_model import run_base_model
    from svemnet.screening import run_screening

    rng = np.random.default_rng(92)
    frame = pd.DataFrame({"Y": rng.normal(size=12), "A": rng.normal(size=12)})
    return {
        False: run_screening(frame, response="Y", factors=["A"], n_boot=2, n_jobs=1),
        True: run_base_model(frame, response="Y", factors=["A"], nfolds=2, repeats=1, n_jobs=1),
    }


def _source(path):
    path.write_text("Y,A\n1,2\n2,4\n3,6\n4,7\n5,9\n6,10\n", encoding="utf-8")
    return path.read_bytes()


@pytest.mark.parametrize("base_model,name", REPORT_CASES)
@pytest.mark.parametrize("save_predictions", [False, True])
def test_cli_all_report_collisions_preserve_source(
    tmp_path, monkeypatch, report_results, base_model, name, save_predictions
):
    from svemnet import base_model as base_api, screening

    path = tmp_path / name
    payload = _source(path)
    result = report_results[base_model]
    api, function = (base_api, "run_base_model") if base_model else (screening, "run_screening")
    monkeypatch.setattr(api, function, lambda *_args, **_kwargs: result)
    argv = ["run", str(path), "--response", "Y", "--factors", "A", "--output", str(tmp_path)]
    if save_predictions:
        argv.append("--save-predictions")
    args = build_parser(base_model=base_model).parse_args(argv)
    with pytest.raises(ValueError, match="overwrite source data"):
        run_cli(args, base_model=base_model)
    assert path.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("base_model,name", REPORT_CASES)
@pytest.mark.parametrize("protect_fit_source", [False, True])
@pytest.mark.parametrize("save_predictions", [False, True])
def test_gui_all_report_collisions_protect_fit_and_current_sources(
    tmp_path, monkeypatch, report_results, base_model, name, protect_fit_source, save_predictions
):
    pytest.importorskip("tkinter")
    from tkinter import filedialog, messagebox

    collision = tmp_path / name
    other = tmp_path / "other_input.csv"
    _source(collision)
    _source(other)
    fit_path, current_path = (collision, other) if protect_fit_source else (other, collision)
    if save_predictions:
        CSVSnapshot.load(fit_path).append_predictions([1, 2, 3, 4, 5, 6], "Predicted Y")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    app = ScreeningApp.__new__(ScreeningApp)
    app.source_snapshot = CSVSnapshot.load(current_path)
    app.file_var = SimpleNamespace(get=lambda: str(current_path))
    errors, notices = [], []
    monkeypatch.setattr(filedialog, "askdirectory", lambda **_: str(tmp_path))
    monkeypatch.setattr(messagebox, "showerror", lambda *args: errors.append(args))
    monkeypatch.setattr(messagebox, "showinfo", lambda *args: notices.append(args))
    app._save_results(report_results[base_model], source=fit_path)
    assert errors and "overwrite source data" in errors[0][1]
    assert not notices
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("base_model", [False, True])
def test_hardlink_report_alias_is_refused(tmp_path, report_results, base_model):
    path = tmp_path / "source.csv"
    payload = _source(path)
    alias = tmp_path / (BASE_OUTPUTS if base_model else SCREENING_OUTPUTS)[-1]
    try:
        os.link(path, alias)
    except OSError:
        pytest.skip("hard links unavailable")
    with pytest.raises(ValueError, match="overwrite source data"):
        report_results[base_model].save(tmp_path, protected=(path,))
    assert path.read_bytes() == payload and alias.read_bytes() == payload
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.skipif(os.name != "nt", reason="Windows case-insensitive path identity")
def test_windows_case_variant_is_protected(tmp_path):
    path = tmp_path / "Data.csv"
    _source(path)
    with pytest.raises(ValueError, match="overwrite source data"):
        assert_no_protected_collision([tmp_path / "DATA.CSV"], [path])


@pytest.mark.parametrize("base_model", [False, True])
def test_noncolliding_reports_save_normally(tmp_path, report_results, base_model):
    path = tmp_path / "input.csv"
    payload = _source(path)
    output = tmp_path / "reports"
    report_results[base_model].save(output, protected=(path,))
    assert path.read_bytes() == payload
    assert set(p.name for p in output.iterdir()) == set(BASE_OUTPUTS if base_model else SCREENING_OUTPUTS)
    json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))


def test_numpy_scalar_json_preserves_types_and_rejects_unknowns():
    encoded = dumps_metadata({"seed": np.int64(12), "score": np.float32(1.5), "flag": np.bool_(True)})
    assert json.loads(encoded) == {"seed": 12, "score": 1.5, "flag": True}
    with pytest.raises(TypeError):
        dumps_metadata({"unsupported": object()})


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission-mode behavior")
def test_atomic_csv_replace_preserves_posix_mode(tmp_path):
    path = tmp_path / "data.csv"
    _source(path)
    path.chmod(0o640)
    CSVSnapshot.load(path).append_predictions(np.arange(6), "Prediction")
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_missing_tk_is_actionable(monkeypatch):
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "tkinter":
            raise ImportError("no tkinter")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(SystemExit, match="Tk support.*command-line"):
        launch_gui()


@pytest.mark.parametrize("save_predictions", [False, True])
def test_gui_fit_carries_original_source_and_literal_role_policy(
    tmp_path, monkeypatch, save_predictions
):
    import queue
    from tkinter import messagebox
    from svemnet import screening_app

    original = tmp_path / "fit.csv"
    original.write_text("Y,Code\n1,001\n2,1\n3,NA\n", encoding="utf-8")
    current = tmp_path / "current.csv"
    current.write_text("Y,Code\n4,2\n5,3\n6,4\n", encoding="utf-8")
    snapshot = CSVSnapshot.load(original)
    original_payload, current_payload = original.read_bytes(), current.read_bytes()
    errors, shown = [], []
    monkeypatch.setattr(messagebox, "showerror", lambda *args: errors.append(args))
    monkeypatch.setattr(messagebox, "showwarning", lambda *args: errors.append(args))
    monkeypatch.setattr(messagebox, "showinfo", lambda *args: None)

    def variable(value):
        return SimpleNamespace(get=lambda: value)

    def selected(values):
        return SimpleNamespace(curselection=lambda: range(len(values)), get=values.__getitem__)

    def runner(frame, **kwargs):
        assert kwargs["categorical"] == ["Code"]
        assert frame["Code"].tolist() == ["001", "1", "NA"]
        result = CapturedFit(frame)
        result.fit = SimpleNamespace(nBoot_used=2)
        result.workers = 1
        result.elapsed_seconds = 0.01
        return result

    app = ScreeningApp.__new__(ScreeningApp)
    app.base_model = False
    app.data, app.source_snapshot = snapshot.data, snapshot
    app.file_var = variable(str(original))
    app.screening = SimpleNamespace(METHOD_FORWARD="forward", run_screening=runner)
    app.response_list = selected(["Y"])
    app.factor_list = selected(["Code"])
    app.categorical_list = selected(["Code"])
    for key, value in {
        "boot": "2", "null_vectors": "0", "seed": "1", "jobs": "1",
        "repeats": "1", "model": "interactions", "center_polynomials": True,
        "method": "forward", "save_predictions": save_predictions, "likelihood": False,
    }.items():
        setattr(app, f"{key}_var", variable(value))
    app.run_button = SimpleNamespace(configure=lambda **_: None)
    app.progress = SimpleNamespace(start=lambda _: None, stop=lambda: None)
    app.status_var = SimpleNamespace(set=lambda _: None)
    app.events = queue.Queue()
    app.root = SimpleNamespace(after=lambda *_: None)
    app._show_results = lambda result, *, source: shown.append(source)

    class ImmediateWorker:
        def __init__(self, *, target, daemon):
            self.target = target

        def start(self):
            # A new browse occurs after fitting captures its source and roles.
            app.source_snapshot = CSVSnapshot.load(current)
            app.data = app.source_snapshot.data
            app.file_var = variable(str(current))
            self.target()

    monkeypatch.setattr(screening_app.threading, "Thread", ImmediateWorker)
    app._run()
    app._poll_events()
    assert not errors
    assert shown == [original]
    assert current.read_bytes() == current_payload
    if save_predictions:
        backups = list(tmp_path.glob("fit.csv*.bak"))
        assert len(backups) == 1 and backups[0].read_bytes() == original_payload
        assert "Predicted Y" in original.read_text(encoding="utf-8")
    else:
        assert original.read_bytes() == original_payload
