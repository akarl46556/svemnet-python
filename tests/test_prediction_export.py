import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from svemnet.base_model import run_base_model
from svemnet.prediction import CSVSnapshot
from svemnet.screening import MODEL_RESPONSE_SURFACE, run_screening
from svemnet.screening_app import build_parser


def sample():
    rng = np.random.default_rng(52)
    frame = pd.DataFrame(
        {
            "Y": rng.normal(size=20),
            "A": rng.normal(size=20) + 10,
            "B": rng.normal(size=20) + 20,
            "Group": ["a", "b"] * 10,
        }
    )
    frame.loc[3, "Y"] = np.nan
    frame.loc[7, "A"] = np.nan
    return frame


@pytest.mark.parametrize("method", ["lasso", "forward", "elastic_net"])
@pytest.mark.parametrize("center", [False, True])
def test_source_predictions_preserve_rows_and_training_schema(method, center):
    frame = sample()
    result = run_screening(
        frame,
        response="Y",
        factors=["A", "B", "Group"],
        model=MODEL_RESPONSE_SURFACE,
        method=method,
        alphas=(0.5, 1) if method == "elastic_net" else (1,),
        center_polynomials=center,
        n_null_vectors=2,
        n_boot=3,
        n_jobs=1,
    )
    values = result.predict_source()
    assert len(values) == len(frame)
    assert np.isnan(values[7])
    assert np.isfinite(values[3])  # response is not required to score a row
    np.testing.assert_allclose(
        values[list(result.design.training_rows)], result.fit.predictions, atol=1e-12
    )
    new = frame.drop(columns="Y").copy()
    for i, name in enumerate(result.design.null_effects):
        new[name] = result.design.null_values[:, i]
    np.testing.assert_allclose(result.predict(new), values)
    new.loc[0, "Group"] = "unseen"
    with pytest.raises(ValueError, match="not seen"):
        result.predict(new)
    with pytest.raises(ValueError, match="prediction columns"):
        result.predict(frame)


def test_csv_backup_fields_and_prediction_column_collision(tmp_path):
    path = tmp_path / "experiment.csv"
    payload = '\ufeffID,Y,A,Predicted Y\r\n001,1,2,old\r\n"two, quoted",2,4,old\r\n003,,8,old\r\n004,4,,old\r\n005,5,16,old\r\n'.encode()
    path.write_bytes(payload)
    snapshot = CSVSnapshot.load(path)
    result = run_screening(
        snapshot.data, response="Y", factors=["A"], n_boot=3, n_jobs=1
    )
    saved = result.save_predictions_csv(snapshot, column_name="Predicted Y")
    assert saved["column"] == "Predicted Y [2]"
    assert Path(saved["backup"]).read_bytes() == payload
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    original = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
    assert [row[:-1] for row in rows] == original
    assert rows[3][-1] != ""  # missing response, complete factor
    assert rows[4][-1] == ""  # missing factor
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with pytest.raises(ValueError, match="changed"):
        result.save_predictions_csv(snapshot)


def test_changed_or_wrong_csv_is_not_overwritten(tmp_path):
    path = tmp_path / "data.csv"
    sample().to_csv(path, index=False)
    snapshot = CSVSnapshot.load(path)
    result = run_screening(
        snapshot.data, response="Y", factors=["A"], n_boot=2, n_jobs=1
    )
    path.write_bytes(path.read_bytes() + b"\n")
    changed = path.read_bytes()
    with pytest.raises(ValueError, match="changed"):
        result.save_predictions_csv(snapshot)
    assert path.read_bytes() == changed
    assert not list(tmp_path.glob("*.bak"))
    wrong = sample().fillna(0)
    wrong.to_csv(tmp_path / "wrong.csv", index=False)
    with pytest.raises(ValueError, match="not the data"):
        result.save_predictions_csv(CSVSnapshot.load(tmp_path / "wrong.csv"))


def test_failed_replace_leaves_original_and_backup(tmp_path, monkeypatch):
    path = tmp_path / "data.csv"
    path.write_text("Y,A\n1,2\n2,3\n3,4\n")
    before = path.read_bytes()
    snapshot = CSVSnapshot.load(path)

    def locked(*_args):
        raise PermissionError("file is locked")

    monkeypatch.setattr("svemnet.prediction.os.replace", locked)
    with pytest.raises(PermissionError, match="locked"):
        snapshot.append_predictions([1, 2, 3], "Predicted Y")
    assert path.read_bytes() == before
    assert len(list(tmp_path.glob("*.bak"))) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_gui_cli_defaults_and_no_write_opt_in():
    args = ["run", "data.csv", "--response", "Y", "--factors", "A"]
    for base in (False, True):
        parsed = build_parser(base_model=base).parse_args(args)
        assert not parsed.save_predictions
        assert parsed.method == "lasso"
        assert parsed.null_vectors == 0


@pytest.mark.parametrize("base_model", [False, True])
@pytest.mark.parametrize("center", [False, True])
def test_numeric_categorical_prediction_export(tmp_path, base_model, center):
    frame = sample()
    frame["Group"] = frame["Group"].map({"a": 1, "b": 2})
    path = tmp_path / "numeric_categories.csv"
    frame.to_csv(path, index=False)
    snapshot = CSVSnapshot.load(path)
    runner = run_base_model if base_model else run_screening
    fit_options = {"nfolds": 3, "repeats": 1} if base_model else {"n_boot": 3}
    result = runner(
        snapshot.data,
        response="Y",
        factors=["A", "B", "Group"],
        categorical=["Group"],
        model=MODEL_RESPONSE_SURFACE,
        center_polynomials=center,
        n_jobs=1,
        **fit_options,
    )
    values = result.predict_source()
    np.testing.assert_allclose(
        values[list(result.design.training_rows)],
        result.fit.predictions,
        atol=1e-12,
    )
    assert np.isfinite(values[3]) and np.isnan(values[7])
    saved = result.save_predictions_csv(snapshot)
    np.testing.assert_allclose(pd.read_csv(path)[saved["column"]], values)
    new = snapshot.data.copy()
    new.loc[0, "Group"] = 3
    with pytest.raises(ValueError, match="not seen"):
        result.predict(new)
