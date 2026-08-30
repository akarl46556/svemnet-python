"""Subprocess CLI tests, including safe default and opt-in CSV replacement."""

import json
import subprocess
import sys

import pandas as pd
import pytest


@pytest.mark.parametrize("base_model", [False, True])
@pytest.mark.parametrize("save_predictions", [False, True])
def test_cli_round_trip(tmp_path, base_model, save_predictions):
    path = tmp_path / "data.csv"
    path.write_text("Y,A,B\n1,1,4\n2,2,1\n3,3,5\n4,4,2\n5,5,7\n6,6,3\n7,7,8\n8,8,6\n")
    original = path.read_bytes()
    output = tmp_path / "results"
    args = [
        sys.executable,
        "-m",
        "svemnet.lasso_app" if base_model else "svemnet.screening_app",
        "run",
        str(path),
        "--response",
        "Y",
        "--factors",
        "A",
        "B",
        "--method",
        "elastic_net",
        "--jobs",
        "2",
        "--output",
        str(output),
    ]
    args += ["--folds", "3", "--repeats", "1"] if base_model else ["--bootstraps", "3"]
    if save_predictions:
        args.append("--save-predictions")
    process = subprocess.run(
        args, capture_output=True, text=True, timeout=60, check=False
    )
    assert process.returncode == 0, process.stdout + process.stderr
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["alpha_candidates"] == [0.5, 1.0]
    assert metadata["workers"] == 2
    backups = list(tmp_path.glob("*.bak"))
    if save_predictions:
        assert len(backups) == 1 and backups[0].read_bytes() == original
        data = pd.read_csv(path)
        assert data.shape == (8, 4)
        assert data.iloc[:, -1].notna().all()
    else:
        assert path.read_bytes() == original and not backups
