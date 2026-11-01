"""Real Tk smoke tests (skipped on headless systems without Tk/display)."""

import gc
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from svemnet.screening_app import ScreeningApp


@pytest.mark.parametrize("base_model", [False, True])
def test_gui_defaults_and_fit_with_opt_in_csv_export(tmp_path, base_model):
    # Production owns one Tcl interpreter for the lifetime of one process.
    # Isolate modes accordingly; repeated Tk initialization is unreliable on
    # some Windows Tcl runtimes and must not silently skip a tested GUI mode.
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path), str(int(base_model))],
        capture_output=True, text=True, timeout=60, env=environment,
        creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0,
    )
    if result.returncode == 77 and result.stdout.startswith("GUI_SKIP:"):
        pytest.skip(result.stdout.removeprefix("GUI_SKIP:").strip())
    assert result.returncode == 0, result.stdout + result.stderr


def _run_gui_smoke(tmp_path, monkeypatch, base_model):
    tk = pytest.importorskip("tkinter")
    from tkinter import filedialog, messagebox

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        if "no display name" in str(exc) or "couldn't connect to display" in str(exc):
            pytest.skip(f"Tk display unavailable: {exc}")
        raise
    root.withdraw()
    path = tmp_path / "experiment.csv"
    path.write_text("Y,A,B\n1,1,4\n2,2,1\n3,3,5\n4,4,2\n5,5,7\n6,6,3\n7,7,8\n8,8,6\n")
    original = path.read_bytes()
    notices, errors = [], []
    monkeypatch.setattr(
        sys, "unraisablehook",
        lambda arg: errors.append(("unraisable", repr(arg.exc_value))),
    )
    monkeypatch.setattr(filedialog, "askopenfilename", lambda **_: str(path))
    monkeypatch.setattr(messagebox, "showinfo", lambda *args: notices.append(args))
    monkeypatch.setattr(messagebox, "showerror", lambda *args: errors.append(args))
    monkeypatch.setattr(messagebox, "showwarning", lambda *args: errors.append(args))
    try:
        app = ScreeningApp(root, base_model=base_model)
        assert not app.save_predictions_var.get()
        assert app.method_var.get() == (
            "Lasso" if base_model else "SVEM forward selection"
        )
        assert app.null_vectors_var.get() == "0"
        if not base_model:
            assert app.likelihood_var.get()
            app.method_var.set(app.screening.METHOD_LASSO)
            assert not app.likelihood_var.get()
            app.method_var.set(app.screening.METHOD_FORWARD)
            assert app.likelihood_var.get()
        app._browse()
        assert path.read_bytes() == original
        app.jobs_var.set("1")
        app.boot_var.set("3")
        app.repeats_var.set("1")
        app.save_predictions_var.set(True)
        deadline = time.monotonic() + 30
        timed_out = []

        def check_completion():
            if notices or errors:
                root.quit()
            elif time.monotonic() >= deadline:
                timed_out.append(True)
                root.quit()
            else:
                root.after(10, check_completion)

        root.report_callback_exception = lambda kind, value, tb: errors.append(
            (kind.__name__, str(value))
        )
        # Tk's real mainloop marshals background-thread image finalizers. An
        # update() polling loop does not provide the same threading contract.
        root.after(0, app._run)
        root.after(10, check_completion)
        root.mainloop()
        assert not errors
        assert not timed_out, "GUI fit did not finish within 30 seconds"
        assert notices, "GUI fit did not finish"
        assert any(isinstance(child, tk.Toplevel) for child in root.winfo_children())
        assert "Predicted Y" in path.read_text()
        if not base_model:
            assert "Predicted Y (forward)" in path.read_text()
        backups = list(tmp_path.glob("*.bak"))
        assert len(backups) == 1 and backups[0].read_bytes() == original
        app._browse()
        assert all(
            not name.startswith("Predicted ") for name in app._selected(app.factor_list)
        )
    finally:
        root.destroy()
        gc.collect()  # Dispose Tk images on the GUI thread before the next worker.
    assert not errors


if __name__ == "__main__":
    try:
        with pytest.MonkeyPatch.context() as patch:
            _run_gui_smoke(Path(sys.argv[1]), patch, bool(int(sys.argv[2])))
    except pytest.skip.Exception as exc:
        print(f"GUI_SKIP:{exc}")
        raise SystemExit(77)
