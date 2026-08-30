"""Real Tk smoke tests (skipped on headless systems without Tk/display)."""

import time

import pytest

from svemnet.screening_app import ScreeningApp


@pytest.mark.parametrize("base_model", [False, True])
def test_gui_defaults_and_fit_with_opt_in_csv_export(tmp_path, monkeypatch, base_model):
    tk = pytest.importorskip("tkinter")
    from tkinter import filedialog, messagebox

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display unavailable")
    root.withdraw()
    path = tmp_path / "experiment.csv"
    path.write_text("Y,A,B\n1,1,4\n2,2,1\n3,3,5\n4,4,2\n5,5,7\n6,6,3\n7,7,8\n8,8,6\n")
    original = path.read_bytes()
    notices, errors = [], []
    monkeypatch.setattr(filedialog, "askopenfilename", lambda **_: str(path))
    monkeypatch.setattr(messagebox, "showinfo", lambda *args: notices.append(args))
    monkeypatch.setattr(messagebox, "showerror", lambda *args: errors.append(args))
    monkeypatch.setattr(messagebox, "showwarning", lambda *args: errors.append(args))
    try:
        app = ScreeningApp(root, base_model=base_model)
        assert not app.save_predictions_var.get()
        assert app.method_var.get() == ("Lasso" if base_model else "SVEM lasso")
        assert app.null_vectors_var.get() == "0"
        app._browse()
        assert path.read_bytes() == original
        app.jobs_var.set("1")
        app.boot_var.set("3")
        app.repeats_var.set("1")
        app.save_predictions_var.set(True)
        app._run()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not notices and not errors:
            root.update()
            time.sleep(0.01)
        assert not errors
        assert notices, "GUI fit did not finish"
        assert any(isinstance(child, tk.Toplevel) for child in root.winfo_children())
        assert "Predicted Y" in path.read_text()
        backups = list(tmp_path.glob("*.bak"))
        assert len(backups) == 1 and backups[0].read_bytes() == original
        app._browse()
        assert all(
            not name.startswith("Predicted ") for name in app._selected(app.factor_list)
        )
    finally:
        root.destroy()
