"""Tk desktop and command-line application for SVEM variable screening."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import queue
import sys
import threading
from pathlib import Path


def _load_screening_api():
    try:
        import pandas as pd

        from . import screening
    except ImportError as exc:
        raise SystemExit(
            "The screening app dependencies are missing. Install them with:\n"
            "  python -m pip install 'svemnet[screening]'\n\n"
            f"Details: {exc}"
        ) from exc
    return pd, screening


class ScreeningApp:
    """Small cross-platform Tk interface around :func:`run_screening`."""

    def __init__(self, root, *, base_model=False):
        import tkinter as tk
        from tkinter import ttk

        _, screening = _load_screening_api()
        self.tk = tk
        self.ttk = ttk
        self.screening = screening
        self.root = root
        self.base_model = base_model
        self.data = None
        self.source_snapshot = None
        self.events: queue.Queue = queue.Queue()

        root.title(
            "Base Lasso / Elastic Net" if base_model else "SVEM Variable Screening"
        )
        root.minsize(1080, 760)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        file_frame = ttk.Frame(root, padding=10)
        file_frame.grid(row=0, column=0, sticky="ew")
        file_frame.columnconfigure(1, weight=1)
        ttk.Label(file_frame, text="CSV data file:").grid(row=0, column=0, padx=(0, 8))
        self.file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_var).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(file_frame, text="Browse…", command=self._browse).grid(
            row=0, column=2, padx=(8, 0)
        )

        body = ttk.Frame(root, padding=(10, 0, 10, 10))
        body.grid(row=1, column=0, sticky="nsew")
        for col in range(3):
            body.columnconfigure(col, weight=1)
        body.rowconfigure(1, weight=1)

        ttk.Label(body, text="Response (select one)").grid(row=0, column=0, sticky="w")
        ttk.Label(body, text="Factors (select one or more)").grid(
            row=0, column=1, sticky="w"
        )
        ttk.Label(body, text="Treat as categorical (optional)").grid(
            row=0, column=2, sticky="w"
        )
        self.response_list = tk.Listbox(
            body, exportselection=False, selectmode="browse"
        )
        self.factor_list = tk.Listbox(
            body, exportselection=False, selectmode="extended"
        )
        self.categorical_list = tk.Listbox(
            body, exportselection=False, selectmode="extended"
        )
        self.response_list.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.factor_list.grid(row=1, column=1, sticky="nsew", padx=(0, 8))
        self.categorical_list.grid(row=1, column=2, sticky="nsew")

        options = ttk.LabelFrame(body, text="Options", padding=10)
        options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        for col in (1, 3, 5):
            options.columnconfigure(col, weight=1)

        ttk.Label(options, text="Candidate model:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value=screening.MODEL_INTERACTIONS)
        ttk.Combobox(
            options,
            textvariable=self.model_var,
            values=screening.MODEL_CHOICES,
            state="readonly",
            width=37,
        ).grid(row=0, column=1, columnspan=2, sticky="ew", padx=(6, 20))
        self.center_polynomials_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options,
            text="Center polynomials",
            variable=self.center_polynomials_var,
        ).grid(row=0, column=3, columnspan=3, sticky="w")
        ttk.Label(options, text="CV folds:" if base_model else "Bootstraps:").grid(
            row=1, column=0, sticky="w", pady=(10, 0)
        )
        self.boot_var = tk.StringVar(value="10" if base_model else "200")
        ttk.Spinbox(
            options,
            from_=1,
            to=100000,
            textvariable=self.boot_var,
            width=9,
        ).grid(row=1, column=1, sticky="w", padx=(6, 20), pady=(10, 0))
        ttk.Label(options, text="Null vectors:").grid(
            row=1, column=2, sticky="w", pady=(10, 0)
        )
        self.null_vectors_var = tk.StringVar(value="0")
        ttk.Spinbox(
            options,
            from_=0,
            to=100000,
            textvariable=self.null_vectors_var,
            width=9,
        ).grid(row=1, column=3, sticky="w", padx=(6, 20), pady=(10, 0))
        ttk.Label(options, text="Seed:").grid(row=1, column=4, sticky="w", pady=(10, 0))
        self.seed_var = tk.StringVar(value="12345")
        ttk.Entry(options, textvariable=self.seed_var, width=12).grid(
            row=1, column=5, sticky="w", padx=(6, 0), pady=(10, 0)
        )
        ttk.Label(options, text="Workers:").grid(
            row=2, column=0, sticky="w", pady=(10, 0)
        )
        self.jobs_var = tk.StringVar(value="-1")
        ttk.Spinbox(
            options,
            from_=-1,
            to=max(1, os.cpu_count() or 1),
            textvariable=self.jobs_var,
            width=7,
        ).grid(row=2, column=1, sticky="w", padx=(6, 20), pady=(10, 0))
        ttk.Label(
            options,
            text=(
                f"-1 = all available CPUs ({os.cpu_count() or 1} logical on "
                "this computer). Null vectors are independent standard-normal "
                "main effects."
            ),
        ).grid(row=2, column=2, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Label(options, text="Fitting method:").grid(
            row=3, column=0, sticky="w", pady=(10, 0)
        )
        choices = (
            ("Lasso", "Elastic net search (0.5, 1)")
            if base_model
            else screening.METHOD_CHOICES
        )
        self.method_var = tk.StringVar(value=choices[0])
        ttk.Combobox(
            options,
            textvariable=self.method_var,
            values=choices,
            state="readonly",
            width=35,
        ).grid(row=3, column=1, columnspan=3, sticky="ew", padx=(6, 20), pady=(10, 0))
        self.repeats_var = tk.StringVar(value="5")
        self.rule_var = tk.StringVar(value="min")
        if base_model:
            ttk.Label(options, text="CV repeats:").grid(
                row=4, column=0, sticky="w", pady=(10, 0)
            )
            ttk.Spinbox(
                options, from_=1, to=100, textvariable=self.repeats_var, width=9
            ).grid(row=4, column=1, sticky="w", padx=6, pady=(10, 0))
            ttk.Label(options, text="CV rule:").grid(
                row=4, column=2, sticky="w", pady=(10, 0)
            )
            ttk.Combobox(
                options,
                textvariable=self.rule_var,
                values=("min", "1se"),
                state="readonly",
                width=9,
            ).grid(row=4, column=3, sticky="w", padx=6, pady=(10, 0))
        else:
            ttk.Label(options, text="SVEM selection objective: wAIC").grid(
                row=4, column=0, columnspan=6, sticky="w", pady=(10, 0)
            )
        self.save_predictions_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options,
            text="Save predictions back to source CSV (backup created)",
            variable=self.save_predictions_var,
        ).grid(row=5, column=0, columnspan=6, sticky="w", pady=(10, 0))

        footer = ttk.Frame(root, padding=10)
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Choose a CSV file to begin.")
        ttk.Label(footer, textvariable=self.status_var).grid(
            row=0, column=0, sticky="w"
        )
        self.progress = ttk.Progressbar(footer, mode="indeterminate")
        self.progress.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.run_button = ttk.Button(
            footer,
            text="Fit base model" if base_model else "Run screening",
            command=self._run,
        )
        self.run_button.grid(
            row=0, column=1, rowspan=2, padx=(12, 0), ipadx=12, ipady=6
        )
        root.after(100, self._poll_events)

    def _browse(self):
        from tkinter import filedialog, messagebox

        path = filedialog.askopenfilename(
            title="Open experiment data",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            pd, _ = _load_screening_api()
            from .prediction import CSVSnapshot

            snapshot = CSVSnapshot.load(path)
            data = snapshot.data
            if data.shape[1] < 2:
                raise ValueError("the CSV must contain at least two columns")
        except Exception as exc:  # noqa: BLE001 - GUI must report parser errors
            messagebox.showerror("Could not open CSV", str(exc))
            return
        self.data = data
        self.source_snapshot = snapshot
        self.file_var.set(path)
        columns = [str(name) for name in data.columns]
        for widget in (
            self.response_list,
            self.factor_list,
            self.categorical_list,
        ):
            widget.delete(0, "end")
            for name in columns:
                widget.insert("end", name)
        self.response_list.selection_set(0)
        for i, name in enumerate(columns[1:], start=1):
            # Avoid silently feeding exported fitted values back into the model.
            if not name.startswith("Predicted "):
                self.factor_list.selection_set(i)
        for i, name in enumerate(columns):
            if not pd.api.types.is_numeric_dtype(data[name]):
                self.categorical_list.selection_set(i)
        self.status_var.set(f"Loaded {len(data):,} rows and {len(columns)} columns.")

    @staticmethod
    def _selected(widget):
        return [widget.get(i) for i in widget.curselection()]

    def _run(self):
        from tkinter import messagebox

        if self.data is None:
            messagebox.showerror("No data", "Choose a CSV data file first.")
            return
        responses = self._selected(self.response_list)
        factors = self._selected(self.factor_list)
        categorical = [
            name for name in self._selected(self.categorical_list) if name in factors
        ]
        try:
            if len(responses) != 1:
                raise ValueError("select exactly one response")
            if not factors:
                raise ValueError("select at least one factor")
            n_boot = int(self.boot_var.get())
            n_null_vectors = int(self.null_vectors_var.get())
            seed = int(self.seed_var.get())
            n_jobs = int(self.jobs_var.get())
            repeats = int(self.repeats_var.get())
            if n_boot < 1 or n_null_vectors < 0 or n_jobs == 0:
                raise ValueError(
                    "bootstraps must be positive; null vectors must be "
                    "nonnegative; workers must be -1 or a nonzero integer"
                )
            if self.base_model and (n_boot < 2 or repeats < 1):
                raise ValueError(
                    "CV folds must be at least two; repeats must be positive"
                )
        except ValueError as exc:
            messagebox.showerror("Invalid selection", str(exc))
            return

        kwargs = {
            "response": responses[0],
            "factors": factors,
            "categorical": categorical,
            "model": self.model_var.get(),
            "center_polynomials": self.center_polynomials_var.get(),
            "n_null_vectors": n_null_vectors,
            "seed": seed,
            "n_jobs": n_jobs,
        }
        method = self.method_var.get()
        alphas = (0.5, 1.0) if "search" in method else (1.0,)
        if self.base_model:
            from .base_model import run_base_model

            runner = run_base_model
            kwargs.update(
                alphas=alphas,
                nfolds=n_boot,
                repeats=repeats,
                choose_rule=self.rule_var.get(),
            )
        else:
            runner = self.screening.run_screening
            method_key = (
                "forward"
                if method == self.screening.METHOD_FORWARD
                else "elastic_net"
                if "search" in method
                else "lasso"
            )
            kwargs.update(method=method_key, alphas=alphas, n_boot=n_boot)
        analysis_data = self.data.copy(deep=True)
        snapshot = self.source_snapshot
        save_predictions = self.save_predictions_var.get()
        self.run_button.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set(
            "Fitting cross-validated base model…"
            if self.base_model
            else f"Fitting {n_boot} SVEM bootstrap members: {method}…"
        )

        def task():
            try:
                result = runner(analysis_data, **kwargs)
            except Exception as exc:  # noqa: BLE001 - return worker errors to Tk
                self.events.put(("error", exc))
                return
            saved, export_error = None, None
            if save_predictions:
                try:
                    if snapshot is None:
                        raise ValueError("load a CSV file before saving predictions")
                    saved = result.save_predictions_csv(snapshot)
                except Exception as exc:  # noqa: BLE001 - fit remains usable
                    export_error = str(exc)
            self.events.put(("done", result, saved, export_error))

        threading.Thread(target=task, daemon=True).start()

    def _poll_events(self):
        from tkinter import messagebox

        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "done":
                    result = event[1]
                    self.run_button.configure(state="normal")
                    self.progress.stop()
                    description = (
                        "Finished base model"
                        if self.base_model
                        else f"Finished {result.fit.nBoot_used} members"
                    )
                    self.status_var.set(
                        f"{description} using "
                        f"{result.workers} workers in "
                        f"{result.elapsed_seconds:.1f} seconds."
                    )
                    self._show_results(result)
                    if event[2]:
                        saved = event[2]
                        messagebox.showinfo(
                            "Predictions saved",
                            f"Added {saved['column']}\n"
                            f"To: {saved['path']}\nBackup: {saved['backup']}",
                        )
                        if Path(self.file_var.get()).absolute() == Path(saved["path"]):
                            from .prediction import CSVSnapshot

                            try:
                                self.source_snapshot = CSVSnapshot.load(saved["path"])
                                self.data = self.source_snapshot.data
                            except (OSError, ValueError) as exc:
                                messagebox.showwarning("Reload source CSV", str(exc))
                    if event[3]:
                        messagebox.showwarning(
                            "Model fitted; CSV not changed", event[3]
                        )
                elif event[0] == "error":
                    self.run_button.configure(state="normal")
                    self.progress.stop()
                    title = (
                        "Base model failed"
                        if self.base_model
                        else "SVEM screening failed"
                    )
                    self.status_var.set(title)
                    messagebox.showerror(title, str(event[1]))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _show_results(self, result):
        from tkinter import filedialog, messagebox, ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        window = self.tk.Toplevel(self.root)
        window.title(
            "Base Model Results"
            if self.base_model
            else "SVEM Variable Screening Results"
        )
        window.geometry("1150x800")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        tabs = ttk.Notebook(window)
        tabs.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        plot_frame = ttk.Frame(tabs)
        tabs.add(plot_frame, text="Pareto plot")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        scroll_canvas = self.tk.Canvas(plot_frame, highlightthickness=0)
        plot_scroll = ttk.Scrollbar(
            plot_frame, orient="vertical", command=scroll_canvas.yview
        )
        scroll_canvas.configure(yscrollcommand=plot_scroll.set)
        scroll_canvas.grid(row=0, column=0, sticky="nsew")
        plot_scroll.grid(row=0, column=1, sticky="ns")
        plot_inner = ttk.Frame(scroll_canvas)
        plot_window = scroll_canvas.create_window(
            (0, 0), window=plot_inner, anchor="nw"
        )
        if self.base_model:
            from .base_model import make_coefficient_figure

            figure = make_coefficient_figure(
                result.coefficient_table, null_effects=result.design.null_effects
            )
        else:
            figure = self.screening.make_pareto_figure(
                result.effect_usage, null_effects=result.design.null_effects
            )
        mpl_canvas = FigureCanvasTkAgg(figure, master=plot_inner)
        mpl_canvas.draw()
        mpl_widget = mpl_canvas.get_tk_widget()
        plot_height = max(650, int(figure.get_figheight() * figure.dpi))
        mpl_widget.configure(height=plot_height)
        mpl_widget.pack(fill="x", expand=True)

        def update_scroll_region(_event=None):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        def resize_plot(event):
            scroll_canvas.itemconfigure(plot_window, width=event.width)
            mpl_widget.configure(width=event.width, height=plot_height)

        def scroll_plot(event):
            direction = -1 if getattr(event, "delta", 0) > 0 else 1
            scroll_canvas.yview_scroll(3 * direction, "units")
            return "break"

        def scroll_plot_linux(event):
            direction = -1 if event.num == 4 else 1
            scroll_canvas.yview_scroll(3 * direction, "units")
            return "break"

        plot_inner.bind("<Configure>", update_scroll_region)
        scroll_canvas.bind("<Configure>", resize_plot)
        for widget in (scroll_canvas, mpl_widget):
            widget.bind("<MouseWheel>", scroll_plot)
            widget.bind("<Button-4>", scroll_plot_linux)
            widget.bind("<Button-5>", scroll_plot_linux)

        def table_tab(title, frame_data):
            frame = ttk.Frame(tabs)
            tabs.add(frame, text=title)
            tree = ttk.Treeview(
                frame, columns=list(frame_data.columns), show="headings"
            )
            for col in frame_data.columns:
                tree.heading(col, text=col)
                tree.column(col, width=180, anchor="w")
            for row in frame_data.itertuples(index=False, name=None):
                tree.insert(
                    "",
                    "end",
                    values=[f"{v:.4g}" if isinstance(v, float) else v for v in row],
                )
            scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scroll.set)
            tree.pack(side="left", fill="both", expand=True)
            scroll.pack(side="right", fill="y")

        if self.base_model:
            table_tab("Coefficients", result.coefficient_table)
        else:
            table_tab("Effect usage", result.effect_usage)
            table_tab("Parameter usage", result.parameter_usage)

        def save():
            directory = filedialog.askdirectory(title="Save screening results")
            if directory:
                result.save(directory)
                messagebox.showinfo("Results saved", f"Saved results to:\n{directory}")

        if self.base_model:
            summary = (
                f"Selected mixing alpha: {result.fit.selected_alpha:g}  |  "
                f"Lambda: {result.fit.selected_lambda:.5g}  |  "
                f"CV: {result.fit.nfolds} folds × {result.fit.repeats} repeats  |  "
                f"Rule: {result.fit.choose_rule}"
            )
        else:
            summary = f"Fitter: {result.method}  |  {result.fit.nBoot_used} bootstrap members  |  wAIC"
        ttk.Label(window, text=summary).grid(row=1, column=0, pady=(0, 8))
        ttk.Button(window, text="Save tables and plot…", command=save).grid(
            row=2, column=0, pady=(0, 10)
        )


def launch_gui(*, base_model=False) -> int:
    import tkinter as tk

    root = tk.Tk()
    ScreeningApp(root, base_model=base_model)
    root.mainloop()
    return 0


def run_cli(args, *, base_model=False) -> int:
    _pd, screening = _load_screening_api()
    from .prediction import CSVSnapshot

    snapshot = CSVSnapshot.load(args.csv)
    data = snapshot.data
    model = (
        screening.MODEL_RESPONSE_SURFACE
        if args.model == "response-surface"
        else screening.MODEL_INTERACTIONS
    )
    kwargs = {
        "response": args.response,
        "factors": args.factors,
        "categorical": args.categorical,
        "model": model,
        "center_polynomials": args.center_polynomials,
        "n_null_vectors": args.null_vectors,
        "seed": args.seed,
        "n_jobs": args.jobs,
    }
    alphas = (
        tuple(args.alphas)
        if args.alphas
        else ((0.5, 1.0) if args.method == "elastic_net" else (1.0,))
    )
    if args.method == "lasso" and alphas != (1.0,):
        raise ValueError("use --method elastic_net for an alpha search")
    if base_model:
        from .base_model import run_base_model

        result = run_base_model(
            data,
            alphas=alphas,
            nfolds=args.folds,
            repeats=args.repeats,
            choose_rule=args.rule,
            **kwargs,
        )
    else:
        result = screening.run_screening(
            data, method=args.method, alphas=alphas, n_boot=args.bootstraps, **kwargs
        )
    output = result.save(args.output)
    if args.save_predictions:
        saved = result.save_predictions_csv(snapshot)
        print(
            f"Predictions: {saved['path']} ({saved['column']}); backup: {saved['backup']}"
        )
    print(
        f"Finished using {result.workers} workers in {result.elapsed_seconds:.2f}s. "
        f"Results: {output.resolve()}"
    )
    return 0


def build_parser(*, base_model=False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single cross-validated lasso/elastic net"
        if base_model
        else "SVEM variable screening"
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("gui", help="open the desktop application")
    run = subparsers.add_parser("run", help="run a CSV analysis without the GUI")
    run.add_argument("csv", type=Path)
    run.add_argument("--response", required=True)
    run.add_argument("--factors", nargs="+", required=True)
    run.add_argument("--categorical", nargs="*", default=[])
    run.add_argument(
        "--method",
        choices=("lasso", "elastic_net")
        if base_model
        else ("lasso", "elastic_net", "forward"),
        default="lasso",
    )
    run.add_argument(
        "--alphas", type=float, nargs="+", help="mixing alphas in (0,1]; 1=lasso"
    )
    run.add_argument(
        "--save-predictions",
        action="store_true",
        help="append predictions to source CSV with an exact-byte backup (default: off)",
    )
    if base_model:
        run.add_argument("--folds", type=int, default=10)
        run.add_argument("--repeats", type=int, default=5)
        run.add_argument("--rule", choices=("min", "1se"), default="min")
    else:
        run.add_argument("--bootstraps", type=int, default=200)
    run.add_argument(
        "--model",
        choices=("interactions", "response-surface"),
        default="interactions",
    )
    run.add_argument(
        "--center-polynomials",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "center uncoded continuous factors inside interactions and powers "
            "while leaving main effects raw (default: on)"
        ),
    )
    run.add_argument(
        "--null-vectors",
        type=int,
        default=0,
        help=(
            "add this many independent standard-normal null main effects (default: 0)"
        ),
    )
    run.add_argument("--seed", type=int, default=12345)
    run.add_argument("--jobs", type=int, default=-1, help="-1 uses all available CPUs")
    run.add_argument(
        "--output",
        type=Path,
        default=Path("svem_lasso_results" if base_model else "svem_screening_results"),
    )
    return parser


def main(argv=None, *, base_model=False) -> int:
    multiprocessing.freeze_support()
    parser = build_parser(base_model=base_model)
    args = parser.parse_args(argv)
    if args.command in (None, "gui"):
        return launch_gui(base_model=base_model)
    return run_cli(args, base_model=base_model)


if __name__ == "__main__":
    sys.exit(main())
