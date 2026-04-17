from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "tase2_phase_switching_mpl"),
)

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import numpy as np

from .analysis import (
    AnalysisParameters,
    AnalysisResult,
    SIMPLE_STATE_COLORS,
    SIMPLE_STATE_NAMES,
    export_analysis,
    run_analysis,
)


FILE_TYPES = [
    ("NetCDF files", "*.nc *.nc4 *.h5 *.hdf5"),
    ("All files", "*.*"),
]


class AnalysisApp:
    VIEW_OPTIONS = [
        "Average normalized total map",
        "Cross mask",
        "Mask occupancy diagnostics",
        "Total intensity",
        "Near-EF intensity",
        "Feature map",
        "Delta feature",
        "Cluster map",
        "Cluster sequence map",
        "Simple state map",
        "Simple state sequence map",
    ]

    def __init__(self, root: tk.Tk, initial_files: list[str] | None = None) -> None:
        self.root = root
        self.root.title("TaSe2 Phase Switching Analyzer")
        self.root.geometry("1680x1020")
        self.root.minsize(1320, 860)

        self.file_paths: list[str] = []
        self.result: AnalysisResult | None = None
        self.selected_pixel: tuple[int, int] | None = None

        defaults = AnalysisParameters()
        self.parameter_vars = {
            "fermi_level_ev": tk.StringVar(value=str(defaults.fermi_level_ev)),
            "ef_window_ev": tk.StringVar(value=str(defaults.ef_window_ev)),
            "wide_window_ev": tk.StringVar(value=str(defaults.wide_window_ev)),
            "n_clusters": tk.StringVar(value=str(defaults.n_clusters)),
            "n_pca_components": tk.StringVar(value=str(defaults.n_pca_components)),
            "cross_threshold_quantile": tk.StringVar(value=str(defaults.cross_threshold_quantile)),
            "cross_row_fraction": tk.StringVar(value=str(defaults.cross_row_fraction)),
            "cross_col_fraction": tk.StringVar(value=str(defaults.cross_col_fraction)),
            "cross_background_quantile": tk.StringVar(value=str(defaults.cross_background_quantile)),
            "cross_pad": tk.StringVar(value=str(defaults.cross_pad)),
            "simple_state_low_quantile": tk.StringVar(value=str(defaults.simple_state_low_quantile)),
            "simple_state_high_quantile": tk.StringVar(value=str(defaults.simple_state_high_quantile)),
        }

        self.status_var = tk.StringVar(
            value="Choose 1 to 4 NetCDF files, adjust the analysis parameters, then run the pipeline."
        )
        self.view_var = tk.StringVar(value=self.VIEW_OPTIONS[0])
        self.state_var = tk.StringVar(value="")
        self.feature_var = tk.StringVar(value="")
        self.compare_from_var = tk.StringVar(value="")
        self.compare_to_var = tk.StringVar(value="")

        self._build_ui()

        if initial_files:
            self._set_files(initial_files)

        self._render_placeholder_text()

    def _build_ui(self) -> None:
        main_pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_pane, padding=12)
        main_pane.add(controls_frame, weight=0)

        right_frame = ttk.Frame(main_pane, padding=(0, 12, 12, 12))
        main_pane.add(right_frame, weight=1)

        self._build_controls_panel(controls_frame)
        self._build_visual_panel(right_frame)

        status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(12, 6))
        status_bar.pack(fill=tk.X)

    def _build_controls_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        files_frame = ttk.LabelFrame(parent, text="Data Files", padding=10)
        files_frame.grid(row=0, column=0, sticky="nsew")
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)

        self.file_listbox = tk.Listbox(files_frame, height=9, exportselection=False)
        self.file_listbox.grid(row=0, column=0, columnspan=2, sticky="nsew")

        ttk.Button(files_frame, text="Add Files", command=self._add_files).grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(files_frame, text="Remove Selected", command=self._remove_selected_files).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        ttk.Button(files_frame, text="Move Up", command=lambda: self._move_selected_file(-1)).grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(files_frame, text="Move Down", command=lambda: self._move_selected_file(1)).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        ttk.Button(files_frame, text="Clear Files", command=self._clear_files).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        spectral_frame = ttk.LabelFrame(parent, text="Spectral Parameters", padding=10)
        spectral_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        self._add_parameter_row(spectral_frame, 0, "Fermi level (eV)", "fermi_level_ev")
        self._add_parameter_row(spectral_frame, 1, "Near-EF window (eV)", "ef_window_ev")
        self._add_parameter_row(spectral_frame, 2, "Wide window (eV)", "wide_window_ev")
        self._add_parameter_row(spectral_frame, 3, "Number of clusters", "n_clusters")
        self._add_parameter_row(spectral_frame, 4, "PCA components", "n_pca_components")

        mask_frame = ttk.LabelFrame(parent, text="Cross Mask and State Thresholds", padding=10)
        mask_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self._add_parameter_row(mask_frame, 0, "Cross threshold quantile", "cross_threshold_quantile")
        self._add_parameter_row(mask_frame, 1, "Cross row fraction", "cross_row_fraction")
        self._add_parameter_row(mask_frame, 2, "Cross column fraction", "cross_col_fraction")
        self._add_parameter_row(mask_frame, 3, "Cross background quantile", "cross_background_quantile")
        self._add_parameter_row(mask_frame, 4, "Cross padding", "cross_pad")
        self._add_parameter_row(mask_frame, 5, "State low quantile", "simple_state_low_quantile")
        self._add_parameter_row(mask_frame, 6, "State high quantile", "simple_state_high_quantile")

        actions_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        actions_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        actions_frame.columnconfigure(0, weight=1)

        ttk.Button(actions_frame, text="Run Analysis", command=self._run_analysis).grid(row=0, column=0, sticky="ew")
        ttk.Button(actions_frame, text="Save Results...", command=self._save_results).grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(actions_frame, text="Save Current Plot...", command=self._save_current_plot).grid(row=2, column=0, sticky="ew", pady=(8, 0))

        help_text = (
            "Sequence order matters.\n"
            "Use the file order to represent the pulse sequence you want to compare.\n"
            "Click any map to inspect that pixel's local spectrum across all states."
        )
        ttk.Label(parent, text=help_text, justify=tk.LEFT, wraplength=320).grid(row=4, column=0, sticky="ew", pady=(12, 0))

    def _build_visual_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        map_frame = ttk.Frame(parent)
        map_frame.grid(row=0, column=0, sticky="nsew")
        map_frame.columnconfigure(0, weight=1)
        map_frame.rowconfigure(2, weight=1)

        controls_row = ttk.Frame(map_frame)
        controls_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(controls_row, text="View").grid(row=0, column=0, sticky="w")
        self.view_combo = ttk.Combobox(
            controls_row,
            textvariable=self.view_var,
            values=self.VIEW_OPTIONS,
            state="readonly",
            width=28,
        )
        self.view_combo.grid(row=1, column=0, sticky="w", padx=(0, 8))
        self.view_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_main_plot())

        ttk.Label(controls_row, text="State").grid(row=0, column=1, sticky="w")
        self.state_combo = ttk.Combobox(controls_row, textvariable=self.state_var, state="readonly", width=24)
        self.state_combo.grid(row=1, column=1, sticky="w", padx=(0, 8))
        self.state_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_main_plot())

        ttk.Label(controls_row, text="Feature").grid(row=0, column=2, sticky="w")
        self.feature_combo = ttk.Combobox(controls_row, textvariable=self.feature_var, state="readonly", width=24)
        self.feature_combo.grid(row=1, column=2, sticky="w", padx=(0, 8))
        self.feature_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_main_plot())

        ttk.Label(controls_row, text="Compare from").grid(row=0, column=3, sticky="w")
        self.compare_from_combo = ttk.Combobox(
            controls_row,
            textvariable=self.compare_from_var,
            state="readonly",
            width=24,
        )
        self.compare_from_combo.grid(row=1, column=3, sticky="w", padx=(0, 8))
        self.compare_from_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_main_plot())

        ttk.Label(controls_row, text="Compare to").grid(row=0, column=4, sticky="w")
        self.compare_to_combo = ttk.Combobox(
            controls_row,
            textvariable=self.compare_to_var,
            state="readonly",
            width=24,
        )
        self.compare_to_combo.grid(row=1, column=4, sticky="w")
        self.compare_to_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_main_plot())

        toolbar_frame = ttk.Frame(map_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")

        self.main_figure = Figure(figsize=(11, 7.2), dpi=100, constrained_layout=True)
        self.main_canvas = FigureCanvasTkAgg(self.main_figure, master=map_frame)
        self.main_canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")
        self.main_canvas.mpl_connect("button_press_event", self._on_main_plot_click)

        self.main_toolbar = NavigationToolbar2Tk(self.main_canvas, toolbar_frame, pack_toolbar=False)
        self.main_toolbar.update()
        self.main_toolbar.pack(side=tk.LEFT, fill=tk.X)

        bottom_pane = ttk.Notebook(parent)
        bottom_pane.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        pixel_frame = ttk.Frame(bottom_pane, padding=8)
        pixel_frame.columnconfigure(0, weight=1)
        pixel_frame.rowconfigure(0, weight=1)
        pixel_frame.rowconfigure(1, weight=0)
        bottom_pane.add(pixel_frame, text="Pixel Inspector")

        self.pixel_figure = Figure(figsize=(11, 4.8), dpi=100, constrained_layout=True)
        self.pixel_canvas = FigureCanvasTkAgg(self.pixel_figure, master=pixel_frame)
        self.pixel_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.pixel_text = tk.Text(pixel_frame, height=10, wrap="word")
        self.pixel_text.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.pixel_text.configure(state="disabled")

        summary_frame = ttk.Frame(bottom_pane, padding=8)
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)
        bottom_pane.add(summary_frame, text="Summary")

        self.summary_text = tk.Text(summary_frame, wrap="word")
        self.summary_text.grid(row=0, column=0, sticky="nsew")
        self.summary_text.configure(state="disabled")

    def _add_parameter_row(self, parent: ttk.LabelFrame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(parent, textvariable=self.parameter_vars[key], width=16)
        entry.grid(row=row, column=1, sticky="e", padx=(10, 0), pady=2)

    def _add_files(self) -> None:
        selected = list(filedialog.askopenfilenames(title="Choose NetCDF files", filetypes=FILE_TYPES))
        if not selected:
            return

        new_paths = [str(Path(path).expanduser().resolve()) for path in selected]
        merged = self.file_paths + [path for path in new_paths if path not in self.file_paths]
        if len(merged) > 4:
            messagebox.showwarning(
                "File limit",
                "The desktop analyzer supports up to four files at a time. Only the first four will be kept.",
            )
            merged = merged[:4]
        self._set_files(merged)

    def _remove_selected_files(self) -> None:
        selection = list(self.file_listbox.curselection())
        if not selection:
            return
        updated_files = list(self.file_paths)
        for index in reversed(selection):
            del updated_files[index]
        self._set_files(updated_files)

    def _move_selected_file(self, direction: int) -> None:
        selection = self.file_listbox.curselection()
        if len(selection) != 1:
            return

        index = selection[0]
        new_index = index + direction
        if not 0 <= new_index < len(self.file_paths):
            return

        updated_files = list(self.file_paths)
        updated_files[index], updated_files[new_index] = updated_files[new_index], updated_files[index]
        self._set_files(updated_files)
        self.file_listbox.selection_set(new_index)

    def _clear_files(self) -> None:
        self._set_files([])

    def _set_files(self, file_paths: list[str]) -> None:
        self.file_paths = list(file_paths)
        self.result = None
        self.selected_pixel = None
        self._sync_file_listbox()
        self._update_selector_values()
        self._render_placeholder_text()

    def _sync_file_listbox(self) -> None:
        self.file_listbox.delete(0, tk.END)
        for index, path in enumerate(self.file_paths):
            self.file_listbox.insert(tk.END, f"{index + 1}. {Path(path).name}")

    def _update_selector_values(self) -> None:
        if self.result is not None:
            state_values = self.result.state_names
            feature_values = self.result.feature_names
        else:
            state_values = [Path(path).name for path in self.file_paths]
            feature_values = []

        self.state_combo["values"] = state_values
        self.compare_from_combo["values"] = state_values
        self.compare_to_combo["values"] = state_values
        self.feature_combo["values"] = feature_values

        if state_values:
            if self.state_var.get() not in state_values:
                self.state_var.set(state_values[0])
            if self.compare_from_var.get() not in state_values:
                self.compare_from_var.set(state_values[0])
            if self.compare_to_var.get() not in state_values:
                self.compare_to_var.set(state_values[min(1, len(state_values) - 1)])
        else:
            self.state_var.set("")
            self.compare_from_var.set("")
            self.compare_to_var.set("")

        if feature_values:
            if self.feature_var.get() not in feature_values:
                self.feature_var.set(feature_values[0])
        else:
            self.feature_var.set("")

    def _parse_parameters(self) -> AnalysisParameters:
        try:
            params = AnalysisParameters(
                fermi_level_ev=float(self.parameter_vars["fermi_level_ev"].get()),
                ef_window_ev=float(self.parameter_vars["ef_window_ev"].get()),
                wide_window_ev=float(self.parameter_vars["wide_window_ev"].get()),
                n_clusters=int(self.parameter_vars["n_clusters"].get()),
                n_pca_components=int(self.parameter_vars["n_pca_components"].get()),
                cross_threshold_quantile=float(self.parameter_vars["cross_threshold_quantile"].get()),
                cross_row_fraction=float(self.parameter_vars["cross_row_fraction"].get()),
                cross_col_fraction=float(self.parameter_vars["cross_col_fraction"].get()),
                cross_background_quantile=float(self.parameter_vars["cross_background_quantile"].get()),
                cross_pad=int(self.parameter_vars["cross_pad"].get()),
                simple_state_low_quantile=float(self.parameter_vars["simple_state_low_quantile"].get()),
                simple_state_high_quantile=float(self.parameter_vars["simple_state_high_quantile"].get()),
            )
        except ValueError as exc:
            raise ValueError(f"Could not parse the parameter form: {exc}") from exc

        params.validate()
        return params

    def _run_analysis(self) -> None:
        if not 1 <= len(self.file_paths) <= 4:
            messagebox.showerror("Missing files", "Please choose between one and four NetCDF files.")
            return

        try:
            parameters = self._parse_parameters()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self.status_var.set("Running analysis...")
        self.root.update_idletasks()

        try:
            self.result = run_analysis(self.file_paths, parameters)
        except Exception as exc:
            self.result = None
            self.status_var.set("Analysis failed.")
            messagebox.showerror("Analysis failed", str(exc))
            self._render_placeholder_text()
            return

        self.selected_pixel = None
        self._update_selector_values()
        self._refresh_main_plot()
        self._update_pixel_details()
        self._update_summary_text()
        self.status_var.set(
            f"Analysis complete. {int(self.result.valid_mask.sum())} valid pixels inside the cross across {self.result.n_states} state(s)."
        )

    def _save_results(self) -> None:
        if self.result is None:
            messagebox.showinfo("No results", "Run the analysis before exporting any results.")
            return

        output_dir = filedialog.askdirectory(title="Choose an output folder")
        if not output_dir:
            return

        try:
            saved_dir = export_analysis(self.result, output_dir)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return

        self.status_var.set(f"Saved analysis outputs to {saved_dir}")
        messagebox.showinfo("Export complete", f"Saved analysis outputs to:\n{saved_dir}")

    def _save_current_plot(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save current plot",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF document", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.main_figure.savefig(path, dpi=220)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        self.status_var.set(f"Saved current plot to {path}")

    def _render_placeholder_text(self) -> None:
        self.main_figure.clear()
        axis = self.main_figure.add_subplot(111)
        if self.file_paths:
            axis.text(
                0.5,
                0.5,
                "Ready to run.\nUse the controls on the left, then click Run Analysis.",
                ha="center",
                va="center",
                fontsize=14,
            )
        else:
            axis.text(
                0.5,
                0.5,
                "Choose 1 to 4 NetCDF files to begin.",
                ha="center",
                va="center",
                fontsize=14,
            )
        axis.set_axis_off()
        self.main_canvas.draw_idle()

        self.pixel_figure.clear()
        pixel_axis = self.pixel_figure.add_subplot(111)
        pixel_axis.text(
            0.5,
            0.5,
            "Click a map after the analysis runs\nto inspect the local spectrum across states.",
            ha="center",
            va="center",
            fontsize=12,
        )
        pixel_axis.set_axis_off()
        self.pixel_canvas.draw_idle()
        self._set_text_widget(self.summary_text, "")
        self._set_text_widget(self.pixel_text, "")

    def _refresh_main_plot(self) -> None:
        if self.result is None:
            self._render_placeholder_text()
            return

        view = self.view_var.get()
        self.main_figure.clear()

        if view == "Average normalized total map":
            axis = self.main_figure.add_subplot(111)
            image = axis.imshow(
                self.result.average_normalized_total_map.T,
                origin="lower",
                cmap="magma",
                aspect="auto",
            )
            axis.set_title("Average normalized total map")
            axis.set_xlabel("x index")
            axis.set_ylabel("y index")
            self.main_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            self._mark_selected_pixel(axis)

        elif view == "Cross mask":
            axes = self.main_figure.subplots(1, 2)
            left, right = axes

            image = left.imshow(
                self.result.average_normalized_total_map.T,
                origin="lower",
                cmap="magma",
                aspect="auto",
            )
            yy, xx = np.where(~self.result.valid_mask.T)
            left.scatter(xx, yy, s=4, c="cyan", alpha=0.7)
            left.set_title("Average map with excluded pixels")
            left.set_xlabel("x index")
            left.set_ylabel("y index")
            self.main_figure.colorbar(image, ax=left, fraction=0.046, pad=0.04)

            right.imshow(self.result.valid_mask.T, origin="lower", cmap="gray", aspect="auto")
            right.set_title("Auto-detected cross mask")
            right.set_xlabel("x index")
            right.set_ylabel("y index")
            self._mark_selected_pixel(left)
            self._mark_selected_pixel(right)

        elif view == "Mask occupancy diagnostics":
            axes = self.main_figure.subplots(1, 2)
            left, right = axes

            image = left.imshow(
                self.result.average_normalized_total_map.T,
                origin="lower",
                cmap="magma",
                aspect="auto",
            )
            left.set_title("Average normalized total map")
            left.set_xlabel("x index")
            left.set_ylabel("y index")
            self.main_figure.colorbar(image, ax=left, fraction=0.046, pad=0.04)
            self._mark_selected_pixel(left)

            right.plot(self.result.row_occupancy, label="row occupancy")
            right.plot(self.result.col_occupancy, label="column occupancy")
            right.axhline(self.result.parameters.cross_row_fraction, linestyle="--", color="#444444", label="row threshold")
            right.axhline(self.result.parameters.cross_col_fraction, linestyle=":", color="#777777", label="column threshold")
            right.set_title("Cross-mask occupancy diagnostics")
            right.set_xlabel("row / column index")
            right.set_ylabel("occupancy fraction")
            right.legend(loc="best")

        elif view in {"Total intensity", "Near-EF intensity", "Feature map", "Cluster map", "Simple state map"}:
            state_index = self._current_state_index()
            axis = self.main_figure.add_subplot(111)
            if view == "Total intensity":
                data = self.result.total_maps[state_index]
                title = f"{self.result.state_names[state_index]}: total intensity"
                image = axis.imshow(data.T, origin="lower", cmap="viridis", aspect="auto")
                self.main_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            elif view == "Near-EF intensity":
                data = self.result.ef_maps[state_index]
                title = f"{self.result.state_names[state_index]}: near-EF intensity"
                image = axis.imshow(data.T, origin="lower", cmap="viridis", aspect="auto")
                self.main_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            elif view == "Feature map":
                feature_name = self.feature_var.get() or self.result.feature_names[0]
                data = self.result.features_by_state[state_index][feature_name]
                title = f"{self.result.state_names[state_index]}: {feature_name}"
                image = axis.imshow(data.T, origin="lower", cmap="viridis", aspect="auto")
                self.main_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            elif view == "Cluster map":
                display = self.result.cluster_maps[state_index].astype(float).copy()
                display[~self.result.valid_mask] = np.nan
                cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(1, len(self.result.cluster_mean_ef_fraction)))
                cmap = cmap.copy()
                cmap.set_bad(color="lightgray")
                image = axis.imshow(
                    display.T,
                    origin="lower",
                    cmap=cmap,
                    vmin=0,
                    vmax=max(0, len(self.result.cluster_mean_ef_fraction) - 1),
                    aspect="auto",
                )
                title = f"{self.result.state_names[state_index]}: cluster map"
                cbar = self.main_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
                cbar.set_label("ordered cluster id")
            else:
                display = self.result.simple_state_code_maps[state_index].astype(float).copy()
                display[~self.result.valid_mask] = np.nan
                cmap = mcolors.ListedColormap([SIMPLE_STATE_COLORS[name] for name in SIMPLE_STATE_NAMES])
                cmap.set_bad(color="lightgray")
                norm = mcolors.BoundaryNorm(np.arange(-0.5, len(SIMPLE_STATE_NAMES) + 0.5, 1), cmap.N)
                image = axis.imshow(display.T, origin="lower", cmap=cmap, norm=norm, aspect="auto")
                title = f"{self.result.state_names[state_index]}: simple state map"
                cbar = self.main_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, ticks=np.arange(len(SIMPLE_STATE_NAMES)))
                cbar.ax.set_yticklabels(list(SIMPLE_STATE_NAMES))

            axis.set_title(title)
            axis.set_xlabel("x index")
            axis.set_ylabel("y index")
            self._mark_selected_pixel(axis)

        elif view == "Delta feature":
            from_index = self._current_compare_index(self.compare_from_var.get(), fallback=0)
            to_index = self._current_compare_index(self.compare_to_var.get(), fallback=min(1, self.result.n_states - 1))
            feature_name = self.feature_var.get() or self.result.feature_names[0]
            axes = self.main_figure.subplots(1, 3)
            first, second, delta_axis = axes

            map_a = self.result.features_by_state[from_index][feature_name]
            map_b = self.result.features_by_state[to_index][feature_name]
            delta = map_b - map_a
            vmax = float(np.nanpercentile(np.abs(delta[self.result.valid_mask]), 99)) if np.any(self.result.valid_mask) else float(np.nanmax(np.abs(delta)))
            if vmax == 0:
                vmax = 1e-6

            image_a = first.imshow(map_a.T, origin="lower", cmap="viridis", aspect="auto")
            image_b = second.imshow(map_b.T, origin="lower", cmap="viridis", aspect="auto")
            image_delta = delta_axis.imshow(
                delta.T,
                origin="lower",
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                aspect="auto",
            )

            first.set_title(f"{self.result.state_names[from_index]}\n{feature_name}")
            second.set_title(f"{self.result.state_names[to_index]}\n{feature_name}")
            delta_axis.set_title(f"Difference\n{self.result.state_names[to_index]} - {self.result.state_names[from_index]}")
            for axis in axes:
                axis.set_xlabel("x index")
                axis.set_ylabel("y index")
                self._mark_selected_pixel(axis)

            self.main_figure.colorbar(image_a, ax=first, fraction=0.046, pad=0.04)
            self.main_figure.colorbar(image_b, ax=second, fraction=0.046, pad=0.04)
            self.main_figure.colorbar(image_delta, ax=delta_axis, fraction=0.046, pad=0.04)

        elif view == "Cluster sequence map":
            axis = self.main_figure.add_subplot(111)
            self._plot_sequence_map(
                axis=axis,
                code_map=self.result.cluster_sequence_code_map,
                ranked_sequences=self.result.cluster_sequences,
                title="Most common per-pixel cluster sequences",
            )

        elif view == "Simple state sequence map":
            axis = self.main_figure.add_subplot(111)
            self._plot_sequence_map(
                axis=axis,
                code_map=self.result.simple_state_sequence_code_map,
                ranked_sequences=self.result.simple_state_sequences,
                title="Most common per-pixel simple-state sequences",
            )

        else:
            axis = self.main_figure.add_subplot(111)
            axis.text(0.5, 0.5, f"Unsupported view: {view}", ha="center", va="center")
            axis.set_axis_off()

        self.main_canvas.draw_idle()

    def _plot_sequence_map(
        self,
        axis: matplotlib.axes.Axes,
        code_map: np.ndarray,
        ranked_sequences: list[tuple[str, int]],
        title: str,
        max_labels: int = 12,
    ) -> None:
        visible_count = min(max_labels, len(ranked_sequences))
        if visible_count == 0:
            axis.text(0.5, 0.5, "No sequences available.", ha="center", va="center")
            axis.set_axis_off()
            return

        display = code_map.astype(float).copy()
        display[~self.result.valid_mask] = np.nan
        for hidden_code in range(visible_count, len(ranked_sequences)):
            display[code_map == hidden_code] = np.nan

        cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(visible_count, 1))
        cmap = cmap.copy()
        cmap.set_bad(color="lightgray")
        norm = mcolors.BoundaryNorm(np.arange(-0.5, visible_count + 0.5, 1), cmap.N)
        image = axis.imshow(display.T, origin="lower", cmap=cmap, norm=norm, aspect="auto")
        cbar = self.main_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, ticks=np.arange(visible_count))
        cbar.ax.set_yticklabels([ranked_sequences[index][0] for index in range(visible_count)])
        axis.set_title(title)
        axis.set_xlabel("x index")
        axis.set_ylabel("y index")
        self._mark_selected_pixel(axis)

    def _on_main_plot_click(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        if self.result is None or event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        x_index = int(round(event.xdata))
        y_index = int(round(event.ydata))
        x_size, y_size = self.result.shape
        if not (0 <= x_index < x_size and 0 <= y_index < y_size):
            return

        self.selected_pixel = (x_index, y_index)
        self._refresh_main_plot()
        self._update_pixel_details()

    def _update_pixel_details(self) -> None:
        self.pixel_figure.clear()

        if self.result is None or self.selected_pixel is None:
            axis = self.pixel_figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                "Click a point on the map to inspect the local spectrum across states.",
                ha="center",
                va="center",
                fontsize=12,
            )
            axis.set_axis_off()
            self.pixel_canvas.draw_idle()
            self._set_text_widget(self.pixel_text, "")
            return

        x_index, y_index = self.selected_pixel
        n_states = self.result.n_states
        grid = self.pixel_figure.add_gridspec(2, n_states, height_ratios=[2.2, 1.0])
        energy_axis = self.result.e_axis
        phi_axis = self.result.phi_axis

        for state_index, state in enumerate(self.result.loaded_states):
            axis = self.pixel_figure.add_subplot(grid[0, state_index])
            spectrum = np.asarray(state.data_array.values[x_index, y_index, :, :], dtype=np.float32)
            image = axis.imshow(
                spectrum,
                origin="lower",
                aspect="auto",
                extent=[float(phi_axis[0]), float(phi_axis[-1]), float(energy_axis[0]), float(energy_axis[-1])],
                cmap="viridis",
            )
            axis.set_title(Path(state.file_path).name)
            axis.set_xlabel("phi")
            if state_index == 0:
                axis.set_ylabel("eV")
            self.pixel_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        profile_axis = self.pixel_figure.add_subplot(grid[1, :])
        for state_index, state in enumerate(self.result.loaded_states):
            spectrum = np.asarray(state.data_array.values[x_index, y_index, :, :], dtype=np.float32)
            profile = spectrum.sum(axis=1)
            if np.nanmax(profile) > 0:
                profile = profile / np.nanmax(profile)
            profile_axis.plot(energy_axis, profile, label=Path(state.file_path).name)
        profile_axis.set_title("Normalized local energy profile (summed over phi)")
        profile_axis.set_xlabel("eV")
        profile_axis.set_ylabel("normalized intensity")
        profile_axis.legend(loc="best", fontsize=8)
        self.pixel_canvas.draw_idle()

        self._set_text_widget(self.pixel_text, self._build_pixel_text(x_index, y_index))

    def _update_summary_text(self) -> None:
        if self.result is None:
            self._set_text_widget(self.summary_text, "")
            return

        summary = self.result.summarize(max_sequences=12)
        lines = [
            "Files:",
            *[f"  - {path}" for path in summary["files"]],
            "",
            f"Valid pixels inside cross: {summary['valid_pixels']}",
            f"Excluded pixels: {summary['excluded_pixels']}",
            f"PCA explained variance ratio: {', '.join(f'{value:.3f}' for value in summary['pca_explained_ratio'])}",
            f"Cluster inertia: {summary['cluster_inertia']:.4g}",
            "",
            "Simple state thresholds:",
            f"  - insulating upper bound: {summary['simple_state_thresholds']['insulating_upper']:.6f}",
            f"  - metallic lower bound: {summary['simple_state_thresholds']['metallic_lower']:.6f}",
            "",
            "Ordered cluster mean ef_fraction:",
        ]
        for cluster_id, mean_ef in summary["cluster_mean_ef_fraction"].items():
            lines.append(f"  - C{cluster_id}: {mean_ef:.6f}")

        lines.extend(["", "Top cluster sequences:"])
        for entry in summary["top_cluster_sequences"]:
            lines.append(f"  - {entry['sequence']}: {entry['count']}")

        lines.extend(["", "Top simple-state sequences (I=insulating, X=intermediate, M=metallic):"])
        for entry in summary["top_simple_state_sequences"]:
            lines.append(f"  - {entry['sequence']}: {entry['count']}")

        if summary["notes"]:
            lines.extend(["", "Notes:"])
            for note in summary["notes"]:
                lines.append(f"  - {note}")

        self._set_text_widget(self.summary_text, "\n".join(lines))

    def _build_pixel_text(self, x_index: int, y_index: int) -> str:
        assert self.result is not None

        inside_cross = bool(self.result.valid_mask[x_index, y_index])
        lines = [
            f"Selected pixel: x={x_index}, y={y_index}",
            f"Inside cross mask: {'yes' if inside_cross else 'no'}",
            f"Cluster sequence: {self.result.cluster_sequence_strings[x_index, y_index]}",
            f"Simple-state sequence: {self.result.simple_state_sequence_strings[x_index, y_index]}",
            "",
        ]

        for state_index, state in enumerate(self.result.loaded_states):
            feature_map = self.result.features_by_state[state_index]
            cluster_id = int(self.result.cluster_maps[state_index][x_index, y_index])
            state_label = str(self.result.simple_state_label_maps[state_index][x_index, y_index])
            lines.append(Path(state.file_path).name)
            lines.append(f"  Cluster: {'outside-cross' if cluster_id < 0 else f'C{cluster_id}'}")
            lines.append(f"  Simple state: {state_label}")
            lines.append(f"  ef_fraction: {feature_map['ef_fraction'][x_index, y_index]:.6f}")
            lines.append(f"  spectral_entropy: {feature_map['spectral_entropy'][x_index, y_index]:.6f}")
            lines.append(f"  spectral_sharpness: {feature_map['spectral_sharpness'][x_index, y_index]:.6f}")
            lines.append(f"  e_centroid: {feature_map['e_centroid'][x_index, y_index]:.6f}")
            lines.append("")

        return "\n".join(lines).rstrip()

    def _set_text_widget(self, widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _mark_selected_pixel(self, axis: matplotlib.axes.Axes) -> None:
        if self.selected_pixel is None:
            return
        x_index, y_index = self.selected_pixel
        axis.scatter([x_index], [y_index], s=80, facecolors="none", edgecolors="white", linewidths=1.8)
        axis.scatter([x_index], [y_index], s=16, c="black")

    def _current_state_index(self) -> int:
        assert self.result is not None
        try:
            return self.result.state_names.index(self.state_var.get())
        except ValueError:
            return 0

    def _current_compare_index(self, value: str, fallback: int) -> int:
        assert self.result is not None
        try:
            return self.result.state_names.index(value)
        except ValueError:
            return fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the TaSe2 phase switching desktop analysis app.")
    parser.add_argument(
        "files",
        nargs="*",
        help="Optional NetCDF files to preload in sequence order.",
    )
    parser.add_argument(
        "--headless-smoke-test",
        action="store_true",
        help="Validate imports and exit without opening the GUI.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.headless_smoke_test:
        AnalysisParameters().validate()
        print("Desktop app imports and parameter validation succeeded.")
        return 0

    initial_files = [str(Path(path).expanduser().resolve()) for path in args.files]
    if len(initial_files) > 4:
        parser.error("At most four files can be preloaded.")

    root = tk.Tk()
    AnalysisApp(root, initial_files=initial_files)
    root.mainloop()
    return 0
