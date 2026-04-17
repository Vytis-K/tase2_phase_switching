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
import numpy as np

from .cdw_model import (
    CDWParameters,
    SequencePulse,
    SequenceSimulationResult,
    TransportParameters,
    build_default_sequence_pulses,
    calibrate_dataset_replay,
    render_sequence_to_netcdf,
    simulate_sequence,
)
from .geometry_inference import (
    GeometryInferenceParameters,
    GeometryInferenceResult,
    SimulationGeometry,
    build_flat_geometry,
    build_gradient_geometry,
    infer_geometry_from_files,
    normalize_inside_mask,
)


FILE_TYPES = [
    ("NetCDF files", "*.nc *.nc4 *.h5 *.hdf5"),
    ("All files", "*.*"),
]


class SimulationApp:
    VIEW_OPTIONS = ("Overview", "Geometry", "Transport", "Residual")
    GEOMETRY_MODES = ("Dataset inferred", "Gradient synthetic", "Flat synthetic")

    def __init__(self, root: tk.Tk, initial_files: list[str] | None = None) -> None:
        self.root = root
        self.root.title("TaSe2 Current + CDW Simulator")
        self.root.geometry("1760x1040")
        self.root.minsize(1360, 860)

        self.file_paths: list[str] = []
        self.geometry_inference: GeometryInferenceResult | None = None
        self.geometry: SimulationGeometry | None = None
        self.simulation_result: SequenceSimulationResult | None = None
        self.last_export_paths: list[Path] = []
        self.controls_canvas: tk.Canvas | None = None
        self.controls_container: ttk.Frame | None = None

        self.status_var = tk.StringVar(
            value="Load a dataset sequence or build a synthetic geometry, then infer/replay the switching."
        )
        self.view_var = tk.StringVar(value=self.VIEW_OPTIONS[0])
        self.state_var = tk.StringVar(value="baseline")
        self.geometry_mode_var = tk.StringVar(value=self.GEOMETRY_MODES[0])

        self.parameter_vars = {
            "shape_x": tk.StringVar(value="61"),
            "shape_y": tk.StringVar(value="61"),
            "flat_thickness": tk.StringVar(value="1.0"),
            "gradient_angle_deg": tk.StringVar(value="35.0"),
            "gradient_strength": tk.StringVar(value="0.35"),
            "defect_strength": tk.StringVar(value="0.45"),
            "pulse_a_angle_deg": tk.StringVar(value="35.0"),
            "pulse_b_angle_deg": tk.StringVar(value="125.0"),
            "pulse_a_amplitude": tk.StringVar(value="1.0"),
            "pulse_b_amplitude": tk.StringVar(value="0.92"),
            "pulse_b_repeats": tk.StringVar(value="1"),
            "contact_width_fraction": tk.StringVar(value="0.22"),
            "solver_iterations": tk.StringVar(value="280"),
            "particle_count": tk.StringVar(value="320"),
            "particle_steps": tk.StringVar(value="64"),
            "particle_diffusion": tk.StringVar(value="0.08"),
            "switch_threshold": tk.StringVar(value="0.32"),
            "write_gain": tk.StringVar(value="0.95"),
            "erase_gain": tk.StringVar(value="0.55"),
            "intermediate_gain": tk.StringVar(value="0.45"),
            "pinning_strength": tk.StringVar(value="0.85"),
            "relaxation": tk.StringVar(value="0.07"),
            "diffusion": tk.StringVar(value="0.10"),
            "e_points": tk.StringVar(value="96"),
            "phi_points": tk.StringVar(value="96"),
        }

        self._build_ui()
        if initial_files:
            self._set_files(initial_files)
        self._refresh_state_combo()
        self._refresh_plot()

    def _build_ui(self) -> None:
        main_pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls_host = ttk.Frame(main_pane, padding=(12, 12, 0, 12))
        visuals = ttk.Frame(main_pane, padding=(0, 12, 12, 12))
        main_pane.add(controls_host, weight=0)
        main_pane.add(visuals, weight=1)

        self._build_scrollable_controls_panel(controls_host)
        self._build_visual_panel(visuals)

        status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(12, 6))
        status_bar.pack(fill=tk.X)

    def _build_scrollable_controls_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        canvas = tk.Canvas(parent, highlightthickness=0, width=430)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        container = ttk.Frame(canvas, padding=(0, 0, 12, 0))

        self.controls_canvas = canvas
        self.controls_container = container

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        window_id = canvas.create_window((0, 0), window=container, anchor="nw")
        container.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window_id, width=event.width),
        )

        self._build_controls_panel(container)
        self._bind_mousewheel_recursive(container)

    def _build_controls_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        files_frame = ttk.LabelFrame(parent, text="Dataset Sequence", padding=10)
        files_frame.grid(row=0, column=0, sticky="nsew")
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)

        self.file_listbox = tk.Listbox(files_frame, height=8, exportselection=False)
        self.file_listbox.grid(row=0, column=0, columnspan=2, sticky="nsew")
        ttk.Button(files_frame, text="Load a-b-c-d", command=self._load_bundled_smoke_test).grid(
            row=1, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Button(files_frame, text="Add Files", command=self._add_files).grid(
            row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0)
        )
        ttk.Button(files_frame, text="Remove Selected", command=self._remove_selected_files).grid(
            row=2, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Button(files_frame, text="Clear Files", command=self._clear_files).grid(
            row=2, column=1, sticky="ew", padx=(8, 0), pady=(8, 0)
        )

        geometry_frame = ttk.LabelFrame(parent, text="Geometry", padding=10)
        geometry_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        geometry_frame.columnconfigure(1, weight=1)

        ttk.Label(geometry_frame, text="Mode").grid(row=0, column=0, sticky="w")
        geometry_combo = ttk.Combobox(
            geometry_frame,
            textvariable=self.geometry_mode_var,
            values=self.GEOMETRY_MODES,
            state="readonly",
            width=22,
        )
        geometry_combo.grid(row=0, column=1, sticky="ew", pady=2)

        self._add_entry(geometry_frame, 1, "Shape x", "shape_x")
        self._add_entry(geometry_frame, 2, "Shape y", "shape_y")
        self._add_entry(geometry_frame, 3, "Flat thickness", "flat_thickness")
        self._add_entry(geometry_frame, 4, "Gradient angle", "gradient_angle_deg")
        self._add_entry(geometry_frame, 5, "Gradient strength", "gradient_strength")
        self._add_entry(geometry_frame, 6, "Defect strength", "defect_strength")

        pulse_frame = ttk.LabelFrame(parent, text="Pulse Preset", padding=10)
        pulse_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self._add_entry(pulse_frame, 0, "Pulse A angle", "pulse_a_angle_deg")
        self._add_entry(pulse_frame, 1, "Pulse B angle", "pulse_b_angle_deg")
        self._add_entry(pulse_frame, 2, "Pulse A amplitude", "pulse_a_amplitude")
        self._add_entry(pulse_frame, 3, "Pulse B amplitude", "pulse_b_amplitude")
        self._add_entry(pulse_frame, 4, "Pulse B repeats", "pulse_b_repeats")

        transport_frame = ttk.LabelFrame(parent, text="Transport + CDW", padding=10)
        transport_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        self._add_entry(transport_frame, 0, "Contact width", "contact_width_fraction")
        self._add_entry(transport_frame, 1, "Solver iterations", "solver_iterations")
        self._add_entry(transport_frame, 2, "Particle count", "particle_count")
        self._add_entry(transport_frame, 3, "Particle steps", "particle_steps")
        self._add_entry(transport_frame, 4, "Particle diffusion", "particle_diffusion")
        self._add_entry(transport_frame, 5, "Switch threshold", "switch_threshold")
        self._add_entry(transport_frame, 6, "Write gain", "write_gain")
        self._add_entry(transport_frame, 7, "Erase gain", "erase_gain")
        self._add_entry(transport_frame, 8, "Intermediate gain", "intermediate_gain")
        self._add_entry(transport_frame, 9, "Pinning strength", "pinning_strength")
        self._add_entry(transport_frame, 10, "Relaxation", "relaxation")
        self._add_entry(transport_frame, 11, "Diffusion", "diffusion")

        export_frame = ttk.LabelFrame(parent, text="Export", padding=10)
        export_frame.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        self._add_entry(export_frame, 0, "Energy points", "e_points")
        self._add_entry(export_frame, 1, "Phi points", "phi_points")

        actions_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        actions_frame.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        actions_frame.columnconfigure(0, weight=1)
        ttk.Button(actions_frame, text="Infer Geometry", command=self._infer_geometry).grid(row=0, column=0, sticky="ew")
        ttk.Button(actions_frame, text="Run Synthetic Sequence", command=self._run_custom_sequence).grid(
            row=1, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Button(actions_frame, text="Run Dataset Replay", command=self._run_dataset_replay).grid(
            row=2, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Button(actions_frame, text="Export Synthetic NetCDF...", command=self._export_synthetic_netcdf).grid(
            row=3, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Button(actions_frame, text="Save Current Plot...", command=self._save_current_plot).grid(
            row=4, column=0, sticky="ew", pady=(8, 0)
        )

        help_text = (
            "Recommended flow:\n"
            "1. Load the bundled a->b->c->d sequence.\n"
            "2. Click Infer Geometry to estimate thickness / pinning / pulse directions.\n"
            "3. Click Run Dataset Replay for the smoke-test fit.\n"
            "4. Tweak the synthetic geometry or pulse controls and re-run.\n"
            "Use the scrollbar or mouse wheel if the action buttons are below the fold."
        )
        ttk.Label(parent, text=help_text, justify=tk.LEFT, wraplength=340).grid(row=6, column=0, sticky="ew", pady=(12, 0))

    def _build_visual_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        parent.rowconfigure(2, weight=0)

        controls = ttk.Frame(parent)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Label(controls, text="View").grid(row=0, column=0, sticky="w")
        view_combo = ttk.Combobox(
            controls,
            textvariable=self.view_var,
            values=self.VIEW_OPTIONS,
            state="readonly",
            width=20,
        )
        view_combo.grid(row=1, column=0, sticky="w", padx=(0, 10))
        view_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_plot())

        ttk.Label(controls, text="State").grid(row=0, column=1, sticky="w")
        self.state_combo = ttk.Combobox(controls, textvariable=self.state_var, state="readonly", width=26)
        self.state_combo.grid(row=1, column=1, sticky="w")
        self.state_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_plot())

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.grid(row=2, column=0, sticky="ew")

        self.figure = Figure(figsize=(12.0, 8.4), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.LEFT, fill=tk.X)

        summary_frame = ttk.LabelFrame(parent, text="Summary", padding=8)
        summary_frame.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)
        self.summary_text = tk.Text(summary_frame, height=12, wrap="word")
        self.summary_text.grid(row=0, column=0, sticky="nsew")
        self.summary_text.configure(state="disabled")

    def _add_entry(self, parent: ttk.LabelFrame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(parent, textvariable=self.parameter_vars[key], width=16)
        entry.grid(row=row, column=1, sticky="e", padx=(10, 0), pady=2)

    def _bind_mousewheel_recursive(self, widget: tk.Misc) -> None:
        widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
        widget.bind("<Button-4>", self._on_mousewheel_linux_up, add="+")
        widget.bind("<Button-5>", self._on_mousewheel_linux_down, add="+")

        for child in widget.winfo_children():
            self._bind_mousewheel_recursive(child)

    def _on_mousewheel(self, event: tk.Event) -> str | None:
        if self.controls_canvas is None:
            return None
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return None
        step = -1 if delta > 0 else 1
        self.controls_canvas.yview_scroll(step, "units")
        return "break"

    def _on_mousewheel_linux_up(self, _event: tk.Event) -> str | None:
        if self.controls_canvas is None:
            return None
        self.controls_canvas.yview_scroll(-1, "units")
        return "break"

    def _on_mousewheel_linux_down(self, _event: tk.Event) -> str | None:
        if self.controls_canvas is None:
            return None
        self.controls_canvas.yview_scroll(1, "units")
        return "break"

    def _load_bundled_smoke_test(self) -> None:
        root = Path(__file__).resolve().parents[2]
        candidate_paths = [
            root / "data" / "a_convert_2_nosm.nc",
            root / "data" / "b_convert_2_nosm.nc",
            root / "data" / "c_convert_2_nosm.nc",
            root / "data" / "d_convert_2_nosm.nc",
        ]
        missing = [str(path) for path in candidate_paths if not path.exists()]
        if missing:
            messagebox.showerror("Bundled Files Missing", "Could not find:\n" + "\n".join(missing))
            return
        self._set_files([str(path) for path in candidate_paths])
        self.status_var.set("Loaded the bundled a->b->c->d smoke-test sequence.")

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(title="Choose NetCDF files", filetypes=FILE_TYPES)
        if not paths:
            return
        self._set_files(list(paths))

    def _remove_selected_files(self) -> None:
        selection = list(self.file_listbox.curselection())
        if not selection:
            return
        for index in reversed(selection):
            del self.file_paths[index]
        self._sync_file_listbox()

    def _clear_files(self) -> None:
        self.file_paths = []
        self.geometry_inference = None
        self.geometry = None
        self.simulation_result = None
        self._sync_file_listbox()
        self._refresh_state_combo()
        self._refresh_plot()
        self.status_var.set("Cleared the dataset sequence.")

    def _set_files(self, paths: list[str]) -> None:
        deduped: list[str] = []
        seen: set[str] = set()
        for path in paths:
            resolved = str(Path(path).expanduser().resolve())
            if resolved not in seen:
                deduped.append(resolved)
                seen.add(resolved)
        self.file_paths = deduped
        self.geometry_inference = None
        self.geometry = None
        self.simulation_result = None
        self._sync_file_listbox()
        self._refresh_state_combo()
        self._refresh_plot()

    def _sync_file_listbox(self) -> None:
        self.file_listbox.delete(0, tk.END)
        for path in self.file_paths:
            self.file_listbox.insert(tk.END, Path(path).name)

    def _infer_geometry(self) -> None:
        try:
            self.status_var.set("Building geometry...")
            self.root.update_idletasks()
            mode = self.geometry_mode_var.get()
            if mode == "Dataset inferred":
                if not self.file_paths:
                    raise ValueError("Choose a dataset sequence first, or switch to a synthetic geometry mode.")
                parameters = GeometryInferenceParameters()
                self.geometry_inference = infer_geometry_from_files(self.file_paths, parameters)
                self.geometry = self.geometry_inference.geometry
                self._apply_inferred_pulse_hints()
                self.status_var.set("Inferred sample geometry and pulse axes from the dataset sequence.")
            elif mode == "Gradient synthetic":
                self.geometry = build_gradient_geometry(
                    shape=self._geometry_shape(),
                    gradient_angle_deg=self._get_float("gradient_angle_deg"),
                    gradient_strength=self._get_float("gradient_strength"),
                    defect_strength=self._get_float("defect_strength"),
                )
                self.geometry_inference = None
                self.status_var.set("Built a synthetic gradient geometry.")
            else:
                self.geometry = build_flat_geometry(
                    shape=self._geometry_shape(),
                    thickness=self._get_float("flat_thickness"),
                )
                self.geometry_inference = None
                self.status_var.set("Built a flat synthetic geometry.")
            self.simulation_result = None
            self._refresh_state_combo()
            self._refresh_plot()
        except Exception as exc:
            messagebox.showerror("Geometry Error", str(exc))
            self.status_var.set("Geometry build failed.")

    def _run_custom_sequence(self) -> None:
        try:
            if self.geometry is None:
                self._infer_geometry()
            if self.geometry is None:
                return
            transport = self._transport_parameters()
            cdw = self._cdw_parameters()
            pulses = build_default_sequence_pulses(
                pulse_a_angle_deg=self._get_float("pulse_a_angle_deg"),
                pulse_b_angle_deg=self._get_float("pulse_b_angle_deg"),
                pulse_a_amplitude=self._get_float("pulse_a_amplitude"),
                pulse_b_amplitude=self._get_float("pulse_b_amplitude"),
                repeat_count=self._get_int("pulse_b_repeats"),
                contact_width_fraction=self._get_float("contact_width_fraction"),
                particle_count=self._get_int("particle_count"),
            )
            self.status_var.set("Running the synthetic pulse sequence...")
            self.root.update_idletasks()
            self.simulation_result = simulate_sequence(
                geometry=self.geometry,
                pulses=pulses,
                transport_parameters=transport,
                cdw_parameters=cdw,
            )
            self._refresh_state_combo()
            self._refresh_plot()
            self.status_var.set("Finished the synthetic pulse sequence.")
        except Exception as exc:
            messagebox.showerror("Simulation Error", str(exc))
            self.status_var.set("Simulation failed.")

    def _run_dataset_replay(self) -> None:
        try:
            if self.geometry_mode_var.get() != "Dataset inferred" or self.geometry_inference is None:
                self.geometry_mode_var.set("Dataset inferred")
                self._infer_geometry()
            if self.geometry is None:
                return
            transport = self._transport_parameters()
            cdw = self._cdw_parameters()
            self.status_var.set("Calibrating the dataset replay...")
            self.root.update_idletasks()
            pulses, result = calibrate_dataset_replay(self.geometry, transport_parameters=transport, cdw_parameters=cdw)
            self.simulation_result = result
            if pulses:
                self.parameter_vars["pulse_a_angle_deg"].set(f"{pulses[0].angle_deg:.2f}")
                if len(pulses) > 1:
                    self.parameter_vars["pulse_b_angle_deg"].set(f"{pulses[1].angle_deg:.2f}")
            self._refresh_state_combo()
            self._refresh_plot()
            self.status_var.set("Finished the dataset replay smoke test.")
        except Exception as exc:
            messagebox.showerror("Replay Error", str(exc))
            self.status_var.set("Dataset replay failed.")

    def _export_synthetic_netcdf(self) -> None:
        if self.simulation_result is None:
            messagebox.showinfo("Nothing To Export", "Run a synthetic sequence or dataset replay first.")
            return
        output_dir = filedialog.askdirectory(title="Choose an export folder")
        if not output_dir:
            return
        try:
            e_points = self._get_int("e_points")
            phi_points = self._get_int("phi_points")
            written = render_sequence_to_netcdf(
                self.simulation_result,
                output_dir,
                e_points=e_points,
                phi_points=phi_points,
                noise_level=0.005,
            )
            self.last_export_paths = written
            self.status_var.set(f"Exported {len(written)} synthetic NetCDF files to {output_dir}")
            self._refresh_summary_text()
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))
            self.status_var.set("Synthetic export failed.")

    def _save_current_plot(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save current plot",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return
        self.figure.savefig(path, dpi=180)
        self.status_var.set(f"Saved the current figure to {path}")

    def _refresh_state_combo(self) -> None:
        values: list[str] = []
        if self.simulation_result is not None:
            values = self.simulation_result.state_names
        elif self.geometry is not None and self.geometry.state_names:
            values = list(self.geometry.state_names)
        if not values:
            values = ["baseline"]
        self.state_combo.configure(values=values)
        if self.state_var.get() not in values:
            self.state_var.set(values[0])

    def _refresh_plot(self) -> None:
        self.figure.clear()
        view = self.view_var.get()
        if view == "Geometry":
            self._render_geometry_view()
        elif view == "Transport":
            self._render_transport_view()
        elif view == "Residual":
            self._render_residual_view()
        else:
            self._render_overview_view()
        self.canvas.draw_idle()
        self._refresh_summary_text()

    def _render_overview_view(self) -> None:
        axes = self.figure.subplots(2, 2)
        state_index = self._current_state_index()
        geometry = self.geometry
        if geometry is None:
            self._render_placeholder("Infer a geometry or run a replay to populate the simulator.")
            return

        simulated = self._state_observable(state_index)
        target = self._state_target(state_index, geometry)
        current = self._state_current_map(state_index)
        metallic = self._state_phase_map(state_index, phase_index=2)

        self._show_map(axes[0, 0], simulated, f"Simulated observable: {self._state_name(state_index)}", cmap="magma")
        self._show_map(axes[0, 1], target, "Observed target / reference", cmap="magma")
        self._show_map(axes[1, 0], current, "Current magnitude", cmap="viridis")
        self._show_map(axes[1, 1], metallic, "Metallic fraction", cmap="inferno", vmin=0.0, vmax=1.0)

    def _render_geometry_view(self) -> None:
        axes = self.figure.subplots(2, 2)
        geometry = self.geometry
        if geometry is None:
            self._render_placeholder("Build or infer a geometry to inspect the sample fields.")
            return
        self._show_map(axes[0, 0], geometry.sample_mask.astype(np.float32), "Sample mask", cmap="gray_r")
        self._show_map(axes[0, 1], geometry.thickness_map, "Thickness map", cmap="terrain")
        self._show_map(axes[1, 0], geometry.roughness_map, "Geometry roughness", cmap="cividis", vmin=0.0, vmax=1.0)
        self._show_map(axes[1, 1], geometry.boundary_pinning_map, "Boundary pinning", cmap="plasma", vmin=0.0, vmax=1.0)

    def _render_transport_view(self) -> None:
        axes = self.figure.subplots(2, 2)
        state_index = self._current_state_index()
        if self.simulation_result is None or state_index == 0:
            self._render_placeholder("Run a pulse sequence and choose a non-baseline state to inspect transport fields.")
            return
        step = self.simulation_result.steps[state_index]
        if step.current_field is None:
            self._render_placeholder("This state does not carry a transport field.")
            return
        self._show_map(axes[0, 0], step.conductivity_map, "Conductivity map", cmap="viridis")
        self._show_map(axes[0, 1], step.current_field.potential_map, "Potential field", cmap="coolwarm")
        self._show_map(axes[1, 0], step.current_field.current_magnitude, "Current magnitude", cmap="magma")
        self._show_map(axes[1, 1], step.current_field.particle_density, "Particle density", cmap="plasma")

    def _render_residual_view(self) -> None:
        axes = self.figure.subplots(2, 2)
        state_index = self._current_state_index()
        geometry = self.geometry
        if geometry is None:
            self._render_placeholder("Infer a geometry first.")
            return

        simulated = self._state_observable(state_index)
        target = self._state_target(state_index, geometry)
        residual = simulated - target
        intermediate = self._state_phase_map(state_index, phase_index=1)

        self._show_map(axes[0, 0], simulated, "Simulated observable", cmap="magma")
        self._show_map(axes[0, 1], target, "Target observable", cmap="magma")
        self._show_map(axes[1, 0], residual, "Residual (sim - target)", cmap="coolwarm")
        self._show_map(axes[1, 1], intermediate, "Intermediate fraction", cmap="YlOrBr", vmin=0.0, vmax=1.0)

    def _render_placeholder(self, text: str) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.axis("off")
        axis.text(0.5, 0.5, text, ha="center", va="center", fontsize=13)

    def _show_map(
        self,
        axis,
        image: np.ndarray,
        title: str,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        data = np.asarray(image, dtype=np.float32)
        im = axis.imshow(data.T, origin="lower", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        self.figure.colorbar(im, ax=axis, shrink=0.85)

    def _refresh_summary_text(self) -> None:
        lines: list[str] = []
        if self.geometry is not None:
            geometry = self.geometry
            lines.append(f"Geometry: {geometry.name}")
            lines.append(f"Shape: {geometry.shape[0]} x {geometry.shape[1]}")
            lines.append(f"Active pixels: {int(geometry.sample_mask.sum())}")
            if geometry.pulse_reports:
                lines.append("Inferred pulse directions:")
                for report in geometry.pulse_reports:
                    lines.append(f"  {report.summary}")
            if geometry.notes:
                lines.append("Geometry notes:")
                for note in geometry.notes:
                    lines.append(f"  {note}")

        if self.simulation_result is not None:
            result = self.simulation_result
            if result.score is not None:
                lines.append(f"Replay score: {result.score:.3f}")
            if result.correlations:
                corr_text = ", ".join(f"{value:.3f}" for value in result.correlations)
                lines.append(f"State correlations: {corr_text}")
            if result.rmse_values:
                rmse_text = ", ".join(f"{value:.3f}" for value in result.rmse_values)
                lines.append(f"State RMSE: {rmse_text}")
            state_index = self._current_state_index()
            state_name = self._state_name(state_index)
            lines.append(f"Current selection: {state_name}")
            phase_map = self._state_phase_map(state_index, phase_index=2)
            lines.append(f"Metallic mean: {float(phase_map.mean()):.3f}")
            if result.notes:
                for note in result.notes:
                    lines.append(note)

        if self.last_export_paths:
            lines.append("Last export:")
            for path in self.last_export_paths:
                lines.append(f"  {path.name}")

        if not lines:
            lines.append("No geometry or simulation loaded yet.")

        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", "\n".join(lines))
        self.summary_text.configure(state="disabled")

    def _state_name(self, state_index: int) -> str:
        if self.simulation_result is not None:
            return self.simulation_result.state_names[state_index]
        if self.geometry is not None and self.geometry.state_names:
            return self.geometry.state_names[min(state_index, len(self.geometry.state_names) - 1)]
        return "baseline"

    def _state_observable(self, state_index: int) -> np.ndarray:
        if self.simulation_result is not None:
            return self.simulation_result.steps[state_index].normalized_observable_map
        if self.geometry is not None:
            return normalize_inside_mask(self.geometry.average_ef_fraction_map, self.geometry.sample_mask)
        return np.zeros((10, 10), dtype=np.float32)

    def _state_target(self, state_index: int, geometry: SimulationGeometry) -> np.ndarray:
        if geometry.target_observable_maps:
            index = min(state_index, len(geometry.target_observable_maps) - 1)
            return normalize_inside_mask(geometry.target_observable_maps[index], geometry.sample_mask)
        return normalize_inside_mask(geometry.average_ef_fraction_map, geometry.sample_mask)

    def _state_current_map(self, state_index: int) -> np.ndarray:
        if self.simulation_result is not None and state_index > 0:
            step = self.simulation_result.steps[state_index]
            if step.current_field is not None:
                return normalize_inside_mask(step.current_field.current_magnitude, self.simulation_result.geometry.sample_mask)
        if self.geometry is not None:
            return normalize_inside_mask(self.geometry.boundary_pinning_map, self.geometry.sample_mask)
        return np.zeros((10, 10), dtype=np.float32)

    def _state_phase_map(self, state_index: int, phase_index: int) -> np.ndarray:
        if self.simulation_result is not None:
            return self.simulation_result.steps[state_index].phase_weights[..., phase_index]
        if self.geometry is not None:
            if phase_index == 2:
                return normalize_inside_mask(self.geometry.average_ef_fraction_map, self.geometry.sample_mask)
            if phase_index == 1:
                return np.where(self.geometry.sample_mask, 0.25, 0.0).astype(np.float32)
            return np.where(self.geometry.sample_mask, 0.75, 0.0).astype(np.float32)
        return np.zeros((10, 10), dtype=np.float32)

    def _current_state_index(self) -> int:
        current = self.state_var.get()
        values = list(self.state_combo.cget("values"))
        if current in values:
            return values.index(current)
        return 0

    def _apply_inferred_pulse_hints(self) -> None:
        if self.geometry is None or not self.geometry.pulse_reports:
            return
        first = self.geometry.pulse_reports[0]
        self.parameter_vars["pulse_a_angle_deg"].set(f"{first.angle_deg:.2f}")
        if len(self.geometry.pulse_reports) > 1:
            second = self.geometry.pulse_reports[1]
            self.parameter_vars["pulse_b_angle_deg"].set(f"{second.angle_deg:.2f}")

    def _geometry_shape(self) -> tuple[int, int]:
        return self._get_int("shape_x"), self._get_int("shape_y")

    def _transport_parameters(self) -> TransportParameters:
        return TransportParameters(
            contact_width_fraction=self._get_float("contact_width_fraction"),
            solver_iterations=self._get_int("solver_iterations"),
            particle_count=self._get_int("particle_count"),
            particle_steps=self._get_int("particle_steps"),
            particle_diffusion=self._get_float("particle_diffusion"),
        )

    def _cdw_parameters(self) -> CDWParameters:
        return CDWParameters(
            switch_threshold=self._get_float("switch_threshold"),
            write_gain=self._get_float("write_gain"),
            erase_gain=self._get_float("erase_gain"),
            intermediate_gain=self._get_float("intermediate_gain"),
            pinning_strength=self._get_float("pinning_strength"),
            relaxation=self._get_float("relaxation"),
            diffusion=self._get_float("diffusion"),
        )

    def _get_float(self, key: str) -> float:
        return float(self.parameter_vars[key].get().strip())

    def _get_int(self, key: str) -> int:
        return int(self.parameter_vars[key].get().strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the TaSe2 current/CDW simulator.")
    parser.add_argument("files", nargs="*", help="Optional NetCDF files to preload.")
    args = parser.parse_args(argv)

    root = tk.Tk()
    app = SimulationApp(root, initial_files=list(args.files) if args.files else None)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
