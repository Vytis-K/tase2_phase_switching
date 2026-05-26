from __future__ import annotations

import argparse
import os
from pathlib import Path
import queue
import tempfile
import threading
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
    ClusterPhysicalInterpretation,
    INITIAL_TRANSITION_GROUPS,
    INITIAL_TRANSITION_NORMALIZATION_MODES,
    LoadedState,
    SPECTRAL_CLUSTER_METHOD_LABELS,
    SIMPLE_STATE_COLORS,
    SIMPLE_STATE_NAMES,
    STATE_CLASSIFICATION_COLORS,
    STATE_CLASSIFICATION_FEATURE_NAMES,
    STATE_CLASSIFICATION_LABELS,
    SWITCHING_COLORS,
    SWITCHING_LABELS,
    SWITCHING_MECHANISM_EDC_NORMALIZATIONS,
    TRANSITION_OUTCOME_COLORS,
    TRANSITION_OUTCOME_LABELS,
    SpectralClusterParameters,
    SpectralClusterResult,
    StateClassificationResult,
    StateClassifierParameters,
    StatePredictionParameters,
    StatePredictionResult,
    SwitchingMechanismDiagnosticsResult,
    SwitchingMechanismParameters,
    SwitchingMapParameters,
    SwitchingMapResult,
    InitialTransitionFeatureParameters,
    InitialTransitionFeatureResult,
    TransitionOutcomeParameters,
    TransitionOutcomeResult,
    align_loaded_states_for_comparison,
    analyze_cluster_physical_interpretation,
    build_cross_mask_from_maps,
    build_simple_state_maps,
    classify_state_feature_maps,
    extract_pixel_features,
    export_cluster_physical_interpretation,
    export_analysis,
    export_initial_transition_feature_analysis,
    export_switching_mechanism_diagnostics,
    export_state_classification,
    export_state_prediction,
    export_switching_map,
    export_transition_outcome_maps,
    load_state,
    run_analysis,
    run_initial_transition_feature_analysis,
    run_switching_mechanism_diagnostics,
    run_spectral_clustering,
    run_state_classification,
    run_state_prediction,
    run_switching_map,
    run_transition_outcome_maps,
    total_and_ef_maps,
)


FILE_TYPES = [
    ("NetCDF files", "*.nc *.nc4 *.h5 *.hdf5"),
    ("All files", "*.*"),
]


class AnalysisApp:
    ANALYSIS_PANEL_OPTIONS = [
        "Analysis",
        "Sequence Viewer",
        "Initial-State Changes",
        "EDC/MDC Compare",
        "Feature Search",
        "Clustering",
        "Switching Map",
        "State Prediction",
        "Transition Outcome Maps",
        "Initial State Transition Features",
        "Switching Mechanism Diagnostics",
    ]
    ANALYSIS_PANEL_MIN_FILES = {
        "Analysis": 1,
        "Sequence Viewer": 1,
        "Initial-State Changes": 1,
        "EDC/MDC Compare": 2,
        "Feature Search": 2,
        "Clustering": 1,
        "Switching Map": 2,
        "State Prediction": 2,
        "Transition Outcome Maps": 2,
        "Initial State Transition Features": 2,
        "Switching Mechanism Diagnostics": 2,
    }
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
        "State comparison",
    ]
    CLUSTER_COLOR_OPTIONS = {
        "Cluster label": "cluster",
        "Near-EF fraction": "ef_fraction",
        "Total intensity": "total_intensity",
        "Spectral entropy": "spectral_entropy",
        "Energy centroid": "e_centroid",
    }
    CHANGE_METRIC_OPTIONS = {
        "Near-EF fraction": "ef_fraction",
        "Near-EF intensity": "ef_intensity",
        "Wide-window fraction": "wide_fraction",
        "Energy centroid": "e_centroid",
        "Total intensity": "total_intensity",
        "Spectral entropy": "spectral_entropy",
    }
    SEQUENCE_MAP_OPTIONS = {
        "Total intensity": "total_intensity",
        "Near-EF intensity": "ef_intensity",
        "Near-EF fraction": "ef_fraction",
    }
    CHANGE_TRANSITION_LABELS = [
        "I->I", "I->X", "I->M",
        "X->I", "X->X", "X->M",
        "M->I", "M->X", "M->M",
    ]
    CHANGE_TRANSITION_COLORS = [
        "#1f3b73",
        "#6fa8dc",
        "#ff6600",
        "#a4c2f4",
        "#aaaaaa",
        "#ff9900",
        "#0a42a8",
        "#6d9eeb",
        "#d62728",
    ]
    CURVE_MAP_OPTIONS = {
        "Total intensity": "total_intensity",
        "Near-EF intensity": "ef_intensity",
        "Near-EF fraction": "ef_fraction",
    }
    FEATURE_MAP_OPTIONS = {
        "Special feature score": "score",
        "Spectral shape change": "spectral_rms",
        "Near-EF fraction change": "delta_ef_fraction",
        "Energy centroid shift": "delta_e_centroid",
        "Total intensity change": "delta_total_intensity",
        "Entropy change": "delta_spectral_entropy",
    }
    STATE_CLASSIFIER_MAP_OPTIONS = {
        "Classified state": "state_code",
        "Total intensity T": "T",
        "Near-EF weight W_EF": "W_EF",
        "LHB / p1 weight W_LHB": "W_LHB",
        "Metallicity ratio I_rat": "I_rat",
        "LHB / p1 peak E_LHB": "E_LHB",
        "Leading edge E_LE": "E_LE",
        "EDC linewidth Gamma": "Gamma_EDC",
        "Orientation marker S_orient": "S_orient",
        "Normalized I_rat": "Irat_norm",
        "Normalized W_EF": "WEF_norm",
        "Leading-edge closeness": "LE_closeness_norm",
        "Normalized linewidth": "Gamma_norm",
        "Orientation shift": "Orient_shift_norm",
    }
    TRANSITION_OUTCOME_MAP_OPTIONS = {
        "Transition label": "transition_label",
        "Delta I_rat": "delta_irat",
        "Relative Delta I_rat": "relative_delta_irat",
        "abs Delta_Irat": "abs_delta_irat",
        "Written mask": "written_mask",
        "Erased mask": "erased_mask",
        "File I_rat": "file_irat",
        "File W_EF": "file_wef",
        "File total intensity": "file_total",
        "Global summary": "global_summary",
    }
    INITIAL_TRANSITION_AGGREGATE_MAP_OPTIONS = {
        "Metallic count": "metallic_count",
        "Erased count": "erased_count",
        "Stable count": "stable_count",
        "Metallic frequency": "metallic_frequency",
        "Erased frequency": "erased_frequency",
        "Stable frequency": "stable_frequency",
        "Max metallicity score": "max_metallicity_score",
        "Mean metallicity score": "mean_metallicity_score",
        "Max erasure score": "max_erasure_score",
        "Mean erasure score": "mean_erasure_score",
        "First metallic transition": "first_metallic_transition",
        "First erased transition": "first_erased_transition",
    }

    def __init__(
        self,
        root: tk.Tk,
        initial_files: list[str] | None = None,
        upload_first: bool = False,
    ) -> None:
        self.root = root
        self.root.title("TaSe2 Phase Switching Analyzer")
        self.root.geometry("1680x1020")
        self.root.minsize(1320, 860)

        self.upload_first = upload_first
        self.uploaded_file_paths: list[str] = []
        self.file_paths: list[str] = []
        self.result: AnalysisResult | None = None
        self.cluster_result: SpectralClusterResult | None = None
        self.cluster_interpretation: ClusterPhysicalInterpretation | None = None
        self.selected_pixel: tuple[int, int] | None = None
        self.cluster_cache: dict[tuple[int, int, int, str], SpectralClusterResult] = {}
        self.cluster_interpretation_cache: dict[
            tuple[int, int, int, str],
            tuple[ClusterPhysicalInterpretation, dict[str, Path]],
        ] = {}
        self.cluster_popup: tk.Toplevel | None = None
        self.cluster_popup_summary_text: tk.Text | None = None
        self.cluster_popup_comparison_text: tk.Text | None = None
        self.cluster_popup_metrics_figure: Figure | None = None
        self.cluster_popup_metrics_canvas: FigureCanvasTkAgg | None = None
        self.cluster_popup_spectra_figure: Figure | None = None
        self.cluster_popup_spectra_canvas: FigureCanvasTkAgg | None = None

        self.sequence_file_paths: list[str] = []
        self.sequence_loaded_states: list[LoadedState] = []
        self.sequence_total_maps: list[np.ndarray] = []
        self.sequence_ef_maps: list[np.ndarray] = []
        self.sequence_selected_indices: list[int] = []
        self.sequence_selected_pixel: tuple[int, int] | None = None
        self.sequence_alignment_notes: list[str] = []
        self.sequence_map_axes: list[matplotlib.axes.Axes] = []
        self.sequence_axis_to_index: dict[matplotlib.axes.Axes, int] = {}
        self.sequence_pixel_marker_artists: list[object] = []
        self.sequence_compare_refresh_after_id: str | None = None

        self.change_file_paths: list[str] = []
        self.change_loaded_states: list[LoadedState] = []
        self.change_total_maps: list[np.ndarray] = []
        self.change_ef_maps: list[np.ndarray] = []
        self.change_features_by_state: list[dict[str, np.ndarray]] = []
        self.change_feature_names: list[str] = []
        self.change_valid_mask: np.ndarray | None = None
        self.change_average_map: np.ndarray | None = None
        self.change_simple_state_label_maps: list[np.ndarray] = []
        self.change_simple_state_code_maps: list[np.ndarray] = []
        self.change_simple_state_thresholds: tuple[float, float] | None = None
        self.change_mean_energy_profiles: list[np.ndarray] = []
        self.change_sequence_stats: list[dict[str, object]] = []
        self.change_initial_path: str | None = None
        self.change_target_path: str | None = None
        self.change_selected_pixel: tuple[int, int] | None = None
        self.change_drag_index: int | None = None

        self.curve_file_paths: list[str] = []
        self.curve_loaded_states: list[LoadedState] = []
        self.curve_total_maps: list[np.ndarray] = []
        self.curve_ef_maps: list[np.ndarray] = []
        self.curve_selected_pixel: tuple[int, int] | None = None
        self.curve_first_path: str | None = None
        self.curve_second_path: str | None = None
        self.curve_map_axes: list[matplotlib.axes.Axes] = []

        self.feature_file_paths: list[str] = []
        self.feature_loaded_states: list[LoadedState] = []
        self.feature_total_maps: list[np.ndarray] = []
        self.feature_ef_maps: list[np.ndarray] = []
        self.feature_features_by_state: list[dict[str, np.ndarray]] = []
        self.feature_valid_mask: np.ndarray | None = None
        self.feature_score_map: np.ndarray | None = None
        self.feature_metric_maps: dict[str, np.ndarray] = {}
        self.feature_hotspots: list[dict[str, object]] = []
        self.feature_selected_pixel: tuple[int, int] | None = None
        self.feature_first_path: str | None = None
        self.feature_second_path: str | None = None
        self.feature_map_axes: list[matplotlib.axes.Axes] = []

        self.classifier_file_path: str | None = None
        self.classifier_result: StateClassificationResult | None = None
        self.classifier_selected_pixel: tuple[int, int] | None = None
        self.classifier_map_axes: list[matplotlib.axes.Axes] = []

        self.switching_file_paths: list[str] = []
        self.switching_result: SwitchingMapResult | None = None
        self.switching_selected_pixel: tuple[int, int] | None = None
        self.switching_map_axes: list[matplotlib.axes.Axes] = []

        self.state_prediction_file_paths: list[str] = []
        self.state_prediction_result: StatePredictionResult | None = None
        self.state_prediction_selected_pixel: tuple[int, int] | None = None
        self.state_prediction_map_axes: list[matplotlib.axes.Axes] = []

        self.transition_outcome_file_paths: list[str] = []
        self.transition_outcome_result: TransitionOutcomeResult | None = None
        self.transition_outcome_selected_pixel: tuple[int, int] | None = None
        self.transition_outcome_hover_pixel: tuple[int, int] | None = None
        self.transition_outcome_focused_transition: int | None = None
        self.transition_outcome_map_axes: list[matplotlib.axes.Axes] = []
        self.transition_outcome_axis_to_transition: dict[matplotlib.axes.Axes, int] = {}
        self.transition_outcome_axis_to_file: dict[matplotlib.axes.Axes, int] = {}
        self.transition_outcome_axis_limits: dict[matplotlib.axes.Axes, tuple[tuple[float, float], tuple[float, float]]] = {}

        self.initial_transition_file_paths: list[str] = []
        self.initial_transition_excluded_indices: set[int] = set()
        self.initial_transition_result: InitialTransitionFeatureResult | None = None
        self.initial_transition_selected_pixel: tuple[int, int] | None = None
        self.initial_transition_map_axes: list[matplotlib.axes.Axes] = []
        self.mechanism_result: SwitchingMechanismDiagnosticsResult | None = None
        self.mechanism_selected_pixel: tuple[int, int] | None = None
        self.mechanism_map_axes: list[matplotlib.axes.Axes] = []
        self.mechanism_file_paths: list[str] = []
        self.mechanism_worker_thread: threading.Thread | None = None
        self.mechanism_worker_queue: queue.Queue[tuple[str, object]] | None = None

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
        cluster_defaults = SpectralClusterParameters()
        self.cluster_parameter_vars = {
            "n_clusters": tk.StringVar(value=str(cluster_defaults.n_clusters)),
            "embedding_components": tk.StringVar(value=str(cluster_defaults.embedding_components)),
        }

        self.status_var = tk.StringVar(
            value="Choose 1 to 4 NetCDF files, adjust the analysis parameters, then run the pipeline."
        )
        self.upload_status_var = tk.StringVar(value="Choose the NetCDF files to analyze.")
        self.global_progress_var = tk.DoubleVar(value=0.0)
        self.global_progress_status_var = tk.StringVar(value="Ready.")
        self.runner_panel_var = tk.StringVar(value=self.ANALYSIS_PANEL_OPTIONS[0])
        self.runner_status_var = tk.StringVar(value="Choose a panel, press Play, then open its result view.")
        self.runner_completed_panel: str | None = None
        self.runner_running_panel: str | None = None
        self.cluster_status_var = tk.StringVar(
            value="Run the main analysis, then use the Clustering panel to cluster registered spectra."
        )
        self.sequence_status_var = tk.StringVar(
            value="Add NetCDF files in sequence order, choose a map, then load the sequence viewer."
        )
        self.view_var = tk.StringVar(value=self.VIEW_OPTIONS[0])
        self.state_var = tk.StringVar(value="")
        self.feature_var = tk.StringVar(value="")
        self.compare_from_var = tk.StringVar(value="")
        self.compare_to_var = tk.StringVar(value="")
        self.cluster_state_var = tk.StringVar(value="")
        self.cluster_method_var = tk.StringVar(value=SPECTRAL_CLUSTER_METHOD_LABELS[cluster_defaults.method_key])
        self.cluster_color_var = tk.StringVar(value="Near-EF fraction")
        self.cluster_focus_var = tk.StringVar(value="")
        self.sequence_map_var = tk.StringVar(value="Near-EF intensity")
        self.sequence_parameter_vars = {
            "fermi_level_ev": tk.StringVar(value=str(defaults.fermi_level_ev)),
            "ef_window_ev": tk.StringVar(value=str(defaults.ef_window_ev)),
        }
        self.change_status_var = tk.StringVar(
            value="Add NetCDF files, label the initial state, then run the initial-state change view."
        )
        self.change_initial_var = tk.StringVar(value="")
        self.change_target_var = tk.StringVar(value="")
        self.change_metric_var = tk.StringVar(value="Near-EF fraction")
        self.change_parameter_vars = {
            "fermi_level_ev": tk.StringVar(value=str(defaults.fermi_level_ev)),
            "ef_window_ev": tk.StringVar(value=str(defaults.ef_window_ev)),
            "wide_window_ev": tk.StringVar(value=str(defaults.wide_window_ev)),
        }
        self.curve_status_var = tk.StringVar(
            value="Add at least two NetCDF files, choose a pair, then run the EDC/MDC comparison."
        )
        self.curve_first_var = tk.StringVar(value="")
        self.curve_second_var = tk.StringVar(value="")
        self.curve_map_var = tk.StringVar(value="Near-EF intensity")
        self.curve_mode_var = tk.StringVar(value="point")
        self.curve_parameter_vars = {
            "fermi_level_ev": tk.StringVar(value=str(defaults.fermi_level_ev)),
            "ef_window_ev": tk.StringVar(value=str(defaults.ef_window_ev)),
        }
        self.feature_status_var = tk.StringVar(
            value="Add at least two NetCDF files, choose a pair, then search for special features."
        )
        self.feature_first_var = tk.StringVar(value="")
        self.feature_second_var = tk.StringVar(value="")
        self.feature_map_var = tk.StringVar(value="Special feature score")
        self.feature_parameter_vars = {
            "fermi_level_ev": tk.StringVar(value=str(defaults.fermi_level_ev)),
            "ef_window_ev": tk.StringVar(value=str(defaults.ef_window_ev)),
            "wide_window_ev": tk.StringVar(value=str(defaults.wide_window_ev)),
            "top_pixels": tk.StringVar(value="12"),
        }
        classifier_defaults = StateClassifierParameters()
        self.classifier_status_var = tk.StringVar(
            value="Choose one NetCDF file, set feature windows, then compute rule-based clustering labels."
        )
        self.classifier_file_var = tk.StringVar(value="")
        self.classifier_map_var = tk.StringVar(value="Classified state")
        self.classifier_parameter_vars = {
            "fermi_level_ev": tk.StringVar(value=str(classifier_defaults.fermi_level_ev)),
            "ef_min_ev": tk.StringVar(value=str(classifier_defaults.ef_min_ev)),
            "ef_max_ev": tk.StringVar(value=str(classifier_defaults.ef_max_ev)),
            "lhb_center_ev": tk.StringVar(value=str(classifier_defaults.lhb_center_ev)),
            "lhb_halfwidth_ev": tk.StringVar(value=str(classifier_defaults.lhb_halfwidth_ev)),
            "leading_edge_min_ev": tk.StringVar(value=str(classifier_defaults.leading_edge_min_ev)),
            "leading_edge_max_ev": tk.StringVar(value=str(classifier_defaults.leading_edge_max_ev)),
            "p3_center_ev": tk.StringVar(value=str(classifier_defaults.p3_center_ev)),
            "p3_halfwidth_ev": tk.StringVar(value=str(classifier_defaults.p3_halfwidth_ev)),
            "smooth_sigma": tk.StringVar(value=str(classifier_defaults.smooth_sigma)),
            "low_quantile": tk.StringVar(value=str(classifier_defaults.low_quantile)),
            "high_quantile": tk.StringVar(value=str(classifier_defaults.high_quantile)),
            "broad_quantile": tk.StringVar(value=str(classifier_defaults.broad_quantile)),
            "orientation_quantile": tk.StringVar(value=str(classifier_defaults.orientation_quantile)),
            "low_signal_quantile": tk.StringVar(value=str(classifier_defaults.low_signal_quantile)),
            "lhb_min_quantile": tk.StringVar(value=str(classifier_defaults.lhb_min_quantile)),
        }
        switching_defaults = SwitchingMapParameters()
        self.switching_status_var = tk.StringVar(
            value="Add chronological NetCDF files, tune EF/LHB windows, then compute switching sites."
        )
        self.switching_parameter_vars = {
            "fermi_level_ev": tk.StringVar(value=str(switching_defaults.fermi_level_ev)),
            "ef_min_ev": tk.StringVar(value=str(switching_defaults.ef_min_ev)),
            "ef_max_ev": tk.StringVar(value=str(switching_defaults.ef_max_ev)),
            "lhb_center_ev": tk.StringVar(value=str(switching_defaults.lhb_center_ev)),
            "lhb_halfwidth_ev": tk.StringVar(value=str(switching_defaults.lhb_halfwidth_ev)),
            "smooth_sigma": tk.StringVar(value=str(switching_defaults.smooth_sigma)),
            "low_switch_quantile": tk.StringVar(value=str(switching_defaults.low_switch_quantile)),
            "high_switch_quantile": tk.StringVar(value=str(switching_defaults.high_switch_quantile)),
            "small_net_quantile": tk.StringVar(value=str(switching_defaults.small_net_quantile)),
            "low_signal_quantile": tk.StringVar(value=str(switching_defaults.low_signal_quantile)),
            "lhb_min_quantile": tk.StringVar(value=str(switching_defaults.lhb_min_quantile)),
        }
        state_prediction_defaults = StatePredictionParameters()
        self.state_prediction_status_var = tk.StringVar(
            value="Add chronological NetCDF files, then compare future switching to initial-state features."
        )
        self.state_prediction_parameter_vars = {
            "fermi_level_ev": tk.StringVar(value=str(state_prediction_defaults.fermi_level_ev)),
            "ef_min_ev": tk.StringVar(value=str(state_prediction_defaults.ef_min_ev)),
            "ef_max_ev": tk.StringVar(value=str(state_prediction_defaults.ef_max_ev)),
            "lhb_center_ev": tk.StringVar(value=str(state_prediction_defaults.lhb_center_ev)),
            "lhb_halfwidth_ev": tk.StringVar(value=str(state_prediction_defaults.lhb_halfwidth_ev)),
            "leading_edge_min_ev": tk.StringVar(value=str(state_prediction_defaults.leading_edge_min_ev)),
            "leading_edge_max_ev": tk.StringVar(value=str(state_prediction_defaults.leading_edge_max_ev)),
            "p3_center_ev": tk.StringVar(value=str(state_prediction_defaults.p3_center_ev)),
            "p3_halfwidth_ev": tk.StringVar(value=str(state_prediction_defaults.p3_halfwidth_ev)),
            "smooth_sigma": tk.StringVar(value=str(state_prediction_defaults.smooth_sigma)),
            "stable_quantile": tk.StringVar(value=str(state_prediction_defaults.stable_quantile)),
            "switch_quantile": tk.StringVar(value=str(state_prediction_defaults.switch_quantile)),
            "net_change_tau": tk.StringVar(value=""),
            "low_signal_quantile": tk.StringVar(value=str(state_prediction_defaults.low_signal_quantile)),
            "lhb_min_quantile": tk.StringVar(value=str(state_prediction_defaults.lhb_min_quantile)),
            "phase_low_quantile": tk.StringVar(value=str(state_prediction_defaults.phase_low_quantile)),
            "phase_high_quantile": tk.StringVar(value=str(state_prediction_defaults.phase_high_quantile)),
            "structural_gradient_quantile": tk.StringVar(value=str(state_prediction_defaults.structural_gradient_quantile)),
        }
        transition_defaults = TransitionOutcomeParameters()
        self.transition_outcome_status_var = tk.StringVar(
            value="Add chronological NetCDF files to map written and erased pixels for each transition."
        )
        self.transition_outcome_pulse_labels_var = tk.StringVar(value="")
        self.transition_outcome_show_wef_var = tk.BooleanVar(value=False)
        self.transition_outcome_show_total_var = tk.BooleanVar(value=False)
        self.transition_outcome_show_global_var = tk.BooleanVar(value=True)
        self.transition_outcome_map_var = tk.StringVar(value="Transition label")
        self.transition_outcome_inspector_file_var = tk.StringVar(value="")
        self.transition_outcome_parameter_vars = {
            "fermi_level_ev": tk.StringVar(value=str(transition_defaults.fermi_level_ev)),
            "ef_min_ev": tk.StringVar(value=str(transition_defaults.ef_min_ev)),
            "ef_max_ev": tk.StringVar(value=str(transition_defaults.ef_max_ev)),
            "lhb_center_ev": tk.StringVar(value=str(transition_defaults.lhb_center_ev)),
            "lhb_halfwidth_ev": tk.StringVar(value=str(transition_defaults.lhb_halfwidth_ev)),
            "smooth_sigma": tk.StringVar(value=str(transition_defaults.smooth_sigma)),
            "user_min_tau": tk.StringVar(value=str(transition_defaults.user_min_tau)),
            "strong_tau_multiplier": tk.StringVar(value=str(transition_defaults.strong_tau_multiplier)),
            "use_relative_delta": tk.BooleanVar(value=transition_defaults.use_relative_delta),
            "low_signal_quantile": tk.StringVar(value=str(transition_defaults.low_signal_quantile)),
            "lhb_min_quantile": tk.StringVar(value=str(transition_defaults.lhb_min_quantile)),
            "color_limit": tk.StringVar(value=""),
        }
        initial_transition_defaults = InitialTransitionFeatureParameters()
        self.initial_transition_status_var = tk.StringVar(
            value="Add a chronological transition sequence, choose a reference file, then compute initial-state transition features."
        )
        self.initial_transition_reference_var = tk.StringVar(value="")
        self.initial_transition_mode_var = tk.StringVar(value="sequential")
        self.initial_transition_normalization_var = tk.StringVar(value=initial_transition_defaults.normalization_mode)
        self.initial_transition_allow_overlap_var = tk.BooleanVar(value=initial_transition_defaults.allow_overlap)
        self.initial_transition_aggregate_map_var = tk.StringVar(value="Metallic count")
        self.initial_transition_selected_transition_var = tk.StringVar(value="")
        self.initial_transition_parameter_vars = {
            "fermi_level_ev": tk.StringVar(value=str(initial_transition_defaults.fermi_level_ev)),
            "ef_min_ev": tk.StringVar(value=str(initial_transition_defaults.ef_min_ev)),
            "ef_max_ev": tk.StringVar(value=str(initial_transition_defaults.ef_max_ev)),
            "feature_min_ev": tk.StringVar(value=str(initial_transition_defaults.feature_min_ev)),
            "feature_max_ev": tk.StringVar(value=str(initial_transition_defaults.feature_max_ev)),
            "asymmetry_split_ev": tk.StringVar(value=str(initial_transition_defaults.asymmetry_split_ev)),
            "metallic_percentile": tk.StringVar(value=str(initial_transition_defaults.metallic_percentile)),
            "erasure_percentile": tk.StringVar(value=str(initial_transition_defaults.erasure_percentile)),
            "stable_percentile": tk.StringVar(value=str(initial_transition_defaults.stable_percentile)),
            "future_metallic_min_count": tk.StringVar(value="1"),
            "future_erased_min_count": tk.StringVar(value="1"),
        }
        mechanism_defaults = SwitchingMechanismParameters()
        self.mechanism_status_var = tk.StringVar(
            value="Use or compute Initial State Transition Features, then run Switching Mechanism Diagnostics."
        )
        self.mechanism_selected_transition_var = tk.StringVar(value="")
        self.mechanism_edc_normalization_var = tk.StringVar(value=mechanism_defaults.edc_normalization)
        self.mechanism_parameter_vars = {
            "future_metallic_min_count": tk.StringVar(value=str(mechanism_defaults.future_metallic_min_count)),
            "future_erased_min_count": tk.StringVar(value=str(mechanism_defaults.future_erased_min_count)),
            "future_metallic_min_frequency": tk.StringVar(value=str(mechanism_defaults.future_metallic_min_frequency)),
            "future_erased_min_frequency": tk.StringVar(value=str(mechanism_defaults.future_erased_min_frequency)),
            "boundary_smooth_sigma": tk.StringVar(value=str(mechanism_defaults.boundary_smooth_sigma)),
            "boundary_percentile": tk.StringVar(value=str(mechanism_defaults.boundary_percentile)),
            "component_min_size": tk.StringVar(value=str(mechanism_defaults.component_min_size)),
            "negative_control_min_ev": tk.StringVar(value=str(mechanism_defaults.negative_control_min_ev)),
            "negative_control_max_ev": tk.StringVar(value=str(mechanism_defaults.negative_control_max_ev)),
            "threshold_sweep_percentiles": tk.StringVar(value=", ".join(str(value) for value in mechanism_defaults.threshold_sweep_percentiles)),
            "permutation_count": tk.StringVar(value=str(mechanism_defaults.permutation_count)),
        }

        self._build_ui()

        if initial_files:
            if self.upload_first:
                self._set_uploaded_files(initial_files)
            else:
                self._set_files(initial_files)
                self._set_sequence_files(initial_files)
                self._set_change_files(initial_files)
                self._set_curve_files(initial_files)
                self._set_feature_files(initial_files)
                self._set_classifier_file(initial_files[0])
                self._set_switching_files(initial_files)
                self._set_state_prediction_files(initial_files)
                self._set_transition_outcome_files(initial_files)
                self._set_initial_transition_files(initial_files)
                self._set_mechanism_files(initial_files)

        self._render_placeholder_text()
        self._render_sequence_placeholder()
        self._render_change_placeholder()
        self._render_curve_placeholder()
        self._render_feature_placeholder()
        self._render_classifier_placeholder()
        self._render_switching_placeholder()
        self._render_state_prediction_placeholder()
        self._render_transition_outcome_placeholder()
        self._render_initial_transition_placeholder()
        self._render_mechanism_placeholder()

    def _build_ui(self) -> None:
        if self.upload_first:
            self.content_frame = ttk.Frame(self.root)
            self.content_frame.pack(fill=tk.BOTH, expand=True)
            self.upload_frame = ttk.Frame(self.content_frame, padding=24)
            self.upload_frame.pack(fill=tk.BOTH, expand=True)
            self._build_upload_gate(self.upload_frame)
            self.runner_frame = ttk.Frame(self.content_frame, padding=24)
            self._build_runner_gate(self.runner_frame)
            notebook_parent = self.content_frame
        else:
            notebook_parent = self.root

        self.top_notebook = ttk.Notebook(notebook_parent)
        if not self.upload_first:
            self.top_notebook.pack(fill=tk.BOTH, expand=True)

        analysis_frame = ttk.Frame(self.top_notebook)
        self.top_notebook.add(analysis_frame, text="Analysis")
        self._build_analysis_panel(analysis_frame)

        sequence_frame = ttk.Frame(self.top_notebook)
        self.top_notebook.add(sequence_frame, text="Sequence Viewer")
        self._build_sequence_panel(sequence_frame)

        change_frame = ttk.Frame(self.top_notebook)
        self.top_notebook.add(change_frame, text="Initial-State Changes")
        self._build_change_panel(change_frame)

        curve_frame = ttk.Frame(self.top_notebook)
        self.top_notebook.add(curve_frame, text="EDC/MDC Compare")
        self._build_curve_panel(curve_frame)

        feature_frame = ttk.Frame(self.top_notebook)
        self.top_notebook.add(feature_frame, text="Feature Search")
        self._build_feature_panel(feature_frame)

        classifier_frame = ttk.Frame(self.top_notebook)
        self.top_notebook.add(classifier_frame, text="Clustering")
        self._build_state_classifier_panel(classifier_frame)

        switching_frame = ttk.Frame(self.top_notebook)
        self.top_notebook.add(switching_frame, text="Switching Map")
        self._build_switching_panel(switching_frame)

        state_prediction_frame = ttk.Frame(self.top_notebook)
        self.top_notebook.add(state_prediction_frame, text="State Prediction")
        self._build_state_prediction_panel(state_prediction_frame)

        transition_outcome_frame = ttk.Frame(self.top_notebook)
        self.top_notebook.add(transition_outcome_frame, text="Transition Outcome Maps")
        self._build_transition_outcome_panel(transition_outcome_frame)

        initial_transition_frame = ttk.Frame(self.top_notebook)
        self.top_notebook.add(initial_transition_frame, text="Initial State Transition Features")
        self._build_initial_transition_panel(initial_transition_frame)

        mechanism_frame = ttk.Frame(self.top_notebook)
        self.top_notebook.add(mechanism_frame, text="Switching Mechanism Diagnostics")
        self._build_switching_mechanism_panel(mechanism_frame)

        if self.upload_first:
            footer = ttk.Frame(self.root, padding=(10, 6))
            footer.pack(fill=tk.X)
            ttk.Button(footer, text="Files", command=self._show_upload_gate, width=10).pack(side=tk.LEFT)
            ttk.Button(footer, text="Run", command=self._show_runner_gate, width=10).pack(side=tk.LEFT, padx=(6, 0))
            ttk.Label(footer, textvariable=self.status_var, anchor="w", padding=(10, 0)).pack(
                side=tk.LEFT,
                fill=tk.X,
                expand=True,
            )
            ttk.Label(footer, textvariable=self.global_progress_status_var, anchor="e").pack(side=tk.LEFT, padx=(8, 6))
            self.global_progress_bar = ttk.Progressbar(
                footer,
                variable=self.global_progress_var,
                maximum=100.0,
                mode="determinate",
                length=260,
            )
            self.global_progress_bar.pack(side=tk.LEFT)
        else:
            status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(12, 6))
            status_bar.pack(fill=tk.X)

    def _build_upload_gate(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        header = ttk.Frame(parent)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Upload data files", font=("TkDefaultFont", 24, "bold")).grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(
            header,
            textvariable=self.upload_status_var,
            justify=tk.LEFT,
            wraplength=1100,
        ).grid(row=1, column=0, sticky="ew", pady=(8, 0))

        files_frame = ttk.LabelFrame(parent, text="NetCDF Files", padding=12)
        files_frame.grid(row=1, column=0, sticky="nsew")
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)

        self.upload_file_listbox = tk.Listbox(files_frame, height=20, exportselection=False)
        self.upload_file_listbox.grid(row=0, column=0, columnspan=4, sticky="nsew")

        ttk.Button(files_frame, text="Add Files", command=self._add_uploaded_files).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(10, 0),
        )
        ttk.Button(files_frame, text="Remove Selected", command=self._remove_selected_uploaded_files).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(10, 0),
        )
        ttk.Button(files_frame, text="Move Up", command=lambda: self._move_selected_uploaded_file(-1)).grid(
            row=1,
            column=2,
            sticky="ew",
            padx=(8, 0),
            pady=(10, 0),
        )
        ttk.Button(files_frame, text="Move Down", command=lambda: self._move_selected_uploaded_file(1)).grid(
            row=1,
            column=3,
            sticky="ew",
            padx=(8, 0),
            pady=(10, 0),
        )
        ttk.Button(files_frame, text="Clear Files", command=self._clear_uploaded_files).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        self.open_uploaded_button = ttk.Button(
            files_frame,
            text="Continue",
            command=self._show_runner_gate,
            state=tk.DISABLED,
        )
        self.open_uploaded_button.grid(row=2, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=(8, 0))

    def _build_runner_gate(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        header = ttk.Frame(parent)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Run analysis panel", font=("TkDefaultFont", 24, "bold")).grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(
            header,
            textvariable=self.runner_status_var,
            justify=tk.LEFT,
            wraplength=1100,
        ).grid(row=1, column=0, sticky="ew", pady=(8, 0))

        runner_frame = ttk.LabelFrame(parent, text="Analysis Panel", padding=12)
        runner_frame.grid(row=1, column=0, sticky="nsew")
        runner_frame.columnconfigure(0, weight=1)
        runner_frame.rowconfigure(4, weight=1)

        ttk.Label(runner_frame, text="Panel").grid(row=0, column=0, sticky="w")
        self.runner_panel_combo = ttk.Combobox(
            runner_frame,
            textvariable=self.runner_panel_var,
            values=self.ANALYSIS_PANEL_OPTIONS,
            state="readonly",
            width=44,
        )
        self.runner_panel_combo.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        self.runner_panel_combo.bind("<<ComboboxSelected>>", self._handle_runner_panel_changed)

        actions = ttk.Frame(runner_frame)
        actions.grid(row=2, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        self.runner_play_button = ttk.Button(actions, text="Play", command=self._run_selected_panel)
        self.runner_play_button.grid(row=0, column=0, sticky="ew")
        self.runner_open_button = ttk.Button(
            actions,
            text="Open View",
            command=self._open_completed_runner_view,
            state=tk.DISABLED,
        )
        self.runner_open_button.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(
            runner_frame,
            textvariable=self.upload_status_var,
            justify=tk.LEFT,
            wraplength=1100,
        ).grid(row=3, column=0, sticky="ew", pady=(14, 8))

        self.runner_file_listbox = tk.Listbox(runner_frame, height=12, exportselection=False)
        self.runner_file_listbox.grid(row=4, column=0, sticky="nsew")

    def _handle_runner_panel_changed(self, _event: tk.Event | None = None) -> None:
        self._sync_runner_buttons()

    def _normalize_file_paths(self, file_paths: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for path in file_paths:
            resolved = str(Path(path).expanduser().resolve())
            if resolved in seen:
                continue
            normalized.append(resolved)
            seen.add(resolved)
        return normalized

    def _add_uploaded_files(self) -> None:
        selected = list(filedialog.askopenfilenames(title="Choose NetCDF files", filetypes=FILE_TYPES))
        if not selected:
            return
        new_paths = self._normalize_file_paths(selected)
        merged = self.uploaded_file_paths + [path for path in new_paths if path not in self.uploaded_file_paths]
        self._set_uploaded_files(merged)

    def _remove_selected_uploaded_files(self) -> None:
        selection = list(self.upload_file_listbox.curselection())
        if not selection:
            return
        updated_files = list(self.uploaded_file_paths)
        for index in reversed(selection):
            del updated_files[index]
        self._set_uploaded_files(updated_files)

    def _move_selected_uploaded_file(self, direction: int) -> None:
        selection = self.upload_file_listbox.curselection()
        if len(selection) != 1:
            return
        index = selection[0]
        new_index = index + direction
        if not 0 <= new_index < len(self.uploaded_file_paths):
            return
        updated_files = list(self.uploaded_file_paths)
        updated_files[index], updated_files[new_index] = updated_files[new_index], updated_files[index]
        self._set_uploaded_files(updated_files)
        self.upload_file_listbox.selection_set(new_index)

    def _clear_uploaded_files(self) -> None:
        self._set_uploaded_files([])

    def _set_uploaded_files(self, file_paths: list[str]) -> None:
        self.uploaded_file_paths = self._normalize_file_paths(file_paths)
        self.runner_completed_panel = None
        self.runner_running_panel = None
        self._sync_uploaded_file_listbox()
        self._sync_runner_file_listbox()
        self._apply_uploaded_files_to_views()
        self._update_upload_status()
        self._sync_runner_buttons()

    def _sync_uploaded_file_listbox(self) -> None:
        if not hasattr(self, "upload_file_listbox"):
            return
        self.upload_file_listbox.delete(0, tk.END)
        for index, path in enumerate(self.uploaded_file_paths):
            self.upload_file_listbox.insert(tk.END, f"{index + 1}. {Path(path).name}")

    def _sync_runner_file_listbox(self) -> None:
        if not hasattr(self, "runner_file_listbox"):
            return
        self.runner_file_listbox.delete(0, tk.END)
        for index, path in enumerate(self.uploaded_file_paths):
            self.runner_file_listbox.insert(tk.END, f"{index + 1}. {Path(path).name}")

    def _update_upload_status(self) -> None:
        count = len(self.uploaded_file_paths)
        if count == 0:
            message = "Choose the NetCDF files to analyze. Sequence order is preserved for chronological views."
        elif count <= 4:
            message = f"{count} file(s) ready. Every view has been preloaded from this upload set."
        else:
            message = (
                f"{count} file(s) ready. Sequence views use all files; the main Analysis tab uses the first four, "
                "and single-file/pair views use the first file or first two files."
            )
        self.upload_status_var.set(message)
        self.status_var.set(message)
        if hasattr(self, "open_uploaded_button"):
            self.open_uploaded_button.configure(state=tk.NORMAL if count else tk.DISABLED)
        self._sync_runner_buttons()

    def _apply_uploaded_files_to_views(self) -> None:
        paths = list(self.uploaded_file_paths)
        analysis_paths = paths[:4]
        self._set_files(analysis_paths)
        self._set_sequence_files(paths)
        self._set_change_files(paths)
        self._set_curve_files(paths)
        self._set_feature_files(paths)
        self._set_classifier_file(paths[0] if paths else None)
        self._set_switching_files(paths)
        self._set_state_prediction_files(paths)
        self._set_transition_outcome_files(paths)
        self._set_initial_transition_files(paths)
        self._set_mechanism_files(paths)

    def _show_runner_gate(self) -> None:
        if not self.uploaded_file_paths:
            messagebox.showinfo("No files uploaded", "Choose at least one NetCDF file before running an analysis panel.")
            return
        self._apply_uploaded_files_to_views()
        if self.upload_frame.winfo_manager():
            self.upload_frame.pack_forget()
        if self.top_notebook.winfo_manager():
            self.top_notebook.pack_forget()
        if not self.runner_frame.winfo_manager():
            self.runner_frame.pack(fill=tk.BOTH, expand=True)
        self._sync_runner_file_listbox()
        self._sync_runner_buttons()
        self.global_progress_status_var.set("Ready.")

    def _open_uploaded_analysis(self) -> None:
        if self.upload_frame.winfo_manager():
            self.upload_frame.pack_forget()
        if self.runner_frame.winfo_manager():
            self.runner_frame.pack_forget()
        if not self.top_notebook.winfo_manager():
            self.top_notebook.pack(fill=tk.BOTH, expand=True)
        self.top_notebook.select(0)
        self.global_progress_status_var.set("Ready.")

    def _show_upload_gate(self) -> None:
        if self.top_notebook.winfo_manager():
            self.top_notebook.pack_forget()
        if self.runner_frame.winfo_manager():
            self.runner_frame.pack_forget()
        if not self.upload_frame.winfo_manager():
            self.upload_frame.pack(fill=tk.BOTH, expand=True)
        self._sync_uploaded_file_listbox()
        self._update_upload_status()

    def _sync_runner_buttons(self) -> None:
        if not hasattr(self, "runner_play_button"):
            return
        selected_panel = self.runner_panel_var.get()
        is_running = self.runner_running_panel is not None
        has_files = bool(self.uploaded_file_paths)
        can_open = self.runner_completed_panel == selected_panel and not is_running
        self.runner_play_button.configure(state=tk.NORMAL if has_files and not is_running else tk.DISABLED)
        self.runner_open_button.configure(state=tk.NORMAL if can_open else tk.DISABLED)

        if is_running:
            self.runner_status_var.set(f"Running {self.runner_running_panel}...")
            return
        if can_open:
            self.runner_status_var.set(f"{selected_panel} finished. Open its view when you are ready.")
            return

        requirement = self._runner_requirement_message(selected_panel)
        if requirement:
            self.runner_status_var.set(requirement)
        else:
            self.runner_status_var.set(f"Ready to run {selected_panel} on the uploaded data files.")

    def _runner_requirement_message(self, panel_name: str) -> str | None:
        required = self.ANALYSIS_PANEL_MIN_FILES.get(panel_name, 1)
        count = len(self.uploaded_file_paths)
        if count < required:
            noun = "file" if required == 1 else "files"
            return f"{panel_name} needs at least {required} uploaded {noun}."
        if panel_name == "Analysis" and count > 4:
            return "Analysis will run on the first four uploaded files."
        return None

    def _run_selected_panel(self) -> None:
        panel_name = self.runner_panel_var.get()
        requirement = self._runner_requirement_message(panel_name)
        if requirement and len(self.uploaded_file_paths) < self.ANALYSIS_PANEL_MIN_FILES.get(panel_name, 1):
            messagebox.showerror("Not enough files", requirement)
            self.runner_status_var.set(requirement)
            return

        self._apply_uploaded_files_to_views()
        self.runner_completed_panel = None
        self.runner_running_panel = panel_name
        self._sync_runner_buttons()

        run_map = {
            "Analysis": self._run_analysis,
            "Sequence Viewer": self._run_sequence_viewer,
            "Initial-State Changes": self._run_change_analysis,
            "EDC/MDC Compare": self._run_curve_comparison,
            "Feature Search": self._run_feature_search,
            "Clustering": self._run_state_classifier,
            "Switching Map": self._run_switching_map,
            "State Prediction": self._run_state_prediction,
            "Transition Outcome Maps": self._run_transition_outcome_maps,
            "Initial State Transition Features": self._run_initial_transition_analysis,
            "Switching Mechanism Diagnostics": lambda: self._run_mechanism_diagnostics(source="files"),
        }
        run_action = run_map.get(panel_name)
        if run_action is None:
            self._mark_runner_failed(panel_name)
            return

        try:
            run_action()
        except Exception as exc:
            self._mark_runner_failed(panel_name)
            messagebox.showerror("Analysis failed", str(exc))
            return

        if panel_name == "Switching Mechanism Diagnostics":
            if self.mechanism_worker_thread is not None and self.mechanism_worker_thread.is_alive():
                self.runner_status_var.set("Switching Mechanism Diagnostics is still running...")
                self._sync_runner_buttons()
                return

        if self._runner_panel_has_result(panel_name):
            self._mark_runner_complete(panel_name)
        else:
            self._mark_runner_failed(panel_name)

    def _runner_panel_has_result(self, panel_name: str) -> bool:
        if panel_name == "Analysis":
            return self.result is not None
        if panel_name == "Sequence Viewer":
            return bool(self.sequence_loaded_states)
        if panel_name == "Initial-State Changes":
            return self.change_valid_mask is not None and bool(self.change_loaded_states)
        if panel_name == "EDC/MDC Compare":
            return len(self.curve_loaded_states) == 2
        if panel_name == "Feature Search":
            return self.feature_score_map is not None
        if panel_name == "Clustering":
            return self.classifier_result is not None
        if panel_name == "Switching Map":
            return self.switching_result is not None
        if panel_name == "State Prediction":
            return self.state_prediction_result is not None
        if panel_name == "Transition Outcome Maps":
            return self.transition_outcome_result is not None
        if panel_name == "Initial State Transition Features":
            return self.initial_transition_result is not None
        if panel_name == "Switching Mechanism Diagnostics":
            return self.mechanism_result is not None
        return False

    def _mark_runner_complete(self, panel_name: str) -> None:
        self.runner_completed_panel = panel_name
        self.runner_running_panel = None
        self.runner_status_var.set(f"{panel_name} finished. Open its view when you are ready.")
        self._finish_global_progress(f"{panel_name} ready to open.")
        self._sync_runner_buttons()

    def _mark_runner_failed(self, panel_name: str) -> None:
        if self.runner_running_panel == panel_name:
            self.runner_running_panel = None
        self.runner_completed_panel = None
        self.runner_status_var.set(f"{panel_name} did not finish. Check the panel settings or uploaded files.")
        self._finish_global_progress(f"{panel_name} did not finish.", success=False)
        self._sync_runner_buttons()

    def _open_completed_runner_view(self) -> None:
        panel_name = self.runner_completed_panel
        if panel_name is None:
            return
        if self.runner_frame.winfo_manager():
            self.runner_frame.pack_forget()
        if self.upload_frame.winfo_manager():
            self.upload_frame.pack_forget()
        if not self.top_notebook.winfo_manager():
            self.top_notebook.pack(fill=tk.BOTH, expand=True)
        self.top_notebook.select(self.ANALYSIS_PANEL_OPTIONS.index(panel_name))

    def _start_global_progress(self, message: str) -> None:
        if not hasattr(self, "global_progress_bar"):
            return
        self.global_progress_bar.stop()
        self.global_progress_var.set(0.0)
        self.global_progress_bar.configure(mode="indeterminate")
        self.global_progress_status_var.set(message)
        self.global_progress_bar.start(12)
        self.root.update_idletasks()

    def _finish_global_progress(self, message: str, *, success: bool = True) -> None:
        if not hasattr(self, "global_progress_bar"):
            return
        self.global_progress_bar.stop()
        self.global_progress_bar.configure(mode="determinate")
        self.global_progress_var.set(100.0 if success else 0.0)
        self.global_progress_status_var.set(message)
        self.root.update_idletasks()

    def _build_analysis_panel(self, parent: ttk.Frame) -> None:
        main_pane = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_pane, padding=12)
        main_pane.add(controls_frame, weight=0)

        right_frame = ttk.Frame(main_pane, padding=(0, 12, 12, 12))
        main_pane.add(right_frame, weight=1)

        self._build_controls_panel(controls_frame)
        self._build_visual_panel(right_frame)

    def _build_sequence_panel(self, parent: ttk.Frame) -> None:
        main_pane = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_pane, padding=12)
        main_pane.add(controls_frame, weight=0)

        right_frame = ttk.Frame(main_pane, padding=(0, 12, 12, 12))
        main_pane.add(right_frame, weight=1)

        self._build_sequence_controls_panel(controls_frame)
        self._build_sequence_visual_panel(right_frame)

    def _build_change_panel(self, parent: ttk.Frame) -> None:
        main_pane = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_pane, padding=12)
        main_pane.add(controls_frame, weight=0)

        right_frame = ttk.Frame(main_pane, padding=(0, 12, 12, 12))
        main_pane.add(right_frame, weight=1)

        self._build_change_controls_panel(controls_frame)
        self._build_change_visual_panel(right_frame)

    def _build_curve_panel(self, parent: ttk.Frame) -> None:
        main_pane = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_pane, padding=12)
        main_pane.add(controls_frame, weight=0)

        right_frame = ttk.Frame(main_pane, padding=(0, 12, 12, 12))
        main_pane.add(right_frame, weight=1)

        self._build_curve_controls_panel(controls_frame)
        self._build_curve_visual_panel(right_frame)

    def _build_feature_panel(self, parent: ttk.Frame) -> None:
        main_pane = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_pane, padding=12)
        main_pane.add(controls_frame, weight=0)

        right_frame = ttk.Frame(main_pane, padding=(0, 12, 12, 12))
        main_pane.add(right_frame, weight=1)

        self._build_feature_controls_panel(controls_frame)
        self._build_feature_visual_panel(right_frame)

    def _build_state_classifier_panel(self, parent: ttk.Frame) -> None:
        main_pane = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_pane, padding=12)
        main_pane.add(controls_frame, weight=0)

        right_frame = ttk.Frame(main_pane, padding=(0, 12, 12, 12))
        main_pane.add(right_frame, weight=1)

        self._build_state_classifier_controls_panel(controls_frame)
        self._build_state_classifier_visual_panel(right_frame)

    def _build_switching_panel(self, parent: ttk.Frame) -> None:
        main_pane = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_pane, padding=12)
        main_pane.add(controls_frame, weight=0)

        right_frame = ttk.Frame(main_pane, padding=(0, 12, 12, 12))
        main_pane.add(right_frame, weight=1)

        self._build_switching_controls_panel(controls_frame)
        self._build_switching_visual_panel(right_frame)

    def _build_state_prediction_panel(self, parent: ttk.Frame) -> None:
        main_pane = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_pane, padding=12)
        main_pane.add(controls_frame, weight=0)

        right_frame = ttk.Frame(main_pane, padding=(0, 12, 12, 12))
        main_pane.add(right_frame, weight=1)

        self._build_state_prediction_controls_panel(controls_frame)
        self._build_state_prediction_visual_panel(right_frame)

    def _build_transition_outcome_panel(self, parent: ttk.Frame) -> None:
        main_pane = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_pane, padding=12)
        main_pane.add(controls_frame, weight=0)

        right_frame = ttk.Frame(main_pane, padding=(0, 12, 12, 12))
        main_pane.add(right_frame, weight=1)

        self._build_transition_outcome_controls_panel(controls_frame)
        self._build_transition_outcome_visual_panel(right_frame)

    def _build_initial_transition_panel(self, parent: ttk.Frame) -> None:
        main_pane = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_pane, padding=12)
        main_pane.add(controls_frame, weight=0)

        right_frame = ttk.Frame(main_pane, padding=(0, 12, 12, 12))
        main_pane.add(right_frame, weight=1)

        self._build_initial_transition_controls_panel(controls_frame)
        self._build_initial_transition_visual_panel(right_frame)

    def _build_switching_mechanism_panel(self, parent: ttk.Frame) -> None:
        main_pane = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_pane, padding=12)
        main_pane.add(controls_frame, weight=0)

        right_frame = ttk.Frame(main_pane, padding=(0, 12, 12, 12))
        main_pane.add(right_frame, weight=1)

        self._build_switching_mechanism_controls_panel(controls_frame)
        self._build_switching_mechanism_visual_panel(right_frame)

    def _build_sequence_controls_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        files_frame = ttk.LabelFrame(parent, text="Sequence Files", padding=10)
        files_frame.grid(row=0, column=0, sticky="nsew")
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)

        self.sequence_file_listbox = tk.Listbox(files_frame, height=15, exportselection=False)
        self.sequence_file_listbox.grid(row=0, column=0, columnspan=2, sticky="nsew")

        ttk.Button(files_frame, text="Add Files", command=self._add_sequence_files).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Use Analysis Files", command=self._copy_analysis_files_to_sequence_panel).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Remove Selected", command=self._remove_selected_sequence_files).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Clear Files", command=self._clear_sequence_files).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Move Up", command=lambda: self._move_selected_sequence_file(-1)).grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Move Down", command=lambda: self._move_selected_sequence_file(1)).grid(
            row=3,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )

        display_frame = ttk.LabelFrame(parent, text="Display", padding=10)
        display_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        display_frame.columnconfigure(0, weight=1)

        ttk.Label(display_frame, text="Map").grid(row=0, column=0, sticky="w")
        self.sequence_map_combo = ttk.Combobox(
            display_frame,
            textvariable=self.sequence_map_var,
            values=list(self.SEQUENCE_MAP_OPTIONS.keys()),
            state="readonly",
            width=24,
        )
        self.sequence_map_combo.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.sequence_map_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_sequence_views())

        self._add_sequence_parameter_row(display_frame, 2, "Fermi level (eV)", "fermi_level_ev")
        self._add_sequence_parameter_row(display_frame, 3, "Near-EF half-window (eV)", "ef_window_ev")

        actions_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        actions_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        actions_frame.columnconfigure(0, weight=1)

        ttk.Button(actions_frame, text="Load Sequence", command=self._run_sequence_viewer).grid(row=0, column=0, sticky="ew")
        ttk.Button(actions_frame, text="Save Overview Plot...", command=self._save_sequence_overview_plot).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(actions_frame, text="Save Comparison Plot...", command=self._save_sequence_comparison_plot).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )

        help_text = (
            "Use the file order as the measurement sequence.\n"
            "The viewer aligns clipped x/y grids when needed.\n"
            "Select up to three files in the bottom panel and click a map pixel for EDC/MDC waterfalls."
        )
        ttk.Label(parent, text=help_text, justify=tk.LEFT, wraplength=320).grid(row=3, column=0, sticky="ew", pady=(12, 0))

    def _build_sequence_visual_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        ttk.Label(
            parent,
            textvariable=self.sequence_status_var,
            justify=tk.LEFT,
            wraplength=1120,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        sequence_split = ttk.Panedwindow(parent, orient=tk.VERTICAL)
        sequence_split.grid(row=1, column=0, sticky="nsew")

        overview_frame = ttk.Frame(sequence_split)
        sequence_split.add(overview_frame, weight=2)
        overview_frame.columnconfigure(0, weight=1)
        overview_frame.rowconfigure(1, weight=1)

        overview_toolbar_frame = ttk.Frame(overview_frame)
        overview_toolbar_frame.grid(row=0, column=0, sticky="ew")

        self.sequence_scroll_canvas = tk.Canvas(overview_frame, highlightthickness=0)
        self.sequence_scrollbar = ttk.Scrollbar(
            overview_frame,
            orient=tk.VERTICAL,
            command=self.sequence_scroll_canvas.yview,
        )
        self.sequence_scroll_canvas.configure(yscrollcommand=self.sequence_scrollbar.set)
        self.sequence_scroll_canvas.grid(row=1, column=0, sticky="nsew")
        self.sequence_scrollbar.grid(row=1, column=1, sticky="ns")

        self.sequence_canvas_frame = ttk.Frame(self.sequence_scroll_canvas)
        self.sequence_canvas_frame.columnconfigure(0, weight=1)
        self.sequence_canvas_frame.rowconfigure(0, weight=1)
        self.sequence_scroll_window = self.sequence_scroll_canvas.create_window(
            (0, 0),
            window=self.sequence_canvas_frame,
            anchor="nw",
        )
        self.sequence_figure = Figure(figsize=(11.2, 6.4), dpi=100, constrained_layout=True)
        self.sequence_canvas = FigureCanvasTkAgg(self.sequence_figure, master=self.sequence_canvas_frame)
        self.sequence_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.sequence_canvas_frame.bind("<Configure>", self._update_sequence_scroll_region)
        self.sequence_scroll_canvas.bind("<Configure>", self._resize_sequence_scroll_window)
        self.sequence_scroll_canvas.bind("<Enter>", self._bind_sequence_mousewheel)
        self.sequence_scroll_canvas.bind("<Leave>", self._unbind_sequence_mousewheel)
        self.sequence_canvas.get_tk_widget().bind("<Enter>", self._bind_sequence_mousewheel)
        self.sequence_canvas.get_tk_widget().bind("<Leave>", self._unbind_sequence_mousewheel)
        self.sequence_canvas.mpl_connect("button_press_event", self._on_sequence_plot_click)

        try:
            self.sequence_toolbar = NavigationToolbar2Tk(self.sequence_canvas, overview_toolbar_frame, pack_toolbar=False)
        except Exception:
            self.sequence_toolbar = None
            ttk.Label(overview_toolbar_frame, text="Matplotlib toolbar unavailable in this environment.").pack(side=tk.LEFT)
        else:
            self.sequence_toolbar.update()
            self.sequence_toolbar.pack(side=tk.LEFT, fill=tk.X)

        comparison_frame = ttk.LabelFrame(sequence_split, text="Selected Pixel EDC/MDC Investigation", padding=8)
        sequence_split.add(comparison_frame, weight=3)
        comparison_frame.columnconfigure(1, weight=1)
        comparison_frame.rowconfigure(0, weight=1)

        selector_frame = ttk.Frame(comparison_frame)
        selector_frame.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        selector_frame.columnconfigure(0, weight=1)
        selector_frame.rowconfigure(1, weight=1)

        ttk.Label(selector_frame, text="Choose two or three files").grid(row=0, column=0, sticky="w")
        self.sequence_selection_listbox = tk.Listbox(
            selector_frame,
            height=10,
            width=34,
            selectmode=tk.EXTENDED,
            exportselection=False,
        )
        self.sequence_selection_listbox.grid(row=1, column=0, sticky="nsew", pady=(4, 8))
        self.sequence_selection_listbox.bind("<<ListboxSelect>>", self._handle_sequence_selection_changed)

        ttk.Button(selector_frame, text="Use First Three", command=self._select_first_sequence_plots).grid(
            row=2,
            column=0,
            sticky="ew",
        )

        self.sequence_summary_text = tk.Text(selector_frame, width=34, height=6, wrap="word")
        self.sequence_summary_text.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.sequence_summary_text.configure(state="disabled")

        compare_canvas_frame = ttk.Frame(comparison_frame)
        compare_canvas_frame.grid(row=0, column=1, sticky="nsew")
        compare_canvas_frame.columnconfigure(0, weight=1)
        compare_canvas_frame.rowconfigure(0, weight=1)

        self.sequence_compare_scroll_canvas = tk.Canvas(compare_canvas_frame, highlightthickness=0)
        self.sequence_compare_h_scrollbar = ttk.Scrollbar(
            compare_canvas_frame,
            orient=tk.HORIZONTAL,
            command=self.sequence_compare_scroll_canvas.xview,
        )
        self.sequence_compare_scroll_canvas.configure(xscrollcommand=self.sequence_compare_h_scrollbar.set)
        self.sequence_compare_scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self.sequence_compare_h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.sequence_compare_canvas_frame = ttk.Frame(self.sequence_compare_scroll_canvas)
        self.sequence_compare_canvas_frame.columnconfigure(0, weight=1)
        self.sequence_compare_canvas_frame.rowconfigure(0, weight=1)
        self.sequence_compare_scroll_window = self.sequence_compare_scroll_canvas.create_window(
            (0, 0),
            window=self.sequence_compare_canvas_frame,
            anchor="nw",
        )
        self.sequence_compare_figure = Figure(figsize=(11.2, 8.6), dpi=100, constrained_layout=False)
        self.sequence_compare_canvas = FigureCanvasTkAgg(
            self.sequence_compare_figure,
            master=self.sequence_compare_canvas_frame,
        )
        self.sequence_compare_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.sequence_compare_canvas_frame.bind("<Configure>", self._update_sequence_compare_scroll_region)
        self.sequence_compare_scroll_canvas.bind("<Configure>", self._resize_sequence_compare_scroll_window)

    def _add_sequence_parameter_row(self, parent: ttk.LabelFrame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.sequence_parameter_vars[key], width=16).grid(
            row=row,
            column=1,
            sticky="e",
            padx=(10, 0),
            pady=2,
        )

    def _update_sequence_scroll_region(self, _event: tk.Event | None = None) -> None:
        if hasattr(self, "sequence_scroll_canvas"):
            self.sequence_scroll_canvas.configure(scrollregion=self.sequence_scroll_canvas.bbox("all"))

    def _resize_sequence_scroll_window(self, event: tk.Event) -> None:
        if hasattr(self, "sequence_scroll_window"):
            width = max(1, int(event.width))
            self.sequence_scroll_canvas.itemconfigure(self.sequence_scroll_window, width=width)

    def _plot_canvas_width_px(self, canvas: tk.Canvas, fallback: int) -> int:
        try:
            width = int(canvas.winfo_width())
        except tk.TclError:
            width = fallback
        if width < 200:
            width = fallback
        return max(360, width - 8)

    def _short_file_label(self, file_path: str, max_chars: int = 34) -> str:
        name = Path(file_path).name
        if len(name) <= max_chars:
            return name
        keep = max(8, max_chars - 3)
        front = max(8, keep // 2)
        back = max(8, keep - front)
        return f"{name[:front]}...{name[-back:]}"

    def _sequence_state_label(self, index: int, max_chars: int = 34) -> str:
        state = self.sequence_loaded_states[index]
        return f"{index + 1}. {self._short_file_label(state.file_path, max_chars=max_chars)}"

    def _update_sequence_compare_scroll_region(self, _event: tk.Event | None = None) -> None:
        if hasattr(self, "sequence_compare_scroll_canvas"):
            self.sequence_compare_scroll_canvas.configure(
                scrollregion=self.sequence_compare_scroll_canvas.bbox("all")
            )

    def _resize_sequence_compare_scroll_window(self, event: tk.Event) -> None:
        if not hasattr(self, "sequence_compare_scroll_window"):
            return
        width = max(1, int(event.width))
        self.sequence_compare_scroll_canvas.itemconfigure(
            self.sequence_compare_scroll_window,
            width=width,
        )

    def _bind_sequence_mousewheel(self, _event: tk.Event | None = None) -> None:
        self.sequence_scroll_canvas.bind_all("<MouseWheel>", self._on_sequence_mousewheel)

    def _unbind_sequence_mousewheel(self, _event: tk.Event | None = None) -> None:
        self.sequence_scroll_canvas.unbind_all("<MouseWheel>")

    def _on_sequence_mousewheel(self, event: tk.Event) -> None:
        delta = int(getattr(event, "delta", 0))
        if delta == 0:
            return
        self.sequence_scroll_canvas.yview_scroll(-1 if delta > 0 else 1, "units")

    def _build_feature_controls_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        files_frame = ttk.LabelFrame(parent, text="Feature Search Files", padding=10)
        files_frame.grid(row=0, column=0, sticky="nsew")
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)

        self.feature_file_listbox = tk.Listbox(files_frame, height=11, exportselection=False)
        self.feature_file_listbox.grid(row=0, column=0, columnspan=2, sticky="nsew")

        ttk.Button(files_frame, text="Add Files", command=self._add_feature_files).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Use Analysis Files", command=self._copy_analysis_files_to_feature_panel).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Remove Selected", command=self._remove_selected_feature_files).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Clear Files", command=self._clear_feature_files).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )

        compare_frame = ttk.LabelFrame(parent, text="Dataset Pair", padding=10)
        compare_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        compare_frame.columnconfigure(0, weight=1)

        ttk.Label(compare_frame, text="First file").grid(row=0, column=0, sticky="w")
        self.feature_first_combo = ttk.Combobox(
            compare_frame,
            textvariable=self.feature_first_var,
            state="readonly",
            width=34,
        )
        self.feature_first_combo.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.feature_first_combo.bind("<<ComboboxSelected>>", self._handle_feature_first_selected)

        ttk.Label(compare_frame, text="Second file").grid(row=2, column=0, sticky="w")
        self.feature_second_combo = ttk.Combobox(
            compare_frame,
            textvariable=self.feature_second_var,
            state="readonly",
            width=34,
        )
        self.feature_second_combo.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        self.feature_second_combo.bind("<<ComboboxSelected>>", self._handle_feature_second_selected)

        ttk.Label(compare_frame, text="Feature map").grid(row=4, column=0, sticky="w")
        self.feature_map_combo = ttk.Combobox(
            compare_frame,
            textvariable=self.feature_map_var,
            values=list(self.FEATURE_MAP_OPTIONS.keys()),
            state="readonly",
            width=34,
        )
        self.feature_map_combo.grid(row=5, column=0, sticky="ew")
        self.feature_map_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_feature_views())

        parameters_frame = ttk.LabelFrame(parent, text="Search Parameters", padding=10)
        parameters_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self._add_feature_parameter_row(parameters_frame, 0, "Fermi level (eV)", "fermi_level_ev")
        self._add_feature_parameter_row(parameters_frame, 1, "Near-EF window (eV)", "ef_window_ev")
        self._add_feature_parameter_row(parameters_frame, 2, "Wide window (eV)", "wide_window_ev")
        self._add_feature_parameter_row(parameters_frame, 3, "Top hotspots", "top_pixels")

        actions_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        actions_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        actions_frame.columnconfigure(0, weight=1)
        ttk.Button(actions_frame, text="Search Special Features", command=self._run_feature_search).grid(
            row=0,
            column=0,
            sticky="ew",
        )
        ttk.Button(actions_frame, text="AI data analysis", command=self._run_ai_data_analysis_placeholder).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(actions_frame, text="Save Feature Plot...", command=self._save_feature_plot).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )

        help_text = (
            "The score combines spectral-shape, near-EF, centroid, total-intensity, and entropy changes.\n"
            "The AI button is intentionally a placeholder for your Ollama hook."
        )
        ttk.Label(parent, text=help_text, justify=tk.LEFT, wraplength=320).grid(
            row=4,
            column=0,
            sticky="ew",
            pady=(12, 0),
        )

    def _build_feature_visual_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        parent.rowconfigure(2, weight=0)

        ttk.Label(
            parent,
            textvariable=self.feature_status_var,
            anchor="w",
            justify=tk.LEFT,
            wraplength=1120,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        feature_frame = ttk.Frame(parent)
        feature_frame.grid(row=1, column=0, sticky="nsew")
        feature_frame.columnconfigure(0, weight=1)
        feature_frame.rowconfigure(1, weight=1)

        toolbar_frame = ttk.Frame(feature_frame)
        toolbar_frame.grid(row=0, column=0, sticky="ew")

        self.feature_figure = Figure(figsize=(11.2, 8.2), dpi=100, constrained_layout=True)
        self.feature_canvas = FigureCanvasTkAgg(self.feature_figure, master=feature_frame)
        self.feature_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.feature_canvas.mpl_connect("button_press_event", self._on_feature_plot_click)

        try:
            self.feature_toolbar = NavigationToolbar2Tk(self.feature_canvas, toolbar_frame, pack_toolbar=False)
        except Exception:
            self.feature_toolbar = None
            ttk.Label(toolbar_frame, text="Matplotlib toolbar unavailable in this environment.").pack(side=tk.LEFT)
        else:
            self.feature_toolbar.update()
            self.feature_toolbar.pack(side=tk.LEFT, fill=tk.X)

        summary_frame = ttk.Frame(parent, padding=(0, 8, 0, 0))
        summary_frame.grid(row=2, column=0, sticky="ew")
        summary_frame.columnconfigure(0, weight=1)

        self.feature_summary_text = tk.Text(summary_frame, height=9, wrap="word")
        self.feature_summary_text.grid(row=0, column=0, sticky="ew")
        self.feature_summary_text.configure(state="disabled")

    def _build_state_classifier_controls_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        file_frame = ttk.LabelFrame(parent, text="Input File", padding=10)
        file_frame.grid(row=0, column=0, sticky="ew")
        file_frame.columnconfigure(0, weight=1)
        ttk.Entry(file_frame, textvariable=self.classifier_file_var, state="readonly", width=34).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
        )
        ttk.Button(file_frame, text="Choose File", command=self._choose_classifier_file).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(file_frame, text="Use Analysis File", command=self._use_analysis_file_for_classifier).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )

        display_frame = ttk.LabelFrame(parent, text="Display", padding=10)
        display_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        display_frame.columnconfigure(0, weight=1)
        ttk.Label(display_frame, text="Map").grid(row=0, column=0, sticky="w")
        self.classifier_map_combo = ttk.Combobox(
            display_frame,
            textvariable=self.classifier_map_var,
            values=list(self.STATE_CLASSIFIER_MAP_OPTIONS.keys()),
            state="readonly",
            width=34,
        )
        self.classifier_map_combo.grid(row=1, column=0, sticky="ew")
        self.classifier_map_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_classifier_views())

        feature_frame = ttk.LabelFrame(parent, text="Feature Windows", padding=10)
        feature_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        feature_frame.columnconfigure(1, weight=1)
        self._add_classifier_parameter_row(feature_frame, 0, "Fermi level (eV)", "fermi_level_ev")
        self._add_classifier_parameter_row(feature_frame, 1, "EF min rel. (eV)", "ef_min_ev")
        self._add_classifier_parameter_row(feature_frame, 2, "EF max rel. (eV)", "ef_max_ev")
        self._add_classifier_parameter_row(feature_frame, 3, "LHB center (eV)", "lhb_center_ev")
        self._add_classifier_parameter_row(feature_frame, 4, "LHB halfwidth (eV)", "lhb_halfwidth_ev")
        self._add_classifier_parameter_row(feature_frame, 5, "LE min rel. (eV)", "leading_edge_min_ev")
        self._add_classifier_parameter_row(feature_frame, 6, "LE max rel. (eV)", "leading_edge_max_ev")
        self._add_classifier_parameter_row(feature_frame, 7, "p3 center (eV)", "p3_center_ev")
        self._add_classifier_parameter_row(feature_frame, 8, "p3 halfwidth (eV)", "p3_halfwidth_ev")
        self._add_classifier_parameter_row(feature_frame, 9, "EDC smooth sigma", "smooth_sigma")

        thresholds_frame = ttk.LabelFrame(parent, text="Rule Threshold Quantiles", padding=10)
        thresholds_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        thresholds_frame.columnconfigure(1, weight=1)
        self._add_classifier_parameter_row(thresholds_frame, 0, "Low state quantile", "low_quantile")
        self._add_classifier_parameter_row(thresholds_frame, 1, "High state quantile", "high_quantile")
        self._add_classifier_parameter_row(thresholds_frame, 2, "Broad linewidth q", "broad_quantile")
        self._add_classifier_parameter_row(thresholds_frame, 3, "Orientation shift q", "orientation_quantile")
        self._add_classifier_parameter_row(thresholds_frame, 4, "Low-signal q", "low_signal_quantile")
        self._add_classifier_parameter_row(thresholds_frame, 5, "Min LHB q", "lhb_min_quantile")

        actions_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        actions_frame.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        actions_frame.columnconfigure(0, weight=1)
        ttk.Button(actions_frame, text="Compute and Classify", command=self._run_state_classifier).grid(
            row=0,
            column=0,
            sticky="ew",
        )
        ttk.Button(actions_frame, text="Update Thresholds", command=self._reclassify_state_classifier).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(actions_frame, text="Save Results...", command=self._save_classifier_results).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(actions_frame, text="Save Plot...", command=self._save_classifier_plot).grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )

    def _build_state_classifier_visual_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        parent.rowconfigure(2, weight=0)

        ttk.Label(
            parent,
            textvariable=self.classifier_status_var,
            anchor="w",
            justify=tk.LEFT,
            wraplength=1120,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        figure_frame = ttk.Frame(parent)
        figure_frame.grid(row=1, column=0, sticky="nsew")
        figure_frame.columnconfigure(0, weight=1)
        figure_frame.rowconfigure(1, weight=1)

        toolbar_frame = ttk.Frame(figure_frame)
        toolbar_frame.grid(row=0, column=0, sticky="ew")

        self.classifier_figure = Figure(figsize=(11.2, 8.2), dpi=100, constrained_layout=True)
        self.classifier_canvas = FigureCanvasTkAgg(self.classifier_figure, master=figure_frame)
        self.classifier_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.classifier_canvas.mpl_connect("button_press_event", self._on_classifier_plot_click)

        try:
            self.classifier_toolbar = NavigationToolbar2Tk(self.classifier_canvas, toolbar_frame, pack_toolbar=False)
        except Exception:
            self.classifier_toolbar = None
            ttk.Label(toolbar_frame, text="Matplotlib toolbar unavailable in this environment.").pack(side=tk.LEFT)
        else:
            self.classifier_toolbar.update()
            self.classifier_toolbar.pack(side=tk.LEFT, fill=tk.X)

        summary_frame = ttk.Frame(parent, padding=(0, 8, 0, 0))
        summary_frame.grid(row=2, column=0, sticky="ew")
        summary_frame.columnconfigure(0, weight=1)
        self.classifier_summary_text = tk.Text(summary_frame, height=10, wrap="word")
        self.classifier_summary_text.grid(row=0, column=0, sticky="ew")
        self.classifier_summary_text.configure(state="disabled")

    def _add_classifier_parameter_row(self, parent: ttk.LabelFrame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.classifier_parameter_vars[key], width=14).grid(
            row=row,
            column=1,
            sticky="e",
            padx=(10, 0),
            pady=2,
        )

    def _build_switching_controls_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        files_frame = ttk.LabelFrame(parent, text="Chronological Files", padding=10)
        files_frame.grid(row=0, column=0, sticky="nsew")
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)

        self.switching_file_listbox = tk.Listbox(files_frame, height=12, exportselection=False)
        self.switching_file_listbox.grid(row=0, column=0, columnspan=2, sticky="nsew")

        ttk.Button(files_frame, text="Add Files", command=self._add_switching_files).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Use Analysis Files", command=self._copy_analysis_files_to_switching_panel).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Remove Selected", command=self._remove_selected_switching_files).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Clear Files", command=self._clear_switching_files).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Move Up", command=lambda: self._move_selected_switching_file(-1)).grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Move Down", command=lambda: self._move_selected_switching_file(1)).grid(
            row=3,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )

        windows_frame = ttk.LabelFrame(parent, text="Spectral Windows", padding=10)
        windows_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        windows_frame.columnconfigure(1, weight=1)
        self._add_switching_parameter_row(windows_frame, 0, "Fermi level (eV)", "fermi_level_ev")
        self._add_switching_parameter_row(windows_frame, 1, "EF min rel. (eV)", "ef_min_ev")
        self._add_switching_parameter_row(windows_frame, 2, "EF max rel. (eV)", "ef_max_ev")
        self._add_switching_parameter_row(windows_frame, 3, "LHB center (eV)", "lhb_center_ev")
        self._add_switching_parameter_row(windows_frame, 4, "LHB halfwidth (eV)", "lhb_halfwidth_ev")
        self._add_switching_parameter_row(windows_frame, 5, "EDC smooth sigma", "smooth_sigma")

        thresholds_frame = ttk.LabelFrame(parent, text="Switching Threshold Quantiles", padding=10)
        thresholds_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        thresholds_frame.columnconfigure(1, weight=1)
        self._add_switching_parameter_row(thresholds_frame, 0, "Stable/low switch q", "low_switch_quantile")
        self._add_switching_parameter_row(thresholds_frame, 1, "Strong switch q", "high_switch_quantile")
        self._add_switching_parameter_row(thresholds_frame, 2, "Small net-change q", "small_net_quantile")
        self._add_switching_parameter_row(thresholds_frame, 3, "Low-signal q", "low_signal_quantile")
        self._add_switching_parameter_row(thresholds_frame, 4, "Min LHB q", "lhb_min_quantile")

        actions_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        actions_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        actions_frame.columnconfigure(0, weight=1)
        ttk.Button(actions_frame, text="Compute Switching Map", command=self._run_switching_map).grid(
            row=0,
            column=0,
            sticky="ew",
        )
        ttk.Button(actions_frame, text="Save Results...", command=self._save_switching_results).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(actions_frame, text="Save Plot...", command=self._save_switching_plot).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )

        help_text = (
            "file 0 is the initial state; file i is after pulse i.\n"
            "The view aligns clipped x/y maps before comparing I_rat pixel by pixel."
        )
        ttk.Label(parent, text=help_text, justify=tk.LEFT, wraplength=320).grid(
            row=4,
            column=0,
            sticky="ew",
            pady=(12, 0),
        )

    def _build_switching_visual_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        parent.rowconfigure(2, weight=0)

        ttk.Label(
            parent,
            textvariable=self.switching_status_var,
            anchor="w",
            justify=tk.LEFT,
            wraplength=1120,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        figure_frame = ttk.Frame(parent)
        figure_frame.grid(row=1, column=0, sticky="nsew")
        figure_frame.columnconfigure(0, weight=1)
        figure_frame.rowconfigure(1, weight=1)

        toolbar_frame = ttk.Frame(figure_frame)
        toolbar_frame.grid(row=0, column=0, sticky="ew")

        self.switching_figure = Figure(figsize=(11.2, 8.8), dpi=100, constrained_layout=True)
        self.switching_canvas = FigureCanvasTkAgg(self.switching_figure, master=figure_frame)
        self.switching_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.switching_canvas.mpl_connect("button_press_event", self._on_switching_plot_click)

        try:
            self.switching_toolbar = NavigationToolbar2Tk(self.switching_canvas, toolbar_frame, pack_toolbar=False)
        except Exception:
            self.switching_toolbar = None
            ttk.Label(toolbar_frame, text="Matplotlib toolbar unavailable in this environment.").pack(side=tk.LEFT)
        else:
            self.switching_toolbar.update()
            self.switching_toolbar.pack(side=tk.LEFT, fill=tk.X)

        summary_frame = ttk.Frame(parent, padding=(0, 8, 0, 0))
        summary_frame.grid(row=2, column=0, sticky="ew")
        summary_frame.columnconfigure(0, weight=1)
        self.switching_summary_text = tk.Text(summary_frame, height=10, wrap="word")
        self.switching_summary_text.grid(row=0, column=0, sticky="ew")
        self.switching_summary_text.configure(state="disabled")

    def _add_switching_parameter_row(self, parent: ttk.LabelFrame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.switching_parameter_vars[key], width=14).grid(
            row=row,
            column=1,
            sticky="e",
            padx=(10, 0),
            pady=2,
        )

    def _build_state_prediction_controls_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self.state_prediction_controls_canvas = tk.Canvas(parent, highlightthickness=0, width=360)
        self.state_prediction_controls_scrollbar = ttk.Scrollbar(
            parent,
            orient=tk.VERTICAL,
            command=self.state_prediction_controls_canvas.yview,
        )
        self.state_prediction_controls_canvas.configure(
            yscrollcommand=self.state_prediction_controls_scrollbar.set
        )
        self.state_prediction_controls_canvas.grid(row=0, column=0, sticky="nsew")
        self.state_prediction_controls_scrollbar.grid(row=0, column=1, sticky="ns")

        content = ttk.Frame(self.state_prediction_controls_canvas)
        content.columnconfigure(0, weight=1)
        self.state_prediction_controls_window = self.state_prediction_controls_canvas.create_window(
            (0, 0),
            window=content,
            anchor="nw",
        )
        content.bind("<Configure>", self._update_state_prediction_controls_scroll_region)
        self.state_prediction_controls_canvas.bind("<Configure>", self._resize_state_prediction_controls_window)
        self.state_prediction_controls_canvas.bind("<Enter>", self._bind_state_prediction_controls_mousewheel)
        self.state_prediction_controls_canvas.bind("<Leave>", self._unbind_state_prediction_controls_mousewheel)
        content.bind("<Enter>", self._bind_state_prediction_controls_mousewheel)
        content.bind("<Leave>", self._unbind_state_prediction_controls_mousewheel)

        parent = content
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        files_frame = ttk.LabelFrame(parent, text="Chronological Files", padding=10)
        files_frame.grid(row=0, column=0, sticky="nsew")
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)

        self.state_prediction_file_listbox = tk.Listbox(files_frame, height=10, exportselection=False)
        self.state_prediction_file_listbox.grid(row=0, column=0, columnspan=2, sticky="nsew")
        ttk.Button(files_frame, text="Add Files", command=self._add_state_prediction_files).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Use Analysis Files", command=self._copy_analysis_files_to_state_prediction_panel).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Remove Selected", command=self._remove_selected_state_prediction_files).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Clear Files", command=self._clear_state_prediction_files).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Move Up", command=lambda: self._move_selected_state_prediction_file(-1)).grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Move Down", command=lambda: self._move_selected_state_prediction_file(1)).grid(
            row=3,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )

        windows_frame = ttk.LabelFrame(parent, text="Initial Spectral Features", padding=10)
        windows_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        windows_frame.columnconfigure(1, weight=1)
        self._add_state_prediction_parameter_row(windows_frame, 0, "Fermi level (eV)", "fermi_level_ev")
        self._add_state_prediction_parameter_row(windows_frame, 1, "EF min rel. (eV)", "ef_min_ev")
        self._add_state_prediction_parameter_row(windows_frame, 2, "EF max rel. (eV)", "ef_max_ev")
        self._add_state_prediction_parameter_row(windows_frame, 3, "LHB center (eV)", "lhb_center_ev")
        self._add_state_prediction_parameter_row(windows_frame, 4, "LHB halfwidth (eV)", "lhb_halfwidth_ev")
        self._add_state_prediction_parameter_row(windows_frame, 5, "LE min rel. (eV)", "leading_edge_min_ev")
        self._add_state_prediction_parameter_row(windows_frame, 6, "LE max rel. (eV)", "leading_edge_max_ev")
        self._add_state_prediction_parameter_row(windows_frame, 7, "p3 center (eV)", "p3_center_ev")
        self._add_state_prediction_parameter_row(windows_frame, 8, "p3 halfwidth (eV)", "p3_halfwidth_ev")
        self._add_state_prediction_parameter_row(windows_frame, 9, "EDC smooth sigma", "smooth_sigma")

        outcomes_frame = ttk.LabelFrame(parent, text="Future Outcome Rules", padding=10)
        outcomes_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        outcomes_frame.columnconfigure(1, weight=1)
        self._add_state_prediction_parameter_row(outcomes_frame, 0, "Stable q", "stable_quantile")
        self._add_state_prediction_parameter_row(outcomes_frame, 1, "Switcher q", "switch_quantile")
        self._add_state_prediction_parameter_row(outcomes_frame, 2, "Net tau (blank=auto)", "net_change_tau")
        self._add_state_prediction_parameter_row(outcomes_frame, 3, "Low-signal q", "low_signal_quantile")
        self._add_state_prediction_parameter_row(outcomes_frame, 4, "Min LHB q", "lhb_min_quantile")
        self._add_state_prediction_parameter_row(outcomes_frame, 5, "Phase low q", "phase_low_quantile")
        self._add_state_prediction_parameter_row(outcomes_frame, 6, "Phase high q", "phase_high_quantile")
        self._add_state_prediction_parameter_row(outcomes_frame, 7, "Structural gradient q", "structural_gradient_quantile")

        actions_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        actions_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        actions_frame.columnconfigure(0, weight=1)
        ttk.Button(actions_frame, text="Compute State Prediction", command=self._run_state_prediction).grid(
            row=0,
            column=0,
            sticky="ew",
        )
        ttk.Button(actions_frame, text="Save Results...", command=self._save_state_prediction_results).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(actions_frame, text="Save Plot...", command=self._save_state_prediction_plot).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )

        help_text = (
            "This diagnostic view asks whether initial spectral or boundary features predict later switching.\n"
            "No machine-learning model is trained here."
        )
        ttk.Label(parent, text=help_text, justify=tk.LEFT, wraplength=320).grid(
            row=4,
            column=0,
            sticky="ew",
            pady=(12, 0),
        )

    def _build_state_prediction_visual_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        parent.rowconfigure(2, weight=0)

        ttk.Label(
            parent,
            textvariable=self.state_prediction_status_var,
            anchor="w",
            justify=tk.LEFT,
            wraplength=1120,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        figure_frame = ttk.Frame(parent)
        figure_frame.grid(row=1, column=0, sticky="nsew")
        figure_frame.columnconfigure(0, weight=1)
        figure_frame.rowconfigure(1, weight=1)

        toolbar_frame = ttk.Frame(figure_frame)
        toolbar_frame.grid(row=0, column=0, sticky="ew")

        plot_frame = ttk.Frame(figure_frame)
        plot_frame.grid(row=1, column=0, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.state_prediction_plot_scroll_canvas = tk.Canvas(plot_frame, highlightthickness=0)
        self.state_prediction_plot_v_scrollbar = ttk.Scrollbar(
            plot_frame,
            orient=tk.VERTICAL,
            command=self.state_prediction_plot_scroll_canvas.yview,
        )
        self.state_prediction_plot_h_scrollbar = ttk.Scrollbar(
            plot_frame,
            orient=tk.HORIZONTAL,
            command=self.state_prediction_plot_scroll_canvas.xview,
        )
        self.state_prediction_plot_scroll_canvas.configure(
            yscrollcommand=self.state_prediction_plot_v_scrollbar.set,
            xscrollcommand=self.state_prediction_plot_h_scrollbar.set,
        )
        self.state_prediction_plot_scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self.state_prediction_plot_v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.state_prediction_plot_h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.state_prediction_plot_canvas_frame = ttk.Frame(self.state_prediction_plot_scroll_canvas)
        self.state_prediction_plot_canvas_frame.columnconfigure(0, weight=1)
        self.state_prediction_plot_canvas_frame.rowconfigure(0, weight=1)
        self.state_prediction_plot_scroll_window = self.state_prediction_plot_scroll_canvas.create_window(
            (0, 0),
            window=self.state_prediction_plot_canvas_frame,
            anchor="nw",
        )

        self.state_prediction_figure = Figure(figsize=(12.8, 13.2), dpi=100, constrained_layout=True)
        self.state_prediction_canvas = FigureCanvasTkAgg(
            self.state_prediction_figure,
            master=self.state_prediction_plot_canvas_frame,
        )
        self.state_prediction_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.state_prediction_canvas.mpl_connect("button_press_event", self._on_state_prediction_plot_click)
        self.state_prediction_plot_canvas_frame.bind("<Configure>", self._update_state_prediction_plot_scroll_region)
        self.state_prediction_plot_scroll_canvas.bind("<Configure>", self._resize_state_prediction_plot_window)
        self.state_prediction_plot_scroll_canvas.bind("<Enter>", self._bind_state_prediction_plot_mousewheel)
        self.state_prediction_plot_scroll_canvas.bind("<Leave>", self._unbind_state_prediction_plot_mousewheel)
        self.state_prediction_canvas.get_tk_widget().bind("<Enter>", self._bind_state_prediction_plot_mousewheel)
        self.state_prediction_canvas.get_tk_widget().bind("<Leave>", self._unbind_state_prediction_plot_mousewheel)

        try:
            self.state_prediction_toolbar = NavigationToolbar2Tk(self.state_prediction_canvas, toolbar_frame, pack_toolbar=False)
        except Exception:
            self.state_prediction_toolbar = None
            ttk.Label(toolbar_frame, text="Matplotlib toolbar unavailable in this environment.").pack(side=tk.LEFT)
        else:
            self.state_prediction_toolbar.update()
            self.state_prediction_toolbar.pack(side=tk.LEFT, fill=tk.X)

        summary_frame = ttk.Frame(parent, padding=(0, 8, 0, 0))
        summary_frame.grid(row=2, column=0, sticky="ew")
        summary_frame.columnconfigure(0, weight=1)
        self.state_prediction_summary_text = tk.Text(summary_frame, height=11, wrap="word")
        self.state_prediction_summary_text.grid(row=0, column=0, sticky="ew")
        self.state_prediction_summary_text.configure(state="disabled")

    def _add_state_prediction_parameter_row(self, parent: ttk.LabelFrame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.state_prediction_parameter_vars[key], width=14).grid(
            row=row,
            column=1,
            sticky="e",
            padx=(10, 0),
            pady=2,
        )

    def _update_state_prediction_controls_scroll_region(self, _event: tk.Event | None = None) -> None:
        if hasattr(self, "state_prediction_controls_canvas"):
            self.state_prediction_controls_canvas.configure(
                scrollregion=self.state_prediction_controls_canvas.bbox("all")
            )

    def _resize_state_prediction_controls_window(self, event: tk.Event) -> None:
        if not hasattr(self, "state_prediction_controls_window"):
            return
        self.state_prediction_controls_canvas.itemconfigure(
            self.state_prediction_controls_window,
            width=max(1, int(event.width)),
        )

    def _bind_state_prediction_controls_mousewheel(self, _event: tk.Event | None = None) -> None:
        self.state_prediction_controls_canvas.bind_all("<MouseWheel>", self._on_state_prediction_controls_mousewheel)

    def _unbind_state_prediction_controls_mousewheel(self, _event: tk.Event | None = None) -> None:
        self.state_prediction_controls_canvas.unbind_all("<MouseWheel>")

    def _on_state_prediction_controls_mousewheel(self, event: tk.Event) -> None:
        delta = int(getattr(event, "delta", 0))
        if delta == 0:
            return
        self.state_prediction_controls_canvas.yview_scroll(-1 if delta > 0 else 1, "units")

    def _update_state_prediction_plot_scroll_region(self, _event: tk.Event | None = None) -> None:
        if hasattr(self, "state_prediction_plot_scroll_canvas"):
            self.state_prediction_plot_scroll_canvas.configure(
                scrollregion=self.state_prediction_plot_scroll_canvas.bbox("all")
            )

    def _resize_state_prediction_plot_window(self, event: tk.Event) -> None:
        if not hasattr(self, "state_prediction_plot_scroll_window"):
            return
        dpi = float(self.state_prediction_figure.dpi or 100.0) if hasattr(self, "state_prediction_figure") else 100.0
        figure_width = int(float(self.state_prediction_figure.get_size_inches()[0]) * dpi)
        self.state_prediction_plot_scroll_canvas.itemconfigure(
            self.state_prediction_plot_scroll_window,
            width=max(1, int(event.width), figure_width),
        )

    def _bind_state_prediction_plot_mousewheel(self, _event: tk.Event | None = None) -> None:
        self.state_prediction_plot_scroll_canvas.bind_all("<MouseWheel>", self._on_state_prediction_plot_mousewheel)
        self.state_prediction_plot_scroll_canvas.bind_all("<Shift-MouseWheel>", self._on_state_prediction_plot_shift_mousewheel)

    def _unbind_state_prediction_plot_mousewheel(self, _event: tk.Event | None = None) -> None:
        self.state_prediction_plot_scroll_canvas.unbind_all("<MouseWheel>")
        self.state_prediction_plot_scroll_canvas.unbind_all("<Shift-MouseWheel>")

    def _on_state_prediction_plot_mousewheel(self, event: tk.Event) -> None:
        delta = int(getattr(event, "delta", 0))
        if delta == 0:
            return
        self.state_prediction_plot_scroll_canvas.yview_scroll(-1 if delta > 0 else 1, "units")

    def _on_state_prediction_plot_shift_mousewheel(self, event: tk.Event) -> None:
        delta = int(getattr(event, "delta", 0))
        if delta == 0:
            return
        self.state_prediction_plot_scroll_canvas.xview_scroll(-1 if delta > 0 else 1, "units")

    def _build_transition_outcome_controls_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self.transition_outcome_controls_canvas = tk.Canvas(parent, highlightthickness=0, width=360)
        self.transition_outcome_controls_scrollbar = ttk.Scrollbar(
            parent,
            orient=tk.VERTICAL,
            command=self.transition_outcome_controls_canvas.yview,
        )
        self.transition_outcome_controls_canvas.configure(
            yscrollcommand=self.transition_outcome_controls_scrollbar.set
        )
        self.transition_outcome_controls_canvas.grid(row=0, column=0, sticky="nsew")
        self.transition_outcome_controls_scrollbar.grid(row=0, column=1, sticky="ns")

        content = ttk.Frame(self.transition_outcome_controls_canvas)
        content.columnconfigure(0, weight=1)
        self.transition_outcome_controls_window = self.transition_outcome_controls_canvas.create_window(
            (0, 0),
            window=content,
            anchor="nw",
        )
        content.bind("<Configure>", self._update_transition_outcome_controls_scroll_region)
        self.transition_outcome_controls_canvas.bind("<Configure>", self._resize_transition_outcome_controls_window)
        self.transition_outcome_controls_canvas.bind("<Enter>", self._bind_transition_outcome_controls_mousewheel)
        self.transition_outcome_controls_canvas.bind("<Leave>", self._unbind_transition_outcome_controls_mousewheel)
        content.bind("<Enter>", self._bind_transition_outcome_controls_mousewheel)
        content.bind("<Leave>", self._unbind_transition_outcome_controls_mousewheel)

        files_frame = ttk.LabelFrame(content, text="Chronological Files", padding=10)
        files_frame.grid(row=0, column=0, sticky="nsew")
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)

        self.transition_outcome_file_listbox = tk.Listbox(files_frame, height=10, exportselection=False)
        self.transition_outcome_file_listbox.grid(row=0, column=0, columnspan=2, sticky="nsew")
        ttk.Button(files_frame, text="Add Files", command=self._add_transition_outcome_files).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Use Analysis Files", command=self._copy_analysis_files_to_transition_outcome_panel).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Remove Selected", command=self._remove_selected_transition_outcome_files).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Clear Files", command=self._clear_transition_outcome_files).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Move Up", command=lambda: self._move_selected_transition_outcome_file(-1)).grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Move Down", command=lambda: self._move_selected_transition_outcome_file(1)).grid(
            row=3,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )

        labels_frame = ttk.LabelFrame(content, text="Pulse Direction Labels", padding=10)
        labels_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        labels_frame.columnconfigure(0, weight=1)
        ttk.Label(labels_frame, text="Comma-separated, one per transition").grid(row=0, column=0, sticky="w")
        ttk.Entry(labels_frame, textvariable=self.transition_outcome_pulse_labels_var).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(4, 0),
        )

        windows_frame = ttk.LabelFrame(content, text="Spectral Windows", padding=10)
        windows_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        windows_frame.columnconfigure(1, weight=1)
        self._add_transition_outcome_parameter_row(windows_frame, 0, "Fermi level (eV)", "fermi_level_ev")
        self._add_transition_outcome_parameter_row(windows_frame, 1, "EF min rel. (eV)", "ef_min_ev")
        self._add_transition_outcome_parameter_row(windows_frame, 2, "EF max rel. (eV)", "ef_max_ev")
        self._add_transition_outcome_parameter_row(windows_frame, 3, "LHB center (eV)", "lhb_center_ev")
        self._add_transition_outcome_parameter_row(windows_frame, 4, "LHB halfwidth (eV)", "lhb_halfwidth_ev")
        self._add_transition_outcome_parameter_row(windows_frame, 5, "EDC smooth sigma", "smooth_sigma")

        thresholds_frame = ttk.LabelFrame(content, text="Transition Thresholds", padding=10)
        thresholds_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        thresholds_frame.columnconfigure(1, weight=1)
        self._add_transition_outcome_parameter_row(thresholds_frame, 0, "Minimum tau", "user_min_tau")
        self._add_transition_outcome_parameter_row(thresholds_frame, 1, "Strong tau multiplier", "strong_tau_multiplier")
        self._add_transition_outcome_parameter_row(thresholds_frame, 2, "Low-signal q", "low_signal_quantile")
        self._add_transition_outcome_parameter_row(thresholds_frame, 3, "Min LHB q", "lhb_min_quantile")
        self._add_transition_outcome_parameter_row(thresholds_frame, 4, "Color limit (blank=auto)", "color_limit")
        ttk.Checkbutton(
            thresholds_frame,
            text="Use relative Delta_Irat",
            variable=self.transition_outcome_parameter_vars["use_relative_delta"],
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))

        display_frame = ttk.LabelFrame(content, text="Display", padding=10)
        display_frame.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        display_frame.columnconfigure(1, weight=1)
        ttk.Label(display_frame, text="Map type").grid(row=0, column=0, sticky="w")
        self.transition_outcome_map_combo = ttk.Combobox(
            display_frame,
            textvariable=self.transition_outcome_map_var,
            values=list(self.TRANSITION_OUTCOME_MAP_OPTIONS.keys()),
            state="readonly",
            width=24,
        )
        self.transition_outcome_map_combo.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self.transition_outcome_map_combo.bind("<<ComboboxSelected>>", self._on_transition_outcome_display_changed)
        ttk.Label(display_frame, text="Inspector file").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.transition_outcome_inspector_file_combo = ttk.Combobox(
            display_frame,
            textvariable=self.transition_outcome_inspector_file_var,
            values=[],
            state="readonly",
            width=24,
        )
        self.transition_outcome_inspector_file_combo.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(8, 0))
        self.transition_outcome_inspector_file_combo.bind(
            "<<ComboboxSelected>>",
            self._on_transition_outcome_inspector_file_changed,
        )

        actions_frame = ttk.LabelFrame(content, text="Actions", padding=10)
        actions_frame.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        actions_frame.columnconfigure(0, weight=1)
        ttk.Button(actions_frame, text="Compute Transition Maps", command=self._run_transition_outcome_maps).grid(
            row=0,
            column=0,
            sticky="ew",
        )
        ttk.Button(actions_frame, text="Show All Transitions", command=self._clear_transition_outcome_focus).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(actions_frame, text="Save Results...", command=self._save_transition_outcome_results).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(actions_frame, text="Save Plot...", command=self._save_transition_outcome_plot).grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )

        help_text = (
            "Erased is a transition event here: a pixel is erased in file i -> i+1 only when I_rat decreases in that step."
        )
        ttk.Label(content, text=help_text, justify=tk.LEFT, wraplength=320).grid(
            row=6,
            column=0,
            sticky="ew",
            pady=(12, 0),
        )

    def _build_transition_outcome_visual_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        ttk.Label(
            parent,
            textvariable=self.transition_outcome_status_var,
            anchor="w",
            justify=tk.LEFT,
            wraplength=1120,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.transition_outcome_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        self.transition_outcome_paned.grid(row=1, column=0, sticky="nsew")

        figure_frame = ttk.Frame(self.transition_outcome_paned)
        figure_frame.columnconfigure(0, weight=1)
        figure_frame.rowconfigure(1, weight=1)
        self.transition_outcome_paned.add(figure_frame, weight=3)

        toolbar_frame = ttk.Frame(figure_frame)
        toolbar_frame.grid(row=0, column=0, sticky="ew")

        plot_frame = ttk.Frame(figure_frame)
        plot_frame.grid(row=1, column=0, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.transition_outcome_plot_scroll_canvas = tk.Canvas(plot_frame, highlightthickness=0)
        self.transition_outcome_plot_h_scrollbar = ttk.Scrollbar(
            plot_frame,
            orient=tk.HORIZONTAL,
            command=self.transition_outcome_plot_scroll_canvas.xview,
        )
        self.transition_outcome_plot_scroll_canvas.configure(
            xscrollcommand=self.transition_outcome_plot_h_scrollbar.set,
        )
        self.transition_outcome_plot_scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self.transition_outcome_plot_h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.transition_outcome_plot_canvas_frame = ttk.Frame(self.transition_outcome_plot_scroll_canvas)
        self.transition_outcome_plot_canvas_frame.grid_propagate(False)
        self.transition_outcome_plot_canvas_frame.columnconfigure(0, weight=1)
        self.transition_outcome_plot_canvas_frame.rowconfigure(0, weight=1)
        self.transition_outcome_plot_scroll_window = self.transition_outcome_plot_scroll_canvas.create_window(
            (0, 0),
            window=self.transition_outcome_plot_canvas_frame,
            anchor="nw",
        )

        self.transition_outcome_figure = Figure(figsize=(13.0, 8.8), dpi=100, constrained_layout=False)
        self.transition_outcome_canvas = FigureCanvasTkAgg(
            self.transition_outcome_figure,
            master=self.transition_outcome_plot_canvas_frame,
        )
        self.transition_outcome_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.transition_outcome_canvas.mpl_connect("button_press_event", self._on_transition_outcome_plot_click)
        self.transition_outcome_canvas.mpl_connect("motion_notify_event", self._on_transition_outcome_plot_hover)
        self.transition_outcome_canvas.mpl_connect("scroll_event", self._on_transition_outcome_mpl_scroll)
        self.transition_outcome_plot_canvas_frame.bind("<Configure>", self._update_transition_outcome_plot_scroll_region)
        self.transition_outcome_plot_scroll_canvas.bind("<Configure>", self._resize_transition_outcome_plot_window)
        for widget in (
            self.transition_outcome_plot_scroll_canvas,
            self.transition_outcome_plot_canvas_frame,
            self.transition_outcome_canvas.get_tk_widget(),
        ):
            self._bind_transition_outcome_plot_scroll_events(widget)

        try:
            self.transition_outcome_toolbar = NavigationToolbar2Tk(self.transition_outcome_canvas, toolbar_frame, pack_toolbar=False)
        except Exception:
            self.transition_outcome_toolbar = None
            ttk.Label(toolbar_frame, text="Matplotlib toolbar unavailable in this environment.").pack(side=tk.LEFT)
        else:
            self.transition_outcome_toolbar.update()
            self.transition_outcome_toolbar.pack(side=tk.LEFT, fill=tk.X)

        bottom_container = ttk.Frame(self.transition_outcome_paned, padding=(0, 8, 0, 0))
        bottom_container.columnconfigure(0, weight=1)
        bottom_container.rowconfigure(0, weight=1)
        self.transition_outcome_paned.add(bottom_container, weight=1)

        bottom_pane = ttk.Notebook(bottom_container)
        bottom_pane.grid(row=0, column=0, sticky="nsew")

        inspector_frame = ttk.Frame(bottom_pane, padding=8)
        inspector_frame.columnconfigure(0, weight=1)
        inspector_frame.rowconfigure(0, weight=1)
        bottom_pane.add(inspector_frame, text="Pixel Inspector")

        self.transition_outcome_inspector_figure = Figure(figsize=(11.0, 4.2), dpi=100, constrained_layout=True)
        self.transition_outcome_inspector_canvas = FigureCanvasTkAgg(
            self.transition_outcome_inspector_figure,
            master=inspector_frame,
        )
        self.transition_outcome_inspector_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        summary_frame = ttk.Frame(bottom_pane, padding=8)
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)
        bottom_pane.add(summary_frame, text="Transition Details")
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.columnconfigure(1, weight=0)
        self.transition_outcome_summary_text = tk.Text(summary_frame, height=11, wrap="word")
        self.transition_outcome_summary_text.grid(row=0, column=0, sticky="nsew")
        self.transition_outcome_summary_scrollbar = ttk.Scrollbar(
            summary_frame,
            orient=tk.VERTICAL,
            command=self.transition_outcome_summary_text.yview,
        )
        self.transition_outcome_summary_scrollbar.grid(row=0, column=1, sticky="ns")
        self.transition_outcome_summary_text.configure(yscrollcommand=self.transition_outcome_summary_scrollbar.set)
        self.transition_outcome_summary_text.configure(state="disabled")

    def _add_transition_outcome_parameter_row(self, parent: ttk.LabelFrame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.transition_outcome_parameter_vars[key], width=14).grid(
            row=row,
            column=1,
            sticky="e",
            padx=(10, 0),
            pady=2,
        )

    def _build_initial_transition_controls_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        canvas = tk.Canvas(parent, highlightthickness=0, width=390)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        content = ttk.Frame(canvas)
        content.columnconfigure(0, weight=1)
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=max(1, int(event.width))))

        files_frame = ttk.LabelFrame(content, text="File Sequence", padding=10)
        files_frame.grid(row=0, column=0, sticky="nsew")
        files_frame.columnconfigure(0, weight=1)
        self.initial_transition_file_tree = ttk.Treeview(
            files_frame,
            columns=("index", "filename", "role", "included", "notes"),
            show="headings",
            height=8,
        )
        for column, width in {
            "index": 46,
            "filename": 170,
            "role": 112,
            "included": 72,
            "notes": 120,
        }.items():
            self.initial_transition_file_tree.heading(column, text=column)
            self.initial_transition_file_tree.column(column, width=width, stretch=(column == "filename"))
        self.initial_transition_file_tree.grid(row=0, column=0, columnspan=2, sticky="nsew")
        ttk.Button(files_frame, text="Add Files", command=self._add_initial_transition_files).grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(files_frame, text="Use Analysis Files", command=self._copy_analysis_files_to_initial_transition_panel).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        ttk.Button(files_frame, text="Remove Selected", command=self._remove_selected_initial_transition_files).grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(files_frame, text="Clear Files", command=self._clear_initial_transition_files).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        ttk.Button(files_frame, text="Move Up", command=lambda: self._move_selected_initial_transition_file(-1)).grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(files_frame, text="Move Down", command=lambda: self._move_selected_initial_transition_file(1)).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        ttk.Button(files_frame, text="Set Selected as Reference", command=self._set_selected_initial_transition_reference).grid(row=4, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(files_frame, text="Toggle Include", command=self._toggle_selected_initial_transition_file).grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        setup_frame = ttk.LabelFrame(content, text="Transition Setup", padding=10)
        setup_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        setup_frame.columnconfigure(1, weight=1)
        ttk.Label(setup_frame, text="Reference").grid(row=0, column=0, sticky="w")
        self.initial_transition_reference_combo = ttk.Combobox(
            setup_frame,
            textvariable=self.initial_transition_reference_var,
            values=[],
            state="readonly",
            width=24,
        )
        self.initial_transition_reference_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.initial_transition_reference_combo.bind("<<ComboboxSelected>>", lambda _event: self._sync_initial_transition_file_tree())
        ttk.Label(setup_frame, text="Mode").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(
            setup_frame,
            textvariable=self.initial_transition_mode_var,
            values=["sequential", "initial_reference"],
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Label(setup_frame, text="Normalization").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(
            setup_frame,
            textvariable=self.initial_transition_normalization_var,
            values=list(INITIAL_TRANSITION_NORMALIZATION_MODES),
            state="readonly",
        ).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Checkbutton(
            setup_frame,
            text="Allow overlapping classes",
            variable=self.initial_transition_allow_overlap_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        windows_frame = ttk.LabelFrame(content, text="Windows and Thresholds", padding=10)
        windows_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        windows_frame.columnconfigure(1, weight=1)
        for row, (label, key) in enumerate(
            [
                ("Fermi level (eV)", "fermi_level_ev"),
                ("EF min rel. (eV)", "ef_min_ev"),
                ("EF max rel. (eV)", "ef_max_ev"),
                ("Feature min (eV)", "feature_min_ev"),
                ("Feature max (eV)", "feature_max_ev"),
                ("Asymmetry split (eV)", "asymmetry_split_ev"),
                ("Metallic percentile", "metallic_percentile"),
                ("Erasure percentile", "erasure_percentile"),
                ("Stable percentile", "stable_percentile"),
                ("Future metallic min count", "future_metallic_min_count"),
                ("Future erased min count", "future_erased_min_count"),
            ]
        ):
            self._add_initial_transition_parameter_row(windows_frame, row, label, key)

        display_frame = ttk.LabelFrame(content, text="Display", padding=10)
        display_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        display_frame.columnconfigure(1, weight=1)
        ttk.Label(display_frame, text="Aggregate map").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            display_frame,
            textvariable=self.initial_transition_aggregate_map_var,
            values=list(self.INITIAL_TRANSITION_AGGREGATE_MAP_OPTIONS.keys()),
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Button(display_frame, text="Refresh Display", command=self._refresh_initial_transition_views).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(display_frame, text="Selected transition").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.initial_transition_selected_transition_combo = ttk.Combobox(
            display_frame,
            textvariable=self.initial_transition_selected_transition_var,
            values=[],
            state="readonly",
        )
        self.initial_transition_selected_transition_combo.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        self.initial_transition_selected_transition_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_initial_transition_views())

        actions_frame = ttk.LabelFrame(content, text="Actions", padding=10)
        actions_frame.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        actions_frame.columnconfigure(0, weight=1)
        ttk.Button(actions_frame, text="Compute Initial Transition Features", command=self._run_initial_transition_analysis).grid(row=0, column=0, sticky="ew")
        ttk.Button(actions_frame, text="Export Results...", command=self._save_initial_transition_results).grid(row=1, column=0, sticky="ew", pady=(8, 0))

    def _add_initial_transition_parameter_row(self, parent: ttk.LabelFrame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.initial_transition_parameter_vars[key], width=14).grid(
            row=row,
            column=1,
            sticky="e",
            padx=(10, 0),
            pady=2,
        )

    def _build_initial_transition_visual_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        ttk.Label(
            parent,
            textvariable=self.initial_transition_status_var,
            anchor="w",
            justify=tk.LEFT,
            wraplength=1120,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        notebook = ttk.Notebook(parent)
        notebook.grid(row=1, column=0, sticky="nsew")

        aggregate_frame = ttk.Frame(notebook, padding=8)
        aggregate_frame.columnconfigure(0, weight=1)
        aggregate_frame.rowconfigure(0, weight=1)
        notebook.add(aggregate_frame, text="Aggregate Maps")
        self.initial_transition_aggregate_figure = Figure(figsize=(12, 8), dpi=100, constrained_layout=True)
        self.initial_transition_aggregate_canvas = FigureCanvasTkAgg(self.initial_transition_aggregate_figure, master=aggregate_frame)
        self.initial_transition_aggregate_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.initial_transition_aggregate_canvas.mpl_connect("button_press_event", self._on_initial_transition_plot_click)

        precursor_frame = ttk.Frame(notebook, padding=8)
        precursor_frame.columnconfigure(0, weight=1)
        precursor_frame.rowconfigure(0, weight=1)
        notebook.add(precursor_frame, text="Initial Precursors")
        self.initial_transition_precursor_figure = Figure(figsize=(12, 4.8), dpi=100, constrained_layout=True)
        self.initial_transition_precursor_canvas = FigureCanvasTkAgg(self.initial_transition_precursor_figure, master=precursor_frame)
        self.initial_transition_precursor_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.initial_transition_precursor_canvas.mpl_connect("button_press_event", self._on_initial_transition_plot_click)

        diagnostics_frame = ttk.Frame(notebook, padding=8)
        diagnostics_frame.columnconfigure(0, weight=1)
        diagnostics_frame.rowconfigure(0, weight=3)
        diagnostics_frame.rowconfigure(1, weight=1)
        notebook.add(diagnostics_frame, text="Pixel Diagnostics")
        self.initial_transition_diagnostics_figure = Figure(figsize=(12, 7.5), dpi=100, constrained_layout=True)
        self.initial_transition_diagnostics_canvas = FigureCanvasTkAgg(self.initial_transition_diagnostics_figure, master=diagnostics_frame)
        self.initial_transition_diagnostics_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.initial_transition_timeline_text = tk.Text(diagnostics_frame, height=9, wrap="word")
        self.initial_transition_timeline_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.initial_transition_timeline_text.configure(state="disabled")

        stats_frame = ttk.Frame(notebook, padding=8)
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=2)
        stats_frame.rowconfigure(1, weight=1)
        notebook.add(stats_frame, text="Population Statistics")
        self.initial_transition_stats_figure = Figure(figsize=(12, 5.5), dpi=100, constrained_layout=True)
        self.initial_transition_stats_canvas = FigureCanvasTkAgg(self.initial_transition_stats_figure, master=stats_frame)
        self.initial_transition_stats_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.initial_transition_stats_text = tk.Text(stats_frame, height=10, wrap="word")
        self.initial_transition_stats_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.initial_transition_stats_text.configure(state="disabled")

    def _build_switching_mechanism_controls_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        canvas = tk.Canvas(parent, highlightthickness=0, width=390)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        content = ttk.Frame(canvas)
        content.columnconfigure(0, weight=1)
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=max(1, int(event.width))))

        source_frame = ttk.LabelFrame(content, text="Data Source", padding=10)
        source_frame.grid(row=0, column=0, sticky="ew")
        source_frame.columnconfigure(0, weight=1)
        self.mechanism_file_listbox = tk.Listbox(source_frame, height=7, exportselection=False)
        self.mechanism_file_listbox.grid(row=0, column=0, columnspan=2, sticky="nsew")
        ttk.Button(source_frame, text="Add Files", command=self._add_mechanism_files).grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(source_frame, text="Use Analysis Files", command=self._copy_analysis_files_to_mechanism).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        ttk.Button(source_frame, text="Use Initial-State Files", command=self._copy_initial_transition_files_to_mechanism).grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(source_frame, text="Remove Selected", command=self._remove_selected_mechanism_file).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        ttk.Button(source_frame, text="Move Up", command=lambda: self._move_selected_mechanism_file(-1)).grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(source_frame, text="Move Down", command=lambda: self._move_selected_mechanism_file(1)).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        ttk.Button(source_frame, text="Clear Files", command=self._clear_mechanism_files).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.mechanism_compute_button = ttk.Button(
            source_frame,
            text="Compute Diagnostics",
            command=lambda: self._run_mechanism_diagnostics(source="files"),
        )
        self.mechanism_compute_button.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self.mechanism_initial_compute_button = ttk.Button(
            source_frame,
            text="Compute From Initial-State Results",
            command=lambda: self._run_mechanism_diagnostics(source="initial"),
        )
        self.mechanism_initial_compute_button.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.mechanism_export_button = ttk.Button(source_frame, text="Export Diagnostics...", command=self._save_mechanism_results)
        self.mechanism_export_button.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.mechanism_progress = ttk.Progressbar(source_frame, mode="indeterminate")
        self.mechanism_progress.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        groups_frame = ttk.LabelFrame(content, text="Group Filters", padding=10)
        groups_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        groups_frame.columnconfigure(1, weight=1)
        for row, (label, key) in enumerate(
            [
                ("Future metallic min count", "future_metallic_min_count"),
                ("Future erased min count", "future_erased_min_count"),
                ("Future metallic min freq", "future_metallic_min_frequency"),
                ("Future erased min freq", "future_erased_min_frequency"),
            ]
        ):
            self._add_mechanism_parameter_row(groups_frame, row, label, key)

        spectral_frame = ttk.LabelFrame(content, text="Spectral and Spatial Controls", padding=10)
        spectral_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        spectral_frame.columnconfigure(1, weight=1)
        ttk.Label(spectral_frame, text="EDC normalization").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            spectral_frame,
            textvariable=self.mechanism_edc_normalization_var,
            values=list(SWITCHING_MECHANISM_EDC_NORMALIZATIONS),
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        for row, (label, key) in enumerate(
            [
                ("Boundary smooth sigma", "boundary_smooth_sigma"),
                ("Boundary percentile", "boundary_percentile"),
                ("Component min size", "component_min_size"),
                ("Control min eV", "negative_control_min_ev"),
                ("Control max eV", "negative_control_max_ev"),
                ("Permutation controls", "permutation_count"),
            ],
            start=1,
        ):
            self._add_mechanism_parameter_row(spectral_frame, row, label, key)

        threshold_frame = ttk.LabelFrame(content, text="Artifact Threshold Sweep", padding=10)
        threshold_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        threshold_frame.columnconfigure(1, weight=1)
        ttk.Label(threshold_frame, text="Percentiles").grid(row=0, column=0, sticky="w")
        ttk.Entry(threshold_frame, textvariable=self.mechanism_parameter_vars["threshold_sweep_percentiles"], width=24).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        transition_frame = ttk.LabelFrame(content, text="Selected Transition", padding=10)
        transition_frame.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        transition_frame.columnconfigure(1, weight=1)
        ttk.Label(transition_frame, text="Transition").grid(row=0, column=0, sticky="w")
        self.mechanism_selected_transition_combo = ttk.Combobox(
            transition_frame,
            textvariable=self.mechanism_selected_transition_var,
            values=[],
            state="readonly",
        )
        self.mechanism_selected_transition_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.mechanism_selected_transition_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_mechanism_views())

        note = (
            "The diagnostics use the same thresholds, transition mode, EF window, feature window, "
            "and normalization from Initial State Transition Features. You can compute directly from files here, "
            "or reuse an already-computed Initial State Transition Features result."
        )
        ttk.Label(content, text=note, wraplength=360, justify=tk.LEFT).grid(row=5, column=0, sticky="ew", pady=(14, 0))

    def _add_mechanism_parameter_row(self, parent: ttk.LabelFrame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=self.mechanism_parameter_vars[key], width=14).grid(
            row=row,
            column=1,
            sticky="e",
            padx=(10, 0),
            pady=2,
        )

    def _build_switching_mechanism_visual_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        ttk.Label(
            parent,
            textvariable=self.mechanism_status_var,
            anchor="w",
            justify=tk.LEFT,
            wraplength=1120,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        notebook = ttk.Notebook(parent)
        notebook.grid(row=1, column=0, sticky="nsew")

        spectral_frame = ttk.Frame(notebook, padding=8)
        spectral_frame.columnconfigure(0, weight=1)
        spectral_frame.rowconfigure(0, weight=4)
        spectral_frame.rowconfigure(1, weight=1)
        notebook.add(spectral_frame, text="Spectral Evidence")

        spectral_plot_frame = ttk.Frame(spectral_frame)
        spectral_plot_frame.grid(row=0, column=0, sticky="nsew")
        spectral_plot_frame.columnconfigure(0, weight=1)
        spectral_plot_frame.rowconfigure(0, weight=1)
        self.mechanism_spectral_scroll_canvas = tk.Canvas(spectral_plot_frame, highlightthickness=0)
        self.mechanism_spectral_h_scrollbar = ttk.Scrollbar(
            spectral_plot_frame,
            orient=tk.HORIZONTAL,
            command=self.mechanism_spectral_scroll_canvas.xview,
        )
        self.mechanism_spectral_scroll_canvas.configure(
            xscrollcommand=self.mechanism_spectral_h_scrollbar.set,
        )
        self.mechanism_spectral_scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self.mechanism_spectral_h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.mechanism_spectral_canvas_frame = ttk.Frame(self.mechanism_spectral_scroll_canvas)
        self.mechanism_spectral_canvas_frame.columnconfigure(0, weight=1)
        self.mechanism_spectral_canvas_frame.rowconfigure(0, weight=1)
        self.mechanism_spectral_scroll_window = self.mechanism_spectral_scroll_canvas.create_window(
            (0, 0),
            window=self.mechanism_spectral_canvas_frame,
            anchor="nw",
        )
        self.mechanism_spectral_figure = Figure(figsize=(22.0, 9.0), dpi=100, constrained_layout=True)
        self.mechanism_spectral_canvas = FigureCanvasTkAgg(
            self.mechanism_spectral_figure,
            master=self.mechanism_spectral_canvas_frame,
        )
        self.mechanism_spectral_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.mechanism_spectral_canvas_frame.bind("<Configure>", self._update_mechanism_spectral_scroll_region)
        self.mechanism_spectral_scroll_canvas.bind("<Configure>", self._resize_mechanism_spectral_scroll_window)
        self.mechanism_spectral_scroll_canvas.bind("<Enter>", self._bind_mechanism_spectral_mousewheel)
        self.mechanism_spectral_scroll_canvas.bind("<Leave>", self._unbind_mechanism_spectral_mousewheel)
        self.mechanism_spectral_canvas.get_tk_widget().bind("<Enter>", self._bind_mechanism_spectral_mousewheel)
        self.mechanism_spectral_canvas.get_tk_widget().bind("<Leave>", self._unbind_mechanism_spectral_mousewheel)

        self.mechanism_spectral_text = tk.Text(spectral_frame, height=7, wrap="word")
        self.mechanism_spectral_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.mechanism_spectral_text.configure(state="disabled")

        spatial_frame = ttk.Frame(notebook, padding=8)
        spatial_frame.columnconfigure(0, weight=1)
        spatial_frame.rowconfigure(0, weight=4)
        spatial_frame.rowconfigure(1, weight=1)
        notebook.add(spatial_frame, text="Spatial Evidence")
        self.mechanism_spatial_figure = Figure(figsize=(13.5, 8.5), dpi=100, constrained_layout=True)
        self.mechanism_spatial_canvas = FigureCanvasTkAgg(self.mechanism_spatial_figure, master=spatial_frame)
        self.mechanism_spatial_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.mechanism_spatial_canvas.mpl_connect("button_press_event", self._on_mechanism_plot_click)
        self.mechanism_spatial_text = tk.Text(spatial_frame, height=7, wrap="word")
        self.mechanism_spatial_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.mechanism_spatial_text.configure(state="disabled")

        history_frame = ttk.Frame(notebook, padding=8)
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=4)
        history_frame.rowconfigure(1, weight=1)
        notebook.add(history_frame, text="Transition History")
        self.mechanism_history_figure = Figure(figsize=(13.5, 10.0), dpi=100, constrained_layout=True)
        self.mechanism_history_canvas = FigureCanvasTkAgg(self.mechanism_history_figure, master=history_frame)
        self.mechanism_history_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.mechanism_history_canvas.mpl_connect("button_press_event", self._on_mechanism_plot_click)
        self.mechanism_history_text = tk.Text(history_frame, height=9, wrap="word")
        self.mechanism_history_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.mechanism_history_text.configure(state="disabled")

        artifact_frame = ttk.Frame(notebook, padding=8)
        artifact_frame.columnconfigure(0, weight=1)
        artifact_frame.rowconfigure(0, weight=4)
        artifact_frame.rowconfigure(1, weight=1)
        notebook.add(artifact_frame, text="Artifact Checks")
        self.mechanism_artifact_figure = Figure(figsize=(13.5, 8.0), dpi=100, constrained_layout=True)
        self.mechanism_artifact_canvas = FigureCanvasTkAgg(self.mechanism_artifact_figure, master=artifact_frame)
        self.mechanism_artifact_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.mechanism_artifact_canvas.mpl_connect("button_press_event", self._on_mechanism_plot_click)
        self.mechanism_artifact_text = tk.Text(artifact_frame, height=8, wrap="word")
        self.mechanism_artifact_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.mechanism_artifact_text.configure(state="disabled")

        summary_frame = ttk.Frame(notebook, padding=8)
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)
        notebook.add(summary_frame, text="Summary Verdict")
        self.mechanism_summary_figure = Figure(figsize=(12, 4.5), dpi=100, constrained_layout=True)
        self.mechanism_summary_canvas = FigureCanvasTkAgg(self.mechanism_summary_figure, master=summary_frame)
        self.mechanism_summary_canvas.get_tk_widget().grid(row=0, column=0, sticky="ew")
        self.mechanism_summary_text = tk.Text(summary_frame, height=18, wrap="word")
        self.mechanism_summary_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.mechanism_summary_text.configure(state="disabled")

    def _update_mechanism_spectral_scroll_region(self, _event: tk.Event | None = None) -> None:
        if hasattr(self, "mechanism_spectral_scroll_canvas"):
            self.mechanism_spectral_scroll_canvas.configure(
                scrollregion=self.mechanism_spectral_scroll_canvas.bbox("all")
            )

    def _resize_mechanism_spectral_scroll_window(self, event: tk.Event) -> None:
        if not hasattr(self, "mechanism_spectral_scroll_window"):
            return
        dpi = float(self.mechanism_spectral_figure.dpi or 100.0) if hasattr(self, "mechanism_spectral_figure") else 100.0
        figure_width = int(float(self.mechanism_spectral_figure.get_size_inches()[0]) * dpi)
        min_width = max(2200, figure_width)
        self.mechanism_spectral_scroll_canvas.itemconfigure(
            self.mechanism_spectral_scroll_window,
            width=max(1, int(event.width), min_width),
            height=max(1, int(event.height)),
        )
        self._update_mechanism_spectral_scroll_region()

    def _bind_mechanism_spectral_mousewheel(self, _event: tk.Event | None = None) -> None:
        self.mechanism_spectral_scroll_canvas.bind_all("<Shift-MouseWheel>", self._on_mechanism_spectral_shift_mousewheel)

    def _unbind_mechanism_spectral_mousewheel(self, _event: tk.Event | None = None) -> None:
        self.mechanism_spectral_scroll_canvas.unbind_all("<Shift-MouseWheel>")

    def _on_mechanism_spectral_shift_mousewheel(self, event: tk.Event) -> None:
        delta = int(getattr(event, "delta", 0))
        if delta == 0:
            return
        self.mechanism_spectral_scroll_canvas.xview_scroll(-1 if delta > 0 else 1, "units")

    def _update_transition_outcome_controls_scroll_region(self, _event: tk.Event | None = None) -> None:
        if hasattr(self, "transition_outcome_controls_canvas"):
            self.transition_outcome_controls_canvas.configure(
                scrollregion=self.transition_outcome_controls_canvas.bbox("all")
            )

    def _resize_transition_outcome_controls_window(self, event: tk.Event) -> None:
        if not hasattr(self, "transition_outcome_controls_window"):
            return
        self.transition_outcome_controls_canvas.itemconfigure(
            self.transition_outcome_controls_window,
            width=max(1, int(event.width)),
        )

    def _bind_transition_outcome_controls_mousewheel(self, _event: tk.Event | None = None) -> None:
        self.transition_outcome_controls_canvas.bind_all("<MouseWheel>", self._on_transition_outcome_controls_mousewheel)

    def _unbind_transition_outcome_controls_mousewheel(self, _event: tk.Event | None = None) -> None:
        self.transition_outcome_controls_canvas.unbind_all("<MouseWheel>")

    def _on_transition_outcome_controls_mousewheel(self, event: tk.Event) -> None:
        delta = int(getattr(event, "delta", 0))
        if delta == 0:
            return
        self.transition_outcome_controls_canvas.yview_scroll(-1 if delta > 0 else 1, "units")

    def _update_transition_outcome_plot_scroll_region(self, _event: tk.Event | None = None) -> None:
        if hasattr(self, "transition_outcome_plot_scroll_canvas"):
            width, height = self._transition_outcome_figure_pixel_size()
            self.transition_outcome_plot_scroll_canvas.configure(scrollregion=(0, 0, width, height))
            self.transition_outcome_plot_scroll_canvas.yview_moveto(0.0)

    def _resize_transition_outcome_plot_window(self, event: tk.Event) -> None:
        if not hasattr(self, "transition_outcome_plot_scroll_window"):
            return
        figure_width, figure_height = self._transition_outcome_figure_pixel_size()
        self.transition_outcome_plot_canvas_frame.configure(width=figure_width, height=figure_height)
        self.transition_outcome_canvas.get_tk_widget().configure(width=figure_width, height=figure_height)
        self.transition_outcome_plot_scroll_canvas.itemconfigure(
            self.transition_outcome_plot_scroll_window,
            width=max(1, figure_width),
            height=max(1, figure_height),
        )
        self._update_transition_outcome_plot_scroll_region()

    def _transition_outcome_figure_pixel_size(self) -> tuple[int, int]:
        if not hasattr(self, "transition_outcome_figure"):
            return (1, 1)
        dpi = float(self.transition_outcome_figure.dpi or 100.0)
        width, height = self.transition_outcome_figure.get_size_inches()
        return (max(1, int(float(width) * dpi)), max(1, int(float(height) * dpi)))

    def _bind_transition_outcome_plot_scroll_events(self, widget: tk.Widget) -> None:
        widget.bind("<MouseWheel>", self._on_transition_outcome_plot_mousewheel)
        widget.bind("<Shift-MouseWheel>", self._on_transition_outcome_plot_shift_mousewheel)
        widget.bind("<Button-4>", self._on_transition_outcome_plot_mousewheel)
        widget.bind("<Button-5>", self._on_transition_outcome_plot_mousewheel)
        widget.bind("<Shift-Button-4>", self._on_transition_outcome_plot_shift_mousewheel)
        widget.bind("<Shift-Button-5>", self._on_transition_outcome_plot_shift_mousewheel)

    def _transition_outcome_scroll_units(self, event: tk.Event) -> int:
        number = int(getattr(event, "num", 0) or 0)
        if number == 4:
            return -12
        if number == 5:
            return 12
        delta = int(getattr(event, "delta", 0) or 0)
        if delta == 0:
            return 0
        units = max(8, min(28, abs(delta) // 10 if abs(delta) >= 10 else 8))
        return -units if delta > 0 else units

    def _on_transition_outcome_plot_mousewheel(self, event: tk.Event) -> None:
        units = self._transition_outcome_scroll_units(event)
        if units == 0:
            return "break"
        self.transition_outcome_plot_scroll_canvas.xview_scroll(units, "units")
        self.transition_outcome_plot_scroll_canvas.yview_moveto(0.0)
        self._restore_transition_outcome_axis_limits()
        return "break"

    def _on_transition_outcome_plot_shift_mousewheel(self, event: tk.Event) -> None:
        units = self._transition_outcome_scroll_units(event)
        if units != 0:
            self.transition_outcome_plot_scroll_canvas.xview_scroll(units, "units")
        self.transition_outcome_plot_scroll_canvas.yview_moveto(0.0)
        self._restore_transition_outcome_axis_limits()
        return "break"

    def _add_feature_parameter_row(self, parent: ttk.LabelFrame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.feature_parameter_vars[key], width=16).grid(
            row=row,
            column=1,
            sticky="e",
            padx=(10, 0),
            pady=2,
        )

    def _build_curve_controls_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        files_frame = ttk.LabelFrame(parent, text="Curve Files", padding=10)
        files_frame.grid(row=0, column=0, sticky="nsew")
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)

        self.curve_file_listbox = tk.Listbox(files_frame, height=11, exportselection=False)
        self.curve_file_listbox.grid(row=0, column=0, columnspan=2, sticky="nsew")

        ttk.Button(files_frame, text="Add Files", command=self._add_curve_files).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Use Analysis Files", command=self._copy_analysis_files_to_curve_panel).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Remove Selected", command=self._remove_selected_curve_files).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Clear Files", command=self._clear_curve_files).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )

        compare_frame = ttk.LabelFrame(parent, text="Pair and Map", padding=10)
        compare_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        compare_frame.columnconfigure(0, weight=1)

        ttk.Label(compare_frame, text="First file").grid(row=0, column=0, sticky="w")
        self.curve_first_combo = ttk.Combobox(
            compare_frame,
            textvariable=self.curve_first_var,
            state="readonly",
            width=34,
        )
        self.curve_first_combo.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.curve_first_combo.bind("<<ComboboxSelected>>", self._handle_curve_first_selected)

        ttk.Label(compare_frame, text="Second file").grid(row=2, column=0, sticky="w")
        self.curve_second_combo = ttk.Combobox(
            compare_frame,
            textvariable=self.curve_second_var,
            state="readonly",
            width=34,
        )
        self.curve_second_combo.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        self.curve_second_combo.bind("<<ComboboxSelected>>", self._handle_curve_second_selected)

        ttk.Label(compare_frame, text="Pixel map").grid(row=4, column=0, sticky="w")
        self.curve_map_combo = ttk.Combobox(
            compare_frame,
            textvariable=self.curve_map_var,
            values=list(self.CURVE_MAP_OPTIONS.keys()),
            state="readonly",
            width=34,
        )
        self.curve_map_combo.grid(row=5, column=0, sticky="ew")
        self.curve_map_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_curve_views())

        mode_frame = ttk.LabelFrame(parent, text="Display Mode", padding=10)
        mode_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        mode_frame.columnconfigure(0, weight=1)
        ttk.Radiobutton(
            mode_frame,
            text="Point curves",
            variable=self.curve_mode_var,
            value="point",
            command=self._refresh_curve_views,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            mode_frame,
            text="EDC/MDC waterfall",
            variable=self.curve_mode_var,
            value="waterfall",
            command=self._refresh_curve_views,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        spectral_frame = ttk.LabelFrame(parent, text="Curve Windows", padding=10)
        spectral_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        self._add_curve_parameter_row(spectral_frame, 0, "Fermi level (eV)", "fermi_level_ev")
        self._add_curve_parameter_row(spectral_frame, 1, "MDC half-window (eV)", "ef_window_ev")

        actions_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        actions_frame.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        actions_frame.columnconfigure(0, weight=1)
        ttk.Button(actions_frame, text="Run EDC/MDC Compare", command=self._run_curve_comparison).grid(
            row=0,
            column=0,
            sticky="ew",
        )
        ttk.Button(actions_frame, text="Save Curve Plot...", command=self._save_curve_plot).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )

        help_text = (
            "Click any map pixel after loading.\n"
            "EDC is summed over phi; MDC is summed over the selected energy window."
        )
        ttk.Label(parent, text=help_text, justify=tk.LEFT, wraplength=320).grid(
            row=5,
            column=0,
            sticky="ew",
            pady=(12, 0),
        )

    def _build_curve_visual_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        parent.rowconfigure(2, weight=0)

        ttk.Label(
            parent,
            textvariable=self.curve_status_var,
            anchor="w",
            justify=tk.LEFT,
            wraplength=1120,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        curve_frame = ttk.Frame(parent)
        curve_frame.grid(row=1, column=0, sticky="nsew")
        curve_frame.columnconfigure(0, weight=1)
        curve_frame.rowconfigure(1, weight=1)

        toolbar_frame = ttk.Frame(curve_frame)
        toolbar_frame.grid(row=0, column=0, sticky="ew")

        self.curve_figure = Figure(figsize=(11.2, 8.2), dpi=100, constrained_layout=True)
        self.curve_canvas = FigureCanvasTkAgg(self.curve_figure, master=curve_frame)
        self.curve_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.curve_canvas.mpl_connect("button_press_event", self._on_curve_plot_click)

        try:
            self.curve_toolbar = NavigationToolbar2Tk(self.curve_canvas, toolbar_frame, pack_toolbar=False)
        except Exception:
            self.curve_toolbar = None
            ttk.Label(toolbar_frame, text="Matplotlib toolbar unavailable in this environment.").pack(side=tk.LEFT)
        else:
            self.curve_toolbar.update()
            self.curve_toolbar.pack(side=tk.LEFT, fill=tk.X)

        summary_frame = ttk.Frame(parent, padding=(0, 8, 0, 0))
        summary_frame.grid(row=2, column=0, sticky="ew")
        summary_frame.columnconfigure(0, weight=1)

        self.curve_summary_text = tk.Text(summary_frame, height=8, wrap="word")
        self.curve_summary_text.grid(row=0, column=0, sticky="ew")
        self.curve_summary_text.configure(state="disabled")

    def _add_curve_parameter_row(self, parent: ttk.LabelFrame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.curve_parameter_vars[key], width=16).grid(
            row=row,
            column=1,
            sticky="e",
            padx=(10, 0),
            pady=2,
        )

    def _build_change_controls_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        files_frame = ttk.LabelFrame(parent, text="State Files", padding=10)
        files_frame.grid(row=0, column=0, sticky="nsew")
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)

        self.change_file_listbox = tk.Listbox(files_frame, height=13, exportselection=False)
        self.change_file_listbox.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.change_file_listbox.bind("<<ListboxSelect>>", self._handle_change_file_selection)
        self.change_file_listbox.bind("<Double-Button-1>", self._handle_change_file_double_click)
        self.change_file_listbox.bind("<ButtonPress-1>", self._start_change_file_drag)
        self.change_file_listbox.bind("<B1-Motion>", self._drag_change_file)
        self.change_file_listbox.bind("<ButtonRelease-1>", self._end_change_file_drag)

        ttk.Button(files_frame, text="Add Files", command=self._add_change_files).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Use Analysis Files", command=self._copy_analysis_files_to_change_panel).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Remove Selected", command=self._remove_selected_change_files).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Set Selected as Initial", command=self._set_selected_change_initial).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Move Up", command=lambda: self._move_selected_change_file(-1)).grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Move Down", command=lambda: self._move_selected_change_file(1)).grid(
            row=3,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )
        ttk.Button(files_frame, text="Clear Files", command=self._clear_change_files).grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(8, 0),
        )

        compare_frame = ttk.LabelFrame(parent, text="Initial-State Comparison", padding=10)
        compare_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        compare_frame.columnconfigure(0, weight=1)

        ttk.Label(compare_frame, text="Initial state").grid(row=0, column=0, sticky="w")
        self.change_initial_combo = ttk.Combobox(
            compare_frame,
            textvariable=self.change_initial_var,
            state="readonly",
            width=34,
        )
        self.change_initial_combo.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.change_initial_combo.bind("<<ComboboxSelected>>", self._handle_change_initial_selected)

        ttk.Label(compare_frame, text="Inspect target").grid(row=2, column=0, sticky="w")
        self.change_target_combo = ttk.Combobox(
            compare_frame,
            textvariable=self.change_target_var,
            state="readonly",
            width=34,
        )
        self.change_target_combo.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        self.change_target_combo.bind("<<ComboboxSelected>>", self._handle_change_target_selected)

        ttk.Label(compare_frame, text="Delta map").grid(row=4, column=0, sticky="w")
        self.change_metric_combo = ttk.Combobox(
            compare_frame,
            textvariable=self.change_metric_var,
            values=list(self.CHANGE_METRIC_OPTIONS.keys()),
            state="readonly",
            width=34,
        )
        self.change_metric_combo.grid(row=5, column=0, sticky="ew")
        self.change_metric_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_change_views())

        spectral_frame = ttk.LabelFrame(parent, text="Energy Windows", padding=10)
        spectral_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self._add_change_parameter_row(spectral_frame, 0, "Fermi level (eV)", "fermi_level_ev")
        self._add_change_parameter_row(spectral_frame, 1, "Near-EF window (eV)", "ef_window_ev")
        self._add_change_parameter_row(spectral_frame, 2, "Wide window (eV)", "wide_window_ev")

        actions_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        actions_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        actions_frame.columnconfigure(0, weight=1)
        ttk.Button(actions_frame, text="Analyze Changes", command=self._run_change_analysis).grid(
            row=0,
            column=0,
            sticky="ew",
        )
        ttk.Button(actions_frame, text="Save Change Plot...", command=self._save_change_plot).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )

        help_text = (
            "The sequence list is draggable.\n"
            "Set one file as the initial state, then order the rest in the sequence you want to compare."
        )
        ttk.Label(parent, text=help_text, justify=tk.LEFT, wraplength=320).grid(
            row=4,
            column=0,
            sticky="ew",
            pady=(12, 0),
        )

    def _build_change_visual_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        parent.rowconfigure(2, weight=1)

        ttk.Label(
            parent,
            textvariable=self.change_status_var,
            anchor="w",
            justify=tk.LEFT,
            wraplength=1120,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        detail_frame = ttk.Frame(parent)
        detail_frame.grid(row=1, column=0, sticky="nsew")
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(1, weight=1)

        toolbar_frame = ttk.Frame(detail_frame)
        toolbar_frame.grid(row=0, column=0, sticky="ew")

        self.change_figure = Figure(figsize=(11, 6.8), dpi=100, constrained_layout=True)
        self.change_canvas = FigureCanvasTkAgg(self.change_figure, master=detail_frame)
        self.change_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.change_canvas.mpl_connect("button_press_event", self._on_change_plot_click)

        try:
            self.change_toolbar = NavigationToolbar2Tk(self.change_canvas, toolbar_frame, pack_toolbar=False)
        except Exception:
            self.change_toolbar = None
            ttk.Label(toolbar_frame, text="Matplotlib toolbar unavailable in this environment.").pack(side=tk.LEFT)
        else:
            self.change_toolbar.update()
            self.change_toolbar.pack(side=tk.LEFT, fill=tk.X)

        bottom_pane = ttk.Notebook(parent)
        bottom_pane.grid(row=2, column=0, sticky="nsew", pady=(12, 0))

        sequence_frame = ttk.Frame(bottom_pane, padding=8)
        sequence_frame.columnconfigure(0, weight=1)
        sequence_frame.rowconfigure(0, weight=1)
        bottom_pane.add(sequence_frame, text="Sequence Overview")

        self.change_sequence_figure = Figure(figsize=(11, 4.8), dpi=100, constrained_layout=True)
        self.change_sequence_canvas = FigureCanvasTkAgg(self.change_sequence_figure, master=sequence_frame)
        self.change_sequence_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        summary_frame = ttk.Frame(bottom_pane, padding=8)
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)
        bottom_pane.add(summary_frame, text="Change Summary")

        self.change_summary_text = tk.Text(summary_frame, wrap="word")
        self.change_summary_text.grid(row=0, column=0, sticky="nsew")
        self.change_summary_text.configure(state="disabled")

    def _add_change_parameter_row(self, parent: ttk.LabelFrame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.change_parameter_vars[key], width=16).grid(
            row=row,
            column=1,
            sticky="e",
            padx=(10, 0),
            pady=2,
        )

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

        # The toolbar is useful but non-essential; some Tk environments report a zero icon size here.
        try:
            self.main_toolbar = NavigationToolbar2Tk(self.main_canvas, toolbar_frame, pack_toolbar=False)
        except Exception:
            self.main_toolbar = None
            ttk.Label(toolbar_frame, text="Matplotlib toolbar unavailable in this environment.").pack(side=tk.LEFT)
        else:
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

        cluster_frame = ttk.Frame(bottom_pane, padding=8)
        cluster_frame.columnconfigure(0, weight=1)
        cluster_frame.rowconfigure(2, weight=1)
        cluster_frame.rowconfigure(3, weight=0)
        bottom_pane.add(cluster_frame, text="Clustering")

        cluster_controls = ttk.Frame(cluster_frame)
        cluster_controls.grid(row=0, column=0, sticky="ew")
        cluster_controls.columnconfigure(1, weight=1)
        cluster_controls.columnconfigure(2, weight=0)
        cluster_controls.columnconfigure(3, weight=0)

        ttk.Label(cluster_controls, text="Cluster state").grid(row=0, column=0, sticky="w")
        self.cluster_state_combo = ttk.Combobox(
            cluster_controls,
            textvariable=self.cluster_state_var,
            state="readonly",
            width=20,
        )
        self.cluster_state_combo.grid(row=1, column=0, sticky="w", padx=(0, 8))
        self.cluster_state_combo.bind("<<ComboboxSelected>>", lambda _event: self._handle_cluster_selector_change())

        ttk.Label(cluster_controls, text="Method").grid(row=0, column=1, sticky="w")
        self.cluster_method_combo = ttk.Combobox(
            cluster_controls,
            textvariable=self.cluster_method_var,
            values=list(SPECTRAL_CLUSTER_METHOD_LABELS.values()),
            state="readonly",
            width=40,
        )
        self.cluster_method_combo.grid(row=1, column=1, sticky="ew", padx=(0, 8))
        self.cluster_method_combo.bind("<<ComboboxSelected>>", lambda _event: self._handle_cluster_selector_change())

        ttk.Label(cluster_controls, text="Clusters").grid(row=0, column=2, sticky="w")
        ttk.Entry(cluster_controls, textvariable=self.cluster_parameter_vars["n_clusters"], width=10).grid(
            row=1,
            column=2,
            sticky="w",
            padx=(0, 8),
        )

        ttk.Label(cluster_controls, text="Embedding PCs").grid(row=0, column=3, sticky="w")
        ttk.Entry(cluster_controls, textvariable=self.cluster_parameter_vars["embedding_components"], width=12).grid(
            row=1,
            column=3,
            sticky="w",
            padx=(0, 8),
        )

        ttk.Label(cluster_controls, text="Scatter color").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.cluster_color_combo = ttk.Combobox(
            cluster_controls,
            textvariable=self.cluster_color_var,
            values=list(self.CLUSTER_COLOR_OPTIONS.keys()),
            state="readonly",
            width=18,
        )
        self.cluster_color_combo.grid(row=3, column=0, sticky="w", padx=(0, 8))
        self.cluster_color_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_cluster_plot())

        ttk.Label(cluster_controls, text="Inspect cluster").grid(row=2, column=1, sticky="w", pady=(8, 0))
        self.cluster_focus_combo = ttk.Combobox(
            cluster_controls,
            textvariable=self.cluster_focus_var,
            state="readonly",
            width=28,
        )
        self.cluster_focus_combo.grid(row=3, column=1, columnspan=2, sticky="ew", padx=(0, 8))
        self.cluster_focus_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_cluster_plot())

        ttk.Button(cluster_controls, text="Run Clustering", command=self._run_cluster_test).grid(
            row=3,
            column=3,
            sticky="e",
            pady=(0, 2),
        )

        ttk.Label(
            cluster_frame,
            textvariable=self.cluster_status_var,
            justify=tk.LEFT,
            wraplength=1100,
        ).grid(row=1, column=0, sticky="ew", pady=(8, 8))

        self.cluster_figure = Figure(figsize=(11, 5.6), dpi=100, constrained_layout=True)
        self.cluster_canvas = FigureCanvasTkAgg(self.cluster_figure, master=cluster_frame)
        self.cluster_canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")

        self.cluster_text = tk.Text(cluster_frame, height=11, wrap="word")
        self.cluster_text.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.cluster_text.configure(state="disabled")

    def _add_parameter_row(self, parent: ttk.LabelFrame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(parent, textvariable=self.parameter_vars[key], width=16)
        entry.grid(row=row, column=1, sticky="e", padx=(10, 0), pady=2)

    def _add_sequence_files(self) -> None:
        selected = list(filedialog.askopenfilenames(title="Choose NetCDF files", filetypes=FILE_TYPES))
        if not selected:
            return

        new_paths = [str(Path(path).expanduser().resolve()) for path in selected]
        merged = self.sequence_file_paths + [path for path in new_paths if path not in self.sequence_file_paths]
        self._set_sequence_files(merged)

    def _copy_analysis_files_to_sequence_panel(self) -> None:
        if not self.file_paths:
            messagebox.showinfo("No analysis files", "Add files to the Analysis panel first, or add files here directly.")
            return
        self._set_sequence_files(self.file_paths)
        self.top_notebook.select(1)

    def _remove_selected_sequence_files(self) -> None:
        selection = list(self.sequence_file_listbox.curselection())
        if not selection:
            return
        updated_files = list(self.sequence_file_paths)
        for index in reversed(selection):
            del updated_files[index]
        self._set_sequence_files(updated_files)

    def _move_selected_sequence_file(self, direction: int) -> None:
        selection = self.sequence_file_listbox.curselection()
        if len(selection) != 1:
            return

        index = selection[0]
        new_index = index + direction
        if not 0 <= new_index < len(self.sequence_file_paths):
            return

        updated_files = list(self.sequence_file_paths)
        updated_files[index], updated_files[new_index] = updated_files[new_index], updated_files[index]
        self._set_sequence_files(updated_files)
        self.sequence_file_listbox.selection_set(new_index)

    def _clear_sequence_files(self) -> None:
        self._set_sequence_files([])

    def _set_sequence_files(self, file_paths: list[str]) -> None:
        self.sequence_file_paths = list(file_paths)
        self._clear_sequence_results()
        self._sync_sequence_file_listbox()
        self._render_sequence_placeholder()

    def _clear_sequence_results(self) -> None:
        self._cancel_sequence_comparison_refresh()
        self._clear_sequence_pixel_markers()
        self.sequence_loaded_states = []
        self.sequence_total_maps = []
        self.sequence_ef_maps = []
        self.sequence_selected_indices = []
        self.sequence_selected_pixel = None
        self.sequence_alignment_notes = []
        self.sequence_map_axes = []
        self.sequence_axis_to_index = {}

    def _cancel_sequence_comparison_refresh(self) -> None:
        refresh_after_id = self.sequence_compare_refresh_after_id
        if refresh_after_id is None:
            return
        try:
            self.root.after_cancel(refresh_after_id)
        except tk.TclError:
            pass
        self.sequence_compare_refresh_after_id = None

    def _schedule_sequence_comparison_refresh(self) -> None:
        self._cancel_sequence_comparison_refresh()
        self.sequence_compare_refresh_after_id = self.root.after(
            40,
            self._run_scheduled_sequence_comparison_refresh,
        )

    def _run_scheduled_sequence_comparison_refresh(self) -> None:
        self.sequence_compare_refresh_after_id = None
        self._refresh_sequence_comparison_plot()

    def _clear_sequence_pixel_markers(self) -> None:
        for artist in self.sequence_pixel_marker_artists:
            remove = getattr(artist, "remove", None)
            if remove is None:
                continue
            try:
                remove()
            except Exception:
                pass
        self.sequence_pixel_marker_artists = []

    def _refresh_sequence_pixel_markers(self) -> None:
        self._clear_sequence_pixel_markers()
        if self.sequence_selected_pixel is None:
            return
        for axis in self.sequence_map_axes:
            self.sequence_pixel_marker_artists.extend(
                self._mark_sequence_selected_pixel(axis, self.sequence_selected_pixel)
            )
        self.sequence_canvas.draw_idle()

    def _sync_sequence_file_listbox(self) -> None:
        self.sequence_file_listbox.delete(0, tk.END)
        for index, path in enumerate(self.sequence_file_paths):
            self.sequence_file_listbox.insert(tk.END, f"{index + 1}. {Path(path).name}")

    def _parse_sequence_parameters(self) -> tuple[float, float]:
        try:
            fermi_level = float(self.sequence_parameter_vars["fermi_level_ev"].get())
            half_window = float(self.sequence_parameter_vars["ef_window_ev"].get())
        except ValueError as exc:
            raise ValueError(f"Could not parse the sequence-viewer controls: {exc}") from exc
        if half_window <= 0:
            raise ValueError("Near-EF half-window must be positive.")
        return fermi_level, half_window

    def _run_sequence_viewer(self) -> None:
        if not self.sequence_file_paths:
            messagebox.showerror("Missing files", "Please choose at least one NetCDF file.")
            return

        try:
            fermi_level, half_window = self._parse_sequence_parameters()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self.sequence_status_var.set("Loading sequence files and preparing maps...")
        self._start_global_progress("Sequence Viewer running...")
        self.root.update_idletasks()

        try:
            loaded_states, alignment_notes = align_loaded_states_for_comparison(
                [load_state(path) for path in self.sequence_file_paths]
            )
            total_maps: list[np.ndarray] = []
            ef_maps: list[np.ndarray] = []
            for state in loaded_states:
                total_map, ef_map = total_and_ef_maps(
                    state.data_array,
                    fermi_level=fermi_level,
                    ef_window=half_window,
                )
                total_maps.append(total_map)
                ef_maps.append(ef_map)

            self.sequence_loaded_states = loaded_states
            self.sequence_total_maps = total_maps
            self.sequence_ef_maps = ef_maps
            self.sequence_alignment_notes = alignment_notes
            self.sequence_selected_indices = list(range(min(3, len(loaded_states))))
            self.sequence_selected_pixel = self._default_sequence_pixel()
        except Exception as exc:
            self._clear_sequence_results()
            self.sequence_status_var.set("Sequence viewer failed.")
            self._finish_global_progress("Sequence Viewer failed.", success=False)
            messagebox.showerror("Sequence viewer failed", str(exc))
            self._render_sequence_placeholder()
            return

        self._sync_sequence_selection_listbox()
        self._refresh_sequence_views()
        shape = self.sequence_total_maps[0].shape if self.sequence_total_maps else (0, 0)
        alignment_suffix = f" {self.sequence_alignment_notes[0]}" if self.sequence_alignment_notes else ""
        self.sequence_status_var.set(
            f"Loaded {len(self.sequence_loaded_states)} sequence file(s) as {shape[0]} x {shape[1]} maps."
            f"{alignment_suffix}"
        )
        self._finish_global_progress("Sequence Viewer complete.")

    def _sequence_map_key(self) -> str:
        return self.SEQUENCE_MAP_OPTIONS.get(self.sequence_map_var.get(), "ef_intensity")

    def _sequence_map_for_index(self, index: int) -> np.ndarray:
        key = self._sequence_map_key()
        if key == "total_intensity":
            return np.asarray(self.sequence_total_maps[index], dtype=np.float32)
        if key == "ef_fraction":
            total = np.asarray(self.sequence_total_maps[index], dtype=np.float32)
            ef = np.asarray(self.sequence_ef_maps[index], dtype=np.float32)
            return (ef / (total + 1e-8)).astype(np.float32)
        return np.asarray(self.sequence_ef_maps[index], dtype=np.float32)

    def _default_sequence_pixel(self) -> tuple[int, int]:
        if not self.sequence_total_maps:
            return (0, 0)
        average = np.mean([np.asarray(total, dtype=np.float32) for total in self.sequence_total_maps], axis=0)
        if not np.any(np.isfinite(average)):
            return (0, 0)
        flat_index = int(np.nanargmax(average))
        return divmod(flat_index, average.shape[1])

    def _sequence_map_limits(self, indices: list[int] | None = None) -> tuple[float, float]:
        if not self.sequence_loaded_states:
            return 0.0, 1.0
        if indices is None:
            indices = list(range(len(self.sequence_loaded_states)))
        values = [
            self._sequence_map_for_index(index).reshape(-1)
            for index in indices
            if 0 <= index < len(self.sequence_loaded_states)
        ]
        if not values:
            return 0.0, 1.0
        combined = np.concatenate(values)
        finite = combined[np.isfinite(combined)]
        if finite.size == 0:
            return 0.0, 1.0
        low = float(np.nanpercentile(finite, 1))
        high = float(np.nanpercentile(finite, 99))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.nanmin(finite))
            high = float(np.nanmax(finite))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return 0.0, 1.0
        return low, high

    def _refresh_sequence_views(self) -> None:
        if not self.sequence_loaded_states:
            self._render_sequence_placeholder()
            return
        self._cancel_sequence_comparison_refresh()
        self.sequence_selected_indices = [
            index for index in self.sequence_selected_indices
            if 0 <= index < len(self.sequence_loaded_states)
        ][:3]
        if self.sequence_selected_pixel is None:
            self.sequence_selected_pixel = self._default_sequence_pixel()
        self._refresh_sequence_overview_plot()
        self._refresh_sequence_comparison_plot()
        self._update_sequence_summary_text()

    def _render_sequence_placeholder(self) -> None:
        if not hasattr(self, "sequence_figure"):
            return

        self._cancel_sequence_comparison_refresh()
        self._clear_sequence_pixel_markers()
        self.sequence_figure.clear()
        axis = self.sequence_figure.add_subplot(111)
        if self.sequence_file_paths:
            message = "Ready to load.\nOrder the files, choose a map, then click Load Sequence."
        else:
            message = "Add NetCDF files to view the full sequence."
        axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=13)
        axis.set_axis_off()
        self.sequence_canvas.draw_idle()
        self._update_sequence_scroll_region()
        self.sequence_map_axes = []
        self.sequence_axis_to_index = {}

        if hasattr(self, "sequence_compare_figure"):
            self.sequence_compare_figure.clear()
            compare_axis = self.sequence_compare_figure.add_subplot(111)
            compare_axis.text(0.5, 0.5, "Select loaded sequence plots here for comparison.", ha="center", va="center", fontsize=12)
            compare_axis.set_axis_off()
            self.sequence_compare_canvas.draw_idle()
            self._update_sequence_compare_scroll_region()

        if hasattr(self, "sequence_selection_listbox"):
            self.sequence_selection_listbox.delete(0, tk.END)
        if hasattr(self, "sequence_summary_text"):
            self._set_text_widget(self.sequence_summary_text, "")
        if not self.sequence_file_paths:
            self.sequence_status_var.set("Add NetCDF files in sequence order, choose a map, then load the sequence viewer.")

    def _sync_sequence_selection_listbox(self) -> None:
        self.sequence_selection_listbox.delete(0, tk.END)
        for index, state in enumerate(self.sequence_loaded_states):
            shape = self.sequence_total_maps[index].shape if index < len(self.sequence_total_maps) else ("?", "?")
            self.sequence_selection_listbox.insert(tk.END, f"{index + 1}. {Path(state.file_path).name} ({shape[0]} x {shape[1]})")
        for index in self.sequence_selected_indices[:3]:
            if 0 <= index < len(self.sequence_loaded_states):
                self.sequence_selection_listbox.selection_set(index)

    def _handle_sequence_selection_changed(self, _event: tk.Event | None = None) -> None:
        if not self.sequence_loaded_states:
            return
        selection = list(self.sequence_selection_listbox.curselection())
        if len(selection) > 3:
            selection = selection[:3]
            self.sequence_selection_listbox.selection_clear(0, tk.END)
            for index in selection:
                self.sequence_selection_listbox.selection_set(index)
            self.sequence_status_var.set("Keeping the first three selected sequence plots for comparison.")
        self.sequence_selected_indices = [int(index) for index in selection]
        self._refresh_sequence_views()

    def _select_first_sequence_plots(self) -> None:
        if not self.sequence_loaded_states:
            return
        self.sequence_selected_indices = list(range(min(3, len(self.sequence_loaded_states))))
        self._sync_sequence_selection_listbox()
        self._refresh_sequence_views()

    def _on_sequence_plot_click(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        if not self.sequence_loaded_states or event.inaxes not in self.sequence_axis_to_index:
            return
        if event.xdata is None or event.ydata is None:
            return
        x_index = int(round(event.xdata))
        y_index = int(round(event.ydata))
        shape = self.sequence_total_maps[0].shape if self.sequence_total_maps else (0, 0)
        if not (0 <= x_index < shape[0] and 0 <= y_index < shape[1]):
            return
        selected_pixel = (x_index, y_index)
        if selected_pixel == self.sequence_selected_pixel:
            return
        self.sequence_selected_pixel = selected_pixel
        self.sequence_status_var.set(f"Highlighted pixel x={x_index}, y={y_index}.")
        self._refresh_sequence_pixel_markers()
        self._schedule_sequence_comparison_refresh()
        self._update_sequence_summary_text()

    def _refresh_sequence_overview_plot(self) -> None:
        count = len(self.sequence_loaded_states)
        self._clear_sequence_pixel_markers()
        self.sequence_figure.clear()
        self.sequence_map_axes = []
        self.sequence_axis_to_index = {}
        if count == 0:
            self.sequence_canvas.draw_idle()
            return

        cols = min(3, count)
        rows = int(np.ceil(count / cols))
        width_px = self._plot_canvas_width_px(self.sequence_scroll_canvas, fallback=1120)
        dpi = float(self.sequence_figure.dpi or 100.0)
        fig_width = width_px / dpi
        fig_height = max(5.2, rows * 4.4)
        self.sequence_figure.set_size_inches(fig_width, fig_height, forward=True)
        self.sequence_figure.set_constrained_layout_pads(
            w_pad=0.03,
            h_pad=0.04,
            wspace=0.04,
            hspace=0.07,
        )
        self.sequence_canvas.get_tk_widget().configure(width=width_px, height=int(fig_height * dpi))
        axes = self.sequence_figure.subplots(rows, cols, squeeze=False)
        vmin, vmax = self._sequence_map_limits()
        selected = set(self.sequence_selected_indices)
        highlighted_pixel = self.sequence_selected_pixel
        last_image = None
        visible_axes: list[matplotlib.axes.Axes] = []

        for axis in axes.reshape(-1):
            axis.set_visible(False)

        for index, _state in enumerate(self.sequence_loaded_states):
            axis = axes[index // cols, index % cols]
            axis.set_visible(True)
            data = self._sequence_map_for_index(index)
            last_image = axis.imshow(data.T, origin="lower", cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
            axis.set_title(self._sequence_state_label(index, max_chars=34), fontsize=9)
            axis.set_xlabel("x index")
            axis.set_ylabel("y index")
            for spine in axis.spines.values():
                spine.set_linewidth(2.2 if index in selected else 0.8)
                spine.set_edgecolor("#ffbf00" if index in selected else "#222222")
            self.sequence_pixel_marker_artists.extend(
                self._mark_sequence_selected_pixel(axis, highlighted_pixel)
            )
            self.sequence_map_axes.append(axis)
            self.sequence_axis_to_index[axis] = index
            visible_axes.append(axis)

        if last_image is not None and visible_axes:
            cbar = self.sequence_figure.colorbar(last_image, ax=visible_axes, fraction=0.022, pad=0.02)
            cbar.set_label(self.sequence_map_var.get())
        self.sequence_canvas.draw_idle()
        self._update_sequence_scroll_region()

    def _mark_sequence_selected_pixel(
        self,
        axis: matplotlib.axes.Axes,
        pixel: tuple[int, int] | None,
    ) -> list[object]:
        if pixel is None:
            return []
        x_index, y_index = pixel
        outline = axis.scatter(
            [x_index],
            [y_index],
            s=118,
            facecolors="none",
            edgecolors="white",
            linewidths=2.2,
        )
        center = axis.scatter([x_index], [y_index], s=32, c="#111111")
        return [outline, center]

    def _sequence_mean_energy_profile(self, index: int) -> np.ndarray:
        data = np.asarray(self.sequence_loaded_states[index].data_array.values, dtype=np.float32)
        profile = np.sum(data, axis=(0, 1, 3), dtype=np.float64).astype(np.float32)
        total = float(np.nansum(profile))
        if abs(total) > 1e-10:
            return (profile / total).astype(np.float32)
        max_value = float(np.nanmax(np.abs(profile))) if profile.size else 0.0
        if max_value > 0:
            return (profile / max_value).astype(np.float32)
        return np.zeros_like(profile, dtype=np.float32)

    def _refresh_sequence_comparison_plot(self) -> None:
        self.sequence_compare_figure.clear()
        selected = [
            index for index in self.sequence_selected_indices
            if 0 <= index < len(self.sequence_loaded_states)
        ][:3]
        if not selected:
            axis = self.sequence_compare_figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                "Select two or three loaded files, then click a map pixel to inspect EDC/MDC waterfalls.",
                ha="center",
                va="center",
                fontsize=12,
                wrap=True,
            )
            axis.set_axis_off()
            self.sequence_compare_canvas.draw_idle()
            self._update_sequence_compare_scroll_region()
            return

        if self.sequence_selected_pixel is None:
            self.sequence_selected_pixel = self._default_sequence_pixel()
        x_index, y_index = self.sequence_selected_pixel
        try:
            fermi_level, half_window = self._parse_sequence_parameters()
        except Exception:
            fermi_level, half_window = 0.0, 0.05

        columns = len(selected)
        width_px = self._plot_canvas_width_px(self.sequence_compare_scroll_canvas, fallback=1060)
        dpi = float(self.sequence_compare_figure.dpi or 100.0)
        figure_width = width_px / dpi
        figure_height = 9.6
        self.sequence_compare_figure.set_size_inches(figure_width, figure_height, forward=True)
        self.sequence_compare_canvas.get_tk_widget().configure(
            width=width_px,
            height=int(figure_height * dpi),
        )
        grid = self.sequence_compare_figure.add_gridspec(
            4,
            columns,
            height_ratios=[0.95, 1.1, 1.1, 1.0],
            left=0.06,
            right=0.97,
            bottom=0.06,
            top=0.955,
            wspace=0.36 if columns > 1 else 0.18,
            hspace=0.72,
        )
        vmin, vmax = self._sequence_map_limits(selected)
        map_axes: list[matplotlib.axes.Axes] = []
        last_image = None
        spectra = [self._sequence_spectrum_at_pixel(index, x_index, y_index) for index in selected]
        spectrum_scale = self._waterfall_scale(spectra)
        reference_spectrum = spectra[0]
        delta_spectra = [spectrum - reference_spectrum for spectrum in spectra[1:]]

        for column, index in enumerate(selected):
            axis = self.sequence_compare_figure.add_subplot(grid[0, column])
            data = self._sequence_map_for_index(index)
            last_image = axis.imshow(data.T, origin="lower", cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
            axis.set_title(self._sequence_state_label(index, max_chars=30), fontsize=9)
            axis.set_xlabel("x index")
            axis.set_ylabel("y index")
            self._mark_sequence_selected_pixel(axis, self.sequence_selected_pixel)
            map_axes.append(axis)

        if last_image is not None and map_axes:
            cbar = self.sequence_compare_figure.colorbar(last_image, ax=map_axes, fraction=0.024, pad=0.018)
            cbar.set_label(self.sequence_map_var.get())

        for column, (index, spectrum) in enumerate(zip(selected, spectra)):
            state = self.sequence_loaded_states[index]
            energy_axis = np.asarray(state.data_array.coords["eV"].values, dtype=np.float32)
            phi_axis = np.asarray(state.data_array.coords["phi"].values, dtype=np.float32)
            edc_axis = self.sequence_compare_figure.add_subplot(grid[1, column])
            self._plot_edc_waterfall(
                edc_axis,
                spectrum,
                energy_axis,
                phi_axis,
                title=f"EDC waterfall\n{self._sequence_state_label(index, max_chars=28)}",
                color="#1f77b4",
                scale=spectrum_scale,
            )

            mdc_axis = self.sequence_compare_figure.add_subplot(grid[2, column])
            self._plot_mdc_waterfall(
                mdc_axis,
                spectrum,
                energy_axis,
                phi_axis,
                title=f"MDC waterfall\n{self._sequence_state_label(index, max_chars=28)}",
                color="#d62728",
                scale=spectrum_scale,
                fermi_level=fermi_level,
                half_window=half_window,
            )

        if len(selected) == 1:
            state = self.sequence_loaded_states[selected[0]]
            energy_axis = np.asarray(state.data_array.coords["eV"].values, dtype=np.float32)
            phi_axis = np.asarray(state.data_array.coords["phi"].values, dtype=np.float32)
            axis = self.sequence_compare_figure.add_subplot(grid[3, 0])
            image = axis.imshow(
                reference_spectrum,
                origin="lower",
                aspect="auto",
                extent=[float(phi_axis[0]), float(phi_axis[-1]), float(energy_axis[0]), float(energy_axis[-1])],
                cmap="viridis",
            )
            axis.axhline(fermi_level, color="#222222", linestyle="--", linewidth=0.9)
            axis.set_title(f"Local spectrum at x={x_index}, y={y_index}")
            axis.set_xlabel("phi")
            axis.set_ylabel("eV")
            self.sequence_compare_figure.colorbar(image, ax=axis, fraction=0.032, pad=0.02)
        else:
            delta_limit = max((self._symmetric_change_limit(delta) for delta in delta_spectra), default=1e-6)
            diff_axes: list[matplotlib.axes.Axes] = []
            diff_image = None
            for diff_column, (index, spectrum_delta) in enumerate(zip(selected[1:], delta_spectra)):
                state = self.sequence_loaded_states[index]
                energy_axis = np.asarray(state.data_array.coords["eV"].values, dtype=np.float32)
                phi_axis = np.asarray(state.data_array.coords["phi"].values, dtype=np.float32)
                axis = self.sequence_compare_figure.add_subplot(grid[3, diff_column])
                diff_image = axis.imshow(
                    spectrum_delta,
                    origin="lower",
                    aspect="auto",
                    extent=[float(phi_axis[0]), float(phi_axis[-1]), float(energy_axis[0]), float(energy_axis[-1])],
                    cmap="coolwarm",
                    vmin=-delta_limit,
                    vmax=delta_limit,
                )
                axis.axhline(fermi_level, color="#222222", linestyle="--", linewidth=0.9)
                axis.set_title(
                    f"Difference\n{self._sequence_state_label(index, max_chars=24)} - "
                    f"{self._sequence_state_label(selected[0], max_chars=24)}",
                    fontsize=9,
                )
                axis.set_xlabel("phi")
                axis.set_ylabel("eV")
                diff_axes.append(axis)

            for empty_column in range(max(0, len(selected) - 1), columns):
                axis = self.sequence_compare_figure.add_subplot(grid[3, empty_column])
                axis.set_axis_off()
            if diff_image is not None and diff_axes:
                self.sequence_compare_figure.colorbar(diff_image, ax=diff_axes, fraction=0.024, pad=0.018)

        self.sequence_compare_canvas.draw_idle()
        self._update_sequence_compare_scroll_region()

    def _sequence_spectrum_at_pixel(self, index: int, x_index: int, y_index: int) -> np.ndarray:
        data = np.asarray(self.sequence_loaded_states[index].data_array.values, dtype=np.float32)
        x_safe = min(max(0, int(x_index)), data.shape[0] - 1)
        y_safe = min(max(0, int(y_index)), data.shape[1] - 1)
        return np.asarray(data[x_safe, y_safe, :, :], dtype=np.float32)

    def _update_sequence_summary_text(self) -> None:
        if not self.sequence_loaded_states:
            self._set_text_widget(self.sequence_summary_text, "")
            return

        selected = [
            index for index in self.sequence_selected_indices
            if 0 <= index < len(self.sequence_loaded_states)
        ][:3]
        lines = [
            f"Loaded files: {len(self.sequence_loaded_states)}",
            f"Map: {self.sequence_map_var.get()}",
            f"Selected: {', '.join(str(index + 1) for index in selected) if selected else 'none'}",
        ]
        if self.sequence_selected_pixel is not None:
            x_index, y_index = self.sequence_selected_pixel
            lines.append(f"Highlighted pixel: x={x_index}, y={y_index}")
        if self.sequence_alignment_notes:
            lines.extend(["", "Alignment:"])
            lines.extend(self.sequence_alignment_notes[:4])
            if len(self.sequence_alignment_notes) > 4:
                lines.append(f"... {len(self.sequence_alignment_notes) - 4} more alignment note(s)")
        self._set_text_widget(self.sequence_summary_text, "\n".join(lines))

    def _save_sequence_overview_plot(self) -> None:
        if not self.sequence_loaded_states:
            messagebox.showinfo("No plot", "Load the sequence before saving the overview plot.")
            return
        path = filedialog.asksaveasfilename(
            title="Save sequence overview plot",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF document", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.sequence_figure.savefig(path, dpi=220)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.sequence_status_var.set(f"Saved sequence overview plot to {path}")

    def _save_sequence_comparison_plot(self) -> None:
        if not self.sequence_loaded_states:
            messagebox.showinfo("No plot", "Load the sequence before saving the comparison plot.")
            return
        path = filedialog.asksaveasfilename(
            title="Save sequence comparison plot",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF document", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.sequence_compare_figure.savefig(path, dpi=220)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.sequence_status_var.set(f"Saved sequence comparison plot to {path}")

    def _add_change_files(self) -> None:
        selected = list(filedialog.askopenfilenames(title="Choose NetCDF files", filetypes=FILE_TYPES))
        if not selected:
            return

        new_paths = [str(Path(path).expanduser().resolve()) for path in selected]
        merged = self.change_file_paths + [path for path in new_paths if path not in self.change_file_paths]
        self._set_change_files(merged)

    def _copy_analysis_files_to_change_panel(self) -> None:
        if not self.file_paths:
            messagebox.showinfo("No analysis files", "Add files to the Analysis panel first, or add files here directly.")
            return
        self._set_change_files(self.file_paths)
        self.top_notebook.select(2)

    def _remove_selected_change_files(self) -> None:
        selection = list(self.change_file_listbox.curselection())
        if not selection:
            return
        updated_files = list(self.change_file_paths)
        for index in reversed(selection):
            del updated_files[index]
        self._set_change_files(updated_files)

    def _move_selected_change_file(self, direction: int) -> None:
        selection = self.change_file_listbox.curselection()
        if len(selection) != 1:
            return

        index = selection[0]
        new_index = index + direction
        if not 0 <= new_index < len(self.change_file_paths):
            return

        self._move_change_file_to(index, new_index)
        self.change_file_listbox.selection_set(new_index)

    def _clear_change_files(self) -> None:
        self._set_change_files([])

    def _set_selected_change_initial(self) -> None:
        selection = self.change_file_listbox.curselection()
        if len(selection) != 1:
            return
        self.change_initial_path = self.change_file_paths[selection[0]]
        if self.change_target_path == self.change_initial_path:
            self.change_target_path = self._first_non_initial_change_path()
        self._rebuild_change_sequence_stats_if_ready()
        self._update_change_selector_values()
        self._sync_change_file_listbox()
        self._refresh_change_views()

    def _handle_change_file_selection(self, _event: tk.Event | None = None) -> None:
        selection = self.change_file_listbox.curselection()
        if len(selection) != 1:
            return
        self.change_target_path = self.change_file_paths[selection[0]]
        self._update_change_selector_values()
        self._refresh_change_views()

    def _handle_change_file_double_click(self, _event: tk.Event | None = None) -> None:
        self._set_selected_change_initial()

    def _start_change_file_drag(self, event: tk.Event) -> None:
        index = self.change_file_listbox.nearest(event.y)
        if 0 <= index < len(self.change_file_paths):
            self.change_drag_index = index
            self.change_file_listbox.selection_clear(0, tk.END)
            self.change_file_listbox.selection_set(index)

    def _drag_change_file(self, event: tk.Event) -> None:
        if self.change_drag_index is None or not self.change_file_paths:
            return
        target_index = self.change_file_listbox.nearest(event.y)
        if target_index == self.change_drag_index or not 0 <= target_index < len(self.change_file_paths):
            return

        self._move_change_file_to(self.change_drag_index, target_index, refresh=False)
        self.change_drag_index = target_index
        self.change_file_listbox.selection_clear(0, tk.END)
        self.change_file_listbox.selection_set(target_index)

    def _end_change_file_drag(self, _event: tk.Event | None = None) -> None:
        if self.change_drag_index is None:
            return
        self.change_drag_index = None
        self._refresh_change_views()

    def _move_change_file_to(self, old_index: int, new_index: int, refresh: bool = True) -> None:
        if old_index == new_index:
            return

        def move_item(values: list[object]) -> None:
            if values:
                values.insert(new_index, values.pop(old_index))

        move_item(self.change_file_paths)
        move_item(self.change_loaded_states)
        move_item(self.change_total_maps)
        move_item(self.change_ef_maps)
        move_item(self.change_features_by_state)
        move_item(self.change_simple_state_label_maps)
        move_item(self.change_simple_state_code_maps)
        move_item(self.change_mean_energy_profiles)

        self._rebuild_change_sequence_stats_if_ready()
        self._sync_change_file_listbox()
        self._update_change_selector_values()
        if refresh:
            self._refresh_change_views()

    def _set_change_files(self, file_paths: list[str]) -> None:
        self.change_file_paths = list(file_paths)
        if self.change_initial_path not in self.change_file_paths:
            self.change_initial_path = self._default_change_initial_path(self.change_file_paths)
        if self.change_target_path not in self.change_file_paths:
            self.change_target_path = self._first_non_initial_change_path()
        self._clear_change_results()
        self._sync_change_file_listbox()
        self._update_change_selector_values()
        self._render_change_placeholder()

    def _clear_change_results(self) -> None:
        self.change_loaded_states = []
        self.change_total_maps = []
        self.change_ef_maps = []
        self.change_features_by_state = []
        self.change_feature_names = []
        self.change_valid_mask = None
        self.change_average_map = None
        self.change_simple_state_label_maps = []
        self.change_simple_state_code_maps = []
        self.change_simple_state_thresholds = None
        self.change_mean_energy_profiles = []
        self.change_sequence_stats = []
        self.change_selected_pixel = None

    def _default_change_initial_path(self, file_paths: list[str]) -> str | None:
        if not file_paths:
            return None
        for path in file_paths:
            stem = Path(path).stem.lower()
            if stem == "a" or stem.startswith(("a_", "a-", "a.", "file_a", "file-a")):
                return path
        return file_paths[0]

    def _first_non_initial_change_path(self) -> str | None:
        if not self.change_file_paths:
            return None
        for path in self.change_file_paths:
            if path != self.change_initial_path:
                return path
        return self.change_file_paths[0]

    def _sync_change_file_listbox(self) -> None:
        self.change_file_listbox.delete(0, tk.END)
        for index, path in enumerate(self.change_file_paths):
            labels: list[str] = []
            if path == self.change_initial_path:
                labels.append("initial")
            if path == self.change_target_path:
                labels.append("view")
            suffix = f" [{', '.join(labels)}]" if labels else ""
            self.change_file_listbox.insert(tk.END, f"{index + 1}. {Path(path).name}{suffix}")

    def _change_label_for_index(self, index: int) -> str:
        return f"{index + 1}. {Path(self.change_file_paths[index]).name}"

    def _change_index_from_label(self, label: str) -> int | None:
        for index in range(len(self.change_file_paths)):
            if label == self._change_label_for_index(index):
                return index
        return None

    def _update_change_selector_values(self) -> None:
        values = [self._change_label_for_index(index) for index in range(len(self.change_file_paths))]
        self.change_initial_combo["values"] = values
        self.change_target_combo["values"] = values

        if self.change_initial_path not in self.change_file_paths:
            self.change_initial_path = self._default_change_initial_path(self.change_file_paths)
        if self.change_target_path not in self.change_file_paths:
            self.change_target_path = self._first_non_initial_change_path()

        if self.change_initial_path in self.change_file_paths:
            self.change_initial_var.set(self._change_label_for_index(self.change_file_paths.index(self.change_initial_path)))
        else:
            self.change_initial_var.set("")

        if self.change_target_path in self.change_file_paths:
            self.change_target_var.set(self._change_label_for_index(self.change_file_paths.index(self.change_target_path)))
        else:
            self.change_target_var.set("")

    def _handle_change_initial_selected(self, _event: tk.Event | None = None) -> None:
        index = self._change_index_from_label(self.change_initial_var.get())
        if index is None:
            return
        self.change_initial_path = self.change_file_paths[index]
        if self.change_target_path == self.change_initial_path:
            self.change_target_path = self._first_non_initial_change_path()
        self._rebuild_change_sequence_stats_if_ready()
        self._sync_change_file_listbox()
        self._update_change_selector_values()
        self._refresh_change_views()

    def _handle_change_target_selected(self, _event: tk.Event | None = None) -> None:
        index = self._change_index_from_label(self.change_target_var.get())
        if index is None:
            return
        self.change_target_path = self.change_file_paths[index]
        self._sync_change_file_listbox()
        self._refresh_change_views()

    def _parse_change_parameters(self) -> AnalysisParameters:
        try:
            params = AnalysisParameters(
                fermi_level_ev=float(self.change_parameter_vars["fermi_level_ev"].get()),
                ef_window_ev=float(self.change_parameter_vars["ef_window_ev"].get()),
                wide_window_ev=float(self.change_parameter_vars["wide_window_ev"].get()),
            )
        except ValueError as exc:
            raise ValueError(f"Could not parse the change-analysis controls: {exc}") from exc
        params.validate()
        return params

    def _run_change_analysis(self) -> None:
        if not self.change_file_paths:
            messagebox.showerror("Missing files", "Please choose at least one NetCDF file.")
            return
        if self.change_initial_path not in self.change_file_paths:
            messagebox.showerror("Missing initial state", "Choose which file is the initial state.")
            return

        try:
            parameters = self._parse_change_parameters()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self.change_status_var.set("Loading files and computing initial-state deltas...")
        self._start_global_progress("Initial-State Changes running...")
        self.root.update_idletasks()

        try:
            loaded_states, alignment_notes = align_loaded_states_for_comparison(
                [load_state(path) for path in self.change_file_paths]
            )

            total_maps: list[np.ndarray] = []
            ef_maps: list[np.ndarray] = []
            features_by_state: list[dict[str, np.ndarray]] = []
            feature_names: list[str] | None = None

            for state in loaded_states:
                total_map, ef_map = total_and_ef_maps(
                    state.data_array,
                    fermi_level=parameters.fermi_level_ev,
                    ef_window=parameters.ef_window_ev,
                )
                features, names, _ = extract_pixel_features(
                    state.data_array,
                    fermi_level=parameters.fermi_level_ev,
                    ef_window=parameters.ef_window_ev,
                    wide_window=parameters.wide_window_ev,
                )
                total_maps.append(total_map)
                ef_maps.append(ef_map)
                features_by_state.append(features)
                if feature_names is None:
                    feature_names = names

            valid_mask, average_map, _, _, _ = build_cross_mask_from_maps(
                total_maps,
                threshold_quantile=parameters.cross_threshold_quantile,
                row_fraction=parameters.cross_row_fraction,
                col_fraction=parameters.cross_col_fraction,
                background_quantile=parameters.cross_background_quantile,
                pad=parameters.cross_pad,
            )
            if not np.any(valid_mask):
                raise ValueError("The current cross-mask settings excluded every pixel.")

            simple_labels, simple_codes, simple_thresholds = build_simple_state_maps(
                features_by_state,
                valid_mask,
                low_quantile=parameters.simple_state_low_quantile,
                high_quantile=parameters.simple_state_high_quantile,
            )

            self.change_loaded_states = loaded_states
            self.change_total_maps = total_maps
            self.change_ef_maps = ef_maps
            self.change_features_by_state = features_by_state
            self.change_feature_names = feature_names or []
            self.change_valid_mask = valid_mask
            self.change_average_map = average_map
            self.change_simple_state_label_maps = simple_labels
            self.change_simple_state_code_maps = simple_codes
            self.change_simple_state_thresholds = simple_thresholds
            self.change_mean_energy_profiles = [
                self._compute_change_mean_energy_profile(state, valid_mask)
                for state in loaded_states
            ]
            self.change_sequence_stats = self._build_change_sequence_stats()
        except Exception as exc:
            self._clear_change_results()
            self.change_status_var.set("Initial-state change analysis failed.")
            self._finish_global_progress("Initial-State Changes failed.", success=False)
            messagebox.showerror("Change analysis failed", str(exc))
            self._render_change_placeholder()
            return

        self.change_selected_pixel = None
        if self.change_target_path not in self.change_file_paths:
            self.change_target_path = self._first_non_initial_change_path()
        self._update_change_selector_values()
        self._sync_change_file_listbox()
        self._refresh_change_views()
        alignment_suffix = f" {alignment_notes[0]}" if alignment_notes else ""
        self.change_status_var.set(
            f"Loaded {len(self.change_loaded_states)} file(s). Comparing every ordered state to initial file {Path(self.change_initial_path or '').name}."
            f"{alignment_suffix}"
        )
        self._finish_global_progress("Initial-State Changes complete.")

    def _compute_change_mean_energy_profile(self, state: LoadedState, valid_mask: np.ndarray) -> np.ndarray:
        data = np.asarray(state.data_array.values, dtype=np.float32)
        x_size, y_size, e_size, phi_size = data.shape
        spectra = data.reshape(x_size * y_size, e_size, phi_size)
        selected = spectra[valid_mask.reshape(-1)]
        if selected.size == 0:
            return np.zeros(e_size, dtype=np.float32)
        return np.sum(selected, axis=(0, 2), dtype=np.float64).astype(np.float32)

    def _normalized_change_profile(self, profile: np.ndarray) -> np.ndarray:
        values = np.asarray(profile, dtype=np.float32)
        total = float(np.nansum(values))
        if abs(total) > 1e-10:
            return (values / total).astype(np.float32)
        max_value = float(np.nanmax(np.abs(values))) if values.size else 0.0
        if max_value > 0:
            return (values / max_value).astype(np.float32)
        return np.zeros_like(values, dtype=np.float32)

    def _current_change_initial_index(self) -> int:
        if self.change_initial_path in self.change_file_paths:
            return self.change_file_paths.index(self.change_initial_path)
        return 0

    def _current_change_target_index(self) -> int:
        if self.change_target_path in self.change_file_paths:
            return self.change_file_paths.index(self.change_target_path)
        return self._current_change_initial_index()

    def _change_metric_key(self) -> str:
        return self.CHANGE_METRIC_OPTIONS.get(self.change_metric_var.get(), "ef_fraction")

    def _rebuild_change_sequence_stats_if_ready(self) -> None:
        if self.change_valid_mask is None or not self.change_features_by_state or not self.change_mean_energy_profiles:
            return
        self.change_sequence_stats = self._build_change_sequence_stats()

    def _build_change_sequence_stats(self) -> list[dict[str, object]]:
        if self.change_valid_mask is None or not self.change_features_by_state or not self.change_mean_energy_profiles:
            return []

        baseline_index = self._current_change_initial_index()
        valid_mask = self.change_valid_mask
        baseline_codes = self.change_simple_state_code_maps[baseline_index]
        baseline_features = self.change_features_by_state[baseline_index]
        baseline_profile = self._normalized_change_profile(self.change_mean_energy_profiles[baseline_index])
        energy_axis = self._change_energy_axis()
        stats: list[dict[str, object]] = []

        for index, path in enumerate(self.change_file_paths):
            target_codes = self.change_simple_state_code_maps[index]
            target_features = self.change_features_by_state[index]
            valid = valid_mask & (baseline_codes >= 0) & (target_codes >= 0)
            changed_count = int(np.sum(valid & (baseline_codes != target_codes)))
            valid_count = max(1, int(np.sum(valid)))
            target_profile = self._normalized_change_profile(self.change_mean_energy_profiles[index])
            delta_profile = target_profile - baseline_profile
            gain_index = int(np.nanargmax(delta_profile)) if delta_profile.size else 0
            loss_index = int(np.nanargmin(delta_profile)) if delta_profile.size else 0
            transition_counts = self._change_transition_counts(baseline_index, index)
            stats.append(
                {
                    "index": index,
                    "path": path,
                    "name": Path(path).name,
                    "changed_count": changed_count,
                    "changed_fraction": float(changed_count / valid_count),
                    "delta_ef_fraction": self._mean_change_delta(target_features, baseline_features, "ef_fraction", valid),
                    "delta_e_centroid": self._mean_change_delta(target_features, baseline_features, "e_centroid", valid),
                    "delta_total_intensity": self._mean_change_delta(target_features, baseline_features, "total_intensity", valid),
                    "delta_profile": delta_profile,
                    "dominant_gain_ev": float(energy_axis[gain_index]) if energy_axis.size else float("nan"),
                    "dominant_loss_ev": float(energy_axis[loss_index]) if energy_axis.size else float("nan"),
                    "profile_rms_delta": float(np.sqrt(np.nanmean(delta_profile**2))) if delta_profile.size else 0.0,
                    "transition_counts": transition_counts,
                }
            )
        return stats

    def _mean_change_delta(
        self,
        target_features: dict[str, np.ndarray],
        baseline_features: dict[str, np.ndarray],
        key: str,
        valid: np.ndarray,
    ) -> float:
        if not np.any(valid):
            return 0.0
        delta = np.asarray(target_features[key], dtype=np.float32) - np.asarray(baseline_features[key], dtype=np.float32)
        return float(np.nanmean(delta[valid]))

    def _change_energy_axis(self) -> np.ndarray:
        if not self.change_loaded_states:
            return np.array([], dtype=np.float32)
        return np.asarray(self.change_loaded_states[0].data_array.coords["eV"].values, dtype=np.float32)

    def _change_transition_counts(self, from_index: int, to_index: int) -> np.ndarray:
        if self.change_valid_mask is None or not self.change_simple_state_code_maps:
            return np.zeros((3, 3), dtype=int)
        from_codes = self.change_simple_state_code_maps[from_index]
        to_codes = self.change_simple_state_code_maps[to_index]
        valid = self.change_valid_mask & (from_codes >= 0) & (to_codes >= 0)
        counts = np.zeros((3, 3), dtype=int)
        for from_code in range(3):
            for to_code in range(3):
                counts[from_code, to_code] = int(np.sum(valid & (from_codes == from_code) & (to_codes == to_code)))
        return counts

    def _add_curve_files(self) -> None:
        selected = list(filedialog.askopenfilenames(title="Choose NetCDF files", filetypes=FILE_TYPES))
        if not selected:
            return

        new_paths = [str(Path(path).expanduser().resolve()) for path in selected]
        merged = self.curve_file_paths + [path for path in new_paths if path not in self.curve_file_paths]
        self._set_curve_files(merged)

    def _copy_analysis_files_to_curve_panel(self) -> None:
        if not self.file_paths:
            messagebox.showinfo("No analysis files", "Add files to the Analysis panel first, or add files here directly.")
            return
        self._set_curve_files(self.file_paths)
        self.top_notebook.select(3)

    def _remove_selected_curve_files(self) -> None:
        selection = list(self.curve_file_listbox.curselection())
        if not selection:
            return
        updated_files = list(self.curve_file_paths)
        for index in reversed(selection):
            del updated_files[index]
        self._set_curve_files(updated_files)

    def _clear_curve_files(self) -> None:
        self._set_curve_files([])

    def _set_curve_files(self, file_paths: list[str]) -> None:
        self.curve_file_paths = list(file_paths)
        if self.curve_first_path not in self.curve_file_paths:
            self.curve_first_path = self.curve_file_paths[0] if self.curve_file_paths else None
        if self.curve_second_path not in self.curve_file_paths or self.curve_second_path == self.curve_first_path:
            self.curve_second_path = self._first_curve_path_excluding(self.curve_first_path)
        self._clear_curve_results()
        self._sync_curve_file_listbox()
        self._update_curve_selector_values()
        self._render_curve_placeholder()

    def _clear_curve_results(self) -> None:
        self.curve_loaded_states = []
        self.curve_total_maps = []
        self.curve_ef_maps = []
        self.curve_selected_pixel = None
        self.curve_map_axes = []

    def _first_curve_path_excluding(self, excluded_path: str | None) -> str | None:
        for path in self.curve_file_paths:
            if path != excluded_path:
                return path
        return self.curve_file_paths[0] if self.curve_file_paths else None

    def _sync_curve_file_listbox(self) -> None:
        self.curve_file_listbox.delete(0, tk.END)
        for index, path in enumerate(self.curve_file_paths):
            labels: list[str] = []
            if path == self.curve_first_path:
                labels.append("first")
            if path == self.curve_second_path:
                labels.append("second")
            suffix = f" [{', '.join(labels)}]" if labels else ""
            self.curve_file_listbox.insert(tk.END, f"{index + 1}. {Path(path).name}{suffix}")

    def _curve_label_for_index(self, index: int) -> str:
        return f"{index + 1}. {Path(self.curve_file_paths[index]).name}"

    def _curve_index_from_label(self, label: str) -> int | None:
        for index in range(len(self.curve_file_paths)):
            if label == self._curve_label_for_index(index):
                return index
        return None

    def _update_curve_selector_values(self) -> None:
        values = [self._curve_label_for_index(index) for index in range(len(self.curve_file_paths))]
        self.curve_first_combo["values"] = values
        self.curve_second_combo["values"] = values

        if self.curve_first_path not in self.curve_file_paths:
            self.curve_first_path = self.curve_file_paths[0] if self.curve_file_paths else None
        if self.curve_second_path not in self.curve_file_paths or self.curve_second_path == self.curve_first_path:
            self.curve_second_path = self._first_curve_path_excluding(self.curve_first_path)

        if self.curve_first_path in self.curve_file_paths:
            self.curve_first_var.set(self._curve_label_for_index(self.curve_file_paths.index(self.curve_first_path)))
        else:
            self.curve_first_var.set("")

        if self.curve_second_path in self.curve_file_paths:
            self.curve_second_var.set(self._curve_label_for_index(self.curve_file_paths.index(self.curve_second_path)))
        else:
            self.curve_second_var.set("")

    def _handle_curve_first_selected(self, _event: tk.Event | None = None) -> None:
        index = self._curve_index_from_label(self.curve_first_var.get())
        if index is None:
            return
        self.curve_first_path = self.curve_file_paths[index]
        if self.curve_second_path == self.curve_first_path:
            self.curve_second_path = self._first_curve_path_excluding(self.curve_first_path)
        self._clear_curve_results()
        self._sync_curve_file_listbox()
        self._update_curve_selector_values()
        self._render_curve_placeholder()

    def _handle_curve_second_selected(self, _event: tk.Event | None = None) -> None:
        index = self._curve_index_from_label(self.curve_second_var.get())
        if index is None:
            return
        self.curve_second_path = self.curve_file_paths[index]
        self._clear_curve_results()
        self._sync_curve_file_listbox()
        self._update_curve_selector_values()
        self._render_curve_placeholder()

    def _parse_curve_parameters(self) -> tuple[float, float]:
        try:
            fermi_level = float(self.curve_parameter_vars["fermi_level_ev"].get())
            half_window = float(self.curve_parameter_vars["ef_window_ev"].get())
        except ValueError as exc:
            raise ValueError(f"Could not parse the EDC/MDC controls: {exc}") from exc
        if half_window <= 0:
            raise ValueError("MDC half-window must be positive.")
        return fermi_level, half_window

    def _run_curve_comparison(self) -> None:
        if len(self.curve_file_paths) < 2:
            messagebox.showerror("Missing files", "Please choose at least two NetCDF files.")
            return
        if self.curve_first_path is None or self.curve_second_path is None:
            messagebox.showerror("Missing pair", "Choose the first and second files to compare.")
            return
        if self.curve_first_path == self.curve_second_path:
            messagebox.showerror("Same file", "Choose two different files for the EDC/MDC comparison.")
            return

        try:
            fermi_level, half_window = self._parse_curve_parameters()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self.curve_status_var.set("Loading selected files and building EDC/MDC comparison...")
        self._start_global_progress("EDC/MDC Compare running...")
        self.root.update_idletasks()

        try:
            loaded_states, alignment_notes = align_loaded_states_for_comparison(
                [load_state(self.curve_first_path), load_state(self.curve_second_path)]
            )

            total_maps: list[np.ndarray] = []
            ef_maps: list[np.ndarray] = []
            for state in loaded_states:
                total_map, ef_map = total_and_ef_maps(
                    state.data_array,
                    fermi_level=fermi_level,
                    ef_window=half_window,
                )
                total_maps.append(total_map)
                ef_maps.append(ef_map)

            self.curve_loaded_states = loaded_states
            self.curve_total_maps = total_maps
            self.curve_ef_maps = ef_maps
            self.curve_selected_pixel = self._default_curve_pixel()
        except Exception as exc:
            self._clear_curve_results()
            self.curve_status_var.set("EDC/MDC comparison failed.")
            self._finish_global_progress("EDC/MDC Compare failed.", success=False)
            messagebox.showerror("EDC/MDC comparison failed", str(exc))
            self._render_curve_placeholder()
            return

        self._sync_curve_file_listbox()
        self._update_curve_selector_values()
        self._refresh_curve_views()
        alignment_suffix = f" {alignment_notes[0]}" if alignment_notes else ""
        self.curve_status_var.set(
            f"Comparing {Path(self.curve_first_path).name} to {Path(self.curve_second_path).name}. Click a map pixel to update curves."
            f"{alignment_suffix}"
        )
        self._finish_global_progress("EDC/MDC Compare complete.")

    def _default_curve_pixel(self) -> tuple[int, int]:
        if not self.curve_total_maps:
            return (0, 0)
        average = np.mean([self._curve_map_for_pair_index(index) for index in range(2)], axis=0)
        if not np.any(np.isfinite(average)):
            return (0, 0)
        flat_index = int(np.nanargmax(average))
        x_size, y_size = average.shape
        return divmod(flat_index, y_size)

    def _curve_map_key(self) -> str:
        return self.CURVE_MAP_OPTIONS.get(self.curve_map_var.get(), "ef_intensity")

    def _curve_map_for_pair_index(self, index: int) -> np.ndarray:
        key = self._curve_map_key()
        if key == "total_intensity":
            return np.asarray(self.curve_total_maps[index], dtype=np.float32)
        if key == "ef_fraction":
            total = np.asarray(self.curve_total_maps[index], dtype=np.float32)
            ef = np.asarray(self.curve_ef_maps[index], dtype=np.float32)
            return (ef / (total + 1e-8)).astype(np.float32)
        return np.asarray(self.curve_ef_maps[index], dtype=np.float32)

    def _refresh_curve_views(self) -> None:
        if not self.curve_loaded_states or len(self.curve_loaded_states) != 2:
            self._render_curve_placeholder()
            return
        if self.curve_selected_pixel is None:
            self.curve_selected_pixel = self._default_curve_pixel()
        if self.curve_mode_var.get() == "waterfall":
            self._refresh_curve_waterfall_plot()
        else:
            self._refresh_curve_plot()
        self._update_curve_summary_text()

    def _render_curve_placeholder(self) -> None:
        if not hasattr(self, "curve_figure"):
            return

        self.curve_figure.clear()
        axis = self.curve_figure.add_subplot(111)
        if len(self.curve_file_paths) >= 2:
            message = "Ready to compare.\nChoose two files, set the MDC window, then click Run EDC/MDC Compare."
        else:
            message = "Add at least two NetCDF files to compare point EDC and MDC curves."
        axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=13)
        axis.set_axis_off()
        self.curve_canvas.draw_idle()
        self.curve_map_axes = []
        self._set_text_widget(self.curve_summary_text, "")
        if len(self.curve_file_paths) < 2:
            self.curve_status_var.set("Add at least two NetCDF files, choose a pair, then run the EDC/MDC comparison.")

    def _refresh_curve_plot(self) -> None:
        assert self.curve_selected_pixel is not None
        first_state, second_state = self.curve_loaded_states
        first_name = Path(first_state.file_path).name
        second_name = Path(second_state.file_path).name
        first_map = self._curve_map_for_pair_index(0)
        second_map = self._curve_map_for_pair_index(1)
        delta_map = second_map - first_map
        x_index, y_index = self.curve_selected_pixel

        first_spectrum = np.asarray(first_state.data_array.values[x_index, y_index, :, :], dtype=np.float32)
        second_spectrum = np.asarray(second_state.data_array.values[x_index, y_index, :, :], dtype=np.float32)
        energy_axis = np.asarray(first_state.data_array.coords["eV"].values, dtype=np.float32)
        phi_axis = np.asarray(first_state.data_array.coords["phi"].values, dtype=np.float32)
        fermi_level, half_window = self._parse_curve_parameters()
        energy_mask = np.abs(energy_axis - fermi_level) <= half_window
        if not np.any(energy_mask):
            energy_mask[np.argmin(np.abs(energy_axis - fermi_level))] = True

        first_edc = first_spectrum.sum(axis=1)
        second_edc = second_spectrum.sum(axis=1)
        first_mdc = first_spectrum[energy_mask, :].sum(axis=0)
        second_mdc = second_spectrum[energy_mask, :].sum(axis=0)

        first_edc_plot, second_edc_plot, delta_edc_plot, edc_scale = self._curve_display_traces(first_edc, second_edc)
        first_mdc_plot, second_mdc_plot, delta_mdc_plot, mdc_scale = self._curve_display_traces(first_mdc, second_mdc)

        self.curve_figure.clear()
        axes = self.curve_figure.subplots(2, 3)
        first_axis, second_axis, overlay_axis = axes[0]
        edc_axis, mdc_axis, diff_axis = axes[1]
        self.curve_map_axes = [first_axis, second_axis, overlay_axis]

        first_image = first_axis.imshow(first_map.T, origin="lower", cmap="viridis", aspect="auto")
        first_axis.set_title(f"First: {first_name}\n{self.curve_map_var.get()}")
        self.curve_figure.colorbar(first_image, ax=first_axis, fraction=0.046, pad=0.04)

        second_image = second_axis.imshow(second_map.T, origin="lower", cmap="viridis", aspect="auto")
        second_axis.set_title(f"Second: {second_name}\n{self.curve_map_var.get()}")
        self.curve_figure.colorbar(second_image, ax=second_axis, fraction=0.046, pad=0.04)

        overlay_axis.imshow(self._build_curve_rgb_overlay(first_map, second_map).transpose(1, 0, 2), origin="lower", aspect="auto")
        if float(np.nanmin(delta_map)) <= 0.0 <= float(np.nanmax(delta_map)):
            overlay_axis.contour(delta_map.T, levels=[0.0], colors=["white"], linewidths=0.7, alpha=0.75)
        overlay_axis.set_title("Map overlay\ncyan=first, magenta=second")

        for axis in self.curve_map_axes:
            axis.set_xlabel("x index")
            axis.set_ylabel("y index")
            self._mark_curve_selected_pixel(axis)

        edc_axis.plot(energy_axis, first_edc_plot, color="#1f77b4", linewidth=1.8, label="first")
        edc_axis.plot(energy_axis, second_edc_plot, color="#d62728", linewidth=1.8, label="second")
        edc_axis.plot(energy_axis, delta_edc_plot, color="#222222", linewidth=1.3, label="second - first")
        edc_axis.axhline(0.0, color="#555555", linewidth=0.8)
        edc_axis.axvline(fermi_level, color="#777777", linestyle="--", linewidth=0.9)
        edc_axis.set_title(f"EDC at x={x_index}, y={y_index}\nshared scale={edc_scale:.4g}")
        edc_axis.set_xlabel("eV")
        edc_axis.set_ylabel("scaled intensity")
        edc_axis.legend(loc="best", fontsize=8)

        mdc_axis.plot(phi_axis, first_mdc_plot, color="#1f77b4", linewidth=1.8, label="first")
        mdc_axis.plot(phi_axis, second_mdc_plot, color="#d62728", linewidth=1.8, label="second")
        mdc_axis.plot(phi_axis, delta_mdc_plot, color="#222222", linewidth=1.3, label="second - first")
        mdc_axis.axhline(0.0, color="#555555", linewidth=0.8)
        mdc_axis.set_title(f"MDC at E={fermi_level:+.3f} +/- {half_window:.3f} eV\nshared scale={mdc_scale:.4g}")
        mdc_axis.set_xlabel("phi")
        mdc_axis.set_ylabel("scaled intensity")
        mdc_axis.legend(loc="best", fontsize=8)

        spectrum_delta = second_spectrum - first_spectrum
        vmax = self._symmetric_change_limit(spectrum_delta)
        diff_image = diff_axis.imshow(
            spectrum_delta,
            origin="lower",
            aspect="auto",
            extent=[float(phi_axis[0]), float(phi_axis[-1]), float(energy_axis[0]), float(energy_axis[-1])],
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        diff_axis.axhline(fermi_level, color="#222222", linestyle="--", linewidth=0.9)
        diff_axis.set_title("Local spectrum difference\nsecond - first")
        diff_axis.set_xlabel("phi")
        diff_axis.set_ylabel("eV")
        self.curve_figure.colorbar(diff_image, ax=diff_axis, fraction=0.046, pad=0.04)

        self.curve_canvas.draw_idle()

    def _refresh_curve_waterfall_plot(self) -> None:
        assert self.curve_selected_pixel is not None
        first_state, second_state = self.curve_loaded_states
        first_name = Path(first_state.file_path).name
        second_name = Path(second_state.file_path).name
        first_map = self._curve_map_for_pair_index(0)
        second_map = self._curve_map_for_pair_index(1)
        delta_map = second_map - first_map
        x_index, y_index = self.curve_selected_pixel

        first_spectrum = np.asarray(first_state.data_array.values[x_index, y_index, :, :], dtype=np.float32)
        second_spectrum = np.asarray(second_state.data_array.values[x_index, y_index, :, :], dtype=np.float32)
        spectrum_delta = second_spectrum - first_spectrum
        energy_axis = np.asarray(first_state.data_array.coords["eV"].values, dtype=np.float32)
        phi_axis = np.asarray(first_state.data_array.coords["phi"].values, dtype=np.float32)
        fermi_level, half_window = self._parse_curve_parameters()

        first_scale = self._waterfall_scale([first_spectrum, second_spectrum])
        delta_scale = self._waterfall_scale([spectrum_delta])

        self.curve_figure.clear()
        axes = self.curve_figure.subplots(3, 3, height_ratios=[0.9, 1.25, 1.25])
        first_axis, second_axis, overlay_axis = axes[0]
        edc_first_axis, edc_second_axis, edc_delta_axis = axes[1]
        mdc_first_axis, mdc_second_axis, mdc_delta_axis = axes[2]
        self.curve_map_axes = [first_axis, second_axis, overlay_axis]

        first_image = first_axis.imshow(first_map.T, origin="lower", cmap="viridis", aspect="auto")
        first_axis.set_title(f"First: {first_name}")
        self.curve_figure.colorbar(first_image, ax=first_axis, fraction=0.046, pad=0.04)

        second_image = second_axis.imshow(second_map.T, origin="lower", cmap="viridis", aspect="auto")
        second_axis.set_title(f"Second: {second_name}")
        self.curve_figure.colorbar(second_image, ax=second_axis, fraction=0.046, pad=0.04)

        overlay_axis.imshow(self._build_curve_rgb_overlay(first_map, second_map).transpose(1, 0, 2), origin="lower", aspect="auto")
        if float(np.nanmin(delta_map)) <= 0.0 <= float(np.nanmax(delta_map)):
            overlay_axis.contour(delta_map.T, levels=[0.0], colors=["white"], linewidths=0.7, alpha=0.75)
        overlay_axis.set_title("Overlay\ncyan=first, magenta=second")

        for axis in self.curve_map_axes:
            axis.set_xlabel("x index")
            axis.set_ylabel("y index")
            self._mark_curve_selected_pixel(axis)

        self._plot_edc_waterfall(
            edc_first_axis,
            first_spectrum,
            energy_axis,
            phi_axis,
            title="EDC waterfall: first",
            color="#1f77b4",
            scale=first_scale,
        )
        self._plot_edc_waterfall(
            edc_second_axis,
            second_spectrum,
            energy_axis,
            phi_axis,
            title="EDC waterfall: second",
            color="#d62728",
            scale=first_scale,
        )
        self._plot_edc_waterfall(
            edc_delta_axis,
            spectrum_delta,
            energy_axis,
            phi_axis,
            title="EDC waterfall: second - first",
            color="#222222",
            scale=delta_scale,
        )

        self._plot_mdc_waterfall(
            mdc_first_axis,
            first_spectrum,
            energy_axis,
            phi_axis,
            title="MDC waterfall: first",
            color="#1f77b4",
            scale=first_scale,
            fermi_level=fermi_level,
            half_window=half_window,
        )
        self._plot_mdc_waterfall(
            mdc_second_axis,
            second_spectrum,
            energy_axis,
            phi_axis,
            title="MDC waterfall: second",
            color="#d62728",
            scale=first_scale,
            fermi_level=fermi_level,
            half_window=half_window,
        )
        self._plot_mdc_waterfall(
            mdc_delta_axis,
            spectrum_delta,
            energy_axis,
            phi_axis,
            title="MDC waterfall: second - first",
            color="#222222",
            scale=delta_scale,
            fermi_level=fermi_level,
            half_window=half_window,
        )

        self.curve_canvas.draw_idle()

    def _waterfall_scale(self, spectra: list[np.ndarray]) -> float:
        values = [np.asarray(spectrum, dtype=np.float32).reshape(-1) for spectrum in spectra]
        combined = np.concatenate(values) if values else np.array([1.0], dtype=np.float32)
        finite = np.abs(combined[np.isfinite(combined)])
        if finite.size == 0:
            return 1.0
        scale = float(np.nanpercentile(finite, 99))
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.nanmax(finite)) if finite.size else 1.0
        return scale if np.isfinite(scale) and scale > 0 else 1.0

    def _waterfall_indices(self, size: int, max_traces: int = 18) -> np.ndarray:
        if size <= 0:
            return np.array([], dtype=int)
        return np.unique(np.linspace(0, size - 1, min(size, max_traces)).astype(int))

    def _plot_edc_waterfall(
        self,
        axis: matplotlib.axes.Axes,
        spectrum: np.ndarray,
        energy_axis: np.ndarray,
        phi_axis: np.ndarray,
        title: str,
        color: str,
        scale: float,
    ) -> None:
        phi_indices = self._waterfall_indices(spectrum.shape[1])
        offset_step = 0.92
        for offset_index, phi_index in enumerate(phi_indices):
            trace = np.asarray(spectrum[:, phi_index], dtype=np.float32) / scale
            offset = offset_index * offset_step
            axis.plot(energy_axis, trace + offset, color=color, linewidth=0.9, alpha=0.88)

        axis.axvline(0.0, color="#777777", linestyle="--", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("eV")
        axis.set_ylabel("offset by phi")
        if phi_indices.size:
            tick_step = max(1, int(np.ceil(phi_indices.size / 5)))
            ticks = np.arange(0, phi_indices.size, tick_step)
            axis.set_yticks(ticks * offset_step)
            axis.set_yticklabels([f"{float(phi_axis[phi_indices[index]]):.2g}" for index in ticks], fontsize=7)

    def _plot_mdc_waterfall(
        self,
        axis: matplotlib.axes.Axes,
        spectrum: np.ndarray,
        energy_axis: np.ndarray,
        phi_axis: np.ndarray,
        title: str,
        color: str,
        scale: float,
        fermi_level: float,
        half_window: float,
    ) -> None:
        energy_indices = self._waterfall_indices(spectrum.shape[0])
        offset_step = 0.92
        for offset_index, energy_index in enumerate(energy_indices):
            trace = np.asarray(spectrum[energy_index, :], dtype=np.float32) / scale
            offset = offset_index * offset_step
            axis.plot(phi_axis, trace + offset, color=color, linewidth=0.9, alpha=0.88)

        window_indices = set(
            int(index)
            for index in np.flatnonzero(np.abs(energy_axis - fermi_level) <= half_window)
        )
        for offset_index, energy_index in enumerate(energy_indices):
            if int(energy_index) in window_indices:
                offset = offset_index * offset_step
                axis.axhspan(offset - 0.18, offset + 0.18, color="#f2d264", alpha=0.18, linewidth=0)

        center_index = int(np.argmin(np.abs(energy_axis - fermi_level))) if energy_axis.size else 0
        if energy_indices.size and center_index in set(int(index) for index in energy_indices):
            center_position = int(np.where(energy_indices == center_index)[0][0]) * offset_step
            axis.axhline(center_position, color="#777777", linestyle="--", linewidth=0.8)
        axis.set_title(f"{title}\nE window marker: {fermi_level:+.3f} +/- {half_window:.3f} eV")
        axis.set_xlabel("phi")
        axis.set_ylabel("offset by energy")
        if energy_indices.size:
            tick_step = max(1, int(np.ceil(energy_indices.size / 5)))
            ticks = np.arange(0, energy_indices.size, tick_step)
            axis.set_yticks(ticks * offset_step)
            axis.set_yticklabels([f"{float(energy_axis[energy_indices[index]]):+.2g}" for index in ticks], fontsize=7)

    def _curve_display_traces(self, first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        first = np.asarray(first, dtype=np.float32)
        second = np.asarray(second, dtype=np.float32)
        delta = second - first
        scale = float(np.nanmax(np.abs(np.concatenate([first.reshape(-1), second.reshape(-1)]))))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        return first / scale, second / scale, delta / scale, scale

    def _build_curve_rgb_overlay(self, first_map: np.ndarray, second_map: np.ndarray) -> np.ndarray:
        first_norm = self._normalize_curve_map(first_map)
        second_norm = self._normalize_curve_map(second_map)
        rgb = np.zeros(first_norm.shape + (3,), dtype=np.float32)
        rgb[..., 0] = second_norm
        rgb[..., 1] = first_norm
        rgb[..., 2] = np.maximum(first_norm, second_norm)
        return np.clip(rgb, 0.0, 1.0)

    def _normalize_curve_map(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        low = float(np.nanpercentile(finite, 1))
        high = float(np.nanpercentile(finite, 99))
        if high <= low:
            high = float(np.nanmax(finite))
            low = float(np.nanmin(finite))
        if high <= low:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - low) / (high - low), 0.0, 1.0).astype(np.float32)

    def _on_curve_plot_click(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        if not self.curve_loaded_states or event.inaxes not in self.curve_map_axes:
            return
        if event.xdata is None or event.ydata is None:
            return

        x_index = int(round(event.xdata))
        y_index = int(round(event.ydata))
        x_size, y_size = self.curve_total_maps[0].shape
        if not (0 <= x_index < x_size and 0 <= y_index < y_size):
            return

        self.curve_selected_pixel = (x_index, y_index)
        self._refresh_curve_views()

    def _mark_curve_selected_pixel(self, axis: matplotlib.axes.Axes) -> None:
        if self.curve_selected_pixel is None:
            return
        x_index, y_index = self.curve_selected_pixel
        axis.scatter([x_index], [y_index], s=80, facecolors="none", edgecolors="white", linewidths=1.8)
        axis.scatter([x_index], [y_index], s=16, c="black")

    def _update_curve_summary_text(self) -> None:
        if not self.curve_loaded_states or self.curve_selected_pixel is None:
            self._set_text_widget(self.curve_summary_text, "")
            return

        x_index, y_index = self.curve_selected_pixel
        first_state, second_state = self.curve_loaded_states
        energy_axis = np.asarray(first_state.data_array.coords["eV"].values, dtype=np.float32)
        phi_axis = np.asarray(first_state.data_array.coords["phi"].values, dtype=np.float32)
        fermi_level, half_window = self._parse_curve_parameters()
        energy_mask = np.abs(energy_axis - fermi_level) <= half_window
        if not np.any(energy_mask):
            energy_mask[np.argmin(np.abs(energy_axis - fermi_level))] = True

        first_spectrum = np.asarray(first_state.data_array.values[x_index, y_index, :, :], dtype=np.float32)
        second_spectrum = np.asarray(second_state.data_array.values[x_index, y_index, :, :], dtype=np.float32)
        first_edc = first_spectrum.sum(axis=1)
        second_edc = second_spectrum.sum(axis=1)
        first_mdc = first_spectrum[energy_mask, :].sum(axis=0)
        second_mdc = second_spectrum[energy_mask, :].sum(axis=0)
        delta_edc = second_edc - first_edc
        delta_mdc = second_mdc - first_mdc
        gain_e = int(np.nanargmax(delta_edc)) if delta_edc.size else 0
        loss_e = int(np.nanargmin(delta_edc)) if delta_edc.size else 0
        gain_phi = int(np.nanargmax(delta_mdc)) if delta_mdc.size else 0
        loss_phi = int(np.nanargmin(delta_mdc)) if delta_mdc.size else 0

        lines = [
            f"Selected pixel: x={x_index}, y={y_index}",
            f"First file: {Path(first_state.file_path).name}",
            f"Second file: {Path(second_state.file_path).name}",
            f"Display mode: {'EDC/MDC waterfall' if self.curve_mode_var.get() == 'waterfall' else 'point curves'}",
            f"MDC window: |E - {fermi_level:+.3f}| <= {half_window:.3f} eV ({int(np.sum(energy_mask))} energy samples)",
            "",
            f"EDC total intensity: first={float(np.sum(first_edc)):.6g}, second={float(np.sum(second_edc)):.6g}, delta={float(np.sum(delta_edc)):+.6g}",
            f"EDC strongest gain/loss: {float(energy_axis[gain_e]):+.4f} eV / {float(energy_axis[loss_e]):+.4f} eV",
            f"MDC total in window: first={float(np.sum(first_mdc)):.6g}, second={float(np.sum(second_mdc)):.6g}, delta={float(np.sum(delta_mdc)):+.6g}",
            f"MDC strongest gain/loss: phi={float(phi_axis[gain_phi]):+.4f} / phi={float(phi_axis[loss_phi]):+.4f}",
            "",
            "The plotted EDC/MDC curves use a shared display scale per curve family; the black curve is second minus first on that same scale.",
        ]
        if self.curve_mode_var.get() == "waterfall":
            lines.append("Waterfall mode stacks EDC traces across phi and MDC traces across energy for the selected pixel.")
        self._set_text_widget(self.curve_summary_text, "\n".join(lines))

    def _save_curve_plot(self) -> None:
        if not self.curve_loaded_states:
            messagebox.showinfo("No plot", "Run the EDC/MDC comparison before saving a plot.")
            return

        path = filedialog.asksaveasfilename(
            title="Save EDC/MDC plot",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF document", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.curve_figure.savefig(path, dpi=220)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        self.curve_status_var.set(f"Saved EDC/MDC plot to {path}")

    def _add_feature_files(self) -> None:
        selected = list(filedialog.askopenfilenames(title="Choose NetCDF files", filetypes=FILE_TYPES))
        if not selected:
            return

        new_paths = [str(Path(path).expanduser().resolve()) for path in selected]
        merged = self.feature_file_paths + [path for path in new_paths if path not in self.feature_file_paths]
        self._set_feature_files(merged)

    def _copy_analysis_files_to_feature_panel(self) -> None:
        if not self.file_paths:
            messagebox.showinfo("No analysis files", "Add files to the Analysis panel first, or add files here directly.")
            return
        self._set_feature_files(self.file_paths)
        self.top_notebook.select(4)

    def _remove_selected_feature_files(self) -> None:
        selection = list(self.feature_file_listbox.curselection())
        if not selection:
            return
        updated_files = list(self.feature_file_paths)
        for index in reversed(selection):
            del updated_files[index]
        self._set_feature_files(updated_files)

    def _clear_feature_files(self) -> None:
        self._set_feature_files([])

    def _set_feature_files(self, file_paths: list[str]) -> None:
        self.feature_file_paths = list(file_paths)
        if self.feature_first_path not in self.feature_file_paths:
            self.feature_first_path = self.feature_file_paths[0] if self.feature_file_paths else None
        if self.feature_second_path not in self.feature_file_paths or self.feature_second_path == self.feature_first_path:
            self.feature_second_path = self._first_feature_path_excluding(self.feature_first_path)
        self._clear_feature_results()
        self._sync_feature_file_listbox()
        self._update_feature_selector_values()
        self._render_feature_placeholder()

    def _clear_feature_results(self) -> None:
        self.feature_loaded_states = []
        self.feature_total_maps = []
        self.feature_ef_maps = []
        self.feature_features_by_state = []
        self.feature_valid_mask = None
        self.feature_score_map = None
        self.feature_metric_maps = {}
        self.feature_hotspots = []
        self.feature_selected_pixel = None
        self.feature_map_axes = []

    def _first_feature_path_excluding(self, excluded_path: str | None) -> str | None:
        for path in self.feature_file_paths:
            if path != excluded_path:
                return path
        return self.feature_file_paths[0] if self.feature_file_paths else None

    def _sync_feature_file_listbox(self) -> None:
        self.feature_file_listbox.delete(0, tk.END)
        for index, path in enumerate(self.feature_file_paths):
            labels: list[str] = []
            if path == self.feature_first_path:
                labels.append("first")
            if path == self.feature_second_path:
                labels.append("second")
            suffix = f" [{', '.join(labels)}]" if labels else ""
            self.feature_file_listbox.insert(tk.END, f"{index + 1}. {Path(path).name}{suffix}")

    def _feature_label_for_index(self, index: int) -> str:
        return f"{index + 1}. {Path(self.feature_file_paths[index]).name}"

    def _feature_index_from_label(self, label: str) -> int | None:
        for index in range(len(self.feature_file_paths)):
            if label == self._feature_label_for_index(index):
                return index
        return None

    def _update_feature_selector_values(self) -> None:
        values = [self._feature_label_for_index(index) for index in range(len(self.feature_file_paths))]
        self.feature_first_combo["values"] = values
        self.feature_second_combo["values"] = values

        if self.feature_first_path not in self.feature_file_paths:
            self.feature_first_path = self.feature_file_paths[0] if self.feature_file_paths else None
        if self.feature_second_path not in self.feature_file_paths or self.feature_second_path == self.feature_first_path:
            self.feature_second_path = self._first_feature_path_excluding(self.feature_first_path)

        if self.feature_first_path in self.feature_file_paths:
            self.feature_first_var.set(self._feature_label_for_index(self.feature_file_paths.index(self.feature_first_path)))
        else:
            self.feature_first_var.set("")

        if self.feature_second_path in self.feature_file_paths:
            self.feature_second_var.set(self._feature_label_for_index(self.feature_file_paths.index(self.feature_second_path)))
        else:
            self.feature_second_var.set("")

    def _handle_feature_first_selected(self, _event: tk.Event | None = None) -> None:
        index = self._feature_index_from_label(self.feature_first_var.get())
        if index is None:
            return
        self.feature_first_path = self.feature_file_paths[index]
        if self.feature_second_path == self.feature_first_path:
            self.feature_second_path = self._first_feature_path_excluding(self.feature_first_path)
        self._clear_feature_results()
        self._sync_feature_file_listbox()
        self._update_feature_selector_values()
        self._render_feature_placeholder()

    def _handle_feature_second_selected(self, _event: tk.Event | None = None) -> None:
        index = self._feature_index_from_label(self.feature_second_var.get())
        if index is None:
            return
        self.feature_second_path = self.feature_file_paths[index]
        self._clear_feature_results()
        self._sync_feature_file_listbox()
        self._update_feature_selector_values()
        self._render_feature_placeholder()

    def _parse_feature_parameters(self) -> AnalysisParameters:
        try:
            params = AnalysisParameters(
                fermi_level_ev=float(self.feature_parameter_vars["fermi_level_ev"].get()),
                ef_window_ev=float(self.feature_parameter_vars["ef_window_ev"].get()),
                wide_window_ev=float(self.feature_parameter_vars["wide_window_ev"].get()),
            )
        except ValueError as exc:
            raise ValueError(f"Could not parse the feature-search controls: {exc}") from exc
        params.validate()
        return params

    def _feature_top_count(self) -> int:
        try:
            count = int(self.feature_parameter_vars["top_pixels"].get())
        except ValueError as exc:
            raise ValueError(f"Top hotspots must be an integer: {exc}") from exc
        return max(1, min(100, count))

    def _run_feature_search(self) -> None:
        if len(self.feature_file_paths) < 2:
            messagebox.showerror("Missing files", "Please choose at least two NetCDF files.")
            return
        if self.feature_first_path is None or self.feature_second_path is None:
            messagebox.showerror("Missing pair", "Choose the first and second files to compare.")
            return
        if self.feature_first_path == self.feature_second_path:
            messagebox.showerror("Same file", "Choose two different files for feature search.")
            return

        try:
            params = self._parse_feature_parameters()
            top_count = self._feature_top_count()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self.feature_status_var.set("Searching for special features between the selected datasets...")
        self._start_global_progress("Feature Search running...")
        self.root.update_idletasks()

        try:
            loaded_states, alignment_notes = align_loaded_states_for_comparison(
                [load_state(self.feature_first_path), load_state(self.feature_second_path)]
            )

            total_maps: list[np.ndarray] = []
            ef_maps: list[np.ndarray] = []
            features_by_state: list[dict[str, np.ndarray]] = []
            for state in loaded_states:
                total_map, ef_map = total_and_ef_maps(
                    state.data_array,
                    fermi_level=params.fermi_level_ev,
                    ef_window=params.ef_window_ev,
                )
                features, _, _ = extract_pixel_features(
                    state.data_array,
                    fermi_level=params.fermi_level_ev,
                    ef_window=params.ef_window_ev,
                    wide_window=params.wide_window_ev,
                )
                total_maps.append(total_map)
                ef_maps.append(ef_map)
                features_by_state.append(features)

            valid_mask, _, _, _, _ = build_cross_mask_from_maps(
                total_maps,
                threshold_quantile=params.cross_threshold_quantile,
                row_fraction=params.cross_row_fraction,
                col_fraction=params.cross_col_fraction,
                background_quantile=params.cross_background_quantile,
                pad=params.cross_pad,
            )
            if not np.any(valid_mask):
                valid_mask = np.ones_like(total_maps[0], dtype=bool)

            spectral_rms = self._compute_feature_spectral_rms_map(loaded_states[0], loaded_states[1])
            metric_maps = self._build_feature_metric_maps(features_by_state, total_maps, spectral_rms)
            score_map = self._build_feature_score_map(metric_maps, valid_mask)
            hotspots = self._select_feature_hotspots(score_map, metric_maps, valid_mask, top_count)

            self.feature_loaded_states = loaded_states
            self.feature_total_maps = total_maps
            self.feature_ef_maps = ef_maps
            self.feature_features_by_state = features_by_state
            self.feature_valid_mask = valid_mask
            self.feature_metric_maps = metric_maps
            self.feature_score_map = score_map
            self.feature_hotspots = hotspots
            self.feature_selected_pixel = self._default_feature_pixel()
        except Exception as exc:
            self._clear_feature_results()
            self.feature_status_var.set("Feature search failed.")
            self._finish_global_progress("Feature Search failed.", success=False)
            messagebox.showerror("Feature search failed", str(exc))
            self._render_feature_placeholder()
            return

        self._sync_feature_file_listbox()
        self._update_feature_selector_values()
        self._refresh_feature_views()
        alignment_suffix = f" {alignment_notes[0]}" if alignment_notes else ""
        self.feature_status_var.set(
            f"Found {len(self.feature_hotspots)} candidate feature hotspot(s) between {Path(self.feature_first_path).name} and {Path(self.feature_second_path).name}."
            f"{alignment_suffix}"
        )
        self._finish_global_progress("Feature Search complete.")

    def _compute_feature_spectral_rms_map(self, first_state: LoadedState, second_state: LoadedState) -> np.ndarray:
        first_data = np.asarray(first_state.data_array.values, dtype=np.float32)
        second_data = np.asarray(second_state.data_array.values, dtype=np.float32)
        x_size, y_size, e_size, phi_size = first_data.shape
        n_pixels = x_size * y_size
        first_flat = first_data.reshape(n_pixels, e_size * phi_size)
        second_flat = second_data.reshape(n_pixels, e_size * phi_size)
        rms = np.zeros(n_pixels, dtype=np.float32)

        chunk_size = 128
        for start in range(0, n_pixels, chunk_size):
            end = min(n_pixels, start + chunk_size)
            first_chunk = np.asarray(first_flat[start:end], dtype=np.float32)
            second_chunk = np.asarray(second_flat[start:end], dtype=np.float32)
            first_total = np.sum(first_chunk, axis=1, keepdims=True) + 1e-8
            second_total = np.sum(second_chunk, axis=1, keepdims=True) + 1e-8
            diff = second_chunk / second_total - first_chunk / first_total
            rms[start:end] = np.sqrt(np.mean(diff * diff, axis=1)).astype(np.float32)

        return rms.reshape(x_size, y_size)

    def _build_feature_metric_maps(
        self,
        features_by_state: list[dict[str, np.ndarray]],
        total_maps: list[np.ndarray],
        spectral_rms: np.ndarray,
    ) -> dict[str, np.ndarray]:
        first_features, second_features = features_by_state
        return {
            "spectral_rms": np.asarray(spectral_rms, dtype=np.float32),
            "delta_ef_fraction": (
                np.asarray(second_features["ef_fraction"], dtype=np.float32)
                - np.asarray(first_features["ef_fraction"], dtype=np.float32)
            ),
            "delta_e_centroid": (
                np.asarray(second_features["e_centroid"], dtype=np.float32)
                - np.asarray(first_features["e_centroid"], dtype=np.float32)
            ),
            "delta_total_intensity": (
                np.asarray(total_maps[1], dtype=np.float32)
                - np.asarray(total_maps[0], dtype=np.float32)
            ),
            "delta_spectral_entropy": (
                np.asarray(second_features["spectral_entropy"], dtype=np.float32)
                - np.asarray(first_features["spectral_entropy"], dtype=np.float32)
            ),
        }

    def _build_feature_score_map(self, metric_maps: dict[str, np.ndarray], valid_mask: np.ndarray) -> np.ndarray:
        components = {
            "spectral_rms": self._feature_scaled_abs(metric_maps["spectral_rms"], valid_mask),
            "delta_ef_fraction": self._feature_scaled_abs(metric_maps["delta_ef_fraction"], valid_mask),
            "delta_e_centroid": self._feature_scaled_abs(metric_maps["delta_e_centroid"], valid_mask),
            "delta_total_intensity": self._feature_scaled_abs(metric_maps["delta_total_intensity"], valid_mask),
            "delta_spectral_entropy": self._feature_scaled_abs(metric_maps["delta_spectral_entropy"], valid_mask),
        }
        score = (
            0.30 * components["spectral_rms"]
            + 0.25 * components["delta_ef_fraction"]
            + 0.20 * components["delta_e_centroid"]
            + 0.15 * components["delta_total_intensity"]
            + 0.10 * components["delta_spectral_entropy"]
        ).astype(np.float32)
        score[~valid_mask] = np.nan
        metric_maps["score"] = score
        metric_maps["scaled_spectral_rms"] = components["spectral_rms"]
        metric_maps["scaled_delta_ef_fraction"] = components["delta_ef_fraction"]
        metric_maps["scaled_delta_e_centroid"] = components["delta_e_centroid"]
        metric_maps["scaled_delta_total_intensity"] = components["delta_total_intensity"]
        metric_maps["scaled_delta_spectral_entropy"] = components["delta_spectral_entropy"]
        return score

    def _feature_scaled_abs(self, values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        valid_values = np.abs(arr[valid_mask & np.isfinite(arr)])
        if valid_values.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        scale = float(np.nanpercentile(valid_values, 97))
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.nanmax(valid_values)) if valid_values.size else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        scaled = np.clip(np.abs(arr) / scale, 0.0, 1.0).astype(np.float32)
        scaled[~valid_mask] = 0.0
        return scaled

    def _select_feature_hotspots(
        self,
        score_map: np.ndarray,
        metric_maps: dict[str, np.ndarray],
        valid_mask: np.ndarray,
        top_count: int,
    ) -> list[dict[str, object]]:
        score = np.asarray(score_map, dtype=np.float32).copy()
        score[~valid_mask] = np.nan
        hotspots: list[dict[str, object]] = []
        exclusion_radius = max(1, int(round(min(score.shape) * 0.035)))

        for _ in range(top_count):
            if not np.any(np.isfinite(score)):
                break
            flat_index = int(np.nanargmax(score))
            x_index, y_index = divmod(flat_index, score.shape[1])
            value = float(score[x_index, y_index])
            if not np.isfinite(value):
                break
            tags = self._feature_tags_at_pixel(metric_maps, x_index, y_index)
            hotspots.append(
                {
                    "x": x_index,
                    "y": y_index,
                    "score": value,
                    "tags": tags,
                    "spectral_rms": float(metric_maps["spectral_rms"][x_index, y_index]),
                    "delta_ef_fraction": float(metric_maps["delta_ef_fraction"][x_index, y_index]),
                    "delta_e_centroid": float(metric_maps["delta_e_centroid"][x_index, y_index]),
                    "delta_total_intensity": float(metric_maps["delta_total_intensity"][x_index, y_index]),
                    "delta_spectral_entropy": float(metric_maps["delta_spectral_entropy"][x_index, y_index]),
                }
            )
            x0 = max(0, x_index - exclusion_radius)
            x1 = min(score.shape[0], x_index + exclusion_radius + 1)
            y0 = max(0, y_index - exclusion_radius)
            y1 = min(score.shape[1], y_index + exclusion_radius + 1)
            score[x0:x1, y0:y1] = np.nan

        return hotspots

    def _feature_tags_at_pixel(self, metric_maps: dict[str, np.ndarray], x_index: int, y_index: int) -> str:
        candidates = [
            ("spectral-shape change", metric_maps.get("scaled_spectral_rms")),
            ("near-EF weight change", metric_maps.get("scaled_delta_ef_fraction")),
            ("energy shift", metric_maps.get("scaled_delta_e_centroid")),
            ("intensity change", metric_maps.get("scaled_delta_total_intensity")),
            ("entropy change", metric_maps.get("scaled_delta_spectral_entropy")),
        ]
        tags = [label for label, values in candidates if values is not None and float(values[x_index, y_index]) >= 0.55]
        return ", ".join(tags) if tags else "mixed weak feature"

    def _default_feature_pixel(self) -> tuple[int, int]:
        if self.feature_score_map is None or not np.any(np.isfinite(self.feature_score_map)):
            return (0, 0)
        flat_index = int(np.nanargmax(self.feature_score_map))
        return divmod(flat_index, self.feature_score_map.shape[1])

    def _feature_map_key(self) -> str:
        return self.FEATURE_MAP_OPTIONS.get(self.feature_map_var.get(), "score")

    def _refresh_feature_views(self) -> None:
        if self.feature_score_map is None or not self.feature_loaded_states:
            self._render_feature_placeholder()
            return
        if self.feature_selected_pixel is None:
            self.feature_selected_pixel = self._default_feature_pixel()
        self._refresh_feature_plot()
        self._update_feature_summary_text()

    def _render_feature_placeholder(self) -> None:
        if not hasattr(self, "feature_figure"):
            return

        self.feature_figure.clear()
        axis = self.feature_figure.add_subplot(111)
        if len(self.feature_file_paths) >= 2:
            message = "Ready to search.\nChoose two files, then click Search Special Features."
        else:
            message = "Add at least two NetCDF files to search for special feature changes."
        axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=13)
        axis.set_axis_off()
        self.feature_canvas.draw_idle()
        self.feature_map_axes = []
        self._set_text_widget(self.feature_summary_text, "")
        if len(self.feature_file_paths) < 2:
            self.feature_status_var.set("Add at least two NetCDF files, choose a pair, then search for special features.")

    def _refresh_feature_plot(self) -> None:
        assert self.feature_selected_pixel is not None
        assert self.feature_score_map is not None
        first_state, second_state = self.feature_loaded_states
        first_name = Path(first_state.file_path).name
        second_name = Path(second_state.file_path).name
        x_index, y_index = self.feature_selected_pixel
        map_key = self._feature_map_key()
        feature_map = np.asarray(self.feature_metric_maps[map_key], dtype=np.float32)

        first_total = np.asarray(self.feature_total_maps[0], dtype=np.float32)
        second_total = np.asarray(self.feature_total_maps[1], dtype=np.float32)
        energy_axis = np.asarray(first_state.data_array.coords["eV"].values, dtype=np.float32)
        phi_axis = np.asarray(first_state.data_array.coords["phi"].values, dtype=np.float32)
        params = self._parse_feature_parameters()
        energy_mask = np.abs(energy_axis - params.fermi_level_ev) <= params.ef_window_ev
        if not np.any(energy_mask):
            energy_mask[np.argmin(np.abs(energy_axis - params.fermi_level_ev))] = True

        first_spectrum = np.asarray(first_state.data_array.values[x_index, y_index, :, :], dtype=np.float32)
        second_spectrum = np.asarray(second_state.data_array.values[x_index, y_index, :, :], dtype=np.float32)
        spectrum_delta = second_spectrum - first_spectrum
        first_edc = first_spectrum.sum(axis=1)
        second_edc = second_spectrum.sum(axis=1)
        first_mdc = first_spectrum[energy_mask, :].sum(axis=0)
        second_mdc = second_spectrum[energy_mask, :].sum(axis=0)
        first_edc_plot, second_edc_plot, delta_edc_plot, _ = self._curve_display_traces(first_edc, second_edc)
        first_mdc_plot, second_mdc_plot, delta_mdc_plot, _ = self._curve_display_traces(first_mdc, second_mdc)

        self.feature_figure.clear()
        axes = self.feature_figure.subplots(2, 3)
        first_axis, second_axis, feature_axis = axes[0]
        spectrum_axis, edc_axis, mdc_axis = axes[1]
        self.feature_map_axes = [first_axis, second_axis, feature_axis]

        first_image = first_axis.imshow(first_total.T, origin="lower", cmap="viridis", aspect="auto")
        first_axis.set_title(f"First: {first_name}\ntotal intensity")
        self.feature_figure.colorbar(first_image, ax=first_axis, fraction=0.046, pad=0.04)

        second_image = second_axis.imshow(second_total.T, origin="lower", cmap="viridis", aspect="auto")
        second_axis.set_title(f"Second: {second_name}\ntotal intensity")
        self.feature_figure.colorbar(second_image, ax=second_axis, fraction=0.046, pad=0.04)

        if map_key.startswith("delta_"):
            vmax = self._symmetric_change_limit(feature_map)
            feature_image = feature_axis.imshow(
                feature_map.T,
                origin="lower",
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                aspect="auto",
            )
        else:
            feature_image = feature_axis.imshow(feature_map.T, origin="lower", cmap="inferno", aspect="auto")
        feature_axis.set_title(self.feature_map_var.get())
        self._plot_feature_hotspot_markers(feature_axis)
        self.feature_figure.colorbar(feature_image, ax=feature_axis, fraction=0.046, pad=0.04)

        for axis in self.feature_map_axes:
            axis.set_xlabel("x index")
            axis.set_ylabel("y index")
            self._mark_feature_selected_pixel(axis)

        vmax = self._symmetric_change_limit(spectrum_delta)
        spectrum_image = spectrum_axis.imshow(
            spectrum_delta,
            origin="lower",
            aspect="auto",
            extent=[float(phi_axis[0]), float(phi_axis[-1]), float(energy_axis[0]), float(energy_axis[-1])],
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        spectrum_axis.axhline(params.fermi_level_ev, color="#222222", linestyle="--", linewidth=0.9)
        spectrum_axis.set_title(f"Local spectrum difference\nx={x_index}, y={y_index}")
        spectrum_axis.set_xlabel("phi")
        spectrum_axis.set_ylabel("eV")
        self.feature_figure.colorbar(spectrum_image, ax=spectrum_axis, fraction=0.046, pad=0.04)

        edc_axis.plot(energy_axis, first_edc_plot, color="#1f77b4", linewidth=1.5, label="first")
        edc_axis.plot(energy_axis, second_edc_plot, color="#d62728", linewidth=1.5, label="second")
        edc_axis.plot(energy_axis, delta_edc_plot, color="#222222", linewidth=1.1, label="second - first")
        edc_axis.axhline(0.0, color="#555555", linewidth=0.8)
        edc_axis.axvline(params.fermi_level_ev, color="#777777", linestyle="--", linewidth=0.9)
        edc_axis.set_title("EDC at selected feature")
        edc_axis.set_xlabel("eV")
        edc_axis.set_ylabel("scaled intensity")
        edc_axis.legend(loc="best", fontsize=8)

        mdc_axis.plot(phi_axis, first_mdc_plot, color="#1f77b4", linewidth=1.5, label="first")
        mdc_axis.plot(phi_axis, second_mdc_plot, color="#d62728", linewidth=1.5, label="second")
        mdc_axis.plot(phi_axis, delta_mdc_plot, color="#222222", linewidth=1.1, label="second - first")
        mdc_axis.axhline(0.0, color="#555555", linewidth=0.8)
        mdc_axis.set_title(f"MDC at E={params.fermi_level_ev:+.3f} +/- {params.ef_window_ev:.3f} eV")
        mdc_axis.set_xlabel("phi")
        mdc_axis.set_ylabel("scaled intensity")
        mdc_axis.legend(loc="best", fontsize=8)

        self.feature_canvas.draw_idle()

    def _plot_feature_hotspot_markers(self, axis: matplotlib.axes.Axes) -> None:
        for rank, hotspot in enumerate(self.feature_hotspots[:10], start=1):
            x_index = int(hotspot["x"])
            y_index = int(hotspot["y"])
            axis.scatter([x_index], [y_index], s=44, facecolors="none", edgecolors="white", linewidths=1.0)
            axis.text(x_index, y_index, str(rank), color="white", fontsize=7, ha="center", va="center")

    def _on_feature_plot_click(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        if self.feature_score_map is None or event.inaxes not in self.feature_map_axes:
            return
        if event.xdata is None or event.ydata is None:
            return

        x_index = int(round(event.xdata))
        y_index = int(round(event.ydata))
        x_size, y_size = self.feature_score_map.shape
        if not (0 <= x_index < x_size and 0 <= y_index < y_size):
            return

        self.feature_selected_pixel = (x_index, y_index)
        self._refresh_feature_views()

    def _mark_feature_selected_pixel(self, axis: matplotlib.axes.Axes) -> None:
        if self.feature_selected_pixel is None:
            return
        x_index, y_index = self.feature_selected_pixel
        axis.scatter([x_index], [y_index], s=92, facecolors="none", edgecolors="white", linewidths=1.8)
        axis.scatter([x_index], [y_index], s=18, c="black")

    def _update_feature_summary_text(self) -> None:
        if self.feature_score_map is None:
            self._set_text_widget(self.feature_summary_text, "")
            return

        first_name = Path(self.feature_loaded_states[0].file_path).name
        second_name = Path(self.feature_loaded_states[1].file_path).name
        lines = [
            f"Feature search: {second_name} - {first_name}",
            "Score weights: spectral-shape 30%, near-EF 25%, energy centroid 20%, total intensity 15%, entropy 10%.",
            "",
            "Top candidate hotspots:",
        ]
        for rank, hotspot in enumerate(self.feature_hotspots, start=1):
            lines.append(
                f"  {rank:02d}. x={hotspot['x']}, y={hotspot['y']}, score={float(hotspot['score']):.3f}, "
                f"{hotspot['tags']}; dEF={float(hotspot['delta_ef_fraction']):+.5f}, "
                f"dEcentroid={float(hotspot['delta_e_centroid']):+.5f} eV"
            )

        if self.feature_selected_pixel is not None:
            x_index, y_index = self.feature_selected_pixel
            lines.extend(
                [
                    "",
                    f"Selected pixel: x={x_index}, y={y_index}",
                    f"Score: {float(self.feature_score_map[x_index, y_index]):.3f}",
                    f"Tags: {self._feature_tags_at_pixel(self.feature_metric_maps, x_index, y_index)}",
                    f"Spectral RMS change: {float(self.feature_metric_maps['spectral_rms'][x_index, y_index]):.6g}",
                    f"Near-EF fraction change: {float(self.feature_metric_maps['delta_ef_fraction'][x_index, y_index]):+.6g}",
                    f"Energy centroid shift: {float(self.feature_metric_maps['delta_e_centroid'][x_index, y_index]):+.6g} eV",
                    f"Total intensity change: {float(self.feature_metric_maps['delta_total_intensity'][x_index, y_index]):+.6g}",
                    f"Entropy change: {float(self.feature_metric_maps['delta_spectral_entropy'][x_index, y_index]):+.6g}",
                ]
            )

        self._set_text_widget(self.feature_summary_text, "\n".join(lines))

    def _run_ai_data_analysis_placeholder(self) -> None:
        messagebox.showinfo(
            "AI data analysis",
            "Placeholder only. Hook this button to your Ollama investigation routine when you are ready.",
        )
        self.feature_status_var.set("AI data analysis placeholder clicked. No external model was called.")

    def _save_feature_plot(self) -> None:
        if self.feature_score_map is None:
            messagebox.showinfo("No plot", "Run the feature search before saving a plot.")
            return

        path = filedialog.asksaveasfilename(
            title="Save feature search plot",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF document", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.feature_figure.savefig(path, dpi=220)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        self.feature_status_var.set(f"Saved feature search plot to {path}")

    def _set_classifier_file(self, file_path: str | None) -> None:
        self.classifier_file_path = str(file_path) if file_path else None
        self.classifier_file_var.set(self.classifier_file_path or "")
        self._clear_classifier_results()
        self._render_classifier_placeholder()

    def _choose_classifier_file(self) -> None:
        selected = filedialog.askopenfilename(title="Choose NetCDF file", filetypes=FILE_TYPES)
        if not selected:
            return
        self._set_classifier_file(selected)

    def _use_analysis_file_for_classifier(self) -> None:
        if not self.file_paths:
            messagebox.showinfo("No analysis files", "Add a file in the Analysis panel first, or choose a classifier file directly.")
            return
        self._set_classifier_file(self.file_paths[0])

    def _clear_classifier_results(self) -> None:
        self.classifier_result = None
        self.classifier_selected_pixel = None
        self.classifier_map_axes = []

    def _parse_classifier_parameters(self) -> StateClassifierParameters:
        try:
            params = StateClassifierParameters(
                fermi_level_ev=float(self.classifier_parameter_vars["fermi_level_ev"].get()),
                ef_min_ev=float(self.classifier_parameter_vars["ef_min_ev"].get()),
                ef_max_ev=float(self.classifier_parameter_vars["ef_max_ev"].get()),
                lhb_center_ev=float(self.classifier_parameter_vars["lhb_center_ev"].get()),
                lhb_halfwidth_ev=float(self.classifier_parameter_vars["lhb_halfwidth_ev"].get()),
                leading_edge_min_ev=float(self.classifier_parameter_vars["leading_edge_min_ev"].get()),
                leading_edge_max_ev=float(self.classifier_parameter_vars["leading_edge_max_ev"].get()),
                p3_center_ev=float(self.classifier_parameter_vars["p3_center_ev"].get()),
                p3_halfwidth_ev=float(self.classifier_parameter_vars["p3_halfwidth_ev"].get()),
                smooth_sigma=float(self.classifier_parameter_vars["smooth_sigma"].get()),
                low_quantile=float(self.classifier_parameter_vars["low_quantile"].get()),
                high_quantile=float(self.classifier_parameter_vars["high_quantile"].get()),
                broad_quantile=float(self.classifier_parameter_vars["broad_quantile"].get()),
                orientation_quantile=float(self.classifier_parameter_vars["orientation_quantile"].get()),
                low_signal_quantile=float(self.classifier_parameter_vars["low_signal_quantile"].get()),
                lhb_min_quantile=float(self.classifier_parameter_vars["lhb_min_quantile"].get()),
            )
        except ValueError as exc:
            raise ValueError(f"Could not parse the clustering controls: {exc}") from exc
        params.validate()
        return params

    def _run_state_classifier(self) -> None:
        if self.classifier_file_path is None:
            messagebox.showerror("Missing file", "Please choose one NetCDF file for clustering.")
            return

        try:
            params = self._parse_classifier_parameters()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self.classifier_status_var.set("Computing eight spectral features and rule-based clustering labels...")
        self._start_global_progress("Clustering running...")
        self.root.update_idletasks()

        try:
            self.classifier_result = run_state_classification(self.classifier_file_path, params)
            self.classifier_selected_pixel = self._default_classifier_pixel()
        except Exception as exc:
            self._clear_classifier_results()
            self.classifier_status_var.set("Clustering failed.")
            self._finish_global_progress("Clustering failed.", success=False)
            messagebox.showerror("Clustering failed", str(exc))
            self._render_classifier_placeholder()
            return

        self._refresh_classifier_views()
        self.classifier_status_var.set(
            f"Classified {Path(self.classifier_file_path).name} as {self.classifier_result.shape[0]} x {self.classifier_result.shape[1]} pixels."
        )
        self._finish_global_progress("Clustering complete.")

    def _reclassify_state_classifier(self) -> None:
        if self.classifier_result is None:
            messagebox.showinfo("No clustering result", "Compute the clustering features before updating thresholds.")
            return
        try:
            params = self._parse_classifier_parameters()
            current = self.classifier_result
            self.classifier_result = classify_state_feature_maps(
                current.state,
                params,
                current.feature_maps,
                normalized_maps=current.normalized_maps,
                orientation_feature_name=current.orientation_feature_name,
                notes=current.notes,
            )
        except Exception as exc:
            messagebox.showerror("Threshold update failed", str(exc))
            return

        self._refresh_classifier_views()
        self.classifier_status_var.set("Updated state labels using the current threshold quantiles.")

    def _classifier_map_key(self) -> str:
        return self.STATE_CLASSIFIER_MAP_OPTIONS.get(self.classifier_map_var.get(), "state_code")

    def _default_classifier_pixel(self) -> tuple[int, int]:
        if self.classifier_result is None:
            return (0, 0)
        score = np.asarray(self.classifier_result.feature_maps["I_rat"], dtype=np.float32).copy()
        score[~self.classifier_result.valid_mask] = np.nan
        if not np.any(np.isfinite(score)):
            return (0, 0)
        flat_index = int(np.nanargmax(score))
        return divmod(flat_index, score.shape[1])

    def _refresh_classifier_views(self) -> None:
        if self.classifier_result is None:
            self._render_classifier_placeholder()
            return
        if self.classifier_selected_pixel is None:
            self.classifier_selected_pixel = self._default_classifier_pixel()
        self._refresh_classifier_plot()
        self._update_classifier_summary_text()

    def _render_classifier_placeholder(self) -> None:
        if not hasattr(self, "classifier_figure"):
            return

        self.classifier_figure.clear()
        axis = self.classifier_figure.add_subplot(111)
        message = (
            "Ready to cluster.\nChoose one NetCDF file, tune the windows, then click Compute and Classify."
            if self.classifier_file_path
            else "Choose one NetCDF file to compute eight per-pixel spectral features."
        )
        axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=13)
        axis.set_axis_off()
        self.classifier_canvas.draw_idle()
        self.classifier_map_axes = []
        if hasattr(self, "classifier_summary_text"):
            self._set_text_widget(self.classifier_summary_text, "")
        if self.classifier_file_path is None:
            self.classifier_status_var.set("Choose one NetCDF file, set feature windows, then compute rule-based state labels.")

    def _classifier_display_map(self, key: str) -> np.ndarray:
        assert self.classifier_result is not None
        if key == "state_code":
            return np.asarray(self.classifier_result.code_map, dtype=np.float32)
        if key in self.classifier_result.feature_maps:
            return np.asarray(self.classifier_result.feature_maps[key], dtype=np.float32)
        return np.asarray(self.classifier_result.normalized_maps[key], dtype=np.float32)

    def _classifier_map_title(self, key: str) -> str:
        for label, value in self.STATE_CLASSIFIER_MAP_OPTIONS.items():
            if value == key:
                return label
        return key

    def _refresh_classifier_plot(self) -> None:
        assert self.classifier_result is not None
        assert self.classifier_selected_pixel is not None
        result = self.classifier_result
        x_index, y_index = self.classifier_selected_pixel
        map_key = self._classifier_map_key()
        selected_map_key = "I_rat" if map_key == "state_code" else map_key
        selected_map = self._classifier_display_map(selected_map_key)

        state = result.state
        data = np.asarray(state.data_array.values, dtype=np.float32)
        energy_axis = np.asarray(state.data_array.coords["eV"].values, dtype=np.float32)
        phi_axis = np.asarray(state.data_array.coords["phi"].values, dtype=np.float32)
        spectrum = data[x_index, y_index, :, :]
        edc = np.sum(spectrum, axis=1)
        edc_scale = float(np.nanmax(np.abs(edc))) if np.any(np.isfinite(edc)) else 1.0
        if not np.isfinite(edc_scale) or edc_scale <= 0:
            edc_scale = 1.0
        params = result.parameters

        self.classifier_figure.clear()
        axes = self.classifier_figure.subplots(2, 3)
        class_axis, feature_axis, total_axis = axes[0]
        spectrum_axis, edc_axis, norms_axis = axes[1]
        self.classifier_map_axes = [class_axis, feature_axis, total_axis]

        colors = [STATE_CLASSIFICATION_COLORS[label] for label in STATE_CLASSIFICATION_LABELS]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(np.arange(-0.5, len(STATE_CLASSIFICATION_LABELS) + 0.5, 1.0), cmap.N)
        class_image = class_axis.imshow(result.code_map.T, origin="lower", cmap=cmap, norm=norm, aspect="auto")
        class_axis.set_title("Classified state")
        cbar = self.classifier_figure.colorbar(class_image, ax=class_axis, fraction=0.046, pad=0.04)
        cbar.set_ticks(np.arange(len(STATE_CLASSIFICATION_LABELS)))
        cbar.ax.set_yticklabels(STATE_CLASSIFICATION_LABELS, fontsize=7)

        vmin, vmax = self._classifier_feature_limits(selected_map)
        feature_image = feature_axis.imshow(selected_map.T, origin="lower", cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        feature_axis.set_title(self._classifier_map_title(selected_map_key))
        self.classifier_figure.colorbar(feature_image, ax=feature_axis, fraction=0.046, pad=0.04)

        total_image = total_axis.imshow(result.feature_maps["T"].T, origin="lower", cmap="inferno", aspect="auto")
        total_axis.set_title("Total intensity T")
        self.classifier_figure.colorbar(total_image, ax=total_axis, fraction=0.046, pad=0.04)

        for axis in self.classifier_map_axes:
            axis.set_xlabel("x index")
            axis.set_ylabel("y index")
            self._mark_classifier_selected_pixel(axis)

        spectrum_image = spectrum_axis.imshow(
            spectrum,
            origin="lower",
            aspect="auto",
            extent=[float(phi_axis[0]), float(phi_axis[-1]), float(energy_axis[0]), float(energy_axis[-1])],
            cmap="viridis",
        )
        spectrum_axis.axhline(params.fermi_level_ev, color="white", linestyle="--", linewidth=0.9)
        spectrum_axis.axhline(float(result.feature_maps["E_LE"][x_index, y_index]), color="#ffbf00", linestyle=":", linewidth=1.1)
        spectrum_axis.set_title(f"Local spectrum\nx={x_index}, y={y_index}")
        spectrum_axis.set_xlabel("phi")
        spectrum_axis.set_ylabel("eV")
        self.classifier_figure.colorbar(spectrum_image, ax=spectrum_axis, fraction=0.046, pad=0.04)

        edc_axis.plot(energy_axis, edc / edc_scale, color="#1f77b4", linewidth=1.5)
        edc_axis.axvline(params.fermi_level_ev, color="#555555", linestyle="--", linewidth=0.9, label="EF")
        edc_axis.axvline(float(result.feature_maps["E_LHB"][x_index, y_index]), color="#d62728", linewidth=1.1, label="E_LHB")
        edc_axis.axvline(float(result.feature_maps["E_LE"][x_index, y_index]), color="#ffbf00", linewidth=1.1, label="E_LE")
        edc_axis.axvspan(
            params.fermi_level_ev + params.ef_min_ev,
            params.fermi_level_ev + params.ef_max_ev,
            color="#6baed6",
            alpha=0.18,
            linewidth=0,
        )
        edc_axis.axvspan(
            params.lhb_center_ev - params.lhb_halfwidth_ev,
            params.lhb_center_ev + params.lhb_halfwidth_ev,
            color="#fb6a4a",
            alpha=0.13,
            linewidth=0,
        )
        edc_axis.set_title("Angle-integrated EDC")
        edc_axis.set_xlabel("eV")
        edc_axis.set_ylabel("scaled intensity")
        edc_axis.legend(loc="best", fontsize=8)

        norm_keys = [
            ("I_rat", "Irat_norm"),
            ("W_EF", "WEF_norm"),
            ("LHB shift", "LHB_shift_norm"),
            ("LE close", "LE_closeness_norm"),
            ("Gamma", "Gamma_norm"),
            ("Orient", "Orient_shift_norm"),
        ]
        values = [
            float(result.normalized_maps[key][x_index, y_index])
            if np.isfinite(result.normalized_maps[key][x_index, y_index])
            else 0.0
            for _label, key in norm_keys
        ]
        norms_axis.bar(np.arange(len(norm_keys)), values, color=["#d62728", "#1f77b4", "#9467bd", "#2ca02c", "#7f7f7f", "#17becf"])
        norms_axis.set_ylim(0.0, 1.05)
        norms_axis.set_xticks(np.arange(len(norm_keys)))
        norms_axis.set_xticklabels([label for label, _key in norm_keys], rotation=25, ha="right", fontsize=8)
        norms_axis.set_ylabel("robust normalized value")
        norms_axis.set_title("Classifier inputs at selected pixel")

        self.classifier_canvas.draw_idle()

    def _classifier_feature_limits(self, values: np.ndarray) -> tuple[float | None, float | None]:
        finite = np.asarray(values, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return None, None
        low = float(np.nanpercentile(finite, 1))
        high = float(np.nanpercentile(finite, 99))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return None, None
        return low, high

    def _on_classifier_plot_click(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        if self.classifier_result is None or event.inaxes not in self.classifier_map_axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        x_index = int(round(event.xdata))
        y_index = int(round(event.ydata))
        x_size, y_size = self.classifier_result.shape
        if not (0 <= x_index < x_size and 0 <= y_index < y_size):
            return
        self.classifier_selected_pixel = (x_index, y_index)
        self._refresh_classifier_views()

    def _mark_classifier_selected_pixel(self, axis: matplotlib.axes.Axes) -> None:
        if self.classifier_selected_pixel is None:
            return
        x_index, y_index = self.classifier_selected_pixel
        axis.scatter([x_index], [y_index], s=92, facecolors="none", edgecolors="white", linewidths=1.8)
        axis.scatter([x_index], [y_index], s=18, c="black")

    def _update_classifier_summary_text(self) -> None:
        result = self.classifier_result
        if result is None:
            self._set_text_widget(self.classifier_summary_text, "")
            return

        lines = [
            f"File: {Path(result.file_path).name}",
            f"Orientation marker: {result.orientation_feature_name}",
            "",
            "State counts:",
        ]
        total_pixels = max(1, result.shape[0] * result.shape[1])
        for label in STATE_CLASSIFICATION_LABELS:
            count = result.counts.get(label, 0)
            lines.append(f"  {label}: {count} ({count / total_pixels:.1%})")

        lines.extend(
            [
                "",
                "Thresholds:",
                f"  I_rat low/high: {result.threshold_values['low_Irat_threshold']:.5g} / {result.threshold_values['high_Irat_threshold']:.5g}",
                f"  W_EF low/high: {result.threshold_values['low_WEF_threshold']:.5g} / {result.threshold_values['high_WEF_threshold']:.5g}",
                f"  E_LE far/close: {result.threshold_values['far_LE_threshold']:.5g} / {result.threshold_values['close_LE_threshold']:.5g} eV",
                f"  LHB ref/close shift: {result.threshold_values['LHB_reference_ev']:.5g} / {result.threshold_values['LHB_close_shift_threshold']:.5g} eV",
                f"  Gamma broad: {result.threshold_values['broad_Gamma_threshold']:.5g}",
                f"  Orientation shift: {result.threshold_values['large_orientation_shift_threshold']:.5g}",
            ]
        )

        if self.classifier_selected_pixel is not None:
            x_index, y_index = self.classifier_selected_pixel
            lines.extend(["", f"Selected pixel: x={x_index}, y={y_index}", f"State: {result.label_map[x_index, y_index]}"])
            for name in STATE_CLASSIFICATION_FEATURE_NAMES:
                lines.append(f"  {name}: {float(result.feature_maps[name][x_index, y_index]):.6g}")
            lines.extend(
                [
                    f"  Irat_norm: {float(result.normalized_maps['Irat_norm'][x_index, y_index]):.3f}",
                    f"  WEF_norm: {float(result.normalized_maps['WEF_norm'][x_index, y_index]):.3f}",
                    f"  LE_closeness_norm: {float(result.normalized_maps['LE_closeness_norm'][x_index, y_index]):.3f}",
                    f"  Gamma_norm: {float(result.normalized_maps['Gamma_norm'][x_index, y_index]):.3f}",
                    f"  Orient_shift_norm: {float(result.normalized_maps['Orient_shift_norm'][x_index, y_index]):.3f}",
                ]
            )

        if result.notes:
            lines.extend(["", "Notes:"])
            lines.extend(f"  {note}" for note in result.notes)

        self._set_text_widget(self.classifier_summary_text, "\n".join(lines))

    def _save_classifier_results(self) -> None:
        if self.classifier_result is None:
            messagebox.showinfo("No clustering result", "Run clustering before saving results.")
            return
        directory = filedialog.askdirectory(title="Choose output folder for clustering results")
        if not directory:
            return
        try:
            paths = export_state_classification(self.classifier_result, directory)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.classifier_status_var.set(f"Saved clustering feature table to {paths['table']}")

    def _save_classifier_plot(self) -> None:
        if self.classifier_result is None:
            messagebox.showinfo("No clustering plot", "Run clustering before saving a plot.")
            return
        path = filedialog.asksaveasfilename(
            title="Save clustering plot",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF document", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.classifier_figure.savefig(path, dpi=220)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.classifier_status_var.set(f"Saved clustering plot to {path}")

    def _add_switching_files(self) -> None:
        selected = list(filedialog.askopenfilenames(title="Choose NetCDF files", filetypes=FILE_TYPES))
        if not selected:
            return

        new_paths = [str(Path(path).expanduser().resolve()) for path in selected]
        merged = self.switching_file_paths + [path for path in new_paths if path not in self.switching_file_paths]
        self._set_switching_files(merged)

    def _copy_analysis_files_to_switching_panel(self) -> None:
        if not self.file_paths:
            messagebox.showinfo("No analysis files", "Add files to the Analysis panel first, or add files here directly.")
            return
        self._set_switching_files(self.file_paths)
        self.top_notebook.select(6)

    def _remove_selected_switching_files(self) -> None:
        selection = list(self.switching_file_listbox.curselection())
        if not selection:
            return
        updated_files = list(self.switching_file_paths)
        for index in reversed(selection):
            del updated_files[index]
        self._set_switching_files(updated_files)

    def _move_selected_switching_file(self, direction: int) -> None:
        selection = self.switching_file_listbox.curselection()
        if len(selection) != 1:
            return

        index = selection[0]
        new_index = index + direction
        if not 0 <= new_index < len(self.switching_file_paths):
            return

        updated_files = list(self.switching_file_paths)
        updated_files[index], updated_files[new_index] = updated_files[new_index], updated_files[index]
        self._set_switching_files(updated_files)
        self.switching_file_listbox.selection_set(new_index)

    def _clear_switching_files(self) -> None:
        self._set_switching_files([])

    def _set_switching_files(self, file_paths: list[str]) -> None:
        self.switching_file_paths = list(file_paths)
        self._clear_switching_results()
        self._sync_switching_file_listbox()
        self._render_switching_placeholder()

    def _clear_switching_results(self) -> None:
        self.switching_result = None
        self.switching_selected_pixel = None
        self.switching_map_axes = []

    def _sync_switching_file_listbox(self) -> None:
        self.switching_file_listbox.delete(0, tk.END)
        for index, path in enumerate(self.switching_file_paths):
            label = "initial" if index == 0 else f"after pulse {index}"
            self.switching_file_listbox.insert(tk.END, f"{index + 1}. {Path(path).name} ({label})")

    def _parse_switching_parameters(self) -> SwitchingMapParameters:
        try:
            params = SwitchingMapParameters(
                fermi_level_ev=float(self.switching_parameter_vars["fermi_level_ev"].get()),
                ef_min_ev=float(self.switching_parameter_vars["ef_min_ev"].get()),
                ef_max_ev=float(self.switching_parameter_vars["ef_max_ev"].get()),
                lhb_center_ev=float(self.switching_parameter_vars["lhb_center_ev"].get()),
                lhb_halfwidth_ev=float(self.switching_parameter_vars["lhb_halfwidth_ev"].get()),
                smooth_sigma=float(self.switching_parameter_vars["smooth_sigma"].get()),
                low_switch_quantile=float(self.switching_parameter_vars["low_switch_quantile"].get()),
                high_switch_quantile=float(self.switching_parameter_vars["high_switch_quantile"].get()),
                small_net_quantile=float(self.switching_parameter_vars["small_net_quantile"].get()),
                low_signal_quantile=float(self.switching_parameter_vars["low_signal_quantile"].get()),
                lhb_min_quantile=float(self.switching_parameter_vars["lhb_min_quantile"].get()),
            )
        except ValueError as exc:
            raise ValueError(f"Could not parse the Switching Map controls: {exc}") from exc
        params.validate()
        return params

    def _run_switching_map(self) -> None:
        if len(self.switching_file_paths) < 2:
            messagebox.showerror("Missing files", "Please choose at least two chronological NetCDF files.")
            return

        try:
            params = self._parse_switching_parameters()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self.switching_status_var.set("Computing Switching Prediction maps from the chronological sequence...")
        self._start_global_progress("Switching Map running...")
        self.root.update_idletasks()

        try:
            self.switching_result = run_switching_map(self.switching_file_paths, params)
            self.switching_selected_pixel = self._default_switching_pixel()
        except Exception as exc:
            self._clear_switching_results()
            self.switching_status_var.set("Switching Map failed.")
            self._finish_global_progress("Switching Map failed.", success=False)
            messagebox.showerror("Switching Map failed", str(exc))
            self._render_switching_placeholder()
            return

        self._refresh_switching_views()
        shape = self.switching_result.shape
        alignment_suffix = f" {self.switching_result.notes[0]}" if self.switching_result.notes else ""
        self.switching_status_var.set(
            f"Computed Switching Map from {self.switching_result.n_states} files as {shape[0]} x {shape[1]} pixels."
            f"{alignment_suffix}"
        )
        self._finish_global_progress("Switching Map complete.")

    def _default_switching_pixel(self) -> tuple[int, int]:
        if self.switching_result is None:
            return (0, 0)
        coefficient = np.asarray(self.switching_result.switching_coefficient_map, dtype=np.float32)
        if not np.any(np.isfinite(coefficient)):
            return (0, 0)
        flat_index = int(np.nanargmax(coefficient))
        return divmod(flat_index, coefficient.shape[1])

    def _refresh_switching_views(self) -> None:
        if self.switching_result is None:
            self._render_switching_placeholder()
            return
        if self.switching_selected_pixel is None:
            self.switching_selected_pixel = self._default_switching_pixel()
        self._refresh_switching_plot()
        self._update_switching_summary_text()

    def _render_switching_placeholder(self) -> None:
        if not hasattr(self, "switching_figure"):
            return

        self.switching_figure.clear()
        axis = self.switching_figure.add_subplot(111)
        message = (
            "Ready to predict switching sites.\nOrder the files chronologically, tune EF/LHB windows, then compute the map."
            if self.switching_file_paths
            else "Add chronological NetCDF files to build a Switching Map."
        )
        axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=13)
        axis.set_axis_off()
        self.switching_canvas.draw_idle()
        self.switching_map_axes = []
        if hasattr(self, "switching_summary_text"):
            self._set_text_widget(self.switching_summary_text, "")
        if not self.switching_file_paths:
            self.switching_status_var.set("Add chronological NetCDF files, tune EF/LHB windows, then compute switching sites.")

    def _refresh_switching_plot(self) -> None:
        assert self.switching_result is not None
        assert self.switching_selected_pixel is not None
        result = self.switching_result
        x_index, y_index = self.switching_selected_pixel

        self.switching_figure.clear()
        axes = self.switching_figure.subplots(3, 3)
        initial_axis, final_axis, coeff_axis = axes[0]
        net_axis, label_axis, trace_axis = axes[1]
        spectrum_axis, edc_axis, mdc_axis = axes[2]
        self.switching_map_axes = [initial_axis, final_axis, coeff_axis, net_axis, label_axis]

        irat_limits = self._switching_feature_limits(np.stack([result.i_rat_maps[0], result.i_rat_maps[-1]], axis=0))
        initial_image = initial_axis.imshow(
            result.i_rat_maps[0].T,
            origin="lower",
            cmap="viridis",
            aspect="auto",
            vmin=irat_limits[0],
            vmax=irat_limits[1],
        )
        initial_axis.set_title("Initial I_rat")
        self.switching_figure.colorbar(initial_image, ax=initial_axis, fraction=0.046, pad=0.04)

        final_image = final_axis.imshow(
            result.i_rat_maps[-1].T,
            origin="lower",
            cmap="viridis",
            aspect="auto",
            vmin=irat_limits[0],
            vmax=irat_limits[1],
        )
        final_axis.set_title("Final I_rat")
        self.switching_figure.colorbar(final_image, ax=final_axis, fraction=0.046, pad=0.04)

        coeff_image = coeff_axis.imshow(
            result.switching_coefficient_map.T,
            origin="lower",
            cmap="magma",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        coeff_axis.set_title("Switching coefficient")
        self.switching_figure.colorbar(coeff_image, ax=coeff_axis, fraction=0.046, pad=0.04)

        net_limit = self._symmetric_change_limit(result.net_change_map)
        net_image = net_axis.imshow(
            result.net_change_map.T,
            origin="lower",
            cmap="coolwarm",
            aspect="auto",
            vmin=-net_limit,
            vmax=net_limit,
        )
        net_axis.set_title("Net change: final - initial")
        self.switching_figure.colorbar(net_image, ax=net_axis, fraction=0.046, pad=0.04)

        colors = [SWITCHING_COLORS[label] for label in SWITCHING_LABELS]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(np.arange(-0.5, len(SWITCHING_LABELS) + 0.5, 1.0), cmap.N)
        label_image = label_axis.imshow(result.code_map.T, origin="lower", cmap=cmap, norm=norm, aspect="auto")
        label_axis.set_title("State label map")
        cbar = self.switching_figure.colorbar(label_image, ax=label_axis, fraction=0.046, pad=0.04)
        cbar.set_ticks(np.arange(len(SWITCHING_LABELS)))
        cbar.ax.set_yticklabels(SWITCHING_LABELS, fontsize=7)

        for axis in self.switching_map_axes:
            axis.set_xlabel("x index")
            axis.set_ylabel("y index")
            self._mark_switching_selected_pixel(axis)

        self._plot_switching_pixel_trace(trace_axis, x_index, y_index)
        self._plot_switching_local_spectrum(spectrum_axis, x_index, y_index)
        self._plot_switching_edc_overlay(edc_axis, x_index, y_index)
        self._plot_switching_mdc_overlay(mdc_axis, x_index, y_index)

        self.switching_canvas.draw_idle()

    def _switching_feature_limits(self, values: np.ndarray) -> tuple[float | None, float | None]:
        finite = np.asarray(values, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return None, None
        low = float(np.nanpercentile(finite, 1))
        high = float(np.nanpercentile(finite, 99))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return None, None
        return low, high

    def _on_switching_plot_click(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        if self.switching_result is None or event.inaxes not in self.switching_map_axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        x_index = int(round(event.xdata))
        y_index = int(round(event.ydata))
        x_size, y_size = self.switching_result.shape
        if not (0 <= x_index < x_size and 0 <= y_index < y_size):
            return
        self.switching_selected_pixel = (x_index, y_index)
        self._refresh_switching_views()

    def _mark_switching_selected_pixel(self, axis: matplotlib.axes.Axes) -> None:
        if self.switching_selected_pixel is None:
            return
        x_index, y_index = self.switching_selected_pixel
        axis.scatter([x_index], [y_index], s=96, facecolors="none", edgecolors="white", linewidths=1.9)
        axis.scatter([x_index], [y_index], s=20, c="black")

    def _switching_spectrum_payload(
        self,
        state_index: int,
        x_index: int,
        y_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.switching_result is not None
        state = self.switching_result.loaded_states[state_index]
        data = np.asarray(state.data_array.values, dtype=np.float32)
        x_safe = min(max(0, int(x_index)), data.shape[0] - 1)
        y_safe = min(max(0, int(y_index)), data.shape[1] - 1)
        energy_axis = np.asarray(state.data_array.coords["eV"].values, dtype=np.float32)
        phi_axis = np.asarray(state.data_array.coords["phi"].values, dtype=np.float32)
        energy_order = np.argsort(energy_axis)
        phi_order = np.argsort(phi_axis)
        spectrum = np.asarray(data[x_safe, y_safe, :, :], dtype=np.float32)
        return spectrum[energy_order][:, phi_order], energy_axis[energy_order], phi_axis[phi_order]

    def _plot_switching_pixel_trace(self, axis: matplotlib.axes.Axes, x_index: int, y_index: int) -> None:
        assert self.switching_result is not None
        result = self.switching_result
        file_indices = np.arange(result.n_states)
        irat_values = np.asarray([maps[x_index, y_index] for maps in result.i_rat_maps], dtype=np.float32)
        wef_values = np.asarray([maps[x_index, y_index] for maps in result.w_ef_maps], dtype=np.float32)
        delta_values = np.asarray([maps[x_index, y_index] for maps in result.delta_irat_maps], dtype=np.float32)

        axis.plot(file_indices, irat_values, marker="o", color="#d62728", label="I_rat")
        if delta_values.size:
            axis.bar(file_indices[1:], delta_values, width=0.34, color="#777777", alpha=0.25, label="Delta I_rat")
        axis.axhline(float(irat_values[0]), color="#d62728", linestyle=":", linewidth=0.9, alpha=0.65)
        axis.set_xlabel("file / pulse index")
        axis.set_ylabel("I_rat")
        axis.set_title("Selected-pixel switching trace")

        twin = axis.twinx()
        twin.plot(file_indices, wef_values, marker="s", color="#1f77b4", linewidth=1.2, label="W_EF")
        twin.set_ylabel("W_EF")
        handles, labels = axis.get_legend_handles_labels()
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        axis.legend(handles + twin_handles, labels + twin_labels, loc="best", fontsize=7)

    def _plot_switching_local_spectrum(self, axis: matplotlib.axes.Axes, x_index: int, y_index: int) -> None:
        spectrum, energy_axis, phi_axis = self._switching_spectrum_payload(0, x_index, y_index)
        image = axis.imshow(
            spectrum,
            origin="lower",
            aspect="auto",
            extent=[float(phi_axis[0]), float(phi_axis[-1]), float(energy_axis[0]), float(energy_axis[-1])],
            cmap="viridis",
        )
        axis.axhline(0.0, color="white", linestyle="--", linewidth=0.8)
        axis.set_title(f"Initial local ARPES image\nx={x_index}, y={y_index}")
        axis.set_xlabel("phi")
        axis.set_ylabel("eV")
        self.switching_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    def _plot_switching_edc_overlay(self, axis: matplotlib.axes.Axes, x_index: int, y_index: int) -> None:
        assert self.switching_result is not None
        for state_index, state in enumerate(self.switching_result.loaded_states):
            spectrum, energy_axis, phi_axis = self._switching_spectrum_payload(state_index, x_index, y_index)
            if phi_axis.size > 1:
                edc = np.trapezoid(spectrum, x=phi_axis, axis=1).astype(np.float32)
            else:
                edc = np.sum(spectrum, axis=1).astype(np.float32)
            scale = float(np.nanmax(np.abs(edc))) if np.any(np.isfinite(edc)) else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            axis.plot(energy_axis, edc / scale, linewidth=1.25, label=f"{state_index}: {self._short_file_label(state.file_path, 18)}")

        params = self.switching_result.parameters
        axis.axvline(params.fermi_level_ev, color="#555555", linestyle="--", linewidth=0.8)
        axis.axvspan(
            params.fermi_level_ev + params.ef_min_ev,
            params.fermi_level_ev + params.ef_max_ev,
            color="#6baed6",
            alpha=0.16,
            linewidth=0,
        )
        axis.axvspan(
            params.lhb_center_ev - params.lhb_halfwidth_ev,
            params.lhb_center_ev + params.lhb_halfwidth_ev,
            color="#fb6a4a",
            alpha=0.13,
            linewidth=0,
        )
        axis.set_title("EDC evolution")
        axis.set_xlabel("eV")
        axis.set_ylabel("normalized intensity")
        axis.legend(loc="best", fontsize=6)

    def _plot_switching_mdc_overlay(self, axis: matplotlib.axes.Axes, x_index: int, y_index: int) -> None:
        assert self.switching_result is not None
        params = self.switching_result.parameters
        for state_index, state in enumerate(self.switching_result.loaded_states):
            spectrum, energy_axis, phi_axis = self._switching_spectrum_payload(state_index, x_index, y_index)
            mask = (
                (energy_axis >= params.fermi_level_ev + params.ef_min_ev)
                & (energy_axis <= params.fermi_level_ev + params.ef_max_ev)
            )
            if not np.any(mask):
                mask[int(np.argmin(np.abs(energy_axis - params.fermi_level_ev)))] = True
            if int(np.count_nonzero(mask)) > 1:
                mdc = np.trapezoid(spectrum[mask, :], x=energy_axis[mask], axis=0).astype(np.float32)
            else:
                mdc = np.sum(spectrum[mask, :], axis=0).astype(np.float32)
            scale = float(np.nanmax(np.abs(mdc))) if np.any(np.isfinite(mdc)) else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            axis.plot(phi_axis, mdc / scale, linewidth=1.25, label=f"{state_index}: {self._short_file_label(state.file_path, 18)}")

        axis.set_title("Near-EF MDC evolution")
        axis.set_xlabel("phi")
        axis.set_ylabel("normalized intensity")
        axis.legend(loc="best", fontsize=6)

    def _update_switching_summary_text(self) -> None:
        result = self.switching_result
        if result is None:
            self._set_text_widget(self.switching_summary_text, "")
            return

        lines = [
            f"Files: {result.n_states}",
            f"Map shape: {result.shape[0]} x {result.shape[1]}",
            "",
            "State counts:",
        ]
        total_pixels = max(1, result.shape[0] * result.shape[1])
        for label in SWITCHING_LABELS:
            count = result.counts.get(label, 0)
            lines.append(f"  {label}: {count} ({count / total_pixels:.1%})")

        lines.extend(
            [
                "",
                "Thresholds:",
                f"  switching low/high: {result.threshold_values['low_switch_threshold']:.5g} / {result.threshold_values['high_switch_threshold']:.5g}",
                f"  small net change: {result.threshold_values['small_net_change_threshold']:.5g}",
                f"  low T / min W_LHB: {result.threshold_values['low_signal_T_threshold']:.5g} / {result.threshold_values['min_W_LHB_threshold']:.5g}",
            ]
        )

        if self.switching_selected_pixel is not None:
            x_index, y_index = self.switching_selected_pixel
            lines.extend(
                [
                    "",
                    f"Selected pixel: x={x_index}, y={y_index}",
                    f"State: {result.label_map[x_index, y_index]}",
                    f"Switching coefficient: {float(result.switching_coefficient_map[x_index, y_index]):.6g}",
                    f"Total change: {float(result.total_change_map[x_index, y_index]):.6g}",
                    f"Max transition change: {float(result.max_change_map[x_index, y_index]):.6g}",
                    f"Net change from initial: {float(result.net_change_map[x_index, y_index]):.6g}",
                    "",
                    "Per-file values:",
                ]
            )
            for state_index, state in enumerate(result.loaded_states):
                lines.append(
                    f"  {state_index}: {Path(state.file_path).name}  "
                    f"I_rat={float(result.i_rat_maps[state_index][x_index, y_index]):.6g}, "
                    f"W_EF={float(result.w_ef_maps[state_index][x_index, y_index]):.6g}, "
                    f"Delta_initial={float(result.initial_delta_irat_maps[state_index][x_index, y_index]):+.6g}"
                )
            if result.delta_irat_maps:
                lines.append("")
                lines.append("Delta_Irat by transition:")
                for transition_index, delta_map in enumerate(result.delta_irat_maps):
                    lines.append(
                        f"  {transition_index} -> {transition_index + 1}: "
                        f"{float(delta_map[x_index, y_index]):+.6g}"
                    )
            lines.extend(["", self._switching_pixel_summary_sentence(x_index, y_index)])

        if result.notes:
            lines.extend(["", "Notes:"])
            lines.extend(f"  {note}" for note in result.notes)

        self._set_text_widget(self.switching_summary_text, "\n".join(lines))

    def _switching_pixel_summary_sentence(self, x_index: int, y_index: int) -> str:
        assert self.switching_result is not None
        result = self.switching_result
        label = str(result.label_map[x_index, y_index])
        delta_values = np.asarray([maps[x_index, y_index] for maps in result.delta_irat_maps], dtype=np.float32)
        if delta_values.size and np.any(np.isfinite(delta_values)):
            largest_transition = int(np.nanargmax(np.abs(delta_values))) + 1
        else:
            largest_transition = 1

        if label == "written / becomes metallic":
            return f"This pixel becomes more metallic overall; the largest I_rat change occurs after pulse {largest_transition}."
        if label == "erased / becomes less metallic":
            return f"This pixel becomes less metallic overall; the largest I_rat change occurs after pulse {largest_transition}."
        if label == "reversible / memory-like":
            return "This pixel changes strongly during the sequence but returns close to its initial I_rat, suggesting reversible or memory-like behavior."
        if label == "stable / unchanged":
            return "This pixel remains comparatively stable across the pulse sequence."
        return "This pixel has weak, noisy, or intermediate switching evidence, so it is marked ambiguous."

    def _save_switching_results(self) -> None:
        if self.switching_result is None:
            messagebox.showinfo("No Switching Map", "Compute the Switching Map before saving results.")
            return
        directory = filedialog.askdirectory(title="Choose output folder for Switching Map results")
        if not directory:
            return
        try:
            paths = export_switching_map(self.switching_result, directory)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.switching_status_var.set(f"Saved Switching Map feature table to {paths['table']}")

    def _save_switching_plot(self) -> None:
        if self.switching_result is None:
            messagebox.showinfo("No Switching Map plot", "Compute the Switching Map before saving a plot.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Switching Map plot",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF document", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.switching_figure.savefig(path, dpi=220)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.switching_status_var.set(f"Saved Switching Map plot to {path}")

    def _add_state_prediction_files(self) -> None:
        selected = list(filedialog.askopenfilenames(title="Choose NetCDF files", filetypes=FILE_TYPES))
        if not selected:
            return

        new_paths = [str(Path(path).expanduser().resolve()) for path in selected]
        merged = self.state_prediction_file_paths + [
            path for path in new_paths if path not in self.state_prediction_file_paths
        ]
        self._set_state_prediction_files(merged)

    def _copy_analysis_files_to_state_prediction_panel(self) -> None:
        if not self.file_paths:
            messagebox.showinfo("No analysis files", "Add files to the Analysis panel first, or add files here directly.")
            return
        self._set_state_prediction_files(self.file_paths)
        self.top_notebook.select(7)

    def _remove_selected_state_prediction_files(self) -> None:
        selection = list(self.state_prediction_file_listbox.curselection())
        if not selection:
            return
        updated_files = list(self.state_prediction_file_paths)
        for index in reversed(selection):
            del updated_files[index]
        self._set_state_prediction_files(updated_files)

    def _move_selected_state_prediction_file(self, direction: int) -> None:
        selection = self.state_prediction_file_listbox.curselection()
        if len(selection) != 1:
            return

        index = selection[0]
        new_index = index + direction
        if not 0 <= new_index < len(self.state_prediction_file_paths):
            return

        updated_files = list(self.state_prediction_file_paths)
        updated_files[index], updated_files[new_index] = updated_files[new_index], updated_files[index]
        self._set_state_prediction_files(updated_files)
        self.state_prediction_file_listbox.selection_set(new_index)

    def _clear_state_prediction_files(self) -> None:
        self._set_state_prediction_files([])

    def _set_state_prediction_files(self, file_paths: list[str]) -> None:
        self.state_prediction_file_paths = list(file_paths)
        self._clear_state_prediction_results()
        self._sync_state_prediction_file_listbox()
        self._render_state_prediction_placeholder()

    def _clear_state_prediction_results(self) -> None:
        self.state_prediction_result = None
        self.state_prediction_selected_pixel = None
        self.state_prediction_map_axes = []

    def _sync_state_prediction_file_listbox(self) -> None:
        self.state_prediction_file_listbox.delete(0, tk.END)
        for index, path in enumerate(self.state_prediction_file_paths):
            label = "initial" if index == 0 else f"after pulse {index}"
            self.state_prediction_file_listbox.insert(tk.END, f"{index + 1}. {Path(path).name} ({label})")

    def _parse_state_prediction_parameters(self) -> StatePredictionParameters:
        tau_text = self.state_prediction_parameter_vars["net_change_tau"].get().strip()
        try:
            net_tau = None if tau_text == "" else float(tau_text)
            params = StatePredictionParameters(
                fermi_level_ev=float(self.state_prediction_parameter_vars["fermi_level_ev"].get()),
                ef_min_ev=float(self.state_prediction_parameter_vars["ef_min_ev"].get()),
                ef_max_ev=float(self.state_prediction_parameter_vars["ef_max_ev"].get()),
                lhb_center_ev=float(self.state_prediction_parameter_vars["lhb_center_ev"].get()),
                lhb_halfwidth_ev=float(self.state_prediction_parameter_vars["lhb_halfwidth_ev"].get()),
                leading_edge_min_ev=float(self.state_prediction_parameter_vars["leading_edge_min_ev"].get()),
                leading_edge_max_ev=float(self.state_prediction_parameter_vars["leading_edge_max_ev"].get()),
                p3_center_ev=float(self.state_prediction_parameter_vars["p3_center_ev"].get()),
                p3_halfwidth_ev=float(self.state_prediction_parameter_vars["p3_halfwidth_ev"].get()),
                smooth_sigma=float(self.state_prediction_parameter_vars["smooth_sigma"].get()),
                stable_quantile=float(self.state_prediction_parameter_vars["stable_quantile"].get()),
                switch_quantile=float(self.state_prediction_parameter_vars["switch_quantile"].get()),
                net_change_tau=net_tau,
                low_signal_quantile=float(self.state_prediction_parameter_vars["low_signal_quantile"].get()),
                lhb_min_quantile=float(self.state_prediction_parameter_vars["lhb_min_quantile"].get()),
                phase_low_quantile=float(self.state_prediction_parameter_vars["phase_low_quantile"].get()),
                phase_high_quantile=float(self.state_prediction_parameter_vars["phase_high_quantile"].get()),
                structural_gradient_quantile=float(
                    self.state_prediction_parameter_vars["structural_gradient_quantile"].get()
                ),
            )
        except ValueError as exc:
            raise ValueError(f"Could not parse the State Prediction controls: {exc}") from exc
        params.validate()
        return params

    def _run_state_prediction(self) -> None:
        if len(self.state_prediction_file_paths) < 2:
            messagebox.showerror("Missing files", "Please choose at least two chronological NetCDF files.")
            return

        try:
            params = self._parse_state_prediction_parameters()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self.state_prediction_status_var.set("Computing initial-state prediction diagnostics...")
        self._start_global_progress("State Prediction running...")
        self.root.update_idletasks()

        try:
            self.state_prediction_result = run_state_prediction(self.state_prediction_file_paths, params)
            self.state_prediction_selected_pixel = self._default_state_prediction_pixel()
        except Exception as exc:
            self._clear_state_prediction_results()
            self.state_prediction_status_var.set("State Prediction failed.")
            self._finish_global_progress("State Prediction failed.", success=False)
            messagebox.showerror("State Prediction failed", str(exc))
            self._render_state_prediction_placeholder()
            return

        self._refresh_state_prediction_views()
        shape = self.state_prediction_result.shape
        alignment_suffix = f" {self.state_prediction_result.notes[0]}" if self.state_prediction_result.notes else ""
        self.state_prediction_status_var.set(
            f"Computed State Prediction from {self.state_prediction_result.n_states} files as {shape[0]} x {shape[1]} pixels."
            f"{alignment_suffix}"
        )
        self._finish_global_progress("State Prediction complete.")

    def _default_state_prediction_pixel(self) -> tuple[int, int]:
        if self.state_prediction_result is None:
            return (0, 0)
        score = np.asarray(
            self.state_prediction_result.switching_result.switching_coefficient_map,
            dtype=np.float32,
        )
        if not np.any(np.isfinite(score)):
            return (0, 0)
        flat_index = int(np.nanargmax(score))
        return divmod(flat_index, score.shape[1])

    def _refresh_state_prediction_views(self) -> None:
        if self.state_prediction_result is None:
            self._render_state_prediction_placeholder()
            return
        if self.state_prediction_selected_pixel is None:
            self.state_prediction_selected_pixel = self._default_state_prediction_pixel()
        self._refresh_state_prediction_plot()
        self._update_state_prediction_summary_text()

    def _render_state_prediction_placeholder(self) -> None:
        if not hasattr(self, "state_prediction_figure"):
            return

        self.state_prediction_figure.clear()
        axis = self.state_prediction_figure.add_subplot(111)
        message = (
            "Ready to compare initial features to future switching.\nOrder chronological files, tune windows, then compute State Prediction."
            if self.state_prediction_file_paths
            else "Add chronological NetCDF files to build State Prediction diagnostics."
        )
        axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=13)
        axis.set_axis_off()
        self.state_prediction_canvas.draw_idle()
        self._update_state_prediction_plot_scroll_region()
        self.state_prediction_map_axes = []
        if hasattr(self, "state_prediction_summary_text"):
            self._set_text_widget(self.state_prediction_summary_text, "")
        if not self.state_prediction_file_paths:
            self.state_prediction_status_var.set(
                "Add chronological NetCDF files, then compare future switching to initial-state features."
            )

    def _refresh_state_prediction_plot(self) -> None:
        assert self.state_prediction_result is not None
        assert self.state_prediction_selected_pixel is not None
        result = self.state_prediction_result
        switching = result.switching_result
        x_index, y_index = self.state_prediction_selected_pixel

        self.state_prediction_figure.clear()
        grid = self.state_prediction_figure.add_gridspec(
            5,
            4,
            height_ratios=[1.05, 1.05, 1.05, 1.05, 0.95],
            hspace=0.72,
            wspace=0.34,
        )

        initial_axis = self.state_prediction_figure.add_subplot(grid[0, 0])
        score_axis = self.state_prediction_figure.add_subplot(grid[0, 1])
        net_axis = self.state_prediction_figure.add_subplot(grid[0, 2])
        label_axis = self.state_prediction_figure.add_subplot(grid[0, 3])
        self.state_prediction_map_axes = [initial_axis, score_axis, net_axis, label_axis]

        self._draw_state_prediction_maps(initial_axis, score_axis, net_axis, label_axis)
        for axis in self.state_prediction_map_axes:
            self._mark_state_prediction_selected_pixel(axis)

        average_axis = self.state_prediction_figure.add_subplot(grid[1, 0:2])
        trace_axis = self.state_prediction_figure.add_subplot(grid[1, 2:4])
        feature_axis = self.state_prediction_figure.add_subplot(grid[2, 0:2])
        distance_axis = self.state_prediction_figure.add_subplot(grid[2, 2:4])
        correlation_axis = self.state_prediction_figure.add_subplot(grid[3, 0:2])
        edc_axis = self.state_prediction_figure.add_subplot(grid[3, 2])
        mdc_axis = self.state_prediction_figure.add_subplot(grid[3, 3])
        scatter_irat_axis = self.state_prediction_figure.add_subplot(grid[4, 0])
        scatter_wef_axis = self.state_prediction_figure.add_subplot(grid[4, 1])
        scatter_gamma_axis = self.state_prediction_figure.add_subplot(grid[4, 2])
        scatter_boundary_axis = self.state_prediction_figure.add_subplot(grid[4, 3])

        self._plot_state_prediction_average_edcs(average_axis)
        self._plot_state_prediction_pixel_trace(trace_axis, x_index, y_index)
        self._plot_state_prediction_feature_distributions(feature_axis)
        self._plot_state_prediction_distance_distributions(distance_axis)
        self._plot_state_prediction_correlations(correlation_axis)
        self._plot_state_prediction_edc_overlay(edc_axis, x_index, y_index)
        self._plot_state_prediction_mdc_overlay(mdc_axis, x_index, y_index)
        self._plot_state_prediction_scatter(
            scatter_irat_axis,
            result.initial_feature_maps["I_rat"],
            switching.switching_coefficient_map,
            "I_rat initial",
        )
        self._plot_state_prediction_scatter(
            scatter_wef_axis,
            result.initial_feature_maps["W_EF"],
            switching.switching_coefficient_map,
            "W_EF initial",
        )
        self._plot_state_prediction_scatter(
            scatter_gamma_axis,
            result.initial_feature_maps["Gamma_EDC"],
            switching.switching_coefficient_map,
            "Gamma initial",
        )
        self._plot_state_prediction_boundary_scatter(scatter_boundary_axis)

        self.state_prediction_canvas.draw_idle()
        self._update_state_prediction_plot_scroll_region()

    def _draw_state_prediction_maps(
        self,
        initial_axis: matplotlib.axes.Axes,
        score_axis: matplotlib.axes.Axes,
        net_axis: matplotlib.axes.Axes,
        label_axis: matplotlib.axes.Axes,
    ) -> None:
        assert self.state_prediction_result is not None
        result = self.state_prediction_result
        switching = result.switching_result
        irat_limits = self._switching_feature_limits(result.initial_feature_maps["I_rat"])
        initial_image = initial_axis.imshow(
            result.initial_feature_maps["I_rat"].T,
            origin="lower",
            cmap="viridis",
            aspect="auto",
            vmin=irat_limits[0],
            vmax=irat_limits[1],
        )
        initial_axis.set_title("Initial I_rat")
        self.state_prediction_figure.colorbar(initial_image, ax=initial_axis, fraction=0.046, pad=0.04)

        score_image = score_axis.imshow(
            switching.switching_coefficient_map.T,
            origin="lower",
            cmap="magma",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        score_axis.set_title("Predictive score")
        self.state_prediction_figure.colorbar(score_image, ax=score_axis, fraction=0.046, pad=0.04)

        net_limit = self._symmetric_change_limit(switching.net_change_map)
        net_image = net_axis.imshow(
            switching.net_change_map.T,
            origin="lower",
            cmap="coolwarm",
            aspect="auto",
            vmin=-net_limit,
            vmax=net_limit,
        )
        net_axis.set_title("Net change")
        self.state_prediction_figure.colorbar(net_image, ax=net_axis, fraction=0.046, pad=0.04)

        colors = [SWITCHING_COLORS[label] for label in SWITCHING_LABELS]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(np.arange(-0.5, len(SWITCHING_LABELS) + 0.5, 1.0), cmap.N)
        label_image = label_axis.imshow(result.code_map.T, origin="lower", cmap=cmap, norm=norm, aspect="auto")
        label_axis.set_title("Future outcome")
        cbar = self.state_prediction_figure.colorbar(label_image, ax=label_axis, fraction=0.046, pad=0.04)
        cbar.set_ticks(np.arange(len(SWITCHING_LABELS)))
        cbar.ax.set_yticklabels(SWITCHING_LABELS, fontsize=7)

        for axis in (initial_axis, score_axis, net_axis, label_axis):
            axis.set_xlabel("x index")
            axis.set_ylabel("y index")

    def _on_state_prediction_plot_click(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        if self.state_prediction_result is None or event.inaxes not in self.state_prediction_map_axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        x_index = int(round(event.xdata))
        y_index = int(round(event.ydata))
        x_size, y_size = self.state_prediction_result.shape
        if not (0 <= x_index < x_size and 0 <= y_index < y_size):
            return
        self.state_prediction_selected_pixel = (x_index, y_index)
        self._refresh_state_prediction_views()

    def _mark_state_prediction_selected_pixel(self, axis: matplotlib.axes.Axes) -> None:
        if self.state_prediction_selected_pixel is None:
            return
        x_index, y_index = self.state_prediction_selected_pixel
        axis.scatter([x_index], [y_index], s=96, facecolors="none", edgecolors="white", linewidths=1.9)
        axis.scatter([x_index], [y_index], s=20, c="black")

    def _state_prediction_spectrum_payload(
        self,
        state_index: int,
        x_index: int,
        y_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.state_prediction_result is not None
        state = self.state_prediction_result.loaded_states[state_index]
        data = np.asarray(state.data_array.values, dtype=np.float32)
        x_safe = min(max(0, int(x_index)), data.shape[0] - 1)
        y_safe = min(max(0, int(y_index)), data.shape[1] - 1)
        energy_axis = np.asarray(state.data_array.coords["eV"].values, dtype=np.float32)
        phi_axis = np.asarray(state.data_array.coords["phi"].values, dtype=np.float32)
        energy_order = np.argsort(energy_axis)
        phi_order = np.argsort(phi_axis)
        spectrum = np.asarray(data[x_safe, y_safe, :, :], dtype=np.float32)
        return spectrum[energy_order][:, phi_order], energy_axis[energy_order], phi_axis[phi_order]

    def _plot_state_prediction_average_edcs(self, axis: matplotlib.axes.Axes) -> None:
        assert self.state_prediction_result is not None
        result = self.state_prediction_result
        energy_axis = np.sort(np.asarray(result.e_axis, dtype=np.float32))
        for code, label in enumerate(SWITCHING_LABELS[:4]):
            edc = np.asarray(result.average_initial_edcs[label], dtype=np.float32)
            if not np.any(np.isfinite(edc)):
                continue
            scale = float(np.nanmax(np.abs(edc)))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            axis.plot(
                energy_axis,
                edc / scale,
                linewidth=1.7,
                color=SWITCHING_COLORS[label],
                label=f"{label} ({result.counts.get(label, 0)})",
            )
        axis.axvline(result.parameters.fermi_level_ev, color="#555555", linestyle="--", linewidth=0.8)
        axis.set_title("Average initial EDCs by future outcome")
        axis.set_xlabel("eV")
        axis.set_ylabel("normalized intensity")
        axis.legend(loc="best", fontsize=7)

    def _plot_state_prediction_pixel_trace(self, axis: matplotlib.axes.Axes, x_index: int, y_index: int) -> None:
        assert self.state_prediction_result is not None
        switching = self.state_prediction_result.switching_result
        file_indices = np.arange(switching.n_states)
        irat_values = np.asarray([maps[x_index, y_index] for maps in switching.i_rat_maps], dtype=np.float32)
        wef_values = np.asarray([maps[x_index, y_index] for maps in switching.w_ef_maps], dtype=np.float32)
        delta_values = np.asarray([maps[x_index, y_index] for maps in switching.delta_irat_maps], dtype=np.float32)

        axis.plot(file_indices, irat_values, marker="o", color="#d62728", label="I_rat")
        if delta_values.size:
            axis.bar(file_indices[1:], delta_values, width=0.34, color="#777777", alpha=0.25, label="Delta I_rat")
        axis.set_xlabel("file / pulse index")
        axis.set_ylabel("I_rat")
        axis.set_title(f"Selected pixel trace: x={x_index}, y={y_index}")
        twin = axis.twinx()
        twin.plot(file_indices, wef_values, marker="s", color="#1f77b4", linewidth=1.2, label="W_EF")
        twin.set_ylabel("W_EF")
        handles, labels = axis.get_legend_handles_labels()
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        axis.legend(handles + twin_handles, labels + twin_labels, loc="best", fontsize=7)

    def _plot_state_prediction_feature_distributions(self, axis: matplotlib.axes.Axes) -> None:
        assert self.state_prediction_result is not None
        result = self.state_prediction_result
        features = [
            ("I_rat", "I_rat"),
            ("W_EF", "W_EF"),
            ("E_LE", "E_LE"),
            ("Gamma", "Gamma_EDC"),
            ("E_p3", "S_orient"),
        ]
        self._plot_grouped_boxplots(
            axis,
            [(label, result.initial_feature_maps[key]) for label, key in features],
            title="Initial feature distributions by future outcome",
            ylabel="robust normalized value",
        )

    def _plot_state_prediction_distance_distributions(self, axis: matplotlib.axes.Axes) -> None:
        assert self.state_prediction_result is not None
        result = self.state_prediction_result
        maps = [
            ("edge", result.distance_maps["distance_to_edge"]),
            ("phase boundary", result.distance_maps["distance_to_phase_boundary"]),
            ("structural boundary", result.distance_maps["distance_to_structural_boundary"]),
        ]
        self._plot_grouped_boxplots(
            axis,
            maps,
            title="Boundary and geometry distances by future outcome",
            ylabel="distance (pixels)",
            normalize=False,
        )

    def _plot_grouped_boxplots(
        self,
        axis: matplotlib.axes.Axes,
        named_maps: list[tuple[str, np.ndarray]],
        title: str,
        ylabel: str,
        normalize: bool = True,
    ) -> None:
        assert self.state_prediction_result is not None
        result = self.state_prediction_result
        labels = list(SWITCHING_LABELS[:4])
        group_width = len(labels) + 1
        positions: list[float] = []
        values: list[np.ndarray] = []
        colors: list[str] = []

        for feature_index, (_name, value_map) in enumerate(named_maps):
            values_map = np.asarray(value_map, dtype=np.float32)
            plot_map = self._normalize_map_for_distribution(values_map, result.valid_mask) if normalize else values_map
            for label_index, outcome_label in enumerate(labels):
                mask = result.valid_mask & (result.code_map == label_index)
                finite = plot_map[mask]
                finite = finite[np.isfinite(finite)]
                if finite.size == 0:
                    continue
                positions.append(feature_index * group_width + label_index)
                values.append(finite)
                colors.append(SWITCHING_COLORS[outcome_label])

        if values:
            box = axis.boxplot(values, positions=positions, widths=0.68, patch_artist=True, showfliers=False)
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
            for median in box["medians"]:
                median.set_color("#111111")
                median.set_linewidth(1.1)

        centers = [feature_index * group_width + (len(labels) - 1) / 2 for feature_index in range(len(named_maps))]
        axis.set_xticks(centers)
        axis.set_xticklabels([name for name, _values in named_maps], rotation=18, ha="right", fontsize=8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)

    def _normalize_map_for_distribution(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        finite = np.asarray(values, dtype=np.float32)
        valid = np.asarray(mask, dtype=bool) & np.isfinite(finite)
        out = np.full(finite.shape, fill_value=np.nan, dtype=np.float32)
        if not np.any(valid):
            return out
        low = float(np.nanpercentile(finite[valid], 2))
        high = float(np.nanpercentile(finite[valid], 98))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            out[valid] = 0.0
        else:
            out[valid] = np.clip((finite[valid] - low) / (high - low), 0.0, 1.0)
        return out

    def _plot_state_prediction_correlations(self, axis: matplotlib.axes.Axes) -> None:
        assert self.state_prediction_result is not None
        correlations = self.state_prediction_result.correlation_values
        labels = list(correlations.keys())
        values = np.asarray([correlations[label] for label in labels], dtype=np.float32)
        colors = ["#d62728" if value >= 0 else "#1f77b4" for value in values]
        axis.bar(np.arange(len(labels)), np.nan_to_num(values, nan=0.0), color=colors, alpha=0.78)
        axis.axhline(0.0, color="#333333", linewidth=0.8)
        axis.set_ylim(-1.0, 1.0)
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels([label.replace("_initial", "").replace("distance_to_", "d_") for label in labels], rotation=28, ha="right", fontsize=7)
        axis.set_ylabel("Pearson r")
        axis.set_title("Initial-feature correlations with switching coefficient")

    def _plot_state_prediction_edc_overlay(self, axis: matplotlib.axes.Axes, x_index: int, y_index: int) -> None:
        assert self.state_prediction_result is not None
        for state_index, state in enumerate(self.state_prediction_result.loaded_states):
            spectrum, energy_axis, phi_axis = self._state_prediction_spectrum_payload(state_index, x_index, y_index)
            if phi_axis.size > 1:
                edc = np.trapezoid(spectrum, x=phi_axis, axis=1).astype(np.float32)
            else:
                edc = np.sum(spectrum, axis=1).astype(np.float32)
            scale = float(np.nanmax(np.abs(edc))) if np.any(np.isfinite(edc)) else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            axis.plot(energy_axis, edc / scale, linewidth=1.15, label=f"{state_index}: {self._short_file_label(state.file_path, 16)}")
        axis.axvline(self.state_prediction_result.parameters.fermi_level_ev, color="#555555", linestyle="--", linewidth=0.8)
        axis.set_title("Selected-pixel EDC evolution")
        axis.set_xlabel("eV")
        axis.set_ylabel("normalized intensity")
        axis.legend(loc="best", fontsize=6)

    def _plot_state_prediction_mdc_overlay(self, axis: matplotlib.axes.Axes, x_index: int, y_index: int) -> None:
        assert self.state_prediction_result is not None
        params = self.state_prediction_result.parameters
        for state_index, state in enumerate(self.state_prediction_result.loaded_states):
            spectrum, energy_axis, phi_axis = self._state_prediction_spectrum_payload(state_index, x_index, y_index)
            mask = (
                (energy_axis >= params.fermi_level_ev + params.ef_min_ev)
                & (energy_axis <= params.fermi_level_ev + params.ef_max_ev)
            )
            if not np.any(mask):
                mask[int(np.argmin(np.abs(energy_axis - params.fermi_level_ev)))] = True
            if int(np.count_nonzero(mask)) > 1:
                mdc = np.trapezoid(spectrum[mask, :], x=energy_axis[mask], axis=0).astype(np.float32)
            else:
                mdc = np.sum(spectrum[mask, :], axis=0).astype(np.float32)
            scale = float(np.nanmax(np.abs(mdc))) if np.any(np.isfinite(mdc)) else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            axis.plot(phi_axis, mdc / scale, linewidth=1.15, label=f"{state_index}: {self._short_file_label(state.file_path, 16)}")
        axis.set_title("Selected-pixel near-EF MDC")
        axis.set_xlabel("phi")
        axis.set_ylabel("normalized intensity")
        axis.legend(loc="best", fontsize=6)

    def _plot_state_prediction_scatter(
        self,
        axis: matplotlib.axes.Axes,
        x_values: np.ndarray,
        y_values: np.ndarray,
        x_label: str,
    ) -> None:
        assert self.state_prediction_result is not None
        result = self.state_prediction_result
        x_arr = np.asarray(x_values, dtype=np.float32)
        y_arr = np.asarray(y_values, dtype=np.float32)
        valid = result.valid_mask & np.isfinite(x_arr) & np.isfinite(y_arr)
        colors = [SWITCHING_COLORS[str(result.label_map[x, y])] for x, y in np.argwhere(valid)]
        axis.scatter(x_arr[valid], y_arr[valid], s=12, c=colors, alpha=0.58, linewidths=0)
        axis.set_xlabel(x_label)
        axis.set_ylabel("switching coefficient")
        axis.set_title(f"{x_label} vs switching")

    def _plot_state_prediction_boundary_scatter(self, axis: matplotlib.axes.Axes) -> None:
        assert self.state_prediction_result is not None
        result = self.state_prediction_result
        y_values = result.switching_result.switching_coefficient_map
        phase = result.distance_maps["distance_to_phase_boundary"]
        structural = result.distance_maps["distance_to_structural_boundary"]
        phase_mask = result.valid_mask & np.isfinite(phase) & np.isfinite(y_values)
        structural_mask = result.valid_mask & np.isfinite(structural) & np.isfinite(y_values)
        axis.scatter(phase[phase_mask], y_values[phase_mask], s=12, color="#9467bd", alpha=0.45, label="phase")
        axis.scatter(structural[structural_mask], y_values[structural_mask], s=12, color="#2ca02c", alpha=0.45, label="structural")
        axis.set_xlabel("boundary distance")
        axis.set_ylabel("switching coefficient")
        axis.set_title("Boundary distances vs switching")
        axis.legend(loc="best", fontsize=7)

    def _update_state_prediction_summary_text(self) -> None:
        result = self.state_prediction_result
        if result is None:
            self._set_text_widget(self.state_prediction_summary_text, "")
            return

        lines = [
            f"Files: {result.n_states}",
            f"Map shape: {result.shape[0]} x {result.shape[1]}",
            f"Orientation marker: {result.orientation_feature_name}",
            "",
            "Future outcome counts:",
        ]
        total_pixels = max(1, result.shape[0] * result.shape[1])
        for label in SWITCHING_LABELS:
            count = result.counts.get(label, 0)
            lines.append(f"  {label}: {count} ({count / total_pixels:.1%})")

        lines.extend(
            [
                "",
                "Outcome thresholds:",
                f"  stable/high switching: {result.threshold_values['stable_switching_threshold']:.5g} / {result.threshold_values['high_switching_threshold']:.5g}",
                f"  net tau: {result.threshold_values['net_change_tau']:.5g}",
                "",
                "Interpretation:",
                result.interpretation,
                "",
                "Correlations with switching coefficient:",
            ]
        )
        for name, value in result.correlation_values.items():
            lines.append(f"  {name}: {value:.3f}" if np.isfinite(value) else f"  {name}: n/a")

        if self.state_prediction_selected_pixel is not None:
            x_index, y_index = self.state_prediction_selected_pixel
            switching = result.switching_result
            lines.extend(
                [
                    "",
                    f"Selected pixel: x={x_index}, y={y_index}",
                    f"Future outcome: {result.label_map[x_index, y_index]}",
                    f"Predictive score: {float(switching.switching_coefficient_map[x_index, y_index]):.6g}",
                    f"Net change: {float(switching.net_change_map[x_index, y_index]):+.6g}",
                    f"Initial I_rat: {float(result.initial_feature_maps['I_rat'][x_index, y_index]):.6g}",
                    f"Initial W_EF: {float(result.initial_feature_maps['W_EF'][x_index, y_index]):.6g}",
                    f"Initial E_LE: {float(result.initial_feature_maps['E_LE'][x_index, y_index]):.6g}",
                    f"Initial Gamma: {float(result.initial_feature_maps['Gamma_EDC'][x_index, y_index]):.6g}",
                    f"Initial E_p3/S_orient: {float(result.initial_feature_maps['S_orient'][x_index, y_index]):.6g}",
                    f"Distance to edge: {float(result.distance_maps['distance_to_edge'][x_index, y_index]):.6g}",
                    f"Distance to phase boundary: {float(result.distance_maps['distance_to_phase_boundary'][x_index, y_index]):.6g}",
                    f"Distance to structural boundary: {float(result.distance_maps['distance_to_structural_boundary'][x_index, y_index]):.6g}",
                    "",
                    "Per-file I_rat / W_EF:",
                ]
            )
            for state_index, state in enumerate(result.loaded_states):
                lines.append(
                    f"  {state_index}: {Path(state.file_path).name}  "
                    f"I_rat={float(switching.i_rat_maps[state_index][x_index, y_index]):.6g}, "
                    f"W_EF={float(switching.w_ef_maps[state_index][x_index, y_index]):.6g}"
                )
            if switching.delta_irat_maps:
                lines.append("")
                lines.append("Delta_Irat by transition:")
                for transition_index, delta_map in enumerate(switching.delta_irat_maps):
                    lines.append(
                        f"  {transition_index} -> {transition_index + 1}: "
                        f"{float(delta_map[x_index, y_index]):+.6g}"
                    )
            lines.extend(["", self._state_prediction_pixel_summary_sentence(x_index, y_index)])

        if result.notes:
            lines.extend(["", "Notes:"])
            lines.extend(f"  {note}" for note in result.notes)

        self._set_text_widget(self.state_prediction_summary_text, "\n".join(lines))

    def _state_prediction_pixel_summary_sentence(self, x_index: int, y_index: int) -> str:
        assert self.state_prediction_result is not None
        result = self.state_prediction_result
        label = str(result.label_map[x_index, y_index])
        irat = float(result.initial_feature_maps["I_rat"][x_index, y_index])
        phase_distance = float(result.distance_maps["distance_to_phase_boundary"][x_index, y_index])
        structural_distance = float(result.distance_maps["distance_to_structural_boundary"][x_index, y_index])
        boundary_hint = ""
        finite_distances = [
            value for value in (phase_distance, structural_distance)
            if np.isfinite(value)
        ]
        if finite_distances and min(finite_distances) <= 1.5:
            boundary_hint = " It is close to a phase or structural boundary."

        if label == "written / becomes metallic":
            return f"This pixel had initial I_rat={irat:.4g} and later became written.{boundary_hint}"
        if label == "erased / becomes less metallic":
            return f"This pixel had initial I_rat={irat:.4g} and later erased or became less metallic.{boundary_hint}"
        if label == "reversible / memory-like":
            return "This pixel has high switching coefficient but low net change, suggesting reversible or memory-like behavior." + boundary_hint
        if label == "stable / unchanged":
            return f"This pixel stayed comparatively stable from an initial I_rat={irat:.4g}.{boundary_hint}"
        return "This pixel does not fit a clean future-outcome rule with the current thresholds." + boundary_hint

    def _save_state_prediction_results(self) -> None:
        if self.state_prediction_result is None:
            messagebox.showinfo("No State Prediction", "Compute State Prediction before saving results.")
            return
        directory = filedialog.askdirectory(title="Choose output folder for State Prediction results")
        if not directory:
            return
        try:
            paths = export_state_prediction(self.state_prediction_result, directory)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.state_prediction_status_var.set(f"Saved State Prediction table to {paths['table']}")

    def _save_state_prediction_plot(self) -> None:
        if self.state_prediction_result is None:
            messagebox.showinfo("No State Prediction plot", "Compute State Prediction before saving a plot.")
            return
        path = filedialog.asksaveasfilename(
            title="Save State Prediction plot",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF document", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.state_prediction_figure.savefig(path, dpi=220)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.state_prediction_status_var.set(f"Saved State Prediction plot to {path}")

    def _add_initial_transition_files(self) -> None:
        selected = list(filedialog.askopenfilenames(title="Choose NetCDF files", filetypes=FILE_TYPES))
        if not selected:
            return
        new_paths = [str(Path(path).expanduser().resolve()) for path in selected]
        merged = self.initial_transition_file_paths + [
            path for path in new_paths if path not in self.initial_transition_file_paths
        ]
        self._set_initial_transition_files(merged)

    def _copy_analysis_files_to_initial_transition_panel(self) -> None:
        if not self.file_paths:
            messagebox.showinfo("No analysis files", "Add files to the Analysis panel first, or add files here directly.")
            return
        self._set_initial_transition_files(self.file_paths)

    def _remove_selected_initial_transition_files(self) -> None:
        selection = self._initial_transition_tree_selection_indices()
        if not selection:
            return
        paths = list(self.initial_transition_file_paths)
        for index in reversed(selection):
            del paths[index]
        self._set_initial_transition_files(paths)

    def _clear_initial_transition_files(self) -> None:
        self._set_initial_transition_files([])

    def _move_selected_initial_transition_file(self, direction: int) -> None:
        selection = self._initial_transition_tree_selection_indices()
        if len(selection) != 1:
            return
        index = selection[0]
        new_index = index + direction
        if not 0 <= new_index < len(self.initial_transition_file_paths):
            return
        reference_index = self._initial_transition_reference_index()
        reference_path = self.initial_transition_file_paths[reference_index] if self.initial_transition_file_paths else None
        paths = list(self.initial_transition_file_paths)
        paths[index], paths[new_index] = paths[new_index], paths[index]
        excluded = set()
        for old_index in self.initial_transition_excluded_indices:
            if old_index == index:
                excluded.add(new_index)
            elif old_index == new_index:
                excluded.add(index)
            else:
                excluded.add(old_index)
        self.initial_transition_file_paths = paths
        self.initial_transition_excluded_indices = excluded
        if reference_path in self.initial_transition_file_paths:
            self.initial_transition_reference_var.set(
                self._initial_transition_reference_value(self.initial_transition_file_paths.index(reference_path))
            )
        self.initial_transition_result = None
        self.initial_transition_selected_pixel = None
        self._sync_initial_transition_file_tree()
        self.initial_transition_file_tree.selection_set(str(new_index))
        self._render_initial_transition_placeholder()

    def _set_selected_initial_transition_reference(self) -> None:
        selection = self._initial_transition_tree_selection_indices()
        if not selection:
            return
        index = selection[0]
        self.initial_transition_reference_var.set(self._initial_transition_reference_value(index))
        self.initial_transition_excluded_indices.discard(index)
        self._sync_initial_transition_file_tree()

    def _toggle_selected_initial_transition_file(self) -> None:
        for index in self._initial_transition_tree_selection_indices():
            reference = self._initial_transition_reference_index()
            if index == reference:
                continue
            if index in self.initial_transition_excluded_indices:
                self.initial_transition_excluded_indices.remove(index)
            else:
                self.initial_transition_excluded_indices.add(index)
        self._sync_initial_transition_file_tree()

    def _initial_transition_tree_selection_indices(self) -> list[int]:
        if not hasattr(self, "initial_transition_file_tree"):
            return []
        indices: list[int] = []
        for item in self.initial_transition_file_tree.selection():
            try:
                indices.append(int(item))
            except ValueError:
                continue
        return sorted(index for index in indices if 0 <= index < len(self.initial_transition_file_paths))

    def _set_initial_transition_files(self, file_paths: list[str]) -> None:
        previous_reference_path = None
        if self.initial_transition_file_paths:
            reference = self._initial_transition_reference_index()
            if 0 <= reference < len(self.initial_transition_file_paths):
                previous_reference_path = self.initial_transition_file_paths[reference]
        self.initial_transition_file_paths = list(file_paths)
        self.initial_transition_excluded_indices = {
            index for index in self.initial_transition_excluded_indices if index < len(file_paths)
        }
        self.initial_transition_result = None
        self.initial_transition_selected_pixel = None
        self.mechanism_result = None
        self.mechanism_selected_pixel = None
        if file_paths:
            reference_index = 0
            if previous_reference_path in self.initial_transition_file_paths:
                reference_index = self.initial_transition_file_paths.index(previous_reference_path)
            self.initial_transition_reference_var.set(self._initial_transition_reference_value(reference_index))
            self.initial_transition_excluded_indices.discard(reference_index)
        else:
            self.initial_transition_reference_var.set("")
        self._sync_initial_transition_file_tree()
        self._render_initial_transition_placeholder()
        self._render_mechanism_placeholder()

    def _initial_transition_reference_value(self, index: int) -> str:
        if not 0 <= index < len(self.initial_transition_file_paths):
            return ""
        return f"{index}: {self._short_file_label(self.initial_transition_file_paths[index], 28)}"

    def _initial_transition_reference_index(self) -> int:
        text = self.initial_transition_reference_var.get().strip()
        try:
            index = int(text.split(":", 1)[0])
        except ValueError:
            index = 0
        if not self.initial_transition_file_paths:
            return 0
        return min(max(0, index), len(self.initial_transition_file_paths) - 1)

    def _sync_initial_transition_file_tree(self) -> None:
        if not hasattr(self, "initial_transition_file_tree"):
            return
        self.initial_transition_file_tree.delete(*self.initial_transition_file_tree.get_children())
        reference = self._initial_transition_reference_index()
        values = [self._initial_transition_reference_value(index) for index in range(len(self.initial_transition_file_paths))]
        self.initial_transition_reference_combo.configure(values=values)
        if values and self.initial_transition_reference_var.get() not in values:
            self.initial_transition_reference_var.set(values[0])
            reference = 0
        for index, path in enumerate(self.initial_transition_file_paths):
            if index in self.initial_transition_excluded_indices:
                role = "excluded"
                included = "no"
            elif index == reference:
                role = "initial reference"
                included = "yes"
            else:
                role = "transition state"
                included = "yes"
            note = "A0" if index == reference else ""
            self.initial_transition_file_tree.insert(
                "",
                tk.END,
                iid=str(index),
                values=(index, Path(path).name, role, included, note),
            )

    def _parse_initial_transition_parameters(self) -> InitialTransitionFeatureParameters:
        included_original_indices = [
            index for index in range(len(self.initial_transition_file_paths))
            if index not in self.initial_transition_excluded_indices
        ]
        reference_original = self._initial_transition_reference_index()
        if reference_original not in included_original_indices:
            included_original_indices.insert(0, reference_original)
        reference_included = included_original_indices.index(reference_original)
        try:
            params = InitialTransitionFeatureParameters(
                fermi_level_ev=float(self.initial_transition_parameter_vars["fermi_level_ev"].get()),
                ef_min_ev=float(self.initial_transition_parameter_vars["ef_min_ev"].get()),
                ef_max_ev=float(self.initial_transition_parameter_vars["ef_max_ev"].get()),
                feature_min_ev=float(self.initial_transition_parameter_vars["feature_min_ev"].get()),
                feature_max_ev=float(self.initial_transition_parameter_vars["feature_max_ev"].get()),
                asymmetry_split_ev=float(self.initial_transition_parameter_vars["asymmetry_split_ev"].get()),
                metallic_percentile=float(self.initial_transition_parameter_vars["metallic_percentile"].get()),
                erasure_percentile=float(self.initial_transition_parameter_vars["erasure_percentile"].get()),
                stable_percentile=float(self.initial_transition_parameter_vars["stable_percentile"].get()),
                transition_mode=self.initial_transition_mode_var.get(),
                reference_index=reference_included,
                normalization_mode=self.initial_transition_normalization_var.get(),
                allow_overlap=bool(self.initial_transition_allow_overlap_var.get()),
            )
        except ValueError as exc:
            raise ValueError(f"Could not parse Initial State Transition Feature controls: {exc}") from exc
        params.validate()
        return params

    def _initial_transition_included_files(self) -> list[str]:
        included = [
            path for index, path in enumerate(self.initial_transition_file_paths)
            if index not in self.initial_transition_excluded_indices
        ]
        reference = self._initial_transition_reference_index()
        if self.initial_transition_file_paths and self.initial_transition_file_paths[reference] not in included:
            included.insert(0, self.initial_transition_file_paths[reference])
        return included

    def _run_initial_transition_analysis(self) -> None:
        files = self._initial_transition_included_files()
        if len(files) < 2:
            messagebox.showerror("Missing files", "Please choose at least two included NetCDF files.")
            return
        try:
            params = self._parse_initial_transition_parameters()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return
        self.initial_transition_status_var.set("Computing initial-state transition features...")
        self._start_global_progress("Initial State Transition Features running...")
        self.root.update_idletasks()
        try:
            self.initial_transition_result = run_initial_transition_feature_analysis(files, params)
        except Exception as exc:
            self.initial_transition_result = None
            self.mechanism_result = None
            self.initial_transition_status_var.set("Initial State Transition Features failed.")
            self._finish_global_progress("Initial State Transition Features failed.", success=False)
            messagebox.showerror("Initial State Transition Features failed", str(exc))
            self._render_initial_transition_placeholder()
            self._render_mechanism_placeholder()
            return
        self.mechanism_result = None
        self.mechanism_selected_pixel = None
        self.initial_transition_selected_pixel = self._default_initial_transition_pixel()
        self._sync_initial_transition_transition_combo()
        self._refresh_initial_transition_views()
        self._render_mechanism_placeholder()
        result = self.initial_transition_result
        self.initial_transition_status_var.set(
            f"Computed {result.n_transitions} transition(s) over {result.shape[0]} x {result.shape[1]} pixels."
        )
        self._finish_global_progress("Initial State Transition Features complete.")

    def _sync_initial_transition_transition_combo(self) -> None:
        if not hasattr(self, "initial_transition_selected_transition_combo"):
            return
        result = self.initial_transition_result
        values = [] if result is None else [
            f"{transition.index}: {transition.name}"
            for transition in result.transitions
        ]
        self.initial_transition_selected_transition_combo.configure(values=values)
        if values and self.initial_transition_selected_transition_var.get() not in values:
            self.initial_transition_selected_transition_var.set(values[0])
        elif not values:
            self.initial_transition_selected_transition_var.set("")

    def _default_initial_transition_pixel(self) -> tuple[int, int]:
        result = self.initial_transition_result
        if result is None:
            return (0, 0)
        score = np.asarray(result.aggregate_maps["metallic_count"] + result.aggregate_maps["erased_count"], dtype=np.float32)
        if np.any(score > 0):
            return divmod(int(np.nanargmax(score)), score.shape[1])
        return (0, 0)

    def _render_initial_transition_placeholder(self) -> None:
        for attr, message in [
            ("initial_transition_aggregate_figure", "Add/order files, choose a reference, then compute aggregate transition maps."),
            ("initial_transition_precursor_figure", "Initial-state precursor masks will appear here."),
            ("initial_transition_diagnostics_figure", "Click a pixel after computing to inspect spectra and transition labels."),
            ("initial_transition_stats_figure", "Population average EDC/MDC curves will appear here."),
        ]:
            if not hasattr(self, attr):
                continue
            figure = getattr(self, attr)
            figure.clear()
            axis = figure.add_subplot(111)
            axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
            axis.set_axis_off()
        if hasattr(self, "initial_transition_aggregate_canvas"):
            self.initial_transition_aggregate_canvas.draw_idle()
            self.initial_transition_precursor_canvas.draw_idle()
            self.initial_transition_diagnostics_canvas.draw_idle()
            self.initial_transition_stats_canvas.draw_idle()
        if hasattr(self, "initial_transition_timeline_text"):
            self._set_text_widget(self.initial_transition_timeline_text, "")
        if hasattr(self, "initial_transition_stats_text"):
            self._set_text_widget(self.initial_transition_stats_text, "")

    def _refresh_initial_transition_views(self) -> None:
        if self.initial_transition_result is None:
            self._render_initial_transition_placeholder()
            return
        if self.initial_transition_selected_pixel is None:
            self.initial_transition_selected_pixel = self._default_initial_transition_pixel()
        self._refresh_initial_transition_aggregate_plot()
        self._refresh_initial_transition_precursor_plot()
        self._refresh_initial_transition_diagnostics_plot()
        self._refresh_initial_transition_population_plot()

    def _future_masks_from_initial_transition_controls(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.initial_transition_result is not None
        result = self.initial_transition_result
        try:
            metallic_min = max(0, int(float(self.initial_transition_parameter_vars["future_metallic_min_count"].get())))
            erased_min = max(0, int(float(self.initial_transition_parameter_vars["future_erased_min_count"].get())))
        except ValueError:
            metallic_min, erased_min = 1, 1
        metallic = result.aggregate_maps["metallic_count"] >= metallic_min
        erased = result.aggregate_maps["erased_count"] >= erased_min
        return metallic, erased, metallic & erased

    def _initial_transition_map_limits(self, data: np.ndarray, symmetric: bool = False) -> tuple[float | None, float | None]:
        finite = np.asarray(data, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return None, None
        if symmetric:
            limit = float(np.nanpercentile(np.abs(finite), 98))
            return -limit, limit
        low = float(np.nanpercentile(finite, 2))
        high = float(np.nanpercentile(finite, 98))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return None, None
        return low, high

    def _refresh_initial_transition_aggregate_plot(self) -> None:
        assert self.initial_transition_result is not None
        result = self.initial_transition_result
        self.initial_transition_aggregate_figure.clear()
        self.initial_transition_map_axes = []
        keys = [
            "metallic_count",
            "erased_count",
            "stable_count",
            "metallic_frequency",
            "erased_frequency",
            self.INITIAL_TRANSITION_AGGREGATE_MAP_OPTIONS.get(self.initial_transition_aggregate_map_var.get(), "max_metallicity_score"),
        ]
        titles = [
            "metallic_count",
            "erased_count",
            "stable_count",
            "metallic_frequency",
            "erased_frequency",
            self.initial_transition_aggregate_map_var.get(),
        ]
        axes = self.initial_transition_aggregate_figure.subplots(2, 3, squeeze=False)
        for axis, key, title in zip(axes.reshape(-1), keys, titles):
            data = np.asarray(result.aggregate_maps[key], dtype=np.float32)
            vmin, vmax = (None, None)
            if "score" in key:
                vmin, vmax = self._initial_transition_map_limits(data, symmetric=False)
            image = axis.imshow(data.T, origin="lower", cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
            axis.set_title(title)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            self._mark_initial_transition_selected_pixel(axis)
            self.initial_transition_map_axes.append(axis)
            self.initial_transition_aggregate_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        self.initial_transition_aggregate_canvas.draw_idle()

    def _refresh_initial_transition_precursor_plot(self) -> None:
        assert self.initial_transition_result is not None
        result = self.initial_transition_result
        metallic, erased, both = self._future_masks_from_initial_transition_controls()
        self.initial_transition_precursor_figure.clear()
        axes = self.initial_transition_precursor_figure.subplots(1, 3)
        for axis, title, mask, color in [
            (axes[0], "Initial State: Pixels That Later Became Metallic", metallic, "Reds"),
            (axes[1], "Initial State: Pixels That Later Erased", erased, "Blues"),
            (axes[2], "Initial State: Pixels That Later Became Both", both, "Purples"),
        ]:
            base = np.asarray(result.initial_near_ef_map, dtype=np.float32)
            vmin, vmax = self._initial_transition_map_limits(base)
            axis.imshow(base.T, origin="lower", cmap="gray", aspect="auto", vmin=vmin, vmax=vmax, alpha=0.55)
            overlay = np.where(mask, 1.0, np.nan)
            axis.imshow(overlay.T, origin="lower", cmap=color, aspect="auto", vmin=0, vmax=1, alpha=0.78)
            axis.set_title(title, fontsize=9)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            self._mark_initial_transition_selected_pixel(axis)
            self.initial_transition_map_axes.append(axis)
        self.initial_transition_precursor_canvas.draw_idle()

    def _selected_initial_transition_index(self) -> int:
        result = self.initial_transition_result
        if result is None or not result.transitions:
            return 0
        text = self.initial_transition_selected_transition_var.get().strip()
        try:
            index = int(text.split(":", 1)[0])
        except ValueError:
            index = 0
        return min(max(0, index), result.n_transitions - 1)

    def _initial_transition_spectrum_payload(self, state_index: int, x_index: int, y_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.initial_transition_result is not None
        state = self.initial_transition_result.loaded_states[state_index]
        data = np.asarray(state.data_array.values, dtype=np.float32)
        x_safe = min(max(0, int(x_index)), data.shape[0] - 1)
        y_safe = min(max(0, int(y_index)), data.shape[1] - 1)
        energy_axis = np.asarray(state.data_array.coords["eV"].values, dtype=np.float32)
        phi_axis = np.asarray(state.data_array.coords["phi"].values, dtype=np.float32)
        energy_order = np.argsort(energy_axis)
        phi_order = np.argsort(phi_axis)
        spectrum = np.asarray(data[x_safe, y_safe, :, :], dtype=np.float32)
        return spectrum[energy_order][:, phi_order], energy_axis[energy_order], phi_axis[phi_order]

    def _refresh_initial_transition_diagnostics_plot(self) -> None:
        assert self.initial_transition_result is not None
        result = self.initial_transition_result
        x_index, y_index = self.initial_transition_selected_pixel or self._default_initial_transition_pixel()
        transition = result.transitions[self._selected_initial_transition_index()]
        self.initial_transition_diagnostics_figure.clear()
        axes = self.initial_transition_diagnostics_figure.subplots(2, 3)
        spectrum, energy_axis, phi_axis = self._initial_transition_spectrum_payload(result.initial_reference_index, x_index, y_index)
        axes[0, 0].imshow(
            spectrum,
            origin="lower",
            aspect="auto",
            extent=[float(phi_axis[0]), float(phi_axis[-1]), float(energy_axis[0]), float(energy_axis[-1])],
            cmap="viridis",
        )
        axes[0, 0].set_title(f"Initial local spectrum\nx={x_index}, y={y_index}")
        axes[0, 0].set_xlabel("phi")
        axes[0, 0].set_ylabel("eV")
        for state_index, state in enumerate(result.loaded_states):
            local_spectrum, e_axis, p_axis = self._initial_transition_spectrum_payload(state_index, x_index, y_index)
            edc = np.trapezoid(local_spectrum, x=p_axis, axis=1) if p_axis.size > 1 else np.sum(local_spectrum, axis=1)
            scale = float(np.nanmax(np.abs(edc))) if np.any(np.isfinite(edc)) else 1.0
            axes[0, 1].plot(e_axis, edc / (scale if scale > 0 else 1.0), linewidth=1.0, alpha=0.7, label=str(state_index))
            ef_mask = (e_axis >= result.parameters.fermi_level_ev + result.parameters.ef_min_ev) & (e_axis <= result.parameters.fermi_level_ev + result.parameters.ef_max_ev)
            if not np.any(ef_mask):
                ef_mask[int(np.argmin(np.abs(e_axis - result.parameters.fermi_level_ev)))] = True
            mdc = np.trapezoid(local_spectrum[ef_mask, :], x=e_axis[ef_mask], axis=0) if int(np.count_nonzero(ef_mask)) > 1 else np.sum(local_spectrum[ef_mask, :], axis=0)
            scale_mdc = float(np.nanmax(np.abs(mdc))) if np.any(np.isfinite(mdc)) else 1.0
            axes[0, 2].plot(p_axis, mdc / (scale_mdc if scale_mdc > 0 else 1.0), linewidth=1.0, alpha=0.7, label=str(state_index))
        axes[0, 1].set_title("EDC across files")
        axes[0, 1].set_xlabel("eV")
        axes[0, 1].legend(fontsize=6)
        axes[0, 2].set_title("Near-EF MDC across files")
        axes[0, 2].set_xlabel("phi")
        axes[0, 2].legend(fontsize=6)
        before = transition.before_index
        after = transition.after_index
        spec_a, e_axis, p_axis = self._initial_transition_spectrum_payload(before, x_index, y_index)
        spec_b, _e, _p = self._initial_transition_spectrum_payload(after, x_index, y_index)
        diff = spec_b - spec_a
        limit = self._symmetric_change_limit(diff)
        axes[1, 0].imshow(diff, origin="lower", aspect="auto", extent=[float(p_axis[0]), float(p_axis[-1]), float(e_axis[0]), float(e_axis[-1])], cmap="coolwarm", vmin=-limit, vmax=limit)
        axes[1, 0].set_title(f"B - A spectrum\n{transition.index}: {transition.name}")
        edc_a = np.trapezoid(spec_a, x=p_axis, axis=1) if p_axis.size > 1 else np.sum(spec_a, axis=1)
        edc_b = np.trapezoid(spec_b, x=p_axis, axis=1) if p_axis.size > 1 else np.sum(spec_b, axis=1)
        axes[1, 1].plot(e_axis, edc_a, label="A")
        axes[1, 1].plot(e_axis, edc_b, label="B")
        axes[1, 1].plot(e_axis, edc_b - edc_a, label="B-A", color="#444444")
        axes[1, 1].set_title("Transition EDC")
        axes[1, 1].set_xlabel("eV")
        axes[1, 1].legend(fontsize=7)
        ef_mask = (e_axis >= result.parameters.fermi_level_ev + result.parameters.ef_min_ev) & (e_axis <= result.parameters.fermi_level_ev + result.parameters.ef_max_ev)
        mdc_a = np.trapezoid(spec_a[ef_mask, :], x=e_axis[ef_mask], axis=0) if int(np.count_nonzero(ef_mask)) > 1 else np.sum(spec_a[ef_mask, :], axis=0)
        mdc_b = np.trapezoid(spec_b[ef_mask, :], x=e_axis[ef_mask], axis=0) if int(np.count_nonzero(ef_mask)) > 1 else np.sum(spec_b[ef_mask, :], axis=0)
        axes[1, 2].plot(p_axis, mdc_a, label="A")
        axes[1, 2].plot(p_axis, mdc_b, label="B")
        axes[1, 2].plot(p_axis, mdc_b - mdc_a, label="B-A", color="#444444")
        axes[1, 2].set_title("Transition MDC")
        axes[1, 2].set_xlabel("phi")
        axes[1, 2].legend(fontsize=7)
        self.initial_transition_diagnostics_canvas.draw_idle()
        self._update_initial_transition_timeline_text(x_index, y_index)

    def _update_initial_transition_timeline_text(self, x_index: int, y_index: int) -> None:
        assert self.initial_transition_result is not None
        result = self.initial_transition_result
        lines = [
            f"Pixel x={x_index}, y={y_index}",
            f"metallic_count={int(result.aggregate_maps['metallic_count'][x_index, y_index])}, "
            f"erased_count={int(result.aggregate_maps['erased_count'][x_index, y_index])}, "
            f"stable_count={int(result.aggregate_maps['stable_count'][x_index, y_index])}",
            f"metallic_frequency={float(result.aggregate_maps['metallic_frequency'][x_index, y_index]):.3f}, "
            f"erased_frequency={float(result.aggregate_maps['erased_frequency'][x_index, y_index]):.3f}",
            "",
            "Transition timeline:",
        ]
        for transition in result.transitions:
            lines.append(
                f"{transition.index}: {transition.name} | "
                f"metallicity={float(transition.metallicity_score[x_index, y_index]):+.5g}, "
                f"erasure={float(transition.erasure_score[x_index, y_index]):+.5g}, "
                f"magnitude={float(transition.transition_magnitude[x_index, y_index]):.5g}, "
                f"metallic={'yes' if transition.metallic_mask[x_index, y_index] else 'no'}, "
                f"erased={'yes' if transition.erased_mask[x_index, y_index] else 'no'}, "
                f"stable={'yes' if transition.stable_mask[x_index, y_index] else 'no'}"
            )
        if result.notes:
            lines.extend(["", "Notes:"])
            lines.extend(f"- {note}" for note in result.notes[:4])
        self._set_text_widget(self.initial_transition_timeline_text, "\n".join(lines))

    def _refresh_initial_transition_population_plot(self) -> None:
        assert self.initial_transition_result is not None
        result = self.initial_transition_result
        self.initial_transition_stats_figure.clear()
        edc_axis, mdc_axis = self.initial_transition_stats_figure.subplots(1, 2)
        colors = {
            "future metallic": "#e6550d",
            "future erased": "#3182bd",
            "both metallic and erased": "#9467bd",
            "stable": "#2ca02c",
            "never switched": "#777777",
        }
        for group, edc in result.average_initial_edcs.items():
            if not np.any(np.isfinite(edc)):
                continue
            scale = float(np.nanmax(np.abs(edc)))
            edc_axis.plot(result.e_axis, edc / (scale if scale > 0 else 1.0), label=group, color=colors.get(group), linewidth=1.5)
        for group, mdc in result.average_initial_mdcs.items():
            if not np.any(np.isfinite(mdc)):
                continue
            scale = float(np.nanmax(np.abs(mdc)))
            mdc_axis.plot(result.phi_axis, mdc / (scale if scale > 0 else 1.0), label=group, color=colors.get(group), linewidth=1.5)
        edc_axis.set_title("Mean initial EDC by future behavior")
        edc_axis.set_xlabel("eV")
        edc_axis.legend(fontsize=7)
        mdc_axis.set_title("Mean initial near-EF MDC by future behavior")
        mdc_axis.set_xlabel("phi")
        mdc_axis.legend(fontsize=7)
        self.initial_transition_stats_canvas.draw_idle()
        lines = ["Group statistics:"]
        for row in result.group_statistics:
            lines.append(
                f"{row['group']}: n={row['number_of_pixels']}, "
                f"mean near_EF={row.get('mean_near_EF_intensity_A0', float('nan')):.5g}, "
                f"mean feature={row.get('mean_feature_window_intensity_A0', float('nan')):.5g}, "
                f"mean peak width={row.get('mean_edc_peak_width_A0', float('nan')):.5g}"
            )
        self._set_text_widget(self.initial_transition_stats_text, "\n".join(lines))

    def _mark_initial_transition_selected_pixel(self, axis: matplotlib.axes.Axes) -> None:
        if self.initial_transition_selected_pixel is None:
            return
        x_index, y_index = self.initial_transition_selected_pixel
        axis.scatter([x_index], [y_index], s=90, facecolors="none", edgecolors="white", linewidths=1.8)
        axis.scatter([x_index], [y_index], s=18, c="black")

    def _on_initial_transition_plot_click(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        if self.initial_transition_result is None or event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        x_index = int(round(event.xdata))
        y_index = int(round(event.ydata))
        x_size, y_size = self.initial_transition_result.shape
        if 0 <= x_index < x_size and 0 <= y_index < y_size:
            self.initial_transition_selected_pixel = (x_index, y_index)
            self._refresh_initial_transition_views()

    def _save_initial_transition_results(self) -> None:
        if self.initial_transition_result is None:
            messagebox.showinfo("No results", "Compute Initial State Transition Features before exporting.")
            return
        directory = filedialog.askdirectory(title="Choose output folder for Initial State Transition Features")
        if not directory:
            return
        try:
            paths = export_initial_transition_feature_analysis(self.initial_transition_result, directory)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.initial_transition_status_var.set(f"Exported transition feature tables to {paths['metrics_table']}")

    def _add_mechanism_files(self) -> None:
        selected = list(filedialog.askopenfilenames(title="Choose NetCDF files", filetypes=FILE_TYPES))
        if not selected:
            return
        new_paths = [str(Path(path).expanduser().resolve()) for path in selected]
        merged = self.mechanism_file_paths + [path for path in new_paths if path not in self.mechanism_file_paths]
        self._set_mechanism_files(merged)

    def _copy_analysis_files_to_mechanism(self) -> None:
        if not self.file_paths:
            messagebox.showinfo("No analysis files", "Add files to the Analysis panel first, or add files here directly.")
            return
        self._set_mechanism_files(self.file_paths)

    def _copy_initial_transition_files_to_mechanism(self) -> None:
        files = self._initial_transition_included_files()
        if len(files) < 2:
            messagebox.showinfo(
                "No initial-state files",
                "Add files to Initial State Transition Features first, or add files here directly.",
            )
            return
        self._set_mechanism_files(files)

    def _remove_selected_mechanism_file(self) -> None:
        if not hasattr(self, "mechanism_file_listbox"):
            return
        selected = list(self.mechanism_file_listbox.curselection())
        if not selected:
            return
        paths = list(self.mechanism_file_paths)
        for index in reversed(selected):
            if 0 <= index < len(paths):
                del paths[index]
        self._set_mechanism_files(paths)

    def _move_selected_mechanism_file(self, direction: int) -> None:
        if not hasattr(self, "mechanism_file_listbox"):
            return
        selected = list(self.mechanism_file_listbox.curselection())
        if len(selected) != 1:
            return
        index = selected[0]
        new_index = index + direction
        if not 0 <= new_index < len(self.mechanism_file_paths):
            return
        paths = list(self.mechanism_file_paths)
        paths[index], paths[new_index] = paths[new_index], paths[index]
        self._set_mechanism_files(paths)
        self.mechanism_file_listbox.selection_set(new_index)
        self.mechanism_file_listbox.see(new_index)

    def _clear_mechanism_files(self) -> None:
        self._set_mechanism_files([])

    def _set_mechanism_files(self, file_paths: list[str]) -> None:
        self.mechanism_file_paths = [str(Path(path).expanduser().resolve()) for path in file_paths]
        self.mechanism_result = None
        self.mechanism_selected_pixel = None
        self._sync_mechanism_file_listbox()
        self._render_mechanism_placeholder()
        if self.mechanism_file_paths:
            self.mechanism_status_var.set(f"Loaded {len(self.mechanism_file_paths)} file(s) for diagnostics.")
        else:
            self.mechanism_status_var.set("Add files here or reuse Initial State Transition Features, then compute diagnostics.")

    def _sync_mechanism_file_listbox(self) -> None:
        if not hasattr(self, "mechanism_file_listbox"):
            return
        self.mechanism_file_listbox.delete(0, tk.END)
        for index, path in enumerate(self.mechanism_file_paths):
            self.mechanism_file_listbox.insert(tk.END, f"{index}: {Path(path).name}")

    def _mechanism_input_files(self) -> list[str]:
        if len(self.mechanism_file_paths) >= 2:
            return list(self.mechanism_file_paths)
        initial_files = self._initial_transition_included_files()
        if len(initial_files) >= 2:
            return initial_files
        if len(self.file_paths) >= 2:
            return list(self.file_paths)
        return []

    def _parse_mechanism_parameters(self) -> SwitchingMechanismParameters:
        try:
            threshold_values = tuple(
                float(part.strip())
                for part in self.mechanism_parameter_vars["threshold_sweep_percentiles"].get().split(",")
                if part.strip()
            )
            transition_params = (
                self.initial_transition_result.parameters
                if self.initial_transition_result is not None
                else self._parse_initial_transition_parameters()
            )
            params = SwitchingMechanismParameters(
                transition_parameters=transition_params,
                future_metallic_min_count=max(0, int(float(self.mechanism_parameter_vars["future_metallic_min_count"].get()))),
                future_erased_min_count=max(0, int(float(self.mechanism_parameter_vars["future_erased_min_count"].get()))),
                future_metallic_min_frequency=float(self.mechanism_parameter_vars["future_metallic_min_frequency"].get()),
                future_erased_min_frequency=float(self.mechanism_parameter_vars["future_erased_min_frequency"].get()),
                edc_normalization=self.mechanism_edc_normalization_var.get(),
                boundary_smooth_sigma=float(self.mechanism_parameter_vars["boundary_smooth_sigma"].get()),
                boundary_percentile=float(self.mechanism_parameter_vars["boundary_percentile"].get()),
                component_min_size=max(0, int(float(self.mechanism_parameter_vars["component_min_size"].get()))),
                threshold_sweep_percentiles=threshold_values,
                negative_control_min_ev=float(self.mechanism_parameter_vars["negative_control_min_ev"].get()),
                negative_control_max_ev=float(self.mechanism_parameter_vars["negative_control_max_ev"].get()),
                permutation_count=max(0, int(float(self.mechanism_parameter_vars["permutation_count"].get()))),
            )
        except ValueError as exc:
            raise ValueError(f"Could not parse Switching Mechanism Diagnostics controls: {exc}") from exc
        params.validate()
        return params

    def _use_initial_transition_for_mechanism(self) -> None:
        self._run_mechanism_diagnostics(source="initial")

    def _run_mechanism_diagnostics(self, source: str = "files") -> None:
        if self.mechanism_worker_thread is not None and self.mechanism_worker_thread.is_alive():
            messagebox.showinfo("Diagnostics running", "Switching Mechanism Diagnostics is already running.")
            return
        try:
            params = self._parse_mechanism_parameters()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return
        transition_result: InitialTransitionFeatureResult | None = None
        files: list[str] | None = None
        if source == "initial":
            transition_result = self.initial_transition_result
            if transition_result is None:
                files = self._initial_transition_included_files()
                if len(files) < 2:
                    files = self._mechanism_input_files()
        else:
            files = self._mechanism_input_files()
            if len(files) < 2:
                messagebox.showerror(
                    "Missing files",
                    "Add at least two NetCDF files in this view, Initial State Transition Features, or Analysis.",
                )
                return
            if files and params.transition_parameters.reference_index >= len(files):
                params.transition_parameters.reference_index = 0
        self._start_mechanism_progress(
            "Computing switching mechanism diagnostics... loading files, computing transition metrics, and building diagnostic plots."
        )
        self.mechanism_worker_queue = queue.Queue()

        def worker() -> None:
            try:
                result = run_switching_mechanism_diagnostics(
                    file_paths=files,
                    transition_result=transition_result,
                    parameters=params,
                )
            except Exception as exc:  # pragma: no cover - handled by UI polling
                self.mechanism_worker_queue.put(("error", exc))
                return
            self.mechanism_worker_queue.put(("result", result))

        self.mechanism_worker_thread = threading.Thread(target=worker, daemon=True)
        self.mechanism_worker_thread.start()
        self.root.after(120, self._poll_mechanism_worker)

    def _start_mechanism_progress(self, message: str) -> None:
        self.mechanism_status_var.set(message)
        self._start_global_progress("Switching Mechanism Diagnostics running...")
        if hasattr(self, "mechanism_progress"):
            self.mechanism_progress.start(12)
        for attr in ("mechanism_compute_button", "mechanism_initial_compute_button", "mechanism_export_button"):
            if hasattr(self, attr):
                getattr(self, attr).configure(state=tk.DISABLED)
        self.root.update_idletasks()

    def _stop_mechanism_progress(self) -> None:
        if hasattr(self, "mechanism_progress"):
            self.mechanism_progress.stop()
        for attr in ("mechanism_compute_button", "mechanism_initial_compute_button", "mechanism_export_button"):
            if hasattr(self, attr):
                getattr(self, attr).configure(state=tk.NORMAL)

    def _poll_mechanism_worker(self) -> None:
        if self.mechanism_worker_queue is None:
            return
        try:
            kind, payload = self.mechanism_worker_queue.get_nowait()
        except queue.Empty:
            self.root.after(120, self._poll_mechanism_worker)
            return
        self._stop_mechanism_progress()
        if kind == "error":
            self.mechanism_result = None
            self.mechanism_status_var.set("Switching Mechanism Diagnostics failed.")
            self._finish_global_progress("Switching Mechanism Diagnostics failed.", success=False)
            if self.runner_running_panel == "Switching Mechanism Diagnostics":
                self._mark_runner_failed("Switching Mechanism Diagnostics")
            messagebox.showerror("Switching Mechanism Diagnostics failed", str(payload))
            self._render_mechanism_placeholder()
            return
        self.mechanism_result = payload  # type: ignore[assignment]
        self.initial_transition_result = self.mechanism_result.transition_result
        self._sync_initial_transition_transition_combo()
        self.mechanism_selected_pixel = self.initial_transition_selected_pixel or self._default_mechanism_pixel()
        self._sync_mechanism_transition_combo()
        self._refresh_mechanism_views()
        verdict = self.mechanism_result.summary_verdict
        self.mechanism_status_var.set(
            "Computed mechanism diagnostics | "
            f"spectral={verdict['spectral_evidence_label']}, "
            f"spatial={verdict['spatial_evidence_label']}, "
            f"history={verdict['transition_history_evidence_label']}, "
            f"artifact risk={verdict['artifact_risk_label']}"
        )
        self._finish_global_progress("Switching Mechanism Diagnostics complete.")
        if self.runner_running_panel == "Switching Mechanism Diagnostics":
            self._mark_runner_complete("Switching Mechanism Diagnostics")

    def _sync_mechanism_transition_combo(self) -> None:
        if not hasattr(self, "mechanism_selected_transition_combo"):
            return
        result = self.mechanism_result
        values = [] if result is None else [f"{transition.index}: {transition.name}" for transition in result.transitions]
        self.mechanism_selected_transition_combo.configure(values=values)
        if values and self.mechanism_selected_transition_var.get() not in values:
            self.mechanism_selected_transition_var.set(values[0])
        elif not values:
            self.mechanism_selected_transition_var.set("")

    def _selected_mechanism_transition_index(self) -> int:
        if self.mechanism_result is None or not self.mechanism_result.transitions:
            return 0
        text = self.mechanism_selected_transition_var.get().strip()
        try:
            index = int(text.split(":", 1)[0])
        except ValueError:
            index = 0
        return min(max(0, index), len(self.mechanism_result.transitions) - 1)

    def _default_mechanism_pixel(self) -> tuple[int, int]:
        result = self.mechanism_result
        if result is None:
            return (0, 0)
        activity = np.asarray(result.transition_result.aggregate_maps["metallic_count"] + result.transition_result.aggregate_maps["erased_count"], dtype=np.float32)
        if activity.size and np.any(activity > 0):
            return divmod(int(np.nanargmax(activity)), activity.shape[1])
        return (0, 0)

    def _render_mechanism_placeholder(self) -> None:
        for attr, message in [
            ("mechanism_spectral_figure", "Run diagnostics to compare initial spectra for future-switching and stable pixels."),
            ("mechanism_spatial_figure", "Spatial maps, boundary distances, and clustering diagnostics will appear here."),
            ("mechanism_history_figure", "Transition-history timelines and selected-pixel switching traces will appear here."),
            ("mechanism_artifact_figure", "Drift, normalization, threshold, edge, and control-window checks will appear here."),
            ("mechanism_summary_figure", "Summary evidence and artifact-risk cards will appear here."),
        ]:
            if not hasattr(self, attr):
                continue
            figure = getattr(self, attr)
            figure.clear()
            axis = figure.add_subplot(111)
            axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
            axis.set_axis_off()
        if hasattr(self, "mechanism_spectral_canvas"):
            self.mechanism_spectral_canvas.draw_idle()
            self._update_mechanism_spectral_scroll_region()
            self.mechanism_spatial_canvas.draw_idle()
            self.mechanism_history_canvas.draw_idle()
            self.mechanism_artifact_canvas.draw_idle()
            self.mechanism_summary_canvas.draw_idle()
        for attr in (
            "mechanism_spectral_text",
            "mechanism_spatial_text",
            "mechanism_history_text",
            "mechanism_artifact_text",
            "mechanism_summary_text",
        ):
            if hasattr(self, attr):
                self._set_text_widget(getattr(self, attr), "")

    def _refresh_mechanism_views(self) -> None:
        if self.mechanism_result is None:
            self._render_mechanism_placeholder()
            return
        if self.mechanism_selected_pixel is None:
            self.mechanism_selected_pixel = self._default_mechanism_pixel()
        self._refresh_mechanism_spectral_plot()
        self._refresh_mechanism_spatial_plot()
        self._refresh_mechanism_history_plot()
        self._refresh_mechanism_artifact_plot()
        self._refresh_mechanism_summary_plot()

    def _mechanism_map_limits(self, data: np.ndarray, symmetric: bool = False) -> tuple[float | None, float | None]:
        finite = np.asarray(data, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return None, None
        if symmetric:
            limit = float(np.nanpercentile(np.abs(finite), 98))
            if not np.isfinite(limit) or limit <= 0:
                return None, None
            return -limit, limit
        low = float(np.nanpercentile(finite, 2))
        high = float(np.nanpercentile(finite, 98))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return None, None
        return low, high

    def _refresh_mechanism_spectral_plot(self) -> None:
        assert self.mechanism_result is not None
        result = self.mechanism_result
        self.mechanism_spectral_figure.clear()
        colors = self._mechanism_group_colors()
        grid = self.mechanism_spectral_figure.add_gridspec(3, 6)
        edc_axis = self.mechanism_spectral_figure.add_subplot(grid[0, 0:2])
        diff_axis = self.mechanism_spectral_figure.add_subplot(grid[0, 2:4])
        mdc_axis = self.mechanism_spectral_figure.add_subplot(grid[0, 4])
        mdc_diff_axis = self.mechanism_spectral_figure.add_subplot(grid[0, 5])
        effect_axis = self.mechanism_spectral_figure.add_subplot(grid[1:, 5])
        for group, edc in result.group_edcs.items():
            if not np.any(np.isfinite(edc)):
                continue
            count = int(np.count_nonzero(result.group_masks[group]))
            edc_axis.plot(result.e_axis, edc, label=f"{group} (N={count})", color=colors.get(group), linewidth=1.4)
            sem = result.group_edc_sem.get(group)
            if sem is not None and np.any(np.isfinite(sem)):
                edc_axis.fill_between(result.e_axis, edc - sem, edc + sem, color=colors.get(group), alpha=0.12)
            mdc = result.group_mdcs.get(group)
            if mdc is not None and np.any(np.isfinite(mdc)):
                mdc_axis.plot(result.phi_axis, mdc, label=f"{group} (N={count})", color=colors.get(group), linewidth=1.3)
                mdc_sem = result.group_mdc_sem.get(group)
                if mdc_sem is not None and np.any(np.isfinite(mdc_sem)):
                    mdc_axis.fill_between(result.phi_axis, mdc - mdc_sem, mdc + mdc_sem, color=colors.get(group), alpha=0.12)
        stable = result.group_edcs.get("stable")
        stable_mdc = result.group_mdcs.get("stable")
        for group in ("future metallic", "future erased", "both metallic and erased", "never switched"):
            edc = result.group_edcs.get(group)
            if edc is not None and stable is not None and np.any(np.isfinite(edc)) and np.any(np.isfinite(stable)):
                diff_axis.plot(result.e_axis, edc - stable, label=f"{group} - stable", color=colors.get(group), linewidth=1.4)
            mdc = result.group_mdcs.get(group)
            if mdc is not None and stable_mdc is not None and np.any(np.isfinite(mdc)) and np.any(np.isfinite(stable_mdc)):
                mdc_diff_axis.plot(result.phi_axis, mdc - stable_mdc, label=f"{group} - stable", color=colors.get(group), linewidth=1.3)
        edc_axis.set_title("Mean Initial EDC by Future Outcome")
        edc_axis.set_xlabel("eV")
        edc_axis.legend(fontsize=6)
        diff_axis.axhline(0, color="#666666", linewidth=0.8)
        diff_axis.set_title("Difference From Stable EDC")
        diff_axis.set_xlabel("eV")
        diff_axis.legend(fontsize=6)
        mdc_axis.set_title("Mean near-EF MDC")
        mdc_axis.set_xlabel("phi")
        mdc_axis.legend(fontsize=5)
        mdc_diff_axis.axhline(0, color="#666666", linewidth=0.8)
        mdc_diff_axis.set_title("MDC - stable")
        mdc_diff_axis.set_xlabel("phi")
        top_effects = sorted(
            [row for row in result.spectral_effect_rows if np.isfinite(row.get("cohens_d", np.nan))],
            key=lambda row: abs(float(row["cohens_d"])),
            reverse=True,
        )[:6]
        labels = [f"{row['group'].split()[1] if ' ' in row['group'] else row['group']}\n{row['feature'].replace('_A0', '')[:12]}" for row in top_effects]
        values = [float(row["cohens_d"]) for row in top_effects]
        effect_axis.barh(range(len(values)), values, color="#6b8fbf")
        effect_axis.set_yticks(range(len(values)), labels, fontsize=6)
        effect_axis.axvline(0, color="#444444", linewidth=0.8)
        effect_axis.set_title("Top Cohen's d")
        spectra_groups = list(INITIAL_TRANSITION_GROUPS)
        all_spectra = [result.group_spectra[group] for group in spectra_groups if np.any(np.isfinite(result.group_spectra[group]))]
        vmin, vmax = self._mechanism_map_limits(np.stack(all_spectra), symmetric=False) if all_spectra else (None, None)
        stable_spectrum = result.group_spectra.get("stable")
        for col, group in enumerate(spectra_groups):
            axis = self.mechanism_spectral_figure.add_subplot(grid[1, col])
            image = axis.imshow(
                result.group_spectra[group],
                origin="lower",
                aspect="auto",
                extent=[float(result.phi_axis[0]), float(result.phi_axis[-1]), float(result.e_axis[0]), float(result.e_axis[-1])],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            axis.set_title(group, fontsize=8)
            axis.set_xlabel("phi")
            if col == 0:
                axis.set_ylabel("eV")
            self.mechanism_spectral_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
            diff_img = result.group_spectra[group] - stable_spectrum if stable_spectrum is not None else np.full_like(result.group_spectra[group], np.nan)
            diff_ax = self.mechanism_spectral_figure.add_subplot(grid[2, col])
            dvmin, dvmax = self._mechanism_map_limits(diff_img, symmetric=True)
            diff_plot = diff_ax.imshow(
                diff_img,
                origin="lower",
                aspect="auto",
                extent=[float(result.phi_axis[0]), float(result.phi_axis[-1]), float(result.e_axis[0]), float(result.e_axis[-1])],
                cmap="coolwarm",
                vmin=dvmin,
                vmax=dvmax,
            )
            diff_ax.set_title(f"{group} - stable", fontsize=8)
            diff_ax.set_xlabel("phi")
            if col == 0:
                diff_ax.set_ylabel("eV")
            self.mechanism_spectral_figure.colorbar(diff_plot, ax=diff_ax, fraction=0.046, pad=0.02)
        self.mechanism_spectral_canvas.draw_idle()
        self._update_mechanism_spectral_scroll_region()
        verdict = result.summary_verdict
        lines = [
            f"Spectral evidence: {verdict['spectral_evidence_label']} ({verdict['spectral_evidence_score']:.2f})",
            f"EDC normalization: {result.parameters.edc_normalization}",
            "Top spectral feature differences:",
        ]
        for row in verdict["top_spectral_features"][:5]:
            lines.append(
                f"- {row['group']} vs stable | {row['feature']}: d={row['cohens_d']:.3g}, "
                f"diff={row['difference']:.3g}, p={row['mannwhitney_p']:.3g}"
            )
        self._set_text_widget(self.mechanism_spectral_text, "\n".join(lines))

    def _refresh_mechanism_spatial_plot(self) -> None:
        assert self.mechanism_result is not None
        result = self.mechanism_result
        self.mechanism_spatial_figure.clear()
        self.mechanism_map_axes = []
        axes = self.mechanism_spatial_figure.subplots(2, 4, squeeze=False)
        base = result.spatial_feature_maps["initial_near_EF"]
        maps = [
            ("Initial near-EF", base, "viridis", False),
            ("Metallic frequency", result.transition_result.aggregate_maps["metallic_frequency"], "Reds", False),
            ("Erased frequency", result.transition_result.aggregate_maps["erased_frequency"], "Blues", False),
            ("Stable frequency", result.transition_result.aggregate_maps["stable_frequency"], "Greens", False),
            ("Boundary mask", result.spatial_feature_maps["domain_boundary_mask"], "gray", False),
            ("Distance to boundary", result.spatial_feature_maps["distance_to_domain_boundary"], "magma", False),
            ("Gradient magnitude", result.spatial_feature_maps["local_intensity_gradient"], "plasma", False),
            ("Texture / contrast", result.spatial_feature_maps["local_contrast_texture"], "cividis", False),
        ]
        for axis, (title, data, cmap, symmetric) in zip(axes.reshape(-1), maps):
            vmin, vmax = self._mechanism_map_limits(data, symmetric=symmetric)
            image = axis.imshow(np.asarray(data).T, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            self._mark_mechanism_selected_pixel(axis)
            self.mechanism_map_axes.append(axis)
            self.mechanism_spatial_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        self.mechanism_spatial_canvas.draw_idle()
        lines = [
            f"Spatial evidence: {result.summary_verdict['spatial_evidence_label']} ({result.summary_verdict['spatial_evidence_score']:.2f})",
            "Connected component diagnostics:",
        ]
        for row in result.connected_component_rows:
            lines.append(
                f"- {row['group']}: pixels={row['pixel_count']}, components={row['connected_component_count']}, "
                f"largest={row['largest_connected_component_size']}, nn={row['mean_nearest_neighbor_distance']:.3g}"
            )
        lines.append("Top spatial feature differences:")
        for row in result.summary_verdict["top_spatial_features"][:4]:
            lines.append(f"- {row['group']} | {row['feature']}: d={row['cohens_d']:.3g}")
        self._set_text_widget(self.mechanism_spatial_text, "\n".join(lines))

    def _refresh_mechanism_history_plot(self) -> None:
        assert self.mechanism_result is not None
        result = self.mechanism_result
        transition_result = result.transition_result
        x_index, y_index = self.mechanism_selected_pixel or self._default_mechanism_pixel()
        self.mechanism_history_figure.clear()
        axes = self.mechanism_history_figure.subplots(3, 4, squeeze=False)
        history_maps = [
            ("First metallic transition", result.transition_history_maps["first_metallic_transition"]),
            ("First erased transition", result.transition_history_maps["first_erased_transition"]),
            ("Last metallic transition", result.transition_history_maps["last_metallic_transition"]),
            ("Last erased transition", result.transition_history_maps["last_erased_transition"]),
        ]
        for axis, (title, data) in zip(axes[0], history_maps):
            image = axis.imshow(np.asarray(data).T, origin="lower", aspect="auto", cmap="tab20")
            axis.set_title(title)
            self._mark_mechanism_selected_pixel(axis)
            self.mechanism_map_axes.append(axis)
            self.mechanism_history_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        transition_indices = [int(row["transition_index"]) for row in result.transition_level_rows]
        axes[1, 0].plot(transition_indices, [row["metallic_pixels"] for row in result.transition_level_rows], label="metallic", color="#e6550d")
        axes[1, 0].plot(transition_indices, [row["erased_pixels"] for row in result.transition_level_rows], label="erased", color="#3182bd")
        axes[1, 0].plot(transition_indices, [row["stable_pixels"] for row in result.transition_level_rows], label="stable", color="#2ca02c")
        axes[1, 0].set_title("Transition pixel counts")
        axes[1, 0].set_xlabel("transition")
        axes[1, 0].legend(fontsize=7)
        axes[1, 1].plot(transition_indices, [row["mean_metallicity_score"] for row in result.transition_level_rows], label="metallicity", color="#e6550d")
        axes[1, 1].plot(transition_indices, [row["mean_erasure_score"] for row in result.transition_level_rows], label="erasure", color="#3182bd")
        axes[1, 1].set_title("Mean transition scores")
        axes[1, 1].legend(fontsize=7)
        timeline = self._mechanism_selected_pixel_timeline(x_index, y_index)
        axes[1, 2].plot(transition_indices, [row["metallicity_score"] for row in timeline], marker="o", label="metallicity", color="#e6550d")
        axes[1, 2].plot(transition_indices, [row["erasure_score"] for row in timeline], marker="o", label="erasure", color="#3182bd")
        axes[1, 2].plot(transition_indices, [row["transition_magnitude"] for row in timeline], marker="o", label="magnitude", color="#555555")
        axes[1, 2].set_title(f"Selected pixel x={x_index}, y={y_index}")
        axes[1, 2].legend(fontsize=7)
        for state_index, state in enumerate(transition_result.loaded_states):
            spectrum, energy_axis, phi_axis = self._mechanism_spectrum_payload(state_index, x_index, y_index)
            edc = np.trapezoid(spectrum, x=phi_axis, axis=1) if phi_axis.size > 1 else np.sum(spectrum, axis=1)
            scale = float(np.nanmax(np.abs(edc))) if np.any(np.isfinite(edc)) else 1.0
            axes[1, 3].plot(energy_axis, edc / (scale if scale > 0 else 1.0), linewidth=1.0, alpha=0.7, label=str(state_index))
        axes[1, 3].set_title("Selected pixel EDC stack")
        axes[1, 3].set_xlabel("eV")
        axes[1, 3].legend(fontsize=6)
        selected_transition = result.transitions[self._selected_mechanism_transition_index()]
        spec_a, e_axis, p_axis = self._mechanism_spectrum_payload(selected_transition.before_index, x_index, y_index)
        spec_b, _e, _p = self._mechanism_spectrum_payload(selected_transition.after_index, x_index, y_index)
        diff = spec_b - spec_a
        limit = self._symmetric_change_limit(diff)
        image = axes[2, 0].imshow(
            diff,
            origin="lower",
            aspect="auto",
            extent=[float(p_axis[0]), float(p_axis[-1]), float(e_axis[0]), float(e_axis[-1])],
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axes[2, 0].set_title(f"Selected transition B-A\n{selected_transition.index}: {selected_transition.name}", fontsize=8)
        axes[2, 0].set_xlabel("phi")
        axes[2, 0].set_ylabel("eV")
        self.mechanism_history_figure.colorbar(image, ax=axes[2, 0], fraction=0.046, pad=0.03)
        edc_a = np.trapezoid(spec_a, x=p_axis, axis=1) if p_axis.size > 1 else np.sum(spec_a, axis=1)
        edc_b = np.trapezoid(spec_b, x=p_axis, axis=1) if p_axis.size > 1 else np.sum(spec_b, axis=1)
        axes[2, 1].plot(e_axis, edc_a, label="A")
        axes[2, 1].plot(e_axis, edc_b, label="B")
        axes[2, 1].plot(e_axis, edc_b - edc_a, label="B-A", color="#444444")
        axes[2, 1].set_title("Selected transition EDC")
        axes[2, 1].set_xlabel("eV")
        axes[2, 1].legend(fontsize=7)
        ef_mask = (
            (e_axis >= transition_result.parameters.fermi_level_ev + transition_result.parameters.ef_min_ev)
            & (e_axis <= transition_result.parameters.fermi_level_ev + transition_result.parameters.ef_max_ev)
        )
        if not np.any(ef_mask):
            ef_mask[int(np.argmin(np.abs(e_axis - transition_result.parameters.fermi_level_ev)))] = True
        mdc_a = np.trapezoid(spec_a[ef_mask, :], x=e_axis[ef_mask], axis=0) if int(np.count_nonzero(ef_mask)) > 1 else np.sum(spec_a[ef_mask, :], axis=0)
        mdc_b = np.trapezoid(spec_b[ef_mask, :], x=e_axis[ef_mask], axis=0) if int(np.count_nonzero(ef_mask)) > 1 else np.sum(spec_b[ef_mask, :], axis=0)
        axes[2, 2].plot(p_axis, mdc_a, label="A")
        axes[2, 2].plot(p_axis, mdc_b, label="B")
        axes[2, 2].plot(p_axis, mdc_b - mdc_a, label="B-A", color="#444444")
        axes[2, 2].set_title("Selected transition MDC")
        axes[2, 2].set_xlabel("phi")
        axes[2, 2].legend(fontsize=7)
        persistence = result.transition_history_maps["switching_persistence"]
        persistence_plot = axes[2, 3].imshow(persistence.T, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
        axes[2, 3].set_title("Switching persistence")
        self._mark_mechanism_selected_pixel(axes[2, 3])
        self.mechanism_history_figure.colorbar(persistence_plot, ax=axes[2, 3], fraction=0.046, pad=0.03)
        self.mechanism_history_canvas.draw_idle()
        lines = [
            f"Transition-history evidence: {result.summary_verdict['transition_history_evidence_label']} ({result.summary_verdict['transition_history_evidence_score']:.2f})",
            f"Selected pixel x={x_index}, y={y_index}",
            f"Selected transition: {selected_transition.index}: {selected_transition.name}",
            "Pixel transition timeline:",
        ]
        for row in timeline:
            labels = []
            if row["metallic"]:
                labels.append("metallic")
            if row["erased"]:
                labels.append("erased")
            if row["stable"]:
                labels.append("stable")
            lines.append(
                f"- {row['transition_index']}: {row['from_file']} -> {row['to_file']} | "
                f"M={row['metallicity_score']:+.4g}, E={row['erasure_score']:+.4g}, "
                f"mag={row['transition_magnitude']:.4g}, labels={','.join(labels) or 'normal'}"
            )
        self._set_text_widget(self.mechanism_history_text, "\n".join(lines))

    def _refresh_mechanism_artifact_plot(self) -> None:
        assert self.mechanism_result is not None
        result = self.mechanism_result
        self.mechanism_artifact_figure.clear()
        axes = self.mechanism_artifact_figure.subplots(2, 3, squeeze=False)
        indices = [row["transition_index"] for row in result.transition_level_rows]
        axes[0, 0].plot(indices, [row["drift_dx"] for row in result.transition_level_rows], marker="o", label="dx")
        axes[0, 0].plot(indices, [row["drift_dy"] for row in result.transition_level_rows], marker="o", label="dy")
        axes[0, 0].set_title("Estimated drift")
        axes[0, 0].legend(fontsize=7)
        axes[0, 1].plot(indices, [row["alignment_score"] for row in result.transition_level_rows], marker="o", color="#6b8fbf")
        axes[0, 1].set_title("Alignment score")
        axes[0, 2].plot([row["file_index"] for row in result.file_intensity_rows], [row["total_intensity"] for row in result.file_intensity_rows], marker="o", label="total")
        axes[0, 2].plot([row["file_index"] for row in result.file_intensity_rows], [row["near_EF_total_intensity"] for row in result.file_intensity_rows], marker="o", label="near EF")
        axes[0, 2].set_title("File intensity stats")
        axes[0, 2].legend(fontsize=7)
        threshold_values = sorted({row["threshold_percentile"] for row in result.threshold_sensitivity_rows})
        metallic_counts = [
            sum(row["metallic_pixels"] for row in result.threshold_sensitivity_rows if row["threshold_percentile"] == threshold)
            for threshold in threshold_values
        ]
        erased_counts = [
            sum(row["erased_pixels"] for row in result.threshold_sensitivity_rows if row["threshold_percentile"] == threshold)
            for threshold in threshold_values
        ]
        axes[1, 0].plot(threshold_values, metallic_counts, marker="o", label="metallic", color="#e6550d")
        axes[1, 0].plot(threshold_values, erased_counts, marker="o", label="erased", color="#3182bd")
        axes[1, 0].set_title("Threshold sensitivity")
        axes[1, 0].set_xlabel("percentile")
        axes[1, 0].legend(fontsize=7)
        for axis, key, title, cmap in [
            (axes[1, 1], "metallic_threshold_robustness", "Metallic robustness", "Reds"),
            (axes[1, 2], "erased_threshold_robustness", "Erased robustness", "Blues"),
        ]:
            data = result.threshold_robustness_maps[key]
            image = axis.imshow(data.T, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1)
            axis.set_title(title)
            self._mark_mechanism_selected_pixel(axis)
            self.mechanism_artifact_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        self.mechanism_artifact_canvas.draw_idle()
        lines = [
            f"Artifact risk: {result.summary_verdict['artifact_risk_label']} ({result.summary_verdict['artifact_risk_score']:.2f})",
            "Top artifact checks:",
        ]
        for row in result.summary_verdict["top_artifact_warnings"][:6]:
            lines.append(f"- {row['check']}: risk={row['risk_score']:.2f}, value={row['value']:.4g} | {row['message']}")
        self._set_text_widget(self.mechanism_artifact_text, "\n".join(lines))

    def _refresh_mechanism_summary_plot(self) -> None:
        assert self.mechanism_result is not None
        result = self.mechanism_result
        verdict = result.summary_verdict
        self.mechanism_summary_figure.clear()
        axis = self.mechanism_summary_figure.add_subplot(111)
        labels = ["Spectral", "Spatial", "History", "Artifact risk"]
        values = [
            verdict["spectral_evidence_score"],
            verdict["spatial_evidence_score"],
            verdict["transition_history_evidence_score"],
            verdict["artifact_risk_score"],
        ]
        colors = ["#6b8fbf", "#2ca02c", "#9467bd", "#d62728"]
        axis.bar(labels, values, color=colors)
        axis.set_ylim(0, 1)
        axis.set_ylabel("heuristic score")
        axis.set_title("Switching Mechanism Diagnostic Summary")
        for index, value in enumerate(values):
            axis.text(index, value + 0.03, f"{value:.2f}", ha="center")
        self.mechanism_summary_canvas.draw_idle()
        lines = [
            "Summary Verdict",
            f"Spectral evidence: {verdict['spectral_evidence_label']} ({verdict['spectral_evidence_score']:.2f})",
            f"Spatial evidence: {verdict['spatial_evidence_label']} ({verdict['spatial_evidence_score']:.2f})",
            f"Transition-history evidence: {verdict['transition_history_evidence_label']} ({verdict['transition_history_evidence_score']:.2f})",
            f"Artifact risk: {verdict['artifact_risk_label']} ({verdict['artifact_risk_score']:.2f})",
            "",
            verdict["interpretation"],
            "",
            "Current settings:",
            f"- transition mode: {verdict['transition_mode']}",
            f"- normalization mode: {verdict['current_normalization_mode']}",
            f"- thresholds: {verdict['current_thresholds']}",
            f"- files: {len(verdict['file_sequence'])}",
            "",
            "Top caveats:",
        ]
        for warning in verdict["top_artifact_warnings"][:3]:
            lines.append(f"- {warning['check']}: {warning['message']}")
        self._set_text_widget(self.mechanism_summary_text, "\n".join(lines))

    def _mechanism_group_colors(self) -> dict[str, str]:
        return {
            "future metallic": "#e6550d",
            "future erased": "#3182bd",
            "both metallic and erased": "#9467bd",
            "stable": "#2ca02c",
            "never switched": "#777777",
        }

    def _mechanism_spectrum_payload(self, state_index: int, x_index: int, y_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.mechanism_result is not None
        state = self.mechanism_result.transition_result.loaded_states[state_index]
        data = np.asarray(state.data_array.values, dtype=np.float32)
        x_safe = min(max(0, int(x_index)), data.shape[0] - 1)
        y_safe = min(max(0, int(y_index)), data.shape[1] - 1)
        energy_axis = np.asarray(state.data_array.coords["eV"].values, dtype=np.float32)
        phi_axis = np.asarray(state.data_array.coords["phi"].values, dtype=np.float32)
        energy_order = np.argsort(energy_axis)
        phi_order = np.argsort(phi_axis)
        spectrum = np.asarray(data[x_safe, y_safe, :, :], dtype=np.float32)
        return spectrum[energy_order][:, phi_order], energy_axis[energy_order], phi_axis[phi_order]

    def _mechanism_selected_pixel_timeline(self, x_index: int, y_index: int) -> list[dict[str, object]]:
        assert self.mechanism_result is not None
        result = self.mechanism_result
        transition_result = result.transition_result
        rows: list[dict[str, object]] = []
        for transition in result.transitions:
            rows.append(
                {
                    "transition_index": transition.index,
                    "from_file": transition_result.loaded_states[transition.before_index].name,
                    "to_file": transition_result.loaded_states[transition.after_index].name,
                    "metallicity_score": float(transition.metallicity_score[x_index, y_index]),
                    "erasure_score": float(transition.erasure_score[x_index, y_index]),
                    "transition_magnitude": float(transition.transition_magnitude[x_index, y_index]),
                    "metallic": bool(transition.metallic_mask[x_index, y_index]),
                    "erased": bool(transition.erased_mask[x_index, y_index]),
                    "stable": bool(transition.stable_mask[x_index, y_index]),
                }
            )
        return rows

    def _mark_mechanism_selected_pixel(self, axis: matplotlib.axes.Axes) -> None:
        if self.mechanism_selected_pixel is None:
            return
        x_index, y_index = self.mechanism_selected_pixel
        axis.scatter([x_index], [y_index], s=90, facecolors="none", edgecolors="white", linewidths=1.8)
        axis.scatter([x_index], [y_index], s=18, c="black")

    def _on_mechanism_plot_click(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        if self.mechanism_result is None or event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        x_index = int(round(event.xdata))
        y_index = int(round(event.ydata))
        x_size, y_size = self.mechanism_result.shape
        if 0 <= x_index < x_size and 0 <= y_index < y_size:
            self.mechanism_selected_pixel = (x_index, y_index)
            self.initial_transition_selected_pixel = (x_index, y_index)
            self.selected_pixel = (x_index, y_index)
            self.sequence_selected_pixel = (x_index, y_index)
            self._refresh_mechanism_views()
            if self.initial_transition_result is not None:
                self._refresh_initial_transition_views()

    def _save_mechanism_results(self) -> None:
        if self.mechanism_result is None:
            messagebox.showinfo("No diagnostics", "Compute Switching Mechanism Diagnostics before exporting.")
            return
        directory = filedialog.askdirectory(title="Choose output folder for Switching Mechanism Diagnostics")
        if not directory:
            return
        try:
            paths = export_switching_mechanism_diagnostics(
                self.mechanism_result,
                directory,
                selected_pixel=self.mechanism_selected_pixel,
            )
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.mechanism_status_var.set(f"Exported mechanism diagnostics to {paths['summary']}")

    def _add_transition_outcome_files(self) -> None:
        selected = list(filedialog.askopenfilenames(title="Choose NetCDF files", filetypes=FILE_TYPES))
        if not selected:
            return

        new_paths = [str(Path(path).expanduser().resolve()) for path in selected]
        merged = self.transition_outcome_file_paths + [
            path for path in new_paths if path not in self.transition_outcome_file_paths
        ]
        self._set_transition_outcome_files(merged)

    def _copy_analysis_files_to_transition_outcome_panel(self) -> None:
        if not self.file_paths:
            messagebox.showinfo("No analysis files", "Add files to the Analysis panel first, or add files here directly.")
            return
        self._set_transition_outcome_files(self.file_paths)
        self.top_notebook.select(8)

    def _remove_selected_transition_outcome_files(self) -> None:
        selection = list(self.transition_outcome_file_listbox.curselection())
        if not selection:
            return
        updated_files = list(self.transition_outcome_file_paths)
        for index in reversed(selection):
            del updated_files[index]
        self._set_transition_outcome_files(updated_files)

    def _move_selected_transition_outcome_file(self, direction: int) -> None:
        selection = self.transition_outcome_file_listbox.curselection()
        if len(selection) != 1:
            return

        index = selection[0]
        new_index = index + direction
        if not 0 <= new_index < len(self.transition_outcome_file_paths):
            return

        updated_files = list(self.transition_outcome_file_paths)
        updated_files[index], updated_files[new_index] = updated_files[new_index], updated_files[index]
        self._set_transition_outcome_files(updated_files)
        self.transition_outcome_file_listbox.selection_set(new_index)

    def _clear_transition_outcome_files(self) -> None:
        self._set_transition_outcome_files([])

    def _set_transition_outcome_files(self, file_paths: list[str]) -> None:
        self.transition_outcome_file_paths = list(file_paths)
        self._clear_transition_outcome_results()
        self._sync_transition_outcome_file_listbox()
        self._sync_transition_outcome_inspector_file_options()
        self._render_transition_outcome_placeholder()

    def _clear_transition_outcome_results(self) -> None:
        self.transition_outcome_result = None
        self.transition_outcome_selected_pixel = None
        self.transition_outcome_hover_pixel = None
        self.transition_outcome_focused_transition = None
        self.transition_outcome_map_axes = []
        self.transition_outcome_axis_to_transition = {}
        self.transition_outcome_axis_to_file = {}
        self.transition_outcome_axis_limits = {}

    def _sync_transition_outcome_file_listbox(self) -> None:
        self.transition_outcome_file_listbox.delete(0, tk.END)
        for index, path in enumerate(self.transition_outcome_file_paths):
            label = "initial" if index == 0 else f"after pulse {index}"
            self.transition_outcome_file_listbox.insert(tk.END, f"{index + 1}. {Path(path).name} ({label})")

    def _sync_transition_outcome_inspector_file_options(self) -> None:
        if not hasattr(self, "transition_outcome_inspector_file_combo"):
            return
        paths = (
            self.transition_outcome_result.file_paths
            if self.transition_outcome_result is not None
            else self.transition_outcome_file_paths
        )
        values = [
            f"{index}: {self._short_file_label(path, 26)}"
            for index, path in enumerate(paths)
        ]
        self.transition_outcome_inspector_file_combo.configure(values=values)
        current = self.transition_outcome_inspector_file_var.get()
        if values and current not in values:
            self.transition_outcome_inspector_file_var.set(values[0])
        elif not values:
            self.transition_outcome_inspector_file_var.set("")

    def _on_transition_outcome_display_changed(self, _event: tk.Event | None = None) -> None:
        if self.transition_outcome_result is not None:
            self._refresh_transition_outcome_views()

    def _on_transition_outcome_inspector_file_changed(self, _event: tk.Event | None = None) -> None:
        if self.transition_outcome_result is None:
            return
        if self.transition_outcome_focused_transition is not None:
            self._refresh_transition_outcome_views()
        else:
            self._refresh_transition_outcome_inspector_panel()
            self._update_transition_outcome_summary_text()

    def _parse_transition_outcome_parameters(self) -> TransitionOutcomeParameters:
        try:
            params = TransitionOutcomeParameters(
                fermi_level_ev=float(self.transition_outcome_parameter_vars["fermi_level_ev"].get()),
                ef_min_ev=float(self.transition_outcome_parameter_vars["ef_min_ev"].get()),
                ef_max_ev=float(self.transition_outcome_parameter_vars["ef_max_ev"].get()),
                lhb_center_ev=float(self.transition_outcome_parameter_vars["lhb_center_ev"].get()),
                lhb_halfwidth_ev=float(self.transition_outcome_parameter_vars["lhb_halfwidth_ev"].get()),
                smooth_sigma=float(self.transition_outcome_parameter_vars["smooth_sigma"].get()),
                user_min_tau=float(self.transition_outcome_parameter_vars["user_min_tau"].get()),
                strong_tau_multiplier=float(self.transition_outcome_parameter_vars["strong_tau_multiplier"].get()),
                use_relative_delta=bool(self.transition_outcome_parameter_vars["use_relative_delta"].get()),
                low_signal_quantile=float(self.transition_outcome_parameter_vars["low_signal_quantile"].get()),
                lhb_min_quantile=float(self.transition_outcome_parameter_vars["lhb_min_quantile"].get()),
            )
        except ValueError as exc:
            raise ValueError(f"Could not parse the Transition Outcome controls: {exc}") from exc
        params.validate()
        return params

    def _transition_outcome_pulse_labels(self) -> list[str]:
        raw = self.transition_outcome_pulse_labels_var.get().strip()
        if not raw:
            return []
        return [piece.strip() for piece in raw.split(",")]

    def _transition_outcome_color_limit(self, values: np.ndarray) -> float:
        text = self.transition_outcome_parameter_vars["color_limit"].get().strip()
        if text:
            try:
                value = abs(float(text))
                if np.isfinite(value) and value > 0:
                    return value
            except ValueError:
                pass
        return self._symmetric_change_limit(values)

    def _run_transition_outcome_maps(self) -> None:
        if len(self.transition_outcome_file_paths) < 2:
            messagebox.showerror("Missing files", "Please choose at least two chronological NetCDF files.")
            return

        try:
            params = self._parse_transition_outcome_parameters()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self.transition_outcome_status_var.set("Computing transition-level writing and erasing maps...")
        self._start_global_progress("Transition Outcome Maps running...")
        self.root.update_idletasks()

        try:
            self.transition_outcome_result = run_transition_outcome_maps(
                self.transition_outcome_file_paths,
                params,
                pulse_labels=self._transition_outcome_pulse_labels(),
            )
            self.transition_outcome_selected_pixel = self._default_transition_outcome_pixel()
            self.transition_outcome_hover_pixel = None
            self.transition_outcome_focused_transition = None
            self._sync_transition_outcome_inspector_file_options()
        except Exception as exc:
            self._clear_transition_outcome_results()
            self.transition_outcome_status_var.set("Transition Outcome Maps failed.")
            self._finish_global_progress("Transition Outcome Maps failed.", success=False)
            messagebox.showerror("Transition Outcome Maps failed", str(exc))
            self._render_transition_outcome_placeholder()
            return

        self._refresh_transition_outcome_views()
        shape = self.transition_outcome_result.shape
        alignment_suffix = f" {self.transition_outcome_result.notes[0]}" if self.transition_outcome_result.notes else ""
        self.transition_outcome_status_var.set(
            f"Computed {self.transition_outcome_result.n_transitions} transition map(s) as {shape[0]} x {shape[1]} pixels."
            f"{alignment_suffix}"
        )
        self._finish_global_progress("Transition Outcome Maps complete.")

    def _default_transition_outcome_pixel(self) -> tuple[int, int]:
        if self.transition_outcome_result is None:
            return (0, 0)
        score = np.asarray(self.transition_outcome_result.activity_count_map, dtype=np.float32)
        if np.any(np.isfinite(score)) and float(np.nanmax(score)) > 0:
            flat_index = int(np.nanargmax(score))
            return divmod(flat_index, score.shape[1])
        net = np.abs(np.asarray(self.transition_outcome_result.net_sequence_change_map, dtype=np.float32))
        if np.any(np.isfinite(net)):
            flat_index = int(np.nanargmax(net))
            return divmod(flat_index, net.shape[1])
        return (0, 0)

    def _clear_transition_outcome_focus(self) -> None:
        if self.transition_outcome_result is None:
            return
        self.transition_outcome_focused_transition = None
        self._refresh_transition_outcome_views()

    def _refresh_transition_outcome_views(self) -> None:
        if self.transition_outcome_result is None:
            self._render_transition_outcome_placeholder()
            return
        if self.transition_outcome_selected_pixel is None:
            self.transition_outcome_selected_pixel = self._default_transition_outcome_pixel()
        self._refresh_transition_outcome_plot()
        self._refresh_transition_outcome_inspector_panel()
        self._update_transition_outcome_summary_text()

    def _render_transition_outcome_placeholder(self) -> None:
        if not hasattr(self, "transition_outcome_figure"):
            return

        self.transition_outcome_figure.clear()
        axis = self.transition_outcome_figure.add_subplot(111)
        message = (
            "Ready to map transition-level writing and erasing.\nAdd/order files, optionally enter pulse labels, then compute."
            if self.transition_outcome_file_paths
            else "Add chronological NetCDF files to build Transition Outcome Maps."
        )
        axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=13)
        axis.set_axis_off()
        self.transition_outcome_canvas.draw_idle()
        self._update_transition_outcome_plot_scroll_region()
        self.transition_outcome_map_axes = []
        self.transition_outcome_axis_to_transition = {}
        self.transition_outcome_axis_to_file = {}
        self.transition_outcome_axis_limits = {}
        if hasattr(self, "transition_outcome_summary_text"):
            self._set_text_widget(self.transition_outcome_summary_text, "")
        if hasattr(self, "transition_outcome_inspector_figure"):
            self.transition_outcome_inspector_figure.clear()
            inspector_axis = self.transition_outcome_inspector_figure.add_subplot(111)
            inspector_axis.text(0.5, 0.5, "Click a map pixel to inspect one file at that point.", ha="center", va="center")
            inspector_axis.set_axis_off()
            self.transition_outcome_inspector_canvas.draw_idle()
        if not self.transition_outcome_file_paths:
            self.transition_outcome_status_var.set(
                "Add chronological NetCDF files to map written and erased pixels for each transition."
            )

    def _refresh_transition_outcome_plot(self) -> None:
        assert self.transition_outcome_result is not None
        if self.transition_outcome_focused_transition is None:
            self._refresh_transition_outcome_overview_plot()
        else:
            self._refresh_transition_outcome_focus_plot(self.transition_outcome_focused_transition)

    def _set_transition_figure_size(self, width: float, height: float) -> None:
        dpi = float(self.transition_outcome_figure.dpi or 100.0)
        self.transition_outcome_figure.set_size_inches(width, height, forward=True)
        width_px = max(1, int(width * dpi))
        height_px = max(1, int(height * dpi))
        self.transition_outcome_plot_canvas_frame.configure(width=width_px, height=height_px)
        self.transition_outcome_canvas.get_tk_widget().configure(width=width_px, height=height_px)
        self.transition_outcome_plot_scroll_canvas.itemconfigure(
            self.transition_outcome_plot_scroll_window,
            width=width_px,
            height=height_px,
        )
        self.transition_outcome_plot_scroll_canvas.configure(scrollregion=(0, 0, width_px, height_px))
        self.transition_outcome_plot_scroll_canvas.yview_moveto(0.0)

    def _refresh_transition_outcome_overview_plot(self) -> None:
        assert self.transition_outcome_result is not None
        result = self.transition_outcome_result
        self.transition_outcome_figure.clear()
        self.transition_outcome_map_axes = []
        self.transition_outcome_axis_to_transition = {}
        self.transition_outcome_axis_to_file = {}
        self.transition_outcome_axis_limits = {}

        map_key = self._transition_outcome_map_key()
        if map_key == "global_summary":
            self._refresh_transition_outcome_global_summary_plot()
            return

        if map_key.startswith("file_"):
            panel_count = result.n_states
            width = max(13.0, panel_count * 5.7 + 0.6)
            self._set_transition_figure_size(width, 6.4)
            grid = self.transition_outcome_figure.add_gridspec(
                1,
                panel_count,
                left=0.045,
                right=0.985,
                bottom=0.12,
                top=0.82,
                wspace=0.46,
            )
            limits = self._transition_outcome_file_map_limits(map_key)
            for index, state in enumerate(result.loaded_states):
                axis = self.transition_outcome_figure.add_subplot(grid[0, index])
                title = f"File {index}: {self._short_file_label(state.file_path, 34)}"
                data, cmap, vmin, vmax, colorbar_label = self._transition_outcome_file_map_payload(
                    map_key,
                    index,
                    limits,
                )
                self._draw_transition_outcome_map_panel(
                    axis,
                    data,
                    title,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    colorbar_label=colorbar_label,
                )
                self.transition_outcome_axis_to_file[axis] = index
            self.transition_outcome_canvas.draw_idle()
            self._update_transition_outcome_plot_scroll_region()
            return

        panel_count = result.n_transitions
        width = max(13.0, panel_count * 5.9 + 0.6)
        self._set_transition_figure_size(width, 6.5)
        grid = self.transition_outcome_figure.add_gridspec(
            1,
            panel_count,
            left=0.045,
            right=0.985,
            bottom=0.12,
            top=0.82,
            wspace=0.5,
        )
        for transition in result.transitions:
            axis = self.transition_outcome_figure.add_subplot(grid[0, transition.index])
            data, cmap, vmin, vmax, norm, colorbar_label = self._transition_outcome_transition_map_payload(
                transition,
                map_key,
            )
            title = self._transition_outcome_panel_title(transition, map_key)
            self._draw_transition_outcome_map_panel(
                axis,
                data,
                title,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                colorbar_label=colorbar_label,
            )
            self.transition_outcome_axis_to_transition[axis] = transition.index

        self.transition_outcome_canvas.draw_idle()
        self._update_transition_outcome_plot_scroll_region()

    def _transition_outcome_map_key(self) -> str:
        return self.TRANSITION_OUTCOME_MAP_OPTIONS.get(
            self.transition_outcome_map_var.get(),
            "transition_label",
        )

    def _refresh_transition_outcome_global_summary_plot(self) -> None:
        assert self.transition_outcome_result is not None
        result = self.transition_outcome_result
        maps = self._transition_outcome_global_map_payloads()
        self._set_transition_figure_size(max(13.0, len(maps) * 5.7 + 0.6), 6.4)
        grid = self.transition_outcome_figure.add_gridspec(
            1,
            len(maps),
            left=0.045,
            right=0.985,
            bottom=0.12,
            top=0.82,
            wspace=0.46,
        )
        for column, (title, data, cmap, vmin, vmax, colorbar_label) in enumerate(maps):
            axis = self.transition_outcome_figure.add_subplot(grid[0, column])
            if title == "Net sequence change":
                limit = self._symmetric_change_limit(data)
                vmin, vmax = -limit, limit
            self._draw_transition_outcome_map_panel(
                axis,
                data,
                title,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                colorbar_label=colorbar_label,
            )
        self.transition_outcome_canvas.draw_idle()
        self._update_transition_outcome_plot_scroll_region()

    def _transition_outcome_global_map_payloads(
        self,
    ) -> list[tuple[str, np.ndarray, str, float | None, float | None, str]]:
        assert self.transition_outcome_result is not None
        result = self.transition_outcome_result
        return [
            ("Write count", result.write_count_map, "Reds", 0.0, None, "transitions"),
            ("Erase count", result.erase_count_map, "Blues", 0.0, None, "transitions"),
            ("Activity count", result.activity_count_map, "viridis", 0.0, None, "transitions"),
            ("Repeated switching", result.repeated_switching_map, "Purples", 0.0, 1.0, "flag"),
            ("Net sequence change", result.net_sequence_change_map, "coolwarm", None, None, "Delta I_rat"),
        ]

    def _transition_outcome_file_map_limits(self, map_key: str) -> tuple[float, float]:
        assert self.transition_outcome_result is not None
        result = self.transition_outcome_result
        if map_key == "file_wef":
            return self._switching_feature_limits(np.stack(result.w_ef_maps, axis=0))
        if map_key == "file_total":
            return self._switching_feature_limits(np.stack(result.total_maps, axis=0))
        return self._switching_feature_limits(np.stack(result.i_rat_maps, axis=0))

    def _transition_outcome_file_map_payload(
        self,
        map_key: str,
        file_index: int,
        limits: tuple[float, float],
    ) -> tuple[np.ndarray, str, float, float, str]:
        assert self.transition_outcome_result is not None
        result = self.transition_outcome_result
        if map_key == "file_wef":
            return result.w_ef_maps[file_index], "magma", limits[0], limits[1], "W_EF"
        if map_key == "file_total":
            return result.total_maps[file_index], "inferno", limits[0], limits[1], "T"
        return result.i_rat_maps[file_index], "viridis", limits[0], limits[1], "I_rat"

    def _transition_outcome_transition_map_payload(
        self,
        transition: object,
        map_key: str,
    ) -> tuple[np.ndarray, object, float | None, float | None, object | None, str]:
        assert self.transition_outcome_result is not None
        result = self.transition_outcome_result
        if map_key == "transition_label":
            cmap, norm = self._transition_label_cmap_norm()
            return transition.code_map, cmap, None, None, norm, "state"
        if map_key == "relative_delta_irat":
            values = np.stack([item.relative_delta_irat_map for item in result.transitions], axis=0)
            limit = self._transition_outcome_color_limit(values)
            return transition.relative_delta_irat_map, "coolwarm", -limit, limit, None, "relative Delta I_rat"
        if map_key == "abs_delta_irat":
            limits = self._switching_feature_limits(
                np.stack([item.abs_delta_irat_map for item in result.transitions], axis=0)
            )
            return transition.abs_delta_irat_map, "plasma", limits[0], limits[1], None, "abs Delta I_rat"
        if map_key == "written_mask":
            data = np.where(np.isin(transition.code_map, [2, 4]), 1.0, 0.0)
            data = np.where(transition.valid_mask, data, np.nan)
            return data, "Reds", 0.0, 1.0, None, "written"
        if map_key == "erased_mask":
            data = np.where(np.isin(transition.code_map, [3, 5]), 1.0, 0.0)
            data = np.where(transition.valid_mask, data, np.nan)
            return data, "Blues", 0.0, 1.0, None, "erased"
        values = np.stack([item.delta_irat_map for item in result.transitions], axis=0)
        limit = self._transition_outcome_color_limit(values)
        return transition.delta_irat_map, "coolwarm", -limit, limit, None, "Delta I_rat"

    def _transition_outcome_panel_title(self, transition: object, map_key: str) -> str:
        pulse = f" ({transition.pulse_label})" if transition.pulse_label else ""
        metric_name = {
            "transition_label": "transition label",
            "delta_irat": "Delta I_rat",
            "relative_delta_irat": "relative Delta I_rat",
            "abs_delta_irat": "abs Delta I_rat",
            "written_mask": "written mask",
            "erased_mask": "erased mask",
        }.get(map_key, "transition map")
        return (
            f"Transition {transition.before_index} -> {transition.after_index}{pulse}: {metric_name}\n"
            f"{self._transition_counts_title(transition).splitlines()[-1]} | "
            f"tau={transition.tau:.4g}, strong={transition.strong_tau:.4g}"
        )

    def _draw_transition_outcome_map_panel(
        self,
        axis: matplotlib.axes.Axes,
        data: np.ndarray,
        title: str,
        *,
        cmap: object,
        vmin: float | None = None,
        vmax: float | None = None,
        norm: object | None = None,
        colorbar_label: str = "",
    ) -> None:
        image = axis.imshow(
            np.asarray(data).T,
            origin="lower",
            cmap=cmap,
            norm=norm,
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal", adjustable="box")
        try:
            axis.set_box_aspect(1)
        except AttributeError:
            pass
        axis.set_navigate(False)
        self._mark_transition_outcome_selected_pixel(axis)
        self.transition_outcome_map_axes.append(axis)
        self.transition_outcome_axis_limits[axis] = (axis.get_xlim(), axis.get_ylim())
        cbar = self.transition_outcome_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=8)
        if norm is not None:
            cbar.set_ticks(np.arange(len(TRANSITION_OUTCOME_LABELS)))
            cbar.ax.set_yticklabels([self._short_transition_label(label) for label in TRANSITION_OUTCOME_LABELS], fontsize=7)

    def _restore_transition_outcome_axis_limits(self) -> None:
        if not getattr(self, "transition_outcome_axis_limits", None):
            return
        changed = False
        for axis, (x_limits, y_limits) in list(self.transition_outcome_axis_limits.items()):
            if axis.figure is not self.transition_outcome_figure:
                continue
            if axis.get_xlim() != x_limits:
                axis.set_xlim(x_limits)
                changed = True
            if axis.get_ylim() != y_limits:
                axis.set_ylim(y_limits)
                changed = True
        if changed:
            self.transition_outcome_canvas.draw_idle()

    def _on_transition_outcome_mpl_scroll(self, _event: matplotlib.backend_bases.MouseEvent) -> None:
        self._restore_transition_outcome_axis_limits()
        self.root.after_idle(self._restore_transition_outcome_axis_limits)

    def _draw_transition_global_maps(self, grid: matplotlib.gridspec.GridSpec, row: int, cols: int) -> None:
        assert self.transition_outcome_result is not None
        result = self.transition_outcome_result
        maps = [
            ("Write count", result.write_count_map, "Reds", None, None),
            ("Erase count", result.erase_count_map, "Blues", None, None),
            ("Activity count", result.activity_count_map, "viridis", None, None),
            ("Repeated switching", result.repeated_switching_map, "Purples", 0, 1),
            ("Net sequence change", result.net_sequence_change_map, "coolwarm", None, None),
        ]
        for column, (title, data, cmap, vmin, vmax) in enumerate(maps[:cols]):
            axis = self.transition_outcome_figure.add_subplot(grid[row, column])
            if title == "Net sequence change":
                limit = self._symmetric_change_limit(data)
                vmin, vmax = -limit, limit
            image = axis.imshow(np.asarray(data).T, origin="lower", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
            axis.set_title(title, fontsize=9)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            self._mark_transition_outcome_selected_pixel(axis)
            self.transition_outcome_map_axes.append(axis)
            self.transition_outcome_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    def _refresh_transition_outcome_focus_plot(self, transition_index: int) -> None:
        assert self.transition_outcome_result is not None
        result = self.transition_outcome_result
        transition = result.transitions[min(max(0, transition_index), result.n_transitions - 1)]
        map_key = self._transition_outcome_map_key()
        if map_key.startswith("file_") or map_key == "global_summary":
            self.transition_outcome_focused_transition = None
            self._refresh_transition_outcome_overview_plot()
            return
        self.transition_outcome_figure.clear()
        self.transition_outcome_map_axes = []
        self.transition_outcome_axis_to_transition = {}
        self.transition_outcome_axis_to_file = {}
        self.transition_outcome_axis_limits = {}
        self._set_transition_figure_size(11.8, 13.4)
        grid = self.transition_outcome_figure.add_gridspec(
            3,
            2,
            left=0.075,
            right=0.965,
            bottom=0.055,
            top=0.945,
            wspace=0.32,
            hspace=0.48,
        )

        map_axis = self.transition_outcome_figure.add_subplot(grid[0, 0])
        data, cmap, vmin, vmax, norm, colorbar_label = self._transition_outcome_transition_map_payload(
            transition,
            map_key,
        )
        self._draw_transition_outcome_map_panel(
            map_axis,
            data,
            self._transition_outcome_panel_title(transition, map_key),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            colorbar_label=colorbar_label,
        )
        self.transition_outcome_axis_to_transition[map_axis] = transition.index

        hist_axis = self.transition_outcome_figure.add_subplot(grid[0, 1])
        values = transition.metric_delta_map[transition.valid_mask]
        values = values[np.isfinite(values)]
        if values.size:
            hist_axis.hist(values, bins=45, color="#777777", alpha=0.82)
        hist_axis.axvline(transition.tau, color="#e6550d", linewidth=1.1)
        hist_axis.axvline(-transition.tau, color="#3182bd", linewidth=1.1)
        hist_axis.axvline(transition.strong_tau, color="#a50f15", linewidth=1.1, linestyle="--")
        hist_axis.axvline(-transition.strong_tau, color="#08519c", linewidth=1.1, linestyle="--")
        hist_axis.set_title("Delta_Irat histogram")
        hist_axis.set_xlabel("relative Delta" if result.parameters.use_relative_delta else "Delta_Irat")
        hist_axis.set_ylabel("pixels")

        x_index, y_index = self.transition_outcome_selected_pixel or self._default_transition_outcome_pixel()
        inspector_file = self._transition_outcome_current_inspector_file_index()
        local_axis = self.transition_outcome_figure.add_subplot(grid[1, 0])
        edc_axis = self.transition_outcome_figure.add_subplot(grid[1, 1])
        mdc_axis = self.transition_outcome_figure.add_subplot(grid[2, 0])
        trace_axis = self.transition_outcome_figure.add_subplot(grid[2, 1])
        self._plot_transition_outcome_local_spectrum(local_axis, inspector_file, x_index, y_index, "selected file")
        self._plot_transition_outcome_edc_single(edc_axis, inspector_file, x_index, y_index)
        self._plot_transition_outcome_mdc_single(mdc_axis, inspector_file, x_index, y_index)
        self._plot_transition_outcome_pixel_trace(trace_axis, x_index, y_index)

        self.transition_outcome_canvas.draw_idle()
        self._update_transition_outcome_plot_scroll_region()

    def _transition_label_cmap_norm(self) -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        colors = [TRANSITION_OUTCOME_COLORS[label] for label in TRANSITION_OUTCOME_LABELS]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(np.arange(-0.5, len(TRANSITION_OUTCOME_LABELS) + 0.5, 1.0), cmap.N)
        return cmap, norm

    def _short_transition_label(self, label: str) -> str:
        return {
            "invalid / low signal": "invalid",
            "written / more metallic": "written",
            "erased / less metallic": "erased",
            "strongly written": "strong write",
            "strongly erased": "strong erase",
        }.get(label, label)

    def _transition_counts_title(self, transition: object) -> str:
        counts = transition.counts
        written = counts.get("written / more metallic", 0) + counts.get("strongly written", 0)
        erased = counts.get("erased / less metallic", 0) + counts.get("strongly erased", 0)
        unchanged = counts.get("unchanged", 0)
        pulse = f" ({transition.pulse_label})" if transition.pulse_label else ""
        return (
            f"{transition.before_index}->{transition.after_index}{pulse}\n"
            f"W {written} | E {erased} | U {unchanged}"
        )

    def _on_transition_outcome_plot_click(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        if self.transition_outcome_result is None or event.inaxes is None:
            return
        self.transition_outcome_focused_transition = None
        if event.inaxes in self.transition_outcome_axis_to_file:
            file_index = self.transition_outcome_axis_to_file[event.inaxes]
            self._set_transition_outcome_inspector_file(file_index)
        if event.inaxes in self.transition_outcome_map_axes and event.xdata is not None and event.ydata is not None:
            x_index = int(round(event.xdata))
            y_index = int(round(event.ydata))
            x_size, y_size = self.transition_outcome_result.shape
            if 0 <= x_index < x_size and 0 <= y_index < y_size:
                self.transition_outcome_selected_pixel = (x_index, y_index)
                self.transition_outcome_hover_pixel = None
        self._refresh_transition_outcome_views()

    def _on_transition_outcome_plot_hover(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        if self.transition_outcome_result is None or event.inaxes not in self.transition_outcome_map_axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        x_index = int(round(event.xdata))
        y_index = int(round(event.ydata))
        x_size, y_size = self.transition_outcome_result.shape
        if not (0 <= x_index < x_size and 0 <= y_index < y_size):
            return
        pixel = (x_index, y_index)
        if pixel == self.transition_outcome_hover_pixel:
            return
        self.transition_outcome_hover_pixel = pixel
        self._update_transition_outcome_summary_text()

    def _set_transition_outcome_inspector_file(self, file_index: int) -> None:
        if self.transition_outcome_result is None:
            return
        safe_index = min(max(0, int(file_index)), self.transition_outcome_result.n_states - 1)
        value = f"{safe_index}: {self._short_file_label(self.transition_outcome_result.file_paths[safe_index], 26)}"
        if self.transition_outcome_inspector_file_var.get() != value:
            self.transition_outcome_inspector_file_var.set(value)

    def _mark_transition_outcome_selected_pixel(self, axis: matplotlib.axes.Axes) -> None:
        if self.transition_outcome_selected_pixel is None:
            return
        x_index, y_index = self.transition_outcome_selected_pixel
        axis.scatter([x_index], [y_index], s=90, facecolors="none", edgecolors="white", linewidths=1.8)
        axis.scatter([x_index], [y_index], s=18, c="black")

    def _transition_outcome_spectrum_payload(
        self,
        state_index: int,
        x_index: int,
        y_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.transition_outcome_result is not None
        state = self.transition_outcome_result.loaded_states[state_index]
        data = np.asarray(state.data_array.values, dtype=np.float32)
        x_safe = min(max(0, int(x_index)), data.shape[0] - 1)
        y_safe = min(max(0, int(y_index)), data.shape[1] - 1)
        energy_axis = np.asarray(state.data_array.coords["eV"].values, dtype=np.float32)
        phi_axis = np.asarray(state.data_array.coords["phi"].values, dtype=np.float32)
        energy_order = np.argsort(energy_axis)
        phi_order = np.argsort(phi_axis)
        spectrum = np.asarray(data[x_safe, y_safe, :, :], dtype=np.float32)
        return spectrum[energy_order][:, phi_order], energy_axis[energy_order], phi_axis[phi_order]

    def _transition_outcome_current_inspector_file_index(self) -> int:
        result = self.transition_outcome_result
        if result is None or result.n_states <= 0:
            return 0
        text = self.transition_outcome_inspector_file_var.get().strip()
        try:
            index = int(text.split(":", 1)[0])
        except ValueError:
            index = 0
        return min(max(0, index), result.n_states - 1)

    def _plot_transition_outcome_edc_overlay(self, axis: matplotlib.axes.Axes, x_index: int, y_index: int) -> None:
        assert self.transition_outcome_result is not None
        for state_index, state in enumerate(self.transition_outcome_result.loaded_states):
            spectrum, energy_axis, phi_axis = self._transition_outcome_spectrum_payload(state_index, x_index, y_index)
            edc = np.trapezoid(spectrum, x=phi_axis, axis=1).astype(np.float32) if phi_axis.size > 1 else np.sum(spectrum, axis=1).astype(np.float32)
            scale = float(np.nanmax(np.abs(edc))) if np.any(np.isfinite(edc)) else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            axis.plot(energy_axis, edc / scale, linewidth=1.1, label=f"{state_index}: {self._short_file_label(state.file_path, 16)}")
        axis.axvline(self.transition_outcome_result.parameters.fermi_level_ev, color="#555555", linestyle="--", linewidth=0.8)
        axis.set_title("EDC evolution")
        axis.set_xlabel("eV")
        axis.set_ylabel("normalized intensity")
        axis.legend(loc="best", fontsize=6)

    def _plot_transition_outcome_edc_single(
        self,
        axis: matplotlib.axes.Axes,
        state_index: int,
        x_index: int,
        y_index: int,
    ) -> None:
        assert self.transition_outcome_result is not None
        spectrum, energy_axis, phi_axis = self._transition_outcome_spectrum_payload(state_index, x_index, y_index)
        edc = np.trapezoid(spectrum, x=phi_axis, axis=1).astype(np.float32) if phi_axis.size > 1 else np.sum(spectrum, axis=1).astype(np.float32)
        axis.plot(energy_axis, edc, linewidth=1.2, color="#d62728")
        axis.axvline(self.transition_outcome_result.parameters.fermi_level_ev, color="#555555", linestyle="--", linewidth=0.8)
        axis.set_title(f"EDC for file {state_index}")
        axis.set_xlabel("eV")
        axis.set_ylabel("intensity")

    def _plot_transition_outcome_mdc_overlay(self, axis: matplotlib.axes.Axes, x_index: int, y_index: int) -> None:
        assert self.transition_outcome_result is not None
        params = self.transition_outcome_result.parameters
        for state_index, state in enumerate(self.transition_outcome_result.loaded_states):
            spectrum, energy_axis, phi_axis = self._transition_outcome_spectrum_payload(state_index, x_index, y_index)
            mask = (
                (energy_axis >= params.fermi_level_ev + params.ef_min_ev)
                & (energy_axis <= params.fermi_level_ev + params.ef_max_ev)
            )
            if not np.any(mask):
                mask[int(np.argmin(np.abs(energy_axis - params.fermi_level_ev)))] = True
            mdc = np.trapezoid(spectrum[mask, :], x=energy_axis[mask], axis=0).astype(np.float32) if int(np.count_nonzero(mask)) > 1 else np.sum(spectrum[mask, :], axis=0).astype(np.float32)
            scale = float(np.nanmax(np.abs(mdc))) if np.any(np.isfinite(mdc)) else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            axis.plot(phi_axis, mdc / scale, linewidth=1.1, label=f"{state_index}: {self._short_file_label(state.file_path, 16)}")
        axis.set_title("Near-EF MDC evolution")
        axis.set_xlabel("phi")
        axis.set_ylabel("normalized intensity")
        axis.legend(loc="best", fontsize=6)

    def _plot_transition_outcome_mdc_single(
        self,
        axis: matplotlib.axes.Axes,
        state_index: int,
        x_index: int,
        y_index: int,
    ) -> None:
        assert self.transition_outcome_result is not None
        params = self.transition_outcome_result.parameters
        spectrum, energy_axis, phi_axis = self._transition_outcome_spectrum_payload(state_index, x_index, y_index)
        mask = (
            (energy_axis >= params.fermi_level_ev + params.ef_min_ev)
            & (energy_axis <= params.fermi_level_ev + params.ef_max_ev)
        )
        if not np.any(mask):
            mask[int(np.argmin(np.abs(energy_axis - params.fermi_level_ev)))] = True
        mdc = np.trapezoid(spectrum[mask, :], x=energy_axis[mask], axis=0).astype(np.float32) if int(np.count_nonzero(mask)) > 1 else np.sum(spectrum[mask, :], axis=0).astype(np.float32)
        axis.plot(phi_axis, mdc, linewidth=1.2, color="#1f77b4")
        axis.set_title(f"Near-EF MDC for file {state_index}")
        axis.set_xlabel("phi")
        axis.set_ylabel("intensity")

    def _plot_transition_outcome_local_spectrum(
        self,
        axis: matplotlib.axes.Axes,
        state_index: int,
        x_index: int,
        y_index: int,
        label: str,
    ) -> None:
        spectrum, energy_axis, phi_axis = self._transition_outcome_spectrum_payload(state_index, x_index, y_index)
        image = axis.imshow(
            spectrum,
            origin="lower",
            aspect="auto",
            extent=[float(phi_axis[0]), float(phi_axis[-1]), float(energy_axis[0]), float(energy_axis[-1])],
            cmap="viridis",
        )
        axis.axhline(0.0, color="white", linestyle="--", linewidth=0.8)
        axis.set_title(f"Local ARPES {label}\nfile {state_index}, x={x_index}, y={y_index}")
        axis.set_xlabel("phi")
        axis.set_ylabel("eV")
        self.transition_outcome_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    def _plot_transition_outcome_local_spectrum_on_figure(
        self,
        figure: Figure,
        axis: matplotlib.axes.Axes,
        state_index: int,
        x_index: int,
        y_index: int,
        title: str,
    ) -> None:
        spectrum, energy_axis, phi_axis = self._transition_outcome_spectrum_payload(state_index, x_index, y_index)
        image = axis.imshow(
            spectrum,
            origin="lower",
            aspect="auto",
            extent=[float(phi_axis[0]), float(phi_axis[-1]), float(energy_axis[0]), float(energy_axis[-1])],
            cmap="viridis",
        )
        axis.axhline(0.0, color="white", linestyle="--", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("phi")
        axis.set_ylabel("eV")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    def _plot_transition_outcome_pixel_trace(self, axis: matplotlib.axes.Axes, x_index: int, y_index: int) -> None:
        assert self.transition_outcome_result is not None
        result = self.transition_outcome_result
        file_indices = np.arange(result.n_states)
        irat_values = np.asarray([maps[x_index, y_index] for maps in result.i_rat_maps], dtype=np.float32)
        wef_values = np.asarray([maps[x_index, y_index] for maps in result.w_ef_maps], dtype=np.float32)
        delta_values = np.asarray([transition.delta_irat_map[x_index, y_index] for transition in result.transitions], dtype=np.float32)
        axis.plot(file_indices, irat_values, marker="o", color="#d62728", label="I_rat")
        if delta_values.size:
            axis.bar(file_indices[1:], delta_values, width=0.34, color="#777777", alpha=0.25, label="Delta I_rat")
        axis.set_xlabel("file / pulse index")
        axis.set_ylabel("I_rat")
        axis.set_title("Selected pixel chronological trace")
        twin = axis.twinx()
        twin.plot(file_indices, wef_values, marker="s", color="#1f77b4", linewidth=1.1, label="W_EF")
        twin.set_ylabel("W_EF")
        handles, labels = axis.get_legend_handles_labels()
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        axis.legend(handles + twin_handles, labels + twin_labels, loc="best", fontsize=7)

    def _refresh_transition_outcome_inspector_panel(self) -> None:
        if not hasattr(self, "transition_outcome_inspector_figure"):
            return
        result = self.transition_outcome_result
        figure = self.transition_outcome_inspector_figure
        figure.clear()
        if result is None:
            axis = figure.add_subplot(111)
            axis.text(0.5, 0.5, "Click a map pixel to inspect one file at that point.", ha="center", va="center")
            axis.set_axis_off()
            self.transition_outcome_inspector_canvas.draw_idle()
            return

        x_index, y_index = self.transition_outcome_selected_pixel or self._default_transition_outcome_pixel()
        file_index = self._transition_outcome_current_inspector_file_index()
        state = result.loaded_states[file_index]
        axes = figure.subplots(1, 4)
        local_axis, edc_axis, mdc_axis, trace_axis = axes

        self._plot_transition_outcome_local_spectrum_on_figure(
            figure,
            local_axis,
            file_index,
            x_index,
            y_index,
            f"File {file_index}: local ARPES\n{self._short_file_label(state.file_path, 26)}",
        )
        self._plot_transition_outcome_edc_single(edc_axis, file_index, x_index, y_index)
        self._plot_transition_outcome_mdc_single(mdc_axis, file_index, x_index, y_index)
        self._plot_transition_outcome_pixel_trace(trace_axis, x_index, y_index)
        figure.suptitle(
            f"Selected pixel x={x_index}, y={y_index} | "
            f"T={float(result.total_maps[file_index][x_index, y_index]):.4g}, "
            f"W_EF={float(result.w_ef_maps[file_index][x_index, y_index]):.4g}, "
            f"W_LHB={float(result.w_lhb_maps[file_index][x_index, y_index]):.4g}, "
            f"I_rat={float(result.i_rat_maps[file_index][x_index, y_index]):.4g}",
            fontsize=10,
        )
        self.transition_outcome_inspector_canvas.draw_idle()

    def _update_transition_outcome_summary_text(self) -> None:
        result = self.transition_outcome_result
        if result is None:
            self._set_text_widget(self.transition_outcome_summary_text, "")
            return

        active_pixel = self._transition_outcome_active_pixel()
        inspector_file = self._transition_outcome_current_inspector_file_index()
        lines = [
            f"Files: {result.n_states}",
            f"Transitions: {result.n_transitions}",
            f"Displayed map: {self.transition_outcome_map_var.get()}",
            f"Metric: {'relative Delta_Irat' if result.parameters.use_relative_delta else 'Delta_Irat'}",
            f"Inspector file: file {inspector_file} ({Path(result.file_paths[inspector_file]).name})",
            "",
            "Transition summaries:",
        ]
        for transition in result.transitions:
            pulse = f" ({transition.pulse_label})" if transition.pulse_label else ""
            lines.append(
                f"  {transition.before_index}->{transition.after_index}{pulse}: "
                f"written={int(transition.stats['written_pixels'])} ({transition.stats['fraction_written']:.1%}), "
                f"erased={int(transition.stats['erased_pixels'])} ({transition.stats['fraction_erased']:.1%}), "
                f"unchanged={int(transition.stats['unchanged_pixels'])}, "
                f"mean dI={transition.stats['mean_delta_irat']:.5g}, tau={transition.tau:.5g}"
            )

        if active_pixel is not None:
            x_index, y_index = active_pixel
            pixel_kind = "Hovered pixel" if self.transition_outcome_hover_pixel is not None else "Selected pixel"
            lines.extend(["", f"{pixel_kind}: x={x_index}, y={y_index}"])
            lines.append(
                f"Selected-file quantities: "
                f"T={float(result.total_maps[inspector_file][x_index, y_index]):.6g}, "
                f"W_EF={float(result.w_ef_maps[inspector_file][x_index, y_index]):.6g}, "
                f"W_LHB={float(result.w_lhb_maps[inspector_file][x_index, y_index]):.6g}, "
                f"I_rat={float(result.i_rat_maps[inspector_file][x_index, y_index]):.6g}"
            )
            lines.append("I_rat by file:")
            irat_values = [
                float(result.i_rat_maps[index][x_index, y_index])
                for index in range(result.n_states)
            ]
            lines.append("  " + ", ".join(f"{index}: {value:.5g}" for index, value in enumerate(irat_values)))
            lines.extend(["", "Transition decision details:"])
            for transition in result.transitions:
                label = self._transition_pixel_event_label(transition, x_index, y_index)
                before = transition.before_index
                after = transition.after_index
                lines.append(
                    f"  file_{transition.before_index} -> file_{transition.after_index}: "
                    f"{label}, "
                    f"I_rat {float(result.i_rat_maps[before][x_index, y_index]):.5g} -> "
                    f"{float(result.i_rat_maps[after][x_index, y_index]):.5g}, "
                    f"Delta_Irat={float(transition.delta_irat_map[x_index, y_index]):+.6g}, "
                    f"metric={float(transition.metric_delta_map[x_index, y_index]):+.6g}, "
                    f"Delta W_EF={float(transition.delta_w_ef_map[x_index, y_index]):+.6g}"
                )
                lines.append(f"    {self._transition_pixel_decision_reason(transition, x_index, y_index)}")
            lines.extend(["", self._transition_pixel_summary_sentence(x_index, y_index)])

        if self.transition_outcome_focused_transition is not None:
            lines.append("")
            lines.append(f"Focused transition: {self.transition_outcome_focused_transition}")

        if result.notes:
            lines.extend(["", "Notes:"])
            lines.extend(f"  {note}" for note in result.notes[:5])
            if len(result.notes) > 5:
                lines.append(f"  ... {len(result.notes) - 5} more note(s)")

        self._set_text_widget(self.transition_outcome_summary_text, "\n".join(lines))

    def _transition_outcome_active_pixel(self) -> tuple[int, int] | None:
        return self.transition_outcome_hover_pixel or self.transition_outcome_selected_pixel

    def _transition_pixel_event_label(self, transition: object, x_index: int, y_index: int) -> str:
        label = str(transition.label_map[x_index, y_index])
        return {
            "written / more metallic": "written",
            "strongly written": "strongly written",
            "erased / less metallic": "erased",
            "strongly erased": "strongly erased",
            "unchanged": "unchanged",
            "invalid / low signal": "invalid",
        }.get(label, label)

    def _transition_pixel_decision_reason(self, transition: object, x_index: int, y_index: int) -> str:
        metric = float(transition.metric_delta_map[x_index, y_index])
        delta_irat = float(transition.delta_irat_map[x_index, y_index])
        relative_delta = float(transition.relative_delta_irat_map[x_index, y_index])
        label = str(transition.label_map[x_index, y_index])
        metric_name = "relative Delta_Irat" if self.transition_outcome_result.parameters.use_relative_delta else "Delta_Irat"
        if not bool(transition.valid_mask[x_index, y_index]) or label == "invalid / low signal":
            return "invalid because the valid-pixel mask failed from low total intensity, low W_LHB, or non-finite values."
        if label == "strongly written":
            return f"strongly written because {metric_name}={metric:+.5g} >= strong_tau={transition.strong_tau:.5g}."
        if label == "written / more metallic":
            return f"written because {metric_name}={metric:+.5g} >= tau={transition.tau:.5g}."
        if label == "strongly erased":
            return f"strongly erased because {metric_name}={metric:+.5g} <= -strong_tau={-transition.strong_tau:.5g}."
        if label == "erased / less metallic":
            return f"erased because {metric_name}={metric:+.5g} <= -tau={-transition.tau:.5g}."
        return (
            f"unchanged because |{metric_name}|={abs(metric):.5g} < tau={transition.tau:.5g}; "
            f"raw Delta_Irat={delta_irat:+.5g}, relative Delta_Irat={relative_delta:+.5g}."
        )

    def _transition_pixel_summary_sentence(self, x_index: int, y_index: int) -> str:
        assert self.transition_outcome_result is not None
        signs: list[int] = []
        for transition in self.transition_outcome_result.transitions:
            code = int(transition.code_map[x_index, y_index])
            if code in (2, 4):
                signs.append(1)
            elif code in (3, 5):
                signs.append(-1)
        if not signs:
            return "This pixel is unchanged or invalid across the transition sequence."
        has_rewrite = any(signs[i] == -1 and any(sign == 1 for sign in signs[i + 1:]) for i in range(len(signs)))
        has_later_erase = any(signs[i] == 1 and any(sign == -1 for sign in signs[i + 1:]) for i in range(len(signs)))
        if has_rewrite and has_later_erase:
            return "This pixel shows write/erase/rewrite behavior and may be memory-like."
        if has_rewrite:
            return "This pixel is erased in an earlier transition and rewritten later."
        if has_later_erase:
            return "This pixel is written in an earlier transition and erased later."
        if len(signs) > 1:
            return "This pixel is repeatedly active across multiple transitions."
        return "This pixel changes in one transition only."

    def _save_transition_outcome_results(self) -> None:
        if self.transition_outcome_result is None:
            messagebox.showinfo("No Transition Outcome Maps", "Compute transition maps before saving results.")
            return
        directory = filedialog.askdirectory(title="Choose output folder for Transition Outcome Maps")
        if not directory:
            return
        try:
            paths = export_transition_outcome_maps(self.transition_outcome_result, directory)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.transition_outcome_status_var.set(f"Saved transition table to {paths['table']}")

    def _save_transition_outcome_plot(self) -> None:
        if self.transition_outcome_result is None:
            messagebox.showinfo("No Transition Outcome plot", "Compute transition maps before saving a plot.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Transition Outcome plot",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF document", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.transition_outcome_figure.savefig(path, dpi=220)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.transition_outcome_status_var.set(f"Saved Transition Outcome plot to {path}")

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
        self.cluster_result = None
        self.cluster_interpretation = None
        self.cluster_cache.clear()
        self.cluster_interpretation_cache.clear()
        self.selected_pixel = None
        if self.cluster_popup is not None and self.cluster_popup.winfo_exists():
            self.cluster_popup.withdraw()
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
        self.cluster_state_combo["values"] = state_values
        self.feature_combo["values"] = feature_values

        if state_values:
            if self.state_var.get() not in state_values:
                self.state_var.set(state_values[0])
            if self.compare_from_var.get() not in state_values:
                self.compare_from_var.set(state_values[0])
            if self.compare_to_var.get() not in state_values:
                self.compare_to_var.set(state_values[min(1, len(state_values) - 1)])
            if self.cluster_state_var.get() not in state_values:
                self.cluster_state_var.set(state_values[0])
        else:
            self.state_var.set("")
            self.compare_from_var.set("")
            self.compare_to_var.set("")
            self.cluster_state_var.set("")

        if feature_values:
            if self.feature_var.get() not in feature_values:
                self.feature_var.set(feature_values[0])
        else:
            self.feature_var.set("")

        if self.cluster_result is not None:
            focus_values = self._cluster_focus_values(self.cluster_result)
            self.cluster_focus_combo["values"] = focus_values
            if focus_values and self.cluster_focus_var.get() not in focus_values:
                self.cluster_focus_var.set(focus_values[0])
        else:
            self.cluster_focus_combo["values"] = []
            self.cluster_focus_var.set("")

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

    def _parse_cluster_parameters(self) -> SpectralClusterParameters:
        try:
            params = SpectralClusterParameters(
                n_clusters=int(self.cluster_parameter_vars["n_clusters"].get()),
                embedding_components=int(self.cluster_parameter_vars["embedding_components"].get()),
                method_key=self._cluster_method_key(),
            )
        except ValueError as exc:
            raise ValueError(f"Could not parse the clustering controls: {exc}") from exc

        params.validate()
        return params

    def _cluster_method_key(self) -> str:
        label = self.cluster_method_var.get()
        for key, value in SPECTRAL_CLUSTER_METHOD_LABELS.items():
            if value == label:
                return key
        return "downsampled_spectra_pca"

    def _cluster_focus_values(self, cluster_result: SpectralClusterResult) -> list[str]:
        return [f"C{stats.cluster_id} | {stats.candidate_label}" for stats in cluster_result.cluster_stats]

    def _selected_cluster_id(self) -> int | None:
        value = self.cluster_focus_var.get().strip()
        if not value.startswith("C"):
            return None
        head = value.split("|", 1)[0].strip()
        try:
            return int(head[1:])
        except ValueError:
            return None

    def _current_cluster_state_index(self) -> int:
        assert self.result is not None
        try:
            return self.result.state_names.index(self.cluster_state_var.get())
        except ValueError:
            return 0

    def _cluster_cache_key(self, state_index: int, params: SpectralClusterParameters) -> tuple[int, int, int, str]:
        return (
            state_index,
            params.n_clusters,
            params.embedding_components,
            params.method_key,
        )

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
        self._start_global_progress("Analysis running...")
        self.root.update_idletasks()

        try:
            self.result = run_analysis(self.file_paths, parameters)
        except Exception as exc:
            self.result = None
            self.cluster_result = None
            self.cluster_interpretation = None
            self.cluster_cache.clear()
            self.cluster_interpretation_cache.clear()
            self.status_var.set("Analysis failed.")
            self._finish_global_progress("Analysis failed.", success=False)
            messagebox.showerror("Analysis failed", str(exc))
            self._render_placeholder_text()
            return

        self.cluster_result = None
        self.cluster_interpretation = None
        self.cluster_cache.clear()
        self.cluster_interpretation_cache.clear()
        self.selected_pixel = None
        self._update_selector_values()
        self._refresh_main_plot()
        self._update_pixel_details()
        self._update_summary_text()
        self._render_cluster_placeholder()
        self.cluster_status_var.set(
            "Choose a clustering method with a resource label, then inspect the mean cluster spectra and compare the classes against the raw map."
        )
        self.status_var.set(
            f"Analysis complete. {int(self.result.valid_mask.sum())} valid pixels inside the cross across {self.result.n_states} state(s)."
        )
        self._finish_global_progress("Analysis complete.")

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
        self._render_cluster_placeholder()

    def _render_change_placeholder(self) -> None:
        if not hasattr(self, "change_figure"):
            return

        self.change_figure.clear()
        axis = self.change_figure.add_subplot(111)
        if self.change_file_paths:
            message = "Ready to analyze.\nChoose the initial state, arrange the sequence, then click Analyze Changes."
        else:
            message = "Add NetCDF files to compare each state against an initial file."
        axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=13)
        axis.set_axis_off()
        self.change_canvas.draw_idle()

        self.change_sequence_figure.clear()
        sequence_axis = self.change_sequence_figure.add_subplot(111)
        sequence_axis.text(
            0.5,
            0.5,
            "The sequence overview will show energy-profile deltas and changed-pixel fractions.",
            ha="center",
            va="center",
            fontsize=12,
        )
        sequence_axis.set_axis_off()
        self.change_sequence_canvas.draw_idle()
        self._set_text_widget(self.change_summary_text, "")
        if not self.change_file_paths:
            self.change_status_var.set("Add NetCDF files, label the initial state, then run the initial-state change view.")

    def _refresh_change_views(self) -> None:
        if self.change_valid_mask is None or not self.change_loaded_states:
            self._render_change_placeholder()
            return
        self._refresh_change_detail_plot()
        self._refresh_change_sequence_plot()
        self._update_change_summary_text()

    def _refresh_change_detail_plot(self) -> None:
        assert self.change_valid_mask is not None
        baseline_index = self._current_change_initial_index()
        target_index = self._current_change_target_index()
        valid_mask = self.change_valid_mask
        energy_axis = self._change_energy_axis()
        metric_key = self._change_metric_key()
        metric_label = self.change_metric_var.get()

        self.change_figure.clear()
        axes = self.change_figure.subplots(2, 3)
        base_axis, target_axis, delta_axis = axes[0]
        transition_axis, profile_axis, delta_profile_axis = axes[1]

        baseline_name = Path(self.change_file_paths[baseline_index]).name
        target_name = Path(self.change_file_paths[target_index]).name

        base_total = np.asarray(self.change_total_maps[baseline_index], dtype=np.float32)
        target_total = np.asarray(self.change_total_maps[target_index], dtype=np.float32)
        base_image = base_axis.imshow(base_total.T, origin="lower", cmap="viridis", aspect="auto")
        target_image = target_axis.imshow(target_total.T, origin="lower", cmap="viridis", aspect="auto")
        base_axis.set_title(f"Initial: {baseline_name}\ntotal intensity")
        target_axis.set_title(f"Target: {target_name}\ntotal intensity")
        self.change_figure.colorbar(base_image, ax=base_axis, fraction=0.046, pad=0.04)
        self.change_figure.colorbar(target_image, ax=target_axis, fraction=0.046, pad=0.04)

        baseline_metric = np.asarray(self.change_features_by_state[baseline_index][metric_key], dtype=np.float32)
        target_metric = np.asarray(self.change_features_by_state[target_index][metric_key], dtype=np.float32)
        metric_delta = target_metric - baseline_metric
        metric_display = metric_delta.astype(float).copy()
        metric_display[~valid_mask] = np.nan
        vmax = self._symmetric_change_limit(metric_display)
        delta_image = delta_axis.imshow(
            metric_display.T,
            origin="lower",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        delta_axis.set_title(f"Delta {metric_label}\n{target_name} - initial")
        self.change_figure.colorbar(delta_image, ax=delta_axis, fraction=0.046, pad=0.04)

        _, transition_display = self._change_transition_map(baseline_index, target_index)
        transition_cmap = mcolors.ListedColormap(self.CHANGE_TRANSITION_COLORS)
        transition_cmap.set_bad(color="lightgray")
        transition_norm = mcolors.BoundaryNorm(np.arange(-0.5, 9.5, 1), transition_cmap.N)
        transition_image = transition_axis.imshow(
            transition_display.T,
            origin="lower",
            cmap=transition_cmap,
            norm=transition_norm,
            aspect="auto",
        )
        transition_axis.set_title("Simple-state transition map\ninitial -> target")
        transition_cbar = self.change_figure.colorbar(
            transition_image,
            ax=transition_axis,
            fraction=0.046,
            pad=0.04,
            ticks=np.arange(9),
        )
        transition_cbar.ax.set_yticklabels(self.CHANGE_TRANSITION_LABELS, fontsize=7)

        baseline_profile = self._normalized_change_profile(self.change_mean_energy_profiles[baseline_index])
        target_profile = self._normalized_change_profile(self.change_mean_energy_profiles[target_index])
        for index, profile in enumerate(self.change_mean_energy_profiles):
            if index in (baseline_index, target_index):
                continue
            profile_axis.plot(
                energy_axis,
                self._normalized_change_profile(profile),
                color="#999999",
                linewidth=0.8,
                alpha=0.35,
            )
        profile_axis.plot(energy_axis, baseline_profile, color="#222222", linewidth=2.4, label="initial")
        profile_axis.plot(energy_axis, target_profile, color="#d62728", linewidth=2.2, label="target")
        profile_axis.axvline(0.0, linestyle="--", color="#666666", linewidth=1.0)
        profile_axis.set_title("Area-normalized energy profile")
        profile_axis.set_xlabel("eV")
        profile_axis.set_ylabel("fraction")
        profile_axis.legend(loc="best", fontsize=8)

        delta_profile = target_profile - baseline_profile
        delta_profile_axis.axhline(0.0, color="#333333", linewidth=0.8)
        delta_profile_axis.plot(energy_axis, delta_profile, color="#333333", linewidth=1.2)
        delta_profile_axis.fill_between(
            energy_axis,
            0.0,
            delta_profile,
            where=delta_profile >= 0,
            color="#d95f02",
            alpha=0.35,
            interpolate=True,
        )
        delta_profile_axis.fill_between(
            energy_axis,
            0.0,
            delta_profile,
            where=delta_profile < 0,
            color="#1f77b4",
            alpha=0.35,
            interpolate=True,
        )
        delta_profile_axis.axvline(0.0, linestyle="--", color="#666666", linewidth=1.0)
        delta_profile_axis.set_title("Energy-profile change")
        delta_profile_axis.set_xlabel("eV")
        delta_profile_axis.set_ylabel("target - initial")

        for axis in (base_axis, target_axis, delta_axis, transition_axis):
            axis.set_xlabel("x index")
            axis.set_ylabel("y index")
            self._mark_change_selected_pixel(axis)

        self.change_canvas.draw_idle()

    def _refresh_change_sequence_plot(self) -> None:
        if self.change_valid_mask is None or not self.change_sequence_stats:
            self._render_change_placeholder()
            return

        baseline_index = self._current_change_initial_index()
        target_index = self._current_change_target_index()
        energy_axis = self._change_energy_axis()
        n_states = len(self.change_sequence_stats)
        positions = np.arange(n_states)

        self.change_sequence_figure.clear()
        axes = self.change_sequence_figure.subplots(3, 1, height_ratios=[1.25, 0.9, 1.15])
        profile_axis, stats_axis, heatmap_axis = axes

        cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(1, n_states))
        for stat in self.change_sequence_stats:
            index = int(stat["index"])
            profile = self._normalized_change_profile(self.change_mean_energy_profiles[index])
            if index == baseline_index:
                color = "#111111"
                linewidth = 2.5
                alpha = 1.0
            elif index == target_index:
                color = "#d62728"
                linewidth = 2.2
                alpha = 1.0
            else:
                color = cmap(index)
                linewidth = 1.0
                alpha = 0.55
            profile_axis.plot(
                energy_axis,
                profile,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=Path(str(stat["path"])).name,
            )
        profile_axis.axvline(0.0, linestyle="--", color="#666666", linewidth=1.0)
        profile_axis.set_title("Energy profiles in sequence order")
        profile_axis.set_xlabel("eV")
        profile_axis.set_ylabel("fraction")
        if n_states <= 8:
            profile_axis.legend(loc="best", fontsize=7)

        changed_fractions = [float(stat["changed_fraction"]) for stat in self.change_sequence_stats]
        delta_ef = [float(stat["delta_ef_fraction"]) for stat in self.change_sequence_stats]
        bar_colors = ["#111111" if int(stat["index"]) == baseline_index else "#7aa6c2" for stat in self.change_sequence_stats]
        stats_axis.bar(positions, changed_fractions, color=bar_colors, alpha=0.78, label="changed pixels")
        stats_axis.set_ylabel("changed fraction")
        stats_axis.set_ylim(0, max(1e-6, min(1.0, max(changed_fractions) * 1.25 if changed_fractions else 1.0)))
        stats_axis.set_xticks(positions)
        stats_axis.set_xticklabels([str(index + 1) for index in positions])
        stats_axis.set_title("Changed simple-state pixels and near-EF shift vs initial")
        stats_axis_2 = stats_axis.twinx()
        stats_axis_2.plot(positions, delta_ef, color="#d62728", marker="o", linewidth=1.4, label="delta near-EF")
        stats_axis_2.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        stats_axis_2.set_ylabel("delta near-EF fraction")

        delta_profiles = np.vstack([np.asarray(stat["delta_profile"], dtype=np.float32) for stat in self.change_sequence_stats])
        heat_vmax = self._symmetric_change_limit(delta_profiles)
        heatmap = heatmap_axis.imshow(
            delta_profiles,
            origin="upper",
            aspect="auto",
            cmap="coolwarm",
            vmin=-heat_vmax,
            vmax=heat_vmax,
            extent=[
                float(energy_axis[0]) if energy_axis.size else 0.0,
                float(energy_axis[-1]) if energy_axis.size else 1.0,
                n_states - 0.5,
                -0.5,
            ],
        )
        heatmap_axis.axvline(0.0, linestyle="--", color="#333333", linewidth=0.9)
        heatmap_axis.set_title("Delta energy profile per file (target - initial)")
        heatmap_axis.set_xlabel("eV")
        heatmap_axis.set_yticks(positions)
        heatmap_axis.set_yticklabels([Path(str(stat["path"])).name for stat in self.change_sequence_stats], fontsize=7)
        self.change_sequence_figure.colorbar(heatmap, ax=heatmap_axis, fraction=0.046, pad=0.04)

        self.change_sequence_canvas.draw_idle()

    def _update_change_summary_text(self) -> None:
        if self.change_valid_mask is None or not self.change_sequence_stats:
            self._set_text_widget(self.change_summary_text, "")
            return

        baseline_index = self._current_change_initial_index()
        target_index = self._current_change_target_index()
        baseline_name = Path(self.change_file_paths[baseline_index]).name
        target_name = Path(self.change_file_paths[target_index]).name
        valid_pixels = int(np.sum(self.change_valid_mask))
        low, high = self.change_simple_state_thresholds or (float("nan"), float("nan"))
        lines = [
            f"Initial state: {baseline_name}",
            f"Selected target: {target_name}",
            f"Valid pixels inside cross: {valid_pixels}",
            f"Simple-state thresholds from all loaded files: insulating <= {low:.6f}, metallic >= {high:.6f}",
            "",
            "Per-file changes from initial:",
        ]

        for stat in self.change_sequence_stats:
            index = int(stat["index"])
            if index == baseline_index:
                lines.append(f"  - {index + 1}. {stat['name']}: initial reference")
                continue
            lines.append(
                f"  - {index + 1}. {stat['name']}: "
                f"changed pixels={float(stat['changed_fraction']):.1%}, "
                f"delta near-EF={float(stat['delta_ef_fraction']):+.6f}, "
                f"delta e-centroid={float(stat['delta_e_centroid']):+.6f} eV, "
                f"dominant gain={float(stat['dominant_gain_ev']):+.4f} eV, "
                f"dominant loss={float(stat['dominant_loss_ev']):+.4f} eV"
            )

        lines.extend(["", "Selected target transition counts (rows=initial, columns=target):"])
        counts = self._change_transition_counts(baseline_index, target_index)
        header = "       to I      to X      to M"
        lines.append(header)
        for row_index, row_label in enumerate(("from I", "from X", "from M")):
            lines.append(
                f"{row_label:>6} {counts[row_index, 0]:9d} {counts[row_index, 1]:9d} {counts[row_index, 2]:9d}"
            )

        if self.change_selected_pixel is not None:
            lines.extend(["", *self._build_change_pixel_lines(baseline_index, target_index)])

        self._set_text_widget(self.change_summary_text, "\n".join(lines))

    def _build_change_pixel_lines(self, baseline_index: int, target_index: int) -> list[str]:
        if self.change_selected_pixel is None:
            return []
        x_index, y_index = self.change_selected_pixel
        baseline_label = str(self.change_simple_state_label_maps[baseline_index][x_index, y_index])
        target_label = str(self.change_simple_state_label_maps[target_index][x_index, y_index])
        valid = bool(self.change_valid_mask[x_index, y_index]) if self.change_valid_mask is not None else False
        metric_key = self._change_metric_key()
        baseline_metric = float(self.change_features_by_state[baseline_index][metric_key][x_index, y_index])
        target_metric = float(self.change_features_by_state[target_index][metric_key][x_index, y_index])
        return [
            f"Selected pixel: x={x_index}, y={y_index}",
            f"Inside cross mask: {'yes' if valid else 'no'}",
            f"Simple state: {baseline_label} -> {target_label}",
            f"{self.change_metric_var.get()}: {baseline_metric:.6f} -> {target_metric:.6f} ({target_metric - baseline_metric:+.6f})",
        ]

    def _change_transition_map(self, from_index: int, to_index: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.change_valid_mask is not None
        from_codes = self.change_simple_state_code_maps[from_index]
        to_codes = self.change_simple_state_code_maps[to_index]
        raw = np.full(from_codes.shape, fill_value=-1, dtype=int)
        valid = self.change_valid_mask & (from_codes >= 0) & (to_codes >= 0)
        raw[valid] = from_codes[valid] * 3 + to_codes[valid]
        display = raw.astype(float)
        display[~self.change_valid_mask] = np.nan
        return raw, display

    def _symmetric_change_limit(self, values: np.ndarray) -> float:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 1e-6
        limit = float(np.nanpercentile(np.abs(finite), 99))
        if not np.isfinite(limit) or limit <= 0:
            limit = float(np.nanmax(np.abs(finite))) if finite.size else 1e-6
        return limit if limit > 0 else 1e-6

    def _on_change_plot_click(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        if self.change_valid_mask is None or event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        x_index = int(round(event.xdata))
        y_index = int(round(event.ydata))
        x_size, y_size = self.change_valid_mask.shape
        if not (0 <= x_index < x_size and 0 <= y_index < y_size):
            return

        self.change_selected_pixel = (x_index, y_index)
        self._refresh_change_views()

    def _mark_change_selected_pixel(self, axis: matplotlib.axes.Axes) -> None:
        if self.change_selected_pixel is None:
            return
        x_index, y_index = self.change_selected_pixel
        axis.scatter([x_index], [y_index], s=80, facecolors="none", edgecolors="white", linewidths=1.8)
        axis.scatter([x_index], [y_index], s=16, c="black")

    def _save_change_plot(self) -> None:
        if self.change_valid_mask is None:
            messagebox.showinfo("No plot", "Run the change analysis before saving a plot.")
            return

        path = filedialog.asksaveasfilename(
            title="Save change plot",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF document", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.change_figure.savefig(path, dpi=220)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        self.change_status_var.set(f"Saved change plot to {path}")

    def _render_cluster_placeholder(self) -> None:
        self.cluster_figure.clear()
        axis = self.cluster_figure.add_subplot(111)
        if self.result is None:
            message = "Run the main analysis first, then use this panel to cluster the registered spectra."
        else:
            message = (
                "Ready for clustering.\n"
                "Choose a state, set the clustering controls, and click Run Clustering."
            )
        axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
        axis.set_axis_off()
        self.cluster_canvas.draw_idle()
        self._set_text_widget(self.cluster_text, "")
        if self.result is None:
            self.cluster_status_var.set("Run the main analysis, then use the Clustering panel to cluster registered spectra.")

    def _handle_cluster_selector_change(self) -> None:
        if self.result is None:
            self._render_cluster_placeholder()
            return

        try:
            params = self._parse_cluster_parameters()
        except Exception:
            self.cluster_result = None
            self.cluster_interpretation = None
            self.cluster_focus_combo["values"] = []
            self.cluster_focus_var.set("")
            self.cluster_status_var.set("Fix the clustering controls, then rerun clustering.")
            self._render_cluster_placeholder()
            return

        cached = self.cluster_cache.get(self._cluster_cache_key(self._current_cluster_state_index(), params))
        if cached is None:
            self.cluster_result = None
            self.cluster_interpretation = None
            self.cluster_focus_combo["values"] = []
            self.cluster_focus_var.set("")
            self.cluster_status_var.set("Selections changed. Click Run Clustering to recompute the clusters.")
            self._render_cluster_placeholder()
            return

        self.cluster_result = cached
        self.cluster_interpretation = None
        focus_values = self._cluster_focus_values(cached)
        self.cluster_focus_combo["values"] = focus_values
        if focus_values and self.cluster_focus_var.get() not in focus_values:
            self.cluster_focus_var.set(focus_values[0])
        self._refresh_cluster_plot()

    def _run_cluster_test(self) -> None:
        if self.result is None:
            messagebox.showinfo("Run analysis first", "Run the main analysis before testing spectral clustering.")
            return

        try:
            params = self._parse_cluster_parameters()
        except Exception as exc:
            messagebox.showerror("Invalid clustering controls", str(exc))
            return

        state_index = self._current_cluster_state_index()
        cache_key = self._cluster_cache_key(state_index, params)
        cached = self.cluster_cache.get(cache_key)
        if cached is not None:
            self.cluster_result = cached
        else:
            self.cluster_status_var.set("Running clustering...")
            self._start_global_progress("Spectral clustering running...")
            self.root.update_idletasks()
            try:
                self.cluster_result = run_spectral_clustering(
                    self.result.loaded_states[state_index],
                    self.result.valid_mask,
                    feature_maps=self.result.features_by_state[state_index],
                    parameters=params,
                    analysis_parameters=self.result.parameters,
                )
            except Exception as exc:
                self.cluster_result = None
                self.cluster_focus_combo["values"] = []
                self.cluster_focus_var.set("")
                self.cluster_status_var.set("Clustering failed.")
                self._render_cluster_placeholder()
                self._finish_global_progress("Spectral clustering failed.", success=False)
                messagebox.showerror("Clustering failed", str(exc))
                return
            self.cluster_cache[cache_key] = self.cluster_result
        if cached is not None:
            self._start_global_progress("Spectral clustering loading cached result...")

        assert self.cluster_result is not None
        focus_values = self._cluster_focus_values(self.cluster_result)
        self.cluster_focus_combo["values"] = focus_values
        if focus_values and self.cluster_focus_var.get() not in focus_values:
            self.cluster_focus_var.set(focus_values[0])

        self.cluster_status_var.set(
            f"Clustering complete for {self.cluster_result.state_name}: {len(self.cluster_result.cluster_stats)} clusters with {SPECTRAL_CLUSTER_METHOD_LABELS[self.cluster_result.parameters.method_key]}."
        )
        self._refresh_cluster_plot()
        self._show_cluster_interpretation_popup(cache_key)
        self._finish_global_progress("Spectral clustering complete.")

    def _show_cluster_interpretation_popup(self, cache_key: tuple[int, int, int, str]) -> None:
        assert self.cluster_result is not None

        cached = self.cluster_interpretation_cache.get(cache_key)
        if cached is None:
            try:
                report = analyze_cluster_physical_interpretation(self.cluster_result)
                csv_paths = export_cluster_physical_interpretation(report)
            except Exception as exc:
                messagebox.showerror(
                    "Interpretation failed",
                    f"Clustering finished, but the physical interpretation panel could not be built:\n{exc}",
                )
                return
            cached = (report, csv_paths)
            self.cluster_interpretation_cache[cache_key] = cached

        self.cluster_interpretation, csv_paths = cached
        self._ensure_cluster_popup()
        assert self.cluster_popup is not None
        self.cluster_popup.deiconify()
        self.cluster_popup.lift()
        self.cluster_popup.focus_force()
        self._update_cluster_popup_contents(self.cluster_result, self.cluster_interpretation, csv_paths)

    def _ensure_cluster_popup(self) -> None:
        if self.cluster_popup is not None and self.cluster_popup.winfo_exists():
            return

        popup = tk.Toplevel(self.root)
        popup.title("Cluster Interpretation")
        popup.geometry("1500x980")
        popup.minsize(1180, 780)
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(popup)
        notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        summary_frame = ttk.Frame(notebook, padding=8)
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)
        notebook.add(summary_frame, text="Summary")

        self.cluster_popup_summary_text = tk.Text(summary_frame, wrap="word")
        self.cluster_popup_summary_text.grid(row=0, column=0, sticky="nsew")
        self.cluster_popup_summary_text.configure(state="disabled")

        metrics_frame = ttk.Frame(notebook, padding=8)
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.rowconfigure(0, weight=1)
        notebook.add(metrics_frame, text="Metrics")

        self.cluster_popup_metrics_figure = Figure(figsize=(12, 8.5), dpi=100, constrained_layout=True)
        self.cluster_popup_metrics_canvas = FigureCanvasTkAgg(self.cluster_popup_metrics_figure, master=metrics_frame)
        self.cluster_popup_metrics_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        spectra_frame = ttk.Frame(notebook, padding=8)
        spectra_frame.columnconfigure(0, weight=1)
        spectra_frame.rowconfigure(0, weight=1)
        notebook.add(spectra_frame, text="Mean Spectra")

        self.cluster_popup_spectra_figure = Figure(figsize=(12, 8.5), dpi=100, constrained_layout=True)
        self.cluster_popup_spectra_canvas = FigureCanvasTkAgg(self.cluster_popup_spectra_figure, master=spectra_frame)
        self.cluster_popup_spectra_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        comparison_frame = ttk.Frame(notebook, padding=8)
        comparison_frame.columnconfigure(0, weight=1)
        comparison_frame.rowconfigure(0, weight=1)
        notebook.add(comparison_frame, text="Comparisons")

        self.cluster_popup_comparison_text = tk.Text(comparison_frame, wrap="word")
        self.cluster_popup_comparison_text.grid(row=0, column=0, sticky="nsew")
        self.cluster_popup_comparison_text.configure(state="disabled")

        def _on_close() -> None:
            if self.cluster_popup is not None:
                self.cluster_popup.withdraw()

        popup.protocol("WM_DELETE_WINDOW", _on_close)
        self.cluster_popup = popup

    def _update_cluster_popup_contents(
        self,
        cluster_result: SpectralClusterResult,
        report: ClusterPhysicalInterpretation,
        csv_paths: dict[str, Path],
    ) -> None:
        summary_lines = [
            f"State: {report.state_name}",
            f"File: {report.state_file}",
            f"Method: {SPECTRAL_CLUSTER_METHOD_LABELS[cluster_result.parameters.method_key]}",
            "",
            "Saved CSV files:",
            f"  - Summary: {csv_paths['summary']}",
            f"  - Per-cluster metrics: {csv_paths['metrics']}",
            f"  - Pairwise differences: {csv_paths['pairwise']}",
            "",
            "Physical interpretation summary:",
        ]
        for summary in report.question_summaries:
            summary_lines.append(f"- {summary.question}")
            summary_lines.append(f"  {summary.answer} {summary.strongest_example} {summary.reasoning}".strip())
        if report.notes:
            summary_lines.extend(["", "Notes:"])
            for note in report.notes:
                summary_lines.append(f"- {note}")

        if self.cluster_popup_summary_text is not None:
            self._set_text_widget(self.cluster_popup_summary_text, "\n".join(summary_lines))

        comparison_lines = ["Pairwise physically interpretable differences:"]
        meaningful_rows = [row for row in report.pairwise_rows if row.overall_physically_distinct]
        if not meaningful_rows:
            comparison_lines.append("- No pair crossed the practical interpretation thresholds on more than one metric.")
        else:
            for row in meaningful_rows:
                comparison_lines.append(f"- C{row.cluster_a} vs C{row.cluster_b}: {row.interpretation}")
        comparison_lines.extend(["", "All pairwise comparisons:"])
        for row in report.pairwise_rows:
            comparison_lines.append(
                f"- C{row.cluster_a} vs C{row.cluster_b}: distinct={'yes' if row.overall_physically_distinct else 'no'}, "
                f"dEF={row.fermi_weight_diff:+.3f}, dGap={row.gap_proxy_diff_ev:+.3f} eV, "
                f"dPeak={row.dominant_peak_diff_ev:+.3f} eV, dWidth={row.dominant_peak_width_diff_ev:+.3f} eV, "
                f"dispersion corr={row.dispersion_shape_correlation:.3f}, "
                f"dWeights(deep/shallow/EF)={row.deep_weight_diff:+.3f}/{row.shallow_weight_diff:+.3f}/{row.near_ef_weight_diff:+.3f}"
            )
        if self.cluster_popup_comparison_text is not None:
            self._set_text_widget(self.cluster_popup_comparison_text, "\n".join(comparison_lines))

        self._draw_cluster_popup_metric_figures(cluster_result, report)
        self._draw_cluster_popup_spectra_figures(cluster_result, report)

    def _draw_cluster_popup_metric_figures(
        self,
        cluster_result: SpectralClusterResult,
        report: ClusterPhysicalInterpretation,
    ) -> None:
        if self.cluster_popup_metrics_figure is None or self.cluster_popup_metrics_canvas is None:
            return

        colors = self._cluster_color_lookup(cluster_result)
        metrics = report.metrics_rows
        cluster_ids = [row.cluster_id for row in metrics]
        x_positions = np.arange(len(cluster_ids))

        self.cluster_popup_metrics_figure.clear()
        axes = self.cluster_popup_metrics_figure.subplots(2, 2)
        profile_axis, ef_axis = axes[0]
        peak_axis, transfer_axis = axes[1]

        for stats in cluster_result.cluster_stats:
            profile = np.asarray(stats.mean_energy_profile, dtype=np.float32).copy()
            max_value = float(np.nanmax(profile)) if profile.size else 0.0
            if max_value > 0:
                profile /= max_value
            profile_axis.plot(
                cluster_result.e_axis,
                profile,
                color=colors[stats.cluster_id],
                linewidth=2.0,
                label=f"C{stats.cluster_id}",
            )
        profile_axis.axvline(0.0, linestyle="--", color="#666666", linewidth=1.0)
        profile_axis.set_title("Mean energy profiles")
        profile_axis.set_xlabel("eV")
        profile_axis.set_ylabel("normalized intensity")
        profile_axis.legend(loc="best", fontsize=8)

        bar_width = 0.36
        ef_axis.bar(
            x_positions - 0.5 * bar_width,
            [row.fermi_weight_fraction for row in metrics],
            width=bar_width,
            color=[colors[row.cluster_id] for row in metrics],
            alpha=0.85,
            label="Fermi-level weight",
        )
        ef_axis.bar(
            x_positions + 0.5 * bar_width,
            [row.gap_fill_ratio for row in metrics],
            width=bar_width,
            color=[colors[row.cluster_id] for row in metrics],
            alpha=0.35,
            label="Gap filling ratio",
        )
        ef_axis.set_title("Fermi-level weight and gap filling")
        ef_axis.set_xticks(x_positions)
        ef_axis.set_xticklabels([f"C{cluster_id}" for cluster_id in cluster_ids])
        ef_axis.set_ylabel("fraction / ratio")
        ef_axis.legend(loc="best", fontsize=8)

        peak_axis.bar(
            x_positions - 0.5 * bar_width,
            [row.dominant_peak_ev for row in metrics],
            width=bar_width,
            color=[colors[row.cluster_id] for row in metrics],
            alpha=0.85,
            label="Peak position (eV)",
        )
        peak_axis.bar(
            x_positions + 0.5 * bar_width,
            [row.dominant_peak_width_ev for row in metrics],
            width=bar_width,
            color=[colors[row.cluster_id] for row in metrics],
            alpha=0.35,
            label="Peak width (eV)",
        )
        peak_axis.axhline(0.0, linestyle="--", color="#666666", linewidth=1.0)
        peak_axis.set_title("Peak positions and widths")
        peak_axis.set_xticks(x_positions)
        peak_axis.set_xticklabels([f"C{cluster_id}" for cluster_id in cluster_ids])
        peak_axis.set_ylabel("eV")
        peak_axis.legend(loc="best", fontsize=8)

        deep = np.asarray([row.deep_weight_fraction for row in metrics], dtype=np.float32)
        shallow = np.asarray([row.shallow_weight_fraction for row in metrics], dtype=np.float32)
        near_ef = np.asarray([row.near_ef_weight_fraction for row in metrics], dtype=np.float32)
        transfer_axis.bar(x_positions, deep, color="#355c7d", label="deep")
        transfer_axis.bar(x_positions, shallow, bottom=deep, color="#f8b195", label="shallow")
        transfer_axis.bar(x_positions, near_ef, bottom=deep + shallow, color="#f67280", label="near-EF")
        transfer_axis.set_title("Relative spectral-weight windows")
        transfer_axis.set_xticks(x_positions)
        transfer_axis.set_xticklabels([f"C{cluster_id}" for cluster_id in cluster_ids])
        transfer_axis.set_ylabel("weight fraction")
        transfer_axis.legend(loc="best", fontsize=8)

        self.cluster_popup_metrics_canvas.draw_idle()

    def _draw_cluster_popup_spectra_figures(
        self,
        cluster_result: SpectralClusterResult,
        report: ClusterPhysicalInterpretation,
    ) -> None:
        if self.cluster_popup_spectra_figure is None or self.cluster_popup_spectra_canvas is None:
            return

        metrics_by_cluster = {row.cluster_id: row for row in report.metrics_rows}
        cluster_count = max(1, len(cluster_result.cluster_stats))
        columns = min(3, cluster_count)
        rows = int(np.ceil(cluster_count / columns))

        self.cluster_popup_spectra_figure.clear()
        axes = self.cluster_popup_spectra_figure.subplots(rows, columns, squeeze=False)

        for axis in axes.reshape(-1):
            axis.set_visible(False)

        for index, stats in enumerate(cluster_result.cluster_stats):
            axis = axes[index // columns, index % columns]
            axis.set_visible(True)
            metric_row = metrics_by_cluster[stats.cluster_id]
            image = axis.imshow(
                stats.mean_spectrum,
                origin="lower",
                aspect="auto",
                extent=[
                    float(cluster_result.phi_axis[0]),
                    float(cluster_result.phi_axis[-1]),
                    float(cluster_result.e_axis[0]),
                    float(cluster_result.e_axis[-1]),
                ],
                cmap="viridis",
            )
            axis.set_title(
                f"C{stats.cluster_id}: {stats.candidate_label}\n"
                f"EF={metric_row.fermi_weight_fraction:.3f}, gap~{metric_row.gap_proxy_ev:.3f} eV"
            )
            axis.set_xlabel("phi")
            axis.set_ylabel("eV")
            self.cluster_popup_spectra_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        self.cluster_popup_spectra_canvas.draw_idle()

    def _cluster_color_lookup(self, cluster_result: SpectralClusterResult) -> dict[int, tuple[float, float, float, float]]:
        cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(1, len(cluster_result.cluster_stats)))
        return {
            stats.cluster_id: cmap(index)
            for index, stats in enumerate(cluster_result.cluster_stats)
        }

    def _cluster_metric_values(self, cluster_result: SpectralClusterResult, metric_key: str) -> np.ndarray:
        metric_maps = {
            "ef_fraction": cluster_result.ef_fraction_map,
            "total_intensity": cluster_result.total_intensity_map,
            "spectral_entropy": cluster_result.spectral_entropy_map,
            "e_centroid": cluster_result.e_centroid_map,
        }
        metric_map = metric_maps[metric_key]
        return np.asarray(metric_map[cluster_result.valid_mask], dtype=np.float32)

    def _refresh_cluster_plot(self) -> None:
        if self.cluster_result is None:
            self._render_cluster_placeholder()
            return

        cluster_result = self.cluster_result
        selected_cluster_id = self._selected_cluster_id()
        cluster_colors = self._cluster_color_lookup(cluster_result)
        cluster_ids = cluster_result.cluster_ids
        labels = np.asarray(cluster_result.cluster_map[cluster_result.valid_mask], dtype=int)

        self.cluster_figure.clear()
        axes = self.cluster_figure.subplots(2, 2)
        map_axis, embedding_axis = axes[0]
        profile_axis, spectrum_axis = axes[1]

        display = cluster_result.cluster_map.astype(float).copy()
        display[~cluster_result.valid_mask] = np.nan
        cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(1, len(cluster_ids))).copy()
        cmap.set_bad((0.0, 0.0, 0.0, 0.0))

        map_axis.imshow(cluster_result.total_intensity_map.T, origin="lower", cmap="gray", aspect="auto")
        image = map_axis.imshow(
            display.T,
            origin="lower",
            cmap=cmap,
            vmin=0,
            vmax=max(0, len(cluster_ids) - 1),
            alpha=0.58,
            aspect="auto",
        )
        if selected_cluster_id is not None and np.any(cluster_result.cluster_map == selected_cluster_id):
            map_axis.contour(
                (cluster_result.cluster_map == selected_cluster_id).T.astype(float),
                levels=[0.5],
                colors=["white"],
                linewidths=1.4,
            )
        map_axis.set_title(f"Spatial clusters over raw total intensity\n{cluster_result.state_name}")
        map_axis.set_xlabel("x index")
        map_axis.set_ylabel("y index")
        cbar = self.cluster_figure.colorbar(image, ax=map_axis, fraction=0.046, pad=0.04, ticks=np.arange(len(cluster_ids)))
        cbar.ax.set_yticklabels([f"C{cluster_id}" for cluster_id in cluster_ids])

        color_mode = self.CLUSTER_COLOR_OPTIONS.get(self.cluster_color_var.get(), "ef_fraction")
        if color_mode == "cluster":
            for stats in cluster_result.cluster_stats:
                mask = labels == stats.cluster_id
                embedding_axis.scatter(
                    cluster_result.embedding_2d[mask, 0],
                    cluster_result.embedding_2d[mask, 1],
                    s=14,
                    alpha=0.70,
                    color=cluster_colors[stats.cluster_id],
                    label=f"C{stats.cluster_id}",
                    linewidths=0,
                )
            if len(cluster_result.cluster_stats) <= 8:
                embedding_axis.legend(loc="best", fontsize=8)
        else:
            metric_values = self._cluster_metric_values(cluster_result, color_mode)
            scatter = embedding_axis.scatter(
                cluster_result.embedding_2d[:, 0],
                cluster_result.embedding_2d[:, 1],
                c=metric_values,
                cmap="viridis",
                s=14,
                alpha=0.75,
                linewidths=0,
            )
            cbar = self.cluster_figure.colorbar(scatter, ax=embedding_axis, fraction=0.046, pad=0.04)
            cbar.set_label(color_mode.replace("_", " "))

        if selected_cluster_id is not None:
            selected_mask = labels == selected_cluster_id
            embedding_axis.scatter(
                cluster_result.embedding_2d[selected_mask, 0],
                cluster_result.embedding_2d[selected_mask, 1],
                facecolors="none",
                edgecolors="black",
                s=28,
                linewidths=0.8,
            )

        explained = cluster_result.embedding_explained_ratio
        x_label = "PC1"
        y_label = "PC2"
        if explained.size >= 2:
            x_label = f"PC1 ({explained[0] * 100:.1f}%)"
            y_label = f"PC2 ({explained[1] * 100:.1f}%)"
        embedding_axis.set_title(f"Spectral embedding colored by {self.cluster_color_var.get().lower()}")
        embedding_axis.set_xlabel(x_label)
        embedding_axis.set_ylabel(y_label)

        for stats in cluster_result.cluster_stats:
            profile = np.asarray(stats.mean_energy_profile, dtype=np.float32).copy()
            max_value = float(np.nanmax(profile)) if profile.size else 0.0
            if max_value > 0:
                profile /= max_value
            line_width = 2.6 if stats.cluster_id == selected_cluster_id else 1.7
            alpha = 1.0 if selected_cluster_id is None or stats.cluster_id == selected_cluster_id else 0.45
            profile_axis.plot(
                cluster_result.e_axis,
                profile,
                color=cluster_colors[stats.cluster_id],
                linewidth=line_width,
                alpha=alpha,
                label=f"C{stats.cluster_id}",
            )
        profile_axis.axvline(0.0, linestyle="--", color="#555555", linewidth=1.0)
        profile_axis.set_title("Mean energy profiles by cluster (summed over phi)")
        profile_axis.set_xlabel("eV")
        profile_axis.set_ylabel("normalized intensity")
        profile_axis.legend(loc="best", fontsize=8)

        selected_stats = next((stats for stats in cluster_result.cluster_stats if stats.cluster_id == selected_cluster_id), None)
        if selected_stats is None and cluster_result.cluster_stats:
            selected_stats = cluster_result.cluster_stats[0]
        if selected_stats is None:
            spectrum_axis.text(0.5, 0.5, "No cluster selected.", ha="center", va="center")
            spectrum_axis.set_axis_off()
        else:
            heatmap = spectrum_axis.imshow(
                selected_stats.mean_spectrum,
                origin="lower",
                aspect="auto",
                extent=[
                    float(cluster_result.phi_axis[0]),
                    float(cluster_result.phi_axis[-1]),
                    float(cluster_result.e_axis[0]),
                    float(cluster_result.e_axis[-1]),
                ],
                cmap="viridis",
            )
            spectrum_axis.set_title(
                f"C{selected_stats.cluster_id} mean spectrum\n{selected_stats.candidate_label}"
            )
            spectrum_axis.set_xlabel("phi")
            spectrum_axis.set_ylabel("eV")
            self.cluster_figure.colorbar(heatmap, ax=spectrum_axis, fraction=0.046, pad=0.04)

        self.cluster_canvas.draw_idle()
        self._update_cluster_summary_text()

    def _update_cluster_summary_text(self) -> None:
        if self.cluster_result is None:
            self._set_text_widget(self.cluster_text, "")
            return

        summary = self.cluster_result.summarize()
        lines = [
            f"State: {summary['state_name']}",
            f"File: {summary['state_file']}",
            f"Method: {summary['method_label']}",
            f"Resource level: {summary['resource_level']}",
            f"Cluster inertia: {summary['cluster_inertia']:.4g}",
            f"Embedding explained ratio: {', '.join(f'{value:.3f}' for value in summary['embedding_explained_ratio'])}",
            "",
            "Candidate labels are heuristic clustering hints based on mean near-EF weight and cluster footprint.",
            "Use the mean spectra and spatial coherence to decide whether a cluster is truly insulating, written-metastable, intermediate, or patch-like.",
            "",
            "Clusters:",
        ]

        for cluster in summary["clusters"]:
            lines.append(
                "  - "
                f"C{cluster['cluster_id']} ({cluster['candidate_label']}): "
                f"{cluster['pixel_count']} px ({cluster['pixel_fraction']:.1%}), "
                f"<ef_fraction>={cluster['mean_ef_fraction']:.5f}, "
                f"<entropy>={cluster['mean_spectral_entropy']:.5f}, "
                f"<e_centroid>={cluster['mean_e_centroid']:.5f}, "
                f"patches={cluster['connected_components']}, "
                f"largest patch={cluster['dominant_component_fraction']:.1%}, "
                f"separation={cluster['separation_ratio']:.2f}"
            )

        if summary["notes"]:
            lines.extend(["", "Notes:"])
            for note in summary["notes"]:
                lines.append(f"  - {note}")

        self._set_text_widget(self.cluster_text, "\n".join(lines))

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

        elif view == "State comparison":
            from_index = self._current_compare_index(self.compare_from_var.get(), fallback=0)
            to_index = self._current_compare_index(self.compare_to_var.get(), fallback=min(1, self.result.n_states - 1))
            self._plot_comparison_view(from_index, to_index)

        else:
            axis = self.main_figure.add_subplot(111)
            axis.text(0.5, 0.5, f"Unsupported view: {view}", ha="center", va="center")
            axis.set_axis_off()

        self.main_canvas.draw_idle()

    def _infer_opposite_pair(self, from_index: int, to_index: int) -> tuple[int, int] | None:
        assert self.result is not None
        n = self.result.n_states
        if n == 2:
            return (to_index, from_index)
        if n == 4:
            remaining = sorted({0, 1, 2, 3} - {from_index, to_index})
            if len(remaining) == 2:
                return (remaining[0], remaining[1])
        return None

    def _compute_state_boundaries(self, code_map: np.ndarray) -> np.ndarray:
        boundary = np.zeros(code_map.shape, dtype=bool)
        for axis_index in [0, 1]:
            shifted = np.roll(code_map, -1, axis=axis_index)
            diff = (code_map != shifted) & (code_map >= 0) & (shifted >= 0)
            boundary |= diff
            boundary |= np.roll(diff, 1, axis=axis_index)
        assert self.result is not None
        return boundary & self.result.valid_mask

    def _plot_comparison_view(self, from_index: int, to_index: int) -> None:
        from matplotlib.lines import Line2D

        assert self.result is not None
        r = self.result

        n_simple = len(SIMPLE_STATE_NAMES)
        state_cmap = mcolors.ListedColormap([SIMPLE_STATE_COLORS[name] for name in SIMPLE_STATE_NAMES])
        state_cmap.set_bad(color="lightgray")
        state_norm = mcolors.BoundaryNorm(np.arange(-0.5, n_simple + 0.5, 1), state_cmap.N)
        state_short = ["I", "X", "M"]

        transition_labels = [
            "I \u2192 I", "I \u2192 X", "I \u2192 M",
            "X \u2192 I", "X \u2192 X", "X \u2192 M",
            "M \u2192 I", "M \u2192 X", "M \u2192 M",
        ]
        transition_colors = [
            "#1f3b73",  # I→I stable insulating
            "#6fa8dc",  # I→X
            "#ff6600",  # I→M strong warming
            "#a4c2f4",  # X→I
            "#aaaaaa",  # X→X stable intermediate
            "#ff9900",  # X→M
            "#0a42a8",  # M→I strong cooling
            "#6d9eeb",  # M→X
            "#d62728",  # M→M stable metallic
        ]
        trans_cmap = mcolors.ListedColormap(transition_colors)
        trans_cmap.set_bad(color="lightgray")
        trans_norm = mcolors.BoundaryNorm(np.arange(-0.5, 9.5, 1), trans_cmap.N)

        def make_transition_map(fi: int, ti: int) -> tuple[np.ndarray, np.ndarray]:
            fc = r.simple_state_code_maps[fi]
            tc = r.simple_state_code_maps[ti]
            raw = np.full(fc.shape, fill_value=-1, dtype=int)
            valid = r.valid_mask & (fc >= 0) & (tc >= 0)
            raw[valid] = fc[valid] * 3 + tc[valid]
            display = raw.astype(float)
            display[~r.valid_mask] = np.nan
            return raw, display

        def make_stat_matrix(raw: np.ndarray) -> np.ndarray:
            mat = np.zeros((3, 3), dtype=int)
            for f in range(3):
                for t in range(3):
                    mat[f, t] = int(np.sum(raw == f * 3 + t))
            return mat

        from_map = r.simple_state_code_maps[from_index].astype(float).copy()
        to_map = r.simple_state_code_maps[to_index].astype(float).copy()
        from_map[~r.valid_mask] = np.nan
        to_map[~r.valid_mask] = np.nan

        trans_raw, trans_display = make_transition_map(from_index, to_index)
        stat_matrix = make_stat_matrix(trans_raw)
        from_boundary = self._compute_state_boundaries(r.simple_state_code_maps[from_index])
        to_boundary = self._compute_state_boundaries(r.simple_state_code_maps[to_index])
        opp = self._infer_opposite_pair(from_index, to_index)

        axes = self.main_figure.subplots(2, 3)

        # [0,0] Before state map
        ax = axes[0, 0]
        img = ax.imshow(from_map.T, origin="lower", cmap=state_cmap, norm=state_norm, aspect="auto")
        ax.set_title(f"Before: {r.state_names[from_index]}")
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        cbar = self.main_figure.colorbar(img, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(n_simple))
        cbar.ax.set_yticklabels(state_short)
        self._mark_selected_pixel(ax)

        # [0,1] After state map
        ax = axes[0, 1]
        img = ax.imshow(to_map.T, origin="lower", cmap=state_cmap, norm=state_norm, aspect="auto")
        ax.set_title(f"After: {r.state_names[to_index]}")
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        cbar = self.main_figure.colorbar(img, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(n_simple))
        cbar.ax.set_yticklabels(state_short)
        self._mark_selected_pixel(ax)

        # [0,2] Transition map
        ax = axes[0, 2]
        img = ax.imshow(trans_display.T, origin="lower", cmap=trans_cmap, norm=trans_norm, aspect="auto")
        ax.set_title(f"Transition map\n{r.state_names[from_index]} \u2192 {r.state_names[to_index]}")
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        cbar = self.main_figure.colorbar(img, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(9))
        cbar.ax.set_yticklabels(transition_labels, fontsize=7)
        self._mark_selected_pixel(ax)

        # [1,0] Boundary overlay
        ax = axes[1, 0]
        ax.imshow(r.average_normalized_total_map.T, origin="lower", cmap="gray", aspect="auto")
        fy, fx = np.where(from_boundary.T)
        if len(fx):
            ax.scatter(fx, fy, s=2, c="#00ccff", alpha=0.85, linewidths=0)
        ty, tx = np.where(to_boundary.T)
        if len(tx):
            ax.scatter(tx, ty, s=2, c="#ff6600", alpha=0.85, linewidths=0)
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#00ccff", markersize=6, label=f"Before ({r.state_names[from_index]})"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff6600", markersize=6, label=f"After ({r.state_names[to_index]})"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=7)
        ax.set_title("Boundary overlay")
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        self._mark_selected_pixel(ax)

        # [1,1] Transition statistics matrix
        ax = axes[1, 1]
        vmax = max(1, int(stat_matrix.max()))
        im = ax.imshow(stat_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["\u2192 I", "\u2192 X", "\u2192 M"])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["I \u2192", "X \u2192", "M \u2192"])
        ax.set_title("Transition statistics\n(pixel counts)")
        ax.set_xlabel("To state")
        ax.set_ylabel("From state")
        for (row, col), count in np.ndenumerate(stat_matrix):
            text_color = "white" if count > vmax * 0.65 else "black"
            ax.text(col, row, str(count), ha="center", va="center", fontsize=9, color=text_color)
        self.main_figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # [1,2] Opposite direction transition map
        ax = axes[1, 2]
        if opp is not None:
            opp_from, opp_to = opp
            _, opp_display = make_transition_map(opp_from, opp_to)
            opp_raw, _ = make_transition_map(opp_from, opp_to)
            img = ax.imshow(opp_display.T, origin="lower", cmap=trans_cmap, norm=trans_norm, aspect="auto")
            ax.set_title(f"Opposite direction\n{r.state_names[opp_from]} \u2192 {r.state_names[opp_to]}")
            ax.set_xlabel("x index")
            ax.set_ylabel("y index")
            cbar = self.main_figure.colorbar(img, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(9))
            cbar.ax.set_yticklabels(transition_labels, fontsize=7)
            self._mark_selected_pixel(ax)
        else:
            ax.text(
                0.5, 0.5,
                "Opposite direction unavailable.\nLoad exactly 2 or 4 states to enable.",
                ha="center", va="center", fontsize=10, transform=ax.transAxes,
            )
            ax.set_axis_off()

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
    parser.add_argument(
        "--upload-first",
        action="store_true",
        help="Start with a file-upload gate, then preload all analysis views from that upload set.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.headless_smoke_test:
        AnalysisParameters().validate()
        InitialTransitionFeatureParameters().validate()
        SwitchingMechanismParameters().validate()
        print("Desktop app imports and parameter validation succeeded.")
        return 0

    initial_files = [str(Path(path).expanduser().resolve()) for path in args.files]
    if len(initial_files) > 4 and not args.upload_first:
        parser.error("At most four files can be preloaded.")

    root = tk.Tk()
    AnalysisApp(root, initial_files=initial_files, upload_first=args.upload_first)
    root.mainloop()
    return 0
