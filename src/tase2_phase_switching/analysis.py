from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import json
from itertools import combinations
import math
import os
import re
from typing import Any

import numpy as np
from scipy import ndimage, signal, stats
from scipy.spatial import cKDTree
import xarray as xr


REQUIRED_DIMS = ("x", "y", "eV", "phi")
TABLE_DATA_EXTENSIONS = {".csv", ".tsv", ".txt", ".dat"}
NUMPY_DATA_EXTENSIONS = {".npy", ".npz"}
SIMPLE_STATE_NAMES = ("insulating", "intermediate", "metallic")
SIMPLE_STATE_COLORS = {
    "insulating": "#1f3b73",
    "intermediate": "#ffbf00",
    "metallic": "#d62728",
}
SIMPLE_STATE_SHORT = {
    "insulating": "I",
    "intermediate": "X",
    "metallic": "M",
}
SPECTRAL_CLUSTER_METHODS = (
    "compressed_features",
    "downsampled_spectra_pca",
    "full_spectra_pca",
)
SPECTRAL_CLUSTER_METHOD_LABELS = {
    "compressed_features": "Compressed spectral summaries (Low, safest)",
    "downsampled_spectra_pca": "Downsampled registered spectra + PCA (Medium)",
    "full_spectra_pca": "Full registered spectra + PCA (High, guarded)",
}
SPECTRAL_CLUSTER_METHOD_RESOURCES = {
    "compressed_features": "Low",
    "downsampled_spectra_pca": "Medium",
    "full_spectra_pca": "High",
}
STATE_CLASSIFICATION_LABELS = (
    "invalid / low signal",
    "CCDW / insulating",
    "metastable metallic",
    "intermediate / erased memory",
    "boundary / mixed",
    "structural / orientation variant",
)
STATE_CLASSIFICATION_COLORS = {
    "invalid / low signal": "#111111",
    "CCDW / insulating": "#1f3b73",
    "metastable metallic": "#d62728",
    "intermediate / erased memory": "#ffbf00",
    "boundary / mixed": "#7f7f7f",
    "structural / orientation variant": "#2ca02c",
}
STATE_CLASSIFICATION_FEATURE_NAMES = (
    "T",
    "W_EF",
    "W_LHB",
    "I_rat",
    "E_LHB",
    "E_LE",
    "Gamma_EDC",
    "S_orient",
)
TILT_DEFECT_LABELS = (
    "none",
    "high tilt",
    "sharp tilt boundary",
    "rough edge / dislocation",
)
TILT_DEFECT_COLORS = {
    "none": "#000000",
    "high tilt": "#fdae6b",
    "sharp tilt boundary": "#e6550d",
    "rough edge / dislocation": "#6a51a3",
}
SWITCHING_LABELS = (
    "stable / unchanged",
    "written / becomes metallic",
    "erased / becomes less metallic",
    "reversible / memory-like",
    "ambiguous",
)
SWITCHING_COLORS = {
    "stable / unchanged": "#1f3b73",
    "written / becomes metallic": "#d62728",
    "erased / becomes less metallic": "#2ca02c",
    "reversible / memory-like": "#9467bd",
    "ambiguous": "#7f7f7f",
}
TRANSITION_OUTCOME_LABELS = (
    "invalid / low signal",
    "unchanged",
    "written / more metallic",
    "erased / less metallic",
    "strongly written",
    "strongly erased",
)
TRANSITION_OUTCOME_COLORS = {
    "invalid / low signal": "#111111",
    "unchanged": "#9a9a9a",
    "written / more metallic": "#e6550d",
    "erased / less metallic": "#3182bd",
    "strongly written": "#a50f15",
    "strongly erased": "#08519c",
}
INITIAL_TRANSITION_MODES = ("sequential", "initial_reference")
INITIAL_TRANSITION_NORMALIZATION_MODES = (
    "none",
    "total_intensity",
    "median_near_ef",
    "high_percentile",
)
INITIAL_TRANSITION_GROUPS = (
    "future metallic",
    "future erased",
    "both metallic and erased",
    "stable",
    "never switched",
)
SWITCHING_MECHANISM_EDC_NORMALIZATIONS = (
    "raw",
    "per_pixel_max",
    "total_spectral_weight",
    "feature_window",
    "near_ef",
)
SWITCHING_MECHANISM_SPECTRAL_FEATURES = (
    "I_rat_A0",
    "W_EF_A0",
    "W_LHB_A0",
    "near_EF_intensity_A0",
    "feature_window_intensity_A0",
    "edc_peak_energy_A0",
    "edc_peak_amplitude_A0",
    "edc_peak_width_A0",
    "total_spectral_weight_A0",
    "edc_center_of_mass_A0",
    "edc_asymmetry_A0",
    "initial_MDC_peak_position_A0",
    "initial_MDC_peak_width_A0",
)


@dataclass(slots=True)
class AnalysisParameters:
    fermi_level_ev: float = 0.0
    ef_window_ev: float = 0.05
    wide_window_ev: float = 0.20
    n_clusters: int = 6
    n_pca_components: int = 8
    cross_threshold_quantile: float = 0.45
    cross_row_fraction: float = 0.18
    cross_col_fraction: float = 0.18
    cross_background_quantile: float = 0.10
    cross_pad: int = 1
    simple_state_low_quantile: float = 0.30
    simple_state_high_quantile: float = 0.70

    def validate(self) -> None:
        quantiles = {
            "cross_threshold_quantile": self.cross_threshold_quantile,
            "cross_background_quantile": self.cross_background_quantile,
            "simple_state_low_quantile": self.simple_state_low_quantile,
            "simple_state_high_quantile": self.simple_state_high_quantile,
        }
        for name, value in quantiles.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}.")

        fractions = {
            "cross_row_fraction": self.cross_row_fraction,
            "cross_col_fraction": self.cross_col_fraction,
        }
        for name, value in fractions.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}.")

        if self.ef_window_ev <= 0:
            raise ValueError("ef_window_ev must be positive.")
        if self.wide_window_ev <= 0:
            raise ValueError("wide_window_ev must be positive.")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        if self.n_pca_components < 1:
            raise ValueError("n_pca_components must be at least 1.")
        if self.cross_pad < 0:
            raise ValueError("cross_pad must be non-negative.")
        if self.simple_state_low_quantile >= self.simple_state_high_quantile:
            raise ValueError(
                "simple_state_low_quantile must be smaller than simple_state_high_quantile."
            )


@dataclass(slots=True)
class LoadedState:
    name: str
    file_path: str
    data_array: xr.DataArray
    dataset: xr.Dataset | None = None


@dataclass(slots=True)
class SpatialAlignmentRecord:
    state_name: str
    file_path: str
    original_shape: tuple[int, int]
    aligned_shape: tuple[int, int]
    x_slice: tuple[int, int]
    y_slice: tuple[int, int]
    method: str
    score: float | None = None


@dataclass(slots=True)
class AnalysisResult:
    parameters: AnalysisParameters
    loaded_states: list[LoadedState]
    feature_names: list[str]
    valid_mask: np.ndarray
    average_normalized_total_map: np.ndarray
    active_mask: np.ndarray
    row_occupancy: np.ndarray
    col_occupancy: np.ndarray
    total_maps: list[np.ndarray]
    ef_maps: list[np.ndarray]
    features_by_state: list[dict[str, np.ndarray]]
    cluster_maps: list[np.ndarray]
    raw_cluster_maps: list[np.ndarray]
    raw_to_ordered_cluster: dict[int, int]
    cluster_mean_ef_fraction: dict[int, float]
    cluster_sequence_strings: np.ndarray
    cluster_sequence_code_map: np.ndarray
    cluster_sequences: list[tuple[str, int]]
    cluster_sequence_to_code: dict[str, int]
    simple_state_label_maps: list[np.ndarray]
    simple_state_code_maps: list[np.ndarray]
    simple_state_thresholds: tuple[float, float]
    simple_state_sequence_strings: np.ndarray
    simple_state_sequence_code_map: np.ndarray
    simple_state_sequences: list[tuple[str, int]]
    simple_state_sequence_to_code: dict[str, int]
    pca_explained_ratio: np.ndarray
    cluster_centroids: np.ndarray
    cluster_inertia: float
    cluster_counts_by_state: list[dict[int, int]]
    notes: list[str] = field(default_factory=list)

    @property
    def state_names(self) -> list[str]:
        return [state.name for state in self.loaded_states]

    @property
    def file_paths(self) -> list[str]:
        return [state.file_path for state in self.loaded_states]

    @property
    def n_states(self) -> int:
        return len(self.loaded_states)

    @property
    def shape(self) -> tuple[int, int]:
        return self.valid_mask.shape

    @property
    def e_axis(self) -> np.ndarray:
        return np.asarray(self.loaded_states[0].data_array.coords["eV"].values, dtype=np.float32)

    @property
    def phi_axis(self) -> np.ndarray:
        return np.asarray(self.loaded_states[0].data_array.coords["phi"].values, dtype=np.float32)

    def summarize(self, max_sequences: int = 12) -> dict[str, Any]:
        return build_summary_dict(self, max_sequences=max_sequences)


@dataclass(slots=True)
class SpectralClusterParameters:
    n_clusters: int = 4
    embedding_components: int = 6
    method_key: str = "downsampled_spectra_pca"
    n_init: int = 12
    n_iter: int = 100
    seed: int = 42

    def validate(self) -> None:
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        if self.embedding_components < 2:
            raise ValueError("embedding_components must be at least 2.")
        if self.method_key not in SPECTRAL_CLUSTER_METHODS:
            supported = ", ".join(SPECTRAL_CLUSTER_METHODS)
            raise ValueError(f"method_key must be one of {supported}, got {self.method_key!r}.")
        if self.n_init < 1:
            raise ValueError("n_init must be at least 1.")
        if self.n_iter < 1:
            raise ValueError("n_iter must be at least 1.")


@dataclass(slots=True)
class StateClassifierParameters:
    fermi_level_ev: float = 0.0
    ef_min_ev: float = -0.05
    ef_max_ev: float = 0.0
    lhb_center_ev: float = -0.18
    lhb_halfwidth_ev: float = 0.05
    leading_edge_min_ev: float = -0.25
    leading_edge_max_ev: float = 0.05
    p3_center_ev: float = -0.35
    p3_halfwidth_ev: float = 0.05
    smooth_sigma: float = 1.0
    low_quantile: float = 0.30
    high_quantile: float = 0.70
    broad_quantile: float = 0.80
    orientation_quantile: float = 0.80
    low_signal_quantile: float = 0.05
    lhb_min_quantile: float = 0.05
    epsilon: float = 1e-8
    use_spatial_boundary: bool = True

    def validate(self) -> None:
        if self.ef_min_ev >= self.ef_max_ev:
            raise ValueError("ef_min_ev must be smaller than ef_max_ev.")
        if self.lhb_halfwidth_ev <= 0:
            raise ValueError("lhb_halfwidth_ev must be positive.")
        if self.leading_edge_min_ev >= self.leading_edge_max_ev:
            raise ValueError("leading_edge_min_ev must be smaller than leading_edge_max_ev.")
        if self.p3_halfwidth_ev <= 0:
            raise ValueError("p3_halfwidth_ev must be positive.")
        if self.smooth_sigma < 0:
            raise ValueError("smooth_sigma must be non-negative.")
        quantiles = {
            "low_quantile": self.low_quantile,
            "high_quantile": self.high_quantile,
            "broad_quantile": self.broad_quantile,
            "orientation_quantile": self.orientation_quantile,
            "low_signal_quantile": self.low_signal_quantile,
            "lhb_min_quantile": self.lhb_min_quantile,
        }
        for name, value in quantiles.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}.")
        if self.low_quantile >= self.high_quantile:
            raise ValueError("low_quantile must be smaller than high_quantile.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")


@dataclass(slots=True)
class SwitchingMapParameters:
    fermi_level_ev: float = 0.0
    ef_min_ev: float = -0.05
    ef_max_ev: float = 0.0
    lhb_center_ev: float = -0.18
    lhb_halfwidth_ev: float = 0.05
    smooth_sigma: float = 1.0
    low_switch_quantile: float = 0.30
    high_switch_quantile: float = 0.75
    small_net_quantile: float = 0.35
    low_signal_quantile: float = 0.05
    lhb_min_quantile: float = 0.05
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.ef_min_ev >= self.ef_max_ev:
            raise ValueError("ef_min_ev must be smaller than ef_max_ev.")
        if self.lhb_halfwidth_ev <= 0:
            raise ValueError("lhb_halfwidth_ev must be positive.")
        if self.smooth_sigma < 0:
            raise ValueError("smooth_sigma must be non-negative.")
        quantiles = {
            "low_switch_quantile": self.low_switch_quantile,
            "high_switch_quantile": self.high_switch_quantile,
            "small_net_quantile": self.small_net_quantile,
            "low_signal_quantile": self.low_signal_quantile,
            "lhb_min_quantile": self.lhb_min_quantile,
        }
        for name, value in quantiles.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}.")
        if self.low_switch_quantile >= self.high_switch_quantile:
            raise ValueError("low_switch_quantile must be smaller than high_switch_quantile.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")


@dataclass(slots=True)
class StatePredictionParameters:
    fermi_level_ev: float = 0.0
    ef_min_ev: float = -0.05
    ef_max_ev: float = 0.0
    lhb_center_ev: float = -0.18
    lhb_halfwidth_ev: float = 0.05
    leading_edge_min_ev: float = -0.25
    leading_edge_max_ev: float = 0.05
    p3_center_ev: float = -0.35
    p3_halfwidth_ev: float = 0.05
    smooth_sigma: float = 1.0
    stable_quantile: float = 0.20
    switch_quantile: float = 0.80
    net_change_tau: float | None = None
    low_signal_quantile: float = 0.05
    lhb_min_quantile: float = 0.05
    phase_low_quantile: float = 0.30
    phase_high_quantile: float = 0.70
    structural_gradient_quantile: float = 0.80
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.ef_min_ev >= self.ef_max_ev:
            raise ValueError("ef_min_ev must be smaller than ef_max_ev.")
        if self.lhb_halfwidth_ev <= 0:
            raise ValueError("lhb_halfwidth_ev must be positive.")
        if self.leading_edge_min_ev >= self.leading_edge_max_ev:
            raise ValueError("leading_edge_min_ev must be smaller than leading_edge_max_ev.")
        if self.p3_halfwidth_ev <= 0:
            raise ValueError("p3_halfwidth_ev must be positive.")
        if self.smooth_sigma < 0:
            raise ValueError("smooth_sigma must be non-negative.")
        quantiles = {
            "stable_quantile": self.stable_quantile,
            "switch_quantile": self.switch_quantile,
            "low_signal_quantile": self.low_signal_quantile,
            "lhb_min_quantile": self.lhb_min_quantile,
            "phase_low_quantile": self.phase_low_quantile,
            "phase_high_quantile": self.phase_high_quantile,
            "structural_gradient_quantile": self.structural_gradient_quantile,
        }
        for name, value in quantiles.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}.")
        if self.stable_quantile >= self.switch_quantile:
            raise ValueError("stable_quantile must be smaller than switch_quantile.")
        if self.phase_low_quantile >= self.phase_high_quantile:
            raise ValueError("phase_low_quantile must be smaller than phase_high_quantile.")
        if self.net_change_tau is not None and self.net_change_tau < 0:
            raise ValueError("net_change_tau must be non-negative when provided.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")


@dataclass(slots=True)
class TransitionOutcomeParameters:
    fermi_level_ev: float = 0.0
    ef_min_ev: float = -0.05
    ef_max_ev: float = 0.0
    lhb_center_ev: float = -0.18
    lhb_halfwidth_ev: float = 0.05
    smooth_sigma: float = 1.0
    user_min_tau: float = 0.0
    strong_tau_multiplier: float = 2.0
    use_relative_delta: bool = False
    low_signal_quantile: float = 0.05
    lhb_min_quantile: float = 0.05
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.ef_min_ev >= self.ef_max_ev:
            raise ValueError("ef_min_ev must be smaller than ef_max_ev.")
        if self.lhb_halfwidth_ev <= 0:
            raise ValueError("lhb_halfwidth_ev must be positive.")
        if self.smooth_sigma < 0:
            raise ValueError("smooth_sigma must be non-negative.")
        if self.user_min_tau < 0:
            raise ValueError("user_min_tau must be non-negative.")
        if self.strong_tau_multiplier < 1.0:
            raise ValueError("strong_tau_multiplier must be at least 1.")
        for name, value in {
            "low_signal_quantile": self.low_signal_quantile,
            "lhb_min_quantile": self.lhb_min_quantile,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")


@dataclass(slots=True)
class SpectralClusterStats:
    cluster_id: int
    pixel_count: int
    pixel_fraction: float
    mean_ef_fraction: float
    mean_total_intensity: float
    mean_spectral_entropy: float
    mean_e_centroid: float
    connected_components: int
    dominant_component_fraction: float
    intra_cluster_rms: float
    nearest_cluster_distance: float
    separation_ratio: float
    embedding_center: np.ndarray
    mean_energy_profile: np.ndarray
    mean_spectrum: np.ndarray
    candidate_label: str


@dataclass(slots=True)
class SpectralClusterResult:
    state_name: str
    state_file: str
    parameters: SpectralClusterParameters
    valid_mask: np.ndarray
    cluster_map: np.ndarray
    raw_cluster_map: np.ndarray
    cluster_counts: dict[int, int]
    cluster_stats: list[SpectralClusterStats]
    cluster_centroids: np.ndarray
    cluster_inertia: float
    embedding_2d: np.ndarray
    embedding_explained_ratio: np.ndarray
    pixel_coordinates: np.ndarray
    total_intensity_map: np.ndarray
    ef_fraction_map: np.ndarray
    spectral_entropy_map: np.ndarray
    e_centroid_map: np.ndarray
    feature_maps: dict[str, np.ndarray]
    e_axis: np.ndarray
    phi_axis: np.ndarray
    notes: list[str] = field(default_factory=list)

    @property
    def cluster_ids(self) -> list[int]:
        return [stats.cluster_id for stats in self.cluster_stats]

    def summarize(self) -> dict[str, Any]:
        return build_spectral_cluster_summary(self)


@dataclass(slots=True)
class ClusterPhysicalMetrics:
    cluster_id: int
    candidate_label: str
    pixel_count: int
    pixel_fraction: float
    fermi_weight_fraction: float
    gap_fill_ratio: float
    gap_proxy_ev: float
    dominant_peak_ev: float
    secondary_peak_ev: float
    dominant_peak_width_ev: float
    dispersion_slope_phi_per_ev: float
    dispersion_curvature_phi_per_ev2: float
    deep_weight_fraction: float
    shallow_weight_fraction: float
    near_ef_weight_fraction: float
    ridge_coverage_fraction: float


@dataclass(slots=True)
class ClusterPairwisePhysicalDifference:
    cluster_a: int
    cluster_b: int
    fermi_weight_diff: float
    fermi_weight_meaningful: bool
    gap_fill_ratio_diff: float
    gap_proxy_diff_ev: float
    gap_difference_meaningful: bool
    dominant_peak_diff_ev: float
    dominant_peak_meaningful: bool
    dominant_peak_width_diff_ev: float
    peak_width_meaningful: bool
    dispersion_shape_correlation: float
    dispersion_slope_diff: float
    dispersion_curvature_diff: float
    dispersion_meaningful: bool
    deep_weight_diff: float
    shallow_weight_diff: float
    near_ef_weight_diff: float
    spectral_weight_transfer_meaningful: bool
    overall_physically_distinct: bool
    interpretation: str


@dataclass(slots=True)
class ClusterPhysicalQuestionSummary:
    question: str
    answer: str
    strongest_example: str
    reasoning: str


@dataclass(slots=True)
class ClusterPhysicalInterpretation:
    state_name: str
    state_file: str
    metrics_rows: list[ClusterPhysicalMetrics]
    pairwise_rows: list[ClusterPairwisePhysicalDifference]
    question_summaries: list[ClusterPhysicalQuestionSummary]
    findings: list[str]
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StateClassificationResult:
    state: LoadedState
    parameters: StateClassifierParameters
    feature_maps: dict[str, np.ndarray]
    normalized_maps: dict[str, np.ndarray]
    threshold_values: dict[str, float]
    label_map: np.ndarray
    code_map: np.ndarray
    valid_mask: np.ndarray
    orientation_feature_name: str
    counts: dict[str, int]
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.code_map.shape

    @property
    def state_name(self) -> str:
        return self.state.name

    @property
    def file_path(self) -> str:
        return self.state.file_path


@dataclass(slots=True)
class TiltMapParameters:
    band_min_ev: float = -0.30
    band_max_ev: float = 0.05
    phi_reference: float = 0.0
    spatial_smooth_sigma: float = 1.0
    defect_tilt_percentile: float = 95.0
    defect_gradient_percentile: float = 95.0
    low_signal_percentile: float = 8.0
    signal_floor_fraction: float = 0.15
    local_window: int = 5
    group_count: int = 5
    min_group_size: int = 8
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.band_min_ev >= self.band_max_ev:
            raise ValueError("band_min_ev must be smaller than band_max_ev.")
        if self.spatial_smooth_sigma < 0:
            raise ValueError("spatial_smooth_sigma must be non-negative.")
        for name, value in {
            "defect_tilt_percentile": self.defect_tilt_percentile,
            "defect_gradient_percentile": self.defect_gradient_percentile,
            "low_signal_percentile": self.low_signal_percentile,
        }.items():
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} must be between 0 and 100.")
        if self.local_window < 1:
            raise ValueError("local_window must be at least 1.")
        if self.group_count < 2:
            raise ValueError("group_count must be at least 2.")
        if self.min_group_size < 0:
            raise ValueError("min_group_size must be non-negative.")
        if not 0.0 <= self.signal_floor_fraction <= 1.0:
            raise ValueError("signal_floor_fraction must be between 0 and 1.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")


@dataclass(slots=True)
class TiltMapResult:
    state: LoadedState
    parameters: TiltMapParameters
    tilt_map: np.ndarray
    peak_tilt_map: np.ndarray
    band_weight_map: np.ndarray
    phi_width_map: np.ndarray
    tilt_gradient_map: np.ndarray
    local_tilt_std_map: np.ndarray
    defect_score_map: np.ndarray
    group_mean_tilt_map: np.ndarray
    defect_mask: np.ndarray
    defect_type_map: np.ndarray
    group_label_map: np.ndarray
    valid_mask: np.ndarray
    thresholds: dict[str, float]
    group_rows: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.tilt_map.shape

    @property
    def file_path(self) -> str:
        return self.state.file_path

    @property
    def state_name(self) -> str:
        return self.state.name

    @property
    def e_axis(self) -> np.ndarray:
        return np.asarray(self.state.data_array.coords["eV"].values, dtype=np.float32)

    @property
    def phi_axis(self) -> np.ndarray:
        return np.asarray(self.state.data_array.coords["phi"].values, dtype=np.float32)


@dataclass(slots=True)
class SwitchingMapResult:
    loaded_states: list[LoadedState]
    parameters: SwitchingMapParameters
    total_maps: list[np.ndarray]
    w_ef_maps: list[np.ndarray]
    w_lhb_maps: list[np.ndarray]
    i_rat_maps: list[np.ndarray]
    delta_irat_maps: list[np.ndarray]
    initial_delta_irat_maps: list[np.ndarray]
    total_change_map: np.ndarray
    max_change_map: np.ndarray
    net_change_map: np.ndarray
    switching_coefficient_map: np.ndarray
    label_map: np.ndarray
    code_map: np.ndarray
    valid_mask: np.ndarray
    threshold_values: dict[str, float]
    counts: dict[str, int]
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.switching_coefficient_map.shape

    @property
    def state_names(self) -> list[str]:
        return [state.name for state in self.loaded_states]

    @property
    def file_paths(self) -> list[str]:
        return [state.file_path for state in self.loaded_states]

    @property
    def n_states(self) -> int:
        return len(self.loaded_states)

    @property
    def e_axis(self) -> np.ndarray:
        return np.asarray(self.loaded_states[0].data_array.coords["eV"].values, dtype=np.float32)

    @property
    def phi_axis(self) -> np.ndarray:
        return np.asarray(self.loaded_states[0].data_array.coords["phi"].values, dtype=np.float32)


@dataclass(slots=True)
class StatePredictionResult:
    switching_result: SwitchingMapResult
    parameters: StatePredictionParameters
    initial_feature_maps: dict[str, np.ndarray]
    distance_maps: dict[str, np.ndarray]
    average_initial_edcs: dict[str, np.ndarray]
    correlation_values: dict[str, float]
    label_map: np.ndarray
    code_map: np.ndarray
    valid_mask: np.ndarray
    threshold_values: dict[str, float]
    counts: dict[str, int]
    orientation_feature_name: str
    interpretation: str
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.code_map.shape

    @property
    def loaded_states(self) -> list[LoadedState]:
        return self.switching_result.loaded_states

    @property
    def state_names(self) -> list[str]:
        return self.switching_result.state_names

    @property
    def file_paths(self) -> list[str]:
        return self.switching_result.file_paths

    @property
    def n_states(self) -> int:
        return self.switching_result.n_states

    @property
    def e_axis(self) -> np.ndarray:
        return self.switching_result.e_axis

    @property
    def phi_axis(self) -> np.ndarray:
        return self.switching_result.phi_axis


@dataclass(slots=True)
class TransitionOutcomeTransition:
    index: int
    before_index: int
    after_index: int
    pulse_label: str
    delta_irat_map: np.ndarray
    abs_delta_irat_map: np.ndarray
    relative_delta_irat_map: np.ndarray
    delta_w_ef_map: np.ndarray
    delta_w_lhb_map: np.ndarray
    metric_delta_map: np.ndarray
    label_map: np.ndarray
    code_map: np.ndarray
    valid_mask: np.ndarray
    tau: float
    strong_tau: float
    counts: dict[str, int]
    stats: dict[str, float]


@dataclass(slots=True)
class TransitionOutcomeResult:
    loaded_states: list[LoadedState]
    parameters: TransitionOutcomeParameters
    pulse_labels: list[str]
    total_maps: list[np.ndarray]
    w_ef_maps: list[np.ndarray]
    w_lhb_maps: list[np.ndarray]
    i_rat_maps: list[np.ndarray]
    transitions: list[TransitionOutcomeTransition]
    write_count_map: np.ndarray
    erase_count_map: np.ndarray
    activity_count_map: np.ndarray
    repeated_switching_map: np.ndarray
    net_sequence_change_map: np.ndarray
    valid_mask: np.ndarray
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.net_sequence_change_map.shape

    @property
    def state_names(self) -> list[str]:
        return [state.name for state in self.loaded_states]

    @property
    def file_paths(self) -> list[str]:
        return [state.file_path for state in self.loaded_states]

    @property
    def n_states(self) -> int:
        return len(self.loaded_states)

    @property
    def n_transitions(self) -> int:
        return len(self.transitions)

    @property
    def e_axis(self) -> np.ndarray:
        return np.asarray(self.loaded_states[0].data_array.coords["eV"].values, dtype=np.float32)

    @property
    def phi_axis(self) -> np.ndarray:
        return np.asarray(self.loaded_states[0].data_array.coords["phi"].values, dtype=np.float32)


@dataclass(slots=True)
class InitialTransitionFeatureParameters:
    """Parameters for mapping future transition behavior back onto the initial state.

    The canonical data convention used by the loader is x, y, eV, phi. All maps
    returned by this analysis therefore have shape (x, y), even when the UI
    labels axes as x/y for display.
    """

    fermi_level_ev: float = 0.0
    ef_min_ev: float = -0.05
    ef_max_ev: float = 0.05
    lhb_center_ev: float = -0.18
    lhb_halfwidth_ev: float = 0.05
    smooth_sigma: float = 1.0
    feature_min_ev: float = -0.30
    feature_max_ev: float = -0.05
    asymmetry_split_ev: float = -0.15
    metallic_percentile: float = 95.0
    erasure_percentile: float = 95.0
    stable_percentile: float = 25.0
    transition_mode: str = "sequential"
    reference_index: int = 0
    normalization_mode: str = "none"
    allow_overlap: bool = True
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.ef_min_ev >= self.ef_max_ev:
            raise ValueError("ef_min_ev must be smaller than ef_max_ev.")
        if self.feature_min_ev >= self.feature_max_ev:
            raise ValueError("feature_min_ev must be smaller than feature_max_ev.")
        if self.lhb_halfwidth_ev <= 0:
            raise ValueError("lhb_halfwidth_ev must be positive.")
        if self.smooth_sigma < 0:
            raise ValueError("smooth_sigma must be non-negative.")
        for name, value in {
            "metallic_percentile": self.metallic_percentile,
            "erasure_percentile": self.erasure_percentile,
            "stable_percentile": self.stable_percentile,
        }.items():
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} must be between 0 and 100, got {value}.")
        if self.transition_mode not in INITIAL_TRANSITION_MODES:
            raise ValueError(f"transition_mode must be one of {INITIAL_TRANSITION_MODES}.")
        if self.normalization_mode not in INITIAL_TRANSITION_NORMALIZATION_MODES:
            raise ValueError(f"normalization_mode must be one of {INITIAL_TRANSITION_NORMALIZATION_MODES}.")
        if self.reference_index < 0:
            raise ValueError("reference_index must be non-negative.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")


@dataclass(slots=True)
class InitialTransitionPairMetrics:
    index: int
    before_index: int
    after_index: int
    name: str
    metallicity_score: np.ndarray
    erasure_score: np.ndarray
    transition_magnitude: np.ndarray
    metallicity_score_norm: np.ndarray
    erasure_score_norm: np.ndarray
    transition_magnitude_norm: np.ndarray
    metallic_mask: np.ndarray
    erased_mask: np.ndarray
    stable_mask: np.ndarray
    metallic_threshold: float
    erasure_threshold: float
    stable_threshold: float


@dataclass(slots=True)
class InitialTransitionFeatureResult:
    loaded_states: list[LoadedState]
    parameters: InitialTransitionFeatureParameters
    transitions: list[InitialTransitionPairMetrics]
    initial_reference_index: int
    initial_near_ef_map: np.ndarray
    initial_feature_map: np.ndarray
    initial_feature_maps: dict[str, np.ndarray]
    aggregate_maps: dict[str, np.ndarray]
    future_metallic_mask: np.ndarray
    future_erased_mask: np.ndarray
    both_metallic_erased_mask: np.ndarray
    stable_mask: np.ndarray
    never_switched_mask: np.ndarray
    average_initial_edcs: dict[str, np.ndarray]
    average_initial_mdcs: dict[str, np.ndarray]
    group_statistics: list[dict[str, Any]]
    valid_mask: np.ndarray
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.initial_near_ef_map.shape

    @property
    def file_paths(self) -> list[str]:
        return [state.file_path for state in self.loaded_states]

    @property
    def state_names(self) -> list[str]:
        return [state.name for state in self.loaded_states]

    @property
    def n_states(self) -> int:
        return len(self.loaded_states)

    @property
    def n_transitions(self) -> int:
        return len(self.transitions)

    @property
    def e_axis(self) -> np.ndarray:
        return np.asarray(self.loaded_states[self.initial_reference_index].data_array.coords["eV"].values, dtype=np.float32)

    @property
    def phi_axis(self) -> np.ndarray:
        return np.asarray(self.loaded_states[self.initial_reference_index].data_array.coords["phi"].values, dtype=np.float32)


@dataclass(slots=True)
class SwitchingMechanismParameters:
    transition_parameters: InitialTransitionFeatureParameters = field(default_factory=InitialTransitionFeatureParameters)
    future_metallic_min_count: int = 1
    future_erased_min_count: int = 1
    future_metallic_min_frequency: float = 0.0
    future_erased_min_frequency: float = 0.0
    edc_normalization: str = "raw"
    boundary_smooth_sigma: float = 1.0
    boundary_percentile: float = 85.0
    component_min_size: int = 0
    threshold_sweep_percentiles: tuple[float, ...] = (90.0, 92.5, 95.0, 97.5, 99.0)
    negative_control_min_ev: float = 0.08
    negative_control_max_ev: float = 0.12
    permutation_count: int = 48
    random_seed: int = 42
    epsilon: float = 1e-8

    def validate(self) -> None:
        self.transition_parameters.validate()
        if self.future_metallic_min_count < 0 or self.future_erased_min_count < 0:
            raise ValueError("Future metallic/erased minimum counts must be non-negative.")
        if not 0.0 <= self.future_metallic_min_frequency <= 1.0:
            raise ValueError("future_metallic_min_frequency must be between 0 and 1.")
        if not 0.0 <= self.future_erased_min_frequency <= 1.0:
            raise ValueError("future_erased_min_frequency must be between 0 and 1.")
        if self.edc_normalization not in SWITCHING_MECHANISM_EDC_NORMALIZATIONS:
            raise ValueError(f"edc_normalization must be one of {SWITCHING_MECHANISM_EDC_NORMALIZATIONS}.")
        if self.boundary_smooth_sigma < 0:
            raise ValueError("boundary_smooth_sigma must be non-negative.")
        if not 0.0 <= self.boundary_percentile <= 100.0:
            raise ValueError("boundary_percentile must be between 0 and 100.")
        if self.component_min_size < 0:
            raise ValueError("component_min_size must be non-negative.")
        if self.negative_control_min_ev >= self.negative_control_max_ev:
            raise ValueError("negative control min energy must be smaller than max energy.")
        for percentile in self.threshold_sweep_percentiles:
            if not 0.0 <= percentile <= 100.0:
                raise ValueError("threshold sweep percentiles must be between 0 and 100.")
        if self.permutation_count < 0:
            raise ValueError("permutation_count must be non-negative.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")


@dataclass(slots=True)
class SwitchingMechanismDiagnosticsResult:
    transition_result: InitialTransitionFeatureResult
    parameters: SwitchingMechanismParameters
    group_masks: dict[str, np.ndarray]
    cleaned_group_masks: dict[str, np.ndarray]
    group_edcs: dict[str, np.ndarray]
    group_edc_sem: dict[str, np.ndarray]
    group_mdcs: dict[str, np.ndarray]
    group_mdc_sem: dict[str, np.ndarray]
    group_spectra: dict[str, np.ndarray]
    group_spectrum_sem: dict[str, np.ndarray]
    spectral_effect_rows: list[dict[str, Any]]
    spatial_feature_maps: dict[str, np.ndarray]
    spatial_effect_rows: list[dict[str, Any]]
    connected_component_rows: list[dict[str, Any]]
    transition_history_maps: dict[str, np.ndarray]
    transition_level_rows: list[dict[str, Any]]
    file_intensity_rows: list[dict[str, Any]]
    artifact_rows: list[dict[str, Any]]
    threshold_sensitivity_rows: list[dict[str, Any]]
    threshold_robustness_maps: dict[str, np.ndarray]
    permutation_control_rows: list[dict[str, Any]]
    negative_control_maps: dict[str, np.ndarray]
    summary_verdict: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.transition_result.shape

    @property
    def e_axis(self) -> np.ndarray:
        return self.transition_result.e_axis

    @property
    def phi_axis(self) -> np.ndarray:
        return self.transition_result.phi_axis

    @property
    def file_paths(self) -> list[str]:
        return self.transition_result.file_paths

    @property
    def transitions(self) -> list[InitialTransitionPairMetrics]:
        return self.transition_result.transitions


def run_analysis(file_paths: list[str] | tuple[str, ...], parameters: AnalysisParameters | None = None) -> AnalysisResult:
    if parameters is None:
        parameters = AnalysisParameters()
    parameters.validate()

    paths = [str(Path(path).expanduser().resolve()) for path in file_paths]
    if not 1 <= len(paths) <= 4:
        raise ValueError("Please provide between 1 and 4 ARPES data files.")

    loaded_states, alignment_notes = align_loaded_states_for_comparison([load_state(path) for path in paths])

    total_maps: list[np.ndarray] = []
    ef_maps: list[np.ndarray] = []
    features_by_state: list[dict[str, np.ndarray]] = []
    feature_matrices: list[np.ndarray] = []
    feature_names: list[str] | None = None
    notes: list[str] = list(alignment_notes)

    for state in loaded_states:
        total_map, ef_map = total_and_ef_maps(
            state.data_array,
            fermi_level=parameters.fermi_level_ev,
            ef_window=parameters.ef_window_ev,
        )
        total_maps.append(total_map)
        ef_maps.append(ef_map)

        features, names, feature_matrix = extract_pixel_features(
            state.data_array,
            fermi_level=parameters.fermi_level_ev,
            ef_window=parameters.ef_window_ev,
            wide_window=parameters.wide_window_ev,
        )
        features_by_state.append(features)
        feature_matrices.append(feature_matrix)
        if feature_names is None:
            feature_names = names

    if feature_names is None:
        raise RuntimeError("No feature names were produced by the analysis pipeline.")

    valid_mask, average_normalized_total_map, active_mask, row_occupancy, col_occupancy = build_cross_mask_from_maps(
        total_maps,
        threshold_quantile=parameters.cross_threshold_quantile,
        row_fraction=parameters.cross_row_fraction,
        col_fraction=parameters.cross_col_fraction,
        background_quantile=parameters.cross_background_quantile,
        pad=parameters.cross_pad,
    )

    valid_pixels = int(valid_mask.sum())
    if valid_pixels == 0:
        raise ValueError(
            "The current cross-mask settings excluded every pixel. Try lowering the mask thresholds."
        )

    if valid_pixels < parameters.n_clusters:
        notes.append(
            f"Reduced cluster count from {parameters.n_clusters} to {valid_pixels} because only {valid_pixels} pixels were inside the cross."
        )

    valid_flat = valid_mask.reshape(-1)
    all_masked = np.concatenate([feature_matrix[valid_flat] for feature_matrix in feature_matrices], axis=0)
    all_masked_z = robust_zscore(all_masked, axis=0)
    all_masked_z = finite_fill(all_masked_z, 0.0)

    masked_chunks: list[np.ndarray] = []
    start = 0
    for _ in loaded_states:
        end = start + valid_pixels
        masked_chunks.append(all_masked_z[start:end])
        start = end

    pca_fit = fit_pca(all_masked_z, n_components=parameters.n_pca_components)
    embeddings = [transform_pca(chunk, pca_fit) for chunk in masked_chunks]
    embedded_all = np.concatenate(embeddings, axis=0)

    k = min(parameters.n_clusters, valid_pixels)
    cluster_labels, cluster_centroids, cluster_inertia = kmeans(
        embedded_all,
        k=k,
        n_iter=100,
        n_init=12,
        seed=42,
    )

    raw_cluster_maps: list[np.ndarray] = []
    x_size, y_size = valid_mask.shape
    valid_indices = np.flatnonzero(valid_flat)

    start = 0
    for _ in loaded_states:
        end = start + valid_pixels
        labels_for_state = cluster_labels[start:end]
        cluster_map = np.full(x_size * y_size, fill_value=-1, dtype=int)
        cluster_map[valid_indices] = labels_for_state
        raw_cluster_maps.append(cluster_map.reshape(x_size, y_size))
        start = end

    raw_to_ordered_cluster, cluster_mean_ef_fraction = order_clusters_by_mean_ef_fraction(
        raw_cluster_maps,
        features_by_state,
        valid_mask,
    )
    cluster_maps = [remap_cluster_map(cluster_map, raw_to_ordered_cluster) for cluster_map in raw_cluster_maps]

    cluster_counts_by_state = [count_labeled_pixels(cluster_map, valid_mask) for cluster_map in cluster_maps]

    cluster_sequence_strings, cluster_sequence_code_map, cluster_sequences, cluster_sequence_to_code = build_sequence_maps(
        cluster_maps,
        valid_mask,
        formatter=lambda values: " -> ".join(f"C{int(value)}" for value in values),
        outside_label="outside-cross",
    )

    simple_state_label_maps, simple_state_code_maps, simple_state_thresholds = build_simple_state_maps(
        features_by_state,
        valid_mask,
        low_quantile=parameters.simple_state_low_quantile,
        high_quantile=parameters.simple_state_high_quantile,
    )

    simple_state_sequence_strings, simple_state_sequence_code_map, simple_state_sequences, simple_state_sequence_to_code = build_sequence_maps(
        simple_state_label_maps,
        valid_mask,
        formatter=lambda values: " -> ".join(SIMPLE_STATE_SHORT[str(value)] for value in values),
        outside_label="outside-cross",
    )

    return AnalysisResult(
        parameters=parameters,
        loaded_states=loaded_states,
        feature_names=feature_names,
        valid_mask=valid_mask,
        average_normalized_total_map=average_normalized_total_map,
        active_mask=active_mask,
        row_occupancy=row_occupancy,
        col_occupancy=col_occupancy,
        total_maps=total_maps,
        ef_maps=ef_maps,
        features_by_state=features_by_state,
        cluster_maps=cluster_maps,
        raw_cluster_maps=raw_cluster_maps,
        raw_to_ordered_cluster=raw_to_ordered_cluster,
        cluster_mean_ef_fraction=cluster_mean_ef_fraction,
        cluster_sequence_strings=cluster_sequence_strings,
        cluster_sequence_code_map=cluster_sequence_code_map,
        cluster_sequences=cluster_sequences,
        cluster_sequence_to_code=cluster_sequence_to_code,
        simple_state_label_maps=simple_state_label_maps,
        simple_state_code_maps=simple_state_code_maps,
        simple_state_thresholds=simple_state_thresholds,
        simple_state_sequence_strings=simple_state_sequence_strings,
        simple_state_sequence_code_map=simple_state_sequence_code_map,
        simple_state_sequences=simple_state_sequences,
        simple_state_sequence_to_code=simple_state_sequence_to_code,
        pca_explained_ratio=pca_fit["explained_ratio"],
        cluster_centroids=cluster_centroids,
        cluster_inertia=cluster_inertia,
        cluster_counts_by_state=cluster_counts_by_state,
        notes=notes,
    )


def run_spectral_clustering(
    state: LoadedState,
    valid_mask: np.ndarray,
    feature_maps: dict[str, np.ndarray] | None = None,
    parameters: SpectralClusterParameters | None = None,
    analysis_parameters: AnalysisParameters | None = None,
) -> SpectralClusterResult:
    if parameters is None:
        parameters = SpectralClusterParameters()
    parameters.validate()

    valid_mask = np.asarray(valid_mask, dtype=bool)
    data = np.asarray(state.data_array.values, dtype=np.float32)
    if data.shape[:2] != valid_mask.shape:
        raise ValueError(
            "valid_mask shape must match the spatial shape of the loaded state. "
            f"Expected {data.shape[:2]}, received {valid_mask.shape}."
        )

    if feature_maps is None:
        spectral_parameters = analysis_parameters or AnalysisParameters()
        feature_maps, _, _ = extract_pixel_features(
            state.data_array,
            fermi_level=spectral_parameters.fermi_level_ev,
            ef_window=spectral_parameters.ef_window_ev,
            wide_window=spectral_parameters.wide_window_ev,
        )

    valid_pixels = int(valid_mask.sum())
    if valid_pixels == 0:
        raise ValueError("Spectral clustering requires at least one valid pixel inside the cross mask.")

    notes: list[str] = []
    if valid_pixels < parameters.n_clusters:
        notes.append(
            f"Reduced spectral-clustering count from {parameters.n_clusters} to {valid_pixels} because only {valid_pixels} pixels were inside the cross."
        )

    x_size, y_size, e_size, phi_size = data.shape
    valid_flat = valid_mask.reshape(-1)
    valid_indices = np.flatnonzero(valid_flat)
    flat_data = data.reshape(x_size * y_size, e_size, phi_size)

    estimated_bytes = estimate_spectral_clustering_working_set_bytes(
        valid_pixels=valid_pixels,
        e_size=e_size,
        phi_size=phi_size,
        method_key=parameters.method_key,
        embedding_components=parameters.embedding_components,
    )
    budget_bytes = get_safe_clustering_memory_budget_bytes()
    if estimated_bytes > budget_bytes:
        raise MemoryError(
            "Refusing to run the selected clustering method because it is estimated to need "
            f"{format_byte_count(estimated_bytes)} of additional working memory, which exceeds the "
            f"safety budget of {format_byte_count(budget_bytes)}.\n"
            f"Try {SPECTRAL_CLUSTER_METHOD_LABELS['compressed_features']} or "
            f"{SPECTRAL_CLUSTER_METHOD_LABELS['downsampled_spectra_pca']} instead."
        )

    cluster_input, embedding, explained_ratio, method_notes = build_cluster_representation(
        flat_data=flat_data,
        valid_indices=valid_indices,
        feature_maps=feature_maps,
        parameters=parameters,
    )
    notes.extend(method_notes)
    embedding_2d = ensure_two_dimensional_embedding(embedding)

    k = min(parameters.n_clusters, valid_pixels)
    raw_labels, raw_centroids, cluster_inertia = kmeans(
        cluster_input,
        k=k,
        n_iter=parameters.n_iter,
        n_init=parameters.n_init,
        seed=parameters.seed,
    )

    raw_cluster_map_flat = np.full(x_size * y_size, fill_value=-1, dtype=int)
    raw_cluster_map_flat[valid_indices] = raw_labels
    raw_cluster_map = raw_cluster_map_flat.reshape(x_size, y_size)

    raw_to_ordered_cluster, _ = order_cluster_map_by_metric(
        raw_cluster_map,
        feature_maps["ef_fraction"],
        valid_mask,
    )
    cluster_map = remap_cluster_map(raw_cluster_map, raw_to_ordered_cluster)
    ordered_labels = np.asarray([raw_to_ordered_cluster[int(label)] for label in raw_labels], dtype=int)
    ordered_centroids = reorder_cluster_centroids(raw_centroids, raw_to_ordered_cluster)
    cluster_counts = count_labeled_pixels(cluster_map, valid_mask)

    cluster_meta = collect_cluster_meta(cluster_map, valid_mask, feature_maps)
    candidate_labels = suggest_hypothesis_candidate_labels(cluster_meta)
    cluster_mean_spectra = compute_cluster_mean_spectra(
        flat_data=flat_data,
        valid_indices=valid_indices,
        ordered_labels=ordered_labels,
        n_clusters=len(cluster_counts),
    )
    cluster_stats = build_spectral_cluster_stats(
        cluster_map=cluster_map,
        valid_mask=valid_mask,
        ordered_labels=ordered_labels,
        cluster_input=cluster_input,
        cluster_centroids=ordered_centroids,
        embedding_2d=embedding_2d,
        cluster_mean_spectra=cluster_mean_spectra,
        feature_maps=feature_maps,
        candidate_labels=candidate_labels,
    )

    return SpectralClusterResult(
        state_name=state.name,
        state_file=state.file_path,
        parameters=parameters,
        valid_mask=valid_mask,
        cluster_map=cluster_map,
        raw_cluster_map=raw_cluster_map,
        cluster_counts=cluster_counts,
        cluster_stats=cluster_stats,
        cluster_centroids=ordered_centroids,
        cluster_inertia=cluster_inertia,
        embedding_2d=embedding_2d,
        embedding_explained_ratio=explained_ratio,
        pixel_coordinates=np.argwhere(valid_mask).astype(int),
        total_intensity_map=np.asarray(feature_maps["total_intensity"], dtype=np.float32),
        ef_fraction_map=np.asarray(feature_maps["ef_fraction"], dtype=np.float32),
        spectral_entropy_map=np.asarray(feature_maps["spectral_entropy"], dtype=np.float32),
        e_centroid_map=np.asarray(feature_maps["e_centroid"], dtype=np.float32),
        feature_maps=feature_maps,
        e_axis=np.asarray(state.data_array.coords["eV"].values, dtype=np.float32),
        phi_axis=np.asarray(state.data_array.coords["phi"].values, dtype=np.float32),
        notes=notes,
    )


def build_cluster_representation(
    flat_data: np.ndarray,
    valid_indices: np.ndarray,
    feature_maps: dict[str, np.ndarray],
    parameters: SpectralClusterParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    method_key = parameters.method_key
    if method_key == "compressed_features":
        return build_compressed_feature_representation(flat_data, valid_indices, feature_maps, parameters)
    if method_key == "downsampled_spectra_pca":
        return build_downsampled_spectra_representation(flat_data, valid_indices, parameters, target_e_bins=20, target_phi_bins=14)
    if method_key == "full_spectra_pca":
        return build_downsampled_spectra_representation(flat_data, valid_indices, parameters, target_e_bins=None, target_phi_bins=None)
    raise ValueError(f"Unsupported clustering method {method_key!r}.")


def build_compressed_feature_representation(
    flat_data: np.ndarray,
    valid_indices: np.ndarray,
    feature_maps: dict[str, np.ndarray],
    parameters: SpectralClusterParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feature_names = [
        "ef_fraction",
        "wide_fraction",
        "e_centroid",
        "e_var",
        "phi_centroid",
        "phi_var",
        "phi_asymmetry",
        "spectral_entropy",
        "spectral_sharpness",
        "ef_neighbor_diff",
        "ef_local_contrast",
    ]
    scalar_matrix = np.stack(
        [np.asarray(feature_maps[name], dtype=np.float32).reshape(-1)[valid_indices] for name in feature_names],
        axis=1,
    ).astype(np.float32)

    n_valid = len(valid_indices)
    sample_spectrum = np.asarray(flat_data[valid_indices[:1]], dtype=np.float32)
    e_size = int(sample_spectrum.shape[1])
    phi_size = int(sample_spectrum.shape[2])
    energy_profiles = np.empty((n_valid, min(12, e_size)), dtype=np.float32)
    phi_profiles = np.empty((n_valid, min(10, phi_size)), dtype=np.float32)

    for start, chunk_indices in iter_valid_index_chunks(valid_indices, chunk_size=256):
        spectra_chunk = np.asarray(flat_data[chunk_indices], dtype=np.float32)
        energy_chunk = normalize_rows(np.sum(spectra_chunk, axis=2))
        phi_chunk = normalize_rows(np.sum(spectra_chunk, axis=1))
        energy_profiles[start : start + len(chunk_indices)] = resize_matrix_rows(energy_chunk, min(12, e_size))
        phi_profiles[start : start + len(chunk_indices)] = resize_matrix_rows(phi_chunk, min(10, phi_size))

    base = np.concatenate([scalar_matrix, energy_profiles, phi_profiles], axis=1).astype(np.float32)
    base = robust_zscore(base, axis=0)
    base = finite_fill(base, 0.0)
    embedding, explained_ratio, pca_components = build_embedding_from_base(base, parameters.embedding_components)
    notes = [
        "Used compressed spectral summaries for safer clustering.",
        f"Representation: {len(feature_names)} scalar features + {energy_profiles.shape[1]} energy bins + {phi_profiles.shape[1]} phi bins.",
    ]
    if pca_components < parameters.embedding_components:
        notes.append(
            f"Reduced embedding components from {parameters.embedding_components} to {pca_components} because the compressed representation had limited rank."
        )
    return embedding, embedding, explained_ratio, notes


def build_downsampled_spectra_representation(
    flat_data: np.ndarray,
    valid_indices: np.ndarray,
    parameters: SpectralClusterParameters,
    target_e_bins: int | None,
    target_phi_bins: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    n_valid = len(valid_indices)
    sample_spectrum = np.asarray(flat_data[valid_indices[:1]], dtype=np.float32)
    e_size = int(sample_spectrum.shape[1])
    phi_size = int(sample_spectrum.shape[2])
    effective_e_bins = e_size if target_e_bins is None else min(target_e_bins, e_size)
    effective_phi_bins = phi_size if target_phi_bins is None else min(target_phi_bins, phi_size)
    base = np.empty((n_valid, effective_e_bins * effective_phi_bins), dtype=np.float32)

    for start, chunk_indices in iter_valid_index_chunks(valid_indices, chunk_size=128):
        spectra_chunk = np.asarray(flat_data[chunk_indices], dtype=np.float32)
        rebinned = rebin_spectra_batch(spectra_chunk, effective_e_bins, effective_phi_bins)
        base[start : start + len(chunk_indices)] = flatten_registered_spectra_rows(rebinned)

    base = robust_zscore(base, axis=0)
    base = finite_fill(base, 0.0)

    embedding, explained_ratio, pca_components = build_embedding_from_base(base, parameters.embedding_components)
    notes: list[str] = []
    if target_e_bins is None or target_phi_bins is None:
        notes.append("Used the full registered spectra with a memory safety guard.")
    else:
        notes.append(
            f"Downsampled spectra to {effective_e_bins} energy bins x {effective_phi_bins} phi bins before clustering to reduce memory use."
        )
    if pca_components < parameters.embedding_components:
        notes.append(
            f"Reduced embedding components from {parameters.embedding_components} to {pca_components} because the spectra matrix had limited rank."
        )
    return embedding, embedding, explained_ratio, notes


def prepare_registered_spectra_for_clustering(spectra: np.ndarray) -> np.ndarray:
    flattened = flatten_registered_spectra_rows(spectra)
    flattened = robust_zscore(flattened, axis=0)
    return finite_fill(flattened, 0.0)


def flatten_registered_spectra_rows(spectra: np.ndarray) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=np.float32)
    flattened = spectra.reshape(spectra.shape[0], -1)
    flattened = np.log1p(np.clip(flattened, a_min=0.0, a_max=None))
    flattened = normalize_rows(flattened)
    return flattened.astype(np.float32)


def build_embedding_from_base(base: np.ndarray, requested_components: int) -> tuple[np.ndarray, np.ndarray, int]:
    base = np.asarray(base, dtype=np.float32)
    n_samples, n_features = base.shape
    n_components = max(2, min(int(requested_components), n_samples, n_features))
    pca_fit = fit_pca(base, n_components=n_components)
    embedding = transform_pca(base, pca_fit)
    return embedding.astype(np.float32), np.asarray(pca_fit["explained_ratio"], dtype=np.float32), n_components


def compute_cluster_mean_spectra(
    flat_data: np.ndarray,
    valid_indices: np.ndarray,
    ordered_labels: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    sample_shape = np.asarray(flat_data[valid_indices[:1]], dtype=np.float32).shape[1:]
    sums = np.zeros((n_clusters, sample_shape[0], sample_shape[1]), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)

    for start, chunk_indices in iter_valid_index_chunks(valid_indices, chunk_size=128):
        spectra_chunk = np.asarray(flat_data[chunk_indices], dtype=np.float32)
        labels_chunk = np.asarray(ordered_labels[start : start + len(chunk_indices)], dtype=int)
        for cluster_id in np.unique(labels_chunk):
            mask = labels_chunk == cluster_id
            sums[cluster_id] += spectra_chunk[mask].sum(axis=0, dtype=np.float64)
            counts[cluster_id] += int(mask.sum())

    mean_spectra = np.zeros_like(sums, dtype=np.float32)
    for cluster_id in range(n_clusters):
        if counts[cluster_id] > 0:
            mean_spectra[cluster_id] = (sums[cluster_id] / counts[cluster_id]).astype(np.float32)
    return mean_spectra


def iter_valid_index_chunks(valid_indices: np.ndarray, chunk_size: int = 256) -> Any:
    valid_indices = np.asarray(valid_indices, dtype=int)
    for start in range(0, len(valid_indices), max(1, int(chunk_size))):
        yield start, valid_indices[start : start + chunk_size]


def resize_matrix_rows(values: np.ndarray, target_width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[1] == target_width:
        return values.astype(np.float32)
    zoom_factors = (1.0, float(target_width) / float(values.shape[1]))
    return ndimage.zoom(values, zoom=zoom_factors, order=1).astype(np.float32)


def rebin_spectra_batch(spectra: np.ndarray, target_e_bins: int, target_phi_bins: int) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=np.float32)
    if spectra.shape[1] == target_e_bins and spectra.shape[2] == target_phi_bins:
        return spectra.astype(np.float32)
    zoom_factors = (
        1.0,
        float(target_e_bins) / float(spectra.shape[1]),
        float(target_phi_bins) / float(spectra.shape[2]),
    )
    return ndimage.zoom(spectra, zoom=zoom_factors, order=1).astype(np.float32)


def get_physical_memory_bytes() -> int | None:
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None
    return None


def get_safe_clustering_memory_budget_bytes() -> int:
    physical_bytes = get_physical_memory_bytes()
    if physical_bytes is None or physical_bytes <= 0:
        return 768 * 1024 * 1024
    return min(int(physical_bytes * 0.12), 1024 * 1024 * 1024)


def estimate_spectral_clustering_working_set_bytes(
    valid_pixels: int,
    e_size: int,
    phi_size: int,
    method_key: str,
    embedding_components: int,
) -> int:
    _, representation_dim = estimate_spectral_clustering_representation_shape(valid_pixels, e_size, phi_size, method_key)
    base_bytes = int(valid_pixels) * int(representation_dim) * 4
    embedding_bytes = int(valid_pixels) * max(2, int(embedding_components)) * 4
    multipliers = {
        "compressed_features": 4.0,
        "downsampled_spectra_pca": 5.0,
        "full_spectra_pca": 7.0,
    }
    multiplier = multipliers.get(method_key, 5.0)
    return int(base_bytes * multiplier + embedding_bytes * 4)


def estimate_spectral_clustering_representation_shape(
    valid_pixels: int,
    e_size: int,
    phi_size: int,
    method_key: str,
) -> tuple[int, int]:
    if method_key == "compressed_features":
        return valid_pixels, 11 + min(12, e_size) + min(10, phi_size)
    if method_key == "downsampled_spectra_pca":
        return valid_pixels, min(20, e_size) * min(14, phi_size)
    if method_key == "full_spectra_pca":
        return valid_pixels, e_size * phi_size
    raise ValueError(f"Unsupported clustering method {method_key!r}.")


def format_byte_count(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def ensure_two_dimensional_embedding(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a two-dimensional embedding array, received shape {values.shape}.")
    if values.shape[1] >= 2:
        return values[:, :2].astype(np.float32)
    return np.column_stack([values[:, 0], np.zeros(values.shape[0], dtype=np.float32)]).astype(np.float32)


def order_cluster_map_by_metric(
    raw_cluster_map: np.ndarray,
    metric_map: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[dict[int, int], dict[int, float]]:
    raw_ids = sorted(int(label) for label in np.unique(raw_cluster_map[valid_mask]))
    means: list[tuple[int, float]] = []
    for raw_id in raw_ids:
        mask = (raw_cluster_map == raw_id) & valid_mask
        mean_value = float(np.mean(np.asarray(metric_map, dtype=np.float32)[mask])) if np.any(mask) else float("nan")
        means.append((raw_id, mean_value))

    means.sort(key=lambda item: item[1])
    mapping = {raw_id: ordered_id for ordered_id, (raw_id, _) in enumerate(means)}
    ordered_means = {mapping[raw_id]: mean_value for raw_id, mean_value in means}
    return mapping, ordered_means


def reorder_cluster_centroids(centroids: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    centroids = np.asarray(centroids, dtype=np.float32)
    reordered = np.empty_like(centroids)
    for raw_id, ordered_id in mapping.items():
        reordered[ordered_id] = centroids[raw_id]
    return reordered


def collect_cluster_meta(
    cluster_map: np.ndarray,
    valid_mask: np.ndarray,
    feature_maps: dict[str, np.ndarray],
) -> list[dict[str, float | int]]:
    total_valid = max(1, int(valid_mask.sum()))
    meta: list[dict[str, float | int]] = []
    for cluster_id in sorted(int(label) for label in np.unique(cluster_map[valid_mask])):
        mask = (cluster_map == cluster_id) & valid_mask
        connected_components, dominant_component_fraction = compute_connected_component_stats(mask)
        meta.append(
            {
                "cluster_id": cluster_id,
                "pixel_count": int(mask.sum()),
                "pixel_fraction": float(mask.sum() / total_valid),
                "mean_ef_fraction": float(np.mean(np.asarray(feature_maps["ef_fraction"], dtype=np.float32)[mask])),
                "connected_components": connected_components,
                "dominant_component_fraction": dominant_component_fraction,
            }
        )
    return meta


def suggest_hypothesis_candidate_labels(cluster_meta: list[dict[str, float | int]]) -> dict[int, str]:
    if not cluster_meta:
        return {}

    ordered = sorted(cluster_meta, key=lambda item: float(item["mean_ef_fraction"]))
    labels = {int(item["cluster_id"]): "erased / intermediate candidate" for item in ordered}

    labels[int(ordered[0]["cluster_id"])] = "insulating / CCDW candidate"
    if len(ordered) >= 2:
        labels[int(ordered[-1]["cluster_id"])] = "written metastable metallic candidate"

    if len(ordered) >= 4:
        interior = ordered[1:-1]
        patch_candidate = min(
            interior,
            key=lambda item: (
                -float(item["mean_ef_fraction"]),
                float(item["pixel_fraction"]),
                -int(item["connected_components"]),
            ),
        )
        labels[int(patch_candidate["cluster_id"])] = "metallic patch / stacking candidate"

    return labels


def build_spectral_cluster_stats(
    cluster_map: np.ndarray,
    valid_mask: np.ndarray,
    ordered_labels: np.ndarray,
    cluster_input: np.ndarray,
    cluster_centroids: np.ndarray,
    embedding_2d: np.ndarray,
    cluster_mean_spectra: np.ndarray,
    feature_maps: dict[str, np.ndarray],
    candidate_labels: dict[int, str],
) -> list[SpectralClusterStats]:
    total_valid = max(1, int(valid_mask.sum()))
    stats: list[SpectralClusterStats] = []

    for cluster_id in sorted(int(label) for label in np.unique(cluster_map[valid_mask])):
        cluster_mask = (cluster_map == cluster_id) & valid_mask
        sample_mask = ordered_labels == cluster_id
        pixel_count = int(sample_mask.sum())

        connected_components, dominant_component_fraction = compute_connected_component_stats(cluster_mask)
        mean_spectrum = np.asarray(cluster_mean_spectra[cluster_id], dtype=np.float32)
        mean_energy_profile = np.sum(mean_spectrum, axis=1).astype(np.float32)
        embedding_center = np.mean(np.asarray(embedding_2d[sample_mask], dtype=np.float32), axis=0).astype(np.float32)

        centroid = np.asarray(cluster_centroids[cluster_id], dtype=np.float32)
        intra_cluster_rms = compute_cluster_rms(cluster_input[sample_mask], centroid)
        nearest_cluster_distance = compute_nearest_centroid_distance(cluster_centroids, cluster_id)
        separation_ratio = (
            float(nearest_cluster_distance / max(intra_cluster_rms, 1e-6))
            if math.isfinite(nearest_cluster_distance)
            else float("nan")
        )

        stats.append(
            SpectralClusterStats(
                cluster_id=cluster_id,
                pixel_count=pixel_count,
                pixel_fraction=float(pixel_count / total_valid),
                mean_ef_fraction=float(np.mean(np.asarray(feature_maps["ef_fraction"], dtype=np.float32)[cluster_mask])),
                mean_total_intensity=float(np.mean(np.asarray(feature_maps["total_intensity"], dtype=np.float32)[cluster_mask])),
                mean_spectral_entropy=float(np.mean(np.asarray(feature_maps["spectral_entropy"], dtype=np.float32)[cluster_mask])),
                mean_e_centroid=float(np.mean(np.asarray(feature_maps["e_centroid"], dtype=np.float32)[cluster_mask])),
                connected_components=connected_components,
                dominant_component_fraction=dominant_component_fraction,
                intra_cluster_rms=intra_cluster_rms,
                nearest_cluster_distance=nearest_cluster_distance,
                separation_ratio=separation_ratio,
                embedding_center=embedding_center,
                mean_energy_profile=mean_energy_profile,
                mean_spectrum=mean_spectrum,
                candidate_label=candidate_labels.get(cluster_id, "candidate class"),
            )
        )

    return stats


def compute_connected_component_stats(mask: np.ndarray) -> tuple[int, float]:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return 0, 0.0

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
    labeled, component_count = ndimage.label(mask.astype(np.int8), structure=structure)
    component_sizes = np.bincount(labeled.reshape(-1))[1:]
    dominant_fraction = float(component_sizes.max() / mask.sum()) if component_sizes.size else 0.0
    return int(component_count), dominant_fraction


def compute_cluster_rms(values: np.ndarray, centroid: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    centroid = np.asarray(centroid, dtype=np.float32)
    if values.size == 0:
        return float("nan")
    distances = np.sum((values - centroid[None, :]) ** 2, axis=1)
    return float(np.sqrt(np.mean(distances)))


def compute_nearest_centroid_distance(cluster_centroids: np.ndarray, cluster_id: int) -> float:
    centroids = np.asarray(cluster_centroids, dtype=np.float32)
    if centroids.shape[0] <= 1:
        return float("nan")
    centroid = centroids[cluster_id]
    distances = np.sqrt(np.sum((centroids - centroid[None, :]) ** 2, axis=1))
    distances[cluster_id] = np.inf
    return float(np.min(distances))


def build_spectral_cluster_summary(result: SpectralClusterResult) -> dict[str, Any]:
    return {
        "state_name": result.state_name,
        "state_file": result.state_file,
        "method_key": result.parameters.method_key,
        "method_label": SPECTRAL_CLUSTER_METHOD_LABELS[result.parameters.method_key],
        "resource_level": SPECTRAL_CLUSTER_METHOD_RESOURCES[result.parameters.method_key],
        "n_clusters": len(result.cluster_stats),
        "cluster_inertia": float(result.cluster_inertia),
        "embedding_explained_ratio": [float(value) for value in result.embedding_explained_ratio.tolist()],
        "clusters": [
            {
                "cluster_id": stats.cluster_id,
                "candidate_label": stats.candidate_label,
                "pixel_count": stats.pixel_count,
                "pixel_fraction": float(stats.pixel_fraction),
                "mean_ef_fraction": float(stats.mean_ef_fraction),
                "mean_total_intensity": float(stats.mean_total_intensity),
                "mean_spectral_entropy": float(stats.mean_spectral_entropy),
                "mean_e_centroid": float(stats.mean_e_centroid),
                "connected_components": int(stats.connected_components),
                "dominant_component_fraction": float(stats.dominant_component_fraction),
                "intra_cluster_rms": float(stats.intra_cluster_rms),
                "nearest_cluster_distance": float(stats.nearest_cluster_distance),
                "separation_ratio": float(stats.separation_ratio),
                "embedding_center": [float(value) for value in stats.embedding_center.tolist()],
            }
            for stats in result.cluster_stats
        ],
        "notes": list(result.notes),
    }


def analyze_cluster_physical_interpretation(result: SpectralClusterResult) -> ClusterPhysicalInterpretation:
    metrics_rows = [
        compute_cluster_physical_metrics(stats, result.e_axis, result.phi_axis)
        for stats in result.cluster_stats
    ]
    pairwise_rows = build_cluster_pairwise_physical_differences(metrics_rows, result.cluster_stats, result.e_axis)
    question_summaries = build_cluster_question_summaries(metrics_rows, pairwise_rows, result.e_axis)
    findings = [
        f"{summary.question} {summary.answer} {summary.strongest_example} {summary.reasoning}".strip()
        for summary in question_summaries
    ]
    notes = [
        "Gap size and gap filling are reported as proxies from the cluster mean energy profile, not as full many-body gap fits.",
        "Dispersion shape is summarized from normalized mean-spectrum correlations plus ridge-line slope and curvature proxies.",
        "Relative spectral-weight transfer compares deep, shallow, and near-EF windows on the cluster mean spectra.",
    ]
    return ClusterPhysicalInterpretation(
        state_name=result.state_name,
        state_file=result.state_file,
        metrics_rows=metrics_rows,
        pairwise_rows=pairwise_rows,
        question_summaries=question_summaries,
        findings=findings,
        notes=notes,
    )


def compute_cluster_physical_metrics(
    stats: SpectralClusterStats,
    energy_axis: np.ndarray,
    phi_axis: np.ndarray,
) -> ClusterPhysicalMetrics:
    mean_spectrum = np.asarray(stats.mean_spectrum, dtype=np.float32)
    energy_axis = np.asarray(energy_axis, dtype=np.float32)
    phi_axis = np.asarray(phi_axis, dtype=np.float32)

    energy_profile = np.sum(mean_spectrum, axis=1).astype(np.float32)
    smoothed = ndimage.gaussian_filter1d(energy_profile, sigma=1.0).astype(np.float32)
    total_weight = max(float(np.sum(energy_profile)), 1e-8)

    near_ef_mask = build_near_ef_mask(energy_axis)
    deep_mask, shallow_mask = build_occupied_weight_masks(energy_axis, near_ef_mask)

    dominant_peak_index = find_dominant_peak_index(energy_axis, smoothed)
    secondary_peak_index = find_secondary_peak_index(smoothed, dominant_peak_index)
    ef_index = int(np.argmin(np.abs(energy_axis)))

    fermi_weight_fraction = float(np.sum(energy_profile[near_ef_mask]) / total_weight)
    gap_fill_ratio = float(smoothed[ef_index] / max(float(smoothed[dominant_peak_index]), 1e-8))
    gap_proxy_ev = estimate_gap_proxy_ev(energy_axis, smoothed, dominant_peak_index)
    dominant_peak_ev = float(energy_axis[dominant_peak_index])
    secondary_peak_ev = float(energy_axis[secondary_peak_index]) if secondary_peak_index is not None else float("nan")
    dominant_peak_width_ev = estimate_peak_width_ev(energy_axis, smoothed, dominant_peak_index)
    dispersion_slope, dispersion_curvature, ridge_coverage = estimate_dispersion_shape_metrics(mean_spectrum, energy_axis, phi_axis)

    deep_weight_fraction = float(np.sum(energy_profile[deep_mask]) / total_weight) if np.any(deep_mask) else 0.0
    shallow_weight_fraction = float(np.sum(energy_profile[shallow_mask]) / total_weight) if np.any(shallow_mask) else 0.0
    near_ef_weight_fraction = float(np.sum(energy_profile[near_ef_mask]) / total_weight) if np.any(near_ef_mask) else 0.0

    return ClusterPhysicalMetrics(
        cluster_id=stats.cluster_id,
        candidate_label=stats.candidate_label,
        pixel_count=stats.pixel_count,
        pixel_fraction=stats.pixel_fraction,
        fermi_weight_fraction=fermi_weight_fraction,
        gap_fill_ratio=gap_fill_ratio,
        gap_proxy_ev=gap_proxy_ev,
        dominant_peak_ev=dominant_peak_ev,
        secondary_peak_ev=secondary_peak_ev,
        dominant_peak_width_ev=dominant_peak_width_ev,
        dispersion_slope_phi_per_ev=dispersion_slope,
        dispersion_curvature_phi_per_ev2=dispersion_curvature,
        deep_weight_fraction=deep_weight_fraction,
        shallow_weight_fraction=shallow_weight_fraction,
        near_ef_weight_fraction=near_ef_weight_fraction,
        ridge_coverage_fraction=ridge_coverage,
    )


def build_near_ef_mask(energy_axis: np.ndarray) -> np.ndarray:
    energy_axis = np.asarray(energy_axis, dtype=np.float32)
    step = median_axis_step(energy_axis)
    halfwidth = max(0.04, 2.0 * step)
    return np.abs(energy_axis) <= halfwidth


def build_occupied_weight_masks(energy_axis: np.ndarray, near_ef_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    energy_axis = np.asarray(energy_axis, dtype=np.float32)
    occupied_indices = np.flatnonzero(energy_axis < -max(0.04, 2.0 * median_axis_step(energy_axis)))
    deep_mask = np.zeros_like(energy_axis, dtype=bool)
    shallow_mask = np.zeros_like(energy_axis, dtype=bool)
    if occupied_indices.size == 0:
        return deep_mask, shallow_mask
    if occupied_indices.size <= 3:
        shallow_mask[occupied_indices] = True
        return deep_mask, shallow_mask

    split = occupied_indices.size // 2
    deep_mask[occupied_indices[:split]] = True
    shallow_mask[occupied_indices[split:]] = True
    shallow_mask &= ~near_ef_mask
    return deep_mask, shallow_mask


def median_axis_step(axis: np.ndarray) -> float:
    axis = np.asarray(axis, dtype=np.float32)
    diffs = np.abs(np.diff(axis))
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return 1e-3
    return float(np.median(positive))


def find_dominant_peak_index(energy_axis: np.ndarray, profile: np.ndarray) -> int:
    energy_axis = np.asarray(energy_axis, dtype=np.float32)
    profile = np.asarray(profile, dtype=np.float32)
    occupied = np.flatnonzero(energy_axis <= 0.02)
    if occupied.size == 0:
        return int(np.argmax(profile))
    occupied_index = occupied[np.argmax(profile[occupied])]
    return int(occupied_index)


def find_secondary_peak_index(profile: np.ndarray, dominant_peak_index: int) -> int | None:
    profile = np.asarray(profile, dtype=np.float32)
    if profile.size < 5:
        return None
    suppressed = profile.copy()
    left = max(0, dominant_peak_index - 2)
    right = min(profile.size, dominant_peak_index + 3)
    suppressed[left:right] = -np.inf
    candidate = int(np.argmax(suppressed))
    if not np.isfinite(suppressed[candidate]) or suppressed[candidate] <= 0:
        return None
    return candidate


def estimate_gap_proxy_ev(energy_axis: np.ndarray, profile: np.ndarray, peak_index: int) -> float:
    energy_axis = np.asarray(energy_axis, dtype=np.float32)
    profile = np.asarray(profile, dtype=np.float32)
    ef_index = int(np.argmin(np.abs(energy_axis)))
    baseline = float(profile[ef_index])
    peak_value = float(profile[peak_index])
    if peak_value <= baseline:
        return 0.0
    half_level = baseline + 0.5 * (peak_value - baseline)
    for index in range(peak_index, ef_index + 1):
        if profile[index] <= half_level:
            return float(abs(energy_axis[index]))
    return 0.0


def estimate_peak_width_ev(energy_axis: np.ndarray, profile: np.ndarray, peak_index: int) -> float:
    energy_axis = np.asarray(energy_axis, dtype=np.float32)
    profile = np.asarray(profile, dtype=np.float32)
    peak_value = float(profile[peak_index])
    if peak_value <= 0:
        return float("nan")
    half_level = 0.5 * peak_value
    left = peak_index
    while left > 0 and profile[left] > half_level:
        left -= 1
    right = peak_index
    while right < len(profile) - 1 and profile[right] > half_level:
        right += 1
    return float(abs(energy_axis[right] - energy_axis[left]))


def estimate_dispersion_shape_metrics(
    mean_spectrum: np.ndarray,
    energy_axis: np.ndarray,
    phi_axis: np.ndarray,
) -> tuple[float, float, float]:
    spectrum = np.asarray(mean_spectrum, dtype=np.float32)
    energy_axis = np.asarray(energy_axis, dtype=np.float32)
    phi_axis = np.asarray(phi_axis, dtype=np.float32)
    smoothed = ndimage.gaussian_filter(spectrum, sigma=(1.0, 1.0)).astype(np.float32)
    energy_profile = np.sum(smoothed, axis=1)
    threshold = 0.35 * float(np.max(energy_profile)) if energy_profile.size else 0.0

    ridge_energy: list[float] = []
    ridge_phi: list[float] = []
    for energy_index, row in enumerate(smoothed):
        if energy_profile[energy_index] < threshold:
            continue
        phi_index = int(np.argmax(row))
        ridge_energy.append(float(energy_axis[energy_index]))
        ridge_phi.append(float(phi_axis[phi_index]))

    if len(ridge_energy) < 3:
        return float("nan"), float("nan"), 0.0

    degree = 2 if len(ridge_energy) >= 4 else 1
    coefficients = np.polyfit(np.asarray(ridge_energy), np.asarray(ridge_phi), deg=degree)
    if degree == 1:
        slope = float(coefficients[0])
        curvature = 0.0
    else:
        slope = float(coefficients[-2])
        curvature = float(2.0 * coefficients[0])
    coverage = float(len(ridge_energy) / max(1, spectrum.shape[0]))
    return slope, curvature, coverage


def build_cluster_pairwise_physical_differences(
    metrics_rows: list[ClusterPhysicalMetrics],
    cluster_stats: list[SpectralClusterStats],
    energy_axis: np.ndarray,
) -> list[ClusterPairwisePhysicalDifference]:
    energy_step = median_axis_step(energy_axis)
    phi_energy_scale = max(0.2, 4.0 * energy_step)

    fermi_values = np.asarray([row.fermi_weight_fraction for row in metrics_rows], dtype=np.float32)
    gap_fill_values = np.asarray([row.gap_fill_ratio for row in metrics_rows], dtype=np.float32)
    near_ef_values = np.asarray([row.near_ef_weight_fraction for row in metrics_rows], dtype=np.float32)
    deep_values = np.asarray([row.deep_weight_fraction for row in metrics_rows], dtype=np.float32)
    shallow_values = np.asarray([row.shallow_weight_fraction for row in metrics_rows], dtype=np.float32)

    fermi_thresh = max(0.015, 0.35 * float(np.ptp(fermi_values)))
    gap_fill_thresh = max(0.08, 0.35 * float(np.ptp(gap_fill_values)))
    gap_ev_thresh = max(0.02, 3.0 * energy_step)
    peak_ev_thresh = max(0.02, 2.0 * energy_step)
    width_thresh = max(0.025, 2.0 * energy_step)
    transfer_thresh = max(0.025, 0.35 * float(max(np.ptp(deep_values), np.ptp(shallow_values), np.ptp(near_ef_values))))

    stat_by_cluster = {stats.cluster_id: stats for stats in cluster_stats}
    rows: list[ClusterPairwisePhysicalDifference] = []

    for first, second in combinations(metrics_rows, 2):
        first_spectrum = np.asarray(stat_by_cluster[first.cluster_id].mean_spectrum, dtype=np.float32)
        second_spectrum = np.asarray(stat_by_cluster[second.cluster_id].mean_spectrum, dtype=np.float32)
        dispersion_corr = normalized_spectrum_correlation(first_spectrum, second_spectrum)

        fermi_diff = float(second.fermi_weight_fraction - first.fermi_weight_fraction)
        gap_fill_diff = float(second.gap_fill_ratio - first.gap_fill_ratio)
        gap_proxy_diff = float(second.gap_proxy_ev - first.gap_proxy_ev)
        dominant_peak_diff = float(second.dominant_peak_ev - first.dominant_peak_ev)
        peak_width_diff = float(second.dominant_peak_width_ev - first.dominant_peak_width_ev)
        dispersion_slope_diff = float(second.dispersion_slope_phi_per_ev - first.dispersion_slope_phi_per_ev)
        dispersion_curvature_diff = float(second.dispersion_curvature_phi_per_ev2 - first.dispersion_curvature_phi_per_ev2)
        deep_diff = float(second.deep_weight_fraction - first.deep_weight_fraction)
        shallow_diff = float(second.shallow_weight_fraction - first.shallow_weight_fraction)
        near_ef_diff = float(second.near_ef_weight_fraction - first.near_ef_weight_fraction)

        fermi_meaningful = abs(fermi_diff) >= fermi_thresh
        gap_meaningful = abs(gap_fill_diff) >= gap_fill_thresh or abs(gap_proxy_diff) >= gap_ev_thresh
        peak_meaningful = abs(dominant_peak_diff) >= peak_ev_thresh
        width_meaningful = abs(peak_width_diff) >= width_thresh
        dispersion_meaningful = (
            dispersion_corr <= 0.96
            and (
                abs(dispersion_slope_diff) >= phi_energy_scale
                or abs(dispersion_curvature_diff) >= max(1.0, phi_energy_scale / max(energy_step, 1e-3))
                or dispersion_corr <= 0.90
            )
        )
        transfer_meaningful = max(abs(deep_diff), abs(shallow_diff), abs(near_ef_diff)) >= transfer_thresh
        overall_distinct = sum(
            int(flag)
            for flag in (
                fermi_meaningful,
                gap_meaningful,
                peak_meaningful,
                width_meaningful,
                dispersion_meaningful,
                transfer_meaningful,
            )
        ) >= 2 or dispersion_corr <= 0.88

        interpretation_parts: list[str] = []
        if fermi_meaningful:
            interpretation_parts.append(f"near-EF weight differs by {fermi_diff:+.3f}")
        if gap_meaningful:
            interpretation_parts.append(f"gap proxy changes by {gap_proxy_diff:+.3f} eV and gap filling by {gap_fill_diff:+.3f}")
        if peak_meaningful:
            interpretation_parts.append(f"dominant peak shifts by {dominant_peak_diff:+.3f} eV")
        if width_meaningful:
            interpretation_parts.append(f"peak width changes by {peak_width_diff:+.3f} eV")
        if dispersion_meaningful:
            interpretation_parts.append(f"dispersion correlation is {dispersion_corr:.3f}")
        if transfer_meaningful:
            interpretation_parts.append(
                f"spectral weight transfers deep/shallow/near-EF by {deep_diff:+.3f}/{shallow_diff:+.3f}/{near_ef_diff:+.3f}"
            )
        if not interpretation_parts:
            interpretation_parts.append("largest differences stay below the practical interpretation thresholds")

        rows.append(
            ClusterPairwisePhysicalDifference(
                cluster_a=first.cluster_id,
                cluster_b=second.cluster_id,
                fermi_weight_diff=fermi_diff,
                fermi_weight_meaningful=fermi_meaningful,
                gap_fill_ratio_diff=gap_fill_diff,
                gap_proxy_diff_ev=gap_proxy_diff,
                gap_difference_meaningful=gap_meaningful,
                dominant_peak_diff_ev=dominant_peak_diff,
                dominant_peak_meaningful=peak_meaningful,
                dominant_peak_width_diff_ev=peak_width_diff,
                peak_width_meaningful=width_meaningful,
                dispersion_shape_correlation=dispersion_corr,
                dispersion_slope_diff=dispersion_slope_diff,
                dispersion_curvature_diff=dispersion_curvature_diff,
                dispersion_meaningful=dispersion_meaningful,
                deep_weight_diff=deep_diff,
                shallow_weight_diff=shallow_diff,
                near_ef_weight_diff=near_ef_diff,
                spectral_weight_transfer_meaningful=transfer_meaningful,
                overall_physically_distinct=overall_distinct,
                interpretation="; ".join(interpretation_parts),
            )
        )

    return rows


def normalized_spectrum_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_flat = normalize_rows(np.asarray(first, dtype=np.float32).reshape(1, -1))[0]
    second_flat = normalize_rows(np.asarray(second, dtype=np.float32).reshape(1, -1))[0]
    first_centered = first_flat - float(np.mean(first_flat))
    second_centered = second_flat - float(np.mean(second_flat))
    denom = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denom <= 1e-10:
        return 1.0
    return float(np.dot(first_centered, second_centered) / denom)


def build_cluster_question_summaries(
    metrics_rows: list[ClusterPhysicalMetrics],
    pairwise_rows: list[ClusterPairwisePhysicalDifference],
    energy_axis: np.ndarray,
) -> list[ClusterPhysicalQuestionSummary]:
    summaries: list[ClusterPhysicalQuestionSummary] = []
    summaries.append(
        build_question_summary(
            "Do the cluster mean spectra differ in Fermi-level weight?",
            pairwise_rows,
            lambda row: row.fermi_weight_meaningful,
            lambda row: abs(row.fermi_weight_diff),
            lambda row: f"Strongest pair: C{row.cluster_a} vs C{row.cluster_b}.",
            lambda row: (
                f"The near-EF spectral-weight fraction shifts by {row.fermi_weight_diff:+.3f}, "
                "which is large enough to matter electronically."
            ),
        )
    )
    summaries.append(
        build_question_summary(
            "Do the cluster mean spectra differ in gap size or gap filling?",
            pairwise_rows,
            lambda row: row.gap_difference_meaningful,
            lambda row: max(abs(row.gap_proxy_diff_ev), abs(row.gap_fill_ratio_diff)),
            lambda row: f"Strongest pair: C{row.cluster_a} vs C{row.cluster_b}.",
            lambda row: (
                f"Gap proxy changes by {row.gap_proxy_diff_ev:+.3f} eV and gap-filling ratio by {row.gap_fill_ratio_diff:+.3f}, "
                "so the difference is more than a tiny numerical shift."
            ),
        )
    )
    summaries.append(
        build_question_summary(
            "Do the cluster mean spectra differ in peak positions?",
            pairwise_rows,
            lambda row: row.dominant_peak_meaningful,
            lambda row: abs(row.dominant_peak_diff_ev),
            lambda row: f"Strongest pair: C{row.cluster_a} vs C{row.cluster_b}.",
            lambda row: f"The dominant peak shifts by {row.dominant_peak_diff_ev:+.3f} eV, which is larger than the practical energy-resolution threshold.",
        )
    )
    summaries.append(
        build_question_summary(
            "Do the cluster mean spectra differ in peak widths?",
            pairwise_rows,
            lambda row: row.peak_width_meaningful,
            lambda row: abs(row.dominant_peak_width_diff_ev),
            lambda row: f"Strongest pair: C{row.cluster_a} vs C{row.cluster_b}.",
            lambda row: f"The dominant peak width changes by {row.dominant_peak_width_diff_ev:+.3f} eV, indicating meaningfully different broadening or filling.",
        )
    )
    summaries.append(
        build_question_summary(
            "Do the cluster mean spectra differ in dispersion shape?",
            pairwise_rows,
            lambda row: row.dispersion_meaningful,
            lambda row: 1.0 - row.dispersion_shape_correlation,
            lambda row: f"Strongest pair: C{row.cluster_a} vs C{row.cluster_b}.",
            lambda row: (
                f"The normalized mean-spectrum correlation drops to {row.dispersion_shape_correlation:.3f}, "
                f"with slope/curvature changes of {row.dispersion_slope_diff:+.3f} and {row.dispersion_curvature_diff:+.3f}."
            ),
        )
    )
    summaries.append(
        build_question_summary(
            "Do the cluster mean spectra differ in relative spectral weight transfers?",
            pairwise_rows,
            lambda row: row.spectral_weight_transfer_meaningful,
            lambda row: max(abs(row.deep_weight_diff), abs(row.shallow_weight_diff), abs(row.near_ef_weight_diff)),
            lambda row: f"Strongest pair: C{row.cluster_a} vs C{row.cluster_b}.",
            lambda row: (
                f"Deep/shallow/near-EF weights shift by {row.deep_weight_diff:+.3f}/{row.shallow_weight_diff:+.3f}/{row.near_ef_weight_diff:+.3f}, "
                "consistent with a real redistribution of spectral weight."
            ),
        )
    )
    return summaries


def build_question_summary(
    question: str,
    pairwise_rows: list[ClusterPairwisePhysicalDifference],
    predicate: Any,
    magnitude: Any,
    strongest_example_builder: Any,
    reasoning_builder: Any,
) -> ClusterPhysicalQuestionSummary:
    if not pairwise_rows:
        return ClusterPhysicalQuestionSummary(
            question=question,
            answer="No clear answer.",
            strongest_example="There were not enough distinct clusters to compare.",
            reasoning="",
        )

    meaningful = [row for row in pairwise_rows if predicate(row)]
    if meaningful:
        strongest = max(meaningful, key=magnitude)
        return ClusterPhysicalQuestionSummary(
            question=question,
            answer="Yes.",
            strongest_example=strongest_example_builder(strongest),
            reasoning=reasoning_builder(strongest),
        )

    strongest = max(pairwise_rows, key=magnitude)
    return ClusterPhysicalQuestionSummary(
        question=question,
        answer="No clear physically interpretable separation.",
        strongest_example=strongest_example_builder(strongest),
        reasoning="The largest observed difference still stayed below the practical threshold used to suppress tiny numerical separations.",
    )


def export_cluster_physical_interpretation(
    report: ClusterPhysicalInterpretation,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    if output_dir is None:
        state_path = Path(report.state_file).expanduser().resolve()
        output_path = state_path.parent / "clustering_reports"
    else:
        output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = sanitize_filename(Path(report.state_file).stem or report.state_name)
    metrics_path = output_path / f"{base_name}_cluster_physical_metrics.csv"
    pairwise_path = output_path / f"{base_name}_cluster_physical_differences.csv"
    summary_path = output_path / f"{base_name}_cluster_physical_summary.csv"

    write_rows_to_csv(metrics_path, [asdict(row) for row in report.metrics_rows])
    write_rows_to_csv(pairwise_path, [asdict(row) for row in report.pairwise_rows])
    write_rows_to_csv(summary_path, [asdict(row) for row in report.question_summaries])

    return {
        "metrics": metrics_path,
        "pairwise": pairwise_path,
        "summary": summary_path,
    }


def write_rows_to_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str]
    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    else:
        fieldnames = ["note"]
        rows = [{"note": "No rows were available."}]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_state(file_path: str, *, load: bool = True) -> LoadedState:
    resolved = str(Path(file_path).expanduser().resolve())
    suffix = Path(resolved).suffix.lower()
    dataset: xr.Dataset | None = None
    if suffix in TABLE_DATA_EXTENSIONS:
        data_array = load_tabular_dataarray(resolved)
    elif suffix in NUMPY_DATA_EXTENSIONS:
        data_array = load_numpy_dataarray(resolved)
    else:
        dataset = open_nc_dataset(resolved)
        data_array = prepare_main_dataarray(dataset)
        if load:
            try:
                data_array = data_array.load()
            finally:
                dataset.close()
                dataset = None

    return LoadedState(
        name=Path(resolved).name,
        file_path=resolved,
        data_array=data_array,
        dataset=dataset,
    )


def align_loaded_states_for_comparison(loaded_states: list[LoadedState]) -> tuple[list[LoadedState], list[str]]:
    """Crop loaded states to a shared spatial window for pixelwise comparisons."""
    states = list(loaded_states)
    if len(states) <= 1:
        return states, []

    validate_comparison_axes(states)

    spatial_shapes = [spatial_shape(state.data_array) for state in states]
    if len(set(spatial_shapes)) == 1 and spatial_coordinates_match(states):
        return states, []

    records = build_coordinate_spatial_alignment_records(states)
    if records is None:
        records = build_intensity_spatial_alignment_records(states)

    aligned_states: list[LoadedState] = []
    for state, record in zip(states, records):
        da = state.data_array.isel(
            x=slice(record.x_slice[0], record.x_slice[1]),
            y=slice(record.y_slice[0], record.y_slice[1]),
        )
        aligned_states.append(LoadedState(name=state.name, file_path=state.file_path, data_array=da))

    aligned_shapes = {spatial_shape(state.data_array) for state in aligned_states}
    if len(aligned_shapes) != 1:
        raise ValueError(f"Spatial alignment failed to produce a shared x/y shape: {sorted(aligned_shapes)}.")

    return aligned_states, format_spatial_alignment_notes(records)


def validate_comparison_axes(loaded_states: list[LoadedState]) -> None:
    reference_dims = loaded_states[0].data_array.dims
    for state in loaded_states[1:]:
        if state.data_array.dims != reference_dims:
            raise ValueError(
                "All files must share the same canonical dimensions after loading.\n"
                f"Expected dims={reference_dims}.\n"
                f"Received dims={state.data_array.dims} for {state.file_path}."
            )

    for dim in ("eV", "phi"):
        reference_axis = np.asarray(loaded_states[0].data_array.coords[dim].values, dtype=np.float32)
        for state in loaded_states[1:]:
            axis = np.asarray(state.data_array.coords[dim].values, dtype=np.float32)
            if axis.shape != reference_axis.shape:
                raise ValueError(
                    "Only x/y spatial clipping is supported for cross-file comparisons.\n"
                    f"Expected {dim} axis length {reference_axis.shape[0]}, received {axis.shape[0]} for {state.file_path}."
                )
            if not np.allclose(axis, reference_axis, rtol=1e-4, atol=1e-6):
                raise ValueError(
                    "Only files with matching eV and phi axes can be compared pixel-by-pixel.\n"
                    f"The {dim} coordinates differ for {state.file_path}."
                )


def spatial_shape(da: xr.DataArray) -> tuple[int, int]:
    return int(da.sizes["x"]), int(da.sizes["y"])


def spatial_coordinates_match(loaded_states: list[LoadedState]) -> bool:
    for dim in ("x", "y"):
        reference_axis = np.asarray(loaded_states[0].data_array.coords[dim].values, dtype=np.float32)
        for state in loaded_states[1:]:
            axis = np.asarray(state.data_array.coords[dim].values, dtype=np.float32)
            if axis.shape != reference_axis.shape or not np.allclose(axis, reference_axis, rtol=1e-4, atol=1e-6):
                return False
    return True


def build_coordinate_spatial_alignment_records(loaded_states: list[LoadedState]) -> list[SpatialAlignmentRecord] | None:
    axis_slices: dict[str, list[slice]] = {}
    target_counts: dict[str, int] = {}

    for dim in ("x", "y"):
        axes = [np.asarray(state.data_array.coords[dim].values, dtype=np.float64) for state in loaded_states]
        sizes = [axis.size for axis in axes]
        if len(set(sizes)) > 1 and all(axis_looks_like_default_index(axis) for axis in axes):
            return None
        if not all(axis_is_monotonic(axis) for axis in axes):
            return None

        lower = max(float(np.nanmin(axis)) for axis in axes)
        upper = min(float(np.nanmax(axis)) for axis in axes)
        if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
            return None

        slices_for_dim: list[slice] = []
        counts: list[int] = []
        for axis in axes:
            tolerance = coordinate_axis_tolerance(axis)
            indices = np.flatnonzero((axis >= lower - tolerance) & (axis <= upper + tolerance))
            if indices.size == 0:
                return None
            if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
                return None
            slices_for_dim.append(slice(int(indices[0]), int(indices[-1]) + 1))
            counts.append(int(indices.size))

        if len(set(counts)) != 1:
            return None
        axis_slices[dim] = slices_for_dim
        target_counts[dim] = counts[0]

    records: list[SpatialAlignmentRecord] = []
    aligned_shape = (target_counts["x"], target_counts["y"])
    for index, state in enumerate(loaded_states):
        x_slice = axis_slices["x"][index]
        y_slice = axis_slices["y"][index]
        records.append(
            SpatialAlignmentRecord(
                state_name=state.name,
                file_path=state.file_path,
                original_shape=spatial_shape(state.data_array),
                aligned_shape=aligned_shape,
                x_slice=(int(x_slice.start), int(x_slice.stop)),
                y_slice=(int(y_slice.start), int(y_slice.stop)),
                method="coordinate overlap",
            )
        )
    return records


def axis_looks_like_default_index(axis: np.ndarray) -> bool:
    axis = np.asarray(axis, dtype=np.float64)
    return axis.ndim == 1 and np.allclose(axis, np.arange(axis.size, dtype=np.float64), rtol=1e-6, atol=1e-6)


def axis_is_monotonic(axis: np.ndarray) -> bool:
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim != 1 or axis.size == 0 or not np.all(np.isfinite(axis)):
        return False
    if axis.size == 1:
        return True
    diff = np.diff(axis)
    return bool(np.all(diff > 0) or np.all(diff < 0))


def coordinate_axis_tolerance(axis: np.ndarray) -> float:
    axis = np.asarray(axis, dtype=np.float64)
    if axis.size < 2:
        return 1e-6
    steps = np.abs(np.diff(axis))
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if steps.size == 0:
        return 1e-6
    return max(1e-6, float(np.nanmedian(steps)) * 0.25)


def build_intensity_spatial_alignment_records(loaded_states: list[LoadedState]) -> list[SpatialAlignmentRecord]:
    shapes = [spatial_shape(state.data_array) for state in loaded_states]
    target_shape = (min(shape[0] for shape in shapes), min(shape[1] for shape in shapes))
    if target_shape[0] <= 0 or target_shape[1] <= 0:
        raise ValueError(f"Cannot align empty spatial shapes: {shapes}.")

    template_index = choose_alignment_template_index(shapes, target_shape)
    template_image_full = build_spatial_alignment_image(loaded_states[template_index].data_array)
    template_bounds = center_crop_bounds(shapes[template_index], target_shape)
    template_image = template_image_full[
        template_bounds[0][0] : template_bounds[0][1],
        template_bounds[1][0] : template_bounds[1][1],
    ]

    records: list[SpatialAlignmentRecord] = []
    for index, state in enumerate(loaded_states):
        image = build_spatial_alignment_image(state.data_array)
        if index == template_index:
            x_slice, y_slice = template_bounds
            score = None
            method = "reference clipped window"
        else:
            x_slice, y_slice, score = find_best_matching_spatial_crop(image, template_image)
            method = "intensity registration"

        records.append(
            SpatialAlignmentRecord(
                state_name=state.name,
                file_path=state.file_path,
                original_shape=spatial_shape(state.data_array),
                aligned_shape=target_shape,
                x_slice=x_slice,
                y_slice=y_slice,
                method=method,
                score=score,
            )
        )
    return records


def choose_alignment_template_index(shapes: list[tuple[int, int]], target_shape: tuple[int, int]) -> int:
    exact_matches = [index for index, shape in enumerate(shapes) if shape == target_shape]
    if exact_matches:
        return exact_matches[0]
    return min(range(len(shapes)), key=lambda index: (shapes[index][0] * shapes[index][1], index))


def center_crop_bounds(shape: tuple[int, int], target_shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    x_extra = shape[0] - target_shape[0]
    y_extra = shape[1] - target_shape[1]
    if x_extra < 0 or y_extra < 0:
        raise ValueError(f"Cannot crop shape {shape} to larger target shape {target_shape}.")
    x_start = max(0, x_extra // 2)
    y_start = max(0, y_extra // 2)
    return (x_start, x_start + target_shape[0]), (y_start, y_start + target_shape[1])


def build_spatial_alignment_image(da: xr.DataArray) -> np.ndarray:
    values = np.asarray(da.values, dtype=np.float32)
    total = np.nansum(values, axis=(2, 3), dtype=np.float64).astype(np.float32)
    finite = total[np.isfinite(total)]
    if finite.size == 0:
        return np.zeros(total.shape, dtype=np.float32)
    shifted = np.clip(total - float(np.nanmin(finite)), a_min=0.0, a_max=None)
    return finite_fill(np.log1p(shifted), 0.0).astype(np.float32)


def find_best_matching_spatial_crop(
    image: np.ndarray,
    template: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], float | None]:
    image = np.asarray(image, dtype=np.float32)
    template = np.asarray(template, dtype=np.float32)
    target_x, target_y = template.shape
    if image.shape[0] < target_x or image.shape[1] < target_y:
        raise ValueError(f"Cannot align image shape {image.shape} to larger template shape {template.shape}.")

    if image.shape == template.shape:
        score = normalized_crop_correlation(image, template)
        return (0, target_x), (0, target_y), score

    best_score = -np.inf
    best_x = 0
    best_y = 0
    max_x = image.shape[0] - target_x
    max_y = image.shape[1] - target_y
    for x_start in range(max_x + 1):
        for y_start in range(max_y + 1):
            crop = image[x_start : x_start + target_x, y_start : y_start + target_y]
            score = normalized_crop_correlation(crop, template)
            if score is not None and score > best_score:
                best_score = score
                best_x = x_start
                best_y = y_start

    if not np.isfinite(best_score):
        (x_start, x_stop), (y_start, y_stop) = center_crop_bounds(image.shape, template.shape)
        return (x_start, x_stop), (y_start, y_stop), None
    return (best_x, best_x + target_x), (best_y, best_y + target_y), float(best_score)


def normalized_crop_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return None
    av = a[mask].astype(np.float64)
    bv = b[mask].astype(np.float64)
    av -= float(av.mean())
    bv -= float(bv.mean())
    denominator = float(np.sqrt(np.sum(av * av) * np.sum(bv * bv)))
    if denominator <= 1e-12:
        return None
    return float(np.sum(av * bv) / denominator)


def format_spatial_alignment_notes(records: list[SpatialAlignmentRecord]) -> list[str]:
    if not records:
        return []
    aligned_shape = records[0].aligned_shape
    notes = [
        f"Spatially aligned files to a shared {aligned_shape[0]} x {aligned_shape[1]} x/y window before pixel comparisons."
    ]
    for record in records:
        full_x = record.x_slice == (0, record.original_shape[0])
        full_y = record.y_slice == (0, record.original_shape[1])
        if record.original_shape == record.aligned_shape and full_x and full_y:
            continue
        score_text = f", match score={record.score:.3f}" if record.score is not None else ""
        notes.append(
            f"{record.state_name}: original x/y={record.original_shape}, using "
            f"x={record.x_slice[0]}:{record.x_slice[1]}, y={record.y_slice[0]}:{record.y_slice[1]} "
            f"({record.method}{score_text})."
        )
    return notes


def open_nc_dataset(file_path: str) -> xr.Dataset:
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    engines_to_try: list[str | None] = ["h5netcdf", "scipy", None]

    errors: list[str] = []
    for engine in engines_to_try:
        try:
            if engine is None:
                dataset = xr.open_dataset(file_path)
            elif engine == "h5netcdf":
                dataset = xr.open_dataset(file_path, engine=engine, chunks="auto")
            else:
                dataset = xr.open_dataset(file_path, engine=engine)
            if dataset.data_vars:
                return dataset
            dataset.close()
            if engine == "h5netcdf":
                for group in numeric_hdf5_groups(file_path):
                    grouped = xr.open_dataset(file_path, engine=engine, group=group, chunks="auto")
                    if grouped.data_vars:
                        return grouped
                    grouped.close()
                errors.append("h5netcdf: root dataset contained no data variables and no numeric groups were found")
        except Exception as exc:  # pragma: no cover - exercised through multiple runtime backends
            label = "default" if engine is None else engine
            errors.append(f"{label}: {exc}")

    joined = "\n".join(errors) if errors else "No engines attempted."
    raise RuntimeError(f"Could not open dataset {file_path}.\n{joined}")


def numeric_hdf5_groups(file_path: str) -> list[str]:
    try:
        import h5py
    except ImportError:
        return []

    groups: list[tuple[str, int]] = []
    try:
        with h5py.File(file_path, "r") as handle:
            def visit_group(name: str, obj: Any) -> None:
                if not isinstance(obj, h5py.Group):
                    return
                numeric_size = 0
                for item in obj.values():
                    if isinstance(item, h5py.Dataset) and np.issubdtype(item.dtype, np.number) and item.shape:
                        numeric_size += int(np.prod(item.shape))
                if numeric_size > 0:
                    groups.append((name, numeric_size))

            handle.visititems(visit_group)
    except Exception:
        return []

    groups.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _size in groups]


def load_tabular_dataarray(file_path: str) -> xr.DataArray:
    """Load long-form ARPES table exports.

    Supported table columns are x, y, eV/energy, phi/angle, and intensity/counts.
    Headerless files are interpreted as the first five columns in that order.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    delimiter = sniff_table_delimiter(path)
    has_header = table_appears_to_have_header(path)
    if has_header:
        table = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=np.float64, encoding=None)
        if table.dtype.names is None:
            raise ValueError(f"Could not read named columns from table file: {file_path}")
        table = np.atleast_1d(table)
        columns = {name: np.asarray(table[name], dtype=np.float64) for name in table.dtype.names}
        x = get_named_table_column(columns, ("x", "x_index", "xindex", "xpos", "x_pos", "pixel_x", "scan_x"))
        y = get_named_table_column(columns, ("y", "y_index", "yindex", "ypos", "y_pos", "pixel_y", "scan_y"))
        e = get_named_table_column(columns, ("ev", "e_v", "energy", "bindingenergy", "binding_energy", "ene", "e"))
        phi = get_named_table_column(columns, ("phi", "angle", "angles", "theta", "momentum", "kx", "ky", "k"))
        intensity = get_named_table_column(columns, ("intensity", "counts", "count", "signal", "value", "i", "z"))
    else:
        raw = np.genfromtxt(path, delimiter=delimiter, dtype=np.float64)
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.ndim != 2 or raw.shape[1] < 5:
            raise ValueError(
                "Headerless ARPES tables must have at least five columns: x, y, eV, phi, intensity."
            )
        x, y, e, phi, intensity = [raw[:, index] for index in range(5)]

    return build_dataarray_from_long_table(x, y, e, phi, intensity, name=path.stem)


def sniff_table_delimiter(path: Path) -> str | None:
    line = first_nonempty_data_line(path)
    if "," in line:
        return ","
    if "\t" in line:
        return "\t"
    return None


def table_appears_to_have_header(path: Path) -> bool:
    line = first_nonempty_data_line(path)
    return any(character.isalpha() for character in line)


def first_nonempty_data_line(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    raise ValueError(f"Table file is empty: {path}")


def normalized_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def get_named_table_column(columns: dict[str, np.ndarray], aliases: tuple[str, ...]) -> np.ndarray:
    normalized = {normalized_column_name(name): value for name, value in columns.items()}
    for alias in aliases:
        value = normalized.get(normalized_column_name(alias))
        if value is not None:
            return value
    normalized_aliases = [normalized_column_name(alias) for alias in aliases]
    for name, value in normalized.items():
        for alias in normalized_aliases:
            if alias in {"x", "y"} and name.startswith(alias):
                return value
            if len(alias) > 1 and alias in name:
                return value
    available = ", ".join(columns.keys())
    raise ValueError(f"Missing required ARPES table column. Expected one of {aliases}; found {available}.")


def build_dataarray_from_long_table(
    x_values: np.ndarray,
    y_values: np.ndarray,
    e_values: np.ndarray,
    phi_values: np.ndarray,
    intensity_values: np.ndarray,
    *,
    name: str,
) -> xr.DataArray:
    arrays = [
        np.asarray(values, dtype=np.float64).reshape(-1)
        for values in (x_values, y_values, e_values, phi_values, intensity_values)
    ]
    lengths = {array.size for array in arrays}
    if len(lengths) != 1:
        raise ValueError("The x, y, eV, phi, and intensity table columns must have the same length.")

    x, y, e, phi, intensity = arrays
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & np.isfinite(phi) & np.isfinite(intensity)
    if not np.any(finite):
        raise ValueError("No finite ARPES table rows were found.")

    x = x[finite]
    y = y[finite]
    e = e[finite]
    phi = phi[finite]
    intensity = intensity[finite]

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    e_unique = np.unique(e)
    phi_unique = np.unique(phi)

    x_axis = x_unique.astype(np.float32)
    y_axis = y_unique.astype(np.float32)
    e_axis = e_unique.astype(np.float32)
    phi_axis = phi_unique.astype(np.float32)

    shape = (x_axis.size, y_axis.size, e_axis.size, phi_axis.size)
    if any(size == 0 for size in shape):
        raise ValueError("The ARPES table did not contain a complete set of coordinate axes.")

    x_index = np.searchsorted(x_unique, x)
    y_index = np.searchsorted(y_unique, y)
    e_index = np.searchsorted(e_unique, e)
    phi_index = np.searchsorted(phi_unique, phi)
    flat_index = np.ravel_multi_index((x_index, y_index, e_index, phi_index), shape)

    sums = np.zeros(int(np.prod(shape)), dtype=np.float64)
    counts = np.zeros(int(np.prod(shape)), dtype=np.int64)
    np.add.at(sums, flat_index, intensity)
    np.add.at(counts, flat_index, 1)

    values = np.full(sums.shape, np.nan, dtype=np.float32)
    present = counts > 0
    values[present] = (sums[present] / counts[present]).astype(np.float32)
    values = values.reshape(shape)

    return xr.DataArray(
        values,
        dims=REQUIRED_DIMS,
        coords={"x": x_axis, "y": y_axis, "eV": e_axis, "phi": phi_axis},
        name=name or "intensity",
    )


def load_numpy_dataarray(file_path: str) -> xr.DataArray:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".npy":
        values = np.load(path)
        coords: dict[str, np.ndarray] = {}
    else:
        archive = np.load(path)
        try:
            data_key = choose_numpy_data_key(archive)
            values = archive[data_key]
            coords = {
                canonical: np.asarray(archive[key], dtype=np.float32)
                for canonical in REQUIRED_DIMS
                for key in matching_numpy_coord_keys(archive, canonical)[:1]
            }
        finally:
            archive.close()

    values = np.asarray(values, dtype=np.float32)
    values = np.squeeze(values)
    if values.ndim != 4:
        raise ValueError(
            f"NumPy ARPES data must be four-dimensional after squeezing; got shape {values.shape}."
        )

    default_coords = {
        "x": np.arange(values.shape[0], dtype=np.float32),
        "y": np.arange(values.shape[1], dtype=np.float32),
        "eV": np.arange(values.shape[2], dtype=np.float32),
        "phi": np.arange(values.shape[3], dtype=np.float32),
    }
    default_coords.update({key: value for key, value in coords.items() if value.shape == default_coords[key].shape})
    return xr.DataArray(values, dims=REQUIRED_DIMS, coords=default_coords, name=path.stem or "intensity")


def choose_numpy_data_key(archive: np.lib.npyio.NpzFile) -> str:
    preferred = ("intensity", "data", "cube", "arpes", "values")
    names = list(archive.files)
    for key in preferred:
        if key in names:
            return key
    numeric_arrays: list[tuple[str, int]] = []
    for key in names:
        array = archive[key]
        if np.issubdtype(array.dtype, np.number) and np.squeeze(array).ndim == 4:
            numeric_arrays.append((key, int(np.prod(array.shape))))
    if not numeric_arrays:
        raise ValueError("No four-dimensional numeric data array was found in the NumPy archive.")
    numeric_arrays.sort(key=lambda item: item[1], reverse=True)
    return numeric_arrays[0][0]


def matching_numpy_coord_keys(archive: np.lib.npyio.NpzFile, canonical: str) -> list[str]:
    aliases = {
        "x": ("x", "x_axis", "xaxis", "x_coords", "xcoords"),
        "y": ("y", "y_axis", "yaxis", "y_coords", "ycoords"),
        "eV": ("ev", "e_v", "energy", "energy_axis", "binding_energy", "bindingenergy"),
        "phi": ("phi", "phi_axis", "angle", "angle_axis", "theta", "theta_axis"),
    }[canonical]
    normalized_aliases = {normalized_column_name(alias) for alias in aliases}
    return [key for key in archive.files if normalized_column_name(key) in normalized_aliases]


def load_topography_image(file_path: str) -> np.ndarray:
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is available in the app environment
        raise RuntimeError("Loading SEM TIFF files requires Pillow to be installed.") from exc

    with Image.open(path) as image:
        image.seek(0)
        array = np.asarray(image)

    array = np.asarray(array)
    array = np.squeeze(array)
    if array.ndim == 3:
        if array.shape[-1] >= 3:
            rgb = array[..., :3].astype(np.float32)
            array = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        elif array.shape[-1] == 1:
            array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"SEM topography image must be two-dimensional after conversion; got shape {array.shape}.")
    return np.asarray(array, dtype=np.float32)


def robust_normalize_map(values: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.float32)
    low = float(np.nanpercentile(finite, lower_percentile))
    high = float(np.nanpercentile(finite, upper_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(array.shape, dtype=np.float32)
    normalized = (array - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)
    return finite_fill(normalized, 0.0).astype(np.float32)


def resample_spatial_map(values: np.ndarray, target_shape: tuple[int, int], order: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Spatial map must be two-dimensional; got shape {array.shape}.")
    if array.shape == target_shape:
        return array.astype(np.float32, copy=True)
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("Cannot resample an empty spatial map.")

    zoom = (target_shape[0] / array.shape[0], target_shape[1] / array.shape[1])
    resized = ndimage.zoom(array, zoom=zoom, order=order).astype(np.float32)
    if resized.shape == target_shape:
        return resized

    out = np.full(target_shape, np.nan, dtype=np.float32)
    x_count = min(target_shape[0], resized.shape[0])
    y_count = min(target_shape[1], resized.shape[1])
    out[:x_count, :y_count] = resized[:x_count, :y_count]
    return out


def prepare_main_dataarray(dataset: xr.Dataset) -> xr.DataArray:
    data_array = get_main_dataarray(dataset).squeeze(drop=True)
    rename_map = guess_dimension_rename_map(data_array.dims)
    if rename_map:
        data_array = data_array.rename(rename_map)

    missing = [dim for dim in REQUIRED_DIMS if dim not in data_array.dims]
    if missing:
        raise ValueError(
            f"Missing required dimensions {missing}. Found dimensions {data_array.dims}."
        )

    extra_dims = [dim for dim in data_array.dims if dim not in REQUIRED_DIMS]
    if extra_dims:
        raise ValueError(
            "Only four analysis dimensions are supported after squeezing singleton axes. "
            f"Unexpected dimensions: {extra_dims}."
        )

    return data_array.transpose(*REQUIRED_DIMS)


def get_main_dataarray(dataset: xr.Dataset) -> xr.DataArray:
    candidates: list[tuple[str, int]] = []
    for name, variable in dataset.data_vars.items():
        try:
            if np.issubdtype(variable.dtype, np.number):
                candidates.append((name, int(np.prod(variable.shape))))
        except TypeError:
            continue

    if not candidates:
        raise ValueError("No numeric data variables were found in the dataset.")

    candidates.sort(key=lambda item: item[1], reverse=True)
    return dataset[candidates[0][0]]


def guess_dimension_rename_map(dims: tuple[str, ...]) -> dict[str, str]:
    rename_map: dict[str, str] = {}
    used: set[str] = set()
    available = list(dims)

    for canonical in REQUIRED_DIMS:
        exact = next((dim for dim in available if dim.lower() == canonical.lower()), None)
        if exact is not None:
            used.add(exact)
            if exact != canonical:
                rename_map[exact] = canonical

    for canonical in REQUIRED_DIMS:
        if canonical in rename_map.values() or canonical in used:
            continue
        guessed = guess_dim_name(available, canonical, used)
        if guessed is not None:
            used.add(guessed)
            rename_map[guessed] = canonical

    return rename_map


def guess_dim_name(dims: list[str], canonical: str, used: set[str]) -> str | None:
    alias_groups = {
        "x": (("x",), ("x_", "_x", "xpos", "x_pos")),
        "y": (("y",), ("y_", "_y", "ypos", "y_pos")),
        "eV": (("ev", "energy", "bindingenergy", "binding_energy", "ene"), ("binding", "energy", "ev")),
        "phi": (
            ("phi", "angle", "angles", "theta", "momentum", "kx", "ky", "kp", "k"),
            ("phi", "angle", "theta", "momentum", "kx", "ky", "kp"),
        ),
    }
    exact_aliases, partial_aliases = alias_groups[canonical]

    ranked: list[tuple[int, str]] = []
    for dim in dims:
        if dim in used:
            continue
        lowered = dim.lower()
        if lowered in exact_aliases:
            ranked.append((0, dim))
            continue
        if any(alias in lowered for alias in partial_aliases):
            ranked.append((1, dim))

    ranked.sort()
    return ranked[0][1] if ranked else None


def total_and_ef_maps(da: xr.DataArray, fermi_level: float = 0.0, ef_window: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    require_dims(da)
    energy_axis = np.asarray(da.coords["eV"].values, dtype=np.float32)
    ef_mask = get_energy_mask(energy_axis, center=fermi_level, halfwidth=ef_window)
    if not ef_mask.any():
        raise ValueError(
            f"No energy samples were found inside |E - {fermi_level:.3f}| <= {ef_window:.3f} eV."
        )

    total_map = np.asarray(da.sum(dim=("eV", "phi")).values, dtype=np.float32)
    ef_map = np.asarray(da.isel(eV=np.flatnonzero(ef_mask)).sum(dim=("eV", "phi")).values, dtype=np.float32)
    return total_map, ef_map


def require_dims(da: xr.DataArray) -> None:
    missing = [dim for dim in REQUIRED_DIMS if dim not in da.dims]
    if missing:
        raise ValueError(f"Missing required dimensions {missing}. Found {da.dims}.")


def get_energy_mask(energy_axis: np.ndarray, center: float = 0.0, halfwidth: float = 0.05) -> np.ndarray:
    energy_axis = np.asarray(energy_axis, dtype=np.float32)
    return np.abs(energy_axis - center) <= halfwidth


def build_cross_mask_from_maps(
    total_maps: list[np.ndarray],
    threshold_quantile: float = 0.45,
    row_fraction: float = 0.18,
    col_fraction: float = 0.18,
    background_quantile: float = 0.10,
    pad: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normalized_maps: list[np.ndarray] = []
    for total_map in total_maps:
        arr = np.asarray(total_map, dtype=np.float32)
        low = float(np.nanmin(arr))
        high = float(np.nanmax(arr))
        if math.isclose(high, low):
            normalized_maps.append(np.zeros_like(arr, dtype=np.float32))
        else:
            normalized_maps.append((arr - low) / (high - low))

    average_normalized_total_map = np.mean(normalized_maps, axis=0).astype(np.float32)
    threshold = float(np.quantile(average_normalized_total_map.reshape(-1), threshold_quantile))
    active_mask = average_normalized_total_map >= threshold

    row_occupancy = active_mask.mean(axis=1)
    col_occupancy = active_mask.mean(axis=0)

    strong_rows = row_occupancy >= row_fraction
    strong_cols = col_occupancy >= col_fraction

    cross_mask = strong_rows[:, None] | strong_cols[None, :]
    background_threshold = float(np.quantile(average_normalized_total_map.reshape(-1), background_quantile))
    cross_mask = cross_mask & (average_normalized_total_map >= background_threshold)

    if pad > 0:
        cross_mask = dilate_mask(cross_mask, n_iter=pad)

    return (
        cross_mask.astype(bool),
        average_normalized_total_map,
        active_mask.astype(bool),
        row_occupancy.astype(np.float32),
        col_occupancy.astype(np.float32),
    )


def dilate_mask(mask: np.ndarray, n_iter: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(n_iter))):
        padded = np.pad(out, 1, mode="edge")
        neighbors = [
            padded[0:-2, 0:-2],
            padded[0:-2, 1:-1],
            padded[0:-2, 2:],
            padded[1:-1, 0:-2],
            padded[1:-1, 1:-1],
            padded[1:-1, 2:],
            padded[2:, 0:-2],
            padded[2:, 1:-1],
            padded[2:, 2:],
        ]
        out = np.logical_or.reduce(neighbors)
    return out


def extract_pixel_features(
    da: xr.DataArray,
    fermi_level: float = 0.0,
    ef_window: float = 0.05,
    wide_window: float = 0.20,
) -> tuple[dict[str, np.ndarray], list[str], np.ndarray]:
    require_dims(da)

    data = np.asarray(da.values, dtype=np.float32)
    x_size, y_size, e_size, phi_size = data.shape

    energy_axis = np.asarray(da.coords["eV"].values, dtype=np.float32)
    phi_axis = np.asarray(da.coords["phi"].values, dtype=np.float32)

    ef_mask = get_energy_mask(energy_axis, center=fermi_level, halfwidth=ef_window)
    wide_mask = get_energy_mask(energy_axis, center=fermi_level, halfwidth=wide_window)
    if not ef_mask.any():
        raise ValueError(
            f"No energy samples were found inside the near-EF window centered at {fermi_level:.3f} eV."
        )
    if not wide_mask.any():
        raise ValueError(
            f"No energy samples were found inside the wide window centered at {fermi_level:.3f} eV."
        )

    spectra = data.reshape(x_size * y_size, e_size, phi_size)
    finite_spectra = finite_fill(spectra, 0.0)
    total_intensity = np.nansum(spectra, axis=(1, 2))
    ef_intensity = np.nansum(spectra[:, ef_mask, :], axis=(1, 2))
    wide_intensity = np.nansum(spectra[:, wide_mask, :], axis=(1, 2))

    ef_fraction = safe_divide(ef_intensity, total_intensity)
    wide_fraction = safe_divide(wide_intensity, total_intensity)

    energy_profile = np.nansum(spectra, axis=2)
    phi_profile = np.nansum(spectra, axis=1)

    energy_profile_norm = normalize_rows(energy_profile)
    phi_profile_norm = normalize_rows(phi_profile)

    e_centroid = (energy_profile_norm * energy_axis[None, :]).sum(axis=1)
    e_var = (energy_profile_norm * (energy_axis[None, :] - e_centroid[:, None]) ** 2).sum(axis=1)

    phi_centroid = (phi_profile_norm * phi_axis[None, :]).sum(axis=1)
    phi_var = (phi_profile_norm * (phi_axis[None, :] - phi_centroid[:, None]) ** 2).sum(axis=1)

    phi_mid = len(phi_axis) // 2
    left_intensity = phi_profile[:, :phi_mid].sum(axis=1)
    right_intensity = phi_profile[:, phi_mid:].sum(axis=1)
    phi_asymmetry = safe_divide(right_intensity - left_intensity, right_intensity + left_intensity)

    spectra_flat = np.clip(finite_spectra.reshape(x_size * y_size, -1), 0.0, None)
    spectra_norm = normalize_rows(spectra_flat)
    spectral_entropy = -np.sum(spectra_norm * np.log(spectra_norm + 1e-12), axis=1)
    spectral_max = spectra_flat.max(axis=1)
    spectral_mean = spectra_flat.mean(axis=1)
    spectral_sharpness = safe_divide(spectral_max, spectral_mean)

    ef_map = ef_intensity.reshape(x_size, y_size)
    grad_x, grad_y = np.gradient(ef_map)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    padded_ef = np.pad(ef_map, 1, mode="edge")
    neighbors = [
        padded_ef[0:-2, 1:-1],
        padded_ef[2:, 1:-1],
        padded_ef[1:-1, 0:-2],
        padded_ef[1:-1, 2:],
    ]
    neighbor_diff = np.mean([np.abs(ef_map - neighbor) for neighbor in neighbors], axis=0).astype(np.float32)

    windows = np.lib.stride_tricks.sliding_window_view(np.pad(ef_map, 1, mode="reflect"), (3, 3))
    local_contrast = windows.std(axis=(-2, -1)).astype(np.float32)

    features = {
        "total_intensity": total_intensity.reshape(x_size, y_size).astype(np.float32),
        "ef_intensity": ef_intensity.reshape(x_size, y_size).astype(np.float32),
        "wide_intensity": wide_intensity.reshape(x_size, y_size).astype(np.float32),
        "ef_fraction": ef_fraction.reshape(x_size, y_size).astype(np.float32),
        "wide_fraction": wide_fraction.reshape(x_size, y_size).astype(np.float32),
        "e_centroid": e_centroid.reshape(x_size, y_size).astype(np.float32),
        "e_var": e_var.reshape(x_size, y_size).astype(np.float32),
        "phi_centroid": phi_centroid.reshape(x_size, y_size).astype(np.float32),
        "phi_var": phi_var.reshape(x_size, y_size).astype(np.float32),
        "phi_asymmetry": phi_asymmetry.reshape(x_size, y_size).astype(np.float32),
        "spectral_entropy": spectral_entropy.reshape(x_size, y_size).astype(np.float32),
        "spectral_sharpness": spectral_sharpness.reshape(x_size, y_size).astype(np.float32),
        "ef_grad_mag": grad_mag.astype(np.float32),
        "ef_neighbor_diff": neighbor_diff,
        "ef_local_contrast": local_contrast,
    }

    feature_names = [
        "total_intensity",
        "ef_intensity",
        "wide_intensity",
        "ef_fraction",
        "wide_fraction",
        "e_centroid",
        "e_var",
        "phi_centroid",
        "phi_var",
        "phi_asymmetry",
        "spectral_entropy",
        "spectral_sharpness",
        "ef_grad_mag",
        "ef_neighbor_diff",
        "ef_local_contrast",
    ]

    feature_matrix = np.stack([features[name].reshape(-1) for name in feature_names], axis=1).astype(np.float32)
    feature_matrix = finite_fill(feature_matrix, 0.0)
    return features, feature_names, feature_matrix


def run_state_classification(
    file_path: str | Path,
    parameters: StateClassifierParameters | None = None,
) -> StateClassificationResult:
    state = load_state(str(file_path))
    return compute_state_classification(state, parameters=parameters)


def compute_state_classification(
    state: LoadedState,
    parameters: StateClassifierParameters | None = None,
) -> StateClassificationResult:
    if parameters is None:
        parameters = StateClassifierParameters()
    parameters.validate()

    feature_maps, normalized_maps, orientation_feature_name, notes = compute_state_classifier_feature_maps(
        state.data_array,
        parameters,
    )
    return classify_state_feature_maps(
        state,
        parameters,
        feature_maps,
        normalized_maps=normalized_maps,
        orientation_feature_name=orientation_feature_name,
        notes=notes,
    )


def compute_state_classifier_feature_maps(
    da: xr.DataArray,
    parameters: StateClassifierParameters,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str, list[str]]:
    require_dims(da)
    parameters.validate()

    data = np.asarray(da.values, dtype=np.float32)
    energy_axis = np.asarray(da.coords["eV"].values, dtype=np.float32)
    phi_axis = np.asarray(da.coords["phi"].values, dtype=np.float32)

    energy_order = np.argsort(energy_axis)
    phi_order = np.argsort(phi_axis)
    energy_axis = energy_axis[energy_order]
    phi_axis = phi_axis[phi_order]
    data = data[:, :, energy_order, :]
    data = data[:, :, :, phi_order]

    x_size, y_size, _e_size, _phi_size = data.shape
    edc = _integrate_along_axis(data, phi_axis, axis=3).astype(np.float32)
    smoothed_edc = (
        ndimage.gaussian_filter1d(edc, sigma=parameters.smooth_sigma, axis=2, mode="nearest")
        if parameters.smooth_sigma > 0
        else edc
    ).astype(np.float32)

    total_intensity = _integrate_along_axis(edc, energy_axis, axis=2).astype(np.float32)

    ef_mask = _energy_window_mask(
        energy_axis,
        parameters.fermi_level_ev + parameters.ef_min_ev,
        parameters.fermi_level_ev + parameters.ef_max_ev,
    )
    lhb_mask = get_energy_mask(
        energy_axis,
        center=parameters.lhb_center_ev,
        halfwidth=parameters.lhb_halfwidth_ev,
    )
    leading_edge_mask = _energy_window_mask(
        energy_axis,
        parameters.fermi_level_ev + parameters.leading_edge_min_ev,
        parameters.fermi_level_ev + parameters.leading_edge_max_ev,
    )
    p3_mask = get_energy_mask(
        energy_axis,
        center=parameters.p3_center_ev,
        halfwidth=parameters.p3_halfwidth_ev,
    )
    if not ef_mask.any():
        raise ValueError("No energy samples were found inside the near-EF classifier window.")
    if not lhb_mask.any():
        raise ValueError("No energy samples were found inside the LHB/p1 classifier window.")
    if not leading_edge_mask.any():
        raise ValueError("No energy samples were found inside the leading-edge classifier window.")

    w_ef = _integrate_window(smoothed_edc, energy_axis, ef_mask).astype(np.float32)
    w_lhb = _integrate_window(smoothed_edc, energy_axis, lhb_mask).astype(np.float32)
    i_rat = safe_divide(w_ef, w_lhb, eps=parameters.epsilon).astype(np.float32)
    e_lhb = _peak_energy_map(smoothed_edc, energy_axis, lhb_mask).astype(np.float32)
    e_le = _leading_edge_map(smoothed_edc, energy_axis, leading_edge_mask, parameters.fermi_level_ev).astype(np.float32)
    gamma_edc = _fwhm_map(smoothed_edc, energy_axis, lhb_mask).astype(np.float32)

    notes: list[str] = []
    if np.count_nonzero(p3_mask) >= 2:
        s_orient = _peak_energy_map(smoothed_edc, energy_axis, p3_mask).astype(np.float32)
        orientation_feature_name = "E_p3"
    else:
        phi_profile = _integrate_along_axis(data, energy_axis, axis=2).reshape(x_size * y_size, len(phi_axis))
        phi_norm = normalize_rows(phi_profile)
        s_orient = (phi_norm * phi_axis[None, :]).sum(axis=1).reshape(x_size, y_size).astype(np.float32)
        orientation_feature_name = "phi_COM"
        notes.append("p3 window did not contain enough energy samples; using phi center-of-mass as S_orient.")

    feature_maps = {
        "T": total_intensity,
        "W_EF": w_ef,
        "W_LHB": w_lhb,
        "I_rat": i_rat,
        "E_LHB": e_lhb,
        "E_LE": e_le,
        "Gamma_EDC": gamma_edc,
        "S_orient": s_orient,
    }
    normalized_maps = build_state_classifier_normalized_maps(
        feature_maps,
        fermi_level=parameters.fermi_level_ev,
    )
    return feature_maps, normalized_maps, orientation_feature_name, notes


def classify_state_feature_maps(
    state: LoadedState,
    parameters: StateClassifierParameters,
    feature_maps: dict[str, np.ndarray],
    normalized_maps: dict[str, np.ndarray] | None = None,
    orientation_feature_name: str = "S_orient",
    notes: list[str] | None = None,
) -> StateClassificationResult:
    parameters.validate()
    feature_maps = {
        name: np.asarray(feature_maps[name], dtype=np.float32)
        for name in STATE_CLASSIFICATION_FEATURE_NAMES
    }
    if normalized_maps is None:
        normalized_maps = build_state_classifier_normalized_maps(
            feature_maps,
            fermi_level=parameters.fermi_level_ev,
        )

    shape = feature_maps["T"].shape
    finite_core = np.ones(shape, dtype=bool)
    for name in STATE_CLASSIFICATION_FEATURE_NAMES:
        finite_core &= np.isfinite(feature_maps[name])

    t_low = _robust_percentile(feature_maps["T"], parameters.low_signal_quantile, finite_core)
    lhb_min = _robust_percentile(feature_maps["W_LHB"], parameters.lhb_min_quantile, finite_core)
    low_signal = (
        ~finite_core
        | (feature_maps["T"] <= t_low)
        | (feature_maps["W_LHB"] <= lhb_min)
    )
    valid_mask = ~low_signal

    low_irat = _robust_percentile(feature_maps["I_rat"], parameters.low_quantile, valid_mask)
    high_irat = _robust_percentile(feature_maps["I_rat"], parameters.high_quantile, valid_mask)
    low_wef = _robust_percentile(feature_maps["W_EF"], parameters.low_quantile, valid_mask)
    high_wef = _robust_percentile(feature_maps["W_EF"], parameters.high_quantile, valid_mask)
    le_far = _robust_percentile(feature_maps["E_LE"], parameters.low_quantile, valid_mask)
    le_close = _robust_percentile(feature_maps["E_LE"], parameters.high_quantile, valid_mask)
    broad = _robust_percentile(feature_maps["Gamma_EDC"], parameters.broad_quantile, valid_mask)
    lhb_reference = _robust_percentile(feature_maps["E_LHB"], 0.50, valid_mask)
    lhb_shift = np.abs(feature_maps["E_LHB"] - lhb_reference).astype(np.float32)
    lhb_close_threshold = _robust_percentile(lhb_shift, parameters.high_quantile, valid_mask)

    orientation_values = feature_maps["S_orient"]
    orientation_median = _robust_percentile(orientation_values, 0.50, valid_mask)
    orientation_shift = np.abs(orientation_values - orientation_median).astype(np.float32)
    orientation_shift_threshold = _robust_percentile(
        orientation_shift,
        parameters.orientation_quantile,
        valid_mask,
    )

    irat_gradient = _gradient_magnitude(feature_maps["I_rat"])
    irat_gradient_threshold = _robust_percentile(irat_gradient, parameters.broad_quantile, valid_mask)

    label_map = np.empty(shape, dtype=object)
    label_map[:] = "intermediate / erased memory"
    code_map = np.full(shape, fill_value=3, dtype=int)

    insulating_mask = (
        valid_mask
        & (feature_maps["I_rat"] <= low_irat)
        & (feature_maps["W_EF"] <= low_wef)
        & (feature_maps["E_LE"] <= le_far)
        & (lhb_shift <= lhb_close_threshold)
        & (feature_maps["Gamma_EDC"] < broad)
    )
    metallic_mask = (
        valid_mask
        & (feature_maps["I_rat"] >= high_irat)
        & (feature_maps["W_EF"] >= high_wef)
        & (feature_maps["E_LE"] >= le_close)
        & (feature_maps["Gamma_EDC"] < broad)
    )
    intermediate_range = (
        valid_mask
        & (feature_maps["I_rat"] > low_irat)
        & (feature_maps["I_rat"] < high_irat)
    )
    ambiguous_metallicity = valid_mask & ~insulating_mask & ~metallic_mask
    structural_mask = (
        ambiguous_metallicity
        & (orientation_shift >= orientation_shift_threshold)
        & (normalized_maps.get("Orient_shift_norm", np.zeros(shape, dtype=np.float32)) >= 0.55)
    )
    boundary_mask = (
        intermediate_range
        & (
            (feature_maps["Gamma_EDC"] >= broad)
            | (irat_gradient >= irat_gradient_threshold)
        )
    )

    preliminary_codes = np.full(shape, fill_value=3, dtype=int)
    preliminary_codes[insulating_mask] = 1
    preliminary_codes[metallic_mask] = 2
    preliminary_codes[low_signal] = 0
    if parameters.use_spatial_boundary:
        boundary_mask |= _metal_insulating_boundary_mask(preliminary_codes) & valid_mask

    label_map[insulating_mask] = STATE_CLASSIFICATION_LABELS[1]
    code_map[insulating_mask] = 1
    label_map[metallic_mask] = STATE_CLASSIFICATION_LABELS[2]
    code_map[metallic_mask] = 2
    label_map[boundary_mask] = STATE_CLASSIFICATION_LABELS[4]
    code_map[boundary_mask] = 4
    label_map[structural_mask] = STATE_CLASSIFICATION_LABELS[5]
    code_map[structural_mask] = 5
    label_map[low_signal] = STATE_CLASSIFICATION_LABELS[0]
    code_map[low_signal] = 0

    counts = {
        label: int(np.count_nonzero(code_map == code))
        for code, label in enumerate(STATE_CLASSIFICATION_LABELS)
    }
    threshold_values = {
        "low_Irat_threshold": float(low_irat),
        "high_Irat_threshold": float(high_irat),
        "low_WEF_threshold": float(low_wef),
        "high_WEF_threshold": float(high_wef),
        "far_LE_threshold": float(le_far),
        "close_LE_threshold": float(le_close),
        "broad_Gamma_threshold": float(broad),
        "LHB_reference_ev": float(lhb_reference),
        "LHB_close_shift_threshold": float(lhb_close_threshold),
        "low_signal_T_threshold": float(t_low),
        "min_W_LHB_threshold": float(lhb_min),
        "orientation_reference": float(orientation_median),
        "large_orientation_shift_threshold": float(orientation_shift_threshold),
        "Irat_gradient_threshold": float(irat_gradient_threshold),
    }
    normalized_maps = dict(normalized_maps)
    normalized_maps["Orient_shift_norm"] = _robust_normalize(orientation_shift, valid_mask)
    normalized_maps["Irat_gradient_norm"] = _robust_normalize(irat_gradient, valid_mask)

    return StateClassificationResult(
        state=state,
        parameters=parameters,
        feature_maps=feature_maps,
        normalized_maps=normalized_maps,
        threshold_values=threshold_values,
        label_map=label_map,
        code_map=code_map,
        valid_mask=valid_mask,
        orientation_feature_name=orientation_feature_name,
        counts=counts,
        notes=list(notes or []),
    )


def build_state_classifier_normalized_maps(
    feature_maps: dict[str, np.ndarray],
    fermi_level: float = 0.0,
) -> dict[str, np.ndarray]:
    shape = np.asarray(feature_maps["T"]).shape
    valid_mask = np.ones(shape, dtype=bool)
    for name in STATE_CLASSIFICATION_FEATURE_NAMES:
        valid_mask &= np.isfinite(np.asarray(feature_maps[name], dtype=np.float32))

    e_lhb = np.asarray(feature_maps["E_LHB"], dtype=np.float32)
    s_orient = np.asarray(feature_maps["S_orient"], dtype=np.float32)
    lhb_reference = _robust_percentile(e_lhb, 0.50, valid_mask, fallback=0.0)
    orientation_reference = _robust_percentile(s_orient, 0.50, valid_mask, fallback=0.0)
    lhb_shift = np.abs(e_lhb - lhb_reference).astype(np.float32)
    le_distance = np.abs(np.asarray(feature_maps["E_LE"], dtype=np.float32) - float(fermi_level)).astype(np.float32)
    orient_shift = np.abs(s_orient - orientation_reference).astype(np.float32)

    le_closeness = 1.0 - _robust_normalize(le_distance, valid_mask)
    le_closeness[~valid_mask] = np.nan

    return {
        "Irat_norm": _robust_normalize(feature_maps["I_rat"], valid_mask),
        "WEF_norm": _robust_normalize(feature_maps["W_EF"], valid_mask),
        "LHB_shift_norm": _robust_normalize(lhb_shift, valid_mask),
        "LE_closeness_norm": le_closeness.astype(np.float32),
        "Gamma_norm": _robust_normalize(feature_maps["Gamma_EDC"], valid_mask),
        "Orient_shift_norm": _robust_normalize(orient_shift, valid_mask),
    }


def state_classification_table_rows(result: StateClassificationResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x_size, y_size = result.shape
    for x_index in range(x_size):
        for y_index in range(y_size):
            row: dict[str, Any] = {
                "x": x_index,
                "y": y_index,
            }
            for name in STATE_CLASSIFICATION_FEATURE_NAMES:
                row[name] = float(result.feature_maps[name][x_index, y_index])
            row["state_code"] = int(result.code_map[x_index, y_index])
            row["state_label"] = str(result.label_map[x_index, y_index])
            rows.append(row)
    return rows


def export_state_classification(result: StateClassificationResult, output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    parameters_path = output_path / "clustering_parameters.json"
    thresholds_path = output_path / "clustering_thresholds.json"
    table_path = output_path / "clustering_feature_table.csv"
    labels_path = output_path / "clustering_state_labels.json"
    code_map_path = output_path / "clustering_state_code_map.npy"

    parameters_path.write_text(json.dumps(asdict(result.parameters), indent=2), encoding="utf-8")
    thresholds_path.write_text(json.dumps(result.threshold_values, indent=2), encoding="utf-8")
    labels_path.write_text(json.dumps(result.label_map.tolist(), indent=2), encoding="utf-8")
    np.save(code_map_path, result.code_map)
    write_rows_to_csv(table_path, state_classification_table_rows(result))

    feature_dir = output_path / "feature_maps"
    feature_dir.mkdir(parents=True, exist_ok=True)
    for feature_name, feature_map in result.feature_maps.items():
        np.save(feature_dir / f"{feature_name}.npy", feature_map)
    for feature_name, feature_map in result.normalized_maps.items():
        np.save(feature_dir / f"{feature_name}.npy", feature_map)

    summary_path = output_path / "clustering_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "file": result.file_path,
                "state_name": result.state_name,
                "shape": {"x": int(result.shape[0]), "y": int(result.shape[1])},
                "orientation_feature_name": result.orientation_feature_name,
                "counts": result.counts,
                "notes": result.notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "parameters": parameters_path,
        "thresholds": thresholds_path,
        "table": table_path,
        "labels": labels_path,
        "code_map": code_map_path,
        "summary": summary_path,
        "feature_maps": feature_dir,
    }


def run_tilt_map(
    file_path: str | Path,
    parameters: TiltMapParameters | None = None,
) -> TiltMapResult:
    if parameters is None:
        parameters = TiltMapParameters()
    parameters.validate()
    state = load_state(str(file_path), load=False)
    return compute_tilt_map_for_state(state, parameters)


def compute_tilt_map_for_state(
    state: LoadedState,
    parameters: TiltMapParameters,
) -> TiltMapResult:
    parameters.validate()
    da, energy_axis, phi_axis = sorted_required_dataarray(state.data_array)
    band_mask = _energy_window_mask(energy_axis, parameters.band_min_ev, parameters.band_max_ev)
    if not np.any(band_mask):
        raise ValueError(
            f"No eV samples were found inside the tilt band window "
            f"{parameters.band_min_ev:g} to {parameters.band_max_ev:g} eV."
        )

    band = da.isel(eV=np.flatnonzero(band_mask)).fillna(0)
    if int(np.count_nonzero(band_mask)) > 1:
        mdc_da = band.integrate(coord="eV")
    else:
        mdc_da = band.sum(dim="eV", skipna=True)
    mdc = np.asarray(mdc_da.values, dtype=np.float32)
    mdc[~np.isfinite(mdc)] = 0.0

    baseline = np.nanpercentile(mdc, 10, axis=2, keepdims=True) if mdc.shape[2] > 1 else 0.0
    weights = np.clip(mdc - baseline, a_min=0.0, a_max=None).astype(np.float32)
    if not np.any(weights > parameters.epsilon):
        weights = np.clip(mdc, a_min=0.0, a_max=None).astype(np.float32)

    band_weight = integrate_phi_profiles(weights, phi_axis).astype(np.float32)
    peak_index = np.argmax(weights, axis=2)
    phi_peak = phi_axis[np.clip(peak_index, 0, max(0, phi_axis.size - 1))].astype(np.float32)
    phi_center = weighted_phi_center(weights, phi_axis, parameters.epsilon).astype(np.float32)
    phi_width = weighted_phi_width(weights, phi_axis, phi_center, parameters.epsilon).astype(np.float32)

    percentile_signal_threshold = finite_percentile(band_weight, parameters.low_signal_percentile)
    high_signal_reference = finite_percentile(band_weight, 98.0)
    signal_floor_threshold = (
        float(high_signal_reference) * float(parameters.signal_floor_fraction)
        if np.isfinite(high_signal_reference)
        else float("nan")
    )
    low_signal_threshold = np.nanmax([percentile_signal_threshold, signal_floor_threshold])
    if not np.isfinite(low_signal_threshold):
        low_signal_threshold = percentile_signal_threshold
    valid_mask = np.isfinite(phi_center) & np.isfinite(band_weight) & (band_weight > low_signal_threshold)
    valid_mask = clean_connected_components(valid_mask, max(1, parameters.min_group_size))
    tilt = (phi_center - float(parameters.phi_reference)).astype(np.float32)
    peak_tilt = (phi_peak - float(parameters.phi_reference)).astype(np.float32)
    tilt[~valid_mask] = np.nan
    peak_tilt[~valid_mask] = np.nan
    phi_width[~valid_mask] = np.nan

    filled_tilt = fill_with_median(tilt)
    if parameters.spatial_smooth_sigma > 0:
        smooth_tilt = ndimage.gaussian_filter(filled_tilt, sigma=parameters.spatial_smooth_sigma, mode="nearest").astype(np.float32)
    else:
        smooth_tilt = filled_tilt.astype(np.float32)
    smooth_tilt[~valid_mask] = np.nan
    tilt_gradient = _gradient_magnitude(smooth_tilt)
    tilt_gradient[~valid_mask] = np.nan
    local_std = local_nan_std(smooth_tilt, valid_mask, parameters.local_window)

    abs_tilt = np.abs(tilt)
    tilt_threshold = finite_percentile(abs_tilt[valid_mask], parameters.defect_tilt_percentile)
    gradient_threshold = finite_percentile(tilt_gradient[valid_mask], parameters.defect_gradient_percentile)
    local_std_threshold = finite_percentile(local_std[valid_mask], parameters.defect_gradient_percentile)
    width_threshold = finite_percentile(phi_width[valid_mask], parameters.defect_gradient_percentile)

    high_tilt = valid_mask & (abs_tilt >= tilt_threshold)
    sharp_boundary = valid_mask & (tilt_gradient >= gradient_threshold)
    rough_or_dislocation = valid_mask & ((local_std >= local_std_threshold) | (phi_width >= width_threshold))

    defect_type_map = np.zeros(tilt.shape, dtype=np.int8)
    defect_type_map[high_tilt] = 1
    defect_type_map[sharp_boundary] = 2
    defect_type_map[rough_or_dislocation] = 3
    defect_mask = defect_type_map > 0
    defect_score = robust_normalize_map(abs_tilt) + robust_normalize_map(tilt_gradient) + robust_normalize_map(local_std)
    defect_score = np.asarray(defect_score / 3.0, dtype=np.float32)
    defect_score[~valid_mask] = np.nan

    group_label_map, group_rows = build_tilt_group_map(
        smooth_tilt,
        valid_mask,
        defect_mask,
        tilt_gradient,
        phi_width,
        parameters.group_count,
        parameters.min_group_size,
    )
    group_rows.extend(build_tilt_defect_rows(defect_type_map, tilt, tilt_gradient, local_std, phi_width))
    group_mean_tilt = build_group_mean_tilt_map(group_label_map, group_rows)

    thresholds = {
        "low_signal_band_weight": float(low_signal_threshold),
        "percentile_low_signal_band_weight": float(percentile_signal_threshold),
        "signal_floor_band_weight": float(signal_floor_threshold),
        "high_abs_tilt": float(tilt_threshold),
        "high_tilt_gradient": float(gradient_threshold),
        "high_local_tilt_std": float(local_std_threshold),
        "high_phi_width": float(width_threshold),
    }
    notes = [
        "Tilt is computed as the intensity-weighted phi center of the selected ARPES band window minus the phi reference.",
        "Sharp boundary defects are high spatial gradients in the local tilt map; rough/dislocation defects are high local tilt variance or broad phi profiles.",
    ]
    return TiltMapResult(
        state=state,
        parameters=parameters,
        tilt_map=tilt,
        peak_tilt_map=peak_tilt,
        band_weight_map=band_weight,
        phi_width_map=phi_width,
        tilt_gradient_map=tilt_gradient,
        local_tilt_std_map=local_std,
        defect_score_map=defect_score,
        group_mean_tilt_map=group_mean_tilt,
        defect_mask=defect_mask,
        defect_type_map=defect_type_map,
        group_label_map=group_label_map,
        valid_mask=valid_mask,
        thresholds=thresholds,
        group_rows=group_rows,
        notes=notes,
    )


def integrate_phi_profiles(values: np.ndarray, phi_axis: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    phi = np.asarray(phi_axis, dtype=np.float32)
    if phi.size > 1:
        return np.trapezoid(arr, x=phi, axis=2).astype(np.float32)
    return np.sum(arr, axis=2).astype(np.float32)


def weighted_phi_center(values: np.ndarray, phi_axis: np.ndarray, epsilon: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    phi = np.asarray(phi_axis, dtype=np.float32)
    denominator = integrate_phi_profiles(arr, phi)
    if phi.size > 1:
        numerator = np.trapezoid(arr * phi[None, None, :], x=phi, axis=2)
    else:
        numerator = np.sum(arr * phi[None, None, :], axis=2)
    out = np.divide(
        numerator,
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=np.float32),
        where=np.abs(denominator) > epsilon,
    )
    return out.astype(np.float32)


def weighted_phi_width(values: np.ndarray, phi_axis: np.ndarray, center: np.ndarray, epsilon: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    phi = np.asarray(phi_axis, dtype=np.float32)
    denominator = integrate_phi_profiles(arr, phi)
    delta = phi[None, None, :] - np.asarray(center, dtype=np.float32)[:, :, None]
    if phi.size > 1:
        numerator = np.trapezoid(arr * delta * delta, x=phi, axis=2)
    else:
        numerator = np.sum(arr * delta * delta, axis=2)
    variance = np.divide(
        numerator,
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=np.float32),
        where=np.abs(denominator) > epsilon,
    )
    return np.sqrt(np.clip(variance, 0.0, None)).astype(np.float32)


def fill_with_median(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    fill = float(np.nanmedian(finite)) if finite.size else 0.0
    return finite_fill(arr, fill).astype(np.float32)


def local_nan_std(values: np.ndarray, valid_mask: np.ndarray, window: int) -> np.ndarray:
    valid = np.asarray(valid_mask, dtype=np.float32)
    size = max(1, int(window))
    filled = finite_fill(values, 0.0).astype(np.float32)
    count = ndimage.uniform_filter(valid, size=size, mode="nearest")
    mean = ndimage.uniform_filter(filled * valid, size=size, mode="nearest") / np.maximum(count, 1e-6)
    mean_sq = ndimage.uniform_filter(filled * filled * valid, size=size, mode="nearest") / np.maximum(count, 1e-6)
    out = np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None)).astype(np.float32)
    out[count <= 1e-6] = np.nan
    return out


def build_tilt_group_map(
    tilt_map: np.ndarray,
    valid_mask: np.ndarray,
    defect_mask: np.ndarray,
    gradient_map: np.ndarray,
    phi_width_map: np.ndarray,
    group_count: int,
    min_group_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    group_base = np.asarray(valid_mask, dtype=bool) & ~np.asarray(defect_mask, dtype=bool) & np.isfinite(tilt_map)
    group_label_map = np.zeros(np.asarray(tilt_map).shape, dtype=np.int16)
    values = np.asarray(tilt_map, dtype=np.float32)[group_base]
    if values.size == 0:
        return group_label_map, []

    quantiles = np.linspace(0.0, 100.0, max(2, int(group_count)) + 1)
    edges = np.unique(np.nanpercentile(values, quantiles))
    if edges.size < 2:
        edges = np.asarray([float(np.nanmin(values)) - 1e-6, float(np.nanmax(values)) + 1e-6], dtype=np.float32)

    label_index = 1
    rows: list[dict[str, Any]] = []
    bins = np.digitize(np.asarray(tilt_map, dtype=np.float32), edges[1:-1], right=False)
    for bin_index in range(edges.size - 1):
        mask = group_base & (bins == bin_index)
        if not np.any(mask):
            continue
        cleaned = clean_connected_components(mask, min_group_size)
        labels, count = ndimage.label(cleaned)
        for component_index in range(1, count + 1):
            component = labels == component_index
            pixel_count = int(np.count_nonzero(component))
            if pixel_count < max(1, min_group_size):
                continue
            group_label_map[component] = label_index
            xs, ys = np.where(component)
            mean_tilt = float(np.nanmean(np.asarray(tilt_map)[component]))
            label = "positive-tilted terrace" if mean_tilt > 0 else "negative-tilted terrace" if mean_tilt < 0 else "near-flat terrace"
            rows.append(
                {
                    "group_id": int(label_index),
                    "group_type": label,
                    "tilt_bin": int(bin_index),
                    "pixel_count": pixel_count,
                    "mean_tilt_phi": mean_tilt,
                    "mean_abs_tilt_phi": float(np.nanmean(np.abs(np.asarray(tilt_map)[component]))),
                    "mean_tilt_gradient": float(np.nanmean(np.asarray(gradient_map)[component])),
                    "mean_phi_width": float(np.nanmean(np.asarray(phi_width_map)[component])),
                    "x_min": int(np.nanmin(xs)),
                    "x_max": int(np.nanmax(xs)),
                    "y_min": int(np.nanmin(ys)),
                    "y_max": int(np.nanmax(ys)),
                }
            )
            label_index += 1
    return group_label_map, rows


def build_tilt_defect_rows(
    defect_type_map: np.ndarray,
    tilt_map: np.ndarray,
    gradient_map: np.ndarray,
    local_std_map: np.ndarray,
    phi_width_map: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for code, label in enumerate(TILT_DEFECT_LABELS):
        if code == 0:
            continue
        labels, count = ndimage.label(np.asarray(defect_type_map) == code)
        for component_index in range(1, count + 1):
            component = labels == component_index
            pixel_count = int(np.count_nonzero(component))
            if pixel_count == 0:
                continue
            xs, ys = np.where(component)
            rows.append(
                {
                    "group_id": int(-1000 * code - component_index),
                    "group_type": label,
                    "tilt_bin": -1,
                    "pixel_count": pixel_count,
                    "mean_tilt_phi": float(np.nanmean(np.asarray(tilt_map)[component])),
                    "mean_abs_tilt_phi": float(np.nanmean(np.abs(np.asarray(tilt_map)[component]))),
                    "mean_tilt_gradient": float(np.nanmean(np.asarray(gradient_map)[component])),
                    "mean_local_tilt_std": float(np.nanmean(np.asarray(local_std_map)[component])),
                    "mean_phi_width": float(np.nanmean(np.asarray(phi_width_map)[component])),
                    "x_min": int(np.nanmin(xs)),
                    "x_max": int(np.nanmax(xs)),
                    "y_min": int(np.nanmin(ys)),
                    "y_max": int(np.nanmax(ys)),
                }
            )
    return rows


def build_group_mean_tilt_map(group_label_map: np.ndarray, group_rows: list[dict[str, Any]]) -> np.ndarray:
    label_map = np.asarray(group_label_map)
    out = np.full(label_map.shape, np.nan, dtype=np.float32)
    for row in group_rows:
        group_id = int(row.get("group_id", 0))
        if group_id <= 0:
            continue
        mean_tilt = row.get("mean_tilt_phi")
        if mean_tilt is None:
            continue
        out[label_map == group_id] = float(mean_tilt)
    return out


def tilt_map_table_rows(result: TiltMapResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x_size, y_size = result.shape
    for x_index in range(x_size):
        for y_index in range(y_size):
            defect_code = int(result.defect_type_map[x_index, y_index])
            rows.append(
                {
                    "x": x_index,
                    "y": y_index,
                    "valid": bool(result.valid_mask[x_index, y_index]),
                    "tilt_phi": float(result.tilt_map[x_index, y_index]),
                    "peak_tilt_phi": float(result.peak_tilt_map[x_index, y_index]),
                    "band_weight": float(result.band_weight_map[x_index, y_index]),
                    "phi_width": float(result.phi_width_map[x_index, y_index]),
                    "tilt_gradient": float(result.tilt_gradient_map[x_index, y_index]),
                    "local_tilt_std": float(result.local_tilt_std_map[x_index, y_index]),
                    "defect_score": float(result.defect_score_map[x_index, y_index]),
                    "defect_code": defect_code,
                    "defect_label": TILT_DEFECT_LABELS[defect_code],
                    "tilt_group_id": int(result.group_label_map[x_index, y_index]),
                    "region_mean_tilt_phi": float(result.group_mean_tilt_map[x_index, y_index]),
                }
            )
    return rows


def export_tilt_map(result: TiltMapResult, output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    maps_dir = output_path / "tilt_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    map_values = {
        "tilt_phi": result.tilt_map,
        "peak_tilt_phi": result.peak_tilt_map,
        "band_weight": result.band_weight_map,
        "phi_width": result.phi_width_map,
        "tilt_gradient": result.tilt_gradient_map,
        "local_tilt_std": result.local_tilt_std_map,
        "defect_score": result.defect_score_map,
        "region_mean_tilt_phi": result.group_mean_tilt_map,
        "defect_mask": result.defect_mask.astype(np.int8),
        "defect_type_map": result.defect_type_map,
        "tilt_group_map": result.group_label_map,
        "valid_mask": result.valid_mask.astype(np.int8),
    }
    for name, values in map_values.items():
        np.save(maps_dir / f"{name}.npy", values)

    pixel_table = output_path / "tilt_map_pixels.csv"
    group_table = output_path / "tilt_map_groups.csv"
    parameters_path = output_path / "tilt_map_parameters.json"
    summary_path = output_path / "tilt_map_summary.json"
    write_rows_to_csv(pixel_table, tilt_map_table_rows(result))
    write_rows_to_csv(group_table, result.group_rows)
    parameters_path.write_text(json.dumps(asdict(result.parameters), indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "file": result.file_path,
                "state_name": result.state_name,
                "shape": {"x": int(result.shape[0]), "y": int(result.shape[1])},
                "thresholds": result.thresholds,
                "valid_pixels": int(np.count_nonzero(result.valid_mask)),
                "defect_pixels": int(np.count_nonzero(result.defect_mask)),
                "groups": len(result.group_rows),
                "notes": result.notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "maps": maps_dir,
        "pixel_table": pixel_table,
        "group_table": group_table,
        "parameters": parameters_path,
        "summary": summary_path,
    }


def run_switching_map(
    file_paths: list[str] | tuple[str, ...],
    parameters: SwitchingMapParameters | None = None,
) -> SwitchingMapResult:
    if parameters is None:
        parameters = SwitchingMapParameters()
    parameters.validate()

    paths = [str(Path(path).expanduser().resolve()) for path in file_paths]
    if len(paths) < 2:
        raise ValueError("Switching Map needs at least two chronological ARPES data files.")

    loaded_states, alignment_notes = align_loaded_states_for_comparison([load_state(path) for path in paths])
    total_maps: list[np.ndarray] = []
    w_ef_maps: list[np.ndarray] = []
    w_lhb_maps: list[np.ndarray] = []
    i_rat_maps: list[np.ndarray] = []

    notes: list[str] = list(alignment_notes)
    for state in loaded_states:
        feature_maps = compute_switching_core_feature_maps(state.data_array, parameters)
        total_maps.append(feature_maps["T"])
        w_ef_maps.append(feature_maps["W_EF"])
        w_lhb_maps.append(feature_maps["W_LHB"])
        i_rat_maps.append(feature_maps["I_rat"])

    return build_switching_map_result(
        loaded_states=loaded_states,
        parameters=parameters,
        total_maps=total_maps,
        w_ef_maps=w_ef_maps,
        w_lhb_maps=w_lhb_maps,
        i_rat_maps=i_rat_maps,
        notes=notes,
    )


def compute_switching_core_feature_maps(
    da: xr.DataArray,
    parameters: SwitchingMapParameters,
) -> dict[str, np.ndarray]:
    require_dims(da)
    parameters.validate()

    data = np.asarray(da.values, dtype=np.float32)
    energy_axis = np.asarray(da.coords["eV"].values, dtype=np.float32)
    phi_axis = np.asarray(da.coords["phi"].values, dtype=np.float32)

    energy_order = np.argsort(energy_axis)
    phi_order = np.argsort(phi_axis)
    energy_axis = energy_axis[energy_order]
    phi_axis = phi_axis[phi_order]
    data = data[:, :, energy_order, :]
    data = data[:, :, :, phi_order]

    edc = _integrate_along_axis(data, phi_axis, axis=3).astype(np.float32)
    smoothed_edc = (
        ndimage.gaussian_filter1d(edc, sigma=parameters.smooth_sigma, axis=2, mode="nearest")
        if parameters.smooth_sigma > 0
        else edc
    ).astype(np.float32)
    total_intensity = _integrate_along_axis(edc, energy_axis, axis=2).astype(np.float32)

    ef_mask = _energy_window_mask(
        energy_axis,
        parameters.fermi_level_ev + parameters.ef_min_ev,
        parameters.fermi_level_ev + parameters.ef_max_ev,
    )
    lhb_mask = get_energy_mask(
        energy_axis,
        center=parameters.lhb_center_ev,
        halfwidth=parameters.lhb_halfwidth_ev,
    )
    if not ef_mask.any():
        raise ValueError("No energy samples were found inside the near-EF switching window.")
    if not lhb_mask.any():
        raise ValueError("No energy samples were found inside the LHB/p1 switching window.")

    w_ef = _integrate_window(smoothed_edc, energy_axis, ef_mask).astype(np.float32)
    w_lhb = _integrate_window(smoothed_edc, energy_axis, lhb_mask).astype(np.float32)
    i_rat = safe_divide(w_ef, w_lhb, eps=parameters.epsilon).astype(np.float32)
    return {
        "T": total_intensity,
        "W_EF": w_ef,
        "W_LHB": w_lhb,
        "I_rat": i_rat,
    }


def build_switching_map_result(
    loaded_states: list[LoadedState],
    parameters: SwitchingMapParameters,
    total_maps: list[np.ndarray],
    w_ef_maps: list[np.ndarray],
    w_lhb_maps: list[np.ndarray],
    i_rat_maps: list[np.ndarray],
    notes: list[str] | None = None,
) -> SwitchingMapResult:
    if len(loaded_states) < 2:
        raise ValueError("Switching Map needs at least two chronological states.")

    i_rat_stack = np.stack([np.asarray(values, dtype=np.float32) for values in i_rat_maps], axis=0)
    w_ef_stack = np.stack([np.asarray(values, dtype=np.float32) for values in w_ef_maps], axis=0)
    w_lhb_stack = np.stack([np.asarray(values, dtype=np.float32) for values in w_lhb_maps], axis=0)
    total_stack = np.stack([np.asarray(values, dtype=np.float32) for values in total_maps], axis=0)

    delta_irat_stack = np.diff(i_rat_stack, axis=0).astype(np.float32)
    initial_delta_stack = (i_rat_stack - i_rat_stack[0:1]).astype(np.float32)
    total_change = np.sum(np.abs(delta_irat_stack), axis=0, dtype=np.float64).astype(np.float32)
    max_change = np.max(np.abs(delta_irat_stack), axis=0).astype(np.float32)
    net_change = (i_rat_stack[-1] - i_rat_stack[0]).astype(np.float32)

    finite_core = (
        np.all(np.isfinite(i_rat_stack), axis=0)
        & np.all(np.isfinite(w_ef_stack), axis=0)
        & np.all(np.isfinite(w_lhb_stack), axis=0)
        & np.all(np.isfinite(total_stack), axis=0)
    )
    total_min = np.min(np.where(np.isfinite(total_stack), total_stack, np.inf), axis=0).astype(np.float32)
    lhb_min_per_pixel = np.min(np.where(np.isfinite(w_lhb_stack), w_lhb_stack, np.inf), axis=0).astype(np.float32)
    low_signal = _robust_percentile(total_min, parameters.low_signal_quantile, finite_core)
    min_lhb = _robust_percentile(lhb_min_per_pixel, parameters.lhb_min_quantile, finite_core)
    valid_mask = finite_core & (total_min > low_signal) & (lhb_min_per_pixel > min_lhb)

    switching_coefficient = _minmax_normalize(total_change, valid_mask)
    low_switch = _robust_percentile(
        switching_coefficient,
        parameters.low_switch_quantile,
        valid_mask,
        fallback=0.0,
    )
    high_switch = _robust_percentile(
        switching_coefficient,
        parameters.high_switch_quantile,
        valid_mask,
        fallback=1.0,
    )
    small_net = _robust_percentile(
        np.abs(net_change),
        parameters.small_net_quantile,
        valid_mask,
        fallback=0.0,
    )

    label_map = np.empty(total_change.shape, dtype=object)
    label_map[:] = SWITCHING_LABELS[4]
    code_map = np.full(total_change.shape, fill_value=4, dtype=int)

    stable_mask = valid_mask & (switching_coefficient <= low_switch)
    high_switch_mask = valid_mask & (switching_coefficient >= high_switch)
    reversible_mask = high_switch_mask & (
        (np.abs(net_change) <= small_net)
        | (np.abs(net_change) <= 0.25 * np.maximum(total_change, parameters.epsilon))
    )
    written_mask = high_switch_mask & ~reversible_mask & (net_change > small_net)
    erased_mask = high_switch_mask & ~reversible_mask & (net_change < -small_net)

    label_map[stable_mask] = SWITCHING_LABELS[0]
    code_map[stable_mask] = 0
    label_map[written_mask] = SWITCHING_LABELS[1]
    code_map[written_mask] = 1
    label_map[erased_mask] = SWITCHING_LABELS[2]
    code_map[erased_mask] = 2
    label_map[reversible_mask] = SWITCHING_LABELS[3]
    code_map[reversible_mask] = 3

    counts = {label: int(np.count_nonzero(code_map == code)) for code, label in enumerate(SWITCHING_LABELS)}
    threshold_values = {
        "low_switch_threshold": float(low_switch),
        "high_switch_threshold": float(high_switch),
        "small_net_change_threshold": float(small_net),
        "low_signal_T_threshold": float(low_signal),
        "min_W_LHB_threshold": float(min_lhb),
    }

    return SwitchingMapResult(
        loaded_states=loaded_states,
        parameters=parameters,
        total_maps=[np.asarray(values, dtype=np.float32) for values in total_maps],
        w_ef_maps=[np.asarray(values, dtype=np.float32) for values in w_ef_maps],
        w_lhb_maps=[np.asarray(values, dtype=np.float32) for values in w_lhb_maps],
        i_rat_maps=[np.asarray(values, dtype=np.float32) for values in i_rat_maps],
        delta_irat_maps=[np.asarray(values, dtype=np.float32) for values in delta_irat_stack],
        initial_delta_irat_maps=[np.asarray(values, dtype=np.float32) for values in initial_delta_stack],
        total_change_map=total_change,
        max_change_map=max_change,
        net_change_map=net_change,
        switching_coefficient_map=switching_coefficient,
        label_map=label_map,
        code_map=code_map,
        valid_mask=valid_mask,
        threshold_values=threshold_values,
        counts=counts,
        notes=list(notes or []),
    )


def switching_map_table_rows(result: SwitchingMapResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x_size, y_size = result.shape
    for x_index in range(x_size):
        for y_index in range(y_size):
            row: dict[str, Any] = {
                "x": x_index,
                "y": y_index,
                "switching_coefficient": float(result.switching_coefficient_map[x_index, y_index]),
                "total_change": float(result.total_change_map[x_index, y_index]),
                "max_change": float(result.max_change_map[x_index, y_index]),
                "net_change_from_initial": float(result.net_change_map[x_index, y_index]),
                "valid": bool(result.valid_mask[x_index, y_index]),
                "state_code": int(result.code_map[x_index, y_index]),
                "state_label": str(result.label_map[x_index, y_index]),
            }
            for state_index, state in enumerate(result.loaded_states):
                prefix = f"file_{state_index}"
                row[f"{prefix}_name"] = state.name
                row[f"{prefix}_I_rat"] = float(result.i_rat_maps[state_index][x_index, y_index])
                row[f"{prefix}_W_EF"] = float(result.w_ef_maps[state_index][x_index, y_index])
                row[f"{prefix}_W_LHB"] = float(result.w_lhb_maps[state_index][x_index, y_index])
                row[f"{prefix}_Delta_Irat_from_initial"] = float(
                    result.initial_delta_irat_maps[state_index][x_index, y_index]
                )
            for transition_index, delta_map in enumerate(result.delta_irat_maps):
                row[f"Delta_Irat_{transition_index}_to_{transition_index + 1}"] = float(
                    delta_map[x_index, y_index]
                )
            rows.append(row)
    return rows


def export_switching_map(result: SwitchingMapResult, output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    parameters_path = output_path / "switching_parameters.json"
    thresholds_path = output_path / "switching_thresholds.json"
    table_path = output_path / "switching_feature_table.csv"
    labels_path = output_path / "switching_state_labels.json"
    code_map_path = output_path / "switching_state_code_map.npy"
    coefficient_path = output_path / "switching_coefficient_map.npy"
    total_change_path = output_path / "switching_total_change_map.npy"
    max_change_path = output_path / "switching_max_change_map.npy"
    net_change_path = output_path / "switching_net_change_map.npy"

    parameters_path.write_text(json.dumps(asdict(result.parameters), indent=2), encoding="utf-8")
    thresholds_path.write_text(json.dumps(result.threshold_values, indent=2), encoding="utf-8")
    labels_path.write_text(json.dumps(result.label_map.tolist(), indent=2), encoding="utf-8")
    write_rows_to_csv(table_path, switching_map_table_rows(result))
    np.save(code_map_path, result.code_map)
    np.save(coefficient_path, result.switching_coefficient_map)
    np.save(total_change_path, result.total_change_map)
    np.save(max_change_path, result.max_change_map)
    np.save(net_change_path, result.net_change_map)

    maps_dir = output_path / "switching_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    np.save(maps_dir / "I_rat_maps.npy", np.stack(result.i_rat_maps, axis=0))
    np.save(maps_dir / "W_EF_maps.npy", np.stack(result.w_ef_maps, axis=0))
    np.save(maps_dir / "W_LHB_maps.npy", np.stack(result.w_lhb_maps, axis=0))
    np.save(maps_dir / "Delta_Irat_maps.npy", np.stack(result.delta_irat_maps, axis=0))
    np.save(maps_dir / "Delta_Irat_from_initial_maps.npy", np.stack(result.initial_delta_irat_maps, axis=0))
    for index, state in enumerate(result.loaded_states):
        state_dir = maps_dir / f"{index:02d}_{sanitize_filename(state.name)}"
        state_dir.mkdir(parents=True, exist_ok=True)
        np.save(state_dir / "T.npy", result.total_maps[index])
        np.save(state_dir / "W_EF.npy", result.w_ef_maps[index])
        np.save(state_dir / "W_LHB.npy", result.w_lhb_maps[index])
        np.save(state_dir / "I_rat.npy", result.i_rat_maps[index])
        np.save(state_dir / "Delta_Irat_from_initial.npy", result.initial_delta_irat_maps[index])

    summary_path = output_path / "switching_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "files": result.file_paths,
                "shape": {"x": int(result.shape[0]), "y": int(result.shape[1])},
                "counts": result.counts,
                "thresholds": result.threshold_values,
                "notes": result.notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "parameters": parameters_path,
        "thresholds": thresholds_path,
        "table": table_path,
        "labels": labels_path,
        "code_map": code_map_path,
        "coefficient_map": coefficient_path,
        "total_change_map": total_change_path,
        "max_change_map": max_change_path,
        "net_change_map": net_change_path,
        "summary": summary_path,
        "maps": maps_dir,
    }


def run_state_prediction(
    file_paths: list[str] | tuple[str, ...],
    parameters: StatePredictionParameters | None = None,
) -> StatePredictionResult:
    if parameters is None:
        parameters = StatePredictionParameters()
    parameters.validate()

    switching_parameters = SwitchingMapParameters(
        fermi_level_ev=parameters.fermi_level_ev,
        ef_min_ev=parameters.ef_min_ev,
        ef_max_ev=parameters.ef_max_ev,
        lhb_center_ev=parameters.lhb_center_ev,
        lhb_halfwidth_ev=parameters.lhb_halfwidth_ev,
        smooth_sigma=parameters.smooth_sigma,
        low_switch_quantile=parameters.stable_quantile,
        high_switch_quantile=parameters.switch_quantile,
        small_net_quantile=0.35,
        low_signal_quantile=parameters.low_signal_quantile,
        lhb_min_quantile=parameters.lhb_min_quantile,
        epsilon=parameters.epsilon,
    )
    switching_result = run_switching_map(file_paths, switching_parameters)
    return build_state_prediction_result(switching_result, parameters)


def build_state_prediction_result(
    switching_result: SwitchingMapResult,
    parameters: StatePredictionParameters,
) -> StatePredictionResult:
    parameters.validate()
    classifier_parameters = StateClassifierParameters(
        fermi_level_ev=parameters.fermi_level_ev,
        ef_min_ev=parameters.ef_min_ev,
        ef_max_ev=parameters.ef_max_ev,
        lhb_center_ev=parameters.lhb_center_ev,
        lhb_halfwidth_ev=parameters.lhb_halfwidth_ev,
        leading_edge_min_ev=parameters.leading_edge_min_ev,
        leading_edge_max_ev=parameters.leading_edge_max_ev,
        p3_center_ev=parameters.p3_center_ev,
        p3_halfwidth_ev=parameters.p3_halfwidth_ev,
        smooth_sigma=parameters.smooth_sigma,
        low_signal_quantile=parameters.low_signal_quantile,
        lhb_min_quantile=parameters.lhb_min_quantile,
        epsilon=parameters.epsilon,
    )
    feature_maps, _normalized_maps, orientation_feature_name, feature_notes = compute_state_classifier_feature_maps(
        switching_result.loaded_states[0].data_array,
        classifier_parameters,
    )
    feature_maps = {name: np.asarray(values, dtype=np.float32) for name, values in feature_maps.items()}

    finite_features = np.ones(switching_result.shape, dtype=bool)
    for name in STATE_CLASSIFICATION_FEATURE_NAMES:
        finite_features &= np.isfinite(feature_maps[name])
    valid_mask = switching_result.valid_mask & finite_features

    coefficient = np.asarray(switching_result.switching_coefficient_map, dtype=np.float32)
    net_change = np.asarray(switching_result.net_change_map, dtype=np.float32)
    low_switch = _robust_percentile(coefficient, parameters.stable_quantile, valid_mask, fallback=0.0)
    high_switch = _robust_percentile(coefficient, parameters.switch_quantile, valid_mask, fallback=1.0)
    if parameters.net_change_tau is None:
        net_tau = _robust_percentile(np.abs(net_change), 0.35, valid_mask, fallback=0.0)
    else:
        net_tau = float(parameters.net_change_tau)

    label_map = np.empty(switching_result.shape, dtype=object)
    label_map[:] = SWITCHING_LABELS[4]
    code_map = np.full(switching_result.shape, fill_value=4, dtype=int)

    stable_mask = valid_mask & (coefficient <= low_switch)
    high_switch_mask = valid_mask & (coefficient >= high_switch)
    reversible_mask = high_switch_mask & (np.abs(net_change) <= net_tau)
    written_mask = high_switch_mask & ~reversible_mask & (net_change > net_tau)
    erased_mask = high_switch_mask & ~reversible_mask & (net_change < -net_tau)

    label_map[stable_mask] = SWITCHING_LABELS[0]
    code_map[stable_mask] = 0
    label_map[written_mask] = SWITCHING_LABELS[1]
    code_map[written_mask] = 1
    label_map[erased_mask] = SWITCHING_LABELS[2]
    code_map[erased_mask] = 2
    label_map[reversible_mask] = SWITCHING_LABELS[3]
    code_map[reversible_mask] = 3

    distance_maps, distance_thresholds = compute_state_prediction_distance_maps(
        feature_maps,
        valid_mask,
        parameters,
    )
    energy_axis, edc_cube = initial_edc_cube(switching_result.loaded_states[0])
    average_initial_edcs = average_edcs_by_outcome(edc_cube, code_map, valid_mask)
    correlation_values = compute_state_prediction_correlations(
        feature_maps,
        distance_maps,
        coefficient,
        valid_mask,
    )
    interpretation = interpret_state_prediction(correlation_values)

    counts = {label: int(np.count_nonzero(code_map == code)) for code, label in enumerate(SWITCHING_LABELS)}
    threshold_values = {
        "stable_switching_threshold": float(low_switch),
        "high_switching_threshold": float(high_switch),
        "net_change_tau": float(net_tau),
        **distance_thresholds,
    }
    notes = list(switching_result.notes) + list(feature_notes)
    if energy_axis.shape != switching_result.e_axis.shape or not np.allclose(energy_axis, switching_result.e_axis):
        notes.append("Initial EDC axis was sorted for averaging; plotted curves use the sorted initial energy axis.")

    return StatePredictionResult(
        switching_result=switching_result,
        parameters=parameters,
        initial_feature_maps=feature_maps,
        distance_maps=distance_maps,
        average_initial_edcs=average_initial_edcs,
        correlation_values=correlation_values,
        label_map=label_map,
        code_map=code_map,
        valid_mask=valid_mask,
        threshold_values=threshold_values,
        counts=counts,
        orientation_feature_name=orientation_feature_name,
        interpretation=interpretation,
        notes=notes,
    )


def compute_state_prediction_distance_maps(
    feature_maps: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    parameters: StatePredictionParameters,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    total = np.asarray(feature_maps["T"], dtype=np.float32)
    irat = np.asarray(feature_maps["I_rat"], dtype=np.float32)
    orient = np.asarray(feature_maps["S_orient"], dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    low_signal = _robust_percentile(total, parameters.low_signal_quantile, valid_mask, fallback=0.0)
    sample_mask = valid_mask & np.isfinite(total) & (total > low_signal)
    distance_to_edge = ndimage.distance_transform_edt(sample_mask).astype(np.float32)
    distance_to_edge[~valid_mask] = np.nan

    low_irat = _robust_percentile(irat, parameters.phase_low_quantile, valid_mask, fallback=0.0)
    high_irat = _robust_percentile(irat, parameters.phase_high_quantile, valid_mask, fallback=1.0)
    insulating = valid_mask & (irat <= low_irat)
    metallic = valid_mask & (irat >= high_irat)
    phase_boundary = _binary_neighbor_touch_mask(insulating, metallic) & valid_mask
    distance_to_phase_boundary = _distance_to_binary_mask(phase_boundary, valid_mask)

    total_grad = _robust_normalize(_gradient_magnitude(total), valid_mask)
    orient_grad = _robust_normalize(_gradient_magnitude(orient), valid_mask)
    structural_score = np.maximum(
        np.nan_to_num(total_grad, nan=0.0),
        np.nan_to_num(orient_grad, nan=0.0),
    ).astype(np.float32)
    structural_score[~valid_mask] = np.nan
    structural_threshold = _robust_percentile(
        structural_score,
        parameters.structural_gradient_quantile,
        valid_mask,
        fallback=1.0,
    )
    structural_boundary = valid_mask & (structural_score >= structural_threshold)
    distance_to_structural_boundary = _distance_to_binary_mask(structural_boundary, valid_mask)

    return (
        {
            "distance_to_edge": distance_to_edge,
            "distance_to_phase_boundary": distance_to_phase_boundary,
            "distance_to_structural_boundary": distance_to_structural_boundary,
            "structural_gradient_score": structural_score,
        },
        {
            "initial_low_signal_T_threshold": float(low_signal),
            "initial_low_Irat_phase_threshold": float(low_irat),
            "initial_high_Irat_phase_threshold": float(high_irat),
            "structural_gradient_threshold": float(structural_threshold),
        },
    )


def initial_edc_cube(state: LoadedState) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(state.data_array.values, dtype=np.float32)
    energy_axis = np.asarray(state.data_array.coords["eV"].values, dtype=np.float32)
    phi_axis = np.asarray(state.data_array.coords["phi"].values, dtype=np.float32)
    energy_order = np.argsort(energy_axis)
    phi_order = np.argsort(phi_axis)
    energy_axis = energy_axis[energy_order]
    phi_axis = phi_axis[phi_order]
    data = data[:, :, energy_order, :]
    data = data[:, :, :, phi_order]
    edc_cube = _integrate_along_axis(data, phi_axis, axis=3).astype(np.float32)
    return energy_axis, edc_cube


def average_edcs_by_outcome(
    edc_cube: np.ndarray,
    code_map: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    averages: dict[str, np.ndarray] = {}
    for code, label in enumerate(SWITCHING_LABELS):
        mask = np.asarray(valid_mask, dtype=bool) & (np.asarray(code_map, dtype=int) == code)
        if np.any(mask):
            averages[label] = np.nanmean(edc_cube[mask], axis=0).astype(np.float32)
        else:
            averages[label] = np.full(edc_cube.shape[2], fill_value=np.nan, dtype=np.float32)
    return averages


def compute_state_prediction_correlations(
    feature_maps: dict[str, np.ndarray],
    distance_maps: dict[str, np.ndarray],
    switching_coefficient: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    sources = {
        "I_rat_initial": feature_maps["I_rat"],
        "W_EF_initial": feature_maps["W_EF"],
        "E_LE_initial": feature_maps["E_LE"],
        "Gamma_initial": feature_maps["Gamma_EDC"],
        "E_p3_initial": feature_maps["S_orient"],
        "distance_to_edge": distance_maps["distance_to_edge"],
        "distance_to_phase_boundary": distance_maps["distance_to_phase_boundary"],
        "distance_to_structural_boundary": distance_maps["distance_to_structural_boundary"],
    }
    return {
        name: _pearson_correlation(values, switching_coefficient, valid_mask)
        for name, values in sources.items()
    }


def interpret_state_prediction(correlation_values: dict[str, float]) -> str:
    spectral_keys = ("I_rat_initial", "W_EF_initial", "E_LE_initial", "Gamma_initial", "E_p3_initial")
    boundary_keys = ("distance_to_edge", "distance_to_phase_boundary", "distance_to_structural_boundary")
    spectral_strength = max((abs(correlation_values.get(key, float("nan"))) for key in spectral_keys), default=float("nan"))
    boundary_strength = max((abs(correlation_values.get(key, float("nan"))) for key in boundary_keys), default=float("nan"))
    spectral_strength = spectral_strength if np.isfinite(spectral_strength) else 0.0
    boundary_strength = boundary_strength if np.isfinite(boundary_strength) else 0.0

    if spectral_strength >= 0.35 and spectral_strength >= boundary_strength - 0.08:
        return (
            "Future switching appears correlated with the initial electronic spectrum. "
            "This suggests switching may be seeded by pre-existing local electronic structure."
        )
    if boundary_strength >= 0.35 and boundary_strength > spectral_strength + 0.08:
        return (
            "Future switching appears more correlated with spatial environment or morphology than with the local spectrum alone. "
            "This suggests switching may be controlled by current flow, heat dissipation, or boundary geometry."
        )
    return (
        "The initial ARPES-derived features do not strongly predict switching. "
        "The decisive variables may be hidden from this analysis, such as local current density, transient temperature, strain, subsurface defects, or contact geometry."
    )


def state_prediction_table_rows(result: StatePredictionResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x_size, y_size = result.shape
    switching = result.switching_result
    for x_index in range(x_size):
        for y_index in range(y_size):
            row: dict[str, Any] = {
                "x": x_index,
                "y": y_index,
                "I_rat_initial": float(result.initial_feature_maps["I_rat"][x_index, y_index]),
                "W_EF_initial": float(result.initial_feature_maps["W_EF"][x_index, y_index]),
                "W_LHB_initial": float(result.initial_feature_maps["W_LHB"][x_index, y_index]),
                "E_LE_initial": float(result.initial_feature_maps["E_LE"][x_index, y_index]),
                "Gamma_initial": float(result.initial_feature_maps["Gamma_EDC"][x_index, y_index]),
                "E_p3_initial": float(result.initial_feature_maps["S_orient"][x_index, y_index]),
                "orientation_feature_name": result.orientation_feature_name,
                "distance_to_edge": float(result.distance_maps["distance_to_edge"][x_index, y_index]),
                "distance_to_phase_boundary": float(result.distance_maps["distance_to_phase_boundary"][x_index, y_index]),
                "distance_to_structural_boundary": float(result.distance_maps["distance_to_structural_boundary"][x_index, y_index]),
                "switching_coefficient": float(switching.switching_coefficient_map[x_index, y_index]),
                "total_change": float(switching.total_change_map[x_index, y_index]),
                "max_change": float(switching.max_change_map[x_index, y_index]),
                "net_change": float(switching.net_change_map[x_index, y_index]),
                "future_outcome_code": int(result.code_map[x_index, y_index]),
                "future_outcome_label": str(result.label_map[x_index, y_index]),
                "valid": bool(result.valid_mask[x_index, y_index]),
            }
            for state_index, state in enumerate(result.loaded_states):
                row[f"file_{state_index}_name"] = state.name
                row[f"file_{state_index}_I_rat"] = float(switching.i_rat_maps[state_index][x_index, y_index])
                row[f"file_{state_index}_W_EF"] = float(switching.w_ef_maps[state_index][x_index, y_index])
            for transition_index, delta_map in enumerate(switching.delta_irat_maps):
                row[f"Delta_Irat_{transition_index}_to_{transition_index + 1}"] = float(delta_map[x_index, y_index])
            rows.append(row)
    return rows


def export_state_prediction(result: StatePredictionResult, output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    parameters_path = output_path / "state_prediction_parameters.json"
    thresholds_path = output_path / "state_prediction_thresholds.json"
    table_path = output_path / "state_prediction_table.csv"
    labels_path = output_path / "state_prediction_labels.json"
    code_map_path = output_path / "state_prediction_code_map.npy"
    score_map_path = output_path / "state_prediction_predictive_score_map.npy"

    parameters_path.write_text(json.dumps(asdict(result.parameters), indent=2), encoding="utf-8")
    thresholds_path.write_text(json.dumps(result.threshold_values, indent=2), encoding="utf-8")
    labels_path.write_text(json.dumps(result.label_map.tolist(), indent=2), encoding="utf-8")
    write_rows_to_csv(table_path, state_prediction_table_rows(result))
    np.save(code_map_path, result.code_map)
    np.save(score_map_path, result.switching_result.switching_coefficient_map)

    maps_dir = output_path / "state_prediction_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    for name, values in result.initial_feature_maps.items():
        np.save(maps_dir / f"{name}_initial.npy", values)
    for name, values in result.distance_maps.items():
        np.save(maps_dir / f"{name}.npy", values)
    np.save(maps_dir / "switching_coefficient.npy", result.switching_result.switching_coefficient_map)
    np.save(maps_dir / "net_change.npy", result.switching_result.net_change_map)

    average_dir = output_path / "average_initial_edcs"
    average_dir.mkdir(parents=True, exist_ok=True)
    for label, values in result.average_initial_edcs.items():
        np.save(average_dir / f"{sanitize_filename(label)}.npy", values)

    summary_path = output_path / "state_prediction_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "files": result.file_paths,
                "shape": {"x": int(result.shape[0]), "y": int(result.shape[1])},
                "counts": result.counts,
                "correlations": result.correlation_values,
                "interpretation": result.interpretation,
                "orientation_feature_name": result.orientation_feature_name,
                "notes": result.notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "parameters": parameters_path,
        "thresholds": thresholds_path,
        "table": table_path,
        "labels": labels_path,
        "code_map": code_map_path,
        "score_map": score_map_path,
        "maps": maps_dir,
        "average_initial_edcs": average_dir,
        "summary": summary_path,
    }


def run_transition_outcome_maps(
    file_paths: list[str] | tuple[str, ...],
    parameters: TransitionOutcomeParameters | None = None,
    pulse_labels: list[str] | tuple[str, ...] | None = None,
) -> TransitionOutcomeResult:
    if parameters is None:
        parameters = TransitionOutcomeParameters()
    parameters.validate()

    paths = [str(Path(path).expanduser().resolve()) for path in file_paths]
    if len(paths) < 2:
        raise ValueError("Transition Outcome Maps needs at least two chronological ARPES data files.")

    loaded_states, alignment_notes = align_loaded_states_for_comparison([load_state(path) for path in paths])
    labels = normalize_transition_pulse_labels(pulse_labels, len(loaded_states) - 1)

    switching_parameters = SwitchingMapParameters(
        fermi_level_ev=parameters.fermi_level_ev,
        ef_min_ev=parameters.ef_min_ev,
        ef_max_ev=parameters.ef_max_ev,
        lhb_center_ev=parameters.lhb_center_ev,
        lhb_halfwidth_ev=parameters.lhb_halfwidth_ev,
        smooth_sigma=parameters.smooth_sigma,
        low_signal_quantile=parameters.low_signal_quantile,
        lhb_min_quantile=parameters.lhb_min_quantile,
        epsilon=parameters.epsilon,
    )

    total_maps: list[np.ndarray] = []
    w_ef_maps: list[np.ndarray] = []
    w_lhb_maps: list[np.ndarray] = []
    i_rat_maps: list[np.ndarray] = []
    for state in loaded_states:
        feature_maps = compute_switching_core_feature_maps(state.data_array, switching_parameters)
        total_maps.append(feature_maps["T"])
        w_ef_maps.append(feature_maps["W_EF"])
        w_lhb_maps.append(feature_maps["W_LHB"])
        i_rat_maps.append(feature_maps["I_rat"])

    return build_transition_outcome_result(
        loaded_states=loaded_states,
        parameters=parameters,
        pulse_labels=labels,
        total_maps=total_maps,
        w_ef_maps=w_ef_maps,
        w_lhb_maps=w_lhb_maps,
        i_rat_maps=i_rat_maps,
        notes=alignment_notes,
    )


def normalize_transition_pulse_labels(
    pulse_labels: list[str] | tuple[str, ...] | None,
    n_transitions: int,
) -> list[str]:
    if pulse_labels is None:
        return ["" for _ in range(n_transitions)]
    labels = [str(label).strip() for label in pulse_labels]
    if len(labels) < n_transitions:
        labels.extend(["" for _ in range(n_transitions - len(labels))])
    return labels[:n_transitions]


def build_transition_outcome_result(
    loaded_states: list[LoadedState],
    parameters: TransitionOutcomeParameters,
    pulse_labels: list[str],
    total_maps: list[np.ndarray],
    w_ef_maps: list[np.ndarray],
    w_lhb_maps: list[np.ndarray],
    i_rat_maps: list[np.ndarray],
    notes: list[str] | None = None,
) -> TransitionOutcomeResult:
    parameters.validate()
    if len(loaded_states) < 2:
        raise ValueError("Transition Outcome Maps needs at least two chronological states.")

    total_stack = np.stack([np.asarray(values, dtype=np.float32) for values in total_maps], axis=0)
    w_ef_stack = np.stack([np.asarray(values, dtype=np.float32) for values in w_ef_maps], axis=0)
    w_lhb_stack = np.stack([np.asarray(values, dtype=np.float32) for values in w_lhb_maps], axis=0)
    i_rat_stack = np.stack([np.asarray(values, dtype=np.float32) for values in i_rat_maps], axis=0)

    finite_core = (
        np.all(np.isfinite(total_stack), axis=0)
        & np.all(np.isfinite(w_ef_stack), axis=0)
        & np.all(np.isfinite(w_lhb_stack), axis=0)
        & np.all(np.isfinite(i_rat_stack), axis=0)
    )
    total_min = np.min(np.where(np.isfinite(total_stack), total_stack, np.inf), axis=0).astype(np.float32)
    lhb_min = np.min(np.where(np.isfinite(w_lhb_stack), w_lhb_stack, np.inf), axis=0).astype(np.float32)
    low_signal = _robust_percentile(total_min, parameters.low_signal_quantile, finite_core, fallback=0.0)
    min_lhb = _robust_percentile(lhb_min, parameters.lhb_min_quantile, finite_core, fallback=0.0)
    global_valid = finite_core & (total_min > low_signal) & (lhb_min > min_lhb)

    transitions: list[TransitionOutcomeTransition] = []
    write_events: list[np.ndarray] = []
    erase_events: list[np.ndarray] = []
    pulse_labels = normalize_transition_pulse_labels(pulse_labels, len(loaded_states) - 1)

    for transition_index in range(len(loaded_states) - 1):
        before = transition_index
        after = transition_index + 1
        delta_irat = (i_rat_stack[after] - i_rat_stack[before]).astype(np.float32)
        abs_delta = np.abs(delta_irat).astype(np.float32)
        relative_delta = safe_divide(delta_irat, np.abs(i_rat_stack[before]), eps=parameters.epsilon).astype(np.float32)
        delta_w_ef = (w_ef_stack[after] - w_ef_stack[before]).astype(np.float32)
        delta_w_lhb = (w_lhb_stack[after] - w_lhb_stack[before]).astype(np.float32)
        metric_delta = relative_delta if parameters.use_relative_delta else delta_irat
        transition_valid = (
            global_valid
            & np.isfinite(delta_irat)
            & np.isfinite(relative_delta)
            & np.isfinite(delta_w_ef)
            & np.isfinite(delta_w_lhb)
        )
        tau = transition_threshold(metric_delta, transition_valid, parameters.user_min_tau)
        strong_tau = max(tau, float(parameters.strong_tau_multiplier) * tau)
        code_map, label_map = label_transition_delta(metric_delta, transition_valid, tau, strong_tau)
        counts = {
            label: int(np.count_nonzero(code_map == code))
            for code, label in enumerate(TRANSITION_OUTCOME_LABELS)
        }
        stats = transition_summary_stats(delta_irat, metric_delta, transition_valid, code_map)
        stats["tau"] = float(tau)
        stats["strong_tau"] = float(strong_tau)

        transition = TransitionOutcomeTransition(
            index=transition_index,
            before_index=before,
            after_index=after,
            pulse_label=pulse_labels[transition_index],
            delta_irat_map=delta_irat,
            abs_delta_irat_map=abs_delta,
            relative_delta_irat_map=relative_delta,
            delta_w_ef_map=delta_w_ef,
            delta_w_lhb_map=delta_w_lhb,
            metric_delta_map=np.asarray(metric_delta, dtype=np.float32),
            label_map=label_map,
            code_map=code_map,
            valid_mask=transition_valid,
            tau=float(tau),
            strong_tau=float(strong_tau),
            counts=counts,
            stats=stats,
        )
        transitions.append(transition)
        write_events.append((code_map == 2) | (code_map == 4))
        erase_events.append((code_map == 3) | (code_map == 5))

    write_stack = np.stack(write_events, axis=0) if write_events else np.zeros((0, *global_valid.shape), dtype=bool)
    erase_stack = np.stack(erase_events, axis=0) if erase_events else np.zeros((0, *global_valid.shape), dtype=bool)
    write_count = np.sum(write_stack, axis=0, dtype=np.int16).astype(np.int16)
    erase_count = np.sum(erase_stack, axis=0, dtype=np.int16).astype(np.int16)
    activity_count = (write_count + erase_count).astype(np.int16)
    repeated_switching = compute_repeated_switching_map(write_stack, erase_stack).astype(np.int8)
    net_sequence_change = (i_rat_stack[-1] - i_rat_stack[0]).astype(np.float32)

    result_notes = list(notes or [])
    result_notes.append(
        f"Valid mask used T > {low_signal:.5g} and W_LHB > {min_lhb:.5g} across all files."
    )

    return TransitionOutcomeResult(
        loaded_states=loaded_states,
        parameters=parameters,
        pulse_labels=pulse_labels,
        total_maps=[np.asarray(values, dtype=np.float32) for values in total_maps],
        w_ef_maps=[np.asarray(values, dtype=np.float32) for values in w_ef_maps],
        w_lhb_maps=[np.asarray(values, dtype=np.float32) for values in w_lhb_maps],
        i_rat_maps=[np.asarray(values, dtype=np.float32) for values in i_rat_maps],
        transitions=transitions,
        write_count_map=write_count,
        erase_count_map=erase_count,
        activity_count_map=activity_count,
        repeated_switching_map=repeated_switching,
        net_sequence_change_map=net_sequence_change,
        valid_mask=global_valid,
        notes=result_notes,
    )


def transition_threshold(delta_map: np.ndarray, valid_mask: np.ndarray, user_min_tau: float) -> float:
    values = np.asarray(delta_map, dtype=np.float32)[np.asarray(valid_mask, dtype=bool)]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(max(user_min_tau, 0.0))
    robust_std = float(np.nanstd(finite))
    tau = max(float(user_min_tau), 0.5 * robust_std)
    return float(tau) if np.isfinite(tau) else float(max(user_min_tau, 0.0))


def label_transition_delta(
    delta_map: np.ndarray,
    valid_mask: np.ndarray,
    tau: float,
    strong_tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(delta_map, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(delta)
    code_map = np.zeros(delta.shape, dtype=int)
    label_map = np.empty(delta.shape, dtype=object)
    label_map[:] = TRANSITION_OUTCOME_LABELS[0]

    unchanged = valid & (np.abs(delta) < tau)
    written = valid & (delta >= tau)
    erased = valid & (delta <= -tau)
    strong_written = valid & (delta >= strong_tau)
    strong_erased = valid & (delta <= -strong_tau)

    code_map[unchanged] = 1
    label_map[unchanged] = TRANSITION_OUTCOME_LABELS[1]
    code_map[written] = 2
    label_map[written] = TRANSITION_OUTCOME_LABELS[2]
    code_map[erased] = 3
    label_map[erased] = TRANSITION_OUTCOME_LABELS[3]
    code_map[strong_written] = 4
    label_map[strong_written] = TRANSITION_OUTCOME_LABELS[4]
    code_map[strong_erased] = 5
    label_map[strong_erased] = TRANSITION_OUTCOME_LABELS[5]
    return code_map, label_map


def transition_summary_stats(
    delta_irat: np.ndarray,
    metric_delta: np.ndarray,
    valid_mask: np.ndarray,
    code_map: np.ndarray,
) -> dict[str, float]:
    valid = np.asarray(valid_mask, dtype=bool)
    delta = np.asarray(delta_irat, dtype=np.float32)
    metric = np.asarray(metric_delta, dtype=np.float32)
    delta_values = delta[valid & np.isfinite(delta)]
    metric_values = metric[valid & np.isfinite(metric)]
    valid_count = max(1, int(np.count_nonzero(valid)))
    written_count = int(np.count_nonzero((code_map == 2) | (code_map == 4)))
    erased_count = int(np.count_nonzero((code_map == 3) | (code_map == 5)))
    unchanged_count = int(np.count_nonzero(code_map == 1))
    return {
        "valid_pixels": float(valid_count),
        "written_pixels": float(written_count),
        "erased_pixels": float(erased_count),
        "unchanged_pixels": float(unchanged_count),
        "fraction_written": float(written_count / valid_count),
        "fraction_erased": float(erased_count / valid_count),
        "mean_delta_irat": float(np.nanmean(delta_values)) if delta_values.size else float("nan"),
        "median_delta_irat": float(np.nanmedian(delta_values)) if delta_values.size else float("nan"),
        "max_positive_delta_irat": float(np.nanmax(delta_values)) if delta_values.size else float("nan"),
        "max_negative_delta_irat": float(np.nanmin(delta_values)) if delta_values.size else float("nan"),
        "mean_metric_delta": float(np.nanmean(metric_values)) if metric_values.size else float("nan"),
        "median_metric_delta": float(np.nanmedian(metric_values)) if metric_values.size else float("nan"),
    }


def compute_repeated_switching_map(write_stack: np.ndarray, erase_stack: np.ndarray) -> np.ndarray:
    if write_stack.shape[0] == 0:
        return np.zeros(write_stack.shape[1:], dtype=bool)
    seen_write = np.zeros(write_stack.shape[1:], dtype=bool)
    seen_erase = np.zeros(write_stack.shape[1:], dtype=bool)
    repeated = np.zeros(write_stack.shape[1:], dtype=bool)
    for index in range(write_stack.shape[0]):
        written = np.asarray(write_stack[index], dtype=bool)
        erased = np.asarray(erase_stack[index], dtype=bool)
        repeated |= (written & seen_erase) | (erased & seen_write)
        seen_write |= written
        seen_erase |= erased
    return repeated


def transition_outcome_table_rows(result: TransitionOutcomeResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x_size, y_size = result.shape
    for transition in result.transitions:
        before = transition.before_index
        after = transition.after_index
        before_name = result.loaded_states[before].name
        after_name = result.loaded_states[after].name
        for x_index in range(x_size):
            for y_index in range(y_size):
                rows.append(
                    {
                        "x": x_index,
                        "y": y_index,
                        "transition_index": transition.index,
                        "file_before": before_name,
                        "file_after": after_name,
                        "pulse_direction": transition.pulse_label,
                        "I_rat_before": float(result.i_rat_maps[before][x_index, y_index]),
                        "I_rat_after": float(result.i_rat_maps[after][x_index, y_index]),
                        "Delta_Irat": float(transition.delta_irat_map[x_index, y_index]),
                        "relative_Delta_Irat": float(transition.relative_delta_irat_map[x_index, y_index]),
                        "W_EF_before": float(result.w_ef_maps[before][x_index, y_index]),
                        "W_EF_after": float(result.w_ef_maps[after][x_index, y_index]),
                        "Delta_W_EF": float(transition.delta_w_ef_map[x_index, y_index]),
                        "W_LHB_before": float(result.w_lhb_maps[before][x_index, y_index]),
                        "W_LHB_after": float(result.w_lhb_maps[after][x_index, y_index]),
                        "Delta_W_LHB": float(transition.delta_w_lhb_map[x_index, y_index]),
                        "transition_code": int(transition.code_map[x_index, y_index]),
                        "transition_label": str(transition.label_map[x_index, y_index]),
                    }
                )
    return rows


def export_transition_outcome_maps(result: TransitionOutcomeResult, output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    parameters_path = output_path / "transition_outcome_parameters.json"
    pulse_labels_path = output_path / "transition_pulse_labels.json"
    table_path = output_path / "transition_outcome_table.csv"
    summary_path = output_path / "transition_outcome_summary.json"
    maps_dir = output_path / "transition_outcome_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    parameters_path.write_text(json.dumps(asdict(result.parameters), indent=2), encoding="utf-8")
    pulse_labels_path.write_text(json.dumps(result.pulse_labels, indent=2), encoding="utf-8")
    write_rows_to_csv(table_path, transition_outcome_table_rows(result))

    np.save(maps_dir / "I_rat_maps.npy", np.stack(result.i_rat_maps, axis=0))
    np.save(maps_dir / "W_EF_maps.npy", np.stack(result.w_ef_maps, axis=0))
    np.save(maps_dir / "W_LHB_maps.npy", np.stack(result.w_lhb_maps, axis=0))
    np.save(maps_dir / "total_intensity_maps.npy", np.stack(result.total_maps, axis=0))
    np.save(maps_dir / "write_count_map.npy", result.write_count_map)
    np.save(maps_dir / "erase_count_map.npy", result.erase_count_map)
    np.save(maps_dir / "activity_count_map.npy", result.activity_count_map)
    np.save(maps_dir / "repeated_switching_map.npy", result.repeated_switching_map)
    np.save(maps_dir / "net_sequence_change_map.npy", result.net_sequence_change_map)

    transition_dir = maps_dir / "transitions"
    transition_dir.mkdir(parents=True, exist_ok=True)
    for transition in result.transitions:
        folder = transition_dir / f"{transition.index:02d}_{sanitize_filename(result.loaded_states[transition.before_index].name)}_to_{sanitize_filename(result.loaded_states[transition.after_index].name)}"
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "Delta_Irat.npy", transition.delta_irat_map)
        np.save(folder / "abs_Delta_Irat.npy", transition.abs_delta_irat_map)
        np.save(folder / "relative_Delta_Irat.npy", transition.relative_delta_irat_map)
        np.save(folder / "Delta_W_EF.npy", transition.delta_w_ef_map)
        np.save(folder / "Delta_W_LHB.npy", transition.delta_w_lhb_map)
        np.save(folder / "transition_code_map.npy", transition.code_map)
        np.save(folder / "written_mask.npy", ((transition.code_map == 2) | (transition.code_map == 4)).astype(np.int8))
        np.save(folder / "erased_mask.npy", ((transition.code_map == 3) | (transition.code_map == 5)).astype(np.int8))

    summary_path.write_text(
        json.dumps(
            {
                "files": result.file_paths,
                "pulse_labels": result.pulse_labels,
                "shape": {"x": int(result.shape[0]), "y": int(result.shape[1])},
                "transition_summaries": [
                    {
                        "transition_index": transition.index,
                        "file_before": result.loaded_states[transition.before_index].name,
                        "file_after": result.loaded_states[transition.after_index].name,
                        "pulse_direction": transition.pulse_label,
                        "counts": transition.counts,
                        "stats": transition.stats,
                    }
                    for transition in result.transitions
                ],
                "notes": result.notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "parameters": parameters_path,
        "pulse_labels": pulse_labels_path,
        "table": table_path,
        "summary": summary_path,
        "maps": maps_dir,
        "transitions": transition_dir,
    }


def run_initial_transition_feature_analysis(
    file_paths: list[str] | tuple[str, ...],
    parameters: InitialTransitionFeatureParameters | None = None,
) -> InitialTransitionFeatureResult:
    if parameters is None:
        parameters = InitialTransitionFeatureParameters()
    parameters.validate()

    paths = [str(Path(path).expanduser().resolve()) for path in file_paths]
    if len(paths) < 2:
        raise ValueError("Please provide at least two ARPES data files.")
    if parameters.reference_index >= len(paths):
        raise ValueError("reference_index is outside the uploaded file list.")

    loaded_states, alignment_notes = load_transition_file_sequence(paths)
    normalized_states = normalize_transition_states(loaded_states, parameters)
    pairs = build_transition_pairs(normalized_states, parameters.transition_mode, parameters.reference_index)
    if not pairs:
        raise ValueError("No transition pairs could be built from the selected files.")

    transitions: list[InitialTransitionPairMetrics] = []
    for index, (before_index, after_index) in enumerate(pairs):
        transition = compute_transition_metrics_for_pair(
            normalized_states[before_index],
            normalized_states[after_index],
            index=index,
            before_index=before_index,
            after_index=after_index,
            parameters=parameters,
        )
        transitions.append(transition)

    aggregate_maps = aggregate_transition_statistics(transitions)
    reference = normalized_states[parameters.reference_index]
    initial_features = extract_initial_state_features(reference, parameters)
    initial_near_ef = initial_features["near_EF_intensity_A0"]
    initial_feature = initial_features["feature_window_intensity_A0"]
    valid_mask = np.isfinite(initial_near_ef) & np.isfinite(initial_feature) & np.isfinite(initial_features["I_rat_A0"])

    future_metallic = build_future_metallic_mask(aggregate_maps)
    future_erased = build_future_erased_mask(aggregate_maps)
    both = future_metallic & future_erased
    stable = (aggregate_maps["stable_count"] > 0) & (aggregate_maps["metallic_count"] == 0) & (aggregate_maps["erased_count"] == 0)
    never_switched = (aggregate_maps["metallic_count"] == 0) & (aggregate_maps["erased_count"] == 0)

    average_initial_edcs, average_initial_mdcs = compute_initial_group_average_spectra(
        reference,
        {
            "future metallic": future_metallic,
            "future erased": future_erased,
            "both metallic and erased": both,
            "stable": stable,
            "never switched": never_switched,
        },
        parameters,
    )
    group_statistics = compute_group_statistics(
        initial_features,
        {
            "future metallic": future_metallic,
            "future erased": future_erased,
            "both metallic and erased": both,
            "stable": stable,
            "never switched": never_switched,
        },
    )

    notes = list(alignment_notes)
    if parameters.normalization_mode != "none":
        notes.append(f"Applied per-file normalization mode: {parameters.normalization_mode}.")
    notes.append(
        "metallic_count counts transitions where a pixel gained I_rat = W_EF / W_LHB; "
        "erased_count counts transitions where I_rat decreased, marking pixels whose metallicity was erased."
    )
    return InitialTransitionFeatureResult(
        loaded_states=normalized_states,
        parameters=parameters,
        transitions=transitions,
        initial_reference_index=parameters.reference_index,
        initial_near_ef_map=initial_near_ef,
        initial_feature_map=initial_feature,
        initial_feature_maps=initial_features,
        aggregate_maps=aggregate_maps,
        future_metallic_mask=future_metallic,
        future_erased_mask=future_erased,
        both_metallic_erased_mask=both,
        stable_mask=stable,
        never_switched_mask=never_switched,
        average_initial_edcs=average_initial_edcs,
        average_initial_mdcs=average_initial_mdcs,
        group_statistics=group_statistics,
        valid_mask=valid_mask,
        notes=notes,
    )


def load_transition_file_sequence(file_paths: list[str] | tuple[str, ...]) -> tuple[list[LoadedState], list[str]]:
    """Load and spatially align a sequence for transition-feature analysis.

    The app-wide canonical array convention is (x, y, eV, phi). This keeps
    existing pixel indexing consistent across the analysis, transition outcome,
    and clustering panels.
    """

    return align_loaded_states_for_comparison([load_state(path, load=False) for path in file_paths])


def normalize_transition_states(
    loaded_states: list[LoadedState],
    parameters: InitialTransitionFeatureParameters,
) -> list[LoadedState]:
    if parameters.normalization_mode == "none":
        return loaded_states

    normalized: list[LoadedState] = []
    for state in loaded_states:
        values = np.asarray(state.data_array.values, dtype=np.float32)
        scale = transition_file_normalization_scale(state.data_array, parameters)
        if not np.isfinite(scale) or scale <= parameters.epsilon:
            scale = 1.0
        da = xr.DataArray(
            values / float(scale),
            dims=state.data_array.dims,
            coords=state.data_array.coords,
            name=state.data_array.name,
            attrs=state.data_array.attrs,
        )
        normalized.append(LoadedState(name=state.name, file_path=state.file_path, data_array=da))
    return normalized


def transition_file_normalization_scale(
    da: xr.DataArray,
    parameters: InitialTransitionFeatureParameters,
) -> float:
    values = np.asarray(da.values, dtype=np.float32)
    if parameters.normalization_mode == "total_intensity":
        return float(np.nanmean(np.nansum(values, axis=(2, 3))))
    if parameters.normalization_mode == "high_percentile":
        finite = values[np.isfinite(values)]
        return float(np.nanpercentile(finite, 98)) if finite.size else 1.0
    if parameters.normalization_mode == "median_near_ef":
        ef_map = compute_integrated_intensity(
            values,
            np.asarray(da.coords["eV"].values, dtype=np.float32),
            np.asarray(da.coords["phi"].values, dtype=np.float32),
            (parameters.fermi_level_ev + parameters.ef_min_ev, parameters.fermi_level_ev + parameters.ef_max_ev),
        )
        finite = ef_map[np.isfinite(ef_map)]
        return float(np.nanmedian(finite)) if finite.size else 1.0
    return 1.0


def build_transition_pairs(
    files: list[LoadedState],
    mode: str,
    reference_index: int = 0,
) -> list[tuple[int, int]]:
    if mode == "initial_reference":
        return [(reference_index, index) for index in range(len(files)) if index != reference_index]
    return [(index, index + 1) for index in range(len(files) - 1)]


def compute_integrated_intensity(
    cube: np.ndarray,
    energy_axis: np.ndarray,
    phi_axis: np.ndarray,
    energy_window: tuple[float, float],
) -> np.ndarray:
    values = np.asarray(cube, dtype=np.float32)
    energy = np.asarray(energy_axis, dtype=np.float32)
    phi = np.asarray(phi_axis, dtype=np.float32)
    mask = (energy >= min(energy_window)) & (energy <= max(energy_window))
    if not np.any(mask):
        raise ValueError(f"No eV samples were found inside energy window {energy_window}.")
    subset = values[:, :, mask, :]
    if phi.size > 1:
        phi_integrated = np.trapezoid(subset, x=phi, axis=3)
    else:
        phi_integrated = np.sum(subset, axis=3)
    if int(np.count_nonzero(mask)) > 1:
        return np.trapezoid(phi_integrated, x=energy[mask], axis=2).astype(np.float32)
    return np.sum(phi_integrated, axis=2).astype(np.float32)


def sorted_required_dataarray(da: xr.DataArray) -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    require_dims(da)
    ordered = da.transpose(*REQUIRED_DIMS)
    energy_axis = np.asarray(ordered.coords["eV"].values, dtype=np.float32)
    phi_axis = np.asarray(ordered.coords["phi"].values, dtype=np.float32)
    energy_order = np.argsort(energy_axis)
    phi_order = np.argsort(phi_axis)
    if not np.array_equal(energy_order, np.arange(energy_axis.size)):
        ordered = ordered.isel(eV=energy_order)
        energy_axis = energy_axis[energy_order]
    if not np.array_equal(phi_order, np.arange(phi_axis.size)):
        ordered = ordered.isel(phi=phi_order)
        phi_axis = phi_axis[phi_order]
    return ordered, energy_axis, phi_axis


def integrate_dataarray_phi(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    ordered, energy_axis, phi_axis = sorted_required_dataarray(da)
    if phi_axis.size > 1:
        edc = ordered.fillna(0).integrate(coord="phi")
    else:
        edc = ordered.sum(dim="phi", skipna=True)
    return np.asarray(edc.values, dtype=np.float32), energy_axis


def integrate_dataarray_phi_energy_range(
    da: xr.DataArray,
    low: float,
    high: float,
    padding_ev: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    ordered, energy_axis, phi_axis = sorted_required_dataarray(da)
    padded_low = min(low, high) - max(0.0, float(padding_ev))
    padded_high = max(low, high) + max(0.0, float(padding_ev))
    mask = (energy_axis >= padded_low) & (energy_axis <= padded_high)
    if not np.any(mask):
        center = 0.5 * (low + high)
        mask[int(np.argmin(np.abs(energy_axis - center)))] = True
    subset = ordered.isel(eV=np.flatnonzero(mask))
    if phi_axis.size > 1:
        edc = subset.fillna(0).integrate(coord="phi")
    else:
        edc = subset.sum(dim="phi", skipna=True)
    return np.asarray(edc.values, dtype=np.float32), energy_axis[mask]


def integrate_dataarray_energy_window(
    da: xr.DataArray,
    energy_window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    ordered, energy_axis, phi_axis = sorted_required_dataarray(da)
    mask = (energy_axis >= min(energy_window)) & (energy_axis <= max(energy_window))
    if not np.any(mask):
        mask[int(np.argmin(np.abs(energy_axis - np.mean(energy_window))))] = True
    subset = ordered.isel(eV=np.flatnonzero(mask))
    if int(np.count_nonzero(mask)) > 1:
        mdc = subset.fillna(0).integrate(coord="eV")
    else:
        mdc = subset.sum(dim="eV", skipna=True)
    return np.asarray(mdc.values, dtype=np.float32), phi_axis


def compute_integrated_intensity_dataarray(
    da: xr.DataArray,
    energy_window: tuple[float, float],
) -> np.ndarray:
    mdc, phi_axis = integrate_dataarray_energy_window(da, energy_window)
    if phi_axis.size > 1:
        return np.trapezoid(mdc, x=phi_axis, axis=2).astype(np.float32)
    return np.sum(mdc, axis=2).astype(np.float32)


def compute_transition_metrics_for_pair(
    state_a: LoadedState,
    state_b: LoadedState,
    index: int,
    before_index: int,
    after_index: int,
    parameters: InitialTransitionFeatureParameters,
) -> InitialTransitionPairMetrics:
    a_features = compute_initial_transition_core_feature_maps(state_a.data_array, parameters)
    b_features = compute_initial_transition_core_feature_maps(state_b.data_array, parameters)
    delta_irat = (b_features["I_rat"] - a_features["I_rat"]).astype(np.float32)
    metallicity_score = delta_irat
    erasure_score = (-delta_irat).astype(np.float32)
    transition_magnitude = np.abs(delta_irat).astype(np.float32)

    metallicity_norm = robust_zscore_map(metallicity_score)
    erasure_norm = robust_zscore_map(erasure_score)
    magnitude_norm = robust_zscore_map(transition_magnitude)
    metallic_mask, erased_mask, stable_mask, thresholds = classify_transition_map(
        metallicity_score,
        erasure_score,
        transition_magnitude,
        parameters.metallic_percentile,
        parameters.erasure_percentile,
        parameters.stable_percentile,
        allow_overlap=parameters.allow_overlap,
    )
    name = f"{state_a.name} -> {state_b.name}"
    return InitialTransitionPairMetrics(
        index=index,
        before_index=before_index,
        after_index=after_index,
        name=name,
        metallicity_score=metallicity_score,
        erasure_score=erasure_score,
        transition_magnitude=transition_magnitude,
        metallicity_score_norm=metallicity_norm,
        erasure_score_norm=erasure_norm,
        transition_magnitude_norm=magnitude_norm,
        metallic_mask=metallic_mask,
        erased_mask=erased_mask,
        stable_mask=stable_mask,
        metallic_threshold=thresholds["metallic"],
        erasure_threshold=thresholds["erased"],
        stable_threshold=thresholds["stable"],
    )


def compute_initial_transition_core_feature_maps(
    da: xr.DataArray,
    parameters: InitialTransitionFeatureParameters,
) -> dict[str, np.ndarray]:
    parameters.validate()
    ef_window = (
        parameters.fermi_level_ev + parameters.ef_min_ev,
        parameters.fermi_level_ev + parameters.ef_max_ev,
    )
    lhb_window = (
        parameters.lhb_center_ev - parameters.lhb_halfwidth_ev,
        parameters.lhb_center_ev + parameters.lhb_halfwidth_ev,
    )
    _, energy_axis, _ = sorted_required_dataarray(da)
    if energy_axis.size > 1:
        steps = np.abs(np.diff(energy_axis))
        steps = steps[np.isfinite(steps) & (steps > 0)]
        energy_padding = float(np.nanmedian(steps)) * max(0.0, parameters.smooth_sigma) * 3.0 if steps.size else 0.0
    else:
        energy_padding = 0.0
    w_ef = compute_smoothed_dataarray_window_weight(da, ef_window, parameters.smooth_sigma, energy_padding)
    w_lhb = compute_smoothed_dataarray_window_weight(da, lhb_window, parameters.smooth_sigma, energy_padding)
    return {
        "T": (w_ef + w_lhb).astype(np.float32),
        "W_EF": w_ef,
        "W_LHB": w_lhb,
        "I_rat": safe_divide(w_ef, w_lhb, eps=parameters.epsilon).astype(np.float32),
    }


def compute_smoothed_dataarray_window_weight(
    da: xr.DataArray,
    energy_window: tuple[float, float],
    smooth_sigma: float,
    padding_ev: float,
) -> np.ndarray:
    edc, energy_axis = integrate_dataarray_phi_energy_range(
        da,
        min(energy_window),
        max(energy_window),
        padding_ev=padding_ev,
    )
    smoothed_edc = (
        ndimage.gaussian_filter1d(edc, sigma=smooth_sigma, axis=2, mode="nearest")
        if smooth_sigma > 0
        else edc
    ).astype(np.float32)
    mask = (energy_axis >= min(energy_window)) & (energy_axis <= max(energy_window))
    if not np.any(mask):
        raise ValueError(f"No eV samples were found inside energy window {energy_window}.")
    return _integrate_window(smoothed_edc, energy_axis, mask).astype(np.float32)


def compute_initial_transition_core_feature_maps_from_edc(
    edc: np.ndarray,
    energy_axis: np.ndarray,
    parameters: InitialTransitionFeatureParameters,
) -> dict[str, np.ndarray]:
    smoothed_edc = (
        ndimage.gaussian_filter1d(edc, sigma=parameters.smooth_sigma, axis=2, mode="nearest")
        if parameters.smooth_sigma > 0
        else edc
    ).astype(np.float32)
    total_intensity = _integrate_along_axis(edc, energy_axis, axis=2).astype(np.float32)
    ef_mask = _energy_window_mask(
        energy_axis,
        parameters.fermi_level_ev + parameters.ef_min_ev,
        parameters.fermi_level_ev + parameters.ef_max_ev,
    )
    lhb_mask = get_energy_mask(
        energy_axis,
        center=parameters.lhb_center_ev,
        halfwidth=parameters.lhb_halfwidth_ev,
    )
    if not ef_mask.any():
        raise ValueError("No energy samples were found inside the near-EF I_rat window.")
    if not lhb_mask.any():
        raise ValueError("No energy samples were found inside the LHB/p1 I_rat window.")
    w_ef = _integrate_window(smoothed_edc, energy_axis, ef_mask).astype(np.float32)
    w_lhb = _integrate_window(smoothed_edc, energy_axis, lhb_mask).astype(np.float32)
    return {
        "T": total_intensity,
        "W_EF": w_ef,
        "W_LHB": w_lhb,
        "I_rat": safe_divide(w_ef, w_lhb, eps=parameters.epsilon).astype(np.float32),
    }


def classify_transition_map(
    metallicity_score: np.ndarray,
    erasure_score: np.ndarray,
    transition_magnitude: np.ndarray,
    metallicity_percentile: float,
    erasure_percentile: float,
    stable_percentile: float,
    allow_overlap: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    metallic = np.asarray(metallicity_score, dtype=np.float32)
    erased = np.asarray(erasure_score, dtype=np.float32)
    magnitude = np.asarray(transition_magnitude, dtype=np.float32)
    valid = np.isfinite(metallic) & np.isfinite(erased) & np.isfinite(magnitude)
    metallic_threshold = finite_percentile(metallic[valid & (metallic > 0)], metallicity_percentile)
    erasure_threshold = finite_percentile(erased[valid & (erased > 0)], erasure_percentile)
    stable_threshold = finite_percentile(magnitude[valid], stable_percentile)
    metallic_mask = valid & (metallic > 0) & (metallic > metallic_threshold)
    erased_mask = valid & (erased > 0) & (erased > erasure_threshold)
    stable_mask = valid & (magnitude < stable_threshold)
    if not allow_overlap:
        both = metallic_mask & erased_mask
        metallic_mask = metallic_mask & ~both
        erased_mask = erased_mask & ~both
        stable_mask = stable_mask & ~metallic_mask & ~erased_mask
    return metallic_mask, erased_mask, stable_mask, {
        "metallic": float(metallic_threshold),
        "erased": float(erasure_threshold),
        "stable": float(stable_threshold),
    }


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanpercentile(finite, percentile))


def robust_zscore_map(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    if finite.size == 0:
        return out
    median = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - median)))
    scale = 1.4826 * mad if mad > 0 else float(np.nanstd(finite))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    out[np.isfinite(arr)] = ((arr[np.isfinite(arr)] - median) / scale).astype(np.float32)
    return out


def aggregate_transition_statistics(transitions: list[InitialTransitionPairMetrics]) -> dict[str, np.ndarray]:
    shape = transitions[0].metallicity_score.shape
    metallic_stack = np.stack([transition.metallic_mask for transition in transitions], axis=0)
    erased_stack = np.stack([transition.erased_mask for transition in transitions], axis=0)
    stable_stack = np.stack([transition.stable_mask for transition in transitions], axis=0)
    metallic_scores = np.stack([transition.metallicity_score for transition in transitions], axis=0)
    erasure_scores = np.stack([transition.erasure_score for transition in transitions], axis=0)
    n_transitions = max(1, len(transitions))
    first_metallic = first_true_index_map(metallic_stack)
    first_erased = first_true_index_map(erased_stack)
    return {
        "metallic_count": np.sum(metallic_stack, axis=0).astype(np.int16),
        "erased_count": np.sum(erased_stack, axis=0).astype(np.int16),
        "stable_count": np.sum(stable_stack, axis=0).astype(np.int16),
        "metallic_frequency": (np.sum(metallic_stack, axis=0) / n_transitions).astype(np.float32),
        "erased_frequency": (np.sum(erased_stack, axis=0) / n_transitions).astype(np.float32),
        "stable_frequency": (np.sum(stable_stack, axis=0) / n_transitions).astype(np.float32),
        "max_metallicity_score": np.nanmax(metallic_scores, axis=0).astype(np.float32),
        "mean_metallicity_score": np.nanmean(metallic_scores, axis=0).astype(np.float32),
        "max_erasure_score": np.nanmax(erasure_scores, axis=0).astype(np.float32),
        "mean_erasure_score": np.nanmean(erasure_scores, axis=0).astype(np.float32),
        "first_metallic_transition": first_metallic,
        "first_erased_transition": first_erased,
        "normal_count": np.full(shape, len(transitions), dtype=np.int16) - np.sum(metallic_stack | erased_stack | stable_stack, axis=0).astype(np.int16),
    }


def first_true_index_map(stack: np.ndarray) -> np.ndarray:
    mask = np.asarray(stack, dtype=bool)
    out = np.full(mask.shape[1:], np.nan, dtype=np.float32)
    for index in range(mask.shape[0]):
        newly_true = mask[index] & ~np.isfinite(out)
        out[newly_true] = float(index)
    return out


def extract_initial_state_features(
    initial_state: LoadedState,
    parameters: InitialTransitionFeatureParameters,
) -> dict[str, np.ndarray]:
    edc, energy_axis = integrate_dataarray_phi(initial_state.data_array)
    ef_window = (parameters.fermi_level_ev + parameters.ef_min_ev, parameters.fermi_level_ev + parameters.ef_max_ev)
    feature_window = (parameters.feature_min_ev, parameters.feature_max_ev)
    ef_mask = (energy_axis >= min(ef_window)) & (energy_axis <= max(ef_window))
    if not np.any(ef_mask):
        raise ValueError(f"No eV samples were found inside energy window {ef_window}.")
    feature_mask_for_intensity = (energy_axis >= min(feature_window)) & (energy_axis <= max(feature_window))
    if not np.any(feature_mask_for_intensity):
        raise ValueError(f"No eV samples were found inside energy window {feature_window}.")
    total_mask = np.isfinite(energy_axis)
    near_ef = _integrate_window(edc, energy_axis, ef_mask).astype(np.float32)
    feature = _integrate_window(edc, energy_axis, feature_mask_for_intensity).astype(np.float32)
    total = _integrate_window(edc, energy_axis, total_mask).astype(np.float32)
    core_features = compute_initial_transition_core_feature_maps_from_edc(edc, energy_axis, parameters)
    mdc, phi_axis = integrate_dataarray_energy_window(initial_state.data_array, ef_window)
    x_size, y_size = near_ef.shape

    peak_energy = np.full((x_size, y_size), np.nan, dtype=np.float32)
    peak_amp = np.full((x_size, y_size), np.nan, dtype=np.float32)
    peak_width = np.full((x_size, y_size), np.nan, dtype=np.float32)
    com = np.full((x_size, y_size), np.nan, dtype=np.float32)
    asymmetry = np.full((x_size, y_size), np.nan, dtype=np.float32)
    mdc_peak_pos = np.full((x_size, y_size), np.nan, dtype=np.float32)
    mdc_peak_width = np.full((x_size, y_size), np.nan, dtype=np.float32)
    feature_mask = feature_mask_for_intensity
    low_mask = energy_axis < parameters.asymmetry_split_ev
    high_mask = energy_axis >= parameters.asymmetry_split_ev

    for x_index in range(x_size):
        for y_index in range(y_size):
            profile = np.asarray(edc[x_index, y_index, :], dtype=np.float32)
            if not np.any(np.isfinite(profile)):
                continue
            finite_profile = finite_fill(profile, 0.0)
            if np.any(feature_mask):
                local = finite_profile[feature_mask]
                local_energy = energy_axis[feature_mask]
            else:
                local = finite_profile
                local_energy = energy_axis
            if local.size:
                peak_idx = int(np.nanargmax(local))
                peak_energy[x_index, y_index] = float(local_energy[peak_idx])
                peak_amp[x_index, y_index] = float(local[peak_idx])
                peak_width[x_index, y_index] = fwhm_1d(local_energy, local)
            denominator = float(np.nansum(np.abs(finite_profile)))
            if denominator > 0:
                com[x_index, y_index] = float(np.nansum(energy_axis * finite_profile) / max(np.nansum(finite_profile), parameters.epsilon))
                low = float(np.nansum(finite_profile[low_mask])) if np.any(low_mask) else 0.0
                high = float(np.nansum(finite_profile[high_mask])) if np.any(high_mask) else 0.0
                asymmetry[x_index, y_index] = float((high - low) / (abs(high) + abs(low) + parameters.epsilon))
            mdc_profile = finite_fill(mdc[x_index, y_index, :], 0.0)
            if mdc_profile.size:
                mdc_idx = int(np.nanargmax(mdc_profile))
                mdc_peak_pos[x_index, y_index] = float(phi_axis[mdc_idx])
                mdc_peak_width[x_index, y_index] = fwhm_1d(phi_axis, mdc_profile)

    grad_x, grad_y = np.gradient(finite_fill(near_ef, 0.0))
    spatial_gradient = np.sqrt(grad_x * grad_x + grad_y * grad_y).astype(np.float32)
    local_mean = ndimage.uniform_filter(finite_fill(near_ef, 0.0), size=3, mode="nearest").astype(np.float32)
    local_sq = ndimage.uniform_filter(finite_fill(near_ef, 0.0) ** 2, size=3, mode="nearest")
    local_std = np.sqrt(np.clip(local_sq - local_mean * local_mean, 0.0, None)).astype(np.float32)
    return {
        "I_rat_A0": core_features["I_rat"].astype(np.float32),
        "W_EF_A0": core_features["W_EF"].astype(np.float32),
        "W_LHB_A0": core_features["W_LHB"].astype(np.float32),
        "near_EF_intensity_A0": near_ef.astype(np.float32),
        "feature_window_intensity_A0": feature.astype(np.float32),
        "edc_peak_energy_A0": peak_energy,
        "edc_peak_amplitude_A0": peak_amp,
        "edc_peak_width_A0": peak_width,
        "total_spectral_weight_A0": total.astype(np.float32),
        "edc_center_of_mass_A0": com,
        "edc_asymmetry_A0": asymmetry,
        "initial_MDC_peak_position_A0": mdc_peak_pos,
        "initial_MDC_peak_width_A0": mdc_peak_width,
        "local_spatial_gradient_A0": spatial_gradient,
        "local_neighborhood_mean_A0": local_mean,
        "local_neighborhood_std_A0": local_std,
    }


def integrate_phi(values: np.ndarray, phi_axis: np.ndarray) -> np.ndarray:
    if np.asarray(phi_axis).size > 1:
        return np.trapezoid(values, x=phi_axis, axis=3).astype(np.float32)
    return np.sum(values, axis=3).astype(np.float32)


def integrate_energy_window(values: np.ndarray, energy_axis: np.ndarray, energy_window: tuple[float, float]) -> np.ndarray:
    energy = np.asarray(energy_axis, dtype=np.float32)
    mask = (energy >= min(energy_window)) & (energy <= max(energy_window))
    if not np.any(mask):
        mask[int(np.argmin(np.abs(energy - np.mean(energy_window))))] = True
    subset = values[:, :, mask, :]
    if int(np.count_nonzero(mask)) > 1:
        return np.trapezoid(subset, x=energy[mask], axis=2).astype(np.float32)
    return np.sum(subset, axis=2).astype(np.float32)


def fwhm_1d(axis: np.ndarray, values: np.ndarray) -> float:
    x = np.asarray(axis, dtype=np.float32)
    y = np.asarray(values, dtype=np.float32)
    if x.size < 2 or y.size != x.size or not np.any(np.isfinite(y)):
        return float("nan")
    y = finite_fill(y, 0.0)
    baseline = float(np.nanmin(y))
    peak = float(np.nanmax(y))
    if not np.isfinite(peak) or peak <= baseline:
        return float("nan")
    half = baseline + 0.5 * (peak - baseline)
    above = np.flatnonzero(y >= half)
    if above.size < 2:
        return float("nan")
    return float(abs(x[int(above[-1])] - x[int(above[0])]))


def build_future_metallic_mask(
    aggregate_stats: dict[str, np.ndarray],
    min_count: int = 1,
    min_frequency: float | None = None,
    score_percentile: float | None = None,
) -> np.ndarray:
    mask = np.asarray(aggregate_stats["metallic_count"] >= min_count, dtype=bool)
    if min_frequency is not None:
        mask &= np.asarray(aggregate_stats["metallic_frequency"] >= min_frequency, dtype=bool)
    if score_percentile is not None:
        threshold = finite_percentile(aggregate_stats["max_metallicity_score"], score_percentile)
        mask &= np.asarray(aggregate_stats["max_metallicity_score"] >= threshold, dtype=bool)
    return mask


def build_future_erased_mask(
    aggregate_stats: dict[str, np.ndarray],
    min_count: int = 1,
    min_frequency: float | None = None,
    score_percentile: float | None = None,
) -> np.ndarray:
    mask = np.asarray(aggregate_stats["erased_count"] >= min_count, dtype=bool)
    if min_frequency is not None:
        mask &= np.asarray(aggregate_stats["erased_frequency"] >= min_frequency, dtype=bool)
    if score_percentile is not None:
        threshold = finite_percentile(aggregate_stats["max_erasure_score"], score_percentile)
        mask &= np.asarray(aggregate_stats["max_erasure_score"] >= threshold, dtype=bool)
    return mask


def compute_initial_group_average_spectra(
    initial_state: LoadedState,
    group_masks: dict[str, np.ndarray],
    parameters: InitialTransitionFeatureParameters,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    edc, energy_axis = integrate_dataarray_phi(initial_state.data_array)
    mdc, phi_axis = integrate_dataarray_energy_window(
        initial_state.data_array,
        (parameters.fermi_level_ev + parameters.ef_min_ev, parameters.fermi_level_ev + parameters.ef_max_ev),
    )
    average_edcs: dict[str, np.ndarray] = {}
    average_mdcs: dict[str, np.ndarray] = {}
    for group, mask in group_masks.items():
        valid = np.asarray(mask, dtype=bool)
        if np.any(valid):
            average_edcs[group] = np.nanmean(edc[valid, :], axis=0).astype(np.float32)
            average_mdcs[group] = np.nanmean(mdc[valid, :], axis=0).astype(np.float32)
        else:
            average_edcs[group] = np.full(energy_axis.shape, np.nan, dtype=np.float32)
            average_mdcs[group] = np.full(phi_axis.shape, np.nan, dtype=np.float32)
    return average_edcs, average_mdcs


def compute_group_statistics(
    initial_features: dict[str, np.ndarray],
    group_masks: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stable_mask = np.asarray(group_masks.get("stable", np.zeros_like(next(iter(initial_features.values())), dtype=bool)), dtype=bool)
    key_features = [
        "I_rat_A0",
        "W_EF_A0",
        "W_LHB_A0",
        "near_EF_intensity_A0",
        "feature_window_intensity_A0",
        "edc_peak_energy_A0",
        "edc_peak_width_A0",
        "total_spectral_weight_A0",
        "edc_center_of_mass_A0",
        "edc_asymmetry_A0",
        "initial_MDC_peak_width_A0",
        "local_spatial_gradient_A0",
    ]
    for group, mask in group_masks.items():
        valid = np.asarray(mask, dtype=bool)
        row: dict[str, Any] = {"group": group, "number_of_pixels": int(np.count_nonzero(valid))}
        for feature_name in key_features:
            values = np.asarray(initial_features[feature_name], dtype=np.float32)
            group_values = values[valid & np.isfinite(values)]
            row[f"mean_{feature_name}"] = float(np.nanmean(group_values)) if group_values.size else float("nan")
            row[f"std_{feature_name}"] = float(np.nanstd(group_values)) if group_values.size else float("nan")
            if group != "stable":
                stable_values = values[stable_mask & np.isfinite(values)]
                row[f"Cohen_d_{group.replace(' ', '_')}_vs_stable_{feature_name}"] = cohen_d(group_values, stable_values)
        rows.append(row)
    return rows


def cohen_d(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = math.sqrt(((a.size - 1) * float(np.nanvar(a)) + (b.size - 1) * float(np.nanvar(b))) / max(1, a.size + b.size - 2))
    if pooled <= 1e-12:
        return float("nan")
    return float((float(np.nanmean(a)) - float(np.nanmean(b))) / pooled)


def initial_transition_metric_rows(result: InitialTransitionFeatureResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x_size, y_size = result.shape
    for x_index in range(x_size):
        for y_index in range(y_size):
            metallic_transition_files = []
            erased_transition_files = []
            stable_transition_files = []
            for transition in result.transitions:
                if bool(transition.metallic_mask[x_index, y_index]):
                    metallic_transition_files.append(transition.name)
                if bool(transition.erased_mask[x_index, y_index]):
                    erased_transition_files.append(transition.name)
                if bool(transition.stable_mask[x_index, y_index]):
                    stable_transition_files.append(transition.name)
            metallic_count = int(result.aggregate_maps["metallic_count"][x_index, y_index])
            erased_count = int(result.aggregate_maps["erased_count"][x_index, y_index])
            stable_count = int(result.aggregate_maps["stable_count"][x_index, y_index])
            if metallic_count and erased_count:
                summary_label = "both metallic and erased"
            elif metallic_count:
                summary_label = "future metallic"
            elif erased_count:
                summary_label = "future erased"
            elif stable_count:
                summary_label = "stable"
            else:
                summary_label = "never switched"
            rows.append(
                {
                    "x": x_index,
                    "y": y_index,
                    "metallic_count": metallic_count,
                    "erased_count": erased_count,
                    "stable_count": stable_count,
                    "metallic_frequency": float(result.aggregate_maps["metallic_frequency"][x_index, y_index]),
                    "erased_frequency": float(result.aggregate_maps["erased_frequency"][x_index, y_index]),
                    "first_metallic_transition": float(result.aggregate_maps["first_metallic_transition"][x_index, y_index]),
                    "first_erased_transition": float(result.aggregate_maps["first_erased_transition"][x_index, y_index]),
                    "max_metallicity_score": float(result.aggregate_maps["max_metallicity_score"][x_index, y_index]),
                    "mean_metallicity_score": float(result.aggregate_maps["mean_metallicity_score"][x_index, y_index]),
                    "max_erasure_score": float(result.aggregate_maps["max_erasure_score"][x_index, y_index]),
                    "mean_erasure_score": float(result.aggregate_maps["mean_erasure_score"][x_index, y_index]),
                    "class_summary_label": summary_label,
                    "metallic_transition_files": "; ".join(metallic_transition_files),
                    "erased_transition_files": "; ".join(erased_transition_files),
                    "stable_transition_files": "; ".join(stable_transition_files),
                }
            )
    return rows


def initial_state_feature_rows(result: InitialTransitionFeatureResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x_size, y_size = result.shape
    feature_names = list(result.initial_feature_maps.keys())
    for x_index in range(x_size):
        for y_index in range(y_size):
            row: dict[str, Any] = {"x": x_index, "y": y_index}
            for name in feature_names:
                row[name] = float(result.initial_feature_maps[name][x_index, y_index])
            rows.append(row)
    return rows


def export_initial_transition_feature_analysis(
    result: InitialTransitionFeatureResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    maps_dir = output_path / "initial_transition_feature_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    transitions_dir = output_path / "initial_transition_pair_maps"
    transitions_dir.mkdir(parents=True, exist_ok=True)

    for name, values in result.aggregate_maps.items():
        np.save(maps_dir / f"{name}.npy", values)
    np.save(maps_dir / "future_metallic_mask.npy", result.future_metallic_mask.astype(np.int8))
    np.save(maps_dir / "future_erased_mask.npy", result.future_erased_mask.astype(np.int8))
    np.save(maps_dir / "both_metallic_erased_mask.npy", result.both_metallic_erased_mask.astype(np.int8))
    np.save(maps_dir / "never_switched_mask.npy", result.never_switched_mask.astype(np.int8))

    for transition in result.transitions:
        transition_dir = transitions_dir / f"{transition.index:02d}_{sanitize_filename(transition.name)}"
        transition_dir.mkdir(parents=True, exist_ok=True)
        np.save(transition_dir / "metallicity_score.npy", transition.metallicity_score)
        np.save(transition_dir / "erasure_score.npy", transition.erasure_score)
        np.save(transition_dir / "transition_magnitude.npy", transition.transition_magnitude)
        np.save(transition_dir / "metallic_mask.npy", transition.metallic_mask.astype(np.int8))
        np.save(transition_dir / "erased_mask.npy", transition.erased_mask.astype(np.int8))
        np.save(transition_dir / "stable_mask.npy", transition.stable_mask.astype(np.int8))

    metrics_path = output_path / "transition_metrics_per_pixel.csv"
    features_path = output_path / "initial_state_features_per_pixel.csv"
    stats_path = output_path / "group_statistics.csv"
    parameters_path = output_path / "initial_transition_feature_parameters.json"
    summary_path = output_path / "initial_transition_feature_summary.json"
    write_rows_to_csv(metrics_path, initial_transition_metric_rows(result))
    write_rows_to_csv(features_path, initial_state_feature_rows(result))
    write_rows_to_csv(stats_path, result.group_statistics)
    parameters_path.write_text(json.dumps(asdict(result.parameters), indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "files": result.file_paths,
                "initial_reference_index": result.initial_reference_index,
                "transitions": [
                    {
                        "index": transition.index,
                        "name": transition.name,
                        "metallic_threshold": transition.metallic_threshold,
                        "erasure_threshold": transition.erasure_threshold,
                        "stable_threshold": transition.stable_threshold,
                    }
                    for transition in result.transitions
                ],
                "notes": result.notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "maps": maps_dir,
        "transitions": transitions_dir,
        "metrics_table": metrics_path,
        "features_table": features_path,
        "group_statistics": stats_path,
        "parameters": parameters_path,
        "summary": summary_path,
    }


def run_switching_mechanism_diagnostics(
    file_paths: list[str] | tuple[str, ...] | None = None,
    transition_result: InitialTransitionFeatureResult | None = None,
    parameters: SwitchingMechanismParameters | None = None,
) -> SwitchingMechanismDiagnosticsResult:
    if parameters is None:
        parameters = SwitchingMechanismParameters()
    parameters.validate()
    if transition_result is None:
        if file_paths is None:
            raise ValueError("Provide file_paths or an existing InitialTransitionFeatureResult.")
        transition_result = run_initial_transition_feature_analysis(file_paths, parameters.transition_parameters)

    group_masks = build_switching_mechanism_group_masks(transition_result, parameters)
    cleaned_group_masks = {
        name: clean_connected_components(mask, parameters.component_min_size)
        for name, mask in group_masks.items()
    }
    (
        group_edcs,
        group_edc_sem,
        group_mdcs,
        group_mdc_sem,
        group_spectra,
        group_spectrum_sem,
    ) = compute_group_spectral_diagnostics(transition_result, group_masks, parameters)
    spectral_effect_rows = compute_effect_size_table(
        transition_result.initial_feature_maps,
        group_masks,
        SWITCHING_MECHANISM_SPECTRAL_FEATURES,
        stable_group="stable",
    )
    spatial_feature_maps = compute_initial_spatial_features(transition_result, parameters)
    spatial_effect_rows = compute_effect_size_table(
        spatial_feature_maps,
        group_masks,
        (
            "local_intensity_gradient",
            "local_neighborhood_mean",
            "local_neighborhood_std",
            "local_contrast_texture",
            "distance_to_domain_boundary",
            "distance_to_valid_edge",
            "x_coordinate",
            "y_coordinate",
        ),
        stable_group="stable",
    )
    connected_component_rows = compute_connected_component_stats_for_groups(group_masks, parameters.component_min_size)
    transition_history_maps = compute_transition_history_maps(transition_result)
    transition_level_rows = compute_transition_level_statistics(transition_result, parameters)
    file_intensity_rows = compute_file_intensity_statistics(transition_result, parameters)
    threshold_sensitivity_rows, threshold_robustness_maps = compute_threshold_sensitivity(transition_result, parameters)
    negative_control_maps = compute_negative_control_scores(transition_result, parameters)
    permutation_control_rows = compute_permutation_control(group_masks, parameters)
    artifact_rows = compute_artifact_diagnostics(
        transition_result,
        group_masks,
        spatial_feature_maps,
        transition_level_rows,
        file_intensity_rows,
        threshold_sensitivity_rows,
        threshold_robustness_maps,
        negative_control_maps,
    )
    summary_verdict = compute_summary_verdict(
        transition_result,
        parameters,
        spectral_effect_rows,
        spatial_effect_rows,
        connected_component_rows,
        transition_history_maps,
        transition_level_rows,
        artifact_rows,
    )
    notes = [
        "Switching Mechanism Diagnostics reuses the I_rat transition labels so every pixel label remains traceable to Delta I_rat scores and thresholds.",
        "Evidence scores are heuristic diagnostics, not proof of a mechanism.",
    ]
    return SwitchingMechanismDiagnosticsResult(
        transition_result=transition_result,
        parameters=parameters,
        group_masks=group_masks,
        cleaned_group_masks=cleaned_group_masks,
        group_edcs=group_edcs,
        group_edc_sem=group_edc_sem,
        group_mdcs=group_mdcs,
        group_mdc_sem=group_mdc_sem,
        group_spectra=group_spectra,
        group_spectrum_sem=group_spectrum_sem,
        spectral_effect_rows=spectral_effect_rows,
        spatial_feature_maps=spatial_feature_maps,
        spatial_effect_rows=spatial_effect_rows,
        connected_component_rows=connected_component_rows,
        transition_history_maps=transition_history_maps,
        transition_level_rows=transition_level_rows,
        file_intensity_rows=file_intensity_rows,
        artifact_rows=artifact_rows,
        threshold_sensitivity_rows=threshold_sensitivity_rows,
        threshold_robustness_maps=threshold_robustness_maps,
        permutation_control_rows=permutation_control_rows,
        negative_control_maps=negative_control_maps,
        summary_verdict=summary_verdict,
        notes=notes,
    )


def build_switching_mechanism_group_masks(
    result: InitialTransitionFeatureResult,
    parameters: SwitchingMechanismParameters,
) -> dict[str, np.ndarray]:
    metallic = build_future_metallic_mask(
        result.aggregate_maps,
        min_count=parameters.future_metallic_min_count,
        min_frequency=parameters.future_metallic_min_frequency,
    )
    erased = build_future_erased_mask(
        result.aggregate_maps,
        min_count=parameters.future_erased_min_count,
        min_frequency=parameters.future_erased_min_frequency,
    )
    both = metallic & erased
    stable = (
        (result.aggregate_maps["stable_count"] > 0)
        & (result.aggregate_maps["metallic_count"] == 0)
        & (result.aggregate_maps["erased_count"] == 0)
    )
    never = (result.aggregate_maps["metallic_count"] == 0) & (result.aggregate_maps["erased_count"] == 0)
    return {
        "future metallic": metallic,
        "future erased": erased,
        "both metallic and erased": both,
        "stable": stable,
        "never switched": never,
    }


def clean_connected_components(mask: np.ndarray, min_size: int = 0) -> np.ndarray:
    bool_mask = np.asarray(mask, dtype=bool)
    if min_size <= 1 or not np.any(bool_mask):
        return bool_mask.copy()
    labels, count = ndimage.label(bool_mask)
    cleaned = np.zeros(bool_mask.shape, dtype=bool)
    for label in range(1, count + 1):
        component = labels == label
        if int(np.count_nonzero(component)) >= min_size:
            cleaned |= component
    return ndimage.binary_fill_holes(cleaned)


def compute_group_spectral_diagnostics(
    result: InitialTransitionFeatureResult,
    group_masks: dict[str, np.ndarray],
    parameters: SwitchingMechanismParameters,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    initial = result.loaded_states[result.initial_reference_index]
    values = np.asarray(initial.data_array.values, dtype=np.float32)
    energy_axis = np.asarray(initial.data_array.coords["eV"].values, dtype=np.float32)
    phi_axis = np.asarray(initial.data_array.coords["phi"].values, dtype=np.float32)
    ef_window = (
        result.parameters.fermi_level_ev + result.parameters.ef_min_ev,
        result.parameters.fermi_level_ev + result.parameters.ef_max_ev,
    )
    edc_stack = integrate_phi(values, phi_axis)
    mdc_stack = integrate_energy_window(values, energy_axis, ef_window)
    group_edcs: dict[str, np.ndarray] = {}
    group_edc_sem: dict[str, np.ndarray] = {}
    group_mdcs: dict[str, np.ndarray] = {}
    group_mdc_sem: dict[str, np.ndarray] = {}
    group_spectra: dict[str, np.ndarray] = {}
    group_spectrum_sem: dict[str, np.ndarray] = {}
    for group, mask in group_masks.items():
        valid = np.asarray(mask, dtype=bool)
        if not np.any(valid):
            group_edcs[group] = np.full(energy_axis.shape, np.nan, dtype=np.float32)
            group_edc_sem[group] = np.full(energy_axis.shape, np.nan, dtype=np.float32)
            group_mdcs[group] = np.full(phi_axis.shape, np.nan, dtype=np.float32)
            group_mdc_sem[group] = np.full(phi_axis.shape, np.nan, dtype=np.float32)
            group_spectra[group] = np.full(values.shape[2:], np.nan, dtype=np.float32)
            group_spectrum_sem[group] = np.full(values.shape[2:], np.nan, dtype=np.float32)
            continue
        edcs = normalize_profiles_for_mechanism(
            edc_stack[valid, :],
            energy_axis,
            parameters.edc_normalization,
            result.parameters,
        )
        mdcs = mdc_stack[valid, :]
        spectra = normalize_spectra_for_mechanism(
            values[valid, :, :],
            energy_axis,
            parameters.edc_normalization,
            result.parameters,
        )
        group_edcs[group] = np.nanmean(edcs, axis=0).astype(np.float32)
        group_edc_sem[group] = nan_sem(edcs, axis=0).astype(np.float32)
        group_mdcs[group] = np.nanmean(mdcs, axis=0).astype(np.float32)
        group_mdc_sem[group] = nan_sem(mdcs, axis=0).astype(np.float32)
        group_spectra[group] = np.nanmean(spectra, axis=0).astype(np.float32)
        group_spectrum_sem[group] = nan_sem(spectra, axis=0).astype(np.float32)
    return group_edcs, group_edc_sem, group_mdcs, group_mdc_sem, group_spectra, group_spectrum_sem


def normalize_profiles_for_mechanism(
    profiles: np.ndarray,
    energy_axis: np.ndarray,
    mode: str,
    transition_parameters: InitialTransitionFeatureParameters,
) -> np.ndarray:
    arr = np.asarray(profiles, dtype=np.float32)
    if mode == "raw":
        return arr.copy()
    denominator = np.ones((arr.shape[0],), dtype=np.float32)
    if mode == "per_pixel_max":
        denominator = np.nanmax(np.abs(arr), axis=1).astype(np.float32)
    elif mode == "total_spectral_weight":
        denominator = np.trapezoid(np.abs(arr), x=energy_axis, axis=1).astype(np.float32)
    elif mode == "feature_window":
        mask = (energy_axis >= transition_parameters.feature_min_ev) & (energy_axis <= transition_parameters.feature_max_ev)
        denominator = np.trapezoid(np.abs(arr[:, mask]), x=energy_axis[mask], axis=1).astype(np.float32) if np.count_nonzero(mask) > 1 else np.sum(np.abs(arr[:, mask]), axis=1).astype(np.float32)
    elif mode == "near_ef":
        lo = transition_parameters.fermi_level_ev + transition_parameters.ef_min_ev
        hi = transition_parameters.fermi_level_ev + transition_parameters.ef_max_ev
        mask = (energy_axis >= lo) & (energy_axis <= hi)
        denominator = np.trapezoid(np.abs(arr[:, mask]), x=energy_axis[mask], axis=1).astype(np.float32) if np.count_nonzero(mask) > 1 else np.sum(np.abs(arr[:, mask]), axis=1).astype(np.float32)
    denominator[~np.isfinite(denominator) | (np.abs(denominator) <= transition_parameters.epsilon)] = 1.0
    return (arr / denominator[:, None]).astype(np.float32)


def normalize_spectra_for_mechanism(
    spectra: np.ndarray,
    energy_axis: np.ndarray,
    mode: str,
    transition_parameters: InitialTransitionFeatureParameters,
) -> np.ndarray:
    arr = np.asarray(spectra, dtype=np.float32)
    if mode == "raw":
        return arr.copy()
    edc = np.nansum(arr, axis=2)
    denominators = normalize_profiles_for_mechanism(
        edc,
        energy_axis,
        mode,
        transition_parameters,
    )
    scale = np.divide(
        edc,
        denominators,
        out=np.ones_like(edc, dtype=np.float32),
        where=np.isfinite(denominators) & (np.abs(denominators) > transition_parameters.epsilon),
    )
    scalar = np.nanmean(np.abs(scale), axis=1)
    scalar[~np.isfinite(scalar) | (scalar <= transition_parameters.epsilon)] = 1.0
    return (arr / scalar[:, None, None]).astype(np.float32)


def nan_sem(values: np.ndarray, axis: int | tuple[int, ...] = 0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    count = np.sum(np.isfinite(arr), axis=axis)
    std = np.nanstd(arr, axis=axis)
    return np.divide(std, np.sqrt(np.maximum(count, 1)), out=np.full_like(std, np.nan, dtype=np.float32), where=count > 0)


def compute_effect_size_table(
    feature_maps: dict[str, np.ndarray],
    group_masks: dict[str, np.ndarray],
    feature_names: tuple[str, ...] | list[str],
    stable_group: str = "stable",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stable_mask = np.asarray(group_masks.get(stable_group, np.zeros_like(next(iter(feature_maps.values())), dtype=bool)), dtype=bool)
    for feature_name in feature_names:
        if feature_name not in feature_maps:
            continue
        values = np.asarray(feature_maps[feature_name], dtype=np.float32)
        stable_values = values[stable_mask & np.isfinite(values)]
        for group, mask in group_masks.items():
            if group == stable_group:
                continue
            group_values = values[np.asarray(mask, dtype=bool) & np.isfinite(values)]
            diff = float(np.nanmean(group_values) - np.nanmean(stable_values)) if group_values.size and stable_values.size else float("nan")
            ci_low, ci_high = bootstrap_mean_difference_ci(group_values, stable_values)
            rows.append(
                {
                    "feature": feature_name,
                    "group": group,
                    "group_n": int(group_values.size),
                    "stable_n": int(stable_values.size),
                    "mean_group": float(np.nanmean(group_values)) if group_values.size else float("nan"),
                    "mean_stable": float(np.nanmean(stable_values)) if stable_values.size else float("nan"),
                    "difference": diff,
                    "cohens_d": cohen_d(group_values, stable_values),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "mannwhitney_p": mannwhitney_pvalue(group_values, stable_values),
                }
            )
    return rows


def bootstrap_mean_difference_ci(
    first: np.ndarray,
    second: np.ndarray,
    n_bootstrap: int = 128,
    seed: int = 123,
) -> tuple[float, float]:
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    a_pool = a if a.size <= 5000 else rng.choice(a, size=5000, replace=False)
    b_pool = b if b.size <= 5000 else rng.choice(b, size=5000, replace=False)
    diffs = np.empty(n_bootstrap, dtype=np.float32)
    for index in range(n_bootstrap):
        a_sample = rng.choice(a_pool, size=a_pool.size, replace=True)
        b_sample = rng.choice(b_pool, size=b_pool.size, replace=True)
        diffs[index] = float(np.nanmean(a_sample) - np.nanmean(b_sample))
    return float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


def mannwhitney_pvalue(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    if a.size > 5000:
        a = np.random.default_rng(11).choice(a, size=5000, replace=False)
    if b.size > 5000:
        b = np.random.default_rng(12).choice(b, size=5000, replace=False)
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def compute_initial_spatial_features(
    result: InitialTransitionFeatureResult,
    parameters: SwitchingMechanismParameters,
) -> dict[str, np.ndarray]:
    base = np.asarray(result.initial_near_ef_map, dtype=np.float32)
    finite_base = finite_fill(base, float(np.nanmedian(base[np.isfinite(base)])) if np.any(np.isfinite(base)) else 0.0)
    smoothed = ndimage.gaussian_filter(finite_base, sigma=parameters.boundary_smooth_sigma) if parameters.boundary_smooth_sigma > 0 else finite_base
    grad_x, grad_y = np.gradient(smoothed)
    gradient = np.sqrt(grad_x * grad_x + grad_y * grad_y).astype(np.float32)
    threshold = finite_percentile(gradient[result.valid_mask], parameters.boundary_percentile) if np.any(result.valid_mask) else finite_percentile(gradient, parameters.boundary_percentile)
    boundary = np.isfinite(gradient) & (gradient >= threshold)
    distance_to_boundary = _distance_to_binary_mask(boundary, result.valid_mask)
    invalid_or_edge = np.zeros(base.shape, dtype=bool)
    invalid_or_edge[0, :] = True
    invalid_or_edge[-1, :] = True
    invalid_or_edge[:, 0] = True
    invalid_or_edge[:, -1] = True
    invalid_or_edge |= ~np.asarray(result.valid_mask, dtype=bool)
    distance_to_edge = ndimage.distance_transform_edt(~invalid_or_edge).astype(np.float32)
    local_mean = ndimage.uniform_filter(finite_base, size=3, mode="nearest").astype(np.float32)
    local_sq = ndimage.uniform_filter(finite_base * finite_base, size=3, mode="nearest").astype(np.float32)
    local_std = np.sqrt(np.clip(local_sq - local_mean * local_mean, 0.0, None)).astype(np.float32)
    contrast = (local_std / (np.abs(local_mean) + parameters.epsilon)).astype(np.float32)
    x_coords = np.broadcast_to(np.arange(base.shape[0], dtype=np.float32)[:, None], base.shape).copy()
    y_coords = np.broadcast_to(np.arange(base.shape[1], dtype=np.float32)[None, :], base.shape).copy()
    return {
        "initial_near_EF": base,
        "local_intensity_gradient": gradient,
        "local_neighborhood_mean": local_mean,
        "local_neighborhood_std": local_std,
        "local_contrast_texture": contrast,
        "domain_boundary_mask": boundary.astype(np.float32),
        "distance_to_domain_boundary": distance_to_boundary,
        "distance_to_valid_edge": distance_to_edge,
        "x_coordinate": x_coords,
        "y_coordinate": y_coords,
    }


def compute_connected_component_stats_for_groups(
    group_masks: dict[str, np.ndarray],
    min_component_size: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, mask in group_masks.items():
        bool_mask = np.asarray(mask, dtype=bool)
        labels, component_count = ndimage.label(bool_mask)
        component_sizes = np.array(
            [int(np.count_nonzero(labels == label)) for label in range(1, component_count + 1)],
            dtype=np.int32,
        )
        coords = np.argwhere(bool_mask)
        mean_nn = mean_nearest_neighbor_distance(coords)
        autocorr = binary_spatial_autocorrelation(bool_mask)
        large_count = int(np.sum(component_sizes >= max(1, min_component_size))) if component_sizes.size else 0
        large_pixels = int(np.sum(component_sizes[component_sizes >= max(1, min_component_size)])) if component_sizes.size else 0
        rows.append(
            {
                "group": group,
                "pixel_count": int(coords.shape[0]),
                "connected_component_count": int(component_count),
                "largest_connected_component_size": int(np.nanmax(component_sizes)) if component_sizes.size else 0,
                "mean_component_size": float(np.nanmean(component_sizes)) if component_sizes.size else float("nan"),
                "mean_nearest_neighbor_distance": mean_nn,
                "large_component_count": large_count,
                "fraction_pixels_in_large_components": float(large_pixels / max(1, coords.shape[0])),
                "spatial_autocorrelation_proxy": autocorr,
            }
        )
    return rows


def mean_nearest_neighbor_distance(coords: np.ndarray) -> float:
    if coords.shape[0] < 2:
        return float("nan")
    sample = coords
    if coords.shape[0] > 8000:
        sample = coords[np.random.default_rng(19).choice(coords.shape[0], size=8000, replace=False)]
    tree = cKDTree(sample.astype(np.float32))
    distances, _indices = tree.query(sample.astype(np.float32), k=2)
    return float(np.nanmean(distances[:, 1]))


def binary_spatial_autocorrelation(mask: np.ndarray) -> float:
    arr = np.asarray(mask, dtype=np.float32)
    if arr.size == 0 or not np.any(arr):
        return 0.0
    centered = arr - float(np.nanmean(arr))
    denominator = float(np.nansum(centered * centered))
    if denominator <= 1e-12:
        return 0.0
    right = float(np.nansum(centered[:-1, :] * centered[1:, :]))
    up = float(np.nansum(centered[:, :-1] * centered[:, 1:]))
    return float((right + up) / denominator)


def compute_transition_history_maps(result: InitialTransitionFeatureResult) -> dict[str, np.ndarray]:
    metallic_stack = np.stack([transition.metallic_mask for transition in result.transitions], axis=0)
    erased_stack = np.stack([transition.erased_mask for transition in result.transitions], axis=0)
    active_stack = metallic_stack | erased_stack
    first_metallic = first_true_index_map(metallic_stack)
    first_erased = first_true_index_map(erased_stack)
    last_metallic = last_true_index_map(metallic_stack)
    last_erased = last_true_index_map(erased_stack)
    first_active = first_true_index_map(active_stack)
    persistence = np.zeros(result.shape, dtype=np.float32)
    for x_index in range(result.shape[0]):
        for y_index in range(result.shape[1]):
            first = first_active[x_index, y_index]
            if not np.isfinite(first):
                persistence[x_index, y_index] = 0.0
                continue
            first_int = int(first)
            possible = max(1, active_stack.shape[0] - first_int)
            persistence[x_index, y_index] = float(np.sum(active_stack[first_int:, x_index, y_index]) / possible)
    active_count = np.sum(active_stack, axis=0).astype(np.int16)
    return {
        "first_metallic_transition": first_metallic,
        "first_erased_transition": first_erased,
        "last_metallic_transition": last_metallic,
        "last_erased_transition": last_erased,
        "metallic_count": result.aggregate_maps["metallic_count"],
        "erased_count": result.aggregate_maps["erased_count"],
        "switching_persistence": persistence,
        "activity_count": active_count,
        "switched_once": (active_count == 1).astype(np.float32),
        "switched_repeatedly": (active_count > 1).astype(np.float32),
    }


def last_true_index_map(stack: np.ndarray) -> np.ndarray:
    mask = np.asarray(stack, dtype=bool)
    out = np.full(mask.shape[1:], np.nan, dtype=np.float32)
    for index in range(mask.shape[0]):
        out[mask[index]] = float(index)
    return out


def compute_transition_level_statistics(
    result: InitialTransitionFeatureResult,
    parameters: SwitchingMechanismParameters,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ef_window = (
        result.parameters.fermi_level_ev + result.parameters.ef_min_ev,
        result.parameters.fermi_level_ev + result.parameters.ef_max_ev,
    )
    feature_window = (result.parameters.feature_min_ev, result.parameters.feature_max_ev)
    for transition in result.transitions:
        state_a = result.loaded_states[transition.before_index]
        state_b = result.loaded_states[transition.after_index]
        a = np.asarray(state_a.data_array.values, dtype=np.float32)
        b = np.asarray(state_b.data_array.values, dtype=np.float32)
        energy_axis = np.asarray(state_a.data_array.coords["eV"].values, dtype=np.float32)
        phi_axis = np.asarray(state_a.data_array.coords["phi"].values, dtype=np.float32)
        total_a = np.nansum(a, axis=(2, 3))
        total_b = np.nansum(b, axis=(2, 3))
        near_a = compute_integrated_intensity(a, energy_axis, phi_axis, ef_window)
        near_b = compute_integrated_intensity(b, energy_axis, phi_axis, ef_window)
        feat_a = compute_integrated_intensity(a, energy_axis, phi_axis, feature_window)
        feat_b = compute_integrated_intensity(b, energy_axis, phi_axis, feature_window)
        drift = estimate_transition_drift(total_a, total_b)
        both = transition.metallic_mask & transition.erased_mask
        rows.append(
            {
                "transition_index": transition.index,
                "from_file": state_a.name,
                "to_file": state_b.name,
                "transition_name": transition.name,
                "metallic_pixels": int(np.count_nonzero(transition.metallic_mask)),
                "erased_pixels": int(np.count_nonzero(transition.erased_mask)),
                "both_pixels": int(np.count_nonzero(both)),
                "stable_pixels": int(np.count_nonzero(transition.stable_mask)),
                "mean_metallicity_score": float(np.nanmean(transition.metallicity_score)),
                "mean_erasure_score": float(np.nanmean(transition.erasure_score)),
                "mean_transition_magnitude": float(np.nanmean(transition.transition_magnitude)),
                "total_intensity_ratio_B_over_A": safe_ratio(np.nansum(total_b), np.nansum(total_a)),
                "near_EF_intensity_ratio_B_over_A": safe_ratio(np.nansum(near_b), np.nansum(near_a)),
                "feature_intensity_ratio_B_over_A": safe_ratio(np.nansum(feat_b), np.nansum(feat_a)),
                "drift_dx": drift["dx"],
                "drift_dy": drift["dy"],
                "alignment_score": drift["alignment_score"],
                "before_alignment_residual": drift["before_residual"],
                "after_alignment_residual": drift["after_residual"],
            }
        )
    return rows


def estimate_transition_drift(map_a: np.ndarray, map_b: np.ndarray) -> dict[str, float]:
    a = finite_fill(np.asarray(map_a, dtype=np.float32), 0.0)
    b = finite_fill(np.asarray(map_b, dtype=np.float32), 0.0)
    a = a - float(np.nanmean(a))
    b = b - float(np.nanmean(b))
    if not np.any(np.isfinite(a)) or not np.any(np.isfinite(b)):
        return {"dx": float("nan"), "dy": float("nan"), "alignment_score": float("nan"), "before_residual": float("nan"), "after_residual": float("nan")}
    corr = signal.correlate2d(b, a, mode="same", boundary="fill", fillvalue=0.0)
    peak = np.unravel_index(int(np.nanargmax(corr)), corr.shape)
    center = (corr.shape[0] // 2, corr.shape[1] // 2)
    dx = float(peak[0] - center[0])
    dy = float(peak[1] - center[1])
    norm = math.sqrt(max(float(np.nansum(a * a) * np.nansum(b * b)), 1e-12))
    score = float(corr[peak] / norm)
    before = normalized_rms(b - a)
    shifted_b = ndimage.shift(b, shift=(-dx, -dy), order=1, mode="nearest")
    after = normalized_rms(shifted_b - a)
    return {"dx": dx, "dy": dy, "alignment_score": score, "before_residual": before, "after_residual": after}


def normalized_rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    scale = float(np.nanstd(finite))
    if scale <= 1e-12:
        scale = 1.0
    return float(math.sqrt(float(np.nanmean(finite * finite))) / scale)


def safe_ratio(numerator: float | np.ndarray, denominator: float | np.ndarray, epsilon: float = 1e-12) -> float:
    num = float(np.asarray(numerator, dtype=np.float64))
    den = float(np.asarray(denominator, dtype=np.float64))
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= epsilon:
        return float("nan")
    return float(num / den)


def compute_file_intensity_statistics(
    result: InitialTransitionFeatureResult,
    parameters: SwitchingMechanismParameters,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ef_window = (
        result.parameters.fermi_level_ev + result.parameters.ef_min_ev,
        result.parameters.fermi_level_ev + result.parameters.ef_max_ev,
    )
    feature_window = (result.parameters.feature_min_ev, result.parameters.feature_max_ev)
    for index, state in enumerate(result.loaded_states):
        values = np.asarray(state.data_array.values, dtype=np.float32)
        energy_axis = np.asarray(state.data_array.coords["eV"].values, dtype=np.float32)
        phi_axis = np.asarray(state.data_array.coords["phi"].values, dtype=np.float32)
        near = compute_integrated_intensity(values, energy_axis, phi_axis, ef_window)
        feature = compute_integrated_intensity(values, energy_axis, phi_axis, feature_window)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "file_index": index,
                "filename": state.name,
                "total_intensity": float(np.nansum(values)),
                "median_intensity": float(np.nanmedian(finite)) if finite.size else float("nan"),
                "near_EF_total_intensity": float(np.nansum(near)),
                "feature_window_total_intensity": float(np.nansum(feature)),
                "high_percentile_intensity": float(np.nanpercentile(finite, 98)) if finite.size else float("nan"),
                "valid_pixel_count": int(np.count_nonzero(np.isfinite(np.nansum(values, axis=(2, 3))))),
            }
        )
    return rows


def compute_threshold_sensitivity(
    result: InitialTransitionFeatureResult,
    parameters: SwitchingMechanismParameters,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    metallic_hits = np.zeros(result.shape, dtype=np.float32)
    erased_hits = np.zeros(result.shape, dtype=np.float32)
    stable_hits = np.zeros(result.shape, dtype=np.float32)
    thresholds = list(parameters.threshold_sweep_percentiles)
    if not thresholds:
        return rows, {
            "metallic_threshold_robustness": metallic_hits,
            "erased_threshold_robustness": erased_hits,
            "stable_threshold_robustness": stable_hits,
        }
    for percentile in thresholds:
        metallic_count = np.zeros(result.shape, dtype=np.int16)
        erased_count = np.zeros(result.shape, dtype=np.int16)
        stable_count = np.zeros(result.shape, dtype=np.int16)
        for transition in result.transitions:
            metallic, erased, stable, threshold_values = classify_transition_map(
                transition.metallicity_score,
                transition.erasure_score,
                transition.transition_magnitude,
                percentile,
                percentile,
                result.parameters.stable_percentile,
                allow_overlap=result.parameters.allow_overlap,
            )
            metallic_count += metallic.astype(np.int16)
            erased_count += erased.astype(np.int16)
            stable_count += stable.astype(np.int16)
            rows.append(
                {
                    "threshold_percentile": float(percentile),
                    "transition_index": transition.index,
                    "metallic_threshold": threshold_values["metallic"],
                    "erasure_threshold": threshold_values["erased"],
                    "stable_threshold": threshold_values["stable"],
                    "metallic_pixels": int(np.count_nonzero(metallic)),
                    "erased_pixels": int(np.count_nonzero(erased)),
                    "both_pixels": int(np.count_nonzero(metallic & erased)),
                    "stable_pixels": int(np.count_nonzero(stable)),
                }
            )
        metallic_hits += (metallic_count > 0).astype(np.float32)
        erased_hits += (erased_count > 0).astype(np.float32)
        stable_hits += (stable_count > 0).astype(np.float32)
    divisor = float(len(thresholds))
    return rows, {
        "metallic_threshold_robustness": metallic_hits / divisor,
        "erased_threshold_robustness": erased_hits / divisor,
        "stable_threshold_robustness": stable_hits / divisor,
    }


def compute_negative_control_scores(
    result: InitialTransitionFeatureResult,
    parameters: SwitchingMechanismParameters,
) -> dict[str, np.ndarray]:
    control_window = (parameters.negative_control_min_ev, parameters.negative_control_max_ev)
    control_scores = []
    for transition in result.transitions:
        state_a = result.loaded_states[transition.before_index]
        state_b = result.loaded_states[transition.after_index]
        energy_axis = np.asarray(state_a.data_array.coords["eV"].values, dtype=np.float32)
        phi_axis = np.asarray(state_a.data_array.coords["phi"].values, dtype=np.float32)
        try:
            a_control = compute_integrated_intensity(np.asarray(state_a.data_array.values, dtype=np.float32), energy_axis, phi_axis, control_window)
            b_control = compute_integrated_intensity(np.asarray(state_b.data_array.values, dtype=np.float32), energy_axis, phi_axis, control_window)
        except ValueError:
            continue
        control_scores.append((b_control - a_control).astype(np.float32))
    if not control_scores:
        empty = np.full(result.shape, np.nan, dtype=np.float32)
        return {
            "negative_control_mean_score": empty,
            "negative_control_max_abs_score": empty,
            "negative_control_activity": empty,
        }
    stack = np.stack(control_scores, axis=0)
    activity_thresholds = np.array([finite_percentile(np.abs(score), 95.0) for score in control_scores], dtype=np.float32)
    activity = np.zeros(result.shape, dtype=np.float32)
    for score, threshold in zip(control_scores, activity_thresholds):
        activity += (np.abs(score) >= threshold).astype(np.float32)
    return {
        "negative_control_mean_score": np.nanmean(stack, axis=0).astype(np.float32),
        "negative_control_max_abs_score": np.nanmax(np.abs(stack), axis=0).astype(np.float32),
        "negative_control_activity": (activity / max(1, len(control_scores))).astype(np.float32),
    }


def compute_permutation_control(
    group_masks: dict[str, np.ndarray],
    parameters: SwitchingMechanismParameters,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(parameters.random_seed)
    rows: list[dict[str, Any]] = []
    for group in ("future metallic", "future erased", "both metallic and erased"):
        mask = np.asarray(group_masks[group], dtype=bool)
        observed = connected_component_summary(mask)
        if parameters.permutation_count == 0 or not np.any(mask):
            shuffled_largest = np.array([], dtype=np.float32)
        else:
            flat = mask.reshape(-1)
            shuffled_largest = np.empty(parameters.permutation_count, dtype=np.float32)
            for index in range(parameters.permutation_count):
                shuffled = rng.permutation(flat).reshape(mask.shape)
                shuffled_largest[index] = connected_component_summary(shuffled)["largest_connected_component_size"]
        shuffled_mean = float(np.nanmean(shuffled_largest)) if shuffled_largest.size else float("nan")
        shuffled_std = float(np.nanstd(shuffled_largest)) if shuffled_largest.size else float("nan")
        observed_largest = float(observed["largest_connected_component_size"])
        rows.append(
            {
                "group": group,
                "observed_largest_connected_component_size": observed_largest,
                "shuffled_mean_largest_connected_component_size": shuffled_mean,
                "shuffled_std_largest_connected_component_size": shuffled_std,
                "largest_component_z_vs_shuffle": (observed_largest - shuffled_mean) / shuffled_std if np.isfinite(shuffled_std) and shuffled_std > 0 else float("nan"),
            }
        )
    return rows


def connected_component_summary(mask: np.ndarray) -> dict[str, float]:
    labels, count = ndimage.label(np.asarray(mask, dtype=bool))
    sizes = [int(np.count_nonzero(labels == label)) for label in range(1, count + 1)]
    return {
        "connected_component_count": float(count),
        "largest_connected_component_size": float(max(sizes) if sizes else 0),
    }


def compute_artifact_diagnostics(
    result: InitialTransitionFeatureResult,
    group_masks: dict[str, np.ndarray],
    spatial_feature_maps: dict[str, np.ndarray],
    transition_level_rows: list[dict[str, Any]],
    file_intensity_rows: list[dict[str, Any]],
    threshold_sensitivity_rows: list[dict[str, Any]],
    threshold_robustness_maps: dict[str, np.ndarray],
    negative_control_maps: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    switching_mask = group_masks["future metallic"] | group_masks["future erased"]
    stable_mask = group_masks["stable"]
    distance_edge = np.asarray(spatial_feature_maps["distance_to_valid_edge"], dtype=np.float32)
    edge_d = cohen_d(distance_edge[switching_mask], distance_edge[stable_mask])
    rows.append(
        {
            "check": "edge_bias",
            "value": edge_d,
            "risk_score": float(np.clip(abs(edge_d) / 1.0, 0.0, 1.0)) if np.isfinite(edge_d) else 0.0,
            "message": "Switching pixels are closer to/farther from valid-image edges than stable pixels." if np.isfinite(edge_d) else "Not enough pixels to estimate edge bias.",
        }
    )
    drift_values = np.array([math.hypot(float(row["drift_dx"]), float(row["drift_dy"])) for row in transition_level_rows], dtype=np.float32)
    max_drift = float(np.nanmax(drift_values)) if drift_values.size else float("nan")
    rows.append(
        {
            "check": "drift_misalignment",
            "value": max_drift,
            "risk_score": float(np.clip(max_drift / 2.0, 0.0, 1.0)) if np.isfinite(max_drift) else 0.0,
            "message": "Large estimated file-to-file drift can create false difference maps.",
        }
    )
    alignment_scores = np.array([float(row["alignment_score"]) for row in transition_level_rows], dtype=np.float32)
    min_alignment = float(np.nanmin(alignment_scores)) if alignment_scores.size else float("nan")
    rows.append(
        {
            "check": "alignment_score",
            "value": min_alignment,
            "risk_score": float(np.clip((0.35 - min_alignment) / 0.35, 0.0, 1.0)) if np.isfinite(min_alignment) else 0.0,
            "message": "Low cross-correlation alignment score suggests possible drift or morphology mismatch.",
        }
    )
    intensity_ratio = np.array([float(row["near_EF_intensity_ratio_B_over_A"]) for row in transition_level_rows], dtype=np.float32)
    metallic_counts = np.array([float(row["metallic_pixels"]) for row in transition_level_rows], dtype=np.float32)
    intensity_corr = safe_pearson(intensity_ratio, metallic_counts)
    rows.append(
        {
            "check": "global_near_EF_intensity_correlation",
            "value": intensity_corr,
            "risk_score": float(np.clip(abs(intensity_corr), 0.0, 1.0)) if np.isfinite(intensity_corr) else 0.0,
            "message": "Metallic counts tracking global near-EF intensity can indicate normalization or detector-scale artifacts.",
        }
    )
    robust_metallic = threshold_robustness_maps.get("metallic_threshold_robustness", np.zeros(result.shape, dtype=np.float32))
    robust_erased = threshold_robustness_maps.get("erased_threshold_robustness", np.zeros(result.shape, dtype=np.float32))
    switching_robust = float(np.nanmean(np.maximum(robust_metallic[switching_mask], robust_erased[switching_mask]))) if np.any(switching_mask) else float("nan")
    rows.append(
        {
            "check": "threshold_robustness",
            "value": switching_robust,
            "risk_score": float(np.clip(1.0 - switching_robust, 0.0, 1.0)) if np.isfinite(switching_robust) else 0.0,
            "message": "Low robustness means labels disappear quickly as percentile thresholds tighten.",
        }
    )
    fake_activity = negative_control_maps.get("negative_control_activity")
    if fake_activity is not None:
        real_activity = np.asarray(result.aggregate_maps["metallic_count"] + result.aggregate_maps["erased_count"], dtype=np.float32)
        fake_corr = safe_pearson(real_activity.reshape(-1), np.asarray(fake_activity, dtype=np.float32).reshape(-1))
    else:
        fake_corr = float("nan")
    rows.append(
        {
            "check": "negative_control_window_similarity",
            "value": fake_corr,
            "risk_score": float(np.clip(abs(fake_corr), 0.0, 1.0)) if np.isfinite(fake_corr) else 0.0,
            "message": "Similarity to a control energy-window map raises artifact risk.",
        }
    )
    low_signal = ~np.asarray(result.valid_mask, dtype=bool)
    overlap = float(np.count_nonzero(switching_mask & low_signal) / max(1, np.count_nonzero(switching_mask)))
    rows.append(
        {
            "check": "invalid_low_signal_overlap",
            "value": overlap,
            "risk_score": float(np.clip(overlap * 3.0, 0.0, 1.0)),
            "message": "Switching labels overlapping invalid/low-signal pixels may be unreliable.",
        }
    )
    return rows


def safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float32).reshape(-1)
    b = np.asarray(second, dtype=np.float32).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    a = a[valid]
    b = b[valid]
    if float(np.nanstd(a)) <= 1e-12 or float(np.nanstd(b)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_summary_verdict(
    result: InitialTransitionFeatureResult,
    parameters: SwitchingMechanismParameters,
    spectral_effect_rows: list[dict[str, Any]],
    spatial_effect_rows: list[dict[str, Any]],
    connected_component_rows: list[dict[str, Any]],
    transition_history_maps: dict[str, np.ndarray],
    transition_level_rows: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    spectral_ranked = sorted(
        [row for row in spectral_effect_rows if np.isfinite(row.get("cohens_d", np.nan))],
        key=lambda row: abs(float(row["cohens_d"])),
        reverse=True,
    )
    spatial_ranked = sorted(
        [row for row in spatial_effect_rows if np.isfinite(row.get("cohens_d", np.nan))],
        key=lambda row: abs(float(row["cohens_d"])),
        reverse=True,
    )
    spectral_score = float(np.clip((abs(float(spectral_ranked[0]["cohens_d"])) if spectral_ranked else 0.0) / 1.0, 0.0, 1.0))
    component_strength = max(
        [float(row.get("fraction_pixels_in_large_components", 0.0)) for row in connected_component_rows if row.get("group") in {"future metallic", "future erased", "both metallic and erased"}],
        default=0.0,
    )
    spatial_effect = abs(float(spatial_ranked[0]["cohens_d"])) if spatial_ranked else 0.0
    spatial_score = float(np.clip(0.6 * min(spatial_effect, 1.5) / 1.5 + 0.4 * component_strength, 0.0, 1.0))
    active_count = np.asarray(transition_history_maps["activity_count"], dtype=np.float32)
    repeated_fraction = float(np.count_nonzero(active_count > 1) / max(1, np.count_nonzero(active_count > 0)))
    transition_counts = np.array([row["metallic_pixels"] + row["erased_pixels"] for row in transition_level_rows], dtype=np.float32)
    specificity = float(np.nanstd(transition_counts) / (np.nanmean(transition_counts) + parameters.epsilon)) if transition_counts.size else 0.0
    history_score = float(np.clip(0.55 * min(specificity, 1.0) + 0.45 * repeated_fraction, 0.0, 1.0))
    artifact_score = float(np.clip(max([float(row.get("risk_score", 0.0)) for row in artifact_rows], default=0.0), 0.0, 1.0))
    top_artifact = sorted(artifact_rows, key=lambda row: float(row.get("risk_score", 0.0)), reverse=True)
    dominant_scores = {
        "spectral": spectral_score,
        "spatial": spatial_score,
        "transition_history": history_score,
    }
    dominant = max(dominant_scores, key=dominant_scores.get)
    interpretation = (
        "Based on the current thresholds and normalization, evidence suggests "
        f"{dominant.replace('_', '-')} structure is the strongest non-artifact diagnostic. "
        "Treat this as a guide for follow-up checks rather than a physics claim."
    )
    if artifact_score >= 0.65:
        interpretation += " Artifact risk is high enough that drift, scaling, threshold, and edge controls should be checked before interpreting the switching maps."
    return {
        "spectral_evidence_score": spectral_score,
        "spectral_evidence_label": evidence_label(spectral_score),
        "spatial_evidence_score": spatial_score,
        "spatial_evidence_label": evidence_label(spatial_score),
        "transition_history_evidence_score": history_score,
        "transition_history_evidence_label": evidence_label(history_score),
        "artifact_risk_score": artifact_score,
        "artifact_risk_label": risk_label(artifact_score),
        "top_spectral_features": spectral_ranked[:5],
        "top_spatial_features": spatial_ranked[:5],
        "top_history_findings": [
            {"finding": "repeat_switching_fraction", "value": repeated_fraction},
            {"finding": "transition_specificity_cv", "value": specificity},
        ],
        "top_artifact_warnings": top_artifact[:5],
        "current_thresholds": {
            "metallic_percentile": result.parameters.metallic_percentile,
            "erasure_percentile": result.parameters.erasure_percentile,
            "stable_percentile": result.parameters.stable_percentile,
        },
        "current_normalization_mode": result.parameters.normalization_mode,
        "file_sequence": result.file_paths,
        "transition_mode": result.parameters.transition_mode,
        "interpretation": interpretation,
    }


def evidence_label(score: float) -> str:
    if score >= 0.67:
        return "strong"
    if score >= 0.34:
        return "moderate"
    return "weak"


def risk_label(score: float) -> str:
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "moderate"
    return "low"


def selected_pixel_transition_timeline_rows(
    result: SwitchingMechanismDiagnosticsResult,
    selected_pixel: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    transition_result = result.transition_result
    if selected_pixel is None:
        activity = np.asarray(transition_result.aggregate_maps["metallic_count"] + transition_result.aggregate_maps["erased_count"], dtype=np.float32)
        selected_pixel = divmod(int(np.nanargmax(activity)), activity.shape[1]) if activity.size else (0, 0)
    x_index = min(max(0, int(selected_pixel[0])), transition_result.shape[0] - 1)
    y_index = min(max(0, int(selected_pixel[1])), transition_result.shape[1] - 1)
    level_by_index = {int(row["transition_index"]): row for row in result.transition_level_rows}
    rows: list[dict[str, Any]] = []
    for transition in transition_result.transitions:
        transition_stats = level_by_index.get(transition.index, {})
        rows.append(
            {
                "x": x_index,
                "y": y_index,
                "transition_index": transition.index,
                "from_file": transition_result.loaded_states[transition.before_index].name,
                "to_file": transition_result.loaded_states[transition.after_index].name,
                "metallicity_score": float(transition.metallicity_score[x_index, y_index]),
                "erasure_score": float(transition.erasure_score[x_index, y_index]),
                "transition_magnitude": float(transition.transition_magnitude[x_index, y_index]),
                "metallic": bool(transition.metallic_mask[x_index, y_index]),
                "erased": bool(transition.erased_mask[x_index, y_index]),
                "stable": bool(transition.stable_mask[x_index, y_index]),
                "global_intensity_ratio": transition_stats.get("total_intensity_ratio_B_over_A", float("nan")),
                "near_EF_intensity_ratio": transition_stats.get("near_EF_intensity_ratio_B_over_A", float("nan")),
                "drift_dx": transition_stats.get("drift_dx", float("nan")),
                "drift_dy": transition_stats.get("drift_dy", float("nan")),
                "alignment_score": transition_stats.get("alignment_score", float("nan")),
            }
        )
    return rows


def boundary_distance_feature_rows(result: SwitchingMechanismDiagnosticsResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    transition_result = result.transition_result
    x_size, y_size = transition_result.shape
    groups = result.group_masks
    features = result.spatial_feature_maps
    for x_index in range(x_size):
        for y_index in range(y_size):
            labels = [group for group, mask in groups.items() if bool(mask[x_index, y_index])]
            rows.append(
                {
                    "x": x_index,
                    "y": y_index,
                    "group_labels": "; ".join(labels),
                    "metallic_count": int(transition_result.aggregate_maps["metallic_count"][x_index, y_index]),
                    "erased_count": int(transition_result.aggregate_maps["erased_count"][x_index, y_index]),
                    "distance_to_domain_boundary": float(features["distance_to_domain_boundary"][x_index, y_index]),
                    "distance_to_valid_edge": float(features["distance_to_valid_edge"][x_index, y_index]),
                    "local_intensity_gradient": float(features["local_intensity_gradient"][x_index, y_index]),
                    "local_neighborhood_std": float(features["local_neighborhood_std"][x_index, y_index]),
                    "local_contrast_texture": float(features["local_contrast_texture"][x_index, y_index]),
                }
            )
    return rows


def export_switching_mechanism_diagnostics(
    result: SwitchingMechanismDiagnosticsResult,
    output_dir: str | Path,
    selected_pixel: tuple[int, int] | None = None,
) -> dict[str, Path]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    spectral_path = output_path / "spectral_diagnostics.csv"
    spatial_path = output_path / "spatial_diagnostics.csv"
    history_path = output_path / "transition_history_diagnostics.csv"
    artifact_path = output_path / "artifact_diagnostics.csv"
    summary_path = output_path / "summary_verdict.json"
    selected_path = output_path / "selected_pixel_transition_timeline.csv"
    threshold_path = output_path / "threshold_sensitivity_results.csv"
    boundary_path = output_path / "boundary_distance_features.csv"
    write_rows_to_csv(spectral_path, result.spectral_effect_rows)
    write_rows_to_csv(spatial_path, result.spatial_effect_rows + result.connected_component_rows)
    write_rows_to_csv(history_path, result.transition_level_rows)
    write_rows_to_csv(artifact_path, result.artifact_rows + result.file_intensity_rows + result.permutation_control_rows)
    write_rows_to_csv(selected_path, selected_pixel_transition_timeline_rows(result, selected_pixel))
    write_rows_to_csv(threshold_path, result.threshold_sensitivity_rows)
    write_rows_to_csv(boundary_path, boundary_distance_feature_rows(result))
    summary_path.write_text(json.dumps(result.summary_verdict, indent=2), encoding="utf-8")
    return {
        "spectral": spectral_path,
        "spatial": spatial_path,
        "transition_history": history_path,
        "artifact": artifact_path,
        "summary": summary_path,
        "selected_pixel": selected_path,
        "threshold_sensitivity": threshold_path,
        "boundary_distance": boundary_path,
    }


def _binary_neighbor_touch_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    near_b = ndimage.binary_dilation(mask_b, structure=structure)
    near_a = ndimage.binary_dilation(mask_a, structure=structure)
    return (mask_a & near_b) | (mask_b & near_a)


def _distance_to_binary_mask(target_mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    target_mask = np.asarray(target_mask, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    out = np.full(valid_mask.shape, fill_value=np.nan, dtype=np.float32)
    if np.any(target_mask):
        out[valid_mask] = ndimage.distance_transform_edt(~target_mask)[valid_mask].astype(np.float32)
    return out


def _pearson_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    valid = np.isfinite(arr_a) & np.isfinite(arr_b)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if int(np.count_nonzero(valid)) < 3:
        return float("nan")
    vec_a = arr_a[valid].astype(np.float64)
    vec_b = arr_b[valid].astype(np.float64)
    if float(np.std(vec_a)) <= 1e-12 or float(np.std(vec_b)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(vec_a, vec_b)[0, 1])


def _minmax_normalize(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    out = np.full(arr.shape, fill_value=np.nan, dtype=np.float32)
    if not np.any(valid):
        return out
    low = float(np.nanmin(arr[valid]))
    high = float(np.nanmax(arr[valid]))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        out[valid] = 0.0
    else:
        out[valid] = np.clip((arr[valid] - low) / (high - low), 0.0, 1.0)
    return out


def _energy_window_mask(energy_axis: np.ndarray, lower: float, upper: float) -> np.ndarray:
    low = min(float(lower), float(upper))
    high = max(float(lower), float(upper))
    return (energy_axis >= low) & (energy_axis <= high)


def _integrate_along_axis(values: np.ndarray, coords: np.ndarray, axis: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.size <= 1:
        return np.sum(values, axis=axis)
    return np.trapezoid(values, x=coords, axis=axis)


def _integrate_window(values: np.ndarray, energy_axis: np.ndarray, mask: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("Cannot integrate over an empty energy window.")
    return _integrate_along_axis(values[:, :, indices], energy_axis[indices], axis=2)


def _peak_energy_map(values: np.ndarray, energy_axis: np.ndarray, mask: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("Cannot find a peak inside an empty energy window.")
    window = values[:, :, indices]
    peak_positions = np.nanargmax(window, axis=2)
    return energy_axis[indices][peak_positions]


def _leading_edge_map(
    values: np.ndarray,
    energy_axis: np.ndarray,
    mask: np.ndarray,
    fermi_level: float,
) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("Cannot calculate leading edges inside an empty energy window.")
    e_window = energy_axis[indices]
    flat = values[:, :, indices].reshape(-1, indices.size)
    out = np.full(flat.shape[0], fill_value=np.nan, dtype=np.float32)

    for row_index, profile in enumerate(flat):
        finite = np.isfinite(profile)
        if not np.any(finite):
            continue
        clean = np.asarray(profile, dtype=np.float32)
        baseline = float(np.nanpercentile(clean[finite], 10))
        peak = float(np.nanmax(clean[finite]))
        if not np.isfinite(peak) or peak <= baseline:
            nearest = int(np.nanargmin(np.abs(e_window - fermi_level)))
            out[row_index] = float(e_window[nearest])
            continue
        half_height = baseline + 0.5 * (peak - baseline)
        centered = clean - half_height
        crossing_energies: list[float] = []
        for index in range(len(e_window) - 1):
            y0 = centered[index]
            y1 = centered[index + 1]
            if not np.isfinite(y0) or not np.isfinite(y1):
                continue
            if y0 == 0:
                crossing_energies.append(float(e_window[index]))
            elif y0 * y1 <= 0 and clean[index + 1] >= clean[index]:
                fraction = abs(y0) / (abs(y0) + abs(y1) + 1e-12)
                crossing_energies.append(float(e_window[index] + fraction * (e_window[index + 1] - e_window[index])))
        if crossing_energies:
            out[row_index] = min(crossing_energies, key=lambda value: abs(value - fermi_level))
        else:
            nearest = int(np.nanargmin(np.abs(centered) + 0.05 * np.abs(e_window - fermi_level)))
            out[row_index] = float(e_window[nearest])

    return out.reshape(values.shape[:2])


def _fwhm_map(values: np.ndarray, energy_axis: np.ndarray, mask: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("Cannot calculate FWHM inside an empty energy window.")
    e_window = energy_axis[indices]
    flat = values[:, :, indices].reshape(-1, indices.size)
    fallback_width = float(abs(e_window[-1] - e_window[0])) if e_window.size > 1 else 0.0
    out = np.full(flat.shape[0], fill_value=fallback_width, dtype=np.float32)

    for row_index, profile in enumerate(flat):
        finite = np.isfinite(profile)
        if not np.any(finite):
            continue
        clean = np.asarray(profile, dtype=np.float32)
        baseline = float(np.nanpercentile(clean[finite], 10))
        peak_index = int(np.nanargmax(clean))
        peak = float(clean[peak_index])
        if not np.isfinite(peak) or peak <= baseline:
            continue
        half_height = baseline + 0.5 * (peak - baseline)

        left_energy: float | None = None
        for index in range(peak_index - 1, -1, -1):
            y0 = clean[index] - half_height
            y1 = clean[index + 1] - half_height
            if y0 * y1 <= 0:
                fraction = abs(y0) / (abs(y0) + abs(y1) + 1e-12)
                left_energy = float(e_window[index] + fraction * (e_window[index + 1] - e_window[index]))
                break

        right_energy: float | None = None
        for index in range(peak_index, len(e_window) - 1):
            y0 = clean[index] - half_height
            y1 = clean[index + 1] - half_height
            if y0 * y1 <= 0:
                fraction = abs(y0) / (abs(y0) + abs(y1) + 1e-12)
                right_energy = float(e_window[index] + fraction * (e_window[index + 1] - e_window[index]))
                break

        if left_energy is not None and right_energy is not None:
            out[row_index] = float(abs(right_energy - left_energy))
        else:
            above = np.flatnonzero(clean >= half_height)
            if above.size >= 2:
                out[row_index] = float(abs(e_window[above[-1]] - e_window[above[0]]))

    return out.reshape(values.shape[:2])


def _gradient_magnitude(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    filled = finite_fill(arr, float(np.nanmedian(arr[np.isfinite(arr)])) if np.any(np.isfinite(arr)) else 0.0)
    grad_x, grad_y = np.gradient(filled)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y).astype(np.float32)


def _metal_insulating_boundary_mask(code_map: np.ndarray) -> np.ndarray:
    padded = np.pad(code_map, 1, mode="constant", constant_values=-1)
    neighbors = [
        padded[0:-2, 0:-2],
        padded[0:-2, 1:-1],
        padded[0:-2, 2:],
        padded[1:-1, 0:-2],
        padded[1:-1, 2:],
        padded[2:, 0:-2],
        padded[2:, 1:-1],
        padded[2:, 2:],
    ]
    insulating_neighbor = np.logical_or.reduce([neighbor == 1 for neighbor in neighbors])
    metallic_neighbor = np.logical_or.reduce([neighbor == 2 for neighbor in neighbors])
    return insulating_neighbor & metallic_neighbor


def _robust_percentile(
    values: np.ndarray,
    quantile: float,
    mask: np.ndarray | None = None,
    fallback: float = 0.0,
) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(fallback)
    return float(np.nanpercentile(finite, 100.0 * float(quantile)))


def _robust_normalize(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    low = _robust_percentile(arr, 0.02, mask, fallback=0.0)
    high = _robust_percentile(arr, 0.98, mask, fallback=1.0)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        high = low + 1.0
    out = np.clip((arr - low) / (high - low), 0.0, 1.0).astype(np.float32)
    out[~np.isfinite(arr)] = np.nan
    if mask is not None:
        out[~np.asarray(mask, dtype=bool)] = np.nan
    return out


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.asarray(a, dtype=np.float32) / (np.asarray(b, dtype=np.float32) + eps)


def normalize_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    totals = values.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(values, dtype=np.float32)
    np.divide(values, totals, out=normalized, where=totals > eps)
    normalized[totals[:, 0] <= eps] = values[totals[:, 0] <= eps]
    return normalized


def finite_fill(values: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    out = np.array(values, copy=True)
    out[~np.isfinite(out)] = fill_value
    return out


def robust_zscore(values: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    median = np.nanmedian(values, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(values - median), axis=axis, keepdims=True)
    return (values - median) / (1.4826 * mad + eps)


def fit_pca(values: np.ndarray, n_components: int = 8) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    n_samples, n_features = values.shape
    n_components = max(1, min(int(n_components), n_samples, n_features))
    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components].astype(np.float32)
    explained = (singular_values ** 2) / max(1, n_samples - 1)
    explained_total = float(explained.sum())
    if explained_total <= 1e-12 or not np.isfinite(explained_total):
        explained_ratio = np.zeros(n_components, dtype=np.float32)
    else:
        explained_ratio = (explained / explained_total)[:n_components].astype(np.float32)
    return {
        "mean": mean.astype(np.float32),
        "components": components,
        "explained_ratio": explained_ratio,
    }


def transform_pca(values: np.ndarray, pca_fit: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centered = values - np.asarray(pca_fit["mean"], dtype=np.float32)
    return centered @ np.asarray(pca_fit["components"], dtype=np.float32).T


def kmeans(
    values: np.ndarray,
    k: int = 6,
    n_iter: int = 100,
    n_init: int = 12,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(values, dtype=np.float32)
    n_samples = values.shape[0]
    if n_samples == 0:
        raise ValueError("K-means requires at least one sample.")

    k = max(1, min(int(k), n_samples))
    rng = np.random.default_rng(seed)

    best_inertia: float | None = None
    best_labels: np.ndarray | None = None
    best_centroids: np.ndarray | None = None

    for _ in range(max(1, int(n_init))):
        initial_indices = rng.choice(n_samples, size=k, replace=False)
        centroids = values[initial_indices].copy()

        for _ in range(max(1, int(n_iter))):
            distances = squared_euclidean_distances(values, centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = centroids.copy()
            for cluster_id in range(k):
                mask = labels == cluster_id
                if np.any(mask):
                    new_centroids[cluster_id] = values[mask].mean(axis=0)
                else:
                    new_centroids[cluster_id] = values[rng.integers(0, n_samples)]

            if np.allclose(new_centroids, centroids, atol=1e-5):
                centroids = new_centroids
                break
            centroids = new_centroids

        distances = squared_euclidean_distances(values, centroids)
        labels = np.argmin(distances, axis=1)
        inertia = float(np.sum(distances[np.arange(n_samples), labels]))

        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    if best_inertia is None or best_labels is None or best_centroids is None:
        raise RuntimeError("K-means did not produce a valid solution.")

    return best_labels.astype(int), best_centroids.astype(np.float32), best_inertia


def squared_euclidean_distances(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.sum((values[:, None, :] - centroids[None, :, :]) ** 2, axis=2)


def order_clusters_by_mean_ef_fraction(
    raw_cluster_maps: list[np.ndarray],
    features_by_state: list[dict[str, np.ndarray]],
    valid_mask: np.ndarray,
) -> tuple[dict[int, int], dict[int, float]]:
    raw_ids = sorted({int(label) for cluster_map in raw_cluster_maps for label in np.unique(cluster_map[valid_mask])})
    means: list[tuple[int, float]] = []
    for raw_id in raw_ids:
        ef_values: list[np.ndarray] = []
        for cluster_map, features in zip(raw_cluster_maps, features_by_state):
            mask = (cluster_map == raw_id) & valid_mask
            if np.any(mask):
                ef_values.append(features["ef_fraction"][mask])
        mean_ef = float(np.mean(np.concatenate(ef_values))) if ef_values else float("nan")
        means.append((raw_id, mean_ef))

    means.sort(key=lambda item: item[1])
    mapping = {raw_id: ordered_id for ordered_id, (raw_id, _) in enumerate(means)}
    ordered_means = {mapping[raw_id]: mean_ef for raw_id, mean_ef in means}
    return mapping, ordered_means


def remap_cluster_map(cluster_map: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    remapped = np.full_like(cluster_map, fill_value=-1)
    for raw_id, ordered_id in mapping.items():
        remapped[cluster_map == raw_id] = ordered_id
    return remapped


def count_labeled_pixels(label_map: np.ndarray, valid_mask: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(label_map[valid_mask], return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def build_simple_state_maps(
    features_by_state: list[dict[str, np.ndarray]],
    valid_mask: np.ndarray,
    low_quantile: float = 0.30,
    high_quantile: float = 0.70,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[float, float]]:
    ef_values = np.concatenate([features["ef_fraction"][valid_mask].reshape(-1) for features in features_by_state])
    finite_ef_values = ef_values[np.isfinite(ef_values)]
    if finite_ef_values.size == 0:
        finite_ef_values = np.zeros(1, dtype=np.float32)
    low = float(np.quantile(finite_ef_values, low_quantile))
    high = float(np.quantile(finite_ef_values, high_quantile))
    if math.isclose(low, high):
        spread = max(1e-6, float(np.std(finite_ef_values)))
        low -= 0.5 * spread
        high += 0.5 * spread

    label_maps: list[np.ndarray] = []
    code_maps: list[np.ndarray] = []
    for features in features_by_state:
        ef_fraction = features["ef_fraction"]
        labels = np.empty(ef_fraction.shape, dtype=object)
        labels[:] = "intermediate"
        labels[ef_fraction <= low] = "insulating"
        labels[ef_fraction >= high] = "metallic"
        label_maps.append(labels)

        codes = np.full(ef_fraction.shape, fill_value=-1, dtype=int)
        for index, state_name in enumerate(SIMPLE_STATE_NAMES):
            codes[labels == state_name] = index
        code_maps.append(codes)

    return label_maps, code_maps, (low, high)


def build_sequence_maps(
    maps_by_state: list[np.ndarray],
    valid_mask: np.ndarray,
    formatter: Any,
    outside_label: str = "outside-cross",
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]], dict[str, int]]:
    x_size, y_size = valid_mask.shape
    sequence_strings = np.empty((x_size, y_size), dtype=object)
    sequence_strings[:] = outside_label

    for x_index in range(x_size):
        for y_index in range(y_size):
            if valid_mask[x_index, y_index]:
                values = [maps[x_index, y_index] for maps in maps_by_state]
                sequence_strings[x_index, y_index] = formatter(values)

    unique_sequences, counts = np.unique(sequence_strings[valid_mask], return_counts=True)
    order = np.argsort(counts)[::-1]
    ordered_sequences = [str(unique_sequences[index]) for index in order]
    ordered_counts = [int(counts[index]) for index in order]
    sequence_to_code = {sequence: code for code, sequence in enumerate(ordered_sequences)}

    code_map = np.full((x_size, y_size), fill_value=-1, dtype=int)
    for sequence, code in sequence_to_code.items():
        code_map[sequence_strings == sequence] = code

    ranked_sequences = list(zip(ordered_sequences, ordered_counts))
    return sequence_strings, code_map, ranked_sequences, sequence_to_code


def build_summary_dict(result: AnalysisResult, max_sequences: int = 12) -> dict[str, Any]:
    thresholds = result.simple_state_thresholds
    cluster_counts = {
        state_name: {str(cluster_id): count for cluster_id, count in counts.items()}
        for state_name, counts in zip(result.state_names, result.cluster_counts_by_state)
    }

    return {
        "files": result.file_paths,
        "state_names": result.state_names,
        "parameters": asdict(result.parameters),
        "shape": {
            "x": int(result.shape[0]),
            "y": int(result.shape[1]),
        },
        "valid_pixels": int(result.valid_mask.sum()),
        "excluded_pixels": int((~result.valid_mask).sum()),
        "pca_explained_ratio": [float(value) for value in result.pca_explained_ratio.tolist()],
        "cluster_inertia": float(result.cluster_inertia),
        "cluster_mean_ef_fraction": {
            str(cluster_id): float(mean_ef)
            for cluster_id, mean_ef in result.cluster_mean_ef_fraction.items()
        },
        "cluster_counts_by_state": cluster_counts,
        "simple_state_thresholds": {
            "insulating_upper": float(thresholds[0]),
            "metallic_lower": float(thresholds[1]),
        },
        "top_cluster_sequences": [
            {"sequence": sequence, "count": count}
            for sequence, count in result.cluster_sequences[:max_sequences]
        ],
        "top_simple_state_sequences": [
            {"sequence": sequence, "count": count}
            for sequence, count in result.simple_state_sequences[:max_sequences]
        ],
        "notes": list(result.notes),
    }


def export_analysis(result: AnalysisResult, output_dir: str | Path) -> Path:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "summary.json").write_text(
        json.dumps(build_summary_dict(result, max_sequences=20), indent=2),
        encoding="utf-8",
    )
    (output_path / "parameters.json").write_text(
        json.dumps(asdict(result.parameters), indent=2),
        encoding="utf-8",
    )
    (output_path / "cluster_sequence_to_code.json").write_text(
        json.dumps(result.cluster_sequence_to_code, indent=2),
        encoding="utf-8",
    )
    (output_path / "simple_state_sequence_to_code.json").write_text(
        json.dumps(result.simple_state_sequence_to_code, indent=2),
        encoding="utf-8",
    )

    np.save(output_path / "valid_cross_mask.npy", result.valid_mask)
    np.save(output_path / "average_normalized_total_map.npy", result.average_normalized_total_map)
    np.save(output_path / "active_mask.npy", result.active_mask)
    np.save(output_path / "row_occupancy.npy", result.row_occupancy)
    np.save(output_path / "col_occupancy.npy", result.col_occupancy)
    np.save(output_path / "cluster_sequence_code_map.npy", result.cluster_sequence_code_map)
    np.save(output_path / "simple_state_sequence_code_map.npy", result.simple_state_sequence_code_map)

    for index, state in enumerate(result.loaded_states):
        safe_name = sanitize_filename(state.name)
        state_dir = output_path / f"state_{index}_{safe_name}"
        state_dir.mkdir(parents=True, exist_ok=True)

        np.save(state_dir / "total_intensity.npy", result.total_maps[index])
        np.save(state_dir / "near_ef_intensity.npy", result.ef_maps[index])
        np.save(state_dir / "cluster_map.npy", result.cluster_maps[index])
        np.save(state_dir / "raw_cluster_map.npy", result.raw_cluster_maps[index])
        np.save(state_dir / "simple_state_code_map.npy", result.simple_state_code_maps[index])
        (state_dir / "simple_state_labels.json").write_text(
            json.dumps(result.simple_state_label_maps[index].tolist(), indent=2),
            encoding="utf-8",
        )

        for feature_name, feature_map in result.features_by_state[index].items():
            np.save(state_dir / f"{feature_name}.npy", feature_map)

    return output_path


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned or "state"
