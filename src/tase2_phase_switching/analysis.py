from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
import re
from typing import Any

import numpy as np
import xarray as xr


REQUIRED_DIMS = ("x", "y", "eV", "phi")
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


def run_analysis(file_paths: list[str] | tuple[str, ...], parameters: AnalysisParameters | None = None) -> AnalysisResult:
    if parameters is None:
        parameters = AnalysisParameters()
    parameters.validate()

    paths = [str(Path(path).expanduser().resolve()) for path in file_paths]
    if not 1 <= len(paths) <= 4:
        raise ValueError("Please provide between 1 and 4 NetCDF files.")

    loaded_states = [load_state(path) for path in paths]
    reference_shape = loaded_states[0].data_array.shape
    reference_dims = loaded_states[0].data_array.dims

    for state in loaded_states[1:]:
        if state.data_array.shape != reference_shape or state.data_array.dims != reference_dims:
            raise ValueError(
                "All files must share the same canonical dimensions and shape.\n"
                f"Expected dims={reference_dims}, shape={reference_shape}.\n"
                f"Received dims={state.data_array.dims}, shape={state.data_array.shape} for {state.file_path}."
            )

    total_maps: list[np.ndarray] = []
    ef_maps: list[np.ndarray] = []
    features_by_state: list[dict[str, np.ndarray]] = []
    feature_matrices: list[np.ndarray] = []
    feature_names: list[str] | None = None
    notes: list[str] = []

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


def load_state(file_path: str) -> LoadedState:
    resolved = str(Path(file_path).expanduser().resolve())
    dataset = open_nc_dataset(resolved)
    try:
        data_array = prepare_main_dataarray(dataset).load()
    finally:
        dataset.close()

    return LoadedState(
        name=Path(resolved).name,
        file_path=resolved,
        data_array=data_array,
    )


def open_nc_dataset(file_path: str) -> xr.Dataset:
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    engines_to_try: list[str | None] = ["h5netcdf", "scipy", None]

    errors: list[str] = []
    for engine in engines_to_try:
        try:
            if engine is None:
                return xr.open_dataset(file_path)
            return xr.open_dataset(file_path, engine=engine)
        except Exception as exc:  # pragma: no cover - exercised through multiple runtime backends
            label = "default" if engine is None else engine
            errors.append(f"{label}: {exc}")

    joined = "\n".join(errors) if errors else "No engines attempted."
    raise RuntimeError(f"Could not open dataset {file_path}.\n{joined}")


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
        "phi": (("phi", "angle", "angles", "theta", "momentum", "kx", "ky", "k"), ("phi", "angle", "theta", "momentum", "kx", "ky")),
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
    total_intensity = spectra.sum(axis=(1, 2))
    ef_intensity = spectra[:, ef_mask, :].sum(axis=(1, 2))
    wide_intensity = spectra[:, wide_mask, :].sum(axis=(1, 2))

    ef_fraction = safe_divide(ef_intensity, total_intensity)
    wide_fraction = safe_divide(wide_intensity, total_intensity)

    energy_profile = spectra.sum(axis=2)
    phi_profile = spectra.sum(axis=1)

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

    spectra_flat = spectra.reshape(x_size * y_size, -1)
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
    explained_ratio = (explained / explained.sum())[:n_components].astype(np.float32)
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
    low = float(np.quantile(ef_values, low_quantile))
    high = float(np.quantile(ef_values, high_quantile))
    if math.isclose(low, high):
        spread = max(1e-6, float(np.std(ef_values)))
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
