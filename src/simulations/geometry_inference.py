from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from tase2_phase_switching.analysis import (
    build_cross_mask_from_maps,
    get_energy_mask,
    open_nc_dataset,
    prepare_main_dataarray,
)


@dataclass(slots=True)
class GeometryInferenceParameters:
    fermi_level_ev: float = 0.0
    ef_window_ev: float = 0.05
    cross_threshold_quantile: float = 0.45
    cross_row_fraction: float = 0.18
    cross_col_fraction: float = 0.18
    cross_background_quantile: float = 0.10
    cross_pad: int = 1
    smoothing_sigma: float = 1.2


@dataclass(slots=True)
class ReferenceAxes:
    x: np.ndarray
    y: np.ndarray
    eV: np.ndarray
    phi: np.ndarray


@dataclass(slots=True)
class StateMapSummary:
    name: str
    file_path: str
    total_map: np.ndarray
    ef_map: np.ndarray
    ef_fraction_map: np.ndarray


@dataclass(slots=True)
class PulseOrientationReport:
    from_state_name: str
    to_state_name: str
    angle_deg: float
    direction_label: str
    strength: float
    center_x: float
    center_y: float
    compared_to_baseline: bool
    orthogonal_to_previous: bool = False

    @property
    def summary(self) -> str:
        angle = f"{self.angle_deg:.1f} deg"
        strength = f"strength={self.strength:.3f}"
        scope = "baseline delta" if self.compared_to_baseline else "step delta"
        return f"{self.to_state_name}: {self.direction_label} ({angle}, {scope}, {strength})"


@dataclass(slots=True)
class SimulationGeometry:
    name: str
    sample_mask: np.ndarray
    thickness_map: np.ndarray
    roughness_map: np.ndarray
    boundary_pinning_map: np.ndarray
    average_total_map: np.ndarray
    average_ef_fraction_map: np.ndarray
    edge_distance_map: np.ndarray
    target_observable_maps: list[np.ndarray] = field(default_factory=list)
    state_names: list[str] = field(default_factory=list)
    pulse_reports: list[PulseOrientationReport] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reference_axes: ReferenceAxes | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.sample_mask.shape)


@dataclass(slots=True)
class GeometryInferenceResult:
    parameters: GeometryInferenceParameters
    geometry: SimulationGeometry
    states: list[StateMapSummary]


def infer_geometry_from_files(
    file_paths: list[str] | tuple[str, ...],
    parameters: GeometryInferenceParameters | None = None,
) -> GeometryInferenceResult:
    if parameters is None:
        parameters = GeometryInferenceParameters()

    paths = [str(Path(path).expanduser().resolve()) for path in file_paths]
    if not paths:
        raise ValueError("Please provide at least one NetCDF file for geometry inference.")

    states, reference_axes = _load_state_maps(paths, parameters)
    total_maps = [state.total_map for state in states]
    ef_fraction_maps = [state.ef_fraction_map for state in states]

    sample_mask, average_total_map, _, _, _ = build_cross_mask_from_maps(
        total_maps,
        threshold_quantile=parameters.cross_threshold_quantile,
        row_fraction=parameters.cross_row_fraction,
        col_fraction=parameters.cross_col_fraction,
        background_quantile=parameters.cross_background_quantile,
        pad=parameters.cross_pad,
    )
    if not np.any(sample_mask):
        raise ValueError("Geometry inference could not find any active sample pixels.")

    average_total_smooth = gaussian_filter(average_total_map.astype(np.float32), sigma=parameters.smoothing_sigma)
    average_ef_fraction_map = np.mean(ef_fraction_maps, axis=0).astype(np.float32)
    average_ef_fraction_map = normalize_inside_mask(average_ef_fraction_map, sample_mask)

    total_norm = normalize_inside_mask(average_total_smooth, sample_mask)
    roughness_seed = np.abs(total_norm - gaussian_filter(total_norm, sigma=max(0.8, parameters.smoothing_sigma)))
    roughness_local = np.hypot(*np.gradient(total_norm))
    roughness_map = normalize_inside_mask(0.6 * roughness_seed + 0.4 * roughness_local, sample_mask)

    sequence_variation = normalize_inside_mask(np.std(ef_fraction_maps, axis=0), sample_mask)
    ef_gradient = normalize_inside_mask(np.hypot(*np.gradient(average_ef_fraction_map)), sample_mask)
    edge_distance = distance_transform_edt(sample_mask)
    edge_distance_norm = normalize_inside_mask(edge_distance, sample_mask)
    edge_term = np.where(sample_mask, 1.0 - edge_distance_norm, 0.0).astype(np.float32)

    thickness_map = np.where(
        sample_mask,
        np.clip(0.45 + 0.75 * total_norm - 0.20 * roughness_map, 0.10, 1.30),
        0.0,
    ).astype(np.float32)

    boundary_pinning_map = np.where(
        sample_mask,
        np.clip(0.25 * edge_term + 0.40 * sequence_variation + 0.35 * ef_gradient, 0.0, 1.0),
        0.0,
    ).astype(np.float32)

    pulse_reports: list[PulseOrientationReport] = []
    notes: list[str] = []
    previous_angle: float | None = None
    baseline = states[0].ef_fraction_map
    for index in range(1, len(states)):
        step_report = infer_pulse_orientation(
            states[index - 1].ef_fraction_map,
            states[index].ef_fraction_map,
            sample_mask,
            from_name=states[index - 1].name,
            to_name=states[index].name,
            compared_to_baseline=False,
            previous_angle=previous_angle,
        )
        pulse_reports.append(step_report)
        previous_angle = step_report.angle_deg

        baseline_report = infer_pulse_orientation(
            baseline,
            states[index].ef_fraction_map,
            sample_mask,
            from_name=states[0].name,
            to_name=states[index].name,
            compared_to_baseline=True,
            previous_angle=None,
        )
        notes.append(
            f"{states[index].name} inferred axis relative to {states[0].name}: "
            f"{baseline_report.direction_label} ({baseline_report.angle_deg:.1f} deg)."
        )

    if len(pulse_reports) >= 2 and pulse_reports[1].orthogonal_to_previous:
        notes.append(
            f"{pulse_reports[0].to_state_name} and {pulse_reports[1].to_state_name} look like different drive directions. "
            f"The second step rotates by about {cyclic_axis_distance(pulse_reports[0].angle_deg, pulse_reports[1].angle_deg):.1f} deg."
        )

    geometry = SimulationGeometry(
        name="Dataset-inferred geometry",
        sample_mask=sample_mask.astype(bool),
        thickness_map=thickness_map,
        roughness_map=roughness_map,
        boundary_pinning_map=boundary_pinning_map,
        average_total_map=average_total_map.astype(np.float32),
        average_ef_fraction_map=average_ef_fraction_map.astype(np.float32),
        edge_distance_map=edge_distance.astype(np.float32),
        target_observable_maps=[state.ef_fraction_map.astype(np.float32) for state in states],
        state_names=[state.name for state in states],
        pulse_reports=pulse_reports,
        notes=notes,
        reference_axes=reference_axes,
    )
    return GeometryInferenceResult(parameters=parameters, geometry=geometry, states=states)


def build_flat_geometry(
    shape: tuple[int, int] = (61, 61),
    thickness: float = 1.0,
    baseline_pinning: float = 0.18,
) -> SimulationGeometry:
    x_size, y_size = int(shape[0]), int(shape[1])
    sample_mask = np.ones((x_size, y_size), dtype=bool)
    thickness_map = np.full((x_size, y_size), fill_value=float(thickness), dtype=np.float32)
    roughness_map = np.zeros((x_size, y_size), dtype=np.float32)
    boundary_pinning_map = np.full((x_size, y_size), fill_value=float(baseline_pinning), dtype=np.float32)
    average_total_map = np.ones((x_size, y_size), dtype=np.float32)
    average_ef_fraction_map = np.full((x_size, y_size), fill_value=0.25, dtype=np.float32)
    edge_distance_map = distance_transform_edt(sample_mask).astype(np.float32)
    return SimulationGeometry(
        name="Flat synthetic geometry",
        sample_mask=sample_mask,
        thickness_map=thickness_map,
        roughness_map=roughness_map,
        boundary_pinning_map=boundary_pinning_map,
        average_total_map=average_total_map,
        average_ef_fraction_map=average_ef_fraction_map,
        edge_distance_map=edge_distance_map,
        state_names=["baseline", "pulse_a", "pulse_b", "pulse_b_repeat"],
    )


def build_gradient_geometry(
    shape: tuple[int, int] = (61, 61),
    gradient_angle_deg: float = 35.0,
    gradient_strength: float = 0.35,
    defect_strength: float = 0.45,
    baseline_pinning: float = 0.22,
) -> SimulationGeometry:
    base = build_flat_geometry(shape=shape, baseline_pinning=baseline_pinning)
    x_grid, y_grid = centered_coordinate_grids(shape)
    theta = math.radians(gradient_angle_deg)
    along = math.cos(theta) * x_grid + math.sin(theta) * y_grid
    along_norm = normalize_inside_mask(along, base.sample_mask)
    dome = np.exp(-2.4 * (x_grid**2 + y_grid**2))
    defect = np.exp(-9.0 * ((x_grid + 0.22) ** 2 + (y_grid - 0.18) ** 2))

    thickness_map = np.clip(0.78 + gradient_strength * along_norm + 0.18 * dome, 0.12, 1.35)
    roughness_map = np.clip(0.08 + defect_strength * defect + 0.10 * (1.0 - dome), 0.0, 1.0)
    boundary_pinning_map = np.clip(
        baseline_pinning + 0.32 * (1.0 - normalize_inside_mask(base.edge_distance_map, base.sample_mask)) + 0.30 * defect,
        0.0,
        1.0,
    )
    average_total_map = thickness_map * (1.08 - 0.22 * roughness_map)
    average_ef_fraction_map = normalize_inside_mask(0.20 + 0.12 * dome - 0.08 * defect + 0.05 * along_norm, base.sample_mask)
    return SimulationGeometry(
        name="Gradient synthetic geometry",
        sample_mask=base.sample_mask,
        thickness_map=thickness_map.astype(np.float32),
        roughness_map=roughness_map.astype(np.float32),
        boundary_pinning_map=boundary_pinning_map.astype(np.float32),
        average_total_map=average_total_map.astype(np.float32),
        average_ef_fraction_map=average_ef_fraction_map.astype(np.float32),
        edge_distance_map=base.edge_distance_map,
        state_names=["baseline", "pulse_a", "pulse_b", "pulse_b_repeat"],
    )


def infer_pulse_orientation(
    from_map: np.ndarray,
    to_map: np.ndarray,
    sample_mask: np.ndarray,
    from_name: str,
    to_name: str,
    compared_to_baseline: bool,
    previous_angle: float | None = None,
) -> PulseOrientationReport:
    delta = np.asarray(to_map, dtype=np.float32) - np.asarray(from_map, dtype=np.float32)
    delta = np.where(sample_mask, delta, 0.0).astype(np.float32)
    weights = np.abs(delta)
    if float(weights.sum()) <= 1e-8:
        return PulseOrientationReport(
            from_state_name=from_name,
            to_state_name=to_name,
            angle_deg=0.0,
            direction_label="weak / unresolved axis",
            strength=0.0,
            center_x=float(sample_mask.shape[0] / 2.0),
            center_y=float(sample_mask.shape[1] / 2.0),
            compared_to_baseline=compared_to_baseline,
            orthogonal_to_previous=False,
        )

    x_index, y_index = np.nonzero(sample_mask)
    masked_weights = weights[sample_mask]
    center_x = float(np.average(x_index, weights=masked_weights))
    center_y = float(np.average(y_index, weights=masked_weights))

    dx = x_index.astype(np.float32) - center_x
    dy = y_index.astype(np.float32) - center_y
    cov_xx = float(np.average(dx * dx, weights=masked_weights))
    cov_yy = float(np.average(dy * dy, weights=masked_weights))
    cov_xy = float(np.average(dx * dy, weights=masked_weights))
    covariance = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float32)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    dominant_vector = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle_deg = float(np.degrees(np.arctan2(dominant_vector[1], dominant_vector[0])) % 180.0)
    direction_label = direction_label_from_angle(angle_deg)

    orthogonal = False
    if previous_angle is not None:
        orthogonal = cyclic_axis_distance(previous_angle, angle_deg) >= 55.0

    return PulseOrientationReport(
        from_state_name=from_name,
        to_state_name=to_name,
        angle_deg=angle_deg,
        direction_label=direction_label,
        strength=float(np.sqrt(np.mean((delta[sample_mask]) ** 2))),
        center_x=center_x,
        center_y=center_y,
        compared_to_baseline=compared_to_baseline,
        orthogonal_to_previous=orthogonal,
    )


def direction_label_from_angle(angle_deg: float) -> str:
    wrapped = angle_deg % 180.0
    if wrapped < 22.5 or wrapped >= 157.5:
        return "sample-x axis"
    if wrapped < 67.5:
        return "rising diagonal"
    if wrapped < 112.5:
        return "sample-y axis"
    return "falling diagonal"


def cyclic_axis_distance(angle_a_deg: float, angle_b_deg: float) -> float:
    diff = abs((float(angle_a_deg) - float(angle_b_deg) + 90.0) % 180.0 - 90.0)
    return float(diff)


def centered_coordinate_grids(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    x_size, y_size = int(shape[0]), int(shape[1])
    x = np.linspace(-1.0, 1.0, x_size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, y_size, dtype=np.float32)
    return np.meshgrid(x, y, indexing="ij")


def normalize_inside_mask(values: np.ndarray, sample_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mask = np.asarray(sample_mask, dtype=bool)
    out = np.zeros_like(arr, dtype=np.float32)
    if not np.any(mask):
        return out
    masked = arr[mask]
    low = float(np.nanmin(masked))
    high = float(np.nanmax(masked))
    if math.isclose(low, high):
        out[mask] = 0.0
        return out
    out[mask] = (masked - low) / (high - low)
    return out


def _load_state_maps(
    file_paths: list[str],
    parameters: GeometryInferenceParameters,
) -> tuple[list[StateMapSummary], ReferenceAxes]:
    states: list[StateMapSummary] = []
    reference_axes: ReferenceAxes | None = None

    for file_path in file_paths:
        resolved = str(Path(file_path).expanduser().resolve())
        dataset = open_nc_dataset(resolved)
        try:
            data_array = prepare_main_dataarray(dataset)
            energy_axis = np.asarray(data_array.coords["eV"].values, dtype=np.float32)
            phi_axis = np.asarray(data_array.coords["phi"].values, dtype=np.float32)
            x_axis = np.asarray(data_array.coords["x"].values, dtype=np.float32)
            y_axis = np.asarray(data_array.coords["y"].values, dtype=np.float32)
            ef_mask = get_energy_mask(
                energy_axis,
                center=parameters.fermi_level_ev,
                halfwidth=parameters.ef_window_ev,
            )
            if not np.any(ef_mask):
                raise ValueError(
                    f"No energy samples were found inside |E - {parameters.fermi_level_ev:.3f}| <= "
                    f"{parameters.ef_window_ev:.3f} eV for {resolved}."
                )

            total_map = np.asarray(data_array.sum(dim=("eV", "phi")).values, dtype=np.float32)
            ef_map = np.asarray(
                data_array.isel(eV=np.flatnonzero(ef_mask)).sum(dim=("eV", "phi")).values,
                dtype=np.float32,
            )
            ef_fraction_map = ef_map / (total_map + 1e-6)
        finally:
            dataset.close()

        if states and total_map.shape != states[0].total_map.shape:
            raise ValueError(
                "All inferred states must share the same x/y shape.\n"
                f"Expected {states[0].total_map.shape}, received {total_map.shape} for {resolved}."
            )

        states.append(
            StateMapSummary(
                name=Path(resolved).name,
                file_path=resolved,
                total_map=total_map.astype(np.float32),
                ef_map=ef_map.astype(np.float32),
                ef_fraction_map=ef_fraction_map.astype(np.float32),
            )
        )
        if reference_axes is None:
            reference_axes = ReferenceAxes(x=x_axis, y=y_axis, eV=energy_axis, phi=phi_axis)

    if reference_axes is None:
        raise RuntimeError("Geometry inference did not load any reference axes.")
    return states, reference_axes
