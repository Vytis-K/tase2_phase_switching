from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import xarray as xr

from .geometry_inference import (
    PulseOrientationReport,
    ReferenceAxes,
    SimulationGeometry,
    build_flat_geometry,
    cyclic_axis_distance,
    normalize_inside_mask,
)
from .particle_model import CurrentFieldResult, solve_current_field


@dataclass(slots=True)
class TransportParameters:
    contact_width_fraction: float = 0.22
    solver_iterations: int = 280
    voltage: float = 1.0
    particle_count: int = 320
    particle_steps: int = 64
    particle_diffusion: float = 0.08
    conductivity_floor: float = 0.03


@dataclass(slots=True)
class CDWParameters:
    switch_threshold: float = 0.32
    write_gain: float = 0.95
    erase_gain: float = 0.55
    intermediate_gain: float = 0.45
    pinning_strength: float = 0.85
    relaxation: float = 0.07
    geometry_gain: float = 0.38
    boundary_gain: float = 0.55
    diffusion: float = 0.10
    director_gain: float = 0.70
    baseline_metallic: float = 0.05
    baseline_intermediate: float = 0.13


@dataclass(slots=True)
class SequencePulse:
    name: str
    angle_deg: float
    amplitude: float = 1.0
    voltage: float = 1.0
    contact_width_fraction: float = 0.22
    particle_count: int = 320
    particle_steps: int = 64
    particle_diffusion: float = 0.08


@dataclass(slots=True)
class SequenceStepResult:
    name: str
    pulse: SequencePulse | None
    phase_weights: np.ndarray
    director_field: np.ndarray
    observable_map: np.ndarray
    normalized_observable_map: np.ndarray
    conductivity_map: np.ndarray
    current_field: CurrentFieldResult | None


@dataclass(slots=True)
class SequenceSimulationResult:
    geometry: SimulationGeometry
    transport_parameters: TransportParameters
    cdw_parameters: CDWParameters
    steps: list[SequenceStepResult]
    score: float | None = None
    correlations: list[float] | None = None
    rmse_values: list[float] | None = None
    notes: list[str] | None = None

    @property
    def observable_maps(self) -> list[np.ndarray]:
        return [step.observable_map for step in self.steps]

    @property
    def normalized_observable_maps(self) -> list[np.ndarray]:
        return [step.normalized_observable_map for step in self.steps]

    @property
    def state_names(self) -> list[str]:
        return [step.name for step in self.steps]


def build_default_sequence_pulses(
    pulse_a_angle_deg: float = 35.0,
    pulse_b_angle_deg: float = 125.0,
    pulse_a_amplitude: float = 1.0,
    pulse_b_amplitude: float = 0.92,
    repeat_count: int = 1,
    contact_width_fraction: float = 0.22,
    particle_count: int = 320,
) -> list[SequencePulse]:
    pulses = [
        SequencePulse(
            name="pulse_a",
            angle_deg=float(pulse_a_angle_deg),
            amplitude=float(pulse_a_amplitude),
            contact_width_fraction=float(contact_width_fraction),
            particle_count=int(particle_count),
        ),
        SequencePulse(
            name="pulse_b",
            angle_deg=float(pulse_b_angle_deg),
            amplitude=float(pulse_b_amplitude),
            contact_width_fraction=float(contact_width_fraction),
            particle_count=int(particle_count),
        ),
    ]
    for index in range(max(0, int(repeat_count))):
        pulses.append(
            SequencePulse(
                name=f"pulse_b_repeat_{index + 1}",
                angle_deg=float(pulse_b_angle_deg),
                amplitude=float(max(0.55, pulse_b_amplitude * 0.85)),
                contact_width_fraction=float(contact_width_fraction),
                particle_count=int(particle_count),
            )
        )
    return pulses


def simulate_sequence(
    geometry: SimulationGeometry | None = None,
    pulses: list[SequencePulse] | None = None,
    transport_parameters: TransportParameters | None = None,
    cdw_parameters: CDWParameters | None = None,
) -> SequenceSimulationResult:
    if geometry is None:
        geometry = build_flat_geometry()
    if pulses is None:
        pulses = build_default_sequence_pulses()
    if transport_parameters is None:
        transport_parameters = TransportParameters()
    if cdw_parameters is None:
        cdw_parameters = CDWParameters()

    weights, director = initialize_phase_field(geometry, cdw_parameters)
    baseline_observable = observable_from_weights(weights, geometry)
    steps = [
        SequenceStepResult(
            name=geometry.state_names[0] if geometry.state_names else "baseline",
            pulse=None,
            phase_weights=weights.copy(),
            director_field=director.copy(),
            observable_map=baseline_observable.copy(),
            normalized_observable_map=normalize_inside_mask(baseline_observable, geometry.sample_mask),
            conductivity_map=conductivity_from_weights(weights, geometry, transport_parameters, cdw_parameters),
            current_field=None,
        )
    ]

    current_weights = weights
    current_director = director
    for index, pulse in enumerate(pulses, start=1):
        conductivity = conductivity_from_weights(current_weights, geometry, transport_parameters, cdw_parameters)
        current_field = solve_current_field(
            sample_mask=geometry.sample_mask,
            conductivity_map=conductivity,
            angle_deg=pulse.angle_deg,
            contact_width_fraction=pulse.contact_width_fraction,
            iterations=transport_parameters.solver_iterations,
            voltage=pulse.voltage * pulse.amplitude * transport_parameters.voltage,
            particle_count=pulse.particle_count or transport_parameters.particle_count,
            particle_steps=pulse.particle_steps or transport_parameters.particle_steps,
            particle_diffusion=pulse.particle_diffusion or transport_parameters.particle_diffusion,
            seed=index,
        )
        current_weights, current_director = evolve_phase_field(
            phase_weights=current_weights,
            director_field=current_director,
            geometry=geometry,
            current_field=current_field,
            pulse=pulse,
            cdw_parameters=cdw_parameters,
        )
        observable = observable_from_weights(current_weights, geometry)
        steps.append(
            SequenceStepResult(
                name=pulse.name,
                pulse=pulse,
                phase_weights=current_weights.copy(),
                director_field=current_director.copy(),
                observable_map=observable.copy(),
                normalized_observable_map=normalize_inside_mask(observable, geometry.sample_mask),
                conductivity_map=conductivity,
                current_field=current_field,
            )
        )

    score, correlations, rmse_values = compare_with_targets(
        simulated_maps=[step.normalized_observable_map for step in steps],
        target_maps=geometry.target_observable_maps,
        sample_mask=geometry.sample_mask,
    )
    notes: list[str] = []
    if score is not None:
        notes.append(f"Replay score: {score:.3f}")

    return SequenceSimulationResult(
        geometry=geometry,
        transport_parameters=transport_parameters,
        cdw_parameters=cdw_parameters,
        steps=steps,
        score=score,
        correlations=correlations,
        rmse_values=rmse_values,
        notes=notes,
    )


def calibrate_dataset_replay(
    geometry: SimulationGeometry,
    transport_parameters: TransportParameters | None = None,
    cdw_parameters: CDWParameters | None = None,
) -> tuple[list[SequencePulse], SequenceSimulationResult]:
    if transport_parameters is None:
        transport_parameters = TransportParameters()
    if cdw_parameters is None:
        cdw_parameters = CDWParameters()

    pulses = pulses_from_reports(geometry.pulse_reports, transport_parameters.contact_width_fraction, transport_parameters.particle_count)
    if not pulses:
        pulses = build_default_sequence_pulses(contact_width_fraction=transport_parameters.contact_width_fraction)

    best_result: SequenceSimulationResult | None = None
    best_pulses: list[SequencePulse] | None = None
    best_score = float("-inf")

    write_grid = (0.70, 0.95, 1.20)
    erase_grid = (0.30, 0.55, 0.80)
    threshold_grid = (0.24, 0.32, 0.40)

    for write_gain in write_grid:
        for erase_gain in erase_grid:
            for threshold in threshold_grid:
                tuned = replace(
                    cdw_parameters,
                    write_gain=float(write_gain),
                    erase_gain=float(erase_gain),
                    switch_threshold=float(threshold),
                )
                result = simulate_sequence(
                    geometry=geometry,
                    pulses=pulses,
                    transport_parameters=transport_parameters,
                    cdw_parameters=tuned,
                )
                score = result.score if result.score is not None else float("-inf")
                if score > best_score:
                    best_score = score
                    best_result = result
                    best_pulses = pulses

    if best_result is None or best_pulses is None:
        raise RuntimeError("Replay calibration did not produce any simulation result.")
    if best_result.notes is None:
        best_result.notes = []
    best_result.notes.append("Replay parameters were selected by a coarse grid search.")
    return best_pulses, best_result


def render_sequence_to_netcdf(
    result: SequenceSimulationResult,
    output_dir: str | Path,
    e_points: int = 96,
    phi_points: int = 96,
    noise_level: float = 0.005,
) -> list[Path]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    geometry = result.geometry
    axes = compact_reference_axes(geometry.reference_axes, geometry.shape, e_points=e_points, phi_points=phi_points)
    basis = compact_basis_spectra(axes.eV, axes.phi)

    written_paths: list[Path] = []
    for index, step in enumerate(result.steps):
        cube = render_spectral_cube(
            phase_weights=step.phase_weights,
            geometry=geometry,
            basis_spectra=basis,
            noise_level=noise_level,
            seed=index,
        )
        dataset = xr.Dataset(
            {
                "intensity": (("x", "y", "eV", "phi"), cube.astype(np.float32)),
            },
            coords={
                "x": axes.x,
                "y": axes.y,
                "eV": axes.eV,
                "phi": axes.phi,
            },
        )
        safe_name = step.name.replace(" ", "_")
        file_path = output_path / f"{index:02d}_{safe_name}_synthetic.nc"
        dataset.to_netcdf(file_path, engine="h5netcdf")
        written_paths.append(file_path)
    return written_paths


def pulses_from_reports(
    pulse_reports: list[PulseOrientationReport],
    contact_width_fraction: float,
    particle_count: int,
) -> list[SequencePulse]:
    pulses: list[SequencePulse] = []
    for report in pulse_reports:
        amplitude = 0.65 + 2.10 * float(report.strength)
        amplitude = float(np.clip(amplitude, 0.55, 1.60))
        pulses.append(
            SequencePulse(
                name=Path(report.to_state_name).stem,
                angle_deg=float(report.angle_deg),
                amplitude=amplitude,
                contact_width_fraction=float(contact_width_fraction),
                particle_count=int(particle_count),
            )
        )
    return pulses


def initialize_phase_field(
    geometry: SimulationGeometry,
    cdw_parameters: CDWParameters,
) -> tuple[np.ndarray, np.ndarray]:
    mask = geometry.sample_mask
    if geometry.target_observable_maps:
        baseline_target = normalize_inside_mask(geometry.target_observable_maps[0], mask)
        metallic = np.clip(0.04 + 0.48 * baseline_target, 0.02, 0.72)
        intermediate = np.clip(0.10 + 0.28 * (1.0 - np.abs(0.5 - baseline_target) * 2.0), 0.08, 0.42)
    else:
        metallic = np.clip(
            cdw_parameters.baseline_metallic
            + 0.04 * geometry.thickness_map
            + 0.05 * geometry.boundary_pinning_map,
            0.02,
            0.26,
        )
        intermediate = np.clip(
            cdw_parameters.baseline_intermediate
            + 0.04 * (1.0 - geometry.roughness_map),
            0.08,
            0.30,
        )

    insulating = np.clip(1.0 - metallic - intermediate, 0.05, 0.95)
    weights = np.stack([insulating, intermediate, metallic], axis=-1).astype(np.float32)
    weights = normalize_phase_weights(weights, mask)
    director = np.zeros(mask.shape + (2,), dtype=np.float32)
    return weights, director


def conductivity_from_weights(
    phase_weights: np.ndarray,
    geometry: SimulationGeometry,
    transport_parameters: TransportParameters,
    cdw_parameters: CDWParameters,
) -> np.ndarray:
    insulating = phase_weights[..., 0]
    intermediate = phase_weights[..., 1]
    metallic = phase_weights[..., 2]
    conductivity = (
        transport_parameters.conductivity_floor
        + 0.45 * geometry.thickness_map
        + 1.30 * metallic
        + 0.55 * intermediate
        - 0.28 * insulating
        - 0.30 * geometry.roughness_map
        - 0.22 * cdw_parameters.pinning_strength * geometry.boundary_pinning_map * insulating
    )
    return np.where(geometry.sample_mask, np.clip(conductivity, transport_parameters.conductivity_floor, None), 0.0).astype(np.float32)


def evolve_phase_field(
    phase_weights: np.ndarray,
    director_field: np.ndarray,
    geometry: SimulationGeometry,
    current_field: CurrentFieldResult,
    pulse: SequencePulse,
    cdw_parameters: CDWParameters,
) -> tuple[np.ndarray, np.ndarray]:
    mask = geometry.sample_mask
    insulating = phase_weights[..., 0]
    intermediate = phase_weights[..., 1]
    metallic = phase_weights[..., 2]

    current_norm = normalize_inside_mask(current_field.current_magnitude, mask)
    particle_norm = normalize_inside_mask(current_field.particle_density, mask)

    pulse_direction = np.array(current_field.direction_vector, dtype=np.float32)
    director_norm = np.linalg.norm(director_field, axis=-1, keepdims=True)
    director_unit = np.divide(director_field, director_norm + 1e-6)
    alignment = np.clip(0.5 * (1.0 + director_unit[..., 0] * pulse_direction[0] + director_unit[..., 1] * pulse_direction[1]), 0.0, 1.0)
    neutral = (director_norm[..., 0] < 1e-4).astype(np.float32)
    alignment = np.where(neutral > 0, 0.5, alignment)
    misalignment = 1.0 - alignment

    activation = np.clip(
        pulse.amplitude
        * (
            0.62 * current_norm
            + 0.18 * particle_norm
            + cdw_parameters.geometry_gain * geometry.thickness_map
            + 0.10 * (1.0 - geometry.roughness_map)
        ),
        0.0,
        2.0,
    )
    threshold = cdw_parameters.switch_threshold * (
        1.0 + cdw_parameters.pinning_strength * geometry.boundary_pinning_map
    )
    write_drive = 1.0 / (1.0 + np.exp(-6.0 * (activation - threshold)))
    erase_drive = (
        pulse.amplitude
        * misalignment
        * current_norm
        * (cdw_parameters.boundary_gain * geometry.boundary_pinning_map + 0.25)
    )

    write_step = cdw_parameters.write_gain * write_drive * (0.60 * insulating + 0.35 * intermediate)
    intermediate_step = cdw_parameters.intermediate_gain * write_drive * (0.35 * insulating + 0.20 * metallic)
    erase_step = cdw_parameters.erase_gain * erase_drive * metallic

    metallic_new = metallic + write_step - erase_step - 0.03 * cdw_parameters.relaxation * metallic
    intermediate_new = intermediate + intermediate_step + 0.55 * erase_step - 0.25 * write_step
    insulating_new = insulating - write_step - 0.45 * intermediate_step + 0.45 * erase_step

    updated = np.stack([insulating_new, intermediate_new, metallic_new], axis=-1)
    updated = diffuse_phase_weights(updated, mask, strength=cdw_parameters.diffusion)
    updated = normalize_phase_weights(updated, mask)

    director_update = cdw_parameters.director_gain * write_drive * (1.0 - 0.45 * geometry.boundary_pinning_map)
    updated_director = (
        (1.0 - director_update)[..., None] * director_field
        + director_update[..., None] * pulse_direction[None, None, :]
    )
    updated_director = np.where(mask[..., None], updated_director, 0.0).astype(np.float32)
    return updated, updated_director


def observable_from_weights(
    phase_weights: np.ndarray,
    geometry: SimulationGeometry,
) -> np.ndarray:
    insulating = phase_weights[..., 0]
    intermediate = phase_weights[..., 1]
    metallic = phase_weights[..., 2]
    phase_signal = 0.20 * insulating + 0.60 * intermediate + 1.00 * metallic
    amplitude = 0.45 + 0.70 * geometry.thickness_map - 0.18 * geometry.roughness_map
    observable = np.where(geometry.sample_mask, phase_signal * amplitude, 0.0)
    return observable.astype(np.float32)


def diffuse_phase_weights(phase_weights: np.ndarray, sample_mask: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return phase_weights
    weights = np.asarray(phase_weights, dtype=np.float32)
    neighbors = (
        np.roll(weights, 1, axis=0)
        + np.roll(weights, -1, axis=0)
        + np.roll(weights, 1, axis=1)
        + np.roll(weights, -1, axis=1)
    ) / 4.0
    mixed = (1.0 - strength) * weights + strength * neighbors
    mixed = np.where(sample_mask[..., None], mixed, 0.0)
    return mixed.astype(np.float32)


def normalize_phase_weights(phase_weights: np.ndarray, sample_mask: np.ndarray) -> np.ndarray:
    weights = np.where(sample_mask[..., None], np.clip(phase_weights, 1e-5, None), 0.0).astype(np.float32)
    total = weights.sum(axis=-1, keepdims=True)
    weights = np.divide(weights, total + 1e-6)
    weights[~sample_mask] = 0.0
    return weights.astype(np.float32)


def compare_with_targets(
    simulated_maps: list[np.ndarray],
    target_maps: list[np.ndarray],
    sample_mask: np.ndarray,
) -> tuple[float | None, list[float] | None, list[float] | None]:
    if not target_maps:
        return None, None, None

    usable = min(len(simulated_maps), len(target_maps))
    correlations: list[float] = []
    rmse_values: list[float] = []
    for index in range(usable):
        simulated = normalize_inside_mask(simulated_maps[index], sample_mask)[sample_mask]
        target = normalize_inside_mask(target_maps[index], sample_mask)[sample_mask]
        if len(simulated) == 0:
            continue
        simulated_centered = simulated - simulated.mean()
        target_centered = target - target.mean()
        denom = float(np.sqrt(np.sum(simulated_centered**2) * np.sum(target_centered**2)) + 1e-6)
        correlation = float(np.sum(simulated_centered * target_centered) / denom)
        rmse = float(np.sqrt(np.mean((simulated - target) ** 2)))
        correlations.append(correlation)
        rmse_values.append(rmse)

    if not correlations:
        return None, None, None
    score = float(np.mean(correlations) - 0.45 * np.mean(rmse_values))
    return score, correlations, rmse_values


def compact_reference_axes(
    reference_axes: ReferenceAxes | None,
    shape: tuple[int, int],
    e_points: int = 96,
    phi_points: int = 96,
) -> ReferenceAxes:
    if reference_axes is None:
        x_axis = np.arange(shape[0], dtype=np.float32)
        y_axis = np.arange(shape[1], dtype=np.float32)
        e_axis = np.linspace(-0.35, 0.12, int(e_points), dtype=np.float32)
        phi_axis = np.linspace(-1.1, 1.1, int(phi_points), dtype=np.float32)
        return ReferenceAxes(x=x_axis, y=y_axis, eV=e_axis, phi=phi_axis)

    x_axis = reference_axes.x.astype(np.float32)
    y_axis = reference_axes.y.astype(np.float32)
    e_axis = np.linspace(float(reference_axes.eV.min()), float(reference_axes.eV.max()), int(e_points), dtype=np.float32)
    phi_axis = np.linspace(float(reference_axes.phi.min()), float(reference_axes.phi.max()), int(phi_points), dtype=np.float32)
    return ReferenceAxes(x=x_axis, y=y_axis, eV=e_axis, phi=phi_axis)


def compact_basis_spectra(energy_axis: np.ndarray, phi_axis: np.ndarray) -> np.ndarray:
    energy_grid, phi_grid = np.meshgrid(energy_axis, phi_axis, indexing="ij")

    insulating = np.exp(-((energy_grid + 0.18) / 0.08) ** 2) * (1.0 - 0.82 * np.exp(-(energy_grid / 0.05) ** 2))
    metallic = (
        0.55 * np.exp(-((energy_grid + 0.12 - 0.10 * phi_grid) / 0.09) ** 2)
        + 0.95 * np.exp(-(energy_grid / 0.045) ** 2) * np.exp(-(phi_grid / 0.35) ** 2)
    )
    intermediate = 0.58 * insulating + 0.42 * metallic

    def _normalize(spectrum: np.ndarray) -> np.ndarray:
        spectrum = np.clip(spectrum, 0.0, None)
        return spectrum / (float(spectrum.max()) + 1e-6)

    return np.stack(
        [
            _normalize(insulating).astype(np.float32),
            _normalize(intermediate).astype(np.float32),
            _normalize(metallic).astype(np.float32),
        ],
        axis=0,
    )


def render_spectral_cube(
    phase_weights: np.ndarray,
    geometry: SimulationGeometry,
    basis_spectra: np.ndarray,
    noise_level: float = 0.005,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    amplitude = (0.45 + 0.70 * geometry.thickness_map - 0.16 * geometry.roughness_map) * geometry.sample_mask
    cube = np.einsum("xyk,kep->xyep", phase_weights, basis_spectra).astype(np.float32)
    cube *= amplitude[:, :, None, None].astype(np.float32)
    if noise_level > 0.0:
        cube += noise_level * rng.standard_normal(cube.shape, dtype=np.float32)
    cube = np.clip(cube, 0.0, None)
    return cube.astype(np.float32)
