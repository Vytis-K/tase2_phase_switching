from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(slots=True)
class CurrentFieldResult:
    potential_map: np.ndarray
    current_x: np.ndarray
    current_y: np.ndarray
    current_magnitude: np.ndarray
    particle_density: np.ndarray
    conductivity_map: np.ndarray
    source_mask: np.ndarray
    sink_mask: np.ndarray
    direction_vector: tuple[float, float]
    angle_deg: float


def solve_current_field(
    sample_mask: np.ndarray,
    conductivity_map: np.ndarray,
    angle_deg: float,
    contact_width_fraction: float = 0.22,
    iterations: int = 280,
    voltage: float = 1.0,
    particle_count: int = 320,
    particle_steps: int = 64,
    particle_diffusion: float = 0.08,
    seed: int = 0,
) -> CurrentFieldResult:
    mask = np.asarray(sample_mask, dtype=bool)
    conductivity = np.where(mask, np.clip(np.asarray(conductivity_map, dtype=np.float32), 1e-4, None), 0.0)
    source_mask, sink_mask, direction = build_contact_masks(mask, angle_deg, width_fraction=contact_width_fraction)
    potential = solve_potential(mask, conductivity, source_mask, sink_mask, iterations=iterations)
    current_x, current_y, current_magnitude = compute_current_map(potential, conductivity, mask, voltage=voltage)
    particle_density = trace_particles(
        current_x=current_x,
        current_y=current_y,
        sample_mask=mask,
        source_mask=source_mask,
        sink_mask=sink_mask,
        n_particles=particle_count,
        n_steps=particle_steps,
        diffusion=particle_diffusion,
        seed=seed,
    )
    return CurrentFieldResult(
        potential_map=potential,
        current_x=current_x,
        current_y=current_y,
        current_magnitude=current_magnitude,
        particle_density=particle_density,
        conductivity_map=conductivity,
        source_mask=source_mask,
        sink_mask=sink_mask,
        direction_vector=direction,
        angle_deg=float(angle_deg),
    )


def build_contact_masks(
    sample_mask: np.ndarray,
    angle_deg: float,
    width_fraction: float = 0.22,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    mask = np.asarray(sample_mask, dtype=bool)
    x_size, y_size = mask.shape
    x = np.linspace(-1.0, 1.0, x_size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, y_size, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

    theta = math.radians(float(angle_deg))
    direction = (float(math.cos(theta)), float(math.sin(theta)))
    projection = direction[0] * x_grid + direction[1] * y_grid
    transverse = -direction[1] * x_grid + direction[0] * y_grid

    mask_projection = projection[mask]
    low_cut = float(np.quantile(mask_projection, 0.07))
    high_cut = float(np.quantile(mask_projection, 0.93))
    width = max(0.08, float(width_fraction)) * 0.95
    support = np.abs(transverse) <= width

    source_mask = mask & support & (projection <= low_cut)
    sink_mask = mask & support & (projection >= high_cut)

    if int(source_mask.sum()) == 0:
        source_mask = mask & (projection <= np.quantile(mask_projection, 0.03))
    if int(sink_mask.sum()) == 0:
        sink_mask = mask & (projection >= np.quantile(mask_projection, 0.97))

    return source_mask, sink_mask, direction


def solve_potential(
    sample_mask: np.ndarray,
    conductivity_map: np.ndarray,
    source_mask: np.ndarray,
    sink_mask: np.ndarray,
    iterations: int = 280,
) -> np.ndarray:
    mask = np.asarray(sample_mask, dtype=bool)
    conductivity = np.asarray(conductivity_map, dtype=np.float32)
    potential = np.where(mask, 0.5, 0.0).astype(np.float32)
    potential[source_mask] = 1.0
    potential[sink_mask] = 0.0
    fixed_mask = source_mask | sink_mask | (~mask)

    west_weight = _neighbor_weight(conductivity, mask, axis=0, shift=1)
    east_weight = _neighbor_weight(conductivity, mask, axis=0, shift=-1)
    south_weight = _neighbor_weight(conductivity, mask, axis=1, shift=1)
    north_weight = _neighbor_weight(conductivity, mask, axis=1, shift=-1)
    denominator = west_weight + east_weight + south_weight + north_weight + 1e-6

    for _ in range(max(1, int(iterations))):
        west = np.roll(potential, 1, axis=0)
        east = np.roll(potential, -1, axis=0)
        south = np.roll(potential, 1, axis=1)
        north = np.roll(potential, -1, axis=1)

        updated = (
            west_weight * west
            + east_weight * east
            + south_weight * south
            + north_weight * north
        ) / denominator
        potential = np.where(~fixed_mask, updated, potential)
        potential[source_mask] = 1.0
        potential[sink_mask] = 0.0
        potential[~mask] = 0.0

    return potential.astype(np.float32)


def compute_current_map(
    potential_map: np.ndarray,
    conductivity_map: np.ndarray,
    sample_mask: np.ndarray,
    voltage: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    potential = np.asarray(potential_map, dtype=np.float32) * float(voltage)
    conductivity = np.asarray(conductivity_map, dtype=np.float32)
    mask = np.asarray(sample_mask, dtype=bool)
    grad_x, grad_y = np.gradient(potential)
    current_x = np.where(mask, -conductivity * grad_x, 0.0).astype(np.float32)
    current_y = np.where(mask, -conductivity * grad_y, 0.0).astype(np.float32)
    current_magnitude = np.where(mask, np.hypot(current_x, current_y), 0.0).astype(np.float32)
    return current_x, current_y, current_magnitude


def trace_particles(
    current_x: np.ndarray,
    current_y: np.ndarray,
    sample_mask: np.ndarray,
    source_mask: np.ndarray,
    sink_mask: np.ndarray,
    n_particles: int = 320,
    n_steps: int = 64,
    diffusion: float = 0.08,
    seed: int = 0,
) -> np.ndarray:
    mask = np.asarray(sample_mask, dtype=bool)
    sources = np.argwhere(source_mask)
    density = np.zeros(mask.shape, dtype=np.float32)
    if len(sources) == 0 or n_particles <= 0:
        return density

    rng = np.random.default_rng(seed)
    positions = sources[rng.integers(0, len(sources), size=int(n_particles))].astype(np.float32)
    positions += rng.uniform(-0.4, 0.4, size=positions.shape).astype(np.float32)
    alive = np.ones(len(positions), dtype=bool)

    for _ in range(max(1, int(n_steps))):
        if not np.any(alive):
            break

        integer_pos = np.rint(positions).astype(int)
        inside = (
            (integer_pos[:, 0] >= 0)
            & (integer_pos[:, 0] < mask.shape[0])
            & (integer_pos[:, 1] >= 0)
            & (integer_pos[:, 1] < mask.shape[1])
        )
        alive &= inside
        if not np.any(alive):
            break

        ix = integer_pos[:, 0].clip(0, mask.shape[0] - 1)
        iy = integer_pos[:, 1].clip(0, mask.shape[1] - 1)
        alive &= mask[ix, iy]
        if not np.any(alive):
            break

        speed_x = current_x[ix, iy]
        speed_y = current_y[ix, iy]
        speed = np.hypot(speed_x, speed_y)
        direction_x = np.divide(speed_x, speed + 1e-6)
        direction_y = np.divide(speed_y, speed + 1e-6)
        step_scale = 0.55 + 0.90 * np.tanh(speed)

        positions[:, 0] += step_scale * direction_x + diffusion * rng.standard_normal(len(positions))
        positions[:, 1] += step_scale * direction_y + diffusion * rng.standard_normal(len(positions))

        integer_pos = np.rint(positions).astype(int)
        inside = (
            (integer_pos[:, 0] >= 0)
            & (integer_pos[:, 0] < mask.shape[0])
            & (integer_pos[:, 1] >= 0)
            & (integer_pos[:, 1] < mask.shape[1])
        )
        alive &= inside
        if not np.any(alive):
            break

        ix = integer_pos[:, 0].clip(0, mask.shape[0] - 1)
        iy = integer_pos[:, 1].clip(0, mask.shape[1] - 1)
        alive &= mask[ix, iy]
        density[ix[alive], iy[alive]] += 1.0
        alive &= ~sink_mask[ix, iy]

    if float(density.max()) > 0.0:
        density /= float(density.max())
    return density.astype(np.float32)


def _neighbor_weight(conductivity_map: np.ndarray, sample_mask: np.ndarray, axis: int, shift: int) -> np.ndarray:
    conductivity = np.asarray(conductivity_map, dtype=np.float32)
    mask = np.asarray(sample_mask, dtype=bool)
    shifted_conductivity = np.roll(conductivity, shift=shift, axis=axis)
    shifted_mask = np.roll(mask, shift=shift, axis=axis)
    weight = 0.5 * (conductivity + shifted_conductivity)
    valid = mask & shifted_mask
    weight = np.where(valid, weight, 0.0).astype(np.float32)

    if axis == 0 and shift == 1:
        weight[0, :] = 0.0
    elif axis == 0 and shift == -1:
        weight[-1, :] = 0.0
    elif axis == 1 and shift == 1:
        weight[:, 0] = 0.0
    elif axis == 1 and shift == -1:
        weight[:, -1] = 0.0
    return weight
