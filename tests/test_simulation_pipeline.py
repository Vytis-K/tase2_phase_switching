from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simulations.cdw_model import (
    CDWParameters,
    SequencePulse,
    TransportParameters,
    render_sequence_to_netcdf,
    simulate_sequence,
)
from simulations.geometry_inference import GeometryInferenceParameters, cyclic_axis_distance, infer_geometry_from_files


def build_switching_dataset(path: Path, stage_index: int) -> Path:
    x_size, y_size = 20, 18
    e_size, phi_size = 24, 16

    x = np.arange(x_size, dtype=np.float32)
    y = np.arange(y_size, dtype=np.float32)
    energy = np.linspace(-0.34, 0.12, e_size, dtype=np.float32)
    phi = np.linspace(-1.05, 1.05, phi_size, dtype=np.float32)
    energy_grid, phi_grid = np.meshgrid(energy, phi, indexing="ij")

    insulating = np.exp(-((energy_grid + 0.18) / 0.08) ** 2) * (1.0 - 0.80 * np.exp(-(energy_grid / 0.05) ** 2))
    metallic = (
        0.48 * np.exp(-((energy_grid + 0.12 - 0.09 * phi_grid) / 0.09) ** 2)
        + 0.92 * np.exp(-(energy_grid / 0.045) ** 2) * np.exp(-(phi_grid / 0.32) ** 2)
    )
    intermediate = 0.58 * insulating + 0.42 * metallic
    basis = np.stack([insulating, intermediate, metallic], axis=0).astype(np.float32)
    basis /= basis.max(axis=(1, 2), keepdims=True) + 1e-6

    x_mid = x_size // 2
    y_mid = y_size // 2
    horizontal = np.abs(y[None, :] - y_mid) <= 1
    vertical = np.abs(x[:, None] - x_mid) <= 1
    cross = horizontal | vertical

    metallic_weight = 0.05 + 0.05 * cross.astype(np.float32)
    intermediate_weight = 0.12 + 0.08 * vertical.astype(np.float32) + np.zeros_like(cross, dtype=np.float32)
    if stage_index >= 1:
        metallic_weight += 0.28 * horizontal.astype(np.float32)
        intermediate_weight += 0.08 * horizontal.astype(np.float32)
    if stage_index >= 2:
        metallic_weight += 0.20 * vertical.astype(np.float32)
        metallic_weight -= 0.10 * horizontal.astype(np.float32)
        intermediate_weight += 0.15 * vertical.astype(np.float32)

    insulating_weight = 1.0 - metallic_weight - intermediate_weight
    weights = np.stack([insulating_weight, intermediate_weight, metallic_weight], axis=-1)
    weights = np.clip(weights, 0.02, None)
    weights /= weights.sum(axis=-1, keepdims=True)

    amplitude = 0.35 + 0.95 * cross.astype(np.float32)
    amplitude += 0.18 * stage_index * horizontal.astype(np.float32)
    amplitude += 0.12 * max(stage_index - 1, 0) * vertical.astype(np.float32)

    cube = np.einsum("xyk,kep->xyep", weights, basis).astype(np.float32)
    cube *= amplitude[:, :, None, None]

    rng = np.random.default_rng(200 + stage_index)
    cube += 0.008 * rng.standard_normal(cube.shape, dtype=np.float32)
    cube = np.clip(cube, 0.0, None)

    dataset = xr.Dataset(
        {
            "intensity": (("x", "y", "eV", "phi"), cube),
        },
        coords={
            "x": x,
            "y": y,
            "eV": energy,
            "phi": phi,
        },
    )
    dataset.to_netcdf(path, engine="h5netcdf")
    return path


class SimulationPipelineTest(unittest.TestCase):
    def test_geometry_simulation_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = [
                build_switching_dataset(tmp_path / f"state_{index}.nc", stage_index=index)
                for index in range(3)
            ]

            inference = infer_geometry_from_files(
                [str(path) for path in files],
                GeometryInferenceParameters(
                    cross_threshold_quantile=0.40,
                    cross_row_fraction=0.15,
                    cross_col_fraction=0.15,
                ),
            )
            self.assertEqual(inference.geometry.shape, (20, 18))
            self.assertGreater(int(inference.geometry.sample_mask.sum()), 0)
            self.assertEqual(len(inference.geometry.pulse_reports), 2)
            for report in inference.geometry.pulse_reports:
                self.assertGreaterEqual(report.angle_deg, 0.0)
                self.assertLess(report.angle_deg, 180.0)
            self.assertGreaterEqual(
                cyclic_axis_distance(
                    inference.geometry.pulse_reports[0].angle_deg,
                    inference.geometry.pulse_reports[1].angle_deg,
                ),
                15.0,
            )

            pulses = [
                SequencePulse(name="state_1", angle_deg=float(inference.geometry.pulse_reports[0].angle_deg), amplitude=1.0),
                SequencePulse(name="state_2", angle_deg=float(inference.geometry.pulse_reports[1].angle_deg), amplitude=0.9),
            ]
            result = simulate_sequence(
                geometry=inference.geometry,
                pulses=pulses,
                transport_parameters=TransportParameters(
                    solver_iterations=90,
                    particle_count=80,
                    particle_steps=24,
                ),
                cdw_parameters=CDWParameters(
                    switch_threshold=0.28,
                    write_gain=0.85,
                    erase_gain=0.45,
                ),
            )

            self.assertEqual(len(result.steps), 3)
            weights = result.steps[-1].phase_weights[inference.geometry.sample_mask]
            self.assertTrue(np.allclose(weights.sum(axis=1), 1.0, atol=1e-4))
            self.assertTrue(np.isfinite(result.steps[-1].observable_map).all())

            export_dir = tmp_path / "synthetic_export"
            written_paths = render_sequence_to_netcdf(result, export_dir, e_points=12, phi_points=10, noise_level=0.0)
            self.assertEqual(len(written_paths), 3)
            self.assertTrue(all(path.exists() for path in written_paths))

            ds = xr.open_dataset(written_paths[0], engine="h5netcdf")
            try:
                self.assertEqual(tuple(ds["intensity"].dims), ("x", "y", "eV", "phi"))
                self.assertEqual(ds["intensity"].shape, (20, 18, 12, 10))
            finally:
                ds.close()


if __name__ == "__main__":
    unittest.main()
