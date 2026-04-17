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

from tase2_phase_switching.analysis import AnalysisParameters, export_analysis, run_analysis


def build_synthetic_dataset(path: Path, state_index: int) -> Path:
    x_size, y_size = 18, 16
    e_size, phi_size = 26, 18

    x = np.arange(x_size, dtype=np.float32)
    y = np.arange(y_size, dtype=np.float32)
    energy = np.linspace(-0.35, 0.12, e_size, dtype=np.float32)
    phi = np.linspace(-1.1, 1.1, phi_size, dtype=np.float32)

    energy_grid, phi_grid = np.meshgrid(energy, phi, indexing="ij")

    insulating = np.exp(-((energy_grid + 0.18) / 0.08) ** 2) * (1.0 - 0.82 * np.exp(-(energy_grid / 0.05) ** 2))
    metallic = (
        0.55 * np.exp(-((energy_grid + 0.12 - 0.10 * phi_grid) / 0.09) ** 2)
        + 0.95 * np.exp(-(energy_grid / 0.045) ** 2) * np.exp(-(phi_grid / 0.35) ** 2)
    )
    intermediate = 0.58 * insulating + 0.42 * metallic

    insulating /= insulating.max()
    metallic /= metallic.max()
    intermediate /= intermediate.max()
    basis = np.stack([insulating, metallic, intermediate], axis=0).astype(np.float32)

    x_mid = x_size // 2
    y_mid = y_size // 2
    horizontal = np.abs(y[None, :] - y_mid) <= 1
    vertical = np.abs(x[:, None] - x_mid) <= 1
    cross = horizontal | vertical

    metallic_weight = (
        0.06
        + 0.10 * cross.astype(np.float32)
        + 0.18 * state_index * horizontal.astype(np.float32)
        + 0.05 * max(state_index - 1, 0) * vertical.astype(np.float32)
    )

    intermediate_weight = (
        0.08
        + 0.18 * vertical.astype(np.float32)
        + 0.07 * state_index * cross.astype(np.float32)
    )

    insulating_weight = 1.0 - metallic_weight - intermediate_weight
    weights = np.stack([insulating_weight, metallic_weight, intermediate_weight], axis=-1)
    weights = np.clip(weights, 0.02, None)
    weights /= weights.sum(axis=-1, keepdims=True)

    cube = np.einsum("xyk,kep->xyep", weights, basis).astype(np.float32)

    amplitude = 0.45 + 0.9 * cross.astype(np.float32) + 0.15 * horizontal.astype(np.float32)
    amplitude += 0.08 * state_index * vertical.astype(np.float32)
    cube *= amplitude[:, :, None, None]

    rng = np.random.default_rng(100 + state_index)
    cube += 0.01 * rng.standard_normal(cube.shape, dtype=np.float32)
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


class AnalysisPipelineTest(unittest.TestCase):
    def test_pipeline_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = [
                build_synthetic_dataset(tmp_path / f"state_{index}.nc", state_index=index)
                for index in range(3)
            ]

            params = AnalysisParameters(
                n_clusters=4,
                n_pca_components=3,
                cross_threshold_quantile=0.40,
                cross_row_fraction=0.15,
                cross_col_fraction=0.15,
            )
            result = run_analysis([str(path) for path in files], params)

            self.assertEqual(result.n_states, 3)
            self.assertEqual(result.shape, (18, 16))
            self.assertGreater(int(result.valid_mask.sum()), 0)
            self.assertEqual(len(result.cluster_maps), 3)
            self.assertEqual(len(result.simple_state_label_maps), 3)
            self.assertTrue(result.cluster_sequences)
            self.assertTrue(result.simple_state_sequences)

            output_dir = export_analysis(result, tmp_path / "exported")
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "cluster_sequence_code_map.npy").exists())
            self.assertTrue((output_dir / "state_0_state_0.nc" / "cluster_map.npy").exists())


if __name__ == "__main__":
    unittest.main()
