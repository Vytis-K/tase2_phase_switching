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

from tase2_phase_switching.ml_switching_dataset import (
    SwitchingMLDatasetParameters,
    build_switching_ml_dataset,
)
from tase2_phase_switching.ml_switching_train import TrainConfig, train_from_dataset


def build_synthetic_transition_file(path: Path, state_index: int) -> Path:
    x_size, y_size = 12, 10
    e_size, phi_size = 22, 12
    x = np.arange(x_size, dtype=np.float32)
    y = np.arange(y_size, dtype=np.float32)
    energy = np.linspace(-0.35, 0.10, e_size, dtype=np.float32)
    phi = np.linspace(-0.8, 0.8, phi_size, dtype=np.float32)
    energy_grid, phi_grid = np.meshgrid(energy, phi, indexing="ij")

    lhb = np.exp(-((energy_grid + 0.18) / 0.07) ** 2)
    near_ef = np.exp(-(energy_grid / 0.04) ** 2) * np.exp(-(phi_grid / 0.32) ** 2)
    base = 0.85 * lhb + 0.12 * near_ef
    switched = 0.58 * lhb + (0.20 + 0.28 * state_index) * near_ef

    xx, yy = np.meshgrid(x, y, indexing="ij")
    future_region = (xx > 5) & (yy > 3)
    weight = future_region.astype(np.float32) * min(1.0, state_index / 2.0)
    cube = (1.0 - weight[:, :, None, None]) * base[None, None, :, :] + weight[:, :, None, None] * switched[None, None, :, :]
    cube *= (0.8 + 0.02 * xx[:, :, None, None] + 0.01 * yy[:, :, None, None]).astype(np.float32)
    cube = np.clip(cube.astype(np.float32), 0.0, None)

    dataset = xr.Dataset(
        {"intensity": (("x", "y", "eV", "phi"), cube)},
        coords={"x": x, "y": y, "eV": energy, "phi": phi},
    )
    dataset.to_netcdf(path, engine="h5netcdf")
    return path


class MLSwitchingPipelineTest(unittest.TestCase):
    def test_dataset_build_and_training_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = [
                build_synthetic_transition_file(tmp_path / f"state_{index}.nc", state_index=index)
                for index in range(3)
            ]
            dataset_paths = build_switching_ml_dataset(
                files,
                tmp_path / "ml_dataset",
                SwitchingMLDatasetParameters(
                    metallic_percentile=80.0,
                    erasure_percentile=80.0,
                    stable_percentile=25.0,
                    boundary_percentile=75.0,
                ),
            )
            self.assertTrue(dataset_paths["dataset"].exists())
            self.assertTrue(dataset_paths["table"].exists())
            self.assertTrue(dataset_paths["transition_audit"].exists())

            arrays = np.load(dataset_paths["dataset"], allow_pickle=True)
            self.assertIn("X", arrays.files)
            self.assertIn("targets", arrays.files)
            self.assertGreater(arrays["X"].shape[0], 0)
            self.assertGreater(arrays["X"].shape[1], 5)
            self.assertIn("I_rat_A0", [str(name) for name in arrays["feature_names"]])

            trained = train_from_dataset(
                TrainConfig(
                    dataset=dataset_paths["dataset"],
                    output_dir=tmp_path / "model",
                    target="future_active",
                    feature_set="spectral_spatial",
                    epochs=20,
                    spatial_block_size=3,
                    seed=4,
                )
            )
            self.assertTrue(trained["model"].exists())
            self.assertTrue(trained["metrics"].exists())
            self.assertTrue(trained["feature_importance"].exists())
            self.assertTrue((trained["maps"] / "probability_future_active_map.npy").exists())
            self.assertTrue((trained["maps"] / "prediction_vs_actual_future_active.png").exists())


if __name__ == "__main__":
    unittest.main()
