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

from tase2_phase_switching.analysis import (
    AnalysisParameters,
    InitialTransitionFeatureParameters,
    SpectralClusterParameters,
    STATE_CLASSIFICATION_FEATURE_NAMES,
    STATE_CLASSIFICATION_LABELS,
    SWITCHING_LABELS,
    TRANSITION_OUTCOME_LABELS,
    StateClassifierParameters,
    StatePredictionParameters,
    SwitchingMechanismParameters,
    SwitchingMapParameters,
    TransitionOutcomeParameters,
    analyze_cluster_physical_interpretation,
    export_analysis,
    export_cluster_physical_interpretation,
    export_initial_transition_feature_analysis,
    export_switching_mechanism_diagnostics,
    export_state_classification,
    export_state_prediction,
    export_switching_map,
    export_transition_outcome_maps,
    initial_state_feature_rows,
    initial_transition_metric_rows,
    run_analysis,
    run_initial_transition_feature_analysis,
    run_switching_mechanism_diagnostics,
    run_spectral_clustering,
    run_state_classification,
    run_state_prediction,
    run_switching_map,
    run_transition_outcome_maps,
    state_classification_table_rows,
    state_prediction_table_rows,
    switching_map_table_rows,
    transition_outcome_table_rows,
)


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


def build_cropped_dataset(source_path: Path, output_path: Path, x_slice: slice, y_slice: slice) -> Path:
    with xr.open_dataset(source_path, engine="h5netcdf") as dataset:
        cropped = dataset["intensity"].isel(x=x_slice, y=y_slice).values
        energy = dataset.coords["eV"].values
        phi = dataset.coords["phi"].values

    cropped_dataset = xr.Dataset(
        {
            "intensity": (("x", "y", "eV", "phi"), cropped),
        },
        coords={
            "x": np.arange(cropped.shape[0], dtype=np.float32),
            "y": np.arange(cropped.shape[1], dtype=np.float32),
            "eV": energy,
            "phi": phi,
        },
    )
    cropped_dataset.to_netcdf(output_path, engine="h5netcdf")
    return output_path


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

    def test_pipeline_aligns_spatially_clipped_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            full_file = build_synthetic_dataset(tmp_path / "state_full.nc", state_index=0)
            full_state_1 = build_synthetic_dataset(tmp_path / "state_full_for_crop.nc", state_index=1)
            cropped_file = build_cropped_dataset(
                full_state_1,
                tmp_path / "state_clipped.nc",
                slice(4, 15),
                slice(3, 14),
            )

            result = run_analysis(
                [str(full_file), str(cropped_file)],
                AnalysisParameters(
                    n_clusters=3,
                    n_pca_components=3,
                    cross_threshold_quantile=0.35,
                    cross_row_fraction=0.15,
                    cross_col_fraction=0.15,
                ),
            )

            self.assertEqual(result.shape, (11, 11))
            self.assertEqual(result.loaded_states[0].data_array.shape[:2], (11, 11))
            self.assertEqual(result.loaded_states[1].data_array.shape[:2], (11, 11))
            self.assertTrue(any("Spatially aligned files" in note for note in result.notes))
            self.assertTrue(any("state_full.nc" in note and "x=4:15" in note and "y=3:14" in note for note in result.notes))

    def test_spectral_cluster_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = [
                build_synthetic_dataset(tmp_path / f"state_{index}.nc", state_index=index)
                for index in range(2)
            ]

            analysis_result = run_analysis(
                [str(path) for path in files],
                AnalysisParameters(
                    n_clusters=4,
                    n_pca_components=3,
                    cross_threshold_quantile=0.40,
                    cross_row_fraction=0.15,
                    cross_col_fraction=0.15,
                ),
            )

            for method_key in ("compressed_features", "downsampled_spectra_pca"):
                with self.subTest(method_key=method_key):
                    cluster_result = run_spectral_clustering(
                        analysis_result.loaded_states[0],
                        analysis_result.valid_mask,
                        feature_maps=analysis_result.features_by_state[0],
                        parameters=SpectralClusterParameters(
                            n_clusters=4,
                            embedding_components=4,
                            method_key=method_key,
                        ),
                        analysis_parameters=analysis_result.parameters,
                    )

                    self.assertEqual(cluster_result.cluster_map.shape, analysis_result.shape)
                    self.assertEqual(len(cluster_result.cluster_stats), 4)
                    self.assertEqual(cluster_result.embedding_2d.shape[0], int(analysis_result.valid_mask.sum()))
                    self.assertEqual(cluster_result.embedding_2d.shape[1], 2)
                    self.assertTrue(np.all(cluster_result.cluster_map[analysis_result.valid_mask] >= 0))
                    self.assertTrue(all(stat.mean_spectrum.ndim == 2 for stat in cluster_result.cluster_stats))
                    self.assertTrue(all(stat.pixel_count > 0 for stat in cluster_result.cluster_stats))

                    interpretation = analyze_cluster_physical_interpretation(cluster_result)
                    self.assertEqual(len(interpretation.metrics_rows), len(cluster_result.cluster_stats))
                    self.assertEqual(len(interpretation.question_summaries), 6)
                    exported = export_cluster_physical_interpretation(interpretation, tmp_path / "cluster_reports")
                    self.assertTrue(exported["metrics"].exists())
                    self.assertTrue(exported["pairwise"].exists())
                    self.assertTrue(exported["summary"].exists())

    def test_state_classifier_features_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            file_path = build_synthetic_dataset(tmp_path / "state_classifier.nc", state_index=1)

            result = run_state_classification(
                file_path,
                StateClassifierParameters(
                    lhb_center_ev=-0.18,
                    lhb_halfwidth_ev=0.07,
                    p3_center_ev=-0.31,
                    p3_halfwidth_ev=0.04,
                ),
            )

            self.assertEqual(result.shape, (18, 16))
            self.assertEqual(tuple(result.feature_maps.keys()), STATE_CLASSIFICATION_FEATURE_NAMES)
            self.assertEqual(
                STATE_CLASSIFICATION_LABELS,
                (
                    "invalid / low signal",
                    "CCDW / insulating",
                    "metastable metallic",
                    "intermediate / erased memory",
                    "boundary / mixed",
                    "structural / orientation variant",
                ),
            )
            self.assertTrue(np.all(np.isfinite(result.feature_maps["T"])))
            self.assertTrue(np.any(result.valid_mask))
            self.assertTrue(set(result.counts).issuperset(STATE_CLASSIFICATION_LABELS))
            self.assertEqual(int(sum(result.counts.values())), 18 * 16)
            for code, label in enumerate(STATE_CLASSIFICATION_LABELS):
                self.assertTrue(np.all(result.label_map[result.code_map == code] == label))

            rows = state_classification_table_rows(result)
            self.assertEqual(len(rows), 18 * 16)
            self.assertIn("I_rat", rows[0])
            self.assertIn("state_label", rows[0])

            exported = export_state_classification(result, tmp_path / "classifier_export")
            self.assertTrue(exported["table"].exists())
            self.assertEqual(exported["table"].name, "clustering_feature_table.csv")
            self.assertTrue(exported["summary"].exists())
            self.assertTrue((exported["feature_maps"] / "I_rat.npy").exists())

    def test_switching_map_sequence_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = [
                build_synthetic_dataset(tmp_path / f"switching_{index}.nc", state_index=index)
                for index in range(3)
            ]

            result = run_switching_map(
                [str(path) for path in files],
                SwitchingMapParameters(
                    lhb_center_ev=-0.18,
                    lhb_halfwidth_ev=0.07,
                    low_switch_quantile=0.25,
                    high_switch_quantile=0.70,
                ),
            )

            self.assertEqual(result.shape, (18, 16))
            self.assertEqual(result.n_states, 3)
            self.assertEqual(len(result.delta_irat_maps), 2)
            self.assertEqual(result.switching_coefficient_map.shape, (18, 16))
            self.assertTrue(np.any(np.isfinite(result.switching_coefficient_map)))
            self.assertTrue(set(result.counts).issuperset(SWITCHING_LABELS))
            self.assertEqual(int(sum(result.counts.values())), 18 * 16)
            for code, label in enumerate(SWITCHING_LABELS):
                self.assertTrue(np.all(result.label_map[result.code_map == code] == label))

            rows = switching_map_table_rows(result)
            self.assertEqual(len(rows), 18 * 16)
            self.assertIn("switching_coefficient", rows[0])
            self.assertIn("file_0_I_rat", rows[0])
            self.assertIn("Delta_Irat_0_to_1", rows[0])

            exported = export_switching_map(result, tmp_path / "switching_export")
            self.assertTrue(exported["table"].exists())
            self.assertEqual(exported["table"].name, "switching_feature_table.csv")
            self.assertTrue(exported["coefficient_map"].exists())
            self.assertTrue(exported["code_map"].exists())
            self.assertTrue((exported["maps"] / "I_rat_maps.npy").exists())

    def test_state_prediction_diagnostics_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = [
                build_synthetic_dataset(tmp_path / f"prediction_{index}.nc", state_index=index)
                for index in range(3)
            ]

            result = run_state_prediction(
                [str(path) for path in files],
                StatePredictionParameters(
                    lhb_center_ev=-0.18,
                    lhb_halfwidth_ev=0.07,
                    p3_center_ev=-0.31,
                    p3_halfwidth_ev=0.04,
                    stable_quantile=0.20,
                    switch_quantile=0.75,
                ),
            )

            self.assertEqual(result.shape, (18, 16))
            self.assertEqual(result.n_states, 3)
            self.assertTrue(set(result.counts).issuperset(SWITCHING_LABELS))
            self.assertEqual(int(sum(result.counts.values())), 18 * 16)
            self.assertIn("I_rat", result.initial_feature_maps)
            self.assertIn("distance_to_edge", result.distance_maps)
            self.assertIn("I_rat_initial", result.correlation_values)
            self.assertIn("distance_to_phase_boundary", result.correlation_values)
            for label in SWITCHING_LABELS:
                self.assertEqual(result.average_initial_edcs[label].shape, result.e_axis.shape)

            rows = state_prediction_table_rows(result)
            self.assertEqual(len(rows), 18 * 16)
            self.assertIn("I_rat_initial", rows[0])
            self.assertIn("switching_coefficient", rows[0])
            self.assertIn("future_outcome_label", rows[0])

            exported = export_state_prediction(result, tmp_path / "state_prediction_export")
            self.assertTrue(exported["table"].exists())
            self.assertEqual(exported["table"].name, "state_prediction_table.csv")
            self.assertTrue(exported["score_map"].exists())
            self.assertTrue((exported["maps"] / "I_rat_initial.npy").exists())
            self.assertTrue((exported["average_initial_edcs"] / "stable_unchanged.npy").exists())

    def test_transition_outcome_maps_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = [
                build_synthetic_dataset(tmp_path / f"transition_{index}.nc", state_index=index)
                for index in range(4)
            ]

            result = run_transition_outcome_maps(
                [str(path) for path in files],
                TransitionOutcomeParameters(
                    lhb_center_ev=-0.18,
                    lhb_halfwidth_ev=0.07,
                    user_min_tau=0.0,
                    strong_tau_multiplier=1.8,
                ),
                pulse_labels=["A", "B", "B"],
            )

            self.assertEqual(result.shape, (18, 16))
            self.assertEqual(result.n_states, 4)
            self.assertEqual(result.n_transitions, 3)
            self.assertEqual(result.pulse_labels, ["A", "B", "B"])
            self.assertEqual(result.write_count_map.shape, (18, 16))
            self.assertEqual(result.erase_count_map.shape, (18, 16))
            self.assertEqual(result.activity_count_map.shape, (18, 16))
            self.assertEqual(result.repeated_switching_map.shape, (18, 16))
            for transition in result.transitions:
                self.assertEqual(transition.delta_irat_map.shape, (18, 16))
                self.assertTrue(set(transition.counts).issuperset(TRANSITION_OUTCOME_LABELS))
                self.assertEqual(int(sum(transition.counts.values())), 18 * 16)
                for code, label in enumerate(TRANSITION_OUTCOME_LABELS):
                    self.assertTrue(np.all(transition.label_map[transition.code_map == code] == label))

            rows = transition_outcome_table_rows(result)
            self.assertEqual(len(rows), 18 * 16 * 3)
            self.assertIn("transition_index", rows[0])
            self.assertIn("Delta_Irat", rows[0])
            self.assertIn("transition_label", rows[0])

            exported = export_transition_outcome_maps(result, tmp_path / "transition_export")
            self.assertTrue(exported["table"].exists())
            self.assertEqual(exported["table"].name, "transition_outcome_table.csv")
            self.assertTrue((exported["maps"] / "write_count_map.npy").exists())
            self.assertTrue((exported["transitions"] / "00_transition_0.nc_to_transition_1.nc" / "Delta_Irat.npy").exists())

    def test_initial_transition_features_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = [
                build_synthetic_dataset(tmp_path / f"initial_transition_{index}.nc", state_index=index)
                for index in range(4)
            ]

            result = run_initial_transition_feature_analysis(
                [str(path) for path in files],
                InitialTransitionFeatureParameters(
                    metallic_percentile=90.0,
                    erasure_percentile=90.0,
                    stable_percentile=25.0,
                    transition_mode="sequential",
                    normalization_mode="none",
                ),
            )

            self.assertEqual(result.shape, (18, 16))
            self.assertEqual(result.n_states, 4)
            self.assertEqual(result.n_transitions, 3)
            self.assertEqual(result.initial_reference_index, 0)
            self.assertEqual(result.initial_near_ef_map.shape, (18, 16))
            self.assertEqual(result.aggregate_maps["metallic_count"].shape, (18, 16))
            self.assertEqual(result.aggregate_maps["erased_count"].shape, (18, 16))
            self.assertEqual(result.aggregate_maps["first_metallic_transition"].shape, (18, 16))
            self.assertTrue(np.any(result.future_metallic_mask))
            self.assertIn("near_EF_intensity_A0", result.initial_feature_maps)
            self.assertIn("local_spatial_gradient_A0", result.initial_feature_maps)
            self.assertIn("future metallic", result.average_initial_edcs)
            self.assertEqual(result.average_initial_edcs["future metallic"].shape, result.e_axis.shape)
            self.assertEqual(result.average_initial_mdcs["future metallic"].shape, result.phi_axis.shape)
            self.assertTrue(result.group_statistics)

            metric_rows = initial_transition_metric_rows(result)
            feature_rows = initial_state_feature_rows(result)
            self.assertEqual(len(metric_rows), 18 * 16)
            self.assertEqual(len(feature_rows), 18 * 16)
            self.assertIn("metallic_transition_files", metric_rows[0])
            self.assertIn("near_EF_intensity_A0", feature_rows[0])

            exported = export_initial_transition_feature_analysis(result, tmp_path / "initial_transition_export")
            self.assertTrue(exported["metrics_table"].exists())
            self.assertTrue(exported["features_table"].exists())
            self.assertTrue(exported["group_statistics"].exists())
            self.assertTrue((exported["maps"] / "metallic_count.npy").exists())
            self.assertTrue((exported["maps"] / "max_metallicity_score.npy").exists())
            self.assertTrue((exported["maps"] / "future_erased_mask.npy").exists())
            self.assertTrue(any(exported["transitions"].iterdir()))

    def test_switching_mechanism_diagnostics_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = [
                build_synthetic_dataset(tmp_path / f"mechanism_{index}.nc", state_index=index)
                for index in range(4)
            ]
            transition_result = run_initial_transition_feature_analysis(
                [str(path) for path in files],
                InitialTransitionFeatureParameters(
                    metallic_percentile=90.0,
                    erasure_percentile=90.0,
                    stable_percentile=25.0,
                ),
            )

            result = run_switching_mechanism_diagnostics(
                transition_result=transition_result,
                parameters=SwitchingMechanismParameters(
                    transition_parameters=transition_result.parameters,
                    future_metallic_min_count=1,
                    future_erased_min_count=1,
                    boundary_percentile=80.0,
                    permutation_count=8,
                    threshold_sweep_percentiles=(90.0, 95.0),
                ),
            )

            self.assertEqual(result.shape, (18, 16))
            self.assertIn("future metallic", result.group_masks)
            self.assertIn("stable", result.group_edcs)
            self.assertEqual(result.group_edcs["stable"].shape, result.e_axis.shape)
            self.assertEqual(result.group_mdcs["stable"].shape, result.phi_axis.shape)
            self.assertIn("distance_to_domain_boundary", result.spatial_feature_maps)
            self.assertIn("first_metallic_transition", result.transition_history_maps)
            self.assertEqual(len(result.transition_level_rows), transition_result.n_transitions)
            self.assertTrue(result.spectral_effect_rows)
            self.assertTrue(result.spatial_effect_rows)
            self.assertTrue(result.artifact_rows)
            self.assertIn("spectral_evidence_score", result.summary_verdict)
            self.assertIn("artifact_risk_score", result.summary_verdict)

            exported = export_switching_mechanism_diagnostics(
                result,
                tmp_path / "mechanism_export",
                selected_pixel=(3, 4),
            )
            self.assertTrue(exported["spectral"].exists())
            self.assertTrue(exported["spatial"].exists())
            self.assertTrue(exported["transition_history"].exists())
            self.assertTrue(exported["artifact"].exists())
            self.assertTrue(exported["summary"].exists())
            self.assertTrue(exported["selected_pixel"].exists())
            self.assertTrue(exported["threshold_sensitivity"].exists())
            self.assertTrue(exported["boundary_distance"].exists())


if __name__ == "__main__":
    unittest.main()
