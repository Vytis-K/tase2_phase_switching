from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
from typing import Any

import numpy as np

from .analysis import (
    InitialTransitionFeatureParameters,
    SwitchingMechanismParameters,
    build_future_erased_mask,
    build_future_metallic_mask,
    compute_initial_spatial_features,
    finite_percentile,
    run_initial_transition_feature_analysis,
    write_rows_to_csv,
)


DEFAULT_SEQUENCE_NAMES = (
    "a_convert_2_nosm.nc",
    "b_convert_2_nosm.nc",
    "c_convert_2_nosm.nc",
    "d_convert_2_nosm.nc",
    "e_convert_2_nosm.nc",
    "f_convert_2_nosm.nc",
    "f2_convert_2_nosm.nc",
    "g_convert_2_nosm.nc",
    "g2_convert_2_nosm.nc",
    "g3_convert_2_nosm.nc",
    "g4_convert_2_nosm.nc",
    "h_adj_convert_2_nosm.nc",
)

TARGET_NAMES = (
    "future_metallic",
    "future_erased",
    "future_active",
    "both_metallic_erased",
    "repeated_switching",
    "stable_control",
    "never_switched",
    "first_switch_transition",
    "outcome_class",
)


@dataclass(slots=True)
class SwitchingMLDatasetParameters:
    """Configuration for the initial-state-to-future-switching ML dataset.

    The underlying analysis convention is (x, y, eV, phi). The exported row
    coordinates are aligned x/y pixel indices after any spatial cropping needed
    to compare 61 x 61 and 41 x 41 files.
    """

    fermi_level_ev: float = 0.0
    ef_min_ev: float = -0.05
    ef_max_ev: float = 0.05
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
    future_metallic_min_count: int = 1
    future_erased_min_count: int = 1
    future_metallic_min_frequency: float = 0.0
    future_erased_min_frequency: float = 0.0
    boundary_smooth_sigma: float = 1.0
    boundary_percentile: float = 85.0
    epsilon: float = 1e-8

    def transition_parameters(self) -> InitialTransitionFeatureParameters:
        return InitialTransitionFeatureParameters(
            fermi_level_ev=self.fermi_level_ev,
            ef_min_ev=self.ef_min_ev,
            ef_max_ev=self.ef_max_ev,
            feature_min_ev=self.feature_min_ev,
            feature_max_ev=self.feature_max_ev,
            asymmetry_split_ev=self.asymmetry_split_ev,
            metallic_percentile=self.metallic_percentile,
            erasure_percentile=self.erasure_percentile,
            stable_percentile=self.stable_percentile,
            transition_mode=self.transition_mode,
            reference_index=self.reference_index,
            normalization_mode=self.normalization_mode,
            allow_overlap=self.allow_overlap,
            epsilon=self.epsilon,
        )

    def mechanism_parameters(self) -> SwitchingMechanismParameters:
        return SwitchingMechanismParameters(
            transition_parameters=self.transition_parameters(),
            future_metallic_min_count=self.future_metallic_min_count,
            future_erased_min_count=self.future_erased_min_count,
            future_metallic_min_frequency=self.future_metallic_min_frequency,
            future_erased_min_frequency=self.future_erased_min_frequency,
            boundary_smooth_sigma=self.boundary_smooth_sigma,
            boundary_percentile=self.boundary_percentile,
            permutation_count=0,
            epsilon=self.epsilon,
        )


def default_transition_files(data_dir: str | Path = "data", include_c2: bool = False) -> list[Path]:
    data_path = Path(data_dir).expanduser().resolve()
    files: list[Path] = []
    for name in DEFAULT_SEQUENCE_NAMES:
        path = data_path / name
        if path.exists():
            files.append(path)
    if include_c2:
        c2 = data_path / "c2_convert_2_nosm.nc"
        if c2.exists():
            insert_at = 3 if len(files) >= 3 else len(files)
            files.insert(insert_at, c2)
    if files:
        return files

    discovered = sorted(data_path.glob("*.nc"), key=lambda p: natural_sort_key(p.name))
    if not include_c2:
        discovered = [path for path in discovered if not path.name.startswith("c2_")]
    return discovered


def natural_sort_key(value: str) -> tuple[Any, ...]:
    import re

    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def build_switching_ml_dataset(
    file_paths: list[str | Path],
    output_dir: str | Path,
    parameters: SwitchingMLDatasetParameters | None = None,
) -> dict[str, Path]:
    if parameters is None:
        parameters = SwitchingMLDatasetParameters()
    if len(file_paths) < 2:
        raise ValueError("At least two NetCDF files are required to build future-switching labels.")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    maps_dir = output_path / "maps"
    maps_dir.mkdir(exist_ok=True)

    transition_result = run_initial_transition_feature_analysis(
        [str(Path(path).expanduser().resolve()) for path in file_paths],
        parameters.transition_parameters(),
    )
    spatial_features = compute_initial_spatial_features(
        transition_result,
        parameters.mechanism_parameters(),
    )

    feature_maps, feature_groups = build_ml_feature_maps(
        transition_result.initial_feature_maps,
        spatial_features,
        transition_result.valid_mask,
        parameters.epsilon,
    )
    target_maps = build_ml_target_maps(transition_result, parameters)

    sample_mask = np.asarray(transition_result.valid_mask, dtype=bool)
    if not np.any(sample_mask):
        raise ValueError("No valid pixels were available after spatial alignment.")

    feature_names = list(feature_maps.keys())
    target_names = list(target_maps.keys())
    coords = np.argwhere(sample_mask).astype(np.int16)
    x_coord = coords[:, 0]
    y_coord = coords[:, 1]
    x_matrix = np.stack([np.asarray(feature_maps[name], dtype=np.float32)[sample_mask] for name in feature_names], axis=1)
    target_matrix = np.stack([np.asarray(target_maps[name])[sample_mask] for name in target_names], axis=1)

    dataset_npz = output_path / "switching_ml_dataset.npz"
    np.savez_compressed(
        dataset_npz,
        X=x_matrix.astype(np.float32),
        targets=target_matrix.astype(np.float32),
        x=x_coord,
        y=y_coord,
        valid_mask=sample_mask.astype(np.int8),
        feature_names=np.asarray(feature_names),
        target_names=np.asarray(target_names),
        map_shape=np.asarray(transition_result.shape, dtype=np.int16),
        row_kind=np.asarray(["aggregate_pixel"]),
    )

    transition_dataset_npz = output_path / "transition_switching_ml_dataset.npz"
    transition_training_table_path = output_path / "transition_switching_ml_training_table.csv"
    write_transition_level_dataset(
        transition_dataset_npz,
        transition_training_table_path,
        x_matrix,
        coords,
        feature_names,
        sample_mask,
        transition_result,
    )

    for name, values in feature_maps.items():
        np.save(maps_dir / f"feature_{safe_name(name)}.npy", np.asarray(values, dtype=np.float32))
    for name, values in target_maps.items():
        np.save(maps_dir / f"target_{safe_name(name)}.npy", np.asarray(values))

    table_rows = ml_dataset_rows(
        feature_maps,
        target_maps,
        sample_mask,
        transition_result,
    )
    table_path = output_path / "switching_ml_training_table.csv"
    write_rows_to_csv(table_path, table_rows)

    transition_table_path = output_path / "transition_label_audit.csv"
    write_rows_to_csv(transition_table_path, transition_audit_rows(transition_result, sample_mask))

    metadata_path = output_path / "metadata.json"
    metadata = build_metadata(
        transition_result=transition_result,
        parameters=parameters,
        feature_names=feature_names,
        feature_groups=feature_groups,
        target_names=target_names,
        target_maps=target_maps,
        sample_mask=sample_mask,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    readme_path = output_path / "README.md"
    readme_path.write_text(dataset_readme(metadata), encoding="utf-8")

    return {
        "dataset": dataset_npz,
        "transition_dataset": transition_dataset_npz,
        "table": table_path,
        "transition_table": transition_training_table_path,
        "transition_audit": transition_table_path,
        "metadata": metadata_path,
        "maps": maps_dir,
        "readme": readme_path,
    }


def build_ml_feature_maps(
    initial_features: dict[str, np.ndarray],
    spatial_features: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    epsilon: float,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    near = np.asarray(initial_features["near_EF_intensity_A0"], dtype=np.float32)
    feature = np.asarray(initial_features["feature_window_intensity_A0"], dtype=np.float32)
    total = np.asarray(initial_features["total_spectral_weight_A0"], dtype=np.float32)
    safe_feature = np.where(np.abs(feature) > epsilon, feature, np.nan)
    safe_total = np.where(np.abs(total) > epsilon, total, np.nan)

    feature_maps: dict[str, np.ndarray] = {
        "near_EF_intensity_A0": near,
        "feature_window_intensity_A0": feature,
        "initial_I_ratio_A0": (near / safe_feature).astype(np.float32),
        "initial_near_EF_fraction_A0": (near / safe_total).astype(np.float32),
        "initial_feature_fraction_A0": (feature / safe_total).astype(np.float32),
        "edc_peak_energy_A0": np.asarray(initial_features["edc_peak_energy_A0"], dtype=np.float32),
        "edc_peak_amplitude_A0": np.asarray(initial_features["edc_peak_amplitude_A0"], dtype=np.float32),
        "edc_peak_width_A0": np.asarray(initial_features["edc_peak_width_A0"], dtype=np.float32),
        "total_spectral_weight_A0": total,
        "edc_center_of_mass_A0": np.asarray(initial_features["edc_center_of_mass_A0"], dtype=np.float32),
        "edc_asymmetry_A0": np.asarray(initial_features["edc_asymmetry_A0"], dtype=np.float32),
        "initial_MDC_peak_position_A0": np.asarray(initial_features["initial_MDC_peak_position_A0"], dtype=np.float32),
        "initial_MDC_peak_width_A0": np.asarray(initial_features["initial_MDC_peak_width_A0"], dtype=np.float32),
        "local_spatial_gradient_A0": np.asarray(initial_features["local_spatial_gradient_A0"], dtype=np.float32),
        "local_neighborhood_mean_A0": np.asarray(initial_features["local_neighborhood_mean_A0"], dtype=np.float32),
        "local_neighborhood_std_A0": np.asarray(initial_features["local_neighborhood_std_A0"], dtype=np.float32),
        "distance_to_domain_boundary": np.asarray(spatial_features["distance_to_domain_boundary"], dtype=np.float32),
        "distance_to_valid_edge": np.asarray(spatial_features["distance_to_valid_edge"], dtype=np.float32),
        "local_contrast_texture": np.asarray(spatial_features["local_contrast_texture"], dtype=np.float32),
        "domain_boundary_mask": np.asarray(spatial_features["domain_boundary_mask"], dtype=np.float32),
        "x_coordinate": np.asarray(spatial_features["x_coordinate"], dtype=np.float32),
        "y_coordinate": np.asarray(spatial_features["y_coordinate"], dtype=np.float32),
        "valid_pixel": np.asarray(valid_mask, dtype=np.float32),
    }
    groups = {
        "spectral": [
            "near_EF_intensity_A0",
            "feature_window_intensity_A0",
            "initial_I_ratio_A0",
            "initial_near_EF_fraction_A0",
            "initial_feature_fraction_A0",
            "edc_peak_energy_A0",
            "edc_peak_amplitude_A0",
            "edc_peak_width_A0",
            "total_spectral_weight_A0",
            "edc_center_of_mass_A0",
            "edc_asymmetry_A0",
            "initial_MDC_peak_position_A0",
            "initial_MDC_peak_width_A0",
        ],
        "spatial": [
            "local_spatial_gradient_A0",
            "local_neighborhood_mean_A0",
            "local_neighborhood_std_A0",
            "distance_to_domain_boundary",
            "local_contrast_texture",
            "domain_boundary_mask",
        ],
        "artifact_position": [
            "total_spectral_weight_A0",
            "distance_to_valid_edge",
            "x_coordinate",
            "y_coordinate",
            "valid_pixel",
        ],
    }
    return feature_maps, groups


def build_ml_target_maps(
    transition_result: Any,
    parameters: SwitchingMLDatasetParameters,
) -> dict[str, np.ndarray]:
    aggregate = transition_result.aggregate_maps
    future_metallic = build_future_metallic_mask(
        aggregate,
        min_count=parameters.future_metallic_min_count,
        min_frequency=parameters.future_metallic_min_frequency,
    )
    future_erased = build_future_erased_mask(
        aggregate,
        min_count=parameters.future_erased_min_count,
        min_frequency=parameters.future_erased_min_frequency,
    )
    both = future_metallic & future_erased
    metallic_count = np.asarray(aggregate["metallic_count"], dtype=np.int16)
    erased_count = np.asarray(aggregate["erased_count"], dtype=np.int16)
    stable_count = np.asarray(aggregate["stable_count"], dtype=np.int16)
    active_count = metallic_count + erased_count
    stable = (stable_count > 0) & (metallic_count == 0) & (erased_count == 0)
    never = (metallic_count == 0) & (erased_count == 0)
    first_metallic = np.asarray(aggregate["first_metallic_transition"], dtype=np.float32)
    first_erased = np.asarray(aggregate["first_erased_transition"], dtype=np.float32)
    first_switch = np.full(first_metallic.shape, -1, dtype=np.int16)
    metal_finite = np.isfinite(first_metallic)
    erased_finite = np.isfinite(first_erased)
    first_switch[metal_finite] = first_metallic[metal_finite].astype(np.int16)
    first_switch[erased_finite & ~metal_finite] = first_erased[erased_finite & ~metal_finite].astype(np.int16)
    both_finite = metal_finite & erased_finite
    first_switch[both_finite] = np.minimum(first_metallic[both_finite], first_erased[both_finite]).astype(np.int16)

    outcome = np.zeros(first_metallic.shape, dtype=np.int16)
    outcome[stable] = 1
    outcome[future_metallic & ~future_erased] = 2
    outcome[future_erased & ~future_metallic] = 3
    outcome[both] = 4

    return {
        "future_metallic": future_metallic.astype(np.int8),
        "future_erased": future_erased.astype(np.int8),
        "future_active": (future_metallic | future_erased).astype(np.int8),
        "both_metallic_erased": both.astype(np.int8),
        "repeated_switching": (active_count >= 2).astype(np.int8),
        "stable_control": stable.astype(np.int8),
        "never_switched": never.astype(np.int8),
        "first_switch_transition": first_switch,
        "outcome_class": outcome,
    }


def write_transition_level_dataset(
    dataset_path: Path,
    table_path: Path,
    base_x_matrix: np.ndarray,
    coords: np.ndarray,
    base_feature_names: list[str],
    sample_mask: np.ndarray,
    transition_result: Any,
) -> None:
    """Export one row per valid pixel per transition for pulse-level map prediction.

    The features are initial-state precursor features plus simple transition
    descriptors. This lets a model answer: given the initial pixel and a pulse
    index, which pixels are predicted to be written/erased in that transition?
    """

    n_pixels = base_x_matrix.shape[0]
    n_transitions = max(1, transition_result.n_transitions)
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    transition_index_rows: list[np.ndarray] = []
    transition_names: list[str] = []
    csv_rows: list[dict[str, Any]] = []

    transition_feature_names = list(base_feature_names) + [
        "transition_index",
        "transition_fraction",
        "before_state_index",
        "after_state_index",
    ]
    transition_target_names = [
        "transition_metallic",
        "transition_erased",
        "transition_active",
        "transition_stable",
        "transition_outcome_class",
    ]

    for transition in transition_result.transitions:
        transition_names.append(transition.name)
        transition_index = np.full((n_pixels, 1), float(transition.index), dtype=np.float32)
        transition_fraction = np.full(
            (n_pixels, 1),
            float(transition.index / max(1, n_transitions - 1)),
            dtype=np.float32,
        )
        before_index = np.full((n_pixels, 1), float(transition.before_index), dtype=np.float32)
        after_index = np.full((n_pixels, 1), float(transition.after_index), dtype=np.float32)
        feature_rows.append(
            np.concatenate(
                [base_x_matrix, transition_index, transition_fraction, before_index, after_index],
                axis=1,
            ).astype(np.float32)
        )

        metallic = np.asarray(transition.metallic_mask[sample_mask], dtype=np.int8)
        erased = np.asarray(transition.erased_mask[sample_mask], dtype=np.int8)
        stable = np.asarray(transition.stable_mask[sample_mask], dtype=np.int8)
        active = ((metallic > 0) | (erased > 0)).astype(np.int8)
        outcome = np.zeros(n_pixels, dtype=np.int16)
        outcome[stable > 0] = 1
        outcome[(metallic > 0) & (erased == 0)] = 2
        outcome[(erased > 0) & (metallic == 0)] = 3
        outcome[(metallic > 0) & (erased > 0)] = 4
        target_rows.append(np.stack([metallic, erased, active, stable, outcome], axis=1))
        x_rows.append(coords[:, 0].astype(np.int16))
        y_rows.append(coords[:, 1].astype(np.int16))
        transition_index_rows.append(np.full(n_pixels, transition.index, dtype=np.int16))

        metal_score = np.asarray(transition.metallicity_score[sample_mask], dtype=np.float32)
        erase_score = np.asarray(transition.erasure_score[sample_mask], dtype=np.float32)
        magnitude = np.asarray(transition.transition_magnitude[sample_mask], dtype=np.float32)
        for row_index, (x_index, y_index) in enumerate(coords):
            row = {
                "x": int(x_index),
                "y": int(y_index),
                "transition_index": int(transition.index),
                "transition_name": transition.name,
                "from_file": transition_result.loaded_states[transition.before_index].name,
                "to_file": transition_result.loaded_states[transition.after_index].name,
                "transition_metallic": int(metallic[row_index]),
                "transition_erased": int(erased[row_index]),
                "transition_active": int(active[row_index]),
                "transition_stable": int(stable[row_index]),
                "transition_outcome_class": int(outcome[row_index]),
                "metallicity_score": float(metal_score[row_index]),
                "erasure_score": float(erase_score[row_index]),
                "transition_magnitude": float(magnitude[row_index]),
            }
            csv_rows.append(row)

    np.savez_compressed(
        dataset_path,
        X=np.concatenate(feature_rows, axis=0).astype(np.float32),
        targets=np.concatenate(target_rows, axis=0).astype(np.float32),
        x=np.concatenate(x_rows, axis=0),
        y=np.concatenate(y_rows, axis=0),
        transition_index=np.concatenate(transition_index_rows, axis=0),
        feature_names=np.asarray(transition_feature_names),
        target_names=np.asarray(transition_target_names),
        transition_names=np.asarray(transition_names),
        map_shape=np.asarray(transition_result.shape, dtype=np.int16),
        row_kind=np.asarray(["transition_pixel"]),
    )
    write_rows_to_csv(table_path, csv_rows)


def ml_dataset_rows(
    feature_maps: dict[str, np.ndarray],
    target_maps: dict[str, np.ndarray],
    sample_mask: np.ndarray,
    transition_result: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    coords = np.argwhere(sample_mask)
    for x_index, y_index in coords:
        row: dict[str, Any] = {"x": int(x_index), "y": int(y_index)}
        for name, values in feature_maps.items():
            row[name] = scalar_for_csv(values[x_index, y_index])
        for name, values in target_maps.items():
            row[name] = scalar_for_csv(values[x_index, y_index])
        row.update(pixel_transition_summary(transition_result, int(x_index), int(y_index)))
        rows.append(row)
    return rows


def transition_audit_rows(transition_result: Any, sample_mask: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    coords = np.argwhere(sample_mask)
    for x_index, y_index in coords:
        for transition in transition_result.transitions:
            rows.append(
                {
                    "x": int(x_index),
                    "y": int(y_index),
                    "transition_index": int(transition.index),
                    "from_file": transition_result.loaded_states[transition.before_index].name,
                    "to_file": transition_result.loaded_states[transition.after_index].name,
                    "metallicity_score": float(transition.metallicity_score[x_index, y_index]),
                    "erasure_score": float(transition.erasure_score[x_index, y_index]),
                    "transition_magnitude": float(transition.transition_magnitude[x_index, y_index]),
                    "metallic_threshold": float(transition.metallic_threshold),
                    "erasure_threshold": float(transition.erasure_threshold),
                    "stable_threshold": float(transition.stable_threshold),
                    "metallic_label": bool(transition.metallic_mask[x_index, y_index]),
                    "erased_label": bool(transition.erased_mask[x_index, y_index]),
                    "stable_label": bool(transition.stable_mask[x_index, y_index]),
                }
            )
    return rows


def pixel_transition_summary(transition_result: Any, x_index: int, y_index: int) -> dict[str, Any]:
    metallic_indices: list[str] = []
    erased_indices: list[str] = []
    stable_indices: list[str] = []
    for transition in transition_result.transitions:
        if bool(transition.metallic_mask[x_index, y_index]):
            metallic_indices.append(str(transition.index))
        if bool(transition.erased_mask[x_index, y_index]):
            erased_indices.append(str(transition.index))
        if bool(transition.stable_mask[x_index, y_index]):
            stable_indices.append(str(transition.index))
    return {
        "metallic_transition_indices": ";".join(metallic_indices),
        "erased_transition_indices": ";".join(erased_indices),
        "stable_transition_indices": ";".join(stable_indices),
    }


def scalar_for_csv(value: Any) -> int | float | str:
    arr = np.asarray(value)
    if arr.dtype.kind in {"i", "u", "b"}:
        return int(arr)
    if arr.dtype.kind == "f":
        val = float(arr)
        return val if np.isfinite(val) else ""
    return str(value)


def build_metadata(
    transition_result: Any,
    parameters: SwitchingMLDatasetParameters,
    feature_names: list[str],
    feature_groups: dict[str, list[str]],
    target_names: list[str],
    target_maps: dict[str, np.ndarray],
    sample_mask: np.ndarray,
) -> dict[str, Any]:
    target_counts: dict[str, dict[str, int]] = {}
    for name, values in target_maps.items():
        sampled = np.asarray(values)[sample_mask]
        unique, counts = np.unique(sampled, return_counts=True)
        target_counts[name] = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    return {
        "description": "Initial-state precursor dataset for predicting future TaS2 switching labels.",
        "array_convention": "NetCDF cubes are standardized as (x, y, eV, phi); dataset rows are valid aligned x/y pixels from the initial reference file.",
        "files": transition_result.file_paths,
        "state_names": transition_result.state_names,
        "reference_index": int(transition_result.initial_reference_index),
        "transitions": [
            {
                "index": int(transition.index),
                "name": transition.name,
                "before_index": int(transition.before_index),
                "after_index": int(transition.after_index),
                "metallic_threshold": float(transition.metallic_threshold),
                "erasure_threshold": float(transition.erasure_threshold),
                "stable_threshold": float(transition.stable_threshold),
            }
            for transition in transition_result.transitions
        ],
        "parameters": asdict(parameters),
        "shape": [int(v) for v in transition_result.shape],
        "n_samples": int(np.count_nonzero(sample_mask)),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "feature_groups": feature_groups,
        "target_names": target_names,
        "target_counts": target_counts,
        "score_threshold_notes": {
            "future_metallic": "1 when metallic_count/frequency filters pass; metallic_count is the number of transitions where near-EF weight increased above the per-transition percentile threshold.",
            "future_erased": "1 when erased_count/frequency filters pass; erased_count is the number of transitions where feature-window weight was lost above the per-transition percentile threshold.",
            "repeated_switching": "1 when metallic_count + erased_count is at least 2.",
            "outcome_class": "0=never switched, 1=stable control, 2=future metallic only, 3=future erased only, 4=both metallic and erased.",
        },
        "notes": transition_result.notes,
    }


def dataset_readme(metadata: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# TaS2 Switching ML Dataset",
            "",
            "This folder contains a row-per-pixel dataset for predicting future switching from the initial ARPES state only.",
            "",
            "Core files:",
            "",
            "- `switching_ml_dataset.npz`: machine-readable arrays.",
            "- `switching_ml_training_table.csv`: auditable feature/target table.",
            "- `transition_label_audit.csv`: per-pixel, per-transition scores and thresholds.",
            "- `metadata.json`: file sequence, thresholds, feature names, target definitions, and counts.",
            "- `maps/`: 2D feature and target maps in aligned initial-state coordinates.",
            "",
            "Recommended first targets:",
            "",
            "- `future_active`: any later metallic or erased behavior.",
            "- `future_metallic`: later gain of near-EF weight.",
            "- `future_erased`: later loss of feature-window weight.",
            "- `repeated_switching`: switched in more than one transition.",
            "",
            "Use spatial-blocked validation when training so neighboring pixels do not leak between train/test.",
            "",
            f"Samples: {metadata['n_samples']}",
            f"Features: {metadata['n_features']}",
        ]
    )


def safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build an ML-ready initial-state precursor dataset for TaS2 switching prediction.")
    parser.add_argument("--data-dir", default="data", help="Directory containing NetCDF files. Used when --nc-files is omitted.")
    parser.add_argument("--nc-files", nargs="*", default=None, help="Chronological NetCDF files. Defaults to data/a,b,c,d,e,f,f2,g,g2,g3,g4,h_adj and excludes c2.")
    parser.add_argument("--include-c2", action="store_true", help="Include c2_convert_2_nosm.nc in the default discovered sequence.")
    parser.add_argument("--output-dir", default="outputs/ml_switching_dataset", help="Output dataset folder.")
    parser.add_argument("--transition-mode", choices=("sequential", "initial_reference"), default="sequential")
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--normalization-mode", choices=("none", "total_intensity", "median_near_ef", "high_percentile"), default="none")
    parser.add_argument("--metallic-percentile", type=float, default=95.0)
    parser.add_argument("--erasure-percentile", type=float, default=95.0)
    parser.add_argument("--stable-percentile", type=float, default=25.0)
    parser.add_argument("--ef-min", type=float, default=-0.05)
    parser.add_argument("--ef-max", type=float, default=0.05)
    parser.add_argument("--feature-min", type=float, default=-0.30)
    parser.add_argument("--feature-max", type=float, default=-0.05)
    parser.add_argument("--future-metallic-min-count", type=int, default=1)
    parser.add_argument("--future-erased-min-count", type=int, default=1)
    args = parser.parse_args(argv)

    files = [Path(path) for path in args.nc_files] if args.nc_files else default_transition_files(args.data_dir, include_c2=args.include_c2)
    if not files:
        raise SystemExit(f"No NetCDF files found in {args.data_dir!r}.")

    params = SwitchingMLDatasetParameters(
        ef_min_ev=args.ef_min,
        ef_max_ev=args.ef_max,
        feature_min_ev=args.feature_min,
        feature_max_ev=args.feature_max,
        metallic_percentile=args.metallic_percentile,
        erasure_percentile=args.erasure_percentile,
        stable_percentile=args.stable_percentile,
        transition_mode=args.transition_mode,
        reference_index=args.reference_index,
        normalization_mode=args.normalization_mode,
        future_metallic_min_count=args.future_metallic_min_count,
        future_erased_min_count=args.future_erased_min_count,
    )
    print("Building switching ML dataset from files:")
    for index, path in enumerate(files):
        print(f"  {index:02d}: {path}")
    paths = build_switching_ml_dataset(files, args.output_dir, params)
    print("Wrote ML dataset:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
