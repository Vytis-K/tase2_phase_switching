#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from tase2_geometry_simulation import (
    DEFAULT_CASE_FILES,
    apply_region_param_overrides,
    build_cross_mask_from_maps,
    build_simulated_cube,
    cube_to_dataarray,
    extract_pixel_features,
    infer_region_map,
    infer_region_params,
    load_dataset_bundle,
    log_progress,
    resolve_case_path,
    rmse,
    to_float32_numpy,
    total_and_ef_maps,
    warp_region_spectra,
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_data_dir = (script_dir / "../../data").resolve()
    default_output_dir = (script_dir / "../../outputs/geometry_mapping").resolve()

    parser = argparse.ArgumentParser(
        description=(
            "Map local nanoARPES geometry from base and pulsed files, compare predicted "
            "geometry fields against experiment, and write figures, a NetCDF geometry bundle, "
            "and a Markdown report."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--base-case", choices=list(DEFAULT_CASE_FILES), default="a")
    parser.add_argument("--target-case", choices=list(DEFAULT_CASE_FILES), default="b")
    parser.add_argument("--base-file", type=Path, default=None)
    parser.add_argument("--target-file", type=Path, default=None)
    parser.add_argument("--fermi-level-ev", type=float, default=0.0)
    parser.add_argument("--ef-window-ev", type=float, default=0.05)
    parser.add_argument("--wide-window-ev", type=float, default=0.20)
    parser.add_argument("--n-regions", type=int, default=6)
    parser.add_argument("--region-map", type=Path, default=None)
    parser.add_argument("--region-params-json", type=Path, default=None)
    parser.add_argument("--cross-threshold-quantile", type=float, default=0.45)
    parser.add_argument("--cross-row-fraction", type=float, default=0.18)
    parser.add_argument("--cross-col-fraction", type=float, default=0.18)
    parser.add_argument("--cross-background-quantile", type=float, default=0.10)
    parser.add_argument("--cross-pad", type=int, default=1)
    parser.add_argument("--tilt-scale-phi", type=float, default=1.0)
    parser.add_argument("--tilt-scale-ev", type=float, default=0.35)
    parser.add_argument("--current-gain", type=float, default=1.0)
    parser.add_argument("--smooth-region-iterations", type=int, default=1)
    return parser.parse_args()


def ensure_region_map(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    region_map = np.load(path)
    if region_map.shape != expected_shape:
        raise ValueError(f"Region map shape {region_map.shape} does not match expected {expected_shape}.")
    return region_map.astype(np.int32)


def compute_geometry_maps(
    features: Dict[str, np.ndarray],
    valid_mask: np.ndarray,
    phi_axis: np.ndarray,
    e_axis: np.ndarray,
    ref_phi_centroid: float,
    ref_e_centroid: float,
) -> Dict[str, np.ndarray]:
    dphi = float(np.median(np.diff(phi_axis)))
    de = float(np.median(np.diff(e_axis)))
    phi_centroid = features["phi_centroid"].astype(np.float32)
    e_centroid = features["e_centroid"].astype(np.float32)
    phi_var = features["phi_var"].astype(np.float32)
    ef_fraction = features["ef_fraction"].astype(np.float32)
    phi_tilt_bins = ((phi_centroid - np.float32(ref_phi_centroid)) / np.float32(dphi + 1e-8)).astype(np.float32)
    e_tilt_bins = ((e_centroid - np.float32(ref_e_centroid)) / np.float32(de + 1e-8)).astype(np.float32)
    phi_width_bins = np.sqrt(np.maximum(phi_var, 0.0)).astype(np.float32) / np.float32(abs(dphi) + 1e-8)

    for arr in (phi_centroid, e_centroid, phi_var, ef_fraction, phi_tilt_bins, e_tilt_bins, phi_width_bins):
        arr[~valid_mask] = np.nan

    return {
        "phi_centroid": phi_centroid,
        "e_centroid": e_centroid,
        "phi_var": phi_var,
        "ef_fraction": ef_fraction,
        "phi_tilt_bins": phi_tilt_bins,
        "e_tilt_bins": e_tilt_bins,
        "phi_width_bins": phi_width_bins,
    }


def build_region_predicted_geometry(
    region_map: np.ndarray,
    valid_mask: np.ndarray,
    region_params,
) -> Dict[str, np.ndarray]:
    phi_shift = np.full(region_map.shape, np.nan, dtype=np.float32)
    e_shift = np.full(region_map.shape, np.nan, dtype=np.float32)
    current_mix = np.full(region_map.shape, np.nan, dtype=np.float32)
    for param in region_params:
        mask = (region_map == param.region_id) & valid_mask
        phi_shift[mask] = np.float32(param.phi_shift_bins)
        e_shift[mask] = np.float32(param.energy_shift_bins)
        current_mix[mask] = np.float32(param.current_mix)
    return {
        "predicted_phi_shift_bins": phi_shift,
        "predicted_e_shift_bins": e_shift,
        "predicted_current_mix": current_mix,
    }


def nan_rmse(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b) & valid_mask
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def nan_corr(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b) & valid_mask
    if np.count_nonzero(mask) < 3:
        return float("nan")
    aa = a[mask].astype(np.float64)
    bb = b[mask].astype(np.float64)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = np.sqrt(np.sum(aa**2) * np.sum(bb**2))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(aa * bb) / denom)


def percentile_vmax(arr: np.ndarray, valid_mask: np.ndarray, q: float = 99.0, fallback: float = 1.0) -> float:
    vals = np.abs(arr[np.isfinite(arr) & valid_mask])
    if vals.size == 0:
        return fallback
    vmax = float(np.nanpercentile(vals, q))
    return vmax if vmax > 1e-12 else fallback


def save_geometry_figure(
    out_path: Path,
    avg_norm_map: np.ndarray,
    region_map: np.ndarray,
    valid_mask: np.ndarray,
    base_geom: Dict[str, np.ndarray],
    target_geom: Dict[str, np.ndarray],
    predicted_geom: Dict[str, np.ndarray],
    simulated_geom: Dict[str, np.ndarray],
    case_label: str,
) -> None:
    region_display = region_map.astype(float)
    region_display[~valid_mask] = np.nan

    phi_diff_pred = predicted_geom["predicted_phi_shift_bins"] - target_geom["phi_tilt_bins"]
    e_diff_pred = predicted_geom["predicted_e_shift_bins"] - target_geom["e_tilt_bins"]
    phi_diff_sim = simulated_geom["phi_tilt_bins"] - target_geom["phi_tilt_bins"]
    ef_diff_sim = simulated_geom["ef_fraction"] - target_geom["ef_fraction"]

    phi_lim = max(
        percentile_vmax(base_geom["phi_tilt_bins"], valid_mask),
        percentile_vmax(target_geom["phi_tilt_bins"], valid_mask),
        percentile_vmax(predicted_geom["predicted_phi_shift_bins"], valid_mask),
        percentile_vmax(simulated_geom["phi_tilt_bins"], valid_mask),
    )
    e_lim = max(
        percentile_vmax(base_geom["e_tilt_bins"], valid_mask),
        percentile_vmax(target_geom["e_tilt_bins"], valid_mask),
        percentile_vmax(predicted_geom["predicted_e_shift_bins"], valid_mask),
        percentile_vmax(simulated_geom["e_tilt_bins"], valid_mask),
    )
    phi_diff_lim = max(percentile_vmax(phi_diff_pred, valid_mask), percentile_vmax(phi_diff_sim, valid_mask))
    e_diff_lim = percentile_vmax(e_diff_pred, valid_mask)
    ef_diff_lim = percentile_vmax(ef_diff_sim, valid_mask)

    fig, axes = plt.subplots(4, 4, figsize=(20, 18))

    panels = [
        (avg_norm_map, "magma", None, None, "Average normalized total intensity"),
        (region_display, "tab10", None, None, "Region map"),
        (predicted_geom["predicted_current_mix"], "viridis", None, None, f"Predicted current mix: {case_label}"),
        (base_geom["ef_fraction"], "viridis", None, None, "Base near-EF fraction"),

        (base_geom["phi_tilt_bins"], "coolwarm", -phi_lim, phi_lim, "Base experimental phi-tilt (bins)"),
        (target_geom["phi_tilt_bins"], "coolwarm", -phi_lim, phi_lim, f"Target experimental phi-tilt (bins): {case_label}"),
        (predicted_geom["predicted_phi_shift_bins"], "coolwarm", -phi_lim, phi_lim, "Predicted phi-tilt from region model"),
        (simulated_geom["phi_tilt_bins"], "coolwarm", -phi_lim, phi_lim, "Simulated-data phi-tilt after geometry routine"),

        (base_geom["e_tilt_bins"], "coolwarm", -e_lim, e_lim, "Base experimental energy-tilt (bins)"),
        (target_geom["e_tilt_bins"], "coolwarm", -e_lim, e_lim, f"Target experimental energy-tilt (bins): {case_label}"),
        (predicted_geom["predicted_e_shift_bins"], "coolwarm", -e_lim, e_lim, "Predicted energy-tilt from region model"),
        (simulated_geom["e_tilt_bins"], "coolwarm", -e_lim, e_lim, "Simulated-data energy-tilt after geometry routine"),

        (phi_diff_pred, "coolwarm", -phi_diff_lim, phi_diff_lim, "Predicted phi-tilt minus target experimental"),
        (phi_diff_sim, "coolwarm", -phi_diff_lim, phi_diff_lim, "Simulated phi-tilt minus target experimental"),
        (e_diff_pred, "coolwarm", -e_diff_lim, e_diff_lim, "Predicted energy-tilt minus target experimental"),
        (ef_diff_sim, "coolwarm", -ef_diff_lim, ef_diff_lim, "Simulated near-EF fraction minus target experimental"),
    ]

    for ax, (img, cmap, vmin, vmax, title) in zip(axes.ravel(), panels):
        im = ax.imshow(img.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_geometry_dataset(
    template_da: xr.DataArray,
    region_map: np.ndarray,
    valid_mask: np.ndarray,
    base_geom: Dict[str, np.ndarray],
    target_geom: Dict[str, np.ndarray],
    predicted_geom: Dict[str, np.ndarray],
    simulated_geom: Dict[str, np.ndarray],
    metrics: Dict[str, float | str | int],
) -> xr.Dataset:
    coords = {"x": template_da.coords["x"], "y": template_da.coords["y"]}
    data_vars = {
        "region_map": xr.DataArray(region_map.astype(np.int32), dims=("x", "y"), coords=coords),
        "valid_mask": xr.DataArray(valid_mask.astype(np.int8), dims=("x", "y"), coords=coords),
    }
    for prefix, geom in [
        ("base", base_geom),
        ("target", target_geom),
        ("predicted", predicted_geom),
        ("simulated", simulated_geom),
    ]:
        for name, arr in geom.items():
            data_vars[f"{prefix}_{name}"] = xr.DataArray(arr.astype(np.float32), dims=("x", "y"), coords=coords)

    ds = xr.Dataset(data_vars)
    for key, value in metrics.items():
        ds.attrs[key] = value
    ds.attrs["description"] = (
        "Geometry prediction bundle containing experimental base geometry, experimental target geometry, "
        "region-model predicted geometry fields, and geometry fields re-extracted from the simulated data cube."
    )
    return ds


def write_markdown_report(
    path: Path,
    run_date: str,
    base_path: Path,
    target_path: Path,
    case_label: str,
    metrics: Dict[str, float | str | int],
    region_params,
    figure_name: str,
    nc_name: str,
) -> None:
    lines = []
    lines.append(f"# Geometry prediction report: {case_label}")
    lines.append("")
    lines.append(f"Run date: {run_date}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"Base file: `{base_path}`")
    lines.append("")
    lines.append(f"Target file: `{target_path}`")
    lines.append("")
    lines.append("## What the geometry fields mean")
    lines.append("")
    lines.append(
        "The local geometry calculation uses the spectral centroids and widths extracted from the `(eV, phi)` cube at each spatial pixel. "
        "The principal geometry observables are the local phi centroid, the local energy centroid, and the phi width. "
        "The phi-tilt map is the local phi centroid expressed in units of detector bins relative to the mean centroid of the base-state file. "
        "The energy-tilt map is the corresponding local energy-centroid offset in units of energy bins. "
        "These fields are not a microscopic surface-normal reconstruction. They are geometry-sensitive observables extracted from the measured spectra that indicate how strongly the local cut is displaced from the reference cut."
    )
    lines.append("")
    lines.append(
        "The region-model prediction is the geometry field implied by the inferred region partition and the per-region tilt parameters. "
        "The simulated-data geometry field is obtained by first generating a simulated cube with the geometry routine and then running the same geometry extraction back over that simulated cube. "
        "This separates the direct parameter prediction from the geometry that is actually recoverable after the forward model has acted on the data."
    )
    lines.append("")
    lines.append("## Main outputs")
    lines.append("")
    lines.append(f"Comparison figure: `{figure_name}`")
    lines.append("")
    lines.append(f"Geometry bundle NetCDF: `{nc_name}`")
    lines.append("")
    lines.append("## Accuracy metrics")
    lines.append("")
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: `{value:.6f}`")
        else:
            lines.append(f"{key}: `{value}`")
        lines.append("")
    lines.append("## Interpreting the comparison figure")
    lines.append("")
    lines.append(
        "The first row gives the spatial support of the calculation, the inferred region map, the predicted current-mix field, and the base near-Fermi spectral-weight map. "
        "The second row compares phi-tilt maps from the base experiment, the pulsed experiment, the direct region-model prediction, and the simulated cube after the geometry extraction routine has been applied. "
        "The third row shows the same comparison for the energy-tilt field. "
        "The final row gives residual maps, so that one can visually distinguish where the region-model geometry itself is inaccurate from where the full forward simulation still fails to match the pulsed data after running the geometry extraction routine."
    )
    lines.append("")
    lines.append("## Inferred region parameters")
    lines.append("")
    lines.append("| Region | Name | Phi shift (bins) | Energy shift (bins) | Current mix |")
    lines.append("|---:|---|---:|---:|---:|")
    for param in region_params:
        lines.append(
            f"| {int(param.region_id)} | {param.name} | {float(param.phi_shift_bins):.6f} | {float(param.energy_shift_bins):.6f} | {float(param.current_mix):.6f} |"
        )
    lines.append("")
    lines.append("## Recommended use")
    lines.append("")
    lines.append(
        "This report is most useful when read in the following order. First inspect the target experimental geometry maps to see whether the pulsed state contains large-scale geometry changes or only weak local changes. "
        "Then compare the direct region-model prediction to determine whether the inferred tilt regions already point in the correct directions and magnitudes. "
        "Finally compare the simulated-data geometry fields to see what geometry the full forward model actually produces after the same extraction routine is reapplied. "
        "The residual maps localize where the inferred tilt partition is physically plausible and where the present region model is still too coarse."
    )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    run_date = datetime.now().strftime("%Y_%m_%d")

    base_path = resolve_case_path(args.data_dir, args.base_case, args.base_file)
    target_path = resolve_case_path(args.data_dir, args.target_case, args.target_file)
    case_label = args.target_case

    log_progress(f"Starting geometry mapping run for target case '{case_label}'")
    log_progress(f"Base path resolved to: {base_path}")
    log_progress(f"Target path resolved to: {target_path}")

    base_bundle = load_dataset_bundle(base_path)
    target_bundle = load_dataset_bundle(target_path)
    base_cube = to_float32_numpy(base_bundle.da)
    target_cube = to_float32_numpy(target_bundle.da)

    log_progress("Computing valid mask from integrated intensity maps")
    base_total, _ = total_and_ef_maps(base_bundle.da, args.fermi_level_ev, args.ef_window_ev)
    target_total, _ = total_and_ef_maps(target_bundle.da, args.fermi_level_ev, args.ef_window_ev)
    valid_mask, avg_norm_map = build_cross_mask_from_maps(
        [base_total, target_total],
        threshold_quantile=args.cross_threshold_quantile,
        row_fraction=args.cross_row_fraction,
        col_fraction=args.cross_col_fraction,
        background_quantile=args.cross_background_quantile,
        pad=args.cross_pad,
    )
    log_progress(f"Valid mask complete: {int(valid_mask.sum())} valid pixels")

    log_progress("Extracting base features")
    base_features, base_feat_matrix, _ = extract_pixel_features(
        base_bundle.da,
        fermi_level=args.fermi_level_ev,
        ef_window=args.ef_window_ev,
        wide_window=args.wide_window_ev,
    )
    log_progress("Extracting target features")
    target_features, _, _ = extract_pixel_features(
        target_bundle.da,
        fermi_level=args.fermi_level_ev,
        ef_window=args.ef_window_ev,
        wide_window=args.wide_window_ev,
    )

    if args.region_map is not None:
        log_progress(f"Loading user-provided region map from {args.region_map}")
        region_map = ensure_region_map(args.region_map, expected_shape=valid_mask.shape)
    else:
        log_progress(f"Inferring region map with n_regions={args.n_regions}")
        region_map = infer_region_map(
            base_features,
            base_feat_matrix,
            valid_mask,
            n_regions=args.n_regions,
            smooth_iterations=args.smooth_region_iterations,
        )

    phi_axis = np.asarray(base_bundle.da.coords["phi"].values, dtype=np.float32)
    e_axis = np.asarray(base_bundle.da.coords["eV"].values, dtype=np.float32)
    ref_phi_centroid = float(np.nanmean(base_features["phi_centroid"][valid_mask]))
    ref_e_centroid = float(np.nanmean(base_features["e_centroid"][valid_mask]))

    log_progress("Inferring region-level geometry parameters and activation")
    region_params, activation_map = infer_region_params(
        base_features,
        target_features,
        region_map,
        valid_mask,
        phi_axis,
        e_axis,
        tilt_scale_phi=args.tilt_scale_phi,
        tilt_scale_ev=args.tilt_scale_ev,
        current_gain=args.current_gain,
    )
    if args.region_params_json is not None:
        log_progress(f"Applying region parameter overrides from {args.region_params_json}")
        region_params = apply_region_param_overrides(region_params, args.region_params_json)

    log_progress("Warping base and target cubes by inferred geometry")
    base_warped = warp_region_spectra(base_cube, region_map, region_params, valid_mask, label="base")
    target_warped = warp_region_spectra(target_cube, region_map, region_params, valid_mask, label="target")

    log_progress("Building simulated cube")
    simulated_cube = build_simulated_cube(
        base_cube,
        base_warped,
        target_cube,
        target_warped,
        region_map,
        valid_mask,
        region_params,
        activation_map,
    )
    simulated_da = cube_to_dataarray(base_bundle.da, simulated_cube, name="simulated_intensity")
    log_progress("Extracting simulated features")
    simulated_features, _, _ = extract_pixel_features(
        simulated_da,
        fermi_level=args.fermi_level_ev,
        ef_window=args.ef_window_ev,
        wide_window=args.wide_window_ev,
    )

    log_progress("Building geometry maps")
    base_geom = compute_geometry_maps(base_features, valid_mask, phi_axis, e_axis, ref_phi_centroid, ref_e_centroid)
    target_geom = compute_geometry_maps(target_features, valid_mask, phi_axis, e_axis, ref_phi_centroid, ref_e_centroid)
    simulated_geom = compute_geometry_maps(simulated_features, valid_mask, phi_axis, e_axis, ref_phi_centroid, ref_e_centroid)
    predicted_geom = build_region_predicted_geometry(region_map, valid_mask, region_params)

    metrics: Dict[str, float | str | int] = {
        "run_date": run_date,
        "base_case": args.base_case,
        "target_case": case_label,
        "valid_pixels": int(valid_mask.sum()),
        "excluded_pixels": int((~valid_mask).sum()),
        "n_regions": int(len(region_params)),
        "rmse_predicted_phi_vs_target_phi_bins": nan_rmse(predicted_geom["predicted_phi_shift_bins"], target_geom["phi_tilt_bins"], valid_mask),
        "rmse_simulated_phi_vs_target_phi_bins": nan_rmse(simulated_geom["phi_tilt_bins"], target_geom["phi_tilt_bins"], valid_mask),
        "rmse_predicted_e_vs_target_e_bins": nan_rmse(predicted_geom["predicted_e_shift_bins"], target_geom["e_tilt_bins"], valid_mask),
        "rmse_simulated_e_vs_target_e_bins": nan_rmse(simulated_geom["e_tilt_bins"], target_geom["e_tilt_bins"], valid_mask),
        "corr_predicted_phi_vs_target_phi_bins": nan_corr(predicted_geom["predicted_phi_shift_bins"], target_geom["phi_tilt_bins"], valid_mask),
        "corr_simulated_phi_vs_target_phi_bins": nan_corr(simulated_geom["phi_tilt_bins"], target_geom["phi_tilt_bins"], valid_mask),
        "corr_predicted_e_vs_target_e_bins": nan_corr(predicted_geom["predicted_e_shift_bins"], target_geom["e_tilt_bins"], valid_mask),
        "corr_simulated_e_vs_target_e_bins": nan_corr(simulated_geom["e_tilt_bins"], target_geom["e_tilt_bins"], valid_mask),
        "rmse_simulated_ef_fraction_vs_target": rmse(simulated_features["ef_fraction"], target_features["ef_fraction"], valid_mask=valid_mask),
    }

    run_dir = (args.output_dir / f"geometry_mapping_{case_label}_{run_date}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    figure_path = run_dir / f"geometry_prediction_comparison_{case_label}_{run_date}.png"
    log_progress("Rendering geometry comparison figure")
    save_geometry_figure(
        figure_path,
        avg_norm_map,
        region_map,
        valid_mask,
        base_geom,
        target_geom,
        predicted_geom,
        simulated_geom,
        case_label,
    )

    nc_path = run_dir / f"geometry_prediction_bundle_{case_label}_{run_date}.nc"
    log_progress("Writing geometry bundle NetCDF")
    geometry_ds = build_geometry_dataset(
        base_bundle.da,
        region_map,
        valid_mask,
        base_geom,
        target_geom,
        predicted_geom,
        simulated_geom,
        metrics,
    )
    geometry_ds.to_netcdf(nc_path, engine="h5netcdf")

    params_path = run_dir / f"region_params_{case_label}_{run_date}.json"
    params_path.write_text(json.dumps({"regions": [p.as_dict() for p in region_params]}, indent=2))
    np.save(run_dir / f"region_map_{case_label}_{run_date}.npy", region_map.astype(np.int32))
    np.save(run_dir / f"valid_mask_{case_label}_{run_date}.npy", valid_mask.astype(np.int8))
    np.save(run_dir / f"activation_map_{case_label}_{run_date}.npy", activation_map.astype(np.float32))

    report_path = run_dir / f"geometry_prediction_report_{case_label}_{run_date}.md"
    log_progress("Writing Markdown report")
    write_markdown_report(
        report_path,
        run_date,
        base_path,
        target_path,
        case_label,
        metrics,
        region_params,
        figure_name=figure_path.name,
        nc_name=nc_path.name,
    )

    log_progress("Geometry mapping complete")
    print(f"Output directory: {run_dir}")
    print(f"Figure: {figure_path}")
    print(f"Geometry NetCDF: {nc_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
