#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


DEFAULT_CASE_FILES = {
    "a": "a_convert_2_nosm.nc",
    "b": "b_convert_2_nosm.nc",
    "c": "c_convert_2_nosm.nc",
    "c2": "c2_convert_2_nosm.nc",
    "d": "d_convert_2_nosm.nc",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_data_dir = (script_dir / "../../data").resolve()
    default_output_dir = (script_dir / "../../outputs/geometry_simulation").resolve()

    parser = argparse.ArgumentParser(
        description=(
            "Simulate spatially varying nanoARPES geometry and current-response maps "
            "for the TaSe2/TaS2-style pulse sequence using file a as the reference state."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--base-case", choices=list(DEFAULT_CASE_FILES), default="a")
    parser.add_argument("--target-case", choices=list(DEFAULT_CASE_FILES), default="b")
    parser.add_argument(
        "--base-file",
        type=Path,
        default=None,
        help="Optional explicit path to the base .nc file. Overrides --base-case.",
    )
    parser.add_argument(
        "--target-file",
        type=Path,
        default=None,
        help="Optional explicit path to the target .nc file. Overrides --target-case.",
    )
    parser.add_argument("--fermi-level-ev", type=float, default=0.0)
    parser.add_argument("--ef-window-ev", type=float, default=0.05)
    parser.add_argument("--wide-window-ev", type=float, default=0.20)
    parser.add_argument("--n-regions", type=int, default=6)
    parser.add_argument(
        "--region-map",
        type=Path,
        default=None,
        help=(
            "Optional .npy file containing an integer region map with shape (x, y). "
            "If absent, a region map is inferred from file a."
        ),
    )
    parser.add_argument(
        "--region-params-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file that overrides the automatically inferred per-region "
            "tilt and current parameters."
        ),
    )
    parser.add_argument(
        "--cross-threshold-quantile", type=float, default=0.45,
        help="Threshold used to infer the valid device mask from total intensity maps."
    )
    parser.add_argument("--cross-row-fraction", type=float, default=0.18)
    parser.add_argument("--cross-col-fraction", type=float, default=0.18)
    parser.add_argument("--cross-background-quantile", type=float, default=0.10)
    parser.add_argument("--cross-pad", type=int, default=1)
    parser.add_argument(
        "--tilt-scale-phi",
        type=float,
        default=1.0,
        help=(
            "Global multiplicative factor applied to automatically inferred phi shifts."
        ),
    )
    parser.add_argument(
        "--tilt-scale-ev",
        type=float,
        default=0.35,
        help=(
            "Global multiplicative factor applied to automatically inferred energy shifts."
        ),
    )
    parser.add_argument(
        "--current-gain",
        type=float,
        default=1.0,
        help=(
            "Global multiplicative factor on the inferred current-response activation."
        ),
    )
    parser.add_argument(
        "--smooth-region-iterations",
        type=int,
        default=1,
        help="Number of majority-smoothing iterations applied to the inferred region map.",
    )
    parser.add_argument(
        "--save-region-products",
        action="store_true",
        help="Save region map and region parameters as standalone files.",
    )
    return parser.parse_args()


@dataclass
class DatasetBundle:
    path: Path
    ds: xr.Dataset
    da: xr.DataArray


@dataclass
class RegionParams:
    region_id: int
    phi_shift_bins: float
    energy_shift_bins: float
    current_mix: float
    name: str

    def as_dict(self) -> dict:
        return {
            "region_id": int(self.region_id),
            "phi_shift_bins": float(self.phi_shift_bins),
            "energy_shift_bins": float(self.energy_shift_bins),
            "current_mix": float(self.current_mix),
            "name": self.name,
        }


def resolve_case_path(data_dir: Path, case_name: str, explicit_path: Optional[Path]) -> Path:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()
    return (data_dir / DEFAULT_CASE_FILES[case_name]).resolve()


def open_nc_dataset(file_path: Path) -> xr.Dataset:
    engines_to_try = ["h5netcdf", "scipy", None, "netcdf4"]
    last_error = None
    for eng in engines_to_try:
        try:
            label = "default" if eng is None else eng
            log_progress(f"Trying xarray engine={label} for {file_path.name}")
            if eng is None:
                ds = xr.open_dataset(file_path)
            else:
                ds = xr.open_dataset(file_path, engine=eng)
            log_progress(f"Opened {file_path.name} with engine={label}")
            return ds
        except Exception as exc:  # pragma: no cover - defensive fallback
            last_error = exc
            log_progress(f"Engine {label} failed for {file_path.name}: {exc}")
    raise RuntimeError(f"Could not open dataset {file_path}: {last_error}")


def get_main_dataarray(ds: xr.Dataset) -> xr.DataArray:
    candidates: List[Tuple[str, int]] = []
    for name, var in ds.data_vars.items():
        try:
            if np.issubdtype(var.dtype, np.number):
                candidates.append((name, int(np.prod(var.shape))))
        except Exception:
            continue
    if not candidates:
        raise ValueError("No numeric data variables found in dataset.")
    candidates.sort(key=lambda item: item[1], reverse=True)
    return ds[candidates[0][0]]


def require_dims(da: xr.DataArray, required: Tuple[str, ...] = ("x", "y", "eV", "phi")) -> None:
    missing = [dim for dim in required if dim not in da.dims]
    if missing:
        raise ValueError(f"Missing required dimensions {missing}; found {da.dims}.")


def load_dataset_bundle(file_path: Path) -> DatasetBundle:
    ds = open_nc_dataset(file_path)
    da = get_main_dataarray(ds)
    require_dims(da)
    return DatasetBundle(path=file_path, ds=ds, da=da)


def to_float32_numpy(da: xr.DataArray) -> np.ndarray:
    return np.asarray(da.values, dtype=np.float32)


def get_energy_mask(e_axis: np.ndarray, center: float, halfwidth: float) -> np.ndarray:
    e_axis = np.asarray(e_axis, dtype=np.float32)
    return np.abs(e_axis - center) <= halfwidth


def total_and_ef_maps(da: xr.DataArray, fermi_level: float, ef_window: float) -> Tuple[np.ndarray, np.ndarray]:
    e_axis = np.asarray(da.coords["eV"].values, dtype=np.float32)
    ef_mask = get_energy_mask(e_axis, center=fermi_level, halfwidth=ef_window)
    total_map = da.sum(dim=("eV", "phi")).values.astype(np.float32)
    ef_map = da.isel(eV=np.where(ef_mask)[0]).sum(dim=("eV", "phi")).values.astype(np.float32)
    return total_map, ef_map


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return a / (b + eps)


def normalize_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    s = np.asarray(v, dtype=np.float32)
    total = s.sum()
    if total <= eps:
        return np.zeros_like(s, dtype=np.float32)
    return s / total


def finite_fill(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    out = np.array(arr, copy=True)
    out[~np.isfinite(out)] = fill_value
    return out


def robust_zscore(arr: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    med = np.nanmedian(arr, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(arr - med), axis=axis, keepdims=True)
    return (arr - med) / (1.4826 * mad + eps)


def dilate_mask(mask: np.ndarray, n_iter: int = 1) -> np.ndarray:
    out = mask.astype(bool).copy()
    for _ in range(max(0, int(n_iter))):
        padded = np.pad(out, 1, mode="edge")
        new_mask = np.zeros_like(out, dtype=bool)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                new_mask |= padded[1 + di : 1 + di + out.shape[0], 1 + dj : 1 + dj + out.shape[1]]
        out = new_mask
    return out


def build_cross_mask_from_maps(
    total_maps: Iterable[np.ndarray],
    threshold_quantile: float,
    row_fraction: float,
    col_fraction: float,
    background_quantile: float,
    pad: int,
) -> Tuple[np.ndarray, np.ndarray]:
    normalized = []
    for m in total_maps:
        arr = np.asarray(m, dtype=np.float32)
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if hi > lo:
            normalized.append((arr - lo) / (hi - lo))
        else:
            normalized.append(np.zeros_like(arr, dtype=np.float32))
    avg = np.mean(normalized, axis=0)
    th = np.quantile(avg.reshape(-1), threshold_quantile)
    active = avg >= th
    row_occ = active.mean(axis=1)
    col_occ = active.mean(axis=0)
    strong_rows = row_occ >= row_fraction
    strong_cols = col_occ >= col_fraction
    cross_mask = strong_rows[:, None] | strong_cols[None, :]
    bg_th = np.quantile(avg.reshape(-1), background_quantile)
    cross_mask = cross_mask & (avg >= bg_th)
    if pad > 0:
        cross_mask = dilate_mask(cross_mask, n_iter=pad)
    return cross_mask.astype(bool), avg.astype(np.float32)


def extract_pixel_features(
    da: xr.DataArray,
    fermi_level: float,
    ef_window: float,
    wide_window: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    data = to_float32_numpy(da)
    x_n, y_n, e_n, p_n = data.shape
    e_axis = np.asarray(da.coords["eV"].values, dtype=np.float32)
    p_axis = np.asarray(da.coords["phi"].values, dtype=np.float32)
    ef_mask = get_energy_mask(e_axis, center=fermi_level, halfwidth=ef_window)
    wide_mask = get_energy_mask(e_axis, center=fermi_level, halfwidth=wide_window)

    flat = data.reshape(x_n * y_n, e_n, p_n)
    total_intensity = flat.sum(axis=(1, 2))
    ef_intensity = flat[:, ef_mask, :].sum(axis=(1, 2))
    wide_intensity = flat[:, wide_mask, :].sum(axis=(1, 2))
    ef_fraction = safe_divide(ef_intensity, total_intensity)
    wide_fraction = safe_divide(wide_intensity, total_intensity)

    e_profile = flat.sum(axis=2)
    p_profile = flat.sum(axis=1)
    e_profile_norm = np.stack([normalize_vector(v) for v in e_profile], axis=0)
    p_profile_norm = np.stack([normalize_vector(v) for v in p_profile], axis=0)
    e_centroid = (e_profile_norm * e_axis[None, :]).sum(axis=1)
    e_var = (e_profile_norm * (e_axis[None, :] - e_centroid[:, None]) ** 2).sum(axis=1)
    phi_centroid = (p_profile_norm * p_axis[None, :]).sum(axis=1)
    phi_var = (p_profile_norm * (p_axis[None, :] - phi_centroid[:, None]) ** 2).sum(axis=1)

    p_mid = len(p_axis) // 2
    left_intensity = p_profile[:, :p_mid].sum(axis=1)
    right_intensity = p_profile[:, p_mid:].sum(axis=1)
    phi_asymmetry = safe_divide(right_intensity - left_intensity, right_intensity + left_intensity)

    flat_all = flat.reshape(x_n * y_n, -1)
    flat_norm = np.stack([normalize_vector(v) for v in flat_all], axis=0)
    spectral_entropy = -np.sum(flat_norm * np.log(flat_norm + 1e-12), axis=1)
    spectral_sharpness = safe_divide(flat_all.max(axis=1), flat_all.mean(axis=1))

    ef_map = ef_intensity.reshape(x_n, y_n)
    gx, gy = np.gradient(ef_map)
    grad_mag = np.sqrt(gx**2 + gy**2)

    neighbor_diff = np.zeros_like(ef_map, dtype=np.float32)
    local_contrast = np.zeros_like(ef_map, dtype=np.float32)
    padded = np.pad(ef_map, 1, mode="reflect")
    for i in range(x_n):
        for j in range(y_n):
            vals = []
            c = ef_map[i, j]
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ii, jj = i + di, j + dj
                if 0 <= ii < x_n and 0 <= jj < y_n:
                    vals.append(abs(c - ef_map[ii, jj]))
            neighbor_diff[i, j] = np.mean(vals) if vals else 0.0
            patch = padded[i : i + 3, j : j + 3]
            local_contrast[i, j] = patch.std()

    features = {
        "total_intensity": total_intensity.reshape(x_n, y_n),
        "ef_intensity": ef_intensity.reshape(x_n, y_n),
        "wide_intensity": wide_intensity.reshape(x_n, y_n),
        "ef_fraction": ef_fraction.reshape(x_n, y_n),
        "wide_fraction": wide_fraction.reshape(x_n, y_n),
        "e_centroid": e_centroid.reshape(x_n, y_n),
        "e_var": e_var.reshape(x_n, y_n),
        "phi_centroid": phi_centroid.reshape(x_n, y_n),
        "phi_var": phi_var.reshape(x_n, y_n),
        "phi_asymmetry": phi_asymmetry.reshape(x_n, y_n),
        "spectral_entropy": spectral_entropy.reshape(x_n, y_n),
        "spectral_sharpness": spectral_sharpness.reshape(x_n, y_n),
        "ef_grad_mag": grad_mag.astype(np.float32),
        "ef_neighbor_diff": neighbor_diff,
        "ef_local_contrast": local_contrast,
    }
    feat_names = [
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
    feat_matrix = np.stack([features[name].reshape(-1) for name in feat_names], axis=1).astype(np.float32)
    feat_matrix = finite_fill(feat_matrix, 0.0)
    return features, feat_matrix, feat_names


def kmeans_numpy(
    x: np.ndarray,
    n_clusters: int,
    n_iter: int = 100,
    n_init: int = 8,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32)
    n, d = x.shape
    best_inertia = None
    best_labels = None
    best_centroids = None
    for _ in range(n_init):
        indices = rng.choice(n, size=n_clusters, replace=False)
        centroids = x[indices].copy()
        for _ in range(n_iter):
            dists = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = dists.argmin(axis=1)
            new_centroids = np.empty_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centroids[k] = x[mask].mean(axis=0)
                else:
                    new_centroids[k] = x[rng.integers(0, n)]
            if np.allclose(new_centroids, centroids, atol=1e-5):
                centroids = new_centroids
                break
            centroids = new_centroids
        dists = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        inertia = float(np.sum(dists[np.arange(n), labels]))
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()
    assert best_labels is not None and best_centroids is not None
    return best_labels.astype(np.int32), best_centroids.astype(np.float32)


def smooth_region_map(region_map: np.ndarray, valid_mask: np.ndarray, n_iter: int) -> np.ndarray:
    out = region_map.copy()
    x_n, y_n = out.shape
    for _ in range(max(0, int(n_iter))):
        new = out.copy()
        for i in range(x_n):
            for j in range(y_n):
                if not valid_mask[i, j]:
                    continue
                neigh = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                    ii, jj = i + di, j + dj
                    if 0 <= ii < x_n and 0 <= jj < y_n and valid_mask[ii, jj]:
                        neigh.append(int(out[ii, jj]))
                if neigh:
                    values, counts = np.unique(np.asarray(neigh), return_counts=True)
                    new[i, j] = int(values[np.argmax(counts)])
        out = new
    return out


def infer_region_map(
    base_features: Dict[str, np.ndarray],
    feat_matrix: np.ndarray,
    valid_mask: np.ndarray,
    n_regions: int,
    smooth_iterations: int,
) -> np.ndarray:
    z = robust_zscore(feat_matrix, axis=0)
    z = finite_fill(z, 0.0)
    valid_flat = valid_mask.reshape(-1)
    useful_cols = [3, 5, 7, 8, 9, 12, 13, 14]
    labels, _ = kmeans_numpy(z[valid_flat][:, useful_cols], n_clusters=n_regions, seed=42)
    region_map = np.full(valid_mask.size, fill_value=-1, dtype=np.int32)
    region_map[valid_flat] = labels
    region_map = region_map.reshape(valid_mask.shape)

    ef = base_features["ef_fraction"]
    region_scores = []
    for rid in sorted(np.unique(region_map[valid_mask])):
        region_scores.append((int(rid), float(np.mean(ef[region_map == rid]))))
    region_scores.sort(key=lambda item: item[1])
    remap = {old: new for new, (old, _) in enumerate(region_scores)}
    for old, new in remap.items():
        region_map[region_map == old] = new
    return smooth_region_map(region_map, valid_mask, n_iter=smooth_iterations)


def shift_last_axis_constant(arr: np.ndarray, shift_bins: float) -> np.ndarray:
    if abs(shift_bins) < 1e-8:
        return arr.copy()
    n = arr.shape[-1]
    coords = np.arange(n, dtype=np.float32)
    source = coords - np.float32(shift_bins)
    source = np.clip(source, 0.0, n - 1.0)
    left = np.floor(source).astype(np.int32)
    right = np.clip(left + 1, 0, n - 1)
    frac = (source - left).astype(np.float32)
    arr32 = np.asarray(arr, dtype=np.float32)
    left_vals = np.take(arr32, left, axis=-1)
    right_vals = np.take(arr32, right, axis=-1)
    reshape = (1,) * (arr32.ndim - 1) + (n,)
    frac = frac.reshape(reshape)
    return ((1.0 - frac) * left_vals + frac * right_vals).astype(np.float32)


def warp_region_spectra(cube: np.ndarray, region_map: np.ndarray, params: List[RegionParams], valid_mask: np.ndarray, label: str = "cube") -> np.ndarray:
    out = cube.copy()
    total_regions = len(params)
    log_progress(f"Warping {label}: starting {total_regions} region(s)")
    for idx, param in enumerate(params, start=1):
        mask = (region_map == param.region_id) & valid_mask
        pixel_count = int(np.count_nonzero(mask))
        if pixel_count == 0:
            log_progress(f"Warping {label}: region {param.region_id} skipped (0 valid pixels)")
            continue
        log_progress(
            f"Warping {label}: region {param.region_id} ({idx}/{total_regions}), "
            f"pixels={pixel_count}, phi_shift={param.phi_shift_bins:.3f}, "
            f"energy_shift={param.energy_shift_bins:.3f}"
        )
        spectra = out[mask]
        warped = spectra
        if abs(param.energy_shift_bins) > 1e-8:
            warped = shift_last_axis_constant(np.swapaxes(warped, 1, 2), param.energy_shift_bins)
            warped = np.swapaxes(warped, 1, 2)
        if abs(param.phi_shift_bins) > 1e-8:
            warped = shift_last_axis_constant(warped, param.phi_shift_bins)
        out[mask] = warped
    log_progress(f"Warping {label}: completed")
    return out


def compute_current_activation(
    base_features: Dict[str, np.ndarray],
    target_features: Dict[str, np.ndarray],
    region_map: np.ndarray,
    valid_mask: np.ndarray,
    current_gain: float,
) -> np.ndarray:
    if target_features is None:
        activation = np.zeros_like(base_features["ef_fraction"], dtype=np.float32)
        activation[valid_mask] = 0.0
        return activation

    delta_ef = target_features["ef_fraction"] - base_features["ef_fraction"]
    delta_wide = target_features["wide_fraction"] - base_features["wide_fraction"]
    delta_entropy = target_features["spectral_entropy"] - base_features["spectral_entropy"]
    response = 0.70 * delta_ef + 0.20 * delta_wide - 0.05 * delta_entropy
    response = response.astype(np.float32)
    valid_vals = response[valid_mask]
    lo = float(np.quantile(valid_vals, 0.05))
    hi = float(np.quantile(valid_vals, 0.95))
    scaled = np.clip((response - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    region_scaled = scaled.copy()
    for rid in np.unique(region_map[valid_mask]):
        mask = (region_map == rid) & valid_mask
        region_scaled[mask] = float(np.mean(scaled[mask]))
    region_scaled = np.clip(region_scaled * np.float32(current_gain), 0.0, 1.0)
    region_scaled[~valid_mask] = 0.0
    return region_scaled.astype(np.float32)


def infer_region_params(
    base_features: Dict[str, np.ndarray],
    target_features: Optional[Dict[str, np.ndarray]],
    region_map: np.ndarray,
    valid_mask: np.ndarray,
    phi_axis: np.ndarray,
    e_axis: np.ndarray,
    tilt_scale_phi: float,
    tilt_scale_ev: float,
    current_gain: float,
) -> Tuple[List[RegionParams], np.ndarray]:
    global_phi = float(np.mean(base_features["phi_centroid"][valid_mask]))
    global_e = float(np.mean(base_features["e_centroid"][valid_mask]))
    dphi = float(np.median(np.diff(phi_axis)))
    de = float(np.median(np.diff(e_axis)))

    activation = compute_current_activation(base_features, target_features, region_map, valid_mask, current_gain)
    params: List[RegionParams] = []
    for rid in sorted(np.unique(region_map[valid_mask]).tolist()):
        mask = (region_map == rid) & valid_mask
        region_phi = float(np.mean(base_features["phi_centroid"][mask]))
        region_e = float(np.mean(base_features["e_centroid"][mask]))
        phi_shift_bins = tilt_scale_phi * (region_phi - global_phi) / (dphi + 1e-8)
        energy_shift_bins = tilt_scale_ev * (region_e - global_e) / (de + 1e-8)
        current_mix = float(np.mean(activation[mask]))
        params.append(
            RegionParams(
                region_id=int(rid),
                phi_shift_bins=float(phi_shift_bins),
                energy_shift_bins=float(energy_shift_bins),
                current_mix=current_mix,
                name=f"region_{int(rid)}",
            )
        )
    return params, activation


def apply_region_param_overrides(params: List[RegionParams], override_json: Path) -> List[RegionParams]:
    payload = json.loads(override_json.read_text())
    lookup = {int(p.region_id): p for p in params}
    for raw in payload.get("regions", payload if isinstance(payload, list) else []):
        rid = int(raw["region_id"])
        if rid not in lookup:
            continue
        cur = lookup[rid]
        if "phi_shift_bins" in raw:
            cur.phi_shift_bins = float(raw["phi_shift_bins"])
        if "energy_shift_bins" in raw:
            cur.energy_shift_bins = float(raw["energy_shift_bins"])
        if "current_mix" in raw:
            cur.current_mix = float(raw["current_mix"])
        if "name" in raw:
            cur.name = str(raw["name"])
    return [lookup[rid] for rid in sorted(lookup)]


def build_simulated_cube(
    base_cube: np.ndarray,
    base_warped: np.ndarray,
    target_cube: Optional[np.ndarray],
    target_warped: Optional[np.ndarray],
    region_map: np.ndarray,
    valid_mask: np.ndarray,
    region_params: List[RegionParams],
    activation_map: np.ndarray,
) -> np.ndarray:
    simulated = base_warped.copy()
    if target_warped is None:
        simulated[~valid_mask] = base_cube[~valid_mask]
        return simulated

    current_mix_by_region = {p.region_id: p.current_mix for p in region_params}
    for rid, mix in current_mix_by_region.items():
        mask = (region_map == rid) & valid_mask
        if not np.any(mask):
            continue
        local_activation = np.clip(activation_map[mask] * np.float32(mix + 1e-8), 0.0, 1.0).astype(np.float32)
        local_activation = local_activation[:, None, None]
        simulated[mask] = (1.0 - local_activation) * base_warped[mask] + local_activation * target_warped[mask]
    simulated[~valid_mask] = base_cube[~valid_mask]
    return simulated.astype(np.float32)


def cube_to_dataarray(template_da: xr.DataArray, cube: np.ndarray, name: str) -> xr.DataArray:
    sim_da = xr.DataArray(
        cube.astype(np.float32),
        dims=template_da.dims,
        coords={coord: template_da.coords[coord] for coord in template_da.coords},
        attrs=dict(template_da.attrs),
        name=name,
    )
    return sim_da


def rmse(a: np.ndarray, b: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> float:
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    if valid_mask is not None:
        mask = valid_mask.astype(bool)
        diff = arr_a[mask] - arr_b[mask]
    else:
        diff = arr_a - arr_b
    return float(np.sqrt(np.mean(diff**2)))


def plot_summary(
    out_png: Path,
    base_features: Dict[str, np.ndarray],
    target_features: Optional[Dict[str, np.ndarray]],
    sim_features: Dict[str, np.ndarray],
    region_map: np.ndarray,
    valid_mask: np.ndarray,
    avg_norm_map: np.ndarray,
    activation_map: np.ndarray,
    case_label: str,
) -> None:
    region_display = region_map.astype(float)
    region_display[~valid_mask] = np.nan
    base_ef = base_features["ef_fraction"].copy()
    base_ef[~valid_mask] = np.nan
    sim_ef = sim_features["ef_fraction"].copy()
    sim_ef[~valid_mask] = np.nan

    if target_features is not None:
        target_ef = target_features["ef_fraction"].copy()
        target_ef[~valid_mask] = np.nan
        diff = sim_ef - target_ef
        diff_v = np.nanpercentile(np.abs(diff[valid_mask]), 99)
    else:
        target_ef = np.full_like(sim_ef, np.nan)
        diff = sim_ef - base_ef
        diff_v = np.nanpercentile(np.abs(diff[valid_mask]), 99)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    im0 = axes[0, 0].imshow(avg_norm_map.T, origin="lower", cmap="magma")
    axes[0, 0].set_title("Average normalized total intensity")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(region_display.T, origin="lower", cmap="tab10")
    axes[0, 1].set_title("Region map")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(activation_map.T, origin="lower", cmap="viridis")
    axes[0, 2].set_title(f"Current activation: {case_label}")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im3 = axes[1, 0].imshow(base_ef.T, origin="lower", cmap="viridis")
    axes[1, 0].set_title("Base near-EF fraction")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(sim_ef.T, origin="lower", cmap="viridis")
    axes[1, 1].set_title("Simulated near-EF fraction")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im5 = axes[1, 2].imshow(diff.T, origin="lower", cmap="coolwarm", vmin=-diff_v, vmax=diff_v)
    title = "Simulated - target near-EF fraction" if target_features is not None else "Simulated - base near-EF fraction"
    axes[1, 2].set_title(title)
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def ensure_region_map(path: Path, expected_shape: Tuple[int, int]) -> np.ndarray:
    region_map = np.load(path)
    if region_map.shape != expected_shape:
        raise ValueError(f"Region map shape {region_map.shape} does not match expected {expected_shape}.")
    return region_map.astype(np.int32)


def save_outputs(
    output_dir: Path,
    run_date: str,
    target_case: str,
    sim_da: xr.DataArray,
    region_map: np.ndarray,
    valid_mask: np.ndarray,
    activation_map: np.ndarray,
    region_params: List[RegionParams],
    metrics: dict,
    figure_path: Path,
    save_region_products: bool,
) -> Path:
    run_dir = output_dir / f"geometry_simulation_{target_case}_{run_date}"
    run_dir.mkdir(parents=True, exist_ok=True)

    ds_out = xr.Dataset(
        {
            "simulated_intensity": sim_da,
            "region_map": xr.DataArray(region_map.astype(np.int32), dims=("x", "y"), coords={"x": sim_da.coords["x"], "y": sim_da.coords["y"]}),
            "valid_mask": xr.DataArray(valid_mask.astype(np.int8), dims=("x", "y"), coords={"x": sim_da.coords["x"], "y": sim_da.coords["y"]}),
            "activation_map": xr.DataArray(activation_map.astype(np.float32), dims=("x", "y"), coords={"x": sim_da.coords["x"], "y": sim_da.coords["y"]}),
        }
    )
    ds_out.attrs["run_date"] = run_date
    ds_out.attrs["description"] = "Geometry/current simulation generated from nanoARPES base state and pulse-state templates."
    for key, value in metrics.items():
        ds_out.attrs[key] = value

    nc_path = run_dir / f"simulated_{target_case}_{run_date}.nc"
    ds_out.to_netcdf(nc_path, engine="h5netcdf")

    metrics_path = run_dir / f"metrics_{target_case}_{run_date}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    params_payload = {"regions": [p.as_dict() for p in region_params]}
    params_path = run_dir / f"region_params_{target_case}_{run_date}.json"
    params_path.write_text(json.dumps(params_payload, indent=2))

    if save_region_products:
        np.save(run_dir / f"region_map_{target_case}_{run_date}.npy", region_map.astype(np.int32))
        np.save(run_dir / f"valid_mask_{target_case}_{run_date}.npy", valid_mask.astype(np.int8))
        np.save(run_dir / f"activation_map_{target_case}_{run_date}.npy", activation_map.astype(np.float32))

    copied_figure = run_dir / figure_path.name
    if copied_figure.resolve() != figure_path.resolve():
        copied_figure.write_bytes(figure_path.read_bytes())

    return nc_path


def main() -> None:
    args = parse_args()
    run_date = datetime.now().strftime("%Y_%m_%d")

    base_path = resolve_case_path(args.data_dir, args.base_case, args.base_file)
    target_path = resolve_case_path(args.data_dir, args.target_case, args.target_file)
    log_progress(f"Starting geometry simulation run for target case '{args.target_case}'")
    log_progress(f"Base path resolved to: {base_path}")
    log_progress(f"Target path resolved to: {target_path}")

    log_progress("Loading base dataset")
    base_bundle = load_dataset_bundle(base_path)
    target_bundle = None
    if target_path != base_path:
        log_progress("Loading target dataset")
        target_bundle = load_dataset_bundle(target_path)
    else:
        log_progress("Base and target are identical; running geometry-only mode")

    log_progress("Materializing base cube into NumPy")
    base_cube = to_float32_numpy(base_bundle.da)
    target_cube = None
    if target_bundle is not None:
        log_progress("Materializing target cube into NumPy")
        target_cube = to_float32_numpy(target_bundle.da)

    log_progress("Computing mask inputs from integrated intensity maps")
    base_total, _ = total_and_ef_maps(base_bundle.da, args.fermi_level_ev, args.ef_window_ev)
    maps_for_mask = [base_total]
    if target_bundle is not None:
        target_total, _ = total_and_ef_maps(target_bundle.da, args.fermi_level_ev, args.ef_window_ev)
        maps_for_mask.append(target_total)

    valid_mask, avg_norm_map = build_cross_mask_from_maps(
        maps_for_mask,
        threshold_quantile=args.cross_threshold_quantile,
        row_fraction=args.cross_row_fraction,
        col_fraction=args.cross_col_fraction,
        background_quantile=args.cross_background_quantile,
        pad=args.cross_pad,
    )
    log_progress(f"Valid-mask inference complete: {int(valid_mask.sum())} valid pixels, {int((~valid_mask).sum())} excluded pixels")

    log_progress("Extracting base spectral features")
    base_features, base_feat_matrix, _ = extract_pixel_features(
        base_bundle.da,
        fermi_level=args.fermi_level_ev,
        ef_window=args.ef_window_ev,
        wide_window=args.wide_window_ev,
    )

    if target_bundle is not None:
        log_progress("Extracting target spectral features")
        target_features, _, _ = extract_pixel_features(
            target_bundle.da,
            fermi_level=args.fermi_level_ev,
            ef_window=args.ef_window_ev,
            wide_window=args.wide_window_ev,
        )
    else:
        target_features = None

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
    log_progress("Inferring per-region tilt and current-response parameters")
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
        log_progress(f"Applying region-parameter overrides from {args.region_params_json}")
        region_params = apply_region_param_overrides(region_params, args.region_params_json)

    log_progress("Warping base cube by inferred region tilts")
    base_warped = warp_region_spectra(base_cube, region_map, region_params, valid_mask, label="base")
    target_warped = None
    if target_cube is not None:
        log_progress("Warping target cube by inferred region tilts")
        target_warped = warp_region_spectra(target_cube, region_map, region_params, valid_mask, label="target")

    log_progress("Building simulated cube from warped base and target states")
    sim_cube = build_simulated_cube(
        base_cube,
        base_warped,
        target_cube,
        target_warped,
        region_map,
        valid_mask,
        region_params,
        activation_map,
    )
    sim_da = cube_to_dataarray(base_bundle.da, sim_cube, name="simulated_intensity")
    log_progress("Extracting simulated spectral features")
    sim_features, _, _ = extract_pixel_features(
        sim_da,
        fermi_level=args.fermi_level_ev,
        ef_window=args.ef_window_ev,
        wide_window=args.wide_window_ev,
    )

    case_label = args.target_case if target_bundle is not None else args.base_case
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = args.output_dir / f"geometry_simulation_summary_{case_label}_{run_date}.png"
    log_progress("Rendering summary figure")
    plot_summary(
        fig_path,
        base_features,
        target_features,
        sim_features,
        region_map,
        valid_mask,
        avg_norm_map,
        activation_map,
        case_label=case_label,
    )

    metrics = {
        "run_date": run_date,
        "base_file": str(base_path),
        "target_file": str(target_path),
        "base_case": args.base_case,
        "target_case": args.target_case,
        "valid_pixels": int(valid_mask.sum()),
        "excluded_pixels": int((~valid_mask).sum()),
        "n_regions": int(len(region_params)),
        "rmse_sim_vs_target_ef_fraction": None,
        "rmse_sim_vs_target_total_intensity": None,
        "rmse_geometry_only_vs_base_ef_fraction": rmse(
            base_features["ef_fraction"],
            sim_features["ef_fraction"],
            valid_mask=valid_mask,
        ) if target_bundle is None else rmse(
            base_features["ef_fraction"],
            extract_pixel_features(cube_to_dataarray(base_bundle.da, base_warped, "base_warped"), args.fermi_level_ev, args.ef_window_ev, args.wide_window_ev)[0]["ef_fraction"],
            valid_mask=valid_mask,
        ),
    }
    if target_bundle is not None:
        metrics["rmse_sim_vs_target_ef_fraction"] = rmse(sim_features["ef_fraction"], target_features["ef_fraction"], valid_mask=valid_mask)
        metrics["rmse_sim_vs_target_total_intensity"] = rmse(sim_features["total_intensity"], target_features["total_intensity"], valid_mask=valid_mask)

    log_progress("Writing NetCDF and run metadata to disk")
    nc_path = save_outputs(
        output_dir=args.output_dir,
        run_date=run_date,
        target_case=case_label,
        sim_da=sim_da,
        region_map=region_map,
        valid_mask=valid_mask,
        activation_map=activation_map,
        region_params=region_params,
        metrics=metrics,
        figure_path=fig_path,
        save_region_products=args.save_region_products,
    )

    log_progress("Simulation complete.")
    print(f"Base file:   {base_path}")
    print(f"Target file: {target_path}")
    print(f"Output NetCDF: {nc_path}")
    print(f"Summary figure: {fig_path}")
    print("Region parameters:")
    for param in region_params:
        print(
            f"  region {param.region_id}: "
            f"phi_shift_bins={param.phi_shift_bins:.3f}, "
            f"energy_shift_bins={param.energy_shift_bins:.3f}, "
            f"current_mix={param.current_mix:.3f}"
        )


if __name__ == "__main__":
    main()
