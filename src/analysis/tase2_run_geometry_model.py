#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from tase2_train_geometry_model import (
    GeometryPatchCNN,
    choose_device,
    extract_cube_feature_maps,
    get_main_dataarray,
    open_dataset,
    require_dims,
    resolve_nc,
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained PyTorch geometry model on a geometry-mapping bundle and simulated cube, "
            "then save predicted geometry maps, figures, and metrics."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mapping-path", type=Path, required=True)
    parser.add_argument("--simulation-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=(script_dir / "../../outputs/geometry_ml").resolve())
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def build_inputs(mapping_ds: xr.Dataset, sim_da: xr.DataArray, feature_names, feature_stats) -> Tuple[np.ndarray, np.ndarray]:
    require_dims(sim_da)
    valid_mask = mapping_ds["valid_mask"].values.astype(bool)
    sim_features = extract_cube_feature_maps(sim_da)
    source_maps = {
        **sim_features,
        "base_phi_tilt_bins": mapping_ds["base_phi_tilt_bins"].values.astype(np.float32),
        "base_e_tilt_bins": mapping_ds["base_e_tilt_bins"].values.astype(np.float32),
        "predicted_phi_shift_bins": mapping_ds["predicted_predicted_phi_shift_bins"].values.astype(np.float32),
        "predicted_e_shift_bins": mapping_ds["predicted_predicted_e_shift_bins"].values.astype(np.float32),
        "predicted_current_mix": mapping_ds["predicted_predicted_current_mix"].values.astype(np.float32),
    }
    maps = []
    for name in feature_names:
        arr = source_maps[name].astype(np.float32)
        mean = feature_stats[name]["mean"]
        std = feature_stats[name]["std"]
        z = ((arr - mean) / std).astype(np.float32)
        z[~valid_mask] = 0.0
        maps.append(z)
    return np.stack(maps, axis=0), valid_mask


class PatchInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, input_maps: np.ndarray, coords: np.ndarray, patch_radius: int) -> None:
        self.coords = coords.astype(np.int64)
        self.r = int(patch_radius)
        self.input_padded = np.pad(
            input_maps,
            ((0, 0), (self.r, self.r), (self.r, self.r)),
            mode="reflect",
        ).astype(np.float32)

    def __len__(self) -> int:
        return self.coords.shape[0]

    def __getitem__(self, idx: int):
        x, y = self.coords[idx]
        patch = self.input_padded[:, x:x + 2 * self.r + 1, y:y + 2 * self.r + 1]
        return torch.from_numpy(patch), torch.tensor([x, y], dtype=torch.int64)


@torch.no_grad()
def predict_maps(model, loader, device, target_mean, target_std, shape):
    pred_phi = np.full(shape, np.nan, dtype=np.float32)
    pred_e = np.full(shape, np.nan, dtype=np.float32)
    model.eval()
    for xb, coordb in loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        pred = pred * target_std[None, :] + target_mean[None, :]
        coords = coordb.numpy()
        pred_phi[coords[:, 0], coords[:, 1]] = pred[:, 0]
        pred_e[coords[:, 0], coords[:, 1]] = pred[:, 1]
    return pred_phi, pred_e


def corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    aa = a[mask].astype(np.float64)
    bb = b[mask].astype(np.float64)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = np.sqrt(np.sum(aa ** 2) * np.sum(bb ** 2))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(aa * bb) / denom)


def rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def percentile_lim(arr: np.ndarray, mask: np.ndarray, q: float = 99.0, fallback: float = 1.0) -> float:
    vals = np.abs(arr[np.isfinite(arr) & mask])
    if vals.size == 0:
        return fallback
    out = float(np.nanpercentile(vals, q))
    return out if out > 1e-12 else fallback


def save_figure(out_path: Path, valid_mask: np.ndarray, pred_phi: np.ndarray, pred_e: np.ndarray,
                target_phi: np.ndarray, target_e: np.ndarray, base_phi: np.ndarray, base_e: np.ndarray) -> None:
    phi_res = pred_phi - target_phi
    e_res = pred_e - target_e
    phi_lim = max(percentile_lim(pred_phi, valid_mask), percentile_lim(target_phi, valid_mask), percentile_lim(base_phi, valid_mask))
    e_lim = max(percentile_lim(pred_e, valid_mask), percentile_lim(target_e, valid_mask), percentile_lim(base_e, valid_mask))
    phi_res_lim = percentile_lim(phi_res, valid_mask)
    e_res_lim = percentile_lim(e_res, valid_mask)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    panels = [
        (base_phi, "coolwarm", -phi_lim, phi_lim, "Base phi-tilt"),
        (target_phi, "coolwarm", -phi_lim, phi_lim, "Target experimental phi-tilt"),
        (pred_phi, "coolwarm", -phi_lim, phi_lim, "ML-predicted phi-tilt"),
        (phi_res, "coolwarm", -phi_res_lim, phi_res_lim, "Prediction residual: phi"),
        (base_e, "coolwarm", -e_lim, e_lim, "Base energy-tilt"),
        (target_e, "coolwarm", -e_lim, e_lim, "Target experimental energy-tilt"),
        (pred_e, "coolwarm", -e_lim, e_lim, "ML-predicted energy-tilt"),
        (e_res, "coolwarm", -e_res_lim, e_res_lim, "Prediction residual: energy"),
    ]
    for ax, (arr, cmap, vmin, vmax, title) in zip(axes.reshape(-1), panels):
        show = arr.copy()
        show[~valid_mask] = np.nan
        im = ax.imshow(show.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    log(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu")
    mapping_nc = resolve_nc(args.mapping_path, "geometry_prediction_bundle")
    sim_nc = resolve_nc(args.simulation_path, "simulated_")
    mapping_ds = open_dataset(mapping_nc)
    sim_ds = open_dataset(sim_nc)
    sim_da = get_main_dataarray(sim_ds)

    feature_names = checkpoint["feature_names"]
    feature_stats = checkpoint["feature_stats"]
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    patch_radius = int(checkpoint["patch_radius"])

    log("Building input tensors")
    input_maps, valid_mask = build_inputs(mapping_ds, sim_da, feature_names, feature_stats)
    coords = np.argwhere(valid_mask)
    infer_ds = PatchInferenceDataset(input_maps, coords, patch_radius)
    infer_loader = torch.utils.data.DataLoader(infer_ds, batch_size=args.batch_size, shuffle=False)

    model = GeometryPatchCNN(in_channels=int(checkpoint["in_channels"]), hidden_channels=int(checkpoint["hidden_channels"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    log("Running model inference")
    pred_phi, pred_e = predict_maps(model, infer_loader, device, target_mean, target_std, valid_mask.shape)

    target_phi = mapping_ds["target_phi_tilt_bins"].values.astype(np.float32)
    target_e = mapping_ds["target_e_tilt_bins"].values.astype(np.float32)
    base_phi = mapping_ds["base_phi_tilt_bins"].values.astype(np.float32)
    base_e = mapping_ds["base_e_tilt_bins"].values.astype(np.float32)

    metrics = {
        "rmse_phi": rmse(pred_phi, target_phi, valid_mask),
        "rmse_energy": rmse(pred_e, target_e, valid_mask),
        "corr_phi": corr(pred_phi, target_phi, valid_mask),
        "corr_energy": corr(pred_e, target_e, valid_mask),
    }
    log(f"Metrics: {metrics}")

    run_date = datetime.now().strftime("%Y_%m_%d")
    run_dir = (args.output_dir / f"geometry_ml_inference_{run_date}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    fig_path = run_dir / "geometry_ml_prediction_comparison.png"
    save_figure(fig_path, valid_mask, pred_phi, pred_e, target_phi, target_e, base_phi, base_e)

    coords_xy = {"x": mapping_ds.coords["x"], "y": mapping_ds.coords["y"]}
    out_ds = xr.Dataset(
        {
            "valid_mask": xr.DataArray(valid_mask.astype(np.int8), dims=("x", "y"), coords=coords_xy),
            "predicted_phi_tilt_bins": xr.DataArray(pred_phi, dims=("x", "y"), coords=coords_xy),
            "predicted_e_tilt_bins": xr.DataArray(pred_e, dims=("x", "y"), coords=coords_xy),
            "target_phi_tilt_bins": xr.DataArray(target_phi, dims=("x", "y"), coords=coords_xy),
            "target_e_tilt_bins": xr.DataArray(target_e, dims=("x", "y"), coords=coords_xy),
            "base_phi_tilt_bins": xr.DataArray(base_phi, dims=("x", "y"), coords=coords_xy),
            "base_e_tilt_bins": xr.DataArray(base_e, dims=("x", "y"), coords=coords_xy),
            "phi_residual": xr.DataArray(pred_phi - target_phi, dims=("x", "y"), coords=coords_xy),
            "e_residual": xr.DataArray(pred_e - target_e, dims=("x", "y"), coords=coords_xy),
        }
    )
    for k, v in metrics.items():
        out_ds.attrs[k] = float(v)
    nc_path = run_dir / "geometry_ml_predictions.nc"
    out_ds.to_netcdf(nc_path, engine="h5netcdf")

    with open(run_dir / "geometry_ml_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log(f"Saved figure to {fig_path}")
    log(f"Saved predictions to {nc_path}")


if __name__ == "__main__":
    main()
