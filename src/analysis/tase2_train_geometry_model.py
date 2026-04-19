#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import xarray as xr


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Train a PyTorch geometry-regression model using a geometry-mapping bundle "
            "and a simulated nanoARPES cube."
        )
    )
    parser.add_argument("--mapping-path", type=Path, required=True,
                        help="Path to geometry_mapping_* directory or geometry_prediction_bundle_*.nc file.")
    parser.add_argument("--simulation-path", type=Path, required=True,
                        help="Path to geometry_simulation_* directory or simulated_*.nc file.")
    parser.add_argument("--output-dir", type=Path, default=(script_dir / "../../outputs/geometry_ml").resolve())
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hidden-channels", type=int, default=32)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_nc(path: Path, prefix: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(path)
    matches = sorted(path.glob(f"{prefix}*.nc"))
    if not matches:
        raise FileNotFoundError(f"No .nc files matching {prefix}*.nc found in {path}")
    return matches[-1]


def open_dataset(path: Path) -> xr.Dataset:
    engines = ["h5netcdf", "scipy", None, "netcdf4"]
    last_error = None
    for eng in engines:
        try:
            label = "default" if eng is None else eng
            log(f"Opening {path.name} with engine={label}")
            if eng is None:
                return xr.open_dataset(path)
            return xr.open_dataset(path, engine=eng)
        except Exception as exc:
            last_error = exc
            log(f"engine={label} failed: {exc}")
    raise RuntimeError(f"Could not open {path}: {last_error}")


def get_main_dataarray(ds: xr.Dataset) -> xr.DataArray:
    candidates = []
    for name, var in ds.data_vars.items():
        try:
            if np.issubdtype(var.dtype, np.number):
                candidates.append((name, int(np.prod(var.shape))))
        except Exception:
            pass
    if not candidates:
        raise ValueError("No numeric data variables found.")
    candidates.sort(key=lambda x: x[1], reverse=True)
    return ds[candidates[0][0]]


def require_dims(da: xr.DataArray, required: Sequence[str] = ("x", "y", "eV", "phi")) -> None:
    missing = [d for d in required if d not in da.dims]
    if missing:
        raise ValueError(f"Missing required dimensions {missing}; found {da.dims}")


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return a / (b + eps)


def normalize_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    total = float(np.sum(v))
    if total <= eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / total).astype(np.float32)


def extract_cube_feature_maps(da: xr.DataArray, fermi_level: float = 0.0, ef_window: float = 0.05) -> Dict[str, np.ndarray]:
    require_dims(da)
    data = np.asarray(da.values, dtype=np.float32)
    xN, yN, eN, pN = data.shape
    e_axis = np.asarray(da.coords["eV"].values, dtype=np.float32)
    p_axis = np.asarray(da.coords["phi"].values, dtype=np.float32)
    ef_mask = np.abs(e_axis - np.float32(fermi_level)) <= np.float32(ef_window)

    X = data.reshape(xN * yN, eN, pN)
    total_intensity = X.sum(axis=(1, 2))
    ef_cube = X[:, ef_mask, :]
    ef_intensity = ef_cube.sum(axis=(1, 2))
    ef_fraction = safe_divide(ef_intensity, total_intensity)

    e_profile = X.sum(axis=2)
    p_profile = X.sum(axis=1)
    e_profile_norm = np.stack([normalize_vector(v) for v in e_profile], axis=0)
    p_profile_norm = np.stack([normalize_vector(v) for v in p_profile], axis=0)
    e_centroid = (e_profile_norm * e_axis[None, :]).sum(axis=1)
    phi_centroid = (p_profile_norm * p_axis[None, :]).sum(axis=1)
    phi_var = (p_profile_norm * (p_axis[None, :] - phi_centroid[:, None]) ** 2).sum(axis=1)

    X_flat = X.reshape(xN * yN, -1)
    X_norm = np.stack([normalize_vector(v) for v in X_flat], axis=0)
    spectral_entropy = -np.sum(X_norm * np.log(X_norm + 1e-12), axis=1)
    spectral_sharpness = safe_divide(X_flat.max(axis=1), X_flat.mean(axis=1))

    return {
        "sim_total_intensity": total_intensity.reshape(xN, yN).astype(np.float32),
        "sim_ef_fraction": ef_fraction.reshape(xN, yN).astype(np.float32),
        "sim_e_centroid": e_centroid.reshape(xN, yN).astype(np.float32),
        "sim_phi_centroid": phi_centroid.reshape(xN, yN).astype(np.float32),
        "sim_phi_var": phi_var.reshape(xN, yN).astype(np.float32),
        "sim_spectral_entropy": spectral_entropy.reshape(xN, yN).astype(np.float32),
        "sim_spectral_sharpness": spectral_sharpness.reshape(xN, yN).astype(np.float32),
    }


def standardize_maps(feature_maps: Dict[str, np.ndarray], stats: Dict[str, Dict[str, float]], valid_mask: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    for name, arr in feature_maps.items():
        mean = stats[name]["mean"]
        std = stats[name]["std"]
        z = (arr - mean) / std
        z = z.astype(np.float32)
        z[~valid_mask] = 0.0
        out[name] = z
    return out


def compute_feature_stats(feature_maps: Dict[str, np.ndarray], valid_mask: np.ndarray) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for name, arr in feature_maps.items():
        vals = arr[valid_mask].astype(np.float64)
        mean = float(vals.mean())
        std = float(vals.std())
        if std < 1e-8:
            std = 1.0
        stats[name] = {"mean": mean, "std": std}
    return stats


class GeometryPatchDataset(Dataset):
    def __init__(
        self,
        input_maps: np.ndarray,
        target_maps: np.ndarray,
        coords: np.ndarray,
        patch_radius: int,
        target_mean: np.ndarray,
        target_std: np.ndarray,
    ) -> None:
        self.coords = coords.astype(np.int64)
        self.r = int(patch_radius)
        self.target_mean = target_mean.astype(np.float32)
        self.target_std = target_std.astype(np.float32)
        self.input_padded = np.pad(
            input_maps,
            ((0, 0), (self.r, self.r), (self.r, self.r)),
            mode="reflect",
        ).astype(np.float32)
        self.target_maps = target_maps.astype(np.float32)

    def __len__(self) -> int:
        return self.coords.shape[0]

    def __getitem__(self, idx: int):
        x, y = self.coords[idx]
        x0 = x
        y0 = y
        patch = self.input_padded[:, x0:x0 + 2 * self.r + 1, y0:y0 + 2 * self.r + 1]
        target = self.target_maps[:, x, y]
        target = (target - self.target_mean) / self.target_std
        return torch.from_numpy(patch), torch.from_numpy(target)


class GeometryPatchCNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


@dataclass
class SplitData:
    train_coords: np.ndarray
    val_coords: np.ndarray
    feature_stats: Dict[str, Dict[str, float]]
    target_mean: np.ndarray
    target_std: np.ndarray
    feature_names: List[str]


def build_inputs_and_targets(mapping_ds: xr.Dataset, sim_da: xr.DataArray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
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
    target_maps = np.stack(
        [
            mapping_ds["target_phi_tilt_bins"].values.astype(np.float32),
            mapping_ds["target_e_tilt_bins"].values.astype(np.float32),
        ],
        axis=0,
    )
    feature_names = list(source_maps.keys())
    input_stack = np.stack([source_maps[name] for name in feature_names], axis=0).astype(np.float32)
    input_stack[:, ~valid_mask] = 0.0
    return input_stack, target_maps, valid_mask, feature_names


def split_coords(valid_mask: np.ndarray, train_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(valid_mask)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(coords.shape[0])
    coords = coords[perm]
    n_train = max(1, int(round(train_frac * coords.shape[0])))
    n_train = min(n_train, coords.shape[0] - 1)
    return coords[:n_train], coords[n_train:]


def prepare_data(mapping_ds: xr.Dataset, sim_da: xr.DataArray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SplitData]:
    input_maps_raw, target_maps, valid_mask, feature_names = build_inputs_and_targets(mapping_ds, sim_da)
    train_coords, val_coords = split_coords(valid_mask, args.train_frac, args.seed)

    feature_maps_raw = {name: input_maps_raw[i] for i, name in enumerate(feature_names)}
    train_mask = np.zeros_like(valid_mask, dtype=bool)
    train_mask[tuple(train_coords.T)] = True
    feature_stats = compute_feature_stats(feature_maps_raw, train_mask)
    feature_maps = standardize_maps(feature_maps_raw, feature_stats, valid_mask)
    input_maps = np.stack([feature_maps[name] for name in feature_names], axis=0)

    train_targets = np.stack([target_maps[:, x, y] for x, y in train_coords], axis=0)
    target_mean = train_targets.mean(axis=0).astype(np.float32)
    target_std = train_targets.std(axis=0).astype(np.float32)
    target_std[target_std < 1e-8] = 1.0

    split = SplitData(
        train_coords=train_coords,
        val_coords=val_coords,
        feature_stats=feature_stats,
        target_mean=target_mean,
        target_std=target_std,
        feature_names=feature_names,
    )
    return input_maps, target_maps, valid_mask, split


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> Dict[str, float]:
    model.eval()
    all_pred = []
    all_true = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            all_pred.append(pred)
            all_true.append(yb.numpy())
    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)
    pred = pred * target_std[None, :] + target_mean[None, :]
    true = true * target_std[None, :] + target_mean[None, :]
    err = pred - true
    rmse_phi = float(np.sqrt(np.mean(err[:, 0] ** 2)))
    rmse_e = float(np.sqrt(np.mean(err[:, 1] ** 2)))

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
        if denom <= 1e-12:
            return float("nan")
        return float(np.sum(a * b) / denom)

    return {
        "rmse_phi": rmse_phi,
        "rmse_energy": rmse_e,
        "corr_phi": corr(pred[:, 0], true[:, 0]),
        "corr_energy": corr(pred[:, 1], true[:, 1]),
    }


def train() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    log(f"Using device: {device}")

    mapping_nc = resolve_nc(args.mapping_path, "geometry_prediction_bundle")
    sim_nc = resolve_nc(args.simulation_path, "simulated_")
    mapping_ds = open_dataset(mapping_nc)
    sim_ds = open_dataset(sim_nc)
    sim_da = get_main_dataarray(sim_ds)

    log("Preparing training tensors")
    input_maps, target_maps, valid_mask, split = prepare_data(mapping_ds, sim_da, args)

    train_ds = GeometryPatchDataset(
        input_maps, target_maps, split.train_coords, args.patch_radius, split.target_mean, split.target_std
    )
    val_ds = GeometryPatchDataset(
        input_maps, target_maps, split.val_coords, args.patch_radius, split.target_mean, split.target_std
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = GeometryPatchCNN(in_channels=input_maps.shape[0], hidden_channels=args.hidden_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    run_date = datetime.now().strftime("%Y_%m_%d")
    run_dir = (args.output_dir / f"geometry_ml_train_{run_date}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            total_count += xb.size(0)

        train_loss = total_loss / max(total_count, 1)
        metrics = evaluate(model, val_loader, device, split.target_mean, split.target_std)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        history.append(metrics)
        log(
            f"epoch {epoch:03d} train_loss={train_loss:.6f} "
            f"val_rmse_phi={metrics['rmse_phi']:.4f} val_rmse_energy={metrics['rmse_energy']:.4f}"
        )
        score = metrics["rmse_phi"] + metrics["rmse_energy"]
        if best is None or score < best:
            best = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    final_val = evaluate(model, val_loader, device, split.target_mean, split.target_std)
    final_train = evaluate(model, train_loader, device, split.target_mean, split.target_std)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "in_channels": int(input_maps.shape[0]),
        "hidden_channels": int(args.hidden_channels),
        "patch_radius": int(args.patch_radius),
        "feature_names": split.feature_names,
        "feature_stats": split.feature_stats,
        "target_mean": split.target_mean.tolist(),
        "target_std": split.target_std.tolist(),
        "mapping_nc": str(mapping_nc),
        "simulation_nc": str(sim_nc),
        "seed": int(args.seed),
    }
    ckpt_path = run_dir / "geometry_model_checkpoint.pt"
    torch.save(checkpoint, ckpt_path)

    summary = {
        "mapping_nc": str(mapping_nc),
        "simulation_nc": str(sim_nc),
        "device": str(device),
        "valid_pixels": int(valid_mask.sum()),
        "train_pixels": int(split.train_coords.shape[0]),
        "val_pixels": int(split.val_coords.shape[0]),
        "feature_names": split.feature_names,
        "patch_radius": int(args.patch_radius),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "hidden_channels": int(args.hidden_channels),
        "final_train_metrics": final_train,
        "final_val_metrics": final_val,
        "history": history,
    }
    with open(run_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log(f"Saved checkpoint to {ckpt_path}")
    log(f"Validation metrics: {final_val}")


if __name__ == "__main__":
    train()
