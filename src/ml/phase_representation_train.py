# phase_representation_train.py
import os, json, argparse
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F


def open_nc_dataset(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    # Avoid netCDF4 engine (can crash in some numpy setups).
    engines_to_try = ["h5netcdf", "scipy"]
    last_err = None
    for eng in engines_to_try:
        try:
            ds = xr.open_dataset(file_path, engine=eng)
            return ds
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not open {file_path}. Last error: {last_err}")


def require_dims(da, required=("x", "y", "eV", "phi")):
    missing = [d for d in required if d not in da.dims]
    if missing:
        raise ValueError(f"Missing dims {missing}. Found: {da.dims}")


def get_main_dataarray(ds):
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


def get_energy_indices(da, fermi_level=0.0, halfwidth=0.20):
    e = np.asarray(da.coords["eV"].values, dtype=np.float32)
    mask = np.abs(e - fermi_level) <= halfwidth
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError("Energy crop is empty. Check fermi_level / halfwidth.")
    return idx


def dilate_mask_maxpool(mask2d: np.ndarray, n_iter: int = 1, kernel_radius: int = 1):
    # mask2d: bool (H,W)
    # Dilate using max_pool2d to stay vectorized.
    x = torch.from_numpy(mask2d.astype(np.float32))[None, None, ...]  # 1x1xHxW
    for _ in range(max(0, int(n_iter))):
        k = 2 * kernel_radius + 1
        x = F.max_pool2d(x, kernel_size=k, stride=1, padding=kernel_radius)
        x = (x > 0.5).float()
    return (x[0, 0].cpu().numpy() > 0.5)


def build_cross_mask_from_total_maps(total_maps, threshold_quantile=0.45,
                                      row_fraction=0.18, col_fraction=0.18,
                                      background_quantile=0.10, pad=1):
    # total_maps: list of (x,y)
    arrs = []
    for m in total_maps:
        m = np.asarray(m, dtype=np.float32)
        lo, hi = np.nanmin(m), np.nanmax(m)
        if hi > lo:
            arrs.append((m - lo) / (hi - lo))
        else:
            arrs.append(np.zeros_like(m, dtype=np.float32))
    avg = np.mean(arrs, axis=0)
    th = np.quantile(avg.reshape(-1), threshold_quantile)
    active = avg >= th
    row_occ = active.mean(axis=1)
    col_occ = active.mean(axis=0)
    strong_rows = row_occ >= row_fraction
    strong_cols = col_occ >= col_fraction
    cross_mask = strong_rows[:, None] | strong_cols[None, :]
    bg_th = np.quantile(avg.reshape(-1), background_quantile)
    cross_mask = cross_mask & (avg >= bg_th)
    if pad and pad > 0:
        cross_mask = dilate_mask_maxpool(cross_mask, n_iter=1, kernel_radius=pad)
    return cross_mask


def normalize_spectrum(spec2d, eps=1e-8):
    # spec2d: (E,P)
    s = np.asarray(spec2d, dtype=np.float32)
    s = np.log1p(np.clip(s, a_min=0.0, a_max=None))
    s = s - s.min()
    total = s.sum()
    if total <= eps:
        return s
    return s / (total + eps)


class SpectrumAE(nn.Module):
    """
    Autoencoder operating on input shaped (E, Phi) after cropping/downsampling.
    """
    def __init__(self, e_len: int, p_len: int, z_dim: int = 16):
        super().__init__()
        self.e_len = e_len
        self.p_len = p_len

        # Encoder: 2D conv -> global pooling -> latent
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # ~E/2 x P/2
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # ~E/4 x P/4
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # ~E/8 x P/8
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_mu = nn.Linear(64 * 4 * 4, z_dim)

        # Decoder: latent -> conv feature -> upsample -> output (E,P)
        self.fc_dec = nn.Linear(z_dim, 64 * 4 * 4)
        self.dec_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def encode(self, x):
        # x: (B,1,E,P)
        h = self.enc(x)
        h = self.pool(h)
        h = h.reshape(h.shape[0], -1)
        z = self.fc_mu(h)
        return z

    def decode(self, z):
        h = self.fc_dec(z).reshape(z.shape[0], 64, 4, 4)
        h = F.interpolate(h, size=(self.e_len, self.p_len), mode="bilinear", align_corners=False)
        xhat = self.dec_conv(h)
        return xhat.squeeze(1)  # (B,E,P)

    def forward(self, x):
        # x: (B, E, P)
        z = self.encode(x[:, None, ...])
        xhat = self.decode(z)
        return xhat, z


class PixelSpectraDataset(torch.utils.data.Dataset):
    def __init__(self, nc_files, valid_mask_2d, fermi_level, rep_halfwidth,
                 phi_downsample=1, cache_in_memory=True):
        self.nc_files = list(nc_files)
        self.valid_mask_2d = valid_mask_2d.astype(bool)
        self.fermi_level = float(fermi_level)
        self.rep_halfwidth = float(rep_halfwidth)
        self.phi_downsample = int(phi_downsample)

        self.valid_flat = np.where(self.valid_mask_2d.reshape(-1))[0]
        if self.valid_flat.size == 0:
            raise ValueError("valid_mask has no True pixels.")

        # Load all states (cropped + downsampled) into memory by default
        self.cache_in_memory = bool(cache_in_memory)
        self.spectra_by_state = [None] * len(self.nc_files)

        self.e_len = None
        self.p_len = None
        self._load_and_cache()

    def _load_and_cache(self):
        for si, fp in enumerate(self.nc_files):
            ds = open_nc_dataset(fp)
            da = get_main_dataarray(ds)
            require_dims(da)
            # crop energy window
            e_idx = get_energy_indices(da, fermi_level=self.fermi_level, halfwidth=self.rep_halfwidth)
            da_c = da.isel(eV=e_idx)
            # downsample phi
            if self.phi_downsample > 1:
                da_c = da_c.isel(phi=slice(None, None, self.phi_downsample))

            data = np.asarray(da_c.values, dtype=np.float32)  # (x,y,E,P)
            xN, yN, E, P = data.shape
            if self.e_len is None:
                self.e_len, self.p_len = E, P
            else:
                if E != self.e_len or P != self.p_len:
                    raise ValueError(f"Shape mismatch across files: got (E={E},P={P})")

            flat = data.reshape(xN * yN, E, P)
            flat = flat[self.valid_flat]  # (Nvalid,E,P)
            if self.cache_in_memory:
                self.spectra_by_state[si] = flat
            else:
                self.spectra_by_state[si] = None
                # We still loaded once to infer shape; we can cache it if desired.
                self.spectra_by_state[si] = flat

    def __len__(self):
        # treat each pixel in each state as a training sample
        return len(self.nc_files) * self.valid_flat.size

    def __getitem__(self, idx):
        # idx -> (state, pixel)
        nvalid = self.valid_flat.size
        si = idx // nvalid
        pi = idx % nvalid

        spec = self.spectra_by_state[si][pi]  # (E,P)
        spec = normalize_spectrum(spec)
        return torch.from_numpy(spec)  # (E,P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc_files", nargs="+", required=True)
    ap.add_argument("--output_dir", default="rep_model_out")
    ap.add_argument("--fermi_level", type=float, default=0.0)

    ap.add_argument("--rep_halfwidth", type=float, default=0.20)  # energy crop for rep AE
    ap.add_argument("--phi_downsample", type=int, default=2)
    ap.add_argument("--z_dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)

    # cross-mask controls (optional: computed from total intensity across states)
    ap.add_argument("--valid_mask_npy", default=None,
                    help="Optional path to boolean mask saved as npy with shape (x,y).")
    ap.add_argument("--threshold_quantile", type=float, default=0.45)
    ap.add_argument("--row_fraction", type=float, default=0.18)
    ap.add_argument("--col_fraction", type=float, default=0.18)
    ap.add_argument("--background_quantile", type=float, default=0.10)
    ap.add_argument("--cross_pad", type=int, default=1)

    ap.add_argument("--cache_in_memory", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # build or load valid_mask
    if args.valid_mask_npy is not None:
        valid_mask = np.load(args.valid_mask_npy).astype(bool)
    else:
        total_maps = []
        # compute total intensity maps using full eV+phi from each file (fast-ish)
        for fp in args.nc_files:
            ds = open_nc_dataset(fp)
            da = get_main_dataarray(ds)
            require_dims(da)
            tot = da.sum(dim=("eV", "phi")).values.astype(np.float32)  # (x,y)
            total_maps.append(tot)
        valid_mask = build_cross_mask_from_total_maps(
            total_maps,
            threshold_quantile=args.threshold_quantile,
            row_fraction=args.row_fraction,
            col_fraction=args.col_fraction,
            background_quantile=args.background_quantile,
            pad=args.cross_pad,
        )
    print("valid pixels:", int(valid_mask.sum()))

    ds_train = PixelSpectraDataset(
        nc_files=args.nc_files,
        valid_mask_2d=valid_mask,
        fermi_level=args.fermi_level,
        rep_halfwidth=args.rep_halfwidth,
        phi_downsample=args.phi_downsample,
        cache_in_memory=args.cache_in_memory or True,  # default True for simplicity/speed
    )
    e_len, p_len = ds_train.e_len, ds_train.p_len
    print("AE input size (E,P) =", (e_len, p_len), " z_dim=", args.z_dim)

    model = SpectrumAE(e_len=e_len, p_len=p_len, z_dim=args.z_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    loader = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device == "cuda")
    )

    model.train()
    for ep in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)  # (B,E,P)
            xhat, _ = model(batch)
            loss = F.mse_loss(xhat, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.size(0)

        avg = total_loss / len(ds_train)
        print(f"epoch {ep+1}/{args.epochs}  mse={avg:.6e}")

        # save checkpoint each epoch (optional)
        ckpt = {
            "model_state": model.state_dict(),
            "e_len": e_len,
            "p_len": p_len,
            "z_dim": args.z_dim,
            "rep_halfwidth": args.rep_halfwidth,
            "phi_downsample": args.phi_downsample,
            "fermi_level": args.fermi_level,
        }
        torch.save(ckpt, os.path.join(args.output_dir, "encoder_checkpoint.pt"))

    # final save (encoder-only usable)
    enc_only = {
        "model_state": model.state_dict(),
        "e_len": e_len,
        "p_len": p_len,
        "z_dim": args.z_dim,
        "rep_halfwidth": args.rep_halfwidth,
        "phi_downsample": args.phi_downsample,
        "fermi_level": args.fermi_level,
    }
    torch.save(enc_only, os.path.join(args.output_dir, "encoder_checkpoint.pt"))

    # save mask for later steps
    np.save(os.path.join(args.output_dir, "valid_cross_mask.npy"), valid_mask.astype(np.uint8))

    meta = {
        "nc_files": args.nc_files,
        "e_len": e_len,
        "p_len": p_len,
        "z_dim": args.z_dim,
        "rep_halfwidth": args.rep_halfwidth,
        "phi_downsample": args.phi_downsample,
        "fermi_level": args.fermi_level,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done. Saved to:", args.output_dir)


if __name__ == "__main__":
    main()