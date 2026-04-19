# phase_state_clustering.py
import os, json, argparse
import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F


def open_nc_dataset(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    engines_to_try = ["h5netcdf", "scipy"]
    last_err = None
    for eng in engines_to_try:
        try:
            return xr.open_dataset(file_path, engine=eng)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not open {file_path}. Last error: {last_err}")


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


def require_dims(da, required=("x", "y", "eV", "phi")):
    missing = [d for d in required if d not in da.dims]
    if missing:
        raise ValueError(f"Missing dims {missing}. Found: {da.dims}")


def get_energy_indices(da, fermi_level=0.0, halfwidth=0.05):
    e = np.asarray(da.coords["eV"].values, dtype=np.float32)
    idx = np.where(np.abs(e - fermi_level) <= halfwidth)[0]
    if idx.size == 0:
        raise ValueError("Energy crop is empty. Check fermi_level / halfwidth.")
    return idx


def normalize_spectrum_for_encoder(spec2d, eps=1e-8):
    s = np.asarray(spec2d, dtype=np.float32)
    s = np.log1p(np.clip(s, a_min=0.0, a_max=None))
    s = s - s.min()
    total = s.sum()
    if total <= eps:
        return s
    return s / (total + eps)


class SpectrumAE(torch.nn.Module):
    def __init__(self, e_len: int, p_len: int, z_dim: int = 16):
        super().__init__()
        self.e_len = e_len
        self.p_len = p_len
        self.enc = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.fc_mu = torch.nn.Linear(64 * 4 * 4, z_dim)

        self.fc_dec = torch.nn.Linear(z_dim, 64 * 4 * 4)
        self.dec_conv = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, kernel_size=1),
        )

    def encode(self, x):
        h = self.enc(x)
        h = self.pool(h)
        h = h.reshape(h.shape[0], -1)
        z = self.fc_mu(h)
        return z

    def forward(self, x):
        z = self.encode(x[:, None, ...])
        # decoder omitted in this script; we only need encoder
        raise NotImplementedError


def torch_kmeans(X_np, k=6, n_iter=100, n_init=20, seed=42, device="cpu"):
    torch.manual_seed(seed)
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    N, D = X.shape
    best_inertia, best_labels, best_centroids = None, None, None

    for _ in range(n_init):
        perm = torch.randperm(N, device=device)
        centroids = X[perm[:k]].clone()
        for __ in range(n_iter):
            dists = torch.cdist(X, centroids)  # (N,k)
            labels = torch.argmin(dists, dim=1)
            new_centroids = []
            for c in range(k):
                mask = labels == c
                if mask.sum() == 0:
                    new_centroids.append(X[torch.randint(0, N, (1,), device=device)].squeeze(0))
                else:
                    new_centroids.append(X[mask].mean(dim=0))
            new_centroids = torch.stack(new_centroids, dim=0)
            if torch.allclose(new_centroids, centroids, atol=1e-5):
                centroids = new_centroids
                break
            centroids = new_centroids

        dists = torch.cdist(X, centroids)
        final_labels = torch.argmin(dists, dim=1)
        inertia = torch.sum((X - centroids[final_labels]) ** 2).item()
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = final_labels.detach().cpu().numpy()
            best_centroids = centroids.detach().cpu().numpy()

    return best_labels, best_centroids, best_inertia


def compute_latents(nc_files, valid_mask_2d, encoder_ckpt,
                     batch_size=256, device="cpu"):
    ckpt = torch.load(encoder_ckpt, map_location=device)
    e_len, p_len, z_dim = ckpt["e_len"], ckpt["p_len"], ckpt["z_dim"]
    rep_halfwidth, phi_downsample, fermi_level = ckpt["rep_halfwidth"], ckpt["phi_downsample"], ckpt["fermi_level"]

    model = SpectrumAE(e_len=e_len, p_len=p_len, z_dim=z_dim).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    valid_flat = np.where(valid_mask_2d.reshape(-1))[0]
    xN, yN = valid_mask_2d.shape

    # latent per state: (Nvalid, z_dim)
    latents = []
    for fp in nc_files:
        ds = open_nc_dataset(fp)
        da = get_main_dataarray(ds)
        require_dims(da)

        e_idx = get_energy_indices(da, fermi_level=fermi_level, halfwidth=rep_halfwidth)
        da_c = da.isel(eV=e_idx)
        if phi_downsample > 1:
            da_c = da_c.isel(phi=slice(None, None, phi_downsample))

        data = np.asarray(da_c.values, dtype=np.float32)  # (x,y,E,P)
        flat = data.reshape(xN * yN, data.shape[2], data.shape[3])
        flat = flat[valid_flat]  # (Nvalid,E,P)

        # batch encode
        z_list = []
        with torch.no_grad():
            for i in range(0, flat.shape[0], batch_size):
                xb = flat[i:i+batch_size]  # (B,E,P) raw
                # apply same normalization as training
                xb_norm = np.stack([normalize_spectrum_for_encoder(s) for s in xb], axis=0).astype(np.float32)
                xb_t = torch.from_numpy(xb_norm).to(device)
                z = model.encode(xb_t[:, None, ...])  # (B,z_dim)
                z_list.append(z.detach().cpu().numpy())
        latents.append(np.concatenate(z_list, axis=0))
    return np.concatenate(latents, axis=0), latents


def ef_fraction_map(nc_file, valid_mask_2d, fermi_level=0.0, ef_halfwidth=0.05, energy_window=None):
    ds = open_nc_dataset(nc_file)
    da = get_main_dataarray(ds)
    require_dims(da)
    e_idx = get_energy_indices(da, fermi_level=fermi_level, halfwidth=ef_halfwidth)
    # EF window intensity = sum over e_idx and phi
    ef = da.isel(eV=e_idx).sum(dim=("eV", "phi")).values.astype(np.float32)  # (x,y)
    total = da.sum(dim=("eV", "phi")).values.astype(np.float32)
    frac = ef / (total + 1e-8)
    frac = np.asarray(frac, dtype=np.float32)
    frac[~valid_mask_2d] = np.nan
    return frac


def build_semantic_mapping(cluster_ids, ef_fraction_t0, delta_ef_tend):
    # cluster_ids: array of all cluster ids (0..K-1)
    # ef_fraction_t0: (K,) mean EF fraction per cluster at t0
    # delta_ef_tend: (K,) mean EF fraction change from t0 to t_last

    # Rank by metallicity proxy: EF fraction at t0
    order = list(cluster_ids)
    order.sort(key=lambda c: ef_fraction_t0[c])  # low -> high

    mapping = {}
    if len(order) >= 1:
        mapping[order[0]] = "insulating"
    if len(order) >= 2:
        mapping[order[1]] = "intermediate"
    if len(order) >= 3:
        mapping[order[-1]] = "metallic"

    # metastable metallic types: pick next-high clusters excluding already assigned
    assigned = set(mapping.keys())
    remaining = [c for c in cluster_ids if c not in assigned]
    remaining.sort(key=lambda c: ef_fraction_t0[c], reverse=True)

    if len(remaining) >= 1:
        mapping[remaining[0]] = "metastable metallic type A"
    if len(remaining) >= 2:
        mapping[remaining[1]] = "metastable metallic type B"

    # erased/reverted: cluster with strongest EF loss
    # (if delta_ef_tend is available)
    if delta_ef_tend is not None:
        # most negative delta
        erase_c = min(cluster_ids, key=lambda c: delta_ef_tend[c])
        # only assign if it's "significantly" negative
        vals = np.array([delta_ef_tend[c] for c in cluster_ids], dtype=np.float32)
        thresh = np.quantile(vals, 0.20)
        if delta_ef_tend[erase_c] < thresh:
            mapping[erase_c] = "erased / reverted"

    # any remaining -> intermediate
    for c in cluster_ids:
        if c not in mapping:
            mapping[c] = "intermediate"

    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc_files", nargs="+", required=True)
    ap.add_argument("--encoder_checkpoint", required=True)
    ap.add_argument("--output_dir", default="state_cluster_out")

    ap.add_argument("--n_clusters", type=int, default=6)
    ap.add_argument("--kmeans_n_init", type=int, default=20)
    ap.add_argument("--kmeans_n_iter", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--valid_mask_npy", default=None,
                    help="Optional npy boolean mask with shape (x,y). If missing, uses saved from encoder step.")
    ap.add_argument("--valid_mask_from_encoder_dir", action="store_true")

    # physics relabeling
    ap.add_argument("--fermi_level", type=float, default=0.0)
    ap.add_argument("--ef_window", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # load valid mask
    if args.valid_mask_npy is not None:
        valid_mask = np.load(args.valid_mask_npy).astype(bool)
    else:
        if args.valid_mask_from_encoder_dir:
            enc_dir = os.path.dirname(args.encoder_checkpoint)
            mask_path = os.path.join(enc_dir, "valid_cross_mask.npy")
        else:
            mask_path = os.path.join(os.path.dirname(args.encoder_checkpoint), "valid_cross_mask.npy")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"valid_mask not provided. Tried: {mask_path}")
        valid_mask = np.load(mask_path).astype(bool)

    print("valid pixels:", int(valid_mask.sum()))

    # compute latents for each state and global kmeans centroids
    embed_all, embeds_per_state = compute_latents(
        n_c_files if False else args.nc_files,
        valid_mask_2d=valid_mask,
        encoder_ckpt=args.encoder_checkpoint,
        batch_size=256,
        device=device,
    )

    # kmeans across time-shared embedding set
    Nvalid = int(valid_mask.sum())
    z_dim = embed_all.shape[1]
    print("latent dims:", z_dim, " embed_all shape:", embed_all.shape)

    shared_labels, centroids, inertia = torch_kmeans(
        embed_all,
        k=args.n_clusters,
        n_iter=args.kmeans_n_iter,
        n_init=args.kmeans_n_init,
        seed=args.seed,
        device=device,
    )
    print("shared kmeans inertia:", inertia)
    np.save(os.path.join(args.output_dir, "kmeans_centroids.npy"), centroids)

    # build per-state cluster maps
    xN, yN = valid_mask.shape
    valid_flat = np.where(valid_mask.reshape(-1))[0]
    state_cluster_maps = []
    start = 0
    for si in range(len(args.nc_files)):
        end = start + Nvalid
        labels_state = shared_labels[start:end]
        cmap = np.full((xN * yN,), fill_value=-1, dtype=np.int32)
        cmap[valid_flat] = labels_state
        cmap = cmap.reshape(xN, yN)
        state_cluster_maps.append(cmap)
        start = end

    # physics-based semantic relabeling using EF fraction at t0 and t_last
    ef0 = ef_fraction_map(
        args.nc_files[0], valid_mask,
        fermi_level=args.fermi_level, ef_halfwidth=args.ef_window
    )
    efT = ef_fraction_map(
        args.nc_files[-1], valid_mask,
        fermi_level=args.fermi_level, ef_halfwidth=args.ef_window
    )
    delta = efT - ef0  # (x,y) nan outside

    cluster_ids = list(range(args.n_clusters))
    ef_frac_t0 = {}
    delta_ef = {}

    for c in cluster_ids:
        mask_c = (state_cluster_maps[0] == c) & valid_mask
        ef_frac_t0[c] = float(np.nanmean(ef0[mask_c])) if mask_c.any() else 0.0
        delta_ef[c] = float(np.nanmean(delta[mask_c])) if mask_c.any() else 0.0

    mapping = build_semantic_mapping(cluster_ids, ef_frac_t0, delta_ef)
    print("cluster -> semantic:")
    for c in cluster_ids:
        print(f"  {c}: {mapping[c]}")

    # encode semantic codes
    state_names = [
        "insulating",
        "metallic",
        "intermediate",
        "metastable metallic type A",
        "metastable metallic type B",
        "erased / reverted",
    ]
    code_map = {name: i for i, name in enumerate(state_names)}

    semantic_maps = []
    for si in range(len(args.nc_files)):
        sm = np.full_like(state_cluster_maps[si], fill_value=-1, dtype=np.int32)
        for c in cluster_ids:
            sm[state_cluster_maps[si] == c] = code_map[mapping[c]]
        sm[~valid_mask] = -1
        semantic_maps.append(sm)

    # save outputs
    for si in range(len(args.nc_files)):
        np.save(os.path.join(args.output_dir, f"cluster_map_state_{si}.npy"), state_cluster_maps[si])
        np.save(os.path.join(args.output_dir, f"semantic_code_map_state_{si}.npy"), semantic_maps[si])

    with open(os.path.join(args.output_dir, "cluster_to_semantic.json"), "w") as f:
        json.dump({str(k): mapping[k] for k in mapping}, f, indent=2)

    meta = {
        "n_clusters": args.n_clusters,
        "ef_window": args.ef_window,
        "fermi_level": args.fermi_level,
        "state_names": state_names,
        "cluster_ids": cluster_ids,
        "kmeans_inertia": inertia,
    }
    with open(os.path.join(args.output_dir, "cluster_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done. Saved to:", args.output_dir)


if __name__ == "__main__":
    main()