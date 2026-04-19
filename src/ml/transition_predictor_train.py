# transition_predictor_train.py
import os, json, argparse
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F


def open_nc_dataset(file_path: str):
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


def get_energy_indices(da, fermi_level=0.0, halfwidth=0.20):
    e = np.asarray(da.coords["eV"].values, dtype=np.float32)
    idx = np.where(np.abs(e - fermi_level) <= halfwidth)[0]
    if idx.size == 0:
        raise ValueError("Energy crop is empty.")
    return idx


def normalize_spectrum_for_encoder(spec2d, eps=1e-8):
    s = np.asarray(spec2d, dtype=np.float32)
    s = np.log1p(np.clip(s, a_min=0.0, a_max=None))
    s = s - s.min()
    total = s.sum()
    if total <= eps:
        return s
    return s / (total + eps)


class SpectrumAE(nn.Module):
    def __init__(self, e_len: int, p_len: int, z_dim: int = 16):
        super().__init__()
        self.e_len = e_len
        self.p_len = p_len
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_mu = nn.Linear(64 * 4 * 4, z_dim)

    def encode(self, x):
        h = self.enc(x)
        h = self.pool(h)
        h = h.reshape(h.shape[0], -1)
        z = self.fc_mu(h)
        return z


def compute_latents_for_all_states(nc_files, valid_mask_2d, encoder_checkpoint,
                                   batch_size=512, device="cpu"):
    ckpt = torch.load(encoder_checkpoint, map_location=device)
    e_len, p_len, z_dim = ckpt["e_len"], ckpt["p_len"], ckpt["z_dim"]
    rep_halfwidth, phi_downsample, fermi_level = ckpt["rep_halfwidth"], ckpt["phi_downsample"], ckpt["fermi_level"]

    model = SpectrumAE(e_len=e_len, p_len=p_len, z_dim=z_dim).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    valid_flat = np.where(valid_mask_2d.reshape(-1))[0]
    xN, yN = valid_mask_2d.shape

    latents_per_state = []
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

        z_list = []
        with torch.no_grad():
            for i in range(0, flat.shape[0], batch_size):
                xb = flat[i:i+batch_size]
                xb_norm = np.stack([normalize_spectrum_for_encoder(s) for s in xb], axis=0).astype(np.float32)
                xb_t = torch.from_numpy(xb_norm).to(device)
                z = model.encode(xb_t[:, None, ...])
                z_list.append(z.detach().cpu().numpy())
        latents_per_state.append(np.concatenate(z_list, axis=0))
    return np.stack(latents_per_state, axis=0)  # (T,Nvalid,z_dim)


class TransitionNet(nn.Module):
    """
    Uses center latent + neighborhood latent mean to predict transition class.
    """
    def __init__(self, z_dim: int, hidden: int = 128, n_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * z_dim + 1, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, z_center, z_neigh_mean, neigh_count):
        # z_center: (B,z), z_neigh_mean: (B,z), neigh_count: (B,1)
        x = torch.cat([z_center, z_neigh_mean, neigh_count], dim=1)
        return self.mlp(x)


def build_neighbor_indices(valid_mask_2d):
    xN, yN = valid_mask_2d.shape
    valid_flat = np.where(valid_mask_2d.reshape(-1))[0]
    # map from flat index -> node idx 0..Nvalid-1
    node_id = -np.ones((xN * yN,), dtype=np.int32)
    node_id[valid_flat] = np.arange(valid_flat.size, dtype=np.int32)

    # for each node, store up to 4 neighbor node ids (-1 if absent)
    neigh = -np.ones((valid_flat.size, 4), dtype=np.int32)
    coords = [(0,1),(0,-1),(1,0),(-1,0)]  # right,left,down,up (in i,j)
    for idx, flat in enumerate(valid_flat):
        i = flat // yN
        j = flat % yN
        for k,(di,dj) in enumerate(coords):
            ii, jj = i + di, j + dj
            if 0 <= ii < xN and 0 <= jj < yN:
                nb_flat = ii * yN + jj
                nid = node_id[nb_flat]
                neigh[idx, k] = nid
    return neigh  # (Nvalid,4) with -1 entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc_files", nargs="+", required=True)
    ap.add_argument("--encoder_checkpoint", required=True)
    ap.add_argument("--state_cluster_dir", required=True,
                    help="Dir containing semantic_code_map_state_{i}.npy produced by step (2).")
    ap.add_argument("--valid_mask_npy", default=None)
    ap.add_argument("--exclude_boundary_code", type=int, default=None,
                    help="If you reserved a special boundary-like code, set it here to exclude.")
    ap.add_argument("--output_dir", default="transition_model_out")

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)

    ap.add_argument("--neigh_kinds", type=str, default="4", help="Currently only 4-neighbor supported.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # valid mask
    if args.valid_mask_npy is not None:
        valid_mask = np.load(args.valid_mask_npy).astype(bool)
    else:
        # fallback: alongside encoder checkpoint dir
        enc_dir = os.path.dirname(args.encoder_checkpoint)
        mask_path = os.path.join(enc_dir, "valid_cross_mask.npy")
        if not os.path.exists(mask_path):
            raise FileNotFoundError("Provide --valid_mask_npy or ensure valid_cross_mask.npy exists near encoder ckpt.")
        valid_mask = np.load(mask_path).astype(bool)

    T = len(args.nc_files)
    # load semantic maps for each state
    semantic_maps = []
    for si in range(T):
        path = os.path.join(args.state_cluster_dir, f"semantic_code_map_state_{si}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        semantic_maps.append(np.load(path))
    semantic_maps = np.stack(semantic_maps, axis=0)  # (T,x,y)

    # build per-node neighbor index list
    neigh_idx = build_neighbor_indices(valid_mask)  # (Nvalid,4)
    Nvalid = neigh_idx.shape[0]
    z_by_state = compute_latents_for_all_states(
        args.nc_files, valid_mask, args.encoder_checkpoint, batch_size=512, device=device
    )  # (T,Nvalid,z_dim)
    z_dim = z_by_state.shape[-1]
    print("z_dim:", z_dim, " Nvalid:", Nvalid)

    # Build transition labels: class = pair (pre_code, post_code)
    # Collect all transitions in available data; then define stable mapping.
    pre_codes = []
    post_codes = []
    for t in range(T - 1):
        pre = semantic_maps[t][valid_mask]
        post = semantic_maps[t + 1][valid_mask]
        pre_codes.append(pre.reshape(-1))
        post_codes.append(post.reshape(-1))
    pre_codes = np.concatenate(pre_codes, axis=0)
    post_codes = np.concatenate(post_codes, axis=0)

    valid_train = np.ones_like(pre_codes, dtype=bool)
    if args.exclude_boundary_code is not None:
        valid_train &= (pre_codes != args.exclude_boundary_code)
        valid_train &= (post_codes != args.exclude_boundary_code)

    pairs = list(set(zip(pre_codes[valid_train].tolist(), post_codes[valid_train].tolist())))
    pairs.sort()  # stable order
    pair_to_class = {pair: ci for ci, pair in enumerate(pairs)}
    n_classes = len(pairs)
    print("transition classes:", n_classes)

    # Create training tensors
    # For each t and each node, build: (z_center, z_neighbor_mean, neigh_count) -> class label
    # We'll construct arrays in memory for simplicity (small dataset).
    neigh_count_tensor = (neigh_idx >= 0).sum(axis=1).astype(np.float32)  # (Nvalid,)
    # neighbor mean latent: mean over existing neighbors
    # We'll precompute neighbor mean for each time state:
    z_neighbor_mean = []
    for t in range(T):
        zt = z_by_state[t].detach().cpu().numpy()  # (Nvalid,z)
        zm = np.zeros_like(zt)
        for n in range(Nvalid):
            nbs = neigh_idx[n]
            nbs = nbs[nbs >= 0]
            if nbs.size > 0:
                zm[n] = zt[nbs].mean(axis=0)
            else:
                zm[n] = 0.0
        z_neighbor_mean.append(zm)
    z_neighbor_mean = np.stack(z_neighbor_mean, axis=0)  # (T,Nvalid,z)

    # Build dataset indices
    X_center = []
    X_neigh = []
    X_count = []
    y = []

    for t in range(T - 1):
        pre = semantic_maps[t][valid_mask].reshape(-1)
        post = semantic_maps[t + 1][valid_mask].reshape(-1)

        # optionally exclude boundary-like code
        mask_ok = np.ones_like(pre, dtype=bool)
        if args.exclude_boundary_code is not None:
            mask_ok &= (pre != args.exclude_boundary_code)
            mask_ok &= (post != args.exclude_boundary_code)

        idxs = np.where(mask_ok)[0]
        for n in idxs:
            zc = z_by_state[t, n].detach().cpu().numpy()
            znm = z_neighbor_mean[t, n]
            cnt = neigh_count_tensor[n:n+1] / 4.0  # normalize
            cls = pair_to_class[(int(pre[n]), int(post[n]))]
            X_center.append(zc)
            X_neigh.append(znm)
            X_count.append(cnt)
            y.append(cls)

    X_center = torch.tensor(np.stack(X_center, axis=0), dtype=torch.float32)
    X_neigh = torch.tensor(np.stack(X_neigh, axis=0), dtype=torch.float32)
    X_count = torch.tensor(np.stack(X_count, axis=0), dtype=torch.float32)
    y = torch.tensor(np.array(y, dtype=np.int64), dtype=torch.long)

    print("train samples:", int(y.shape[0]))

    model = TransitionNet(z_dim=z_dim, hidden=args.hidden, n_classes=n_classes, dropout=0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # class weights
    counts = torch.bincount(y, minlength=n_classes).float()
    w = (counts.sum() / (counts + 1e-8))
    w = w / w.mean()
    w = w.to(device)

    ds = torch.utils.data.TensorDataset(X_center, X_neigh, X_count, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    for ep in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for zc, znm, cnt, yy in loader:
            zc, znm, cnt, yy = zc.to(device), znm.to(device), cnt.to(device), yy.to(device)
            logits = model(zc, znm, cnt)
            loss = F.cross_entropy(logits, yy, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * yy.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yy).sum().item()
            seen += yy.size(0)
        print(f"epoch {ep+1}/{args.epochs}  loss={total_loss/seen:.4e}  acc={correct/seen:.4f}")

    # Save model + mapping
    torch.save({
        "model_state": model.state_dict(),
        "z_dim": z_dim,
        "n_classes": n_classes,
        "transition_pairs": pairs,
    }, os.path.join(args.output_dir, "transition_model.pt"))

    with open(os.path.join(args.output_dir, "transition_pairs.json"), "w") as f:
        json.dump({"pairs": pairs}, f, indent=2)

    print("Saved transition model to:", args.output_dir)

    # ---- Interpretability (gradient saliency on latent inputs) ----
    # For one random sample, compute grads w.r.t. z_center and z_neigh_mean.
    model.eval()
    i0 = np.random.randint(0, y.shape[0])
    zc = X_center[i0:i0+1].to(device).requires_grad_(True)
    znm = X_neigh[i0:i0+1].to(device).requires_grad_(True)
    cnt = X_count[i0:i0+1].to(device)

    logits = model(zc, znm, cnt)
    pred_cls = int(logits.argmax(dim=1).item())
    logit = logits[0, pred_cls]
    grads = torch.autograd.grad(logit, [zc, znm], retain_graph=False)

    gzc = grads[0].detach().cpu().numpy()[0]   # (z_dim,)
    gzn = grads[1].detach().cpu().numpy()[0]   # (z_dim,)

    np.save(os.path.join(args.output_dir, "latent_saliency_center.npy"), np.abs(gzc))
    np.save(os.path.join(args.output_dir, "latent_saliency_neighbors.npy"), np.abs(gzn))

    print("Wrote latent saliency (abs gradients) for 1 sample:")
    print("  ", os.path.join(args.output_dir, "latent_saliency_center.npy"))
    print("  ", os.path.join(args.output_dir, "latent_saliency_neighbors.npy"))

    # If you want (E,phi) saliency too, you can extend this by backpropagating through the encoder.
    # Tell me your preferred preprocessing (exact crop/downsample) and I’ll add that function.


if __name__ == "__main__":
    main()