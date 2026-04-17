import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tv_loss(w_patch: torch.Tensor) -> torch.Tensor:
    """
    Total variation loss on patch weights.
    w_patch: (B, P, P, K)
    """
    dx = w_patch[:, 1:, :, :] - w_patch[:, :-1, :, :]
    dy = w_patch[:, :, 1:, :] - w_patch[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def entropy_loss(w: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    Encourage lower-entropy mixtures (crisper components).
    w: (N, K) simplex
    """
    return (-w * torch.log(w + eps)).sum(dim=-1).mean()


def save_img(path, img2d, title="", xlabel="", ylabel=""):
    plt.figure(figsize=(5, 4))
    plt.imshow(img2d.T, origin="lower", aspect="auto")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_spectrum(path, spec2d, title=""):
    plt.figure(figsize=(6, 4))
    plt.imshow(spec2d.T, origin="lower", aspect="auto")
    plt.title(title)
    plt.xlabel("k index")
    plt.ylabel("E index")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ----------------------------
# Dataset: sample patches so we can do spatial regularization
# ----------------------------
class PatchARPESDataset(Dataset):
    """
    Produces patches of spectra from a cube for TV regularization.
    Each sample:
      x_patch: (P, P, 1, Nk, Ne)
      meta: (sid, i0, j0)
    """
    def __init__(self, cubes, patch=8, stride=8, max_patches_per_cube=None, seed=0):
        """
        cubes: list of np arrays each (nx, ny, Nk, Ne)
        """
        self.cubes = cubes
        self.patch = patch
        self.items = []
        rng = np.random.default_rng(seed)

        for sid, cube in enumerate(cubes):
            coords = []
            for i0 in range(0, cube.shape[0] - patch + 1, stride):
                for j0 in range(0, cube.shape[1] - patch + 1, stride):
                    coords.append((sid, i0, j0))
            rng.shuffle(coords)
            if max_patches_per_cube is not None:
                coords = coords[:max_patches_per_cube]
            self.items.extend(coords)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sid, i0, j0 = self.items[idx]
        cube = self.cubes[sid]
        x = cube[i0:i0+self.patch, j0:j0+self.patch]          # (P,P,Nk,Ne)
        x = x[:, :, None, :, :]                               # (P,P,1,Nk,Ne)
        return torch.from_numpy(x.astype(np.float32)), torch.tensor([sid, i0, j0], dtype=torch.long)


# ----------------------------
# Model: Unwarp + Unmix
# ----------------------------
class UnwarpUnmix(nn.Module):
    """
    Two-stage in one model:
      - CNN predicts per-spectrum affine params (theta, shear, tx, ty) and mixture weights w
      - Learn K nonnegative basis spectra S_k (canonical)
      - Reconstruct canonical spectrum as sum_k w_k S_k
      - Self-supervised losses:
           (1) alignment: invwarp(x_obs) ~ x_can_hat
           (2) cycle: warp(x_can_hat) ~ x_obs
      - Output: theta map, weights, bases
    """
    def __init__(self, Nk, Ne, K=3):
        super().__init__()
        self.Nk, self.Ne, self.K = Nk, Ne, K

        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )

        # Affine params: theta, shear, tx, ty
        self.out_aff = nn.Linear(128, 4)

        # Simplex weights
        self.out_w = nn.Linear(128, K)

        # Canonical basis spectra (nonnegative)
        self.S_raw = nn.Parameter(0.05 * torch.randn(K, Nk, Ne))

    def _A(self, theta, shear, tx, ty):
        """
        Build (B,2,3) affine matrix for grid_sample (normalized coords).
        """
        B = theta.shape[0]
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        A = torch.zeros((B, 2, 3), device=theta.device, dtype=theta.dtype)
        A[:, 0, 0] = cos
        A[:, 0, 1] = -sin + shear
        A[:, 1, 0] = sin
        A[:, 1, 1] = cos
        A[:, 0, 2] = tx
        A[:, 1, 2] = ty
        return A

    def warp(self, img, A):
        """
        img: (B,1,Nk,Ne)
        A: (B,2,3)
        """
        grid = F.affine_grid(A, img.shape, align_corners=False)
        return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def forward(self, x_obs):
        """
        x_obs: (B,1,Nk,Ne)
        returns:
          x_can_obs: invwarped observed into canonical
          x_can_hat: canonical reconstruction from mixture
          x_hat_obs: forward-warped reconstruction back to observed
          theta_deg, shear, tx, ty
          w (B,K)
          S (K,Nk,Ne)
        """
        h = self.head(self.enc(x_obs))

        a = self.out_aff(h)
        # constrain transforms (tweak as needed)
        theta = 0.35 * torch.tanh(a[:, 0])   # radians ~ +/-20 deg
        shear = 0.20 * torch.tanh(a[:, 1])
        tx    = 0.12 * torch.tanh(a[:, 2])   # normalized coords
        ty    = 0.12 * torch.tanh(a[:, 3])

        w = F.softmax(self.out_w(h), dim=-1)  # simplex

        # nonnegative bases
        S = F.softplus(self.S_raw)
        S = S / (S.amax(dim=(1,2), keepdim=True) + 1e-8)

        # canonical recon
        x_can_hat = torch.einsum("bk,kij->bij", w, S).unsqueeze(1)  # (B,1,Nk,Ne)

        # inverse-warp observed -> canonical
        # (approx inverse: good for small transforms; if needed, compute true inverse A^{-1})
        A_inv = self._A(-theta, -shear, -tx, -ty)
        x_can_obs = self.warp(x_obs, A_inv)

        # forward-warp canonical recon -> observed
        A_fwd = self._A(theta, shear, tx, ty)
        x_hat_obs = self.warp(x_can_hat, A_fwd)

        theta_deg = theta * (180.0 / math.pi)
        return x_can_obs, x_can_hat, x_hat_obs, theta_deg, shear, tx, ty, w, S


# ----------------------------
# Training loop (self-supervised)
# ----------------------------
def train_unwarp_unmix(
    cubes,
    Nk, Ne,
    K=3,
    patch=8,
    stride=8,
    max_patches_per_cube=60,
    batch_size=4,
    epochs=10,
    lr=2e-3,
    device="cpu",
    outdir="idea1_out",
    seed=0,
):
    os.makedirs(outdir, exist_ok=True)
    set_seed(seed)

    ds = PatchARPESDataset(
        cubes=cubes,
        patch=patch,
        stride=stride,
        max_patches_per_cube=max_patches_per_cube,
        seed=seed,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = UnwarpUnmix(Nk, Ne, K=K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # loss weights (tune these)
    lam_align = 1.0   # invwarp(x_obs) matches canonical recon
    lam_cycle = 0.7   # warp(canonical recon) matches x_obs
    lam_tv    = 0.2   # patch smoothness in w
    lam_ent   = 0.02  # slightly prefer crisp mixtures

    history = []
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n_batches = 0

        for x_patch, meta in loader:
            # x_patch: (B,P,P,1,Nk,Ne)
            B, P, _, C, _, _ = x_patch.shape
            x_obs = x_patch.reshape(B * P * P, C, Nk, Ne).to(device)

            x_can_obs, x_can_hat, x_hat_obs, theta_deg, shear, tx, ty, w, S = model(x_obs)

            loss_align = F.mse_loss(x_can_obs, x_can_hat)
            loss_cycle = F.mse_loss(x_hat_obs, x_obs)

            w_patch = w.reshape(B, P, P, K)
            loss_tv = tv_loss(w_patch)

            loss_ent = entropy_loss(w)

            loss = (
                lam_align * loss_align
                + lam_cycle * loss_cycle
                + lam_tv * loss_tv
                + lam_ent * loss_ent
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            ep_loss += loss.item()
            n_batches += 1

        ep_loss /= max(n_batches, 1)
        history.append(ep_loss)

        if ep == 1 or ep % 2 == 0 or ep == epochs:
            print(f"epoch {ep:02d} loss={ep_loss:.5f}")

    train_time = time.time() - t0

    # Save model + training curve
    torch.save(model.state_dict(), os.path.join(outdir, "unwarp_unmix_model.pt"))

    plt.figure(figsize=(5,3))
    plt.plot(history)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "train_curve.png"), dpi=200)
    plt.close()

    print(f"Training done in {train_time:.1f}s, saved to {outdir}")
    return model


# ----------------------------
# Apply model to full cube -> per-pixel outputs
# ----------------------------
@torch.no_grad()
def apply_model_to_cube(model, cube, device="cpu", batch=512):
    """
    cube: (nx,ny,Nk,Ne)
    Returns:
      theta_map, shear_map, tx_map, ty_map, w_map, recon_err_map
      plus bases S (K,Nk,Ne)
    """
    model.eval()
    nx, ny, Nk, Ne = cube.shape
    flat = cube.reshape(nx * ny, Nk, Ne).astype(np.float32)

    theta = np.zeros((nx * ny,), np.float32)
    shear = np.zeros((nx * ny,), np.float32)
    tx    = np.zeros((nx * ny,), np.float32)
    ty    = np.zeros((nx * ny,), np.float32)
    w_map = np.zeros((nx * ny, model.K), np.float32)
    err   = np.zeros((nx * ny,), np.float32)

    for s in range(0, nx * ny, batch):
        x = torch.from_numpy(flat[s:s+batch]).unsqueeze(1).to(device)  # (B,1,Nk,Ne)

        x_can_obs, x_can_hat, x_hat_obs, thdeg, sh, txx, tyy, w, S = model(x)

        theta[s:s+len(thdeg)] = thdeg.cpu().numpy()
        shear[s:s+len(thdeg)] = sh.cpu().numpy()
        tx[s:s+len(thdeg)]    = txx.cpu().numpy()
        ty[s:s+len(thdeg)]    = tyy.cpu().numpy()
        w_map[s:s+len(thdeg)] = w.cpu().numpy()

        # reconstruction error in observed space
        mse = ((x_hat_obs - x) ** 2).mean(dim=(1,2,3)).cpu().numpy()
        err[s:s+len(thdeg)] = mse

    theta = theta.reshape(nx, ny)
    shear = shear.reshape(nx, ny)
    tx    = tx.reshape(nx, ny)
    ty    = ty.reshape(nx, ny)
    w_map = w_map.reshape(nx, ny, model.K)
    err   = err.reshape(nx, ny)

    # basis spectra (canonical)
    S_np = S.cpu().numpy()  # (K,Nk,Ne)
    return theta, shear, tx, ty, w_map, err, S_np


# ----------------------------
# Main runnable
# ----------------------------
def main():
    npz_path = "synth_tas2_pulsing_v1.npz"
    outdir = "idea1_out"

    d = np.load(npz_path)
    cube0  = d["cube0"].astype(np.float32)
    cubeAB = d["cubeAB"].astype(np.float32)
    theta_gt = d["theta_map"].astype(np.float32)  # ONLY for sanity plotting (not used in training)

    nx, ny, Nk, Ne = cube0.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # 1) Train self-supervised
    model = train_unwarp_unmix(
        cubes=[cube0, cubeAB],
        Nk=Nk, Ne=Ne,
        K=3,
        patch=8,
        stride=8,
        max_patches_per_cube=80,
        batch_size=4,
        epochs=12,
        lr=2e-3,
        device=device,
        outdir=outdir,
        seed=0,
    )

    # 2) Apply to each cube
    th0, sh0, tx0, ty0, w0_pred, err0, S = apply_model_to_cube(model, cube0, device=device, batch=512)
    thAB, shAB, txAB, tyAB, wAB_pred, errAB, _ = apply_model_to_cube(model, cubeAB, device=device, batch=512)

    # 3) Save predictions
    np.savez(
        os.path.join(outdir, "predictions.npz"),
        theta_cube0=th0, theta_cubeAB=thAB,
        shear_cube0=sh0, shear_cubeAB=shAB,
        tx_cube0=tx0, tx_cubeAB=txAB,
        ty_cube0=ty0, ty_cubeAB=tyAB,
        w_cube0=w0_pred, w_cubeAB=wAB_pred,
        recon_err_cube0=err0, recon_err_cubeAB=errAB,
        S_basis=S,
        theta_gt=theta_gt,  # only for checking synthetic
    )

    # 4) Plots (outputs you want)
    os.makedirs(outdir, exist_ok=True)

    # Orientation maps
    save_img(os.path.join(outdir, "theta_gt.png"), theta_gt, "theta GT (deg)", "x", "y")
    save_img(os.path.join(outdir, "theta_pred_cube0.png"), th0, "theta pred (cube0)", "x", "y")
    save_img(os.path.join(outdir, "theta_pred_cubeAB.png"), thAB, "theta pred (cubeAB)", "x", "y")
    save_img(os.path.join(outdir, "theta_err_cube0.png"), th0 - theta_gt, "theta pred - GT (cube0)", "x", "y")

    # Mixture weight maps for component 0/1/2
    for j in range(S.shape[0]):
        save_img(os.path.join(outdir, f"w{j}_cube0.png"), w0_pred[..., j], f"w[{j}] (cube0)", "x", "y")
        save_img(os.path.join(outdir, f"w{j}_cubeAB.png"), wAB_pred[..., j], f"w[{j}] (cubeAB)", "x", "y")

    # Reconstruction error map
    save_img(os.path.join(outdir, "recon_err_cube0.png"), err0, "reconstruction MSE (cube0)", "x", "y")
    save_img(os.path.join(outdir, "recon_err_cubeAB.png"), errAB, "reconstruction MSE (cubeAB)", "x", "y")

    # Basis spectra
    for j in range(S.shape[0]):
        save_spectrum(os.path.join(outdir, f"S{j}_basis.png"), S[j], title=f"Learned basis S[{j}]")

    # Scatter: GT vs pred (synthetic sanity check)
    gt = theta_gt.reshape(-1)
    pr = th0.reshape(-1)
    plt.figure(figsize=(4.5, 4))
    plt.scatter(gt, pr, s=5, alpha=0.4)
    lo = min(gt.min(), pr.min())
    hi = max(gt.max(), pr.max())
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("theta GT (deg)")
    plt.ylabel("theta pred (deg)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "theta_scatter_cube0.png"), dpi=200)
    plt.close()

    print("Wrote outputs to:", outdir)


if __name__ == "__main__":
    main()
