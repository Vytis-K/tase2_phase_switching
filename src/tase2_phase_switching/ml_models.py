"""ML models and training utilities for predictive phase-transition cartography."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Iterator
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_spectra_tensor(
    data_array_values: np.ndarray,
    valid_mask: np.ndarray,
) -> torch.Tensor:
    """Flatten (x, y, eV, phi) → (N_valid, eV*phi) for valid pixels only."""
    x, y, e, p = data_array_values.shape
    flat = data_array_values.reshape(x * y, e * p).astype(np.float32)
    valid_flat = valid_mask.reshape(-1)
    spectra = flat[valid_flat]
    total = spectra.sum(axis=1, keepdims=True) + 1e-8
    return torch.from_numpy(spectra / total)


def build_energy_profiles(
    data_array_values: np.ndarray,
    valid_mask: np.ndarray,
) -> torch.Tensor:
    """Flatten (x, y, eV, phi) → (N_valid, eV) energy profiles (phi-summed, normalised)."""
    x, y, e, p = data_array_values.shape
    profiles = data_array_values.sum(axis=3).reshape(x * y, e).astype(np.float32)
    valid_flat = valid_mask.reshape(-1)
    profiles = profiles[valid_flat]
    total = profiles.sum(axis=1, keepdims=True) + 1e-8
    return torch.from_numpy(profiles / total)


def build_neighborhood_embeddings(
    embeddings_map: np.ndarray,
    valid_mask: np.ndarray,
    radius: int = 1,
) -> torch.Tensor:
    """Average embeddings in a square neighbourhood around each valid pixel."""
    x_size, y_size, d = embeddings_map.shape
    result = np.zeros((x_size, y_size, d), dtype=np.float32)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            shifted = np.roll(np.roll(embeddings_map, dx, axis=0), dy, axis=1)
            result += shifted
    count = (2 * radius + 1) ** 2
    result /= count
    valid_flat = valid_mask.reshape(-1)
    flat = result.reshape(x_size * y_size, d)
    return torch.from_numpy(flat[valid_flat])


def build_transition_labels(
    simple_state_code_maps: list[np.ndarray],
    valid_mask: np.ndarray,
    from_index: int,
    to_index: int,
) -> torch.Tensor:
    """9-class transition label: from_code * 3 + to_code for each valid pixel."""
    fc = simple_state_code_maps[from_index]
    tc = simple_state_code_maps[to_index]
    valid_flat = valid_mask.reshape(-1)
    fc_valid = fc.reshape(-1)[valid_flat]
    tc_valid = tc.reshape(-1)[valid_flat]
    labels = fc_valid * 3 + tc_valid
    labels = np.clip(labels, 0, 8)
    return torch.from_numpy(labels.astype(np.int64))


# ---------------------------------------------------------------------------
# Stage A: Encoder (1D convolutional autoencoder on energy profiles)
# ---------------------------------------------------------------------------

class PatchEncoder(nn.Module):
    """Conv1D autoencoder over energy profiles. Produces a compact embedding."""

    def __init__(self, input_dim: int, latent_dim: int = 16, hidden: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, hidden)
        self.enc_fc2 = nn.Linear(hidden, hidden)
        self.enc_out = nn.Linear(hidden, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, hidden)
        self.dec_fc2 = nn.Linear(hidden, hidden)
        self.dec_out = nn.Linear(hidden, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.enc_fc1(x))
        h = F.gelu(self.enc_fc2(h))
        return self.enc_out(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.dec_fc1(z))
        h = F.gelu(self.dec_fc2(h))
        return torch.sigmoid(self.dec_out(h))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


# ---------------------------------------------------------------------------
# Stage C: Transition predictor MLP
# ---------------------------------------------------------------------------

class TransitionPredictor(nn.Module):
    """Predicts transition class from pre-pulse embedding + neighbourhood embedding."""

    def __init__(self, embedding_dim: int, n_classes: int = 9, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, emb: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb, nbr], dim=1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Training generators (yield loss dicts so the UI can plot live)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 64
    n_epochs: int = 60
    weight_decay: float = 1e-4
    device: str = "cpu"


def train_encoder(
    spectra: torch.Tensor,
    config: TrainConfig,
    latent_dim: int = 16,
    hidden: int = 64,
    stop_flag: list[bool] | None = None,
) -> Generator[dict[str, Any], None, PatchEncoder]:
    """
    Yields {"epoch": int, "loss": float, "phase": "encoder"} each epoch.
    Returns the trained PatchEncoder when done (StopIteration value).
    """
    device = torch.device(config.device)
    model = PatchEncoder(spectra.shape[1], latent_dim=latent_dim, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.n_epochs)
    dataset = TensorDataset(spectra)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    for epoch in range(1, config.n_epochs + 1):
        if stop_flag and stop_flag[0]:
            break
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            _, recon = model(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch)
        scheduler.step()
        avg_loss = total_loss / max(1, len(spectra))
        yield {"epoch": epoch, "n_epochs": config.n_epochs, "loss": avg_loss, "phase": "encoder"}

    return model


def train_predictor(
    embeddings: torch.Tensor,
    neighborhoods: torch.Tensor,
    labels: torch.Tensor,
    config: TrainConfig,
    n_classes: int = 9,
    hidden: int = 64,
    stop_flag: list[bool] | None = None,
) -> Generator[dict[str, Any], None, TransitionPredictor]:
    """
    Yields {"epoch": int, "loss": float, "acc": float, "phase": "predictor"} each epoch.
    Returns the trained TransitionPredictor.
    """
    device = torch.device(config.device)
    emb_dim = embeddings.shape[1]
    model = TransitionPredictor(emb_dim, n_classes=n_classes, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.n_epochs)

    # Class-balanced weights
    class_counts = torch.bincount(labels, minlength=n_classes).float()
    class_weights = (class_counts.sum() / (n_classes * class_counts.clamp(min=1))).to(device)

    dataset = TensorDataset(embeddings, neighborhoods, labels)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    for epoch in range(1, config.n_epochs + 1):
        if stop_flag and stop_flag[0]:
            break
        model.train()
        total_loss = 0.0
        correct = 0
        n_total = 0
        for emb_b, nbr_b, lbl_b in loader:
            emb_b, nbr_b, lbl_b = emb_b.to(device), nbr_b.to(device), lbl_b.to(device)
            opt.zero_grad()
            logits = model(emb_b, nbr_b)
            loss = F.cross_entropy(logits, lbl_b, weight=class_weights)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(lbl_b)
            correct += (logits.argmax(1) == lbl_b).sum().item()
            n_total += len(lbl_b)
        scheduler.step()
        avg_loss = total_loss / max(1, n_total)
        acc = correct / max(1, n_total)
        yield {"epoch": epoch, "n_epochs": config.n_epochs, "loss": avg_loss, "acc": acc, "phase": "predictor"}

    return model


# ---------------------------------------------------------------------------
# Stage B: Cluster embeddings
# ---------------------------------------------------------------------------

def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """KMeans on learned embeddings. Returns (labels, centroids)."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels.astype(np.int32), km.cluster_centers_.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage D: Interpretability
# ---------------------------------------------------------------------------

def compute_saliency(
    model: PatchEncoder,
    spectra: torch.Tensor,
    n_samples: int = 200,
) -> np.ndarray:
    """Gradient saliency: d||z||/dx for each input dimension."""
    model.eval()
    device = next(model.parameters()).device
    indices = np.random.choice(len(spectra), size=min(n_samples, len(spectra)), replace=False)
    sample = spectra[indices].to(device).requires_grad_(True)
    z = model.encode(sample)
    score = (z ** 2).sum()
    score.backward()
    saliency = sample.grad.abs().mean(dim=0).cpu().detach().numpy()
    return saliency


def compute_transition_saliency(
    encoder: PatchEncoder,
    predictor: TransitionPredictor,
    spectra: torch.Tensor,
    neighborhoods: torch.Tensor,
    target_class: int,
    n_samples: int = 100,
) -> np.ndarray:
    """Gradient of predictor logit[target_class] w.r.t. input spectrum."""
    encoder.eval()
    predictor.eval()
    device = next(encoder.parameters()).device
    indices = np.random.choice(len(spectra), size=min(n_samples, len(spectra)), replace=False)
    spec = spectra[indices].to(device).requires_grad_(True)
    nbr = neighborhoods[indices].to(device)
    z = encoder.encode(spec)
    logits = predictor(z, nbr)
    logits[:, target_class].sum().backward()
    saliency = spec.grad.abs().mean(dim=0).cpu().detach().numpy()
    return saliency


def find_prototypes(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    n_proto: int = 3,
) -> dict[int, np.ndarray]:
    """For each cluster, return indices of pixels closest to the centroid."""
    from sklearn.metrics import pairwise_distances
    prototypes: dict[int, np.ndarray] = {}
    unique = np.unique(cluster_labels)
    for cluster_id in unique:
        mask = cluster_labels == cluster_id
        cluster_embs = embeddings[mask]
        centroid = cluster_embs.mean(axis=0, keepdims=True)
        dists = pairwise_distances(cluster_embs, centroid).ravel()
        top_k = np.argsort(dists)[:n_proto]
        global_indices = np.where(mask)[0][top_k]
        prototypes[int(cluster_id)] = global_indices
    return prototypes


# ---------------------------------------------------------------------------
# Model save / load
# ---------------------------------------------------------------------------

@dataclass
class MLResult:
    encoder: PatchEncoder | None = None
    encoder_losses: list[float] = field(default_factory=list)
    predictor: TransitionPredictor | None = None
    predictor_losses: list[float] = field(default_factory=list)
    predictor_accs: list[float] = field(default_factory=list)
    cluster_labels: np.ndarray | None = None
    cluster_centroids: np.ndarray | None = None
    embeddings_by_state: list[np.ndarray] = field(default_factory=list)
    saliency: np.ndarray | None = None
    transition_saliency: dict[int, np.ndarray] = field(default_factory=dict)
    prototype_indices: dict[int, np.ndarray] = field(default_factory=dict)
    n_clusters: int = 8
    from_state_index: int = 0
    to_state_index: int = 1


def save_ml_result(result: MLResult, path: str) -> None:
    payload: dict[str, Any] = {
        "encoder_losses": result.encoder_losses,
        "predictor_losses": result.predictor_losses,
        "predictor_accs": result.predictor_accs,
        "cluster_labels": result.cluster_labels,
        "cluster_centroids": result.cluster_centroids,
        "embeddings_by_state": result.embeddings_by_state,
        "saliency": result.saliency,
        "transition_saliency": {str(k): v for k, v in result.transition_saliency.items()},
        "prototype_indices": {str(k): v for k, v in result.prototype_indices.items()},
        "n_clusters": result.n_clusters,
        "from_state_index": result.from_state_index,
        "to_state_index": result.to_state_index,
    }
    torch_payload: dict[str, Any] = {}
    if result.encoder is not None:
        torch_payload["encoder_state"] = result.encoder.state_dict()
        torch_payload["encoder_input_dim"] = result.encoder.input_dim
        torch_payload["encoder_latent_dim"] = result.encoder.latent_dim
    if result.predictor is not None:
        torch_payload["predictor_state"] = result.predictor.state_dict()
    torch.save({**payload, **torch_payload}, path)


def load_ml_result(path: str) -> MLResult:
    data = torch.load(path, map_location="cpu", weights_only=False)
    result = MLResult(
        encoder_losses=data.get("encoder_losses", []),
        predictor_losses=data.get("predictor_losses", []),
        predictor_accs=data.get("predictor_accs", []),
        cluster_labels=data.get("cluster_labels"),
        cluster_centroids=data.get("cluster_centroids"),
        embeddings_by_state=data.get("embeddings_by_state", []),
        saliency=data.get("saliency"),
        transition_saliency={int(k): v for k, v in data.get("transition_saliency", {}).items()},
        prototype_indices={int(k): v for k, v in data.get("prototype_indices", {}).items()},
        n_clusters=data.get("n_clusters", 8),
        from_state_index=data.get("from_state_index", 0),
        to_state_index=data.get("to_state_index", 1),
    )
    if "encoder_state" in data:
        enc = PatchEncoder(data["encoder_input_dim"], latent_dim=data["encoder_latent_dim"])
        enc.load_state_dict(data["encoder_state"])
        result.encoder = enc
    if "predictor_state" in data and result.encoder is not None:
        pred = TransitionPredictor(result.encoder.latent_dim)
        pred.load_state_dict(data["predictor_state"])
        result.predictor = pred
    return result
