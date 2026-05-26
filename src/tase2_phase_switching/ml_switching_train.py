from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import csv
import json
from typing import Any

import numpy as np


@dataclass(slots=True)
class TrainConfig:
    dataset: Path
    output_dir: Path
    target: str = "future_active"
    feature_set: str = "all"
    model: str = "logistic"
    test_fraction: float = 0.20
    val_fraction: float = 0.20
    spatial_block_size: int = 6
    seed: int = 42
    epochs: int = 800
    lr: float = 0.03
    l2: float = 1e-3
    class_weight: str = "balanced"
    shuffle_labels: bool = False


def train_from_dataset(config: TrainConfig) -> dict[str, Path]:
    output_path = config.output_dir.expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    arrays = np.load(config.dataset, allow_pickle=True)
    metadata = load_metadata(config.dataset)
    feature_names = [str(name) for name in arrays["feature_names"].tolist()]
    target_names = [str(name) for name in arrays["target_names"].tolist()]
    if config.target not in target_names:
        raise ValueError(f"Unknown target {config.target!r}. Available targets: {target_names}")

    selected_features = select_feature_names(feature_names, metadata, config.feature_set)
    selected_indices = [feature_names.index(name) for name in selected_features]
    x_raw = np.asarray(arrays["X"], dtype=np.float32)[:, selected_indices]
    y_all = np.asarray(arrays["targets"] if "targets" in arrays.files else arrays["y"])
    y_raw = y_all[:, target_names.index(config.target)].astype(np.int64)
    coords = np.stack([arrays["x"].astype(np.int16), arrays["y"].astype(np.int16)], axis=1)
    map_shape = tuple(int(v) for v in arrays["map_shape"].tolist())

    if config.shuffle_labels:
        rng = np.random.default_rng(config.seed)
        y_raw = rng.permutation(y_raw)

    split = make_spatial_block_split(
        coords,
        map_shape,
        test_fraction=config.test_fraction,
        val_fraction=config.val_fraction,
        block_size=config.spatial_block_size,
        seed=config.seed,
    )
    x_train, x_val, x_test = x_raw[split == "train"], x_raw[split == "val"], x_raw[split == "test"]
    y_train, y_val, y_test = y_raw[split == "train"], y_raw[split == "val"], y_raw[split == "test"]

    preprocessor = fit_robust_preprocessor(x_train)
    x_train_s = apply_robust_preprocessor(x_train, preprocessor)
    x_val_s = apply_robust_preprocessor(x_val, preprocessor)
    x_test_s = apply_robust_preprocessor(x_test, preprocessor)

    class_values = np.unique(y_train)
    if class_values.size < 2:
        raise ValueError(f"Training split for target {config.target!r} contains fewer than two classes.")
    class_to_index = {int(value): idx for idx, value in enumerate(class_values.tolist())}
    y_train_i = encode_classes(y_train, class_to_index)
    y_val_i = encode_classes(y_val, class_to_index)
    y_test_i = encode_classes(y_test, class_to_index)

    fit = fit_linear_classifier(
        x_train_s,
        y_train_i,
        x_val_s,
        y_val_i,
        n_classes=class_values.size,
        config=config,
    )

    decision_threshold = binary_validation_threshold(fit["weights"], fit["bias"], x_val_s, y_val_i, class_values)
    train_metrics = evaluate_classifier(fit["weights"], fit["bias"], x_train_s, y_train_i, class_values, decision_threshold)
    val_metrics = evaluate_classifier(fit["weights"], fit["bias"], x_val_s, y_val_i, class_values, decision_threshold)
    test_metrics = evaluate_classifier(fit["weights"], fit["bias"], x_test_s, y_test_i, class_values, decision_threshold)

    x_all_s = apply_robust_preprocessor(x_raw, preprocessor)
    proba_all = predict_proba(fit["weights"], fit["bias"], x_all_s)
    pred_i = predict_indices_from_probabilities(proba_all, class_values, decision_threshold)
    pred_raw = class_values[pred_i]

    model_path = output_path / "switching_predictor_model.npz"
    np.savez_compressed(
        model_path,
        weights=fit["weights"],
        bias=fit["bias"],
        center=preprocessor["center"],
        scale=preprocessor["scale"],
        fill=preprocessor["fill"],
        feature_names=np.asarray(selected_features),
        target=np.asarray([config.target]),
        class_values=class_values,
    )

    predictions_path = output_path / "predictions.csv"
    write_prediction_rows(predictions_path, coords, y_raw, pred_raw, proba_all, class_values, split)

    feature_importance_path = output_path / "feature_importance.csv"
    write_feature_importance(feature_importance_path, selected_features, fit["weights"])

    maps_dir = output_path / "maps"
    maps_dir.mkdir(exist_ok=True)
    transition_index = arrays["transition_index"].astype(np.int16) if "transition_index" in arrays.files else None
    transition_names = [str(name) for name in arrays["transition_names"].tolist()] if "transition_names" in arrays.files else None
    comparison_paths = write_prediction_maps(
        maps_dir,
        coords,
        map_shape,
        y_raw,
        pred_raw,
        proba_all,
        class_values,
        split,
        config.target,
        transition_index=transition_index,
        transition_names=transition_names,
    )

    metrics_path = output_path / "metrics.json"
    metrics = {
        "config": config_for_json(config),
        "dataset_metadata": {
            "dataset": str(config.dataset.expanduser().resolve()),
            "files": metadata.get("files", []),
            "transition_mode": metadata.get("parameters", {}).get("transition_mode"),
            "normalization_mode": metadata.get("parameters", {}).get("normalization_mode"),
            "feature_set": config.feature_set,
            "selected_features": selected_features,
            "target_counts_all": class_counts(y_raw),
            "target_counts_train": class_counts(y_train),
            "target_counts_val": class_counts(y_val),
            "target_counts_test": class_counts(y_test),
        },
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "best_validation_epoch": fit["best_epoch"],
        "decision_threshold": decision_threshold,
        "loss_history": fit["history"],
        "visual_outputs": [str(path) for path in comparison_paths],
    }
    if class_values.size == 2:
        metrics["single_feature_baseline"] = single_feature_baseline(
            x_train_s,
            y_train_i,
            x_test_s,
            y_test_i,
            selected_features,
        )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_path = output_path / "training_report.md"
    report_path.write_text(training_report(metrics), encoding="utf-8")

    return {
        "model": model_path,
        "metrics": metrics_path,
        "predictions": predictions_path,
        "feature_importance": feature_importance_path,
        "maps": maps_dir,
        "report": report_path,
    }


def load_metadata(dataset_path: Path) -> dict[str, Any]:
    metadata_path = dataset_path.expanduser().resolve().parent / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def select_feature_names(feature_names: list[str], metadata: dict[str, Any], feature_set: str) -> list[str]:
    if feature_set == "all":
        return feature_names
    groups = metadata.get("feature_groups", {})
    selected: list[str] = []
    if feature_set == "spectral":
        selected = groups.get("spectral", [])
    elif feature_set == "spatial":
        selected = groups.get("spatial", [])
    elif feature_set in {"artifact", "artifact_position"}:
        selected = groups.get("artifact_position", [])
    elif feature_set == "spectral_spatial":
        selected = groups.get("spectral", []) + groups.get("spatial", [])
    elif feature_set == "no_position":
        excluded = set(groups.get("artifact_position", []))
        selected = [name for name in feature_names if name not in excluded]
    else:
        raise ValueError(
            "feature_set must be one of all, spectral, spatial, artifact, spectral_spatial, no_position."
        )
    selected = [name for name in dict.fromkeys(selected) if name in feature_names]
    if not selected:
        raise ValueError(f"Feature set {feature_set!r} did not select any available features.")
    return selected


def make_spatial_block_split(
    coords: np.ndarray,
    shape: tuple[int, int],
    test_fraction: float,
    val_fraction: float,
    block_size: int,
    seed: int,
) -> np.ndarray:
    if not 0.0 < test_fraction < 1.0 or not 0.0 <= val_fraction < 1.0:
        raise ValueError("test_fraction must be in (0,1), val_fraction must be in [0,1).")
    block_size = max(1, int(block_size))
    block_ids = (coords[:, 0] // block_size) * (1 + shape[1] // block_size) + (coords[:, 1] // block_size)
    unique_blocks = np.unique(block_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_blocks)
    n_test = max(1, int(round(unique_blocks.size * test_fraction)))
    n_val = max(1, int(round(unique_blocks.size * val_fraction))) if val_fraction > 0 else 0
    test_blocks = set(unique_blocks[:n_test].tolist())
    val_blocks = set(unique_blocks[n_test:n_test + n_val].tolist())
    split = np.full(coords.shape[0], "train", dtype=object)
    for index, block_id in enumerate(block_ids):
        if int(block_id) in test_blocks:
            split[index] = "test"
        elif int(block_id) in val_blocks:
            split[index] = "val"
    if np.count_nonzero(split == "train") == 0 or np.count_nonzero(split == "test") == 0:
        raise ValueError("Spatial split produced an empty train or test set; lower --spatial-block-size or split fractions.")
    if val_fraction > 0 and np.count_nonzero(split == "val") == 0:
        raise ValueError("Spatial split produced an empty validation set; lower --spatial-block-size or --val-fraction.")
    return split


def fit_robust_preprocessor(x_train: np.ndarray) -> dict[str, np.ndarray]:
    center = np.nanmedian(x_train, axis=0).astype(np.float32)
    q25 = np.nanpercentile(x_train, 25, axis=0).astype(np.float32)
    q75 = np.nanpercentile(x_train, 75, axis=0).astype(np.float32)
    scale = (q75 - q25).astype(np.float32)
    std = np.nanstd(x_train, axis=0).astype(np.float32)
    scale[~np.isfinite(scale) | (np.abs(scale) <= 1e-8)] = std[~np.isfinite(scale) | (np.abs(scale) <= 1e-8)]
    scale[~np.isfinite(scale) | (np.abs(scale) <= 1e-8)] = 1.0
    fill = center.copy()
    fill[~np.isfinite(fill)] = 0.0
    return {"center": center, "scale": scale, "fill": fill}


def apply_robust_preprocessor(x: np.ndarray, preprocessor: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32).copy()
    fill = preprocessor["fill"]
    missing = ~np.isfinite(values)
    if np.any(missing):
        values[missing] = np.take(fill, np.where(missing)[1])
    return ((values - preprocessor["center"]) / preprocessor["scale"]).astype(np.float32)


def encode_classes(y: np.ndarray, class_to_index: dict[int, int]) -> np.ndarray:
    encoded = np.empty(y.shape[0], dtype=np.int64)
    for index, value in enumerate(y.astype(np.int64)):
        if int(value) not in class_to_index:
            encoded[index] = -1
        else:
            encoded[index] = class_to_index[int(value)]
    if np.any(encoded < 0):
        missing = sorted(set(y[encoded < 0].astype(int).tolist()))
        raise ValueError(f"Validation/test split contains classes absent from training split: {missing}")
    return encoded


def fit_linear_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    config: TrainConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed)
    n_features = x_train.shape[1]
    weights = rng.normal(0.0, 0.01, size=(n_features, n_classes)).astype(np.float32)
    bias = np.zeros(n_classes, dtype=np.float32)
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = np.zeros_like(bias)
    v_b = np.zeros_like(bias)
    beta1, beta2 = 0.9, 0.999
    class_weights = balanced_class_weights(y_train, n_classes) if config.class_weight == "balanced" else np.ones(n_classes, dtype=np.float32)
    best_weights = weights.copy()
    best_bias = bias.copy()
    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        logits = x_train @ weights + bias
        probabilities = softmax(logits)
        y_onehot = onehot(y_train, n_classes)
        sample_weights = class_weights[y_train]
        loss = weighted_cross_entropy(probabilities, y_onehot, sample_weights) + 0.5 * config.l2 * float(np.sum(weights * weights))
        grad_logits = (probabilities - y_onehot) * sample_weights[:, None] / max(1.0, float(np.sum(sample_weights)))
        grad_w = x_train.T @ grad_logits + config.l2 * weights
        grad_b = np.sum(grad_logits, axis=0)

        m_w = beta1 * m_w + (1.0 - beta1) * grad_w
        v_w = beta2 * v_w + (1.0 - beta2) * (grad_w * grad_w)
        m_b = beta1 * m_b + (1.0 - beta1) * grad_b
        v_b = beta2 * v_b + (1.0 - beta2) * (grad_b * grad_b)
        m_w_hat = m_w / (1.0 - beta1 ** epoch)
        v_w_hat = v_w / (1.0 - beta2 ** epoch)
        m_b_hat = m_b / (1.0 - beta1 ** epoch)
        v_b_hat = v_b / (1.0 - beta2 ** epoch)
        weights -= config.lr * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)
        bias -= config.lr * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)

        val_prob = softmax(x_val @ weights + bias) if x_val.size else probabilities
        val_onehot = onehot(y_val, n_classes) if y_val.size else y_onehot
        val_weights = class_weights[y_val] if y_val.size else sample_weights
        val_loss = weighted_cross_entropy(val_prob, val_onehot, val_weights)
        history.append({"epoch": float(epoch), "train_loss": float(loss), "val_loss": float(val_loss)})
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_weights = weights.copy()
            best_bias = bias.copy()
            best_epoch = epoch

    return {
        "weights": best_weights,
        "bias": best_bias,
        "best_epoch": best_epoch,
        "history": history,
    }


def balanced_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = np.sum(counts) / np.maximum(counts, 1.0)
    weights /= np.nanmean(weights)
    return weights.astype(np.float32)


def onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.nanmax(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def weighted_cross_entropy(probabilities: np.ndarray, y_onehot: np.ndarray, sample_weights: np.ndarray) -> float:
    losses = -np.sum(y_onehot * np.log(np.clip(probabilities, 1e-8, 1.0)), axis=1)
    return float(np.sum(losses * sample_weights) / max(1e-8, float(np.sum(sample_weights))))


def predict_proba(weights: np.ndarray, bias: np.ndarray, x: np.ndarray) -> np.ndarray:
    return softmax(x @ weights + bias)


def binary_validation_threshold(
    weights: np.ndarray,
    bias: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_values: np.ndarray,
) -> float | None:
    if class_values.size != 2 or not np.any(class_values == 1) or x_val.size == 0:
        return None
    positive_index = int(np.where(class_values == 1)[0][0])
    probabilities = predict_proba(weights, bias, x_val)[:, positive_index]
    finite_probabilities = probabilities[np.isfinite(probabilities)]
    if finite_probabilities.size == 0:
        return 0.5
    candidates = np.unique(np.nanpercentile(finite_probabilities, np.linspace(5, 95, 37)))
    best_threshold = 0.5
    best_score = -np.inf
    y_positive = y_val == positive_index
    for threshold in candidates:
        predicted = probabilities >= threshold
        tp = float(np.count_nonzero(predicted & y_positive))
        tn = float(np.count_nonzero(~predicted & ~y_positive))
        fp = float(np.count_nonzero(predicted & ~y_positive))
        fn = float(np.count_nonzero(~predicted & y_positive))
        recall_pos = tp / max(1.0, tp + fn)
        recall_neg = tn / max(1.0, tn + fp)
        balanced_accuracy = 0.5 * (recall_pos + recall_neg)
        precision = tp / max(1.0, tp + fp)
        f1 = 2.0 * precision * recall_pos / max(1e-8, precision + recall_pos)
        score = 0.65 * balanced_accuracy + 0.35 * f1
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def predict_indices_from_probabilities(
    probabilities: np.ndarray,
    class_values: np.ndarray,
    binary_threshold: float | None,
) -> np.ndarray:
    if binary_threshold is not None and class_values.size == 2 and np.any(class_values == 1):
        positive_index = int(np.where(class_values == 1)[0][0])
        negative_index = 1 - positive_index
        return np.where(probabilities[:, positive_index] >= binary_threshold, positive_index, negative_index)
    return np.argmax(probabilities, axis=1)


def evaluate_classifier(
    weights: np.ndarray,
    bias: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    class_values: np.ndarray,
    binary_threshold: float | None = None,
) -> dict[str, Any]:
    probabilities = predict_proba(weights, bias, x)
    pred = predict_indices_from_probabilities(probabilities, class_values, binary_threshold)
    metrics = classification_metrics(y, pred, n_classes=class_values.size)
    if class_values.size == 2:
        positive_index = int(np.where(class_values == 1)[0][0]) if np.any(class_values == 1) else 1
        metrics["auroc"] = auc_roc(y == positive_index, probabilities[:, positive_index])
        metrics["average_precision"] = average_precision(y == positive_index, probabilities[:, positive_index])
    return metrics


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, Any]:
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        confusion[int(true), int(pred)] += 1
    accuracy = float(np.trace(confusion) / max(1, np.sum(confusion)))
    recalls = []
    precisions = []
    f1s = []
    for cls in range(n_classes):
        tp = float(confusion[cls, cls])
        fp = float(np.sum(confusion[:, cls]) - tp)
        fn = float(np.sum(confusion[cls, :]) - tp)
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.nanmean(recalls)),
        "macro_precision": float(np.nanmean(precisions)),
        "macro_recall": float(np.nanmean(recalls)),
        "macro_f1": float(np.nanmean(f1s)),
        "confusion_matrix": confusion.tolist(),
        "n_samples": int(y_true.shape[0]),
    }


def auc_roc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=np.float64)
    valid = np.isfinite(s)
    y = y[valid]
    s = s[valid]
    n_pos = int(np.count_nonzero(y))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    return float((np.sum(ranks[y]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=np.float64)
    valid = np.isfinite(s)
    y = y[valid]
    s = s[valid]
    n_pos = int(np.count_nonzero(y))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, y_sorted.size + 1)
    return float(np.sum(precision[y_sorted]) / n_pos)


def single_feature_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    best = {"feature": "", "train_auroc": -np.inf, "direction": 1.0}
    for idx, name in enumerate(feature_names):
        score = x_train[:, idx]
        auc = auc_roc(y_train == 1, score)
        direction = 1.0
        if np.isfinite(auc) and auc < 0.5:
            auc = 1.0 - auc
            direction = -1.0
        if np.isfinite(auc) and auc > float(best["train_auroc"]):
            best = {"feature": name, "train_auroc": float(auc), "direction": direction, "index": idx}
    if "index" not in best:
        return {"note": "No finite single-feature baseline could be computed."}
    test_score = x_test[:, int(best["index"])] * float(best["direction"])
    return {
        "feature": best["feature"],
        "train_auroc": best["train_auroc"],
        "test_auroc": auc_roc(y_test == 1, test_score),
        "test_average_precision": average_precision(y_test == 1, test_score),
    }


def class_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


def write_prediction_maps(
    maps_dir: Path,
    coords: np.ndarray,
    map_shape: tuple[int, int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_values: np.ndarray,
    split: np.ndarray,
    target: str,
    transition_index: np.ndarray | None = None,
    transition_names: list[str] | None = None,
) -> list[Path]:
    if transition_index is None:
        prefix = target
        actual_map, pred_map, prob_map, topk_map, error_map, split_map = maps_for_rows(
            coords,
            map_shape,
            y_true,
            y_pred,
            probabilities,
            class_values,
            split,
        )
        np.save(maps_dir / f"actual_{prefix}_map.npy", actual_map)
        np.save(maps_dir / f"predicted_{prefix}_map.npy", pred_map)
        np.save(maps_dir / f"probability_{prefix}_map.npy", prob_map)
        np.save(maps_dir / f"top_count_predicted_{prefix}_map.npy", topk_map)
        np.save(maps_dir / f"error_{prefix}_map.npy", error_map)
        np.save(maps_dir / f"split_{prefix}_map.npy", split_map)
        figure_path = maps_dir / f"prediction_vs_actual_{prefix}.png"
        save_prediction_comparison_figure(
            figure_path,
            actual_map,
            prob_map,
            pred_map,
            topk_map,
            error_map,
            title=f"{target}: predicted map vs real map",
            class_values=class_values,
        )
        return [figure_path]

    output_paths: list[Path] = []
    for transition_id in np.unique(transition_index):
        mask = transition_index == transition_id
        if not np.any(mask):
            continue
        name = (
            transition_names[int(transition_id)]
            if transition_names is not None and 0 <= int(transition_id) < len(transition_names)
            else f"transition {int(transition_id)}"
        )
        safe = f"{int(transition_id):02d}_{safe_filename(name)}"
        actual_map, pred_map, prob_map, topk_map, error_map, split_map = maps_for_rows(
            coords[mask],
            map_shape,
            y_true[mask],
            y_pred[mask],
            probabilities[mask],
            class_values,
            split[mask],
        )
        np.save(maps_dir / f"actual_{target}_{safe}.npy", actual_map)
        np.save(maps_dir / f"predicted_{target}_{safe}.npy", pred_map)
        np.save(maps_dir / f"probability_{target}_{safe}.npy", prob_map)
        np.save(maps_dir / f"top_count_predicted_{target}_{safe}.npy", topk_map)
        np.save(maps_dir / f"error_{target}_{safe}.npy", error_map)
        np.save(maps_dir / f"split_{target}_{safe}.npy", split_map)
        figure_path = maps_dir / f"prediction_vs_actual_{target}_{safe}.png"
        save_prediction_comparison_figure(
            figure_path,
            actual_map,
            prob_map,
            pred_map,
            topk_map,
            error_map,
            title=f"{target}: {name}",
            class_values=class_values,
        )
        output_paths.append(figure_path)
    return output_paths


def maps_for_rows(
    coords: np.ndarray,
    map_shape: tuple[int, int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_values: np.ndarray,
    split: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actual_map = np.full(map_shape, np.nan, dtype=np.float32)
    pred_map = np.full(map_shape, np.nan, dtype=np.float32)
    prob_map = np.full(map_shape, np.nan, dtype=np.float32)
    topk_map = np.full(map_shape, np.nan, dtype=np.float32)
    error_map = np.full(map_shape, np.nan, dtype=np.float32)
    split_map = np.full(map_shape, np.nan, dtype=np.float32)
    if class_values.size == 2 and np.any(class_values == 1):
        display_probability = probabilities[:, int(np.where(class_values == 1)[0][0])]
        positive_count = int(np.count_nonzero(y_true == 1))
        topk = np.zeros_like(display_probability, dtype=np.float32)
        if positive_count > 0:
            top_indices = np.argsort(-display_probability)[:positive_count]
            topk[top_indices] = 1.0
    else:
        display_probability = np.nanmax(probabilities, axis=1)
        topk = np.full_like(display_probability, np.nan, dtype=np.float32)
    split_codes = {"train": 0.0, "val": 1.0, "test": 2.0}
    for row_index, (x_index, y_index) in enumerate(coords):
        actual_map[int(x_index), int(y_index)] = float(y_true[row_index])
        pred_map[int(x_index), int(y_index)] = float(y_pred[row_index])
        prob_map[int(x_index), int(y_index)] = float(display_probability[row_index])
        topk_map[int(x_index), int(y_index)] = float(topk[row_index])
        error_map[int(x_index), int(y_index)] = float(y_true[row_index] != y_pred[row_index])
        split_map[int(x_index), int(y_index)] = split_codes.get(str(split[row_index]), np.nan)
    return actual_map, pred_map, prob_map, topk_map, error_map, split_map


def save_prediction_comparison_figure(
    path: Path,
    actual_map: np.ndarray,
    probability_map: np.ndarray,
    predicted_map: np.ndarray,
    topk_map: np.ndarray,
    error_map: np.ndarray,
    title: str,
    class_values: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.8), constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    is_binary = class_values.size == 2 and np.any(class_values == 1)
    panels = [
        ("Real map", actual_map, "viridis", None, None),
        ("Predicted probability" if is_binary else "Prediction confidence", probability_map, "magma", 0.0, 1.0),
        ("Predicted label", predicted_map, "viridis", None, None),
        ("Top-N probability pixels", topk_map, "viridis", 0.0, 1.0),
        ("Incorrect pixels", error_map, "Reds", 0.0, 1.0),
    ]
    for axis, (panel_title, values, cmap, vmin, vmax) in zip(axes, panels):
        image = axis.imshow(values.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        axis.set_title(panel_title, fontsize=10)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def safe_filename(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in name)


def write_prediction_rows(
    path: Path,
    coords: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_values: np.ndarray,
    split: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["x", "y", "split", "target", "prediction"] + [f"prob_class_{int(v)}" for v in class_values]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (x, y) in enumerate(coords):
            row: dict[str, Any] = {
                "x": int(x),
                "y": int(y),
                "split": str(split[index]),
                "target": int(y_true[index]),
                "prediction": int(y_pred[index]),
            }
            for class_index, class_value in enumerate(class_values):
                row[f"prob_class_{int(class_value)}"] = float(probabilities[index, class_index])
            writer.writerow(row)


def write_feature_importance(path: Path, feature_names: list[str], weights: np.ndarray) -> None:
    importance = np.linalg.norm(weights, axis=1)
    order = np.argsort(-importance)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "feature", "importance"])
        writer.writeheader()
        for rank, index in enumerate(order, start=1):
            writer.writerow({"rank": rank, "feature": feature_names[int(index)], "importance": float(importance[int(index)])})


def config_for_json(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    data["dataset"] = str(config.dataset.expanduser().resolve())
    data["output_dir"] = str(config.output_dir.expanduser().resolve())
    return data


def training_report(metrics: dict[str, Any]) -> str:
    test = metrics["test"]
    lines = [
        "# Switching Predictor Training Report",
        "",
        f"Target: `{metrics['config']['target']}`",
        f"Feature set: `{metrics['config']['feature_set']}`",
        "",
        "## Test Metrics",
        "",
        f"- Accuracy: {test.get('accuracy', float('nan')):.3f}",
        f"- Balanced accuracy: {test.get('balanced_accuracy', float('nan')):.3f}",
        f"- Macro F1: {test.get('macro_f1', float('nan')):.3f}",
    ]
    if "auroc" in test:
        lines.append(f"- AUROC: {test.get('auroc', float('nan')):.3f}")
        lines.append(f"- Average precision: {test.get('average_precision', float('nan')):.3f}")
    lines.extend(
        [
            "",
            "The split is spatial-blocked by default so neighboring pixels are not freely mixed between train and test.",
            "Use `--feature-set spectral`, `--feature-set spatial`, and `--feature-set spectral_spatial` to run ablations.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a leakage-aware initial-state switching predictor from an exported ML dataset.")
    parser.add_argument("--dataset", default="outputs/ml_switching_dataset/switching_ml_dataset.npz")
    parser.add_argument("--output-dir", default="outputs/ml_switching_model")
    parser.add_argument("--target", default="future_active", choices=TARGET_CHOICES)
    parser.add_argument("--feature-set", default="all", choices=("all", "spectral", "spatial", "artifact", "spectral_spatial", "no_position"))
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--spatial-block-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--shuffle-labels", action="store_true", help="Train on shuffled labels as a permutation/null control.")
    args = parser.parse_args(argv)

    config = TrainConfig(
        dataset=Path(args.dataset),
        output_dir=Path(args.output_dir),
        target=args.target,
        feature_set=args.feature_set,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        spatial_block_size=args.spatial_block_size,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        shuffle_labels=args.shuffle_labels,
    )
    print(f"Training switching predictor for target={config.target!r}, feature_set={config.feature_set!r}")
    paths = train_from_dataset(config)
    print("Wrote training outputs:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    test = metrics["test"]
    print(
        "Test metrics: "
        f"accuracy={test.get('accuracy', float('nan')):.3f}, "
        f"balanced_accuracy={test.get('balanced_accuracy', float('nan')):.3f}, "
        f"macro_f1={test.get('macro_f1', float('nan')):.3f}"
    )
    if "auroc" in test:
        print(f"Binary metrics: AUROC={test['auroc']:.3f}, AP={test['average_precision']:.3f}")


TARGET_CHOICES = (
    "future_metallic",
    "future_erased",
    "future_active",
    "both_metallic_erased",
    "repeated_switching",
    "stable_control",
    "never_switched",
    "first_switch_transition",
    "outcome_class",
    "transition_metallic",
    "transition_erased",
    "transition_active",
    "transition_stable",
    "transition_outcome_class",
)


if __name__ == "__main__":
    main()
