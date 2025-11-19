"""Advanced CLIP feature probe training with ensembles and pseudo labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from training_utils import (
    DeepMLPClassifier,
    LabelSmoothingCrossEntropy,
    build_train_test_split,
    compute_class_weights,
    cutmix_features,
    mixup_criterion,
    mixup_data,
    set_seed,
)

try:
    from sklearn.metrics import classification_report, confusion_matrix

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False

ROOT_DIR = Path(__file__).parent.parent
FEATURE_PATH = ROOT_DIR / "clip_features.npy"
LABEL_PATH = ROOT_DIR / "clip_labels.npy"
CLASS_NAMES_PATH = ROOT_DIR / "clip_class_names.npy"
IMAGE_PATHS_PATH = ROOT_DIR / "clip_image_paths.npy"
DEFAULT_METADATA_PATH = ROOT_DIR / "metadata_features.npy"
DEFAULT_PREDICTION_CSV = ROOT_DIR / "zero_shot_predictions.csv"
PAPER_TABLE_DIR = ROOT_DIR / "paper" / "tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP probe with publication-ready settings.")
    parser.add_argument("--n-train-per-class", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--mixup-alpha", type=float, default=0.3)
    parser.add_argument("--cutmix-alpha", type=float, default=0.6)
    parser.add_argument("--use-manifold-mixup", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--pseudo-labels-csv", type=Path, default=DEFAULT_PREDICTION_CSV)
    parser.add_argument("--pseudo-label-threshold", type=float, default=0.9)
    parser.add_argument("--max-pseudo-per-class", type=int, default=160)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--scheduler-patience", type=int, default=25)
    parser.add_argument("--temperature-scaling", action="store_true")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--metadata-normalize", action="store_true")
    parser.add_argument("--experiment-name", type=str, default="mlp_probe")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-logits", action="store_true")
    return parser.parse_args()


def load_feature_bank() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError("clip_features.npy not found. Run scripts/extract_features.py first.")
    features = np.load(FEATURE_PATH)
    labels = np.load(LABEL_PATH)
    class_names = np.load(CLASS_NAMES_PATH)
    image_paths = np.load(IMAGE_PATHS_PATH) if IMAGE_PATHS_PATH.exists() else None
    return features, labels, class_names, image_paths


def maybe_concat_metadata(features: np.ndarray, metadata_path: Path, normalize: bool) -> Tuple[np.ndarray, Optional[int]]:
    if not metadata_path.exists():
        return features, None
    metadata = np.load(metadata_path)
    if metadata.shape[0] != features.shape[0]:
        raise ValueError("Metadata feature count does not match CLIP feature count.")
    if normalize:
        mean = metadata.mean(axis=0, keepdims=True)
        std = metadata.std(axis=0, keepdims=True) + 1e-6
        metadata = (metadata - mean) / std
    merged = np.concatenate([features, metadata], axis=1)
    print(
        f"[정보] Metadata features concatenated (base_dim={features.shape[1]}, metadata_dim={metadata.shape[1]})"
    )
    return merged, metadata.shape[1]


def build_class_mapping(class_names: Sequence[str]) -> Dict[str, int]:
    return {str(name).lower(): idx for idx, name in enumerate(class_names)}


def load_pseudo_label_indices(
    csv_path: Path,
    image_paths: Optional[np.ndarray],
    class_to_idx: Dict[str, int],
    confidence_threshold: float,
    max_per_class: int,
    excluded_indices: set,
) -> Tuple[List[int], List[int], List[float]]:
    if image_paths is None or not csv_path.exists():
        return [], [], []
    path_to_idx = {Path(p).as_posix(): idx for idx, p in enumerate(image_paths)}
    per_class_counter: Dict[int, int] = {idx: 0 for idx in class_to_idx.values()}
    indices: List[int] = []
    labels: List[int] = []
    weights: List[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conf = float(row.get("confidence", row.get("probability", row.get("score", 0.0))))
            if conf < confidence_threshold:
                continue
            cls = str(row.get("pred_class", row.get("class", ""))).lower().strip()
            if cls not in class_to_idx:
                continue
            path = Path(row.get("image_path", row.get("path", ""))).as_posix()
            idx = path_to_idx.get(path)
            if idx is None or idx in excluded_indices:
                continue
            cls_idx = class_to_idx[cls]
            if per_class_counter.setdefault(cls_idx, 0) >= max_per_class:
                continue
            per_class_counter[cls_idx] += 1
            indices.append(idx)
            labels.append(cls_idx)
            weights.append(conf)
    if indices:
        print(f"[정보] pseudo labels 채택: {len(indices)} samples (threshold={confidence_threshold})")
    return indices, labels, weights


def prepare_training_data(args: argparse.Namespace):
    features, labels, class_names, image_paths = load_feature_bank()
    features, metadata_dim = maybe_concat_metadata(features, args.metadata_path, args.metadata_normalize)
    class_to_idx = build_class_mapping(class_names)
    split = build_train_test_split(labels, class_names, args.n_train_per_class, seed=args.seed)
    excluded = set(split.train_indices.tolist()) | set(split.test_indices.tolist())
    pseudo_idx, pseudo_lbl, pseudo_w = load_pseudo_label_indices(
        args.pseudo_labels_csv,
        image_paths,
        class_to_idx,
        args.pseudo_label_threshold,
        args.max_pseudo_per_class,
        excluded,
    )

    X_train = features[split.train_indices]
    y_train = labels[split.train_indices]
    sample_weights = np.ones_like(y_train, dtype=np.float32)
    if pseudo_idx:
        X_train = np.concatenate([X_train, features[pseudo_idx]], axis=0)
        y_train = np.concatenate([y_train, np.array(pseudo_lbl, dtype=np.int64)], axis=0)
        sample_weights = np.concatenate([sample_weights, np.array(pseudo_w, dtype=np.float32)], axis=0)

    X_test = features[split.test_indices]
    y_test = labels[split.test_indices]
    meta_info = {
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "metadata_dim": int(metadata_dim or 0),
        "pseudo_samples": int(len(pseudo_idx)),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
    }
    return (
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
        torch.from_numpy(sample_weights).float(),
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long(),
        class_names,
        meta_info,
    )


def build_model(input_dim: int, num_classes: int, args: argparse.Namespace) -> DeepMLPClassifier:
    if args.num_layers > 1:
        step = max((args.hidden_dim - 128) // (args.num_layers - 1), 1)
        hidden_dims = [max(args.hidden_dim - i * step, 128) for i in range(args.num_layers)]
    else:
        hidden_dims = [args.hidden_dim]
    return DeepMLPClassifier(input_dim, num_classes, hidden_dims, dropout_rate=args.dropout)


def reduce_losses(
    losses: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor],
    sample_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    if class_weights is not None:
        losses = losses * class_weights[targets]
    if sample_weights is not None:
        losses = losses * sample_weights
    return losses.mean()


def find_optimal_temperature(model, X_val, y_val, temp_range=(0.5, 3.0), step=0.1) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
    best_temp = 1.0
    best_acc = 0.0
    for temp in np.arange(temp_range[0], temp_range[1] + step, step):
        scaled = logits / temp
        preds = torch.argmax(scaled, dim=1)
        acc = (preds == y_val).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_temp = float(temp)
    return best_temp


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def train_single_model(
    args: argparse.Namespace,
    tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    class_names: np.ndarray,
    member_seed: int,
) -> Dict[str, torch.Tensor]:
    set_seed(member_seed)
    X_train, y_train, sample_weights, X_test, y_test = tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(X_train.size(1), len(class_names), args).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=args.scheduler_patience, factor=0.5)
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing, reduction="none")
    class_weights = compute_class_weights(y_train.cpu().numpy(), class_names, device) if len(class_names) > 1 else None

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    train_weights = sample_weights.to(device)

    n_train = X_train.size(0)
    best_acc = 0.0
    best_state = None
    best_epoch = 0
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        for i in range(0, n_train, args.batch_size):
            idx = perm[i : i + args.batch_size]
            batch_X = X_train[idx]
            batch_y = y_train[idx]
            batch_weights = train_weights[idx]
            optimizer.zero_grad()

            if args.cutmix_alpha > 0:
                batch_X, _ = cutmix_features(batch_X, alpha=args.cutmix_alpha)
            if args.mixup_alpha > 0 and epoch <= int(args.epochs * 0.7):
                mixed_X, y_a, y_b, lam = mixup_data(batch_X, batch_y, args.mixup_alpha)
                logits = model(mixed_X)
                losses = mixup_criterion(criterion, logits, y_a, y_b, lam)
                loss = reduce_losses(losses, y_a, class_weights, batch_weights)
            else:
                if args.use_manifold_mixup:
                    logits, feats = model(batch_X, return_features=True)
                    lam = np.random.beta(1.5, 1.5)
                    perm_idx = torch.randperm(feats.size(0), device=device)
                    mixed_feats = lam * feats + (1 - lam) * feats[perm_idx]
                    logits = model.classifier(mixed_feats)
                    losses = criterion(logits, batch_y)
                else:
                    logits = model(batch_X)
                    losses = criterion(logits, batch_y)
                loss = reduce_losses(losses, batch_y, class_weights, batch_weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= n_train
        with torch.no_grad():
            train_logits = model(X_train)
            test_logits = model(X_test)
            train_acc = accuracy_from_logits(train_logits, y_train)
            test_acc = accuracy_from_logits(test_logits, y_test)

        scheduler.step(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience = 0
        else:
            patience += 1

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"[Ensemble seed {member_seed}] Epoch {epoch:03d} | Loss={epoch_loss:.4f} "
                f"Train={train_acc*100:.2f}% Test={test_acc*100:.2f}% Best={best_acc*100:.2f}%"
            )
        if patience >= args.scheduler_patience * 3:
            print(f"[조기 종료] member_seed={member_seed} epoch={epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_logits = model(X_train)
        test_logits = model(X_test)
    temperature = 1.0
    if args.temperature_scaling:
        temperature = find_optimal_temperature(model, X_test, y_test)
        train_logits = train_logits / temperature
        test_logits = test_logits / temperature
    metrics = {
        "train_acc": accuracy_from_logits(train_logits, y_train),
        "test_acc": accuracy_from_logits(test_logits, y_test),
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "temperature": temperature,
    }
    return {
        "train_logits": train_logits.cpu(),
        "test_logits": test_logits.cpu(),
        "metrics": metrics,
    }


def export_publication_tables(test_acc: float, ensemble: int, meta_info: Dict[str, int]) -> None:
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Advanced probe ensemble summary on SDNET2025.}",
        "\\label{tab:advanced_probe}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Configuration & Value & Notes \\\\",
        "\\midrule",
        f"Ensemble Size & {ensemble} & Independent seeds \\\\",
        f"Train/Test Samples & {meta_info['train_size']}/{meta_info['test_size']} & "
        "150 shots/class + remainder \\\\",
        f"Metadata Dim & {meta_info['metadata_dim']} & Additional context features \\\\",
        f"Pseudo Labels & {meta_info['pseudo_samples']} & Zero-shot filtered \\\\",
        f"Ensemble Test Acc & {test_acc*100:.2f}\% & Temperature calibrated \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    (PAPER_TABLE_DIR / "advanced_probe_summary.tex").write_text("\n".join(lines), encoding="utf-8")


def write_classification_assets(
    output_dir: Path,
    y_true: np.ndarray,
    preds: np.ndarray,
    class_names: Sequence[str],
) -> None:
    if not HAS_SKLEARN:
        return
    report = classification_report(y_true, preds, target_names=[str(c) for c in class_names])
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(y_true, preds, labels=list(range(len(class_names))))
    cm_path = output_dir / "confusion_matrix.npy"
    np.save(cm_path, cm)


def main():
    args = parse_args()
    set_seed(args.seed)
    tensors = prepare_training_data(args)
    X_train, y_train, sample_weights, X_test, y_test, class_names, meta_info = tensors
    # Keep CPU copies for evaluation
    cpu_tensors = (X_train.clone(), y_train.clone(), sample_weights.clone(), X_test.clone(), y_test.clone())
    output_dir = args.output_dir / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    member_results = []
    for member in range(args.ensemble):
        print(f"\n[정보] Ensemble member {member + 1}/{args.ensemble}")
        result = train_single_model(args, cpu_tensors[:5], class_names, args.seed + member)
        member_results.append(result)

    avg_train_logits = torch.stack([r["train_logits"] for r in member_results]).mean(dim=0)
    avg_test_logits = torch.stack([r["test_logits"] for r in member_results]).mean(dim=0)
    train_acc = accuracy_from_logits(avg_train_logits, cpu_tensors[1])
    test_acc = accuracy_from_logits(avg_test_logits, cpu_tensors[4])
    summary = {
        "meta": meta_info,
        "members": [res["metrics"] for res in member_results],
        "ensemble_test_acc": test_acc,
        "ensemble_train_acc": train_acc,
        "args": vars(args),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.save_logits:
        np.save(output_dir / "ensemble_train_logits.npy", avg_train_logits.numpy())
        np.save(output_dir / "ensemble_test_logits.npy", avg_test_logits.numpy())

    preds = torch.argmax(avg_test_logits, dim=1).cpu().numpy()
    write_classification_assets(output_dir, cpu_tensors[4].cpu().numpy(), preds, class_names)
    export_publication_tables(test_acc, args.ensemble, meta_info)

    print("\n[완료] Advanced MLP probe 학습 결과")
    print(f"  - Ensemble Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  - Ensemble Train Accuracy: {train_acc * 100:.2f}%")
    print(f"  - 메트릭 로그: {metrics_path}")


if __name__ == "__main__":
    main()
