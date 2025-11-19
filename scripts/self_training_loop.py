"""Self-training loop combining CLIP features with pseudo labels."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from training_utils import DeepMLPClassifier, build_train_test_split, set_seed

ROOT = Path(__file__).parent.parent
FEATURE_PATH = ROOT / "clip_features.npy"
LABEL_PATH = ROOT / "clip_labels.npy"
CLASS_PATH = ROOT / "clip_class_names.npy"
LOG_PATH = ROOT / "experiments" / "self_training_history.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative self-training on CLIP features.")
    parser.add_argument("--labeled-per-class", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--max-add-per-iter", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_feature_bank() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.load(FEATURE_PATH), np.load(LABEL_PATH), np.load(CLASS_PATH)


def train_probe(X: torch.Tensor, y: torch.Tensor, hidden_dim: int, epochs: int, num_classes: int) -> DeepMLPClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepMLPClassifier(X.size(1), num_classes, [hidden_dim], dropout_rate=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    X = X.to(device)
    y = y.to(device)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    return model.cpu()


def self_training_loop(args):
    features, labels, class_names = load_feature_bank()
    split = build_train_test_split(labels, class_names, args.labeled_per_class, seed=args.seed)
    labeled = set(split.train_indices.tolist())
    unlabeled = set(split.test_indices.tolist())
    history: List[Dict[str, float]] = []
    current_indices = np.array(sorted(labeled), dtype=np.int64)
    current_labels = labels[current_indices]

    for iteration in range(1, args.iterations + 1):
        print(f"\n[Self-training] Iteration {iteration}/{args.iterations}")
        X = torch.from_numpy(features[current_indices]).float()
        y = torch.from_numpy(current_labels).long()
        model = train_probe(X, y, args.hidden_dim, args.epochs, num_classes=len(class_names))
        with torch.no_grad():
            logits = model(torch.from_numpy(features).float())
            probs = torch.softmax(logits, dim=1).numpy()
        candidate_indices = np.array(sorted(unlabeled))
        candidate_probs = probs[candidate_indices]
        best_class = candidate_probs.argmax(axis=1)
        best_conf = candidate_probs.max(axis=1)
        mask = best_conf >= args.confidence
        selected = candidate_indices[mask][: args.max_add_per_iter]
        selected_labels = best_class[mask][: args.max_add_per_iter]
        current_indices = np.concatenate([current_indices, selected])
        current_labels = np.concatenate([current_labels, selected_labels])
        unlabeled -= set(selected.tolist())
        entry = {
            "iteration": iteration,
            "labeled": len(current_indices),
            "added": len(selected),
            "remaining": len(unlabeled),
        }
        print(f"  - added {len(selected)} pseudo labels (confidence>={args.confidence})")
        history.append(entry)

    if history:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["iteration", "labeled", "added", "remaining"])
            writer.writeheader()
            writer.writerows(history)
        print(f"[로그] Self-training history saved to {LOG_PATH}")


def main():
    args = parse_args()
    set_seed(args.seed)
    self_training_loop(args)


if __name__ == "__main__":
    main()
