"""Utility modules for bolt classification experiments.

This module centralizes reusable helpers for training probes,
self-training loops, and LoRA fine-tuning scripts.  All functions are
pure-Python and torch-based so they can be imported without introducing
heavy framework dependencies.
"""
from __future__ import annotations

"""Shared training helpers for CLIP bolt classification experiments."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TrainTestSplit:
    """Container describing an index-based train/test split."""

    train_indices: np.ndarray
    test_indices: np.ndarray


def set_seed(seed: int = 42) -> None:
    """Set all RNG seeds for reproducibility."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion: nn.Module, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_features(x: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, float]:
    """CutMix-style interpolation for feature vectors.

    Instead of patch swapping (which is not defined in feature space), we
    randomly drop a contiguous span of feature dimensions and replace it
    with another sample.  This approximates CutMix regularisation while
    staying in the latent space.
    """

    if alpha <= 0:
        return x, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size, feat_dim = x.size()
    perm = torch.randperm(batch_size, device=x.device)
    span = int(feat_dim * (1 - lam))
    start = np.random.randint(0, feat_dim - span + 1)
    mixed = x.clone()
    mixed[:, start:start + span] = x[perm, start:start + span]
    lam_adjusted = 1 - span / feat_dim
    return mixed, lam_adjusted


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        num_classes = inputs.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        losses = torch.sum(-true_dist * log_probs, dim=1)
        if self.reduction == "none":
            return losses
        if self.reduction == "sum":
            return torch.sum(losses)
        return torch.mean(losses)


class DeepMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: Sequence[int], dropout_rate: float = 0.3):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


def build_train_test_split(labels: np.ndarray, class_names: Sequence[str], n_train_per_class: int, seed: int = 42) -> TrainTestSplit:
    set_seed(seed)
    train_indices: List[int] = []
    test_indices: List[int] = []

    for cls_idx, _ in enumerate(class_names):
        cls_idxs = np.where(labels == cls_idx)[0]
        np.random.shuffle(cls_idxs)
        n_train = min(n_train_per_class, len(cls_idxs))
        train_indices.extend(cls_idxs[:n_train].tolist())
        test_indices.extend(cls_idxs[n_train:].tolist())

    return TrainTestSplit(
        train_indices=np.array(train_indices, dtype=np.int64),
        test_indices=np.array(test_indices, dtype=np.int64),
    )


def compute_class_weights(labels: np.ndarray, class_names: Sequence[str], device: torch.device, missing_class_boost: float = 1.5) -> torch.Tensor:
    class_counts = np.bincount(labels, minlength=len(class_names))
    total = class_counts.sum()
    weights = total / (len(class_names) * class_counts)
    missing_idx = None
    for i, name in enumerate(class_names):
        if str(name).lower() == "missing":
            missing_idx = i
            break
    if missing_idx is not None:
        weights[missing_idx] *= missing_class_boost
    weights = weights / weights.min()
    return torch.from_numpy(weights).float().to(device)


def softmax_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).float().sum().item()
    return correct / targets.size(0)
