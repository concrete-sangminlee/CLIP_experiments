import csv
import json
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "SDNET2025"
TABLE_DIR = ROOT / "paper" / "tables"
FIGURE_DIR = ROOT / "paper" / "figures"
FEATURE_PATH = ROOT / "clip_features.npy"
LABEL_PATH = ROOT / "clip_labels.npy"
CLASS_NAMES_PATH = ROOT / "clip_class_names.npy"


def ensure_dirs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def collect_dataset_stats():
    class_dirs = {
        "loosened": "Dataset/Defected/Annotated Loosen bolt & nuts/Resized images 640-640",
        "missing": "Dataset/Defected/Annotated Missing bolt & nuts/Resized- 640-640",
        "fixed": "Dataset/Fixed/640-640",
    }
    counts = {}
    total = 0
    for name, rel in class_dirs.items():
        dir_path = DATA_ROOT / rel
        count = sum(1 for p in dir_path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        counts[name] = count
        total += count
    for name in counts:
        counts[name] = {
            "count": counts[name],
            "ratio": counts[name] / total if total else 0.0,
            "description": {
                "loosened": "Loose bolt or nut requiring tightening",
                "missing": "Completely missing bolt or nut",
                "fixed": "Properly fastened bolt without defect",
            }[name],
        }
    return counts


def write_dataset_table(counts):
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{SDNET2025 Class Distribution (Resized 640$\\times$640 subset).}",
        "\\label{tab:dataset}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Class & Description & Images & Share (\\%) \\\\",
        "\\midrule",
    ]
    for name, stats in counts.items():
        lines.append(
            f"{name.title()} & {stats['description']} & {stats['count']} & {stats['ratio']*100:.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLE_DIR / "dataset_overview.tex").write_text("\n".join(lines), encoding="utf-8")


def stage_progression():
    stages = [
        ("Zero-shot CLIP", "ViT-B/32 + domain prompts", 51.70),
        ("Baseline Probe", "Linear classifier, 100 shots", 51.70),
        ("Improved Probe", "MLP + mixup + class weights", 63.30),
        ("Prompt + Backbone", "Prompt engineering + ViT-L/14", 67.55),
        ("Grid Search", "Best LR/WD/dropout combo", 69.68),
    ]
    return stages


def write_progress_table(stages):
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Progressive accuracy improvements on SDNET2025.}",
        "\\label{tab:progress}",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Stage & Key Technique & Test Acc (\\%) & $\\Delta$ (pp) \\\\",
        "\\midrule",
    ]
    prev = None
    for stage, desc, acc in stages:
        delta = acc - prev if prev is not None else 0.0
        lines.append(f"{stage} & {desc} & {acc:.2f} & {delta:+.2f} \\\\")
        prev = acc
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLE_DIR / "performance_progression.tex").write_text("\n".join(lines), encoding="utf-8")


def parse_grid_results():
    pattern = re.compile(
        r"(?P<lr>[0-9.e+-]+)\s+(?P<wd>[0-9.e+-]+)\s+(?P<dropout>[0-9.]+)\s+(?P<acc>[0-9.]+)"
    )
    entries = []
    for line in (ROOT / "grid_search_results.txt").read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            entries.append(
                {
                    "lr": float(match.group("lr")),
                    "wd": float(match.group("wd")),
                    "dropout": float(match.group("dropout")),
                    "acc": float(match.group("acc")),
                }
            )
    return entries


def write_grid_table(entries, top_k=5):
    ordered = sorted(entries, key=lambda x: x["acc"], reverse=True)[:top_k]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Top hyperparameter combinations from the 27-run grid search.}",
        "\\label{tab:grid}",
        "\\begin{tabular}{ccccc}",
        "\\toprule",
        "Rank & LR & Weight Decay & Dropout & Test Acc (\\%) \\\\",
        "\\midrule",
    ]
    for idx, entry in enumerate(ordered, start=1):
        lines.append(
            f"{idx} & {entry['lr']:.0e} & {entry['wd']:.0e} & {entry['dropout']:.1f} & {entry['acc']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLE_DIR / "grid_search_top.tex").write_text("\n".join(lines), encoding="utf-8")
    return ordered


def plot_progression(stages):
    accs = [acc for _, _, acc in stages]
    labels = [stage for stage, _, _ in stages]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(len(accs)), accs, marker="o", linewidth=2, color="#1b6ca8")
    for idx, acc in enumerate(accs):
        ax.text(idx, acc + 0.6, f"{acc:.2f}%", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(48, 72)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("Step-wise Accuracy Improvements")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "performance_progression.png", dpi=300)
    plt.close(fig)


def plot_grid_heatmap(entries):
    lrs = sorted({entry["lr"] for entry in entries})
    wds = sorted({entry["wd"] for entry in entries})
    dropouts = sorted({entry["dropout"] for entry in entries})
    vmin = min(entry["acc"] for entry in entries)
    vmax = max(entry["acc"] for entry in entries)
    fig, axes = plt.subplots(1, len(dropouts), figsize=(13, 4.2), sharey=True)
    if len(dropouts) == 1:
        axes = [axes]
    for ax, dropout in zip(axes, dropouts):
        grid = np.full((len(wds), len(lrs)), np.nan)
        for entry in entries:
            if abs(entry["dropout"] - dropout) < 1e-6:
                i = wds.index(entry["wd"])
                j = lrs.index(entry["lr"])
                grid[i, j] = entry["acc"]
        im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="viridis")
        for i in range(len(wds)):
            for j in range(len(lrs)):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white", fontsize=8)
        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f"{lr:.0e}" for lr in lrs], rotation=45, ha='right')
        ax.set_yticks(range(len(wds)))
        ax.set_yticklabels([f"{wd:.0e}" for wd in wds])
        ax.set_xlabel("Learning Rate", fontsize=10)
        ax.set_title(f"Dropout = {dropout:.1f}", fontsize=10, pad=8)
    axes[0].set_ylabel("Weight Decay", fontsize=10)
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.08, label="Accuracy (%)")
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle("Grid Search Accuracy Landscape", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0.02, 0.98, 0.96])
    fig.savefig(FIGURE_DIR / "grid_search_heatmap.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def load_feature_bank():
    features = np.load(FEATURE_PATH)
    labels = np.load(LABEL_PATH)
    class_names = np.load(CLASS_NAMES_PATH)
    return features, labels, class_names


def plot_tsne_projection(features, labels, class_names, max_points=2000, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    if len(indices) > max_points:
        indices = indices[:max_points]
    subset_feat = features[indices]
    subset_lbl = labels[indices]

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    embedding = tsne.fit_transform(subset_feat)

    unique_labels = sorted(np.unique(labels))
    colors = ["#1b6ca8", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, cls in enumerate(unique_labels):
        mask = subset_lbl == cls
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=16,
            color=colors[idx % len(colors)],
            label=str(class_names[cls]),
            alpha=0.7,
            edgecolors="none",
        )
    ax.set_title("t-SNE Projection of CLIP Features")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(markerscale=2, loc="best", frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "tsne_projection.png", dpi=300)
    plt.close(fig)


def train_confusion_classifier(features, labels, class_names, n_train_per_class=150, seed=42):
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for cls in range(len(class_names)):
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        if len(cls_indices) <= 1:
            train_idx.extend(cls_indices)
            continue
        n_train = min(n_train_per_class, len(cls_indices) - 1)
        n_train = max(n_train, 1)
        train_idx.extend(cls_indices[:n_train])
        test_idx.extend(cls_indices[n_train:])
    train_idx = np.array(train_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)
    X_train = features[train_idx]
    y_train = labels[train_idx]
    X_test = features[test_idx]
    y_test = labels[test_idx]

    clf = LogisticRegression(
        max_iter=5000,
        solver="saga",
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    cm = confusion_matrix(y_test, preds, labels=range(len(class_names)))
    report = classification_report(
        y_test,
        preds,
        target_names=[str(name) for name in class_names],
        digits=4,
    )
    (TABLE_DIR / "confusion_report.txt").write_text(report, encoding="utf-8")
    return cm, [str(name) for name in class_names]


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Balanced Logistic Probe)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=300)
    plt.close(fig)


def csv_to_latex_table(csv_path: Path, output_path: Path, caption: str, label: str):
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]
    if not rows:
        return
    header = rows[0]
    body = rows[1:]
    col_format = "".join(["l" for _ in header])

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{tabular}{{{col_format}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for row in body:
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def maybe_generate_csv_tables():
    csv_configs = [
        (ROOT / "ensemble_results.csv", TABLE_DIR / "ensemble_results.tex", "Ensemble experiments summary.", "tab:ensemble"),
        (ROOT / "lora_results.csv", TABLE_DIR / "lora_results.tex", "LoRA fine-tuning experiments.", "tab:lora"),
    ]
    for csv_path, out_path, caption, label in csv_configs:
        if csv_path.exists():
            csv_to_latex_table(csv_path, out_path, caption, label)


def main():
    ensure_dirs()
    dataset_stats = collect_dataset_stats()
    write_dataset_table(dataset_stats)

    stages = stage_progression()
    write_progress_table(stages)

    entries = parse_grid_results()
    write_grid_table(entries)

    plot_progression(stages)
    plot_grid_heatmap(entries)

    features = None
    labels = None
    class_names = None
    if FEATURE_PATH.exists() and LABEL_PATH.exists() and CLASS_NAMES_PATH.exists():
        features, labels, class_names = load_feature_bank()
        plot_tsne_projection(features, labels, class_names)
        cm, class_labels = train_confusion_classifier(features, labels, class_names)
        plot_confusion_matrix(cm, class_labels)

    maybe_generate_csv_tables()


if __name__ == "__main__":
    main()

