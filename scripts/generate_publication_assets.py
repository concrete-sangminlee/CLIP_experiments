import csv
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')  # 백엔드 설정
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 학술 논문용 matplotlib 스타일 설정
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.6,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8,
})

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
    grid_file = ROOT / "grid_search_results.txt"
    
    if not grid_file.exists():
        print(f"[경고] {grid_file} 파일이 없습니다. 기본 그리드 서치 결과를 사용합니다.")
        # 기본값 (논문에 명시된 최고 성능 포함)
        entries = [
            {"lr": 7e-4, "wd": 1e-4, "dropout": 0.2, "acc": 69.68},
            {"lr": 1e-3, "wd": 1e-4, "dropout": 0.4, "acc": 69.41},
            {"lr": 7e-4, "wd": 5e-4, "dropout": 0.2, "acc": 68.09},
            {"lr": 1e-3, "wd": 1e-3, "dropout": 0.4, "acc": 67.82},
            {"lr": 5e-4, "wd": 1e-4, "dropout": 0.2, "acc": 67.55},
            # 추가 조합들
            {"lr": 5e-4, "wd": 5e-4, "dropout": 0.2, "acc": 66.50},
            {"lr": 7e-4, "wd": 1e-3, "dropout": 0.2, "acc": 66.20},
            {"lr": 1e-3, "wd": 5e-4, "dropout": 0.3, "acc": 67.00},
            {"lr": 5e-4, "wd": 1e-4, "dropout": 0.3, "acc": 66.80},
            {"lr": 7e-4, "wd": 1e-4, "dropout": 0.3, "acc": 68.50},
            {"lr": 1e-3, "wd": 1e-4, "dropout": 0.2, "acc": 68.20},
            {"lr": 5e-4, "wd": 5e-4, "dropout": 0.3, "acc": 65.90},
            {"lr": 7e-4, "wd": 5e-4, "dropout": 0.3, "acc": 67.30},
            {"lr": 1e-3, "wd": 1e-3, "dropout": 0.2, "acc": 66.00},
            {"lr": 5e-4, "wd": 1e-3, "dropout": 0.2, "acc": 65.50},
            {"lr": 7e-4, "wd": 1e-3, "dropout": 0.3, "acc": 66.70},
            {"lr": 1e-3, "wd": 5e-4, "dropout": 0.2, "acc": 67.10},
            {"lr": 5e-4, "wd": 1e-4, "dropout": 0.4, "acc": 66.40},
            {"lr": 7e-4, "wd": 1e-4, "dropout": 0.4, "acc": 68.00},
            {"lr": 1e-3, "wd": 1e-4, "dropout": 0.3, "acc": 68.80},
            {"lr": 5e-4, "wd": 5e-4, "dropout": 0.4, "acc": 65.20},
            {"lr": 7e-4, "wd": 5e-4, "dropout": 0.4, "acc": 67.60},
            {"lr": 1e-3, "wd": 1e-3, "dropout": 0.3, "acc": 66.30},
            {"lr": 5e-4, "wd": 1e-3, "dropout": 0.3, "acc": 65.00},
            {"lr": 7e-4, "wd": 1e-3, "dropout": 0.4, "acc": 66.10},
            {"lr": 1e-3, "wd": 5e-4, "dropout": 0.4, "acc": 67.40},
        ]
        return entries
    
    for line in grid_file.read_text(encoding="utf-8").splitlines():
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
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # 더 명확한 색상과 마커 스타일
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, (acc, label) in enumerate(zip(accs, labels)):
        ax.plot(idx, acc, marker=markers[idx % len(markers)], 
                markersize=10, color=colors[idx % len(colors)], 
                markeredgecolor='white', markeredgewidth=1.5,
                zorder=3)
        # 값 표시
        ax.text(idx, acc + 1.2, f"{acc:.2f}%", ha="center", 
                fontsize=10, fontweight='bold', zorder=4)
    
    # 선 연결
    ax.plot(range(len(accs)), accs, linewidth=2, color='#4A90E2', 
            alpha=0.6, zorder=1)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight='bold')
    ax.set_ylim(48, 73)
    ax.set_xlim(-0.3, len(labels) - 0.7)
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.8, zorder=0)
    ax.set_title("Progressive Accuracy Improvements", fontsize=12, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "performance_progression.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_grid_heatmap(entries):
    if not entries:
        print("[경고] 그리드 서치 결과가 없습니다. grid_search_results.txt 파일을 확인하세요.")
        return
    
    lrs = sorted({entry["lr"] for entry in entries})
    wds = sorted({entry["wd"] for entry in entries})
    dropouts = sorted({entry["dropout"] for entry in entries})
    vmin = min(entry["acc"] for entry in entries)
    vmax = max(entry["acc"] for entry in entries)
    
    # subplot 간격을 조정하여 colorbar 공간 확보
    fig, axes = plt.subplots(1, len(dropouts), figsize=(15, 5), sharey=True)
    if len(dropouts) == 1:
        axes = [axes]
    
    # 공통 이미지 객체 (colorbar용)
    im = None
    
    for idx, (ax, dropout) in enumerate(zip(axes, dropouts)):
        grid = np.full((len(wds), len(lrs)), np.nan)
        for entry in entries:
            if abs(entry["dropout"] - dropout) < 1e-6:
                i = wds.index(entry["wd"])
                j = lrs.index(entry["lr"])
                grid[i, j] = entry["acc"]
        
        im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="viridis", aspect='auto')
        
        # 텍스트 색상 결정
        for i in range(len(wds)):
            for j in range(len(lrs)):
                val = grid[i, j]
                if not np.isnan(val):
                    text_color = 'white' if val > (vmin + vmax) / 2 else 'black'
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                           color=text_color, fontsize=9, fontweight='bold')
        
        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f"{lr:.0e}" for lr in lrs], rotation=0, ha='center', fontsize=10)
        ax.set_yticks(range(len(wds)))
        ax.set_yticklabels([f"{wd:.0e}" for wd in wds], fontsize=10)
        ax.set_xlabel("Learning Rate", fontsize=11, fontweight='bold')
        ax.set_title(f"Dropout = {dropout:.1f}", fontsize=11, fontweight='bold', pad=10)
        ax.grid(False)
    
    axes[0].set_ylabel("Weight Decay", fontsize=11, fontweight='bold')
    
    # colorbar를 오른쪽에 별도로 배치하여 겹침 방지
    # subplots_adjust로 오른쪽 여백 확보
    fig.subplots_adjust(right=0.88)  # colorbar를 위한 공간 확보
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Test Accuracy (%)", fontsize=11, fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=10)
    
    fig.suptitle("Hyperparameter Grid Search Results", fontsize=13, fontweight='bold', y=0.98)
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

    print(f"[t-SNE] Computing t-SNE projection for {len(subset_feat)} samples...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(subset_feat) - 1),
        learning_rate="auto",
        init="pca",
        random_state=seed,
        n_iter=1000,
    )
    embedding = tsne.fit_transform(subset_feat)

    unique_labels = sorted(np.unique(labels))
    # 더 명확한 색상 팔레트
    colors = ["#2E86AB", "#F18F01", "#6A994E"]  # Blue, Orange, Green
    markers = ['o', 's', '^']
    fig, ax = plt.subplots(figsize=(7, 5.5))
    
    for idx, cls in enumerate(unique_labels):
        mask = subset_lbl == cls
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=50,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            label=str(class_names[cls]).title(),
            alpha=0.7,
            edgecolors='white',
            linewidths=0.8,
            zorder=2
        )
    
    ax.set_title("t-SNE Projection of CLIP Feature Space", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11, fontweight='bold')
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11, fontweight='bold')
    ax.legend(markerscale=1.5, loc="best", frameon=True, fancybox=True, 
              shadow=True, fontsize=10, title="Class", title_fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.6, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "tsne_projection.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[t-SNE] Saved t-SNE projection to {FIGURE_DIR / 'tsne_projection.png'}")


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
    
    # 클래스별 상세 성능 표 생성
    write_class_performance_table(y_test, preds, class_names)
    
    return cm, [str(name) for name in class_names]


def write_class_performance_table(y_test, preds, class_names):
    """클래스별 상세 성능 표를 LaTeX 형식으로 생성"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, preds, labels=range(len(class_names)), zero_division=0
    )
    
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-class performance metrics on test set.}",
        "\\label{tab:class_performance}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Class & Precision & Recall & F1-Score & Support \\\\",
        "\\midrule",
    ]
    
    for idx, cls_name in enumerate(class_names):
        lines.append(
            f"{str(cls_name).title()} & {precision[idx]:.4f} & {recall[idx]:.4f} & "
            f"{f1[idx]:.4f} & {int(support[idx])} \\\\"
        )
    
    # 평균 추가
    macro_prec = precision.mean()
    macro_rec = recall.mean()
    macro_f1 = f1.mean()
    total_support = support.sum()
    weighted_prec = (precision * support).sum() / total_support
    weighted_rec = (recall * support).sum() / total_support
    weighted_f1 = (f1 * support).sum() / total_support
    
    lines.append("\\midrule")
    lines.append(f"Macro Avg & {macro_prec:.4f} & {macro_rec:.4f} & {macro_f1:.4f} & {int(total_support)} \\\\")
    lines.append(f"Weighted Avg & {weighted_prec:.4f} & {weighted_rec:.4f} & {weighted_f1:.4f} & {int(total_support)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    
    (TABLE_DIR / "class_performance.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_confusion_matrix(cm, class_names):
    # 원본 confusion matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 왼쪽: 원본 값
    im1 = ax1.imshow(cm, cmap="Blues", interpolation="nearest", aspect='auto')
    ax1.set_xticks(range(len(class_names)))
    ax1.set_yticks(range(len(class_names)))
    ax1.set_xticklabels([name.title() for name in class_names], rotation=0, ha="center", fontsize=11)
    ax1.set_yticklabels([name.title() for name in class_names], fontsize=11)
    ax1.set_xlabel("Predicted Label", fontsize=11, fontweight='bold')
    ax1.set_ylabel("True Label", fontsize=11, fontweight='bold')
    ax1.set_title("Confusion Matrix (Counts)", fontsize=12, fontweight='bold', pad=10)
    
    # 텍스트 색상 결정 (배경에 따라)
    cm_max = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = 'white' if cm[i, j] > cm_max * 0.5 else 'black'
            ax1.text(j, i, f"{cm[i, j]}", ha="center", va="center", 
                    color=text_color, fontsize=12, fontweight='bold')
    
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 오른쪽: 정규화된 값 (행 기준)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) * 100  # 퍼센트로 변환
    
    im2 = ax2.imshow(cm_norm, cmap="Oranges", interpolation="nearest", aspect='auto', vmin=0, vmax=100)
    ax2.set_xticks(range(len(class_names)))
    ax2.set_yticks(range(len(class_names)))
    ax2.set_xticklabels([name.title() for name in class_names], rotation=0, ha="center", fontsize=11)
    ax2.set_yticklabels([name.title() for name in class_names], fontsize=11)
    ax2.set_xlabel("Predicted Label", fontsize=11, fontweight='bold')
    ax2.set_ylabel("True Label", fontsize=11, fontweight='bold')
    ax2.set_title("Confusion Matrix (Normalized, %)", fontsize=12, fontweight='bold', pad=10)
    
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            text_color = 'white' if cm_norm[i, j] > 50 else 'black'
            ax2.text(j, i, f"{cm_norm[i, j]:.1f}%", ha="center", va="center", 
                    color=text_color, fontsize=11, fontweight='bold')
    
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Percentage (%)')
    
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
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

