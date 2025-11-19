"""Zero-shot CLIP evaluation with prompt ensembles and CSV export."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

import open_clip

from prompt_library import DEFAULT_TEMPLATE_KEYS, build_prompts

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "SDNET2025"
DEFAULT_CSV = ROOT / "zero_shot_predictions.csv"

CLASS_DIRS = [
    "Dataset/Defected/Annotated Loosen bolt & nuts/Resized images 640-640",
    "Dataset/Defected/Annotated Missing bolt & nuts/Resized- 640-640",
    "Dataset/Fixed/640-640",
]
CLASS_DIR_TO_NAME = {
    "Dataset/Defected/Annotated Loosen bolt & nuts/Resized images 640-640": "loosened",
    "Dataset/Defected/Annotated Missing bolt & nuts/Resized- 640-640": "missing",
    "Dataset/Fixed/640-640": "fixed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLIP zero-shot baseline with richer prompts.")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--max-images-per-class", type=int, default=80)
    parser.add_argument("--templates", type=str, default=",".join(DEFAULT_TEMPLATE_KEYS))
    parser.add_argument("--languages", type=str, default="en,ko")
    parser.add_argument("--save-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--temperature", type=float, default=100.0)
    return parser.parse_args()


def collect_image_paths(max_per_class: int) -> Tuple[List[Path], List[str]]:
    image_paths: List[Path] = []
    labels: List[str] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for cls_dir_path in CLASS_DIRS:
        cls_dir = DATA_ROOT / cls_dir_path
        cls_name = CLASS_DIR_TO_NAME.get(cls_dir_path, cls_dir_path)
        cls_images = [p for p in cls_dir.rglob("*") if p.suffix.lower() in exts]
        if max_per_class:
            cls_images = cls_images[:max_per_class]
        image_paths.extend(cls_images)
        labels.extend([cls_name] * len(cls_images))
        print(f"[정보] 클래스 '{cls_name}'에서 {len(cls_images)}개 샘플 사용")
    return image_paths, labels


def encode_prompts(model, tokenizer, class_names: List[str], template_keys: List[str], languages: List[str], device) -> torch.Tensor:
    per_class_embeddings = []
    for cls in class_names:
        prompts = build_prompts(cls, template_keys, languages)
        if not prompts:
            continue
        with torch.no_grad():
            tokens = tokenizer(prompts).to(device)
            text_feats = model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        per_class_embeddings.append(text_feats.mean(dim=0, keepdim=True))
    if not per_class_embeddings:
        raise RuntimeError("No prompts were generated. Check template/language arguments.")
    return torch.cat(per_class_embeddings, dim=0)


def load_image(path: Path, preprocess) -> torch.Tensor:
    return preprocess(Image.open(path).convert("RGB"))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[정보] 모델 {args.model} ({args.pretrained}) on {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)
    model.eval()

    template_keys = [t.strip() for t in args.templates.split(",") if t.strip()]
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]

    image_paths, str_labels = collect_image_paths(args.max_images_per_class)
    class_names = sorted({CLASS_DIR_TO_NAME[c] for c in CLASS_DIRS})
    text_features = encode_prompts(model, tokenizer, class_names, template_keys, languages, device)

    rows = []
    correct = 0
    total = 0
    with torch.no_grad():
        for path, label in zip(image_paths, str_labels):
            img_tensor = load_image(path, preprocess).unsqueeze(0).to(device)
            image_feat = model.encode_image(img_tensor)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            logits = args.temperature * image_feat @ text_features.T
            probs = logits.softmax(dim=-1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_name = class_names[pred_idx]
            total += 1
            if pred_name == label:
                correct += 1
            rows.append(
                {
                    "image_path": path.as_posix(),
                    "gt_class": label,
                    "pred_class": pred_name,
                    "confidence": float(probs[pred_idx]),
                }
            )
            print(
                f"[{total:03d}] {path.name} | GT={label:8s} | Pred={pred_name:8s} | p={probs[pred_idx]:.3f}"
            )

    acc = correct / total if total else 0.0
    print("\n[결과] Zero-shot 정확도")
    print(f"  - Samples: {total}")
    print(f"  - Accuracy: {acc * 100:.2f}%")

    if rows:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.save_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "gt_class", "pred_class", "confidence"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[저장] CSV 로그: {args.save_csv}")


if __name__ == "__main__":
    main()
