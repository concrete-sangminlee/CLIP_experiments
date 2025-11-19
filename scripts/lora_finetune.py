"""LoRA fine-tuning entrypoint for CLIP image tower."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import open_clip

try:
    from peft import LoraConfig, get_peft_model
except ImportError as exc:  # pragma: no cover
    raise SystemExit("peft is required for LoRA fine-tuning. Install with `pip install peft`." ) from exc


ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "SDNET2025"

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


class BoltDataset(Dataset):
    def __init__(self, preprocess, max_per_class: int = None):
        self.preprocess = preprocess
        self.samples: List[Tuple[Path, int]] = []
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for cls_dir in CLASS_DIRS:
            cls_name = CLASS_DIR_TO_NAME[cls_dir]
            cls_idx = list(CLASS_DIR_TO_NAME.values()).index(cls_name)
            folder = DATA_ROOT / cls_dir
            imgs = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
            if max_per_class:
                imgs = imgs[:max_per_class]
            self.samples.extend((img, cls_idx) for img in imgs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.preprocess(img), label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CLIP image encoder with LoRA adapters.")
    parser.add_argument("--model", type=str, default="ViT-L-14")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-per-class", type=int, default=200)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "lora_clip.pt")
    return parser.parse_args()


def collect_target_modules(model) -> List[str]:
    targets = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ("c_proj" in name or "c_fc" in name):
            targets.add(name.split(".")[-1])
    return sorted(targets)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    dataset = BoltDataset(preprocess, max_per_class=args.max_per_class)
    if len(dataset) == 0:
        raise SystemExit('데이터셋이 비어 있습니다. data/SDNET2025 경로를 확인하세요.')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    target_modules = collect_target_modules(model.visual)
    lora_cfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank * 2, target_modules=target_modules, lora_dropout=0.1)
    model.visual = get_peft_model(model.visual, lora_cfg)
    model = model.to(device)
    sample_img, _ = dataset[0]
    feat_dim = model.encode_image(sample_img.unsqueeze(0).to(device)).shape[-1]
    head = torch.nn.Linear(feat_dim, len(CLASS_DIR_TO_NAME)).to(device)
    params = list(model.visual.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            feats = model.encode_image(images)
            logits = head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
        print(f"[LoRA] Epoch {epoch}: Loss={running_loss/total:.4f} Acc={correct/total*100:.2f}%")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"visual": model.visual.state_dict(), "head": head.state_dict()}, args.output)
    print(f"[저장] LoRA 가중치: {args.output}")


if __name__ == "__main__":
    main()
