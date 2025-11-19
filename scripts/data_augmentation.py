"""Generate augmented bolt images for class rebalancing."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "SDNET2025"
AUG_ROOT = ROOT / "data" / "SDNET2025_augmented"

CLASS_MAP = {
    "loosened": "Dataset/Defected/Annotated Loosen bolt & nuts/Resized images 640-640",
    "missing": "Dataset/Defected/Annotated Missing bolt & nuts/Resized- 640-640",
    "fixed": "Dataset/Fixed/640-640",
}


AUG_STEPS = ["color", "blur", "noise", "cutout"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create augmented dataset variants.")
    parser.add_argument("--copies", type=int, default=2, help="Number of augmented copies per image")
    parser.add_argument("--target-root", type=Path, default=AUG_ROOT)
    parser.add_argument("--steps", type=str, default=",".join(AUG_STEPS))
    return parser.parse_args()


def apply_color_ops(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Color(img).enhance(random.uniform(0.6, 1.4))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    return img


def apply_blur(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    return img


def apply_noise(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        arr = np.asarray(img).astype("int16")
        noise = np.random.randint(-25, 25, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype("uint8")
        return Image.fromarray(arr)
    return img


def apply_cutout(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        w, h = img.size
        mask_w = random.randint(int(0.1 * w), int(0.3 * w))
        mask_h = random.randint(int(0.1 * h), int(0.3 * h))
        x0 = random.randint(0, w - mask_w)
        y0 = random.randint(0, h - mask_h)
        overlay = Image.new("RGB", (mask_w, mask_h), (0, 0, 0))
        img.paste(overlay, (x0, y0))
    return img


STEP_FUNCS = {
    "color": apply_color_ops,
    "blur": apply_blur,
    "noise": apply_noise,
    "cutout": apply_cutout,
}


def augment_image(img: Image.Image, steps):
    for step in steps:
        func = STEP_FUNCS.get(step)
        if func:
            img = func(img)
    return img


def main():
    args = parse_args()
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    for cls, rel_path in CLASS_MAP.items():
        src_dir = DATA_ROOT / rel_path
        dst_dir = args.target_root / rel_path
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img_path in src_dir.rglob("*.jpg"):
            img = Image.open(img_path).convert("RGB")
            for copy_idx in range(args.copies):
                aug = augment_image(img.copy(), steps)
                out_path = dst_dir / f"{img_path.stem}_aug{copy_idx}.jpg"
                aug.save(out_path, quality=95)
        print(f"[증강] {cls} -> {dst_dir}")


if __name__ == "__main__":
    main()
