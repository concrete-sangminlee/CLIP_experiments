import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image

import open_clip


# ---------- 설정 부분 ----------

# SDNET2025 데이터가 있는 루트 디렉토리
DATA_ROOT = Path(__file__).parent.parent / "data" / "SDNET2025"

# 클래스별 폴더 이름과 텍스트 프롬프트 매핑 (개선된 버전)
CLASS_PROMPTS = {
    "loosened": "a close-up photo of a loosened steel bolt that is not properly tightened and needs repair on an industrial structure",
    "missing": "a close-up photo showing an empty bolt hole where a steel bolt or nut is completely missing from a metal structure",
    "fixed": "a close-up photo of a properly installed and tightly secured steel bolt with no defects or damage on a structure",
}

# 실제로 사용할 클래스 폴더들 (DATA_ROOT 아래 폴더 이름과 동일하게)
# 실제 데이터 구조가 다르면 여기를 수정하세요.
# 예: ["Annotated Loosen bolt & nuts/Resized images 640-640", "Annotated Missing bolt & nuts/Resized- 640-640", "Fixed/640-640"]
CLASS_DIRS = [
    "Dataset/Defected/Annotated Loosen bolt & nuts/Resized images 640-640",
    "Dataset/Defected/Annotated Missing bolt & nuts/Resized- 640-640",
    "Dataset/Fixed/640-640"
]

# CLASS_DIRS와 CLASS_PROMPTS의 키를 매핑 (폴더 경로 -> 클래스 이름)
# CLASS_DIRS의 순서와 CLASS_PROMPTS.keys()의 순서가 일치해야 함
CLASS_DIR_TO_NAME = {
    "Dataset/Defected/Annotated Loosen bolt & nuts/Resized images 640-640": "loosened",
    "Dataset/Defected/Annotated Missing bolt & nuts/Resized- 640-640": "missing",
    "Dataset/Fixed/640-640": "fixed"
}

# 사용할 CLIP 모델 이름
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# 테스트할 최대 이미지 수 (너무 많으면 시간 오래 걸리니까 제한)
MAX_IMAGES_PER_CLASS = 50  # 원하면 10 정도로 줄여도 됨


# ---------- 유틸 함수 ----------

def collect_image_paths(root: Path, class_dirs, class_dir_to_name, max_per_class=None):
    image_paths = []
    labels = []

    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for cls_dir_path in class_dirs:
        cls_dir = root / cls_dir_path
        if not cls_dir.exists():
            print(f"[경고] 폴더가 없습니다: {cls_dir} (건너뜀)")
            continue

        # 폴더 경로를 클래스 이름으로 변환
        cls_name = class_dir_to_name.get(cls_dir_path, cls_dir_path)
        
        cls_images = []
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in exts:
                cls_images.append(p)

        if max_per_class is not None:
            cls_images = cls_images[:max_per_class]

        print(f"[정보] 클래스 '{cls_name}' (폴더: {cls_dir_path})에서 {len(cls_images)}개 이미지 사용")

        image_paths.extend(cls_images)
        labels.extend([cls_name] * len(cls_images))

    return image_paths, labels


def load_image(path, preprocess):
    img = Image.open(path).convert("RGB")
    return preprocess(img)


# ---------- 메인 ----------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[정보] 사용 디바이스: {device}")

    # 1. CLIP 모델 불러오기
    print(f"[정보] CLIP 모델 로드: {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    model = model.to(device)
    model.eval()

    # 2. 텍스트 프롬프트 인코딩
    class_names = list(CLASS_PROMPTS.keys())
    prompts = [CLASS_PROMPTS[c] for c in class_names]

    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    print("[정보] 텍스트 프롬프트 인코딩 완료")
    for name, prompt in zip(class_names, prompts):
        print(f"  - {name}: \"{prompt}\"")

    # 3. 이미지 경로 모으기
    image_paths, labels = collect_image_paths(
        DATA_ROOT, CLASS_DIRS, CLASS_DIR_TO_NAME, max_per_class=MAX_IMAGES_PER_CLASS
    )

    if len(image_paths) == 0:
        print("[오류] 사용할 이미지가 없습니다. DATA_ROOT와 폴더 구조를 확인하세요.")
        return

    # 4. 이미지 인코딩 + zero-shot 분류
    correct = 0
    total = 0

    print("\n[정보] Zero-shot CLIP 예측 시작")

    with torch.no_grad():
        for img_path, true_cls in zip(image_paths, labels):
            # 이미지 로드 및 전처리
            image_tensor = load_image(img_path, preprocess).unsqueeze(0).to(device)

            # 이미지 feature 추출
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 텍스트 feature와 코사인 유사도 계산
            # (batch=1 이므로 text_features (N_classes, D)와 matmul)
            logits = 100.0 * image_features @ text_features.T  # temperature scaling
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

            pred_idx = int(np.argmax(probs))
            pred_cls = class_names[pred_idx]
            pred_prob = probs[pred_idx]

            total += 1
            if pred_cls == true_cls:
                correct += 1

            # 파일 이름 + 예측 결과 출력
            print(
                f"[{total:03d}] {img_path.name} | "
                f"GT: {true_cls:8s} | Pred: {pred_cls:8s} "
                f"(p={pred_prob:.3f})"
            )

    acc = correct / total if total > 0 else 0.0
    print("\n[결과] Zero-shot CLIP 정확도")
    print(f"  - 총 이미지 수: {total}")
    print(f"  - 정답 수: {correct}")
    print(f"  - Accuracy: {acc * 100:.2f}%")

    print("\n[참고] 정확도가 낮더라도 괜찮습니다.")
    print(" - 여기서부터 few-shot linear probe나 LoRA fine-tuning으로 확장하면 됨.")


if __name__ == "__main__":
    main()

