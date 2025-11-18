import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import open_clip

# ---------- 설정 부분 ----------

# SDNET2025 데이터 루트
DATA_ROOT = Path(__file__).parent.parent / "data" / "SDNET2025"

# feature, label 결과를 저장할 곳 (프로젝트 루트)
OUTPUT_DIR = Path(__file__).parent.parent

# 클래스별 폴더 이름 (DATA_ROOT 아래 실제 폴더명과 맞춰주세요)
# 실제 데이터 구조가 다르면 여기를 수정하세요.
CLASS_DIRS = [
    "Dataset/Defected/Annotated Loosen bolt & nuts/Resized images 640-640",
    "Dataset/Defected/Annotated Missing bolt & nuts/Resized- 640-640",
    "Dataset/Fixed/640-640"
]

# CLASS_DIRS와 클래스 이름을 매핑 (폴더 경로 -> 클래스 이름)
CLASS_DIR_TO_NAME = {
    "Dataset/Defected/Annotated Loosen bolt & nuts/Resized images 640-640": "loosened",
    "Dataset/Defected/Annotated Missing bolt & nuts/Resized- 640-640": "missing",
    "Dataset/Fixed/640-640": "fixed"
}

# 사용할 CLIP 모델
# ViT-L-14: 더 큰 모델, 더 높은 성능 (느리지만 정확도 향상)
# ViT-B-32: 기본 모델, 빠름
CLIP_MODEL_NAME = "ViT-L-14"  # "ViT-B-32"에서 변경
CLIP_PRETRAINED = "openai"

# 최대 이미지 수 (테스트용으로 제한하고 싶으면 사용, 아니면 None)
MAX_IMAGES_PER_CLASS = None  # 예: 100 으로 제한하고 싶으면 100

# ---------- 유틸 함수 ----------

def collect_image_paths(root: Path, class_dirs, class_dir_to_name, max_per_class=None):
    """클래스별 이미지 경로와 클래스 이름 리스트를 만든다."""
    image_paths = []
    labels = []  # 문자열 라벨 (클래스 이름)

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

    # 1. CLIP 모델 로드
    print(f"[정보] CLIP 모델 로드: {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )

    model = model.to(device)
    model.eval()

    # 2. 이미지 경로 및 문자열 라벨 수집
    image_paths, str_labels = collect_image_paths(
        DATA_ROOT, CLASS_DIRS, CLASS_DIR_TO_NAME, max_per_class=MAX_IMAGES_PER_CLASS
    )

    if len(image_paths) == 0:
        print("[오류] 사용할 이미지가 없습니다. DATA_ROOT와 폴더 구조를 확인하세요.")
        return

    # 3. 문자열 라벨 -> 숫자 라벨 매핑
    #    클래스 이름을 정렬하여 일관된 인덱스 매핑 생성
    class_names = sorted(set(str_labels))  # ["fixed", "loosened", "missing"] 순서로 정렬
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    int_labels = np.array([class_to_idx[c] for c in str_labels], dtype=np.int64)
    image_paths_str = np.array([str(p) for p in image_paths])

    print("\n[정보] 클래스 인덱스 매핑:")
    for cls, idx in class_to_idx.items():
        print(f"  - {cls}: {idx}")

    # 4. 이미지 feature 추출
    all_features = []
    total = len(image_paths)
    print(f"\n[정보] 총 {total}개 이미지에서 CLIP feature 추출 시작")

    with torch.no_grad():
        for i, img_path in enumerate(image_paths, start=1):
            img_tensor = load_image(img_path, preprocess).unsqueeze(0).to(device)
            feats = model.encode_image(img_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # 정규화
            feats = feats.cpu().numpy()[0]  # (D,)
            all_features.append(feats)

            if i % 50 == 0 or i == total:
                print(f"  - 진행 상황: {i}/{total}")

    features_array = np.stack(all_features, axis=0)  # (N, D)

    print(f"\n[정보] feature shape: {features_array.shape}")
    print(f"[정보] label shape: {int_labels.shape}")

    # 5. 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "clip_features.npy", features_array)
    np.save(OUTPUT_DIR / "clip_labels.npy", int_labels)
    np.save(OUTPUT_DIR / "clip_class_names.npy", np.array(class_names))
    np.save(OUTPUT_DIR / "clip_image_paths.npy", image_paths_str)

    print("\n[완료] 다음 파일들이 저장되었습니다:")
    print(f"  - {OUTPUT_DIR / 'clip_features.npy'}")
    print(f"  - {OUTPUT_DIR / 'clip_labels.npy'}")
    print(f"  - {OUTPUT_DIR / 'clip_class_names.npy'}")
    print(f"  - {OUTPUT_DIR / 'clip_image_paths.npy'}")


if __name__ == "__main__":
    main()

