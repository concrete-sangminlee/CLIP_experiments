# CLIP-based Bolt Classification Research

Vision-Language Model (VLM) 기반 볼트 분류 연구 프로젝트입니다. CLIP 모델을 활용하여 SDNET2025 데이터셋의 볼트 이미지를 분류합니다.

## 📊 프로젝트 개요

### 연구 목표
CLIP zero-shot 베이스라인(51.70%)에서 시작하여 domain-aware prompt engineering, 아키텍처 개선, 하이퍼파라미터 최적화를 통해 **69.68%** 테스트 정확도 달성 (+17.98 percentage points)

### 데이터셋
- **SDNET2025 Bolt Classification Dataset**
  - 3개 클래스: Loosened (324), Missing (200), Fixed (302)
  - 총 826개 이미지 (640×640 해상도)
  - 클래스 불균형 문제 존재 (1.62× 비율)

### 핵심 방법론
1. **CLIP Feature Extraction**: ViT-L/14 백본으로 768차원 feature 추출
2. **Domain-aware Prompt Engineering**: 산업용 볼트 특화 텍스트 프롬프트 설계
3. **MLP Probe Architecture**: 2-layer MLP with BatchNorm, Dropout
4. **Regularization Techniques**: Mixup augmentation, Class re-weighting, Label smoothing
5. **Hyperparameter Optimization**: 27개 조합 그리드 서치 (LR, Weight Decay, Dropout)

### 주요 기여
- ✅ **체계적인 점진적 개선 파이프라인**: Zero-shot → Linear Probe → MLP → Hyperparameter Search
- ✅ **클래스 불균형 대응**: Class weighting, Mixup을 통한 소수 클래스 성능 향상
- ✅ **재현 가능한 실험 설계**: 고정된 random seed, 명확한 train/test split
- ✅ **논문용 고품질 시각화**: 자동화된 그림/표 생성 스크립트

## 📁 프로젝트 구조

```
research_VLM/
├── paper/                      # 논문 관련 파일
│   ├── figures/               # 논문 그림 (PNG)
│   │   ├── confusion_matrix.png
│   │   ├── grid_search_heatmap.png
│   │   ├── performance_progression.png
│   │   └── tsne_projection.png
│   └── tables/                # 논문 테이블 (LaTeX)
│       ├── confusion_report.txt
│       ├── dataset_overview.tex
│       ├── grid_search_top.tex
│       ├── performance_progression.tex
│       └── class_performance.tex
│
├── scripts/                   # 실험 스크립트
│   ├── extract_features.py           # CLIP feature 추출
│   ├── train_linear_probe.py         # Linear Probe 학습 (최종 버전)
│   ├── zero_shot_baseline.py         # Zero-shot 베이스라인
│   └── generate_publication_assets.py # 논문 자료 생성
│
├── data/                      # 데이터 (gitignore)
│   └── SDNET2025/            # 데이터셋
│
├── .gitignore
└── README.md
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 필요한 패키지 설치
pip install torch torchvision
pip install open-clip-torch
pip install scikit-learn matplotlib seaborn pandas numpy
```

### 2. 데이터 준비

SDNET2025 데이터셋을 `data/SDNET2025/` 폴더에 배치합니다.

### 3. 실험 실행

#### Step 1: CLIP Feature 추출

```bash
python scripts/extract_features.py
```

- ViT-L-14 모델 사용
- 추출된 feature는 프로젝트 루트에 `.npy` 파일로 저장

#### Step 2: Zero-shot 베이스라인 평가

```bash
python scripts/zero_shot_baseline.py
```

- 예상 성능: ~51.70%

#### Step 3: Linear Probe 학습

```bash
python scripts/train_linear_probe.py
```

- MLP probe with mixup, class re-weighting
- 최적 하이퍼파라미터 적용
- 예상 성능: ~69.68%

#### Step 4: 논문 자료 생성

```bash
python scripts/generate_publication_assets.py
```

- `paper/figures/` 및 `paper/tables/` 폴더에 자료 생성

## 📈 성능 향상 과정

### 단계별 개선 전략

| 단계 | 방법 | 정확도 | 향상 | 주요 기법 |
|------|------|--------|------|----------|
| **Baseline** | CLIP Zero-shot | 51.70% | - | ViT-B/32, 기본 프롬프트 |
| **Stage 1** | Baseline Linear Probe | 51.70% | +0.00% | Linear classifier, 100 shots/class |
| **Stage 2** | Improved MLP Probe | 63.30% | +11.60% | MLP (256-dim), Mixup (α=0.3), Class weights |
| **Stage 3** | Prompt + Backbone Upgrade | 67.55% | +4.25% | Domain prompts, ViT-L/14 (768-dim) |
| **Final** | Hyperparameter Optimization | **69.68%** | +2.13% | Grid search (LR=7e-4, WD=1e-4, Dropout=0.2) |

**총 향상**: +17.98 percentage points (34.8% relative improvement)

### 핵심 발견사항
1. **Prompt Engineering의 중요성**: Domain-specific 프롬프트가 +4.25% 향상
2. **아키텍처 깊이의 효과**: MLP probe가 Linear보다 +11.6% 향상
3. **하이퍼파라미터 민감도**: 최적 조합으로 +2.13% 추가 향상
4. **클래스 불균형 문제**: Missing 클래스의 낮은 precision (41.79%) 확인

## 📄 논문 작성

### 논문용 시각화 자료 생성

논문에 사용할 모든 그림과 표는 자동으로 생성됩니다:

```bash
python scripts/generate_publication_assets.py
```

이 스크립트는 다음을 생성합니다:
- **그림** (`paper/figures/`):
  - `performance_progression.png`: 단계별 성능 향상 그래프
  - `grid_search_heatmap.png`: 하이퍼파라미터 그리드 서치 히트맵
  - `confusion_matrix.png`: 혼동 행렬 (원본 및 정규화 버전)
  - `tsne_projection.png`: CLIP feature 공간의 t-SNE 시각화

- **표** (`paper/tables/`):
  - `dataset_overview.tex`: 데이터셋 클래스 분포
  - `performance_progression.tex`: 단계별 성능 향상
  - `grid_search_top.tex`: 그리드 서치 상위 5개 결과
  - `class_performance.tex`: 클래스별 상세 성능 지표 (Precision, Recall, F1)
  - `confusion_report.txt`: 분류 리포트 (텍스트 형식)

**참고**: 생성된 LaTeX 표 파일들은 Word 문서에 직접 복사-붙여넣기하거나, 필요시 수정하여 사용할 수 있습니다.

## 🔧 주요 기술 및 실험 설정

### 모델 및 프레임워크
- **Backbone**: OpenAI CLIP ViT-L/14 (768-dim features)
- **Framework**: PyTorch 2.2, open-clip-torch
- **Probe Architecture**: 2-layer MLP (768 → 256 → 3)

### 핵심 기법 상세

#### 1. Domain-aware Prompt Engineering
```
- Loosened: "a close-up photo of a loosened steel bolt that is not properly tightened..."
- Missing: "a close-up photo showing an empty bolt hole where a steel bolt is completely missing..."
- Fixed: "a close-up photo of a properly installed and tightly secured steel bolt..."
```

#### 2. MLP Probe Architecture
- **구조**: Linear(768→256) → BatchNorm → ReLU → Dropout(0.2) → Linear(256→3)
- **정규화**: BatchNorm, Dropout, Gradient clipping (max_norm=1.0)

#### 3. Regularization Techniques
- **Mixup**: α=0.3, 학습 초기 70% epoch에 적용
- **Class Re-weighting**: Inverse frequency weighting + Missing 클래스 1.5× boost
- **Label Smoothing**: 0.1 smoothing factor

#### 4. 최적화 설정
- **Optimizer**: AdamW (LR=7e-4, Weight Decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (patience=25, factor=0.5)
- **Training**: 500 epochs, Batch size=32, Early stopping (patience=75)
- **Data Split**: 150 samples/class for training, 나머지 test set

## 📚 참고 자료

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [SDNET2025 Dataset](https://github.com/sdnet2025/sdnet2025)

## 🤝 기여

이 프로젝트는 연구 목적으로 개발되었습니다.

## 📝 라이선스

연구 및 교육 목적으로 사용 가능합니다.

## 📋 파일 설명

### 스크립트 파일
- `extract_features.py`: CLIP 모델을 사용하여 이미지에서 feature 벡터 추출
- `zero_shot_baseline.py`: Zero-shot CLIP 분류 성능 평가
- `train_linear_probe.py`: MLP probe를 사용한 분류기 학습 및 평가
- `generate_publication_assets.py`: 논문용 그림과 표 자동 생성

### 생성되는 파일
- `clip_features.npy`: 추출된 CLIP feature 벡터 (N×768)
- `clip_labels.npy`: 클래스 레이블 (0, 1, 2)
- `clip_class_names.npy`: 클래스 이름 배열
- `clip_image_paths.npy`: 이미지 파일 경로 배열

**주의**: `.npy` 파일들은 `.gitignore`에 포함되어 있어 Git에 추적되지 않습니다. 필요시 `extract_features.py`를 실행하여 재생성할 수 있습니다.

---

**최종 업데이트**: 2025-01-XX
