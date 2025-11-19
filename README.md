# CLIP-based Bolt Classification Research

Vision-Language Model (VLM) 기반 볼트 결함 분류 연구 프로젝트입니다. CLIP 모델을 활용하여 SDNET2025 데이터셋의 볼트 이미지를 분류합니다.

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
│   ├── figures/               # 논문 그림 (PNG, 300 DPI)
│   │   ├── confusion_matrix.png
│   │   ├── grid_search_heatmap.png
│   │   ├── performance_progression.png
│   │   └── tsne_projection.png
│   └── tables/                # 논문 테이블 (LaTeX 형식)
│       ├── confusion_report.txt
│       ├── dataset_overview.tex
│       ├── grid_search_top.tex
│       ├── performance_progression.tex
│       └── class_performance.tex
│
├── scripts/                   # 실험 스크립트
│   ├── extract_features.py           # CLIP feature 추출
│   ├── train_linear_probe.py         # 앙상블 + pseudo label 기반 MLP Probe
│   ├── zero_shot_baseline.py         # Prompt ensemble zero-shot 평가
│   ├── self_training_loop.py         # Iterative pseudo labeling
│   ├── lora_finetune.py              # CLIP 시각 백본 LoRA 파인튜닝
│   ├── data_augmentation.py          # 클래스 불균형 대응 증강 생성
│   ├── prompt_library.py             # 다국어 프롬프트 템플릿 정의
│   └── generate_publication_assets.py # 논문 자료 자동 생성
│
├── data/                      # 데이터셋 (Git에 추적되지 않음)
│   └── SDNET2025/
│       └── Dataset/
│           ├── Defected/
│           │   ├── Annotated Loosen bolt & nuts/
│           │   └── Annotated Missing bolt & nuts/
│           └── Fixed/
│
├── venv/                      # Python 가상환경 (Git에 추적되지 않음)
├── .gitignore                 # Git 무시 파일 목록
└── README.md                  # 프로젝트 문서
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

SDNET2025 데이터셋을 `data/SDNET2025/` 폴더에 배치합니다. 데이터 구조는 다음과 같아야 합니다:

```
data/SDNET2025/
└── Dataset/
    ├── Defected/
    │   ├── Annotated Loosen bolt & nuts/Resized images 640-640/
    │   └── Annotated Missing bolt & nuts/Resized- 640-640/
    └── Fixed/640-640/
```

### 3. 실험 실행

#### Step 1: CLIP Feature 추출

```bash
python scripts/extract_features.py
```

**설정**:
- 모델: ViT-L-14 (OpenAI pretrained)
- 출력: 프로젝트 루트에 `.npy` 파일로 저장
  - `clip_features.npy`: Feature 벡터 (N×768)
  - `clip_labels.npy`: 클래스 레이블 (0, 1, 2)
  - `clip_class_names.npy`: 클래스 이름 배열
  - `clip_image_paths.npy`: 이미지 파일 경로 배열

**참고**: Feature 파일들은 `.gitignore`에 포함되어 Git에 추적되지 않습니다.

#### Step 2: Zero-shot 베이스라인 평가

```bash
python scripts/zero_shot_baseline.py \
  --templates base_en,materials_en,context_ko \
  --languages en,ko --max-images-per-class 120
```

**업데이트 내용**:
- 다국어 prompt ensemble을 구성하여 CLIP 텍스트 임베딩을 평균내고, 각 이미지별 로그를 `zero_shot_predictions.csv`로 저장합니다.
- CSV 로그는 이후 pseudo labeling, self-training, probe 학습에서 바로 사용할 수 있습니다.

#### Step 3: MLP Probe 학습

```bash
python scripts/train_linear_probe.py \
  --ensemble 3 --temperature-scaling \
  --pseudo-labels-csv zero_shot_predictions.csv \
  --metadata-normalize
```

**주요 개선점**:
- Mixup, CutMix, Manifold Mixup을 조합하고 pseudo label 가중치를 자동으로 부여합니다.
- `metadata_features.npy`가 존재하면 feature와 concat하여 조도·각도 메타 정보를 함께 학습합니다.
- 서로 다른 seed의 모델을 `--ensemble` 옵션으로 학습해 로짓을 평균내고, Temperature scaling 결과까지 JSON으로 기록합니다.
- `paper/tables/advanced_probe_summary.tex`가 자동 생성되어 논문 부록으로 바로 활용할 수 있습니다.

#### Step 4: 논문 자료 생성

```bash
python scripts/generate_publication_assets.py
```

**생성되는 파일**:
- `paper/figures/`: 논문용 고해상도 그림 (300 DPI)
- `paper/tables/`: LaTeX 형식 표 파일

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

### 🔧 추가 고도화 모듈

- `scripts/self_training_loop.py`: labeled 60장 seed → pseudo label confidence 기반 확장, 로그(`experiments/self_training_history.csv`) 생성.
- `scripts/data_augmentation.py`: `data/SDNET2025_augmented/`에 Cutout/ColorJitter/Blur 증강본을 자동 생성해 Missing 클래스 수를 보정합니다.
- `scripts/lora_finetune.py`: `peft` LoRA 어댑터를 CLIP 시각 백본에 주입해 5 epoch 정도의 경량 파인튜닝을 수행하고 `experiments/lora_clip.pt`로 저장합니다.
- `scripts/prompt_library.py`: 영어/한국어 템플릿을 한 곳에서 관리해 zero-shot, LoRA, self-training 모두 동일한 조건 묘사를 공유합니다.

### 📄 논문 재현성 체크리스트

1. `python scripts/zero_shot_baseline.py --save-csv zero_shot_predictions.csv`
2. `python scripts/train_linear_probe.py --ensemble 3 --pseudo-labels-csv zero_shot_predictions.csv`
3. (옵션) `python scripts/self_training_loop.py --iterations 4 --confidence 0.9`
4. (옵션) `python scripts/lora_finetune.py --lora-rank 8 --epochs 5`
5. `python scripts/generate_publication_assets.py`

prompt ensemble → pseudo labeling → 앙상블 probe → self-training/LoRA → 논문 figure/table 생성 순으로 실행하면 실험-논문 전체 파이프라인을 한 번에 재현할 수 있습니다.

## 📄 논문 작성

### 논문용 시각화 자료 생성

논문에 사용할 모든 그림과 표는 자동으로 생성됩니다:

```bash
python scripts/generate_publication_assets.py
```

**생성되는 자료**:

#### 그림 (`paper/figures/`)
- `performance_progression.png`: 단계별 성능 향상 그래프
- `grid_search_heatmap.png`: 하이퍼파라미터 그리드 서치 히트맵
- `confusion_matrix.png`: 혼동 행렬 (원본 카운트 + 정규화 퍼센트)
- `tsne_projection.png`: CLIP feature 공간의 t-SNE 시각화

#### 표 (`paper/tables/`)
- `dataset_overview.tex`: 데이터셋 클래스 분포
- `performance_progression.tex`: 단계별 성능 향상
- `grid_search_top.tex`: 그리드 서치 상위 5개 결과
- `class_performance.tex`: 클래스별 상세 성능 지표 (Precision, Recall, F1)
- `self_training.tex`: self-training iteration 기록 (존재 시 자동 생성)
- `confusion_report.txt`: 분류 리포트 (텍스트 형식)

**참고**: 생성된 LaTeX 표 파일들은 Word 문서에 직접 복사-붙여넣기하거나, 필요시 수정하여 사용할 수 있습니다.

## 🔧 주요 기술 및 실험 설정

### 모델 및 프레임워크
- **Backbone**: OpenAI CLIP ViT-L/14 (768-dim features)
- **Framework**: PyTorch 2.2, open-clip-torch
- **Probe Architecture**: 2-layer MLP (768 → 256 → 3)

### 핵심 기법 상세

#### 1. Domain-aware Prompt Engineering

각 클래스에 대한 도메인 특화 프롬프트:

- **Loosened**: "a close-up photo of a loosened steel bolt that is not properly tightened and needs repair on an industrial structure"
- **Missing**: "a close-up photo showing an empty bolt hole where a steel bolt or nut is completely missing from a metal structure"
- **Fixed**: "a close-up photo of a properly installed and tightly secured steel bolt with no defects or damage on a structure"

#### 2. MLP Probe Architecture

```
Input (768-dim) 
  → Linear(768→256) 
  → BatchNorm1d 
  → ReLU 
  → Dropout(0.2) 
  → Linear(256→3) 
  → Output (3 classes)
```

**정규화 기법**:
- BatchNorm: 학습 안정화
- Dropout: 과적합 방지 (rate=0.2)
- Gradient clipping: 최대 norm=1.0

#### 3. Regularization Techniques

- **Mixup**: α=0.3, 학습 초기 70% epoch에 적용
- **Class Re-weighting**: Inverse frequency weighting + Missing 클래스 1.5× boost
- **Label Smoothing**: 0.1 smoothing factor

#### 4. 최적화 설정

- **Optimizer**: AdamW
  - Learning Rate: 7e-4
  - Weight Decay: 1e-4
- **Scheduler**: ReduceLROnPlateau
  - Patience: 25 epochs
  - Factor: 0.5
- **Training**:
  - Epochs: 500 (최대)
  - Batch size: 32
  - Early stopping: Patience=75 epochs
- **Data Split**: 150 samples/class for training, 나머지 test set
- **Random Seed**: 42 (재현성 보장)

## 📋 파일 설명

### 스크립트 파일

#### `extract_features.py`
CLIP 모델을 사용하여 이미지에서 feature 벡터를 추출합니다.
- 입력: `data/SDNET2025/` 폴더의 이미지
- 출력: 프로젝트 루트에 `.npy` 파일 저장
- 모델: ViT-L-14 (OpenAI pretrained)

#### `zero_shot_baseline.py`
Prompt ensemble 기반 zero-shot 평가 및 pseudo label CSV 생성을 담당합니다.
- 모델: 기본 ViT-B-32 (옵션으로 변경 가능)
- 영어/한국어 템플릿을 동시에 사용하고 temperature 스케일링을 적용합니다.
- `--save-csv` 옵션으로 모든 이미지의 GT/예측/신뢰도를 저장하여 self-training, probe 학습에 재사용합니다.

#### `train_linear_probe.py`
Pseudo label + metadata + ensemble을 지원하는 MLP probe 학습 스크립트입니다.
- Mixup/CutMix/Manifold Mixup 조합과 class weighting, temperature scaling을 지원합니다.
- `experiments/<name>/metrics.json`에 모든 설정과 성능을 저장하고, 논문용 `advanced_probe_summary.tex`를 자동 생성합니다.
- `--ensemble` 옵션으로 다중 seed를 학습해 로짓 평균을 수행합니다.

#### `self_training_loop.py`
Pseudo label confidence를 기반으로 labeled set을 반복적으로 확장하는 self-training 실험 도구입니다.
- Iteration별로 추가된 샘플 수와 잔여 unlabeled 수를 CSV(`experiments/self_training_history.csv`)로 저장합니다.
- `generate_publication_assets.py`가 CSV를 감지하면 자동으로 LaTeX 표를 생성합니다.

#### `data_augmentation.py`
Missing 클래스 증강을 위해 Cutout/색감/블러/노이즈를 적용한 synthetic 이미지를 `data/SDNET2025_augmented/`에 생성합니다.

#### `lora_finetune.py`
`peft` LoRA 어댑터를 CLIP 비전 백본에 삽입해 경량 파인튜닝을 수행합니다.
- `experiments/lora_clip.pt`에 visual backbone과 분류 head state dict를 저장합니다.
- CLIP text tower는 고정하고 이미지 인코더만 업데이트하여 GPU 메모리를 절약합니다.

#### `prompt_library.py`
Zero-shot/LoRA/self-training에서 사용하는 영어·한국어 템플릿과 클래스 설명을 한 곳에서 관리합니다.

#### `generate_publication_assets.py`
논문용 그림과 표를 자동으로 생성합니다.
- 데이터셋 통계 분석
- 성능 진행 그래프 생성
- 그리드 서치 결과 시각화
- t-SNE feature 시각화
- 혼동 행렬 및 성능 리포트 생성

### 생성되는 파일

다음 파일들은 스크립트 실행 시 생성되며, `.gitignore`에 포함되어 Git에 추적되지 않습니다:

- `clip_features.npy`: 추출된 CLIP feature 벡터 (N×768)
- `clip_labels.npy`: 클래스 레이블 (0, 1, 2)
- `clip_class_names.npy`: 클래스 이름 배열
- `clip_image_paths.npy`: 이미지 파일 경로 배열

**재생성**: 필요시 `extract_features.py`를 실행하여 재생성할 수 있습니다.

## 📚 참고 자료

- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Radford et al., 2021
- [OpenCLIP](https://github.com/mlfoundations/open_clip) - Open-source CLIP implementation
- [SDNET2025 Dataset](https://github.com/sdnet2025/sdnet2025) - Structural defect dataset

## 🤝 기여

이 프로젝트는 연구 목적으로 개발되었습니다. 질문이나 제안사항이 있으시면 이슈를 등록해주세요.

## 📝 라이선스

연구 및 교육 목적으로 사용 가능합니다.

---

**최종 업데이트**: 2025-11
