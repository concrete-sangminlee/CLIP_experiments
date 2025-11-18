# CLIP-based Bolt Classification Research

Vision-Language Model (VLM) 기반 볼트 분류 연구 프로젝트입니다. CLIP 모델을 활용하여 SDNET2025 데이터셋의 볼트 이미지를 분류합니다.

## 📊 프로젝트 개요

- **목표**: CLIP zero-shot 베이스라인(51.70%)에서 시작하여 domain-aware prompt engineering, 아키텍처 개선, 하이퍼파라미터 최적화를 통해 **69.68%** 테스트 정확도 달성
- **데이터셋**: SDNET2025 볼트 분류
- **방법론**: CLIP feature extraction + Linear Probe with advanced techniques

## 📁 프로젝트 구조

```
research_VLM/
├── paper/                      # 논문 관련 파일
│   ├── main.tex               # LaTeX 논문 본문
│   ├── references.bib         # 참고문헌
│   ├── figures/               # 논문 그림 (PNG)
│   │   ├── confusion_matrix.png
│   │   ├── grid_search_heatmap.png
│   │   ├── performance_progression.png
│   │   └── tsne_projection.png
│   └── tables/                # 논문 테이블 (LaTeX)
│       ├── confusion_report.txt
│       ├── dataset_overview.tex
│       ├── grid_search_top.tex
│       └── performance_progression.tex
│
├── scripts/                   # 실험 스크립트
│   ├── extract_features.py           # CLIP feature 추출
│   ├── train_linear_probe.py         # Linear Probe 학습 (최종 버전)
│   ├── zero_shot_baseline.py         # Zero-shot 베이스라인
│   └── generate_publication_assets.py # 논문 자료 생성
│
├── docs/                      # 문서
│   ├── EXECUTION_GUIDE.md     # 실행 가이드
│   └── OVERLEAF_SETUP.md      # Overleaf 설정 가이드
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

| 단계 | 방법 | 정확도 | 향상 |
|------|------|--------|------|
| Baseline | CLIP Zero-shot | 51.70% | - |
| Stage 1 | Prompt Engineering + ViT-L-14 | ~60% | +8.3% |
| Stage 2 | Linear Probe + MLP | ~65% | +5% |
| Stage 3 | Mixup + Class Re-weighting | ~68% | +3% |
| Final | Hyperparameter Optimization | **69.68%** | +1.68% |

**총 향상**: +17.98 percentage points

## 📄 논문 작성

### Overleaf에서 컴파일

자세한 내용은 [`docs/OVERLEAF_SETUP.md`](docs/OVERLEAF_SETUP.md)를 참조하세요.

**요약**:
1. `paper/` 폴더의 모든 파일을 Overleaf에 업로드
2. Compiler를 `pdfLaTeX`로 설정
3. Main document를 `main.tex`로 설정
4. Recompile 실행

### 로컬에서 컴파일

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 🔧 주요 기술

- **Model**: OpenAI CLIP (ViT-L-14)
- **Framework**: PyTorch, open-clip-torch
- **Techniques**: 
  - Domain-aware prompt engineering
  - MLP probe architecture
  - Mixup augmentation
  - Class re-weighting
  - Grid search hyperparameter optimization

## 📚 참고 자료

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- 자세한 실행 가이드: [`docs/EXECUTION_GUIDE.md`](docs/EXECUTION_GUIDE.md)

## 🤝 기여

이 프로젝트는 연구 목적으로 개발되었습니다.

## 📝 라이선스

연구 및 교육 목적으로 사용 가능합니다.

---

**최종 업데이트**: 2025-11-18
