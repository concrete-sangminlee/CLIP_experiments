# Overleaf 프로젝트 설정 가이드

## 1. 파일 업로드 순서

Overleaf에서 이 프로젝트를 컴파일하려면 다음 파일/폴더를 모두 업로드해야 합니다:

### 필수 파일
- `main.tex` (메인 LaTeX 파일)
- `references.bib` (참고문헌 파일, 없으면 빈 파일 생성)

### 필수 폴더 (하위 파일 포함)
- `tables/` 폴더 전체
  - `dataset_overview.tex`
  - `performance_progression.tex`
  - `grid_search_top.tex`
  - `confusion_report.txt`
  
- `figures/` 폴더 전체
  - `performance_progression.png`
  - `grid_search_heatmap.png`
  - `tsne_projection.png`
  - `confusion_matrix.png`

## 2. Overleaf 업로드 방법

### 방법 A: 직접 업로드
1. Overleaf 프로젝트에서 좌측 상단 "Upload" 버튼 클릭
2. `main.tex` 업로드
3. "New Folder" 버튼으로 `tables` 폴더 생성
4. `tables` 폴더 안에 들어가서 모든 `.tex` 파일 업로드
5. "New Folder" 버튼으로 `figures` 폴더 생성
6. `figures` 폴더 안에 들어가서 모든 `.png` 파일 업로드

### 방법 B: ZIP 파일로 업로드
1. 로컬에서 다음 구조로 ZIP 파일 생성:
   ```
   project.zip
   ├── main.tex
   ├── references.bib
   ├── tables/
   │   ├── dataset_overview.tex
   │   ├── performance_progression.tex
   │   ├── grid_search_top.tex
   │   └── confusion_report.txt
   └── figures/
       ├── performance_progression.png
       ├── grid_search_heatmap.png
       ├── tsne_projection.png
       └── confusion_matrix.png
   ```
2. Overleaf에서 "New Project" → "Upload Project" → ZIP 선택

## 3. 컴파일 설정

1. Overleaf 좌측 상단 메뉴 → "Settings" 클릭
2. "Main document" 를 `main.tex`로 설정 (기본값)
3. "Compiler"를 `pdfLaTeX`로 설정
4. "Recompile" 버튼 클릭

## 4. 빈 references.bib 생성 (참고문헌 없을 경우)

Overleaf에서 "New File" 버튼 클릭 → `references.bib` 생성 후 아래 내용 입력:

```bibtex
@article{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  journal={International Conference on Machine Learning},
  year={2021}
}

@inproceedings{doersch2020crosstransformers,
  title={CrossTransformers: spatially-aware few-shot transfer},
  author={Doersch, Carl and Gupta, Ankush and Zisserman, Andrew},
  booktitle={NeurIPS},
  year={2020}
}

@article{cha2018autonomous,
  title={Autonomous structural visual inspection using region-based deep learning for detecting multiple damage types},
  author={Cha, Young-Jin and Choi, Wooram and B{\"u}y{\"u}k{\"o}zt{\"u}rk, Oral},
  journal={Computer-Aided Civil and Infrastructure Engineering},
  volume={33},
  number={9},
  pages={731--747},
  year={2018}
}
```

## 5. 문제 해결

### 오류: "File not found"
- `tables/` 또는 `figures/` 폴더가 업로드되지 않았거나 이름이 다른 경우
- 폴더 구조가 정확한지 확인 (대소문자 구분)

### 오류: "Reference undefined"
- `references.bib` 파일이 없는 경우
- 위 4번 항목대로 빈 파일 생성

### 컴파일이 느린 경우
- 이미지 파일이 큰 경우 정상 (최초 1회만 느림)
- 이후 재컴파일은 빠름

## 6. 로컬 컴파일 (선택사항)

TeX Live 또는 MiKTeX 설치 후:

```bash
cd C:\Users\snuai_006\Desktop\research_VLM
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 7. 추가 팁

- Overleaf는 자동 저장되므로 별도 저장 불필요
- PDF 미리보기는 우측 패널에 자동 표시
- 오류 메시지는 하단 로그 창에서 확인
- "Recompile from scratch" 옵션으로 캐시 초기화 가능

