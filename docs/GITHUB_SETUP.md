# GitHub 리포지토리 연결 가이드

이 문서는 현재 프로젝트를 GitHub 리포지토리와 연결하는 방법을 안내합니다.

## 사전 준비

1. GitHub 계정이 있어야 합니다
2. Git이 설치되어 있어야 합니다 (`git --version`으로 확인)

## 단계별 가이드

### 1단계: GitHub에서 새 리포지토리 생성

1. [GitHub.com](https://github.com)에 로그인
2. 우측 상단의 `+` 버튼 클릭 → `New repository` 선택
3. 리포지토리 설정:
   - **Repository name**: `research_VLM` (또는 원하는 이름)
   - **Description**: "CLIP-based Bolt Classification Research"
   - **Public** 또는 **Private** 선택
   - ⚠️ **중요**: "Add a README file" 체크 해제 (로컬에 이미 파일이 있음)
   - ⚠️ **중요**: "Add .gitignore" 체크 해제 (이미 존재함)
4. `Create repository` 클릭

### 2단계: 로컬 Git 리포지토리 확인

프로젝트 루트에서 다음 명령어 실행:

```powershell
cd C:\Users\snuai_006\Desktop\research_VLM
git status
```

이미 Git 리포지토리가 초기화되어 있습니다 (`.git` 폴더 존재).

### 3단계: 원격 저장소 연결

GitHub에서 생성한 리포지토리의 URL을 사용하여 연결:

#### HTTPS 방식 (추천)

```powershell
git remote add origin https://github.com/[사용자명]/[리포지토리명].git
```

**예시**:
```powershell
git remote add origin https://github.com/myusername/research_VLM.git
```

#### SSH 방식 (SSH 키 설정 필요)

```powershell
git remote add origin git@github.com:[사용자명]/[리포지토리명].git
```

### 4단계: 연결 확인

```powershell
git remote -v
```

**예상 출력**:
```
origin  https://github.com/myusername/research_VLM.git (fetch)
origin  https://github.com/myusername/research_VLM.git (push)
```

### 5단계: 파일 스테이징 및 커밋

```powershell
# 모든 변경사항 추가
git add .

# 첫 번째 커밋 생성
git commit -m "Initial commit: CLIP-based bolt classification project"
```

### 6단계: GitHub에 푸시

```powershell
# main 브랜치로 설정 (최신 GitHub 기본 브랜치명)
git branch -M main

# GitHub에 푸시
git push -u origin main
```

## 문제 해결

### 오류: "remote origin already exists"

이미 원격 저장소가 설정되어 있는 경우:

```powershell
# 기존 원격 저장소 확인
git remote -v

# 기존 원격 저장소 제거
git remote remove origin

# 새로운 원격 저장소 추가 (3단계 반복)
git remote add origin https://github.com/[사용자명]/[리포지토리명].git
```

### 오류: "failed to push some refs"

GitHub 리포지토리에 이미 파일이 있는 경우:

```powershell
# 원격 변경사항 가져오기
git pull origin main --allow-unrelated-histories

# 다시 푸시
git push -u origin main
```

### 인증 오류 (HTTPS 사용 시)

Windows에서는 Git Credential Manager가 자동으로 인증을 처리합니다.
처음 푸시 시 GitHub 로그인 창이 나타나면 로그인하세요.

Personal Access Token이 필요한 경우:
1. GitHub → Settings → Developer settings → Personal access tokens
2. "Generate new token" 클릭
3. `repo` 권한 선택
4. 생성된 토큰을 비밀번호 대신 사용

## .gitignore 확인

현재 프로젝트의 `.gitignore`는 다음을 제외합니다:

- `venv/` - 가상환경
- `data/SDNET2025/` - 데이터셋 (용량이 큼)
- `*.npy` - 추출된 feature 파일 (재생성 가능)
- `paper/*.pdf` - 컴파일된 LaTeX PDF
- `__pycache__/` - Python 캐시
- 기타 임시 파일

필요시 `.gitignore` 파일을 수정하여 추가/제외할 파일을 조정하세요.

## 일상적인 Git 사용

### 변경사항 커밋 및 푸시

```powershell
# 변경된 파일 확인
git status

# 변경사항 추가
git add .

# 커밋
git commit -m "설명 메시지"

# GitHub에 푸시
git push
```

### 변경사항 가져오기

```powershell
# 원격 저장소의 최신 변경사항 가져오기
git pull
```

### 브랜치 작업

```powershell
# 새 브랜치 생성 및 전환
git checkout -b feature/new-experiment

# 브랜치 목록 확인
git branch

# 브랜치 전환
git checkout main

# 브랜치 병합
git merge feature/new-experiment

# 브랜치 삭제
git branch -d feature/new-experiment
```

## 추천 커밋 메시지 규칙

- `feat:` - 새로운 기능 추가
- `fix:` - 버그 수정
- `docs:` - 문서 수정
- `refactor:` - 코드 리팩토링
- `test:` - 테스트 추가/수정
- `chore:` - 기타 작업

**예시**:
```
feat: Add MLP probe architecture
fix: Correct data path in extract_features.py
docs: Update README with installation guide
refactor: Reorganize project structure
```

## 다음 단계

1. **README.md 업데이트**: GitHub에서 보이는 프로젝트 설명 개선
2. **LICENSE 추가**: 라이선스 파일 추가 (MIT, Apache 등)
3. **GitHub Actions**: CI/CD 파이프라인 설정 (선택사항)
4. **Issues/Projects**: GitHub Issues로 작업 관리 (선택사항)

---

**참고 자료**:
- [Git 공식 문서](https://git-scm.com/doc)
- [GitHub 가이드](https://guides.github.com/)
- [Git 치트시트](https://education.github.com/git-cheat-sheet-education.pdf)

