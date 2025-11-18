# 성능 개선 실행 가이드

## 완료된 작업

### ✅ 1. 프롬프트 개선
- `zero_shot_clip_bolt.py`의 프롬프트를 더 구체적으로 수정
- 더 명확한 설명으로 클래스 구분 향상

### ✅ 2. 더 큰 모델 적용
- `extract_clip_features.py`에서 ViT-B-32 → ViT-L-14로 변경
- 더 높은 성능 기대 (느리지만 정확도 향상)

### ✅ 3. 하이퍼파라미터 그리드 서치 스크립트 생성
- `train_grid_search_clip.py` 생성
- LR, Weight Decay, Dropout을 체계적으로 실험

## 실행 순서

### 1단계: Feature 재추출 (프롬프트 개선 + ViT-L-14 적용)

**중요**: 프롬프트가 변경되었고 모델이 ViT-L-14로 변경되었으므로 feature를 다시 추출해야 합니다.

```powershell
python extract_clip_features.py
```

**예상 시간**: ViT-L-14는 ViT-B-32보다 느리므로 시간이 더 걸립니다 (약 2-3배)

**예상 효과**: 
- 프롬프트 개선: +1~2%
- ViT-L-14: +3~5%
- 총 예상: 63.30% → **67~70%**

### 2단계: 하이퍼파라미터 그리드 서치 실행

```powershell
python train_grid_search_clip.py
```

**실행 내용**:
- 총 27개 조합 실험 (LR: 3개 × Weight Decay: 3개 × Dropout: 3개)
- 각 조합마다 모델 학습 및 평가
- 상위 10개 조합 출력
- 결과를 `grid_search_results.txt`에 저장

**예상 시간**: 각 조합당 약 5-10분 (CPU 기준) × 27개 = 약 2-4시간

**예상 효과**: 최적 하이퍼파라미터 찾기 → +2~4% 추가 향상

### 3단계: 최적 하이퍼파라미터로 최종 학습

그리드 서치 결과를 확인한 후, 최고 성능 조합으로 `train_optimized_clip.py`를 수정하여 최종 학습:

```python
# train_optimized_clip.py에서 최고 성능 조합으로 수정
LR = <그리드 서치 최고값>
WEIGHT_DECAY = <그리드 서치 최고값>
DROPOUT_RATE = <그리드 서치 최고값>
```

그 다음:
```powershell
python train_optimized_clip.py
```

## 예상 성능 향상 로드맵

| 단계 | 현재 | 목표 | 방법 |
|------|------|------|------|
| 1단계 | 63.30% | 67~70% | 프롬프트 개선 + ViT-L-14 |
| 2단계 | 67% | 69~72% | 하이퍼파라미터 그리드 서치 |
| 3단계 | 69% | **71~75%** | 최적 조합으로 최종 학습 |

## 주의사항

1. **ViT-L-14는 메모리를 많이 사용합니다**
   - CPU에서 실행 시 메모리 부족 가능성
   - GPU가 있으면 더 빠르게 실행 가능

2. **그리드 서치는 시간이 오래 걸립니다**
   - 27개 조합 × 5-10분 = 약 2-4시간
   - 필요시 조합 수를 줄여서 실행 가능

3. **Feature 재추출은 필수입니다**
   - 프롬프트와 모델이 변경되었으므로 반드시 재추출 필요

## 빠른 실행 (시간 절약)

시간이 부족하면 그리드 서치 조합 수를 줄일 수 있습니다:

`train_grid_search_clip.py`에서:
```python
LR_VALUES = [7e-4, 1e-3]  # 3개 -> 2개
WEIGHT_DECAY_VALUES = [5e-4, 1e-3]  # 3개 -> 2개
DROPOUT_VALUES = [0.2, 0.3]  # 3개 -> 2개
# 총 8개 조합으로 감소
```

## 결과 확인

1. **Feature 추출 후**: `clip_features.npy` 파일 크기 확인 (ViT-L-14는 더 큰 feature)
2. **그리드 서치 후**: `grid_search_results.txt` 파일 확인
3. **최종 학습 후**: Test Accuracy 확인

## 다음 단계 (선택사항)

1. **앙상블**: 최고 성능 모델들을 앙상블하여 추가 향상
2. **Cross-Validation**: 더 안정적인 평가
3. **LoRA Fine-tuning**: CLIP 모델 자체를 fine-tuning (최고 성능)

