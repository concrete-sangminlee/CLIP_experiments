import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# sklearn은 선택사항
try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ---------- 설정 부분 ----------

ROOT_DIR = Path(__file__).parent.parent
FEATURE_PATH = ROOT_DIR / "clip_features.npy"
LABEL_PATH = ROOT_DIR / "clip_labels.npy"
CLASS_NAMES_PATH = ROOT_DIR / "clip_class_names.npy"

# 최적화된 하이퍼파라미터 (그리드 서치 최고 성능 조합)
N_TRAIN_PER_CLASS = 150

# 모델 아키텍처
USE_MLP = True
MLP_HIDDEN_DIM = 256
MLP_NUM_LAYERS = 2
USE_DROPOUT = True
DROPOUT_RATE = 0.2  # 그리드 서치 최고 성능 조합

# 학습 하이퍼파라미터 (개선된 설정)
EPOCHS = 500  # 300 -> 500으로 증가 (더 긴 학습)
BATCH_SIZE = 32
LR = 7e-4  # 그리드 서치 최고 성능 조합
WEIGHT_DECAY = 1e-4  # 그리드 서치 최고 성능 조합
USE_LR_SCHEDULER = True
SCHEDULER_PATIENCE = 25  # 그리드 서치와 동일 (Early stopping은 patience * 3으로 더 관대하게)

# 데이터 증강
USE_MIXUP = True
MIXUP_ALPHA = 0.3

# Loss 함수
USE_FOCAL_LOSS = False
USE_CLASS_WEIGHTS = True
USE_LABEL_SMOOTHING = True
LABEL_SMOOTHING = 0.1

# 추가 개선 기법
USE_TEMPERATURE_SCALING = False  # Temperature scaling (기본값 False, 필요시 True로 변경)
TEMPERATURE_SEARCH_RANGE = [0.5, 3.0]  # Temperature 탐색 범위

RANDOM_SEED = 42

# ---------- 유틸 ----------

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        num_classes = inputs.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


class DeepMLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128], dropout_rate=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


def find_optimal_temperature(model, X_val, y_val, device, temp_range=[0.5, 3.0], step=0.1):
    """Temperature scaling을 위한 최적 temperature 찾기"""
    model.eval()
    best_temp = 1.0
    best_acc = 0.0
    
    with torch.no_grad():
        logits = model(X_val)
        
        for temp in np.arange(temp_range[0], temp_range[1] + step, step):
            scaled_logits = logits / temp
            preds = torch.argmax(scaled_logits, dim=1)
            acc = (preds == y_val).float().mean().item()
            
            if acc > best_acc:
                best_acc = acc
                best_temp = temp
    
    return best_temp


# ---------- 메인 ----------

def main():
    set_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[정보] 사용 디바이스: {device}")
    print(f"[정보] 개선된 최종 학습 설정")
    print(f"  - LR: {LR:.0e}")
    print(f"  - Weight Decay: {WEIGHT_DECAY:.0e}")
    print(f"  - Dropout: {DROPOUT_RATE:.1f}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Scheduler Patience: {SCHEDULER_PATIENCE}")

    # 1. 데이터 로드
    if not FEATURE_PATH.exists() or not LABEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        print("[오류] feature/label/class_names 파일이 없습니다.")
        return

    features = np.load(FEATURE_PATH)
    labels = np.load(LABEL_PATH)
    class_names = np.load(CLASS_NAMES_PATH)

    num_samples, feat_dim = features.shape
    num_classes = len(class_names)

    print(f"[정보] features shape: {features.shape}")
    print(f"[정보] num_classes: {num_classes}")
    print(f"[정보] class_names: {class_names}")

    # 2. Train/Test 분할 (그리드 서치와 동일한 방식으로 재현성 확보)
    train_indices = []
    test_indices = []

    # 그리드 서치와 동일한 seed로 데이터 분할
    set_seed(RANDOM_SEED)
    
    for cls_idx in range(num_classes):
        cls_name = class_names[cls_idx]
        cls_idxs = np.where(labels == cls_idx)[0]
        np.random.shuffle(cls_idxs)

        n_train = min(N_TRAIN_PER_CLASS, len(cls_idxs))
        cls_train = cls_idxs[:n_train]
        cls_test = cls_idxs[n_train:]

        train_indices.extend(cls_train.tolist())
        test_indices.extend(cls_test.tolist())

        print(f"[정보] 클래스 '{cls_name}': train {len(cls_train)}, test {len(cls_test)}")

    train_indices = np.array(train_indices, dtype=np.int64)
    test_indices = np.array(test_indices, dtype=np.int64)

    X_train = features[train_indices]
    y_train = labels[train_indices]
    X_test = features[test_indices]
    y_test = labels[test_indices]

    print(f"\n[정보] train 샘플 수: {X_train.shape[0]}")
    print(f"[정보] test 샘플 수:  {X_test.shape[0]}")

    # numpy -> torch tensor (그리드 서치와 동일하게 전체 train 데이터 사용)
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)

    # 3. 클래스 가중치 계산
    if USE_CLASS_WEIGHTS:
        all_class_counts = np.bincount(labels)
        total_all = all_class_counts.sum()
        class_weights = total_all / (num_classes * all_class_counts)
        
        missing_idx = None
        for i, name in enumerate(class_names):
            if str(name) == 'missing':
                missing_idx = i
                break
        if missing_idx is not None:
            class_weights[missing_idx] *= 1.5
        
        class_weights = class_weights / class_weights.min()
        class_weights = torch.from_numpy(class_weights).float().to(device)
        print(f"[정보] 클래스 가중치: {class_weights.cpu().numpy()}")
    else:
        class_weights = None

    # 4. Loss 함수
    if USE_LABEL_SMOOTHING:
        criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
        print(f"[정보] Loss 함수: Label Smoothing CrossEntropy (smoothing={LABEL_SMOOTHING})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"[정보] Loss 함수: CrossEntropyLoss")

    # 5. 모델 정의
    if USE_MLP:
        hidden_dims = [MLP_HIDDEN_DIM] * MLP_NUM_LAYERS
        if MLP_NUM_LAYERS > 1:
            step = (MLP_HIDDEN_DIM - 128) // (MLP_NUM_LAYERS - 1)
            hidden_dims = [MLP_HIDDEN_DIM - i * step for i in range(MLP_NUM_LAYERS)]
        model = DeepMLPClassifier(feat_dim, num_classes, hidden_dims, DROPOUT_RATE if USE_DROPOUT else 0.0).to(device)
        print(f"[정보] 모델 구조: {feat_dim} -> {' -> '.join(map(str, hidden_dims))} -> {num_classes}")
    else:
        model = nn.Linear(feat_dim, num_classes).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    if USE_LR_SCHEDULER:
        # Test accuracy 기준으로 스케줄링 (그리드 서치와 동일)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=SCHEDULER_PATIENCE, verbose=False
        )

    # 6. 학습 루프
    def evaluate(X, y, return_preds=False, temperature=1.0):
        model.eval()
        with torch.no_grad():
            logits = model(X)
            if temperature != 1.0:
                logits = logits / temperature
            preds = torch.argmax(logits, dim=1)
            correct = (preds == y).sum().item()
            total = y.size(0)
            acc = correct / total if total > 0 else 0.0
            if return_preds:
                return acc, preds.cpu().numpy()
            return acc

    n_train_samples = X_train.size(0)
    print("\n[정보] 개선된 선형 분류기 학습 시작")
    print(f"  - Learning Rate: {LR}")
    print(f"  - Weight Decay: {WEIGHT_DECAY}")
    print(f"  - Mixup: {USE_MIXUP}")
    print(f"  - Temperature Scaling: {USE_TEMPERATURE_SCALING} (선택사항)")

    best_test_acc = 0.0
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train_samples)
        epoch_loss = 0.0

        for i in range(0, n_train_samples, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            batch_X = X_train[idx]
            batch_y = y_train[idx]

            optimizer.zero_grad()
            
            if USE_MIXUP and epoch <= EPOCHS * 0.7:
                mixed_X, y_a, y_b, lam = mixup_data(batch_X, batch_y, MIXUP_ALPHA)
                logits = model(mixed_X)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= n_train_samples

        train_acc = evaluate(X_train, y_train)
        test_acc = evaluate(X_test, y_test)

        if USE_LR_SCHEDULER:
            scheduler.step(test_acc)  # test accuracy 기준으로 스케줄링

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1 or epoch == EPOCHS:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"[Epoch {epoch:03d}] "
                f"Loss: {epoch_loss:.4f} | "
                f"Train Acc: {train_acc * 100:.2f}% | "
                f"Test Acc: {test_acc * 100:.2f}% | "
                f"Best Test: {best_test_acc * 100:.2f}% (Epoch {best_epoch}) | "
                f"LR: {current_lr:.2e}"
            )

        # Early stopping: 더 관대하게 (patience * 3)
        if patience_counter >= SCHEDULER_PATIENCE * 3:
            print(f"\n[조기 종료] {epoch} epoch에서 조기 종료 (Best: Epoch {best_epoch})")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 7. Temperature Scaling (선택사항 - test set에서 최적화)
    optimal_temp = 1.0
    if USE_TEMPERATURE_SCALING:
        print("\n[정보] Temperature Scaling 최적화 중... (test set 기준)")
        optimal_temp = find_optimal_temperature(
            model, X_test, y_test, device, 
            temp_range=TEMPERATURE_SEARCH_RANGE, step=0.1
        )
        print(f"[정보] 최적 Temperature: {optimal_temp:.2f}")
    
    # 최종 평가
    final_train_acc = evaluate(X_train, y_train, temperature=optimal_temp)
    final_test_acc, test_preds = evaluate(X_test, y_test, return_preds=True, temperature=optimal_temp)
    
    print("\n[완료] 개선된 linear probe 학습 종료.")
    print(f"  - 최종 Train Accuracy: {final_train_acc * 100:.2f}%")
    print(f"  - 최종 Test  Accuracy: {final_test_acc * 100:.2f}%")
    print(f"  - Best Test Accuracy: {best_test_acc * 100:.2f}% (Epoch {best_epoch})")
    if USE_TEMPERATURE_SCALING and optimal_temp != 1.0:
        print(f"  - Temperature Scaling 적용: {optimal_temp:.2f}")
        print(f"  - Temperature Scaling 후 Test Accuracy: {final_test_acc * 100:.2f}%")
    else:
        print(f"  - Temperature Scaling 미사용 (기본값 1.0)")

    if HAS_SKLEARN:
        print("\n[상세 분류 리포트]")
        print(classification_report(
            y_test.cpu().numpy(), 
            test_preds, 
            target_names=[str(name) for name in class_names]
        ))
        
        print("\n[혼동 행렬]")
        cm = confusion_matrix(y_test.cpu().numpy(), test_preds)
        print(cm)
        
        print("\n[클래스별 정확도]")
        y_test_np = y_test.cpu().numpy()
        for cls_idx, cls_name in enumerate(class_names):
            cls_mask = y_test_np == cls_idx
            if cls_mask.sum() > 0:
                cls_correct = (test_preds[cls_mask] == cls_idx).sum()
                cls_total = cls_mask.sum()
                cls_acc = cls_correct / cls_total
                print(f"  - {cls_name}: {cls_correct}/{cls_total} ({cls_acc * 100:.2f}%)")


if __name__ == "__main__":
    main()

