import time
import os
import torch, torch.nn as nn, torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# ===== 0) 可复现性 =====
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ===== 1) 数据 =====
X, y = load_iris(return_X_y=True)            # X: (150, 4), y: 0/1/2
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=SEED, test_size=0.2
)

# ===== 2) 设备与版本信息 =====
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
print("===== Runtime Info =====")
print("Torch:", torch.__version__, "| CUDA build:", torch.version.cuda)
print("CUDA available:", use_cuda, "| Device:", device_name)

# ===== 3) 转张量 =====
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.long,   device=device)
X_test  = torch.tensor(X_test,  dtype=torch.float32, device=device)
y_test  = torch.tensor(y_test,  dtype=torch.long,    device=device)

# ===== 4) 模型/优化器/损失 =====
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 3)
).to(device)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}, "
      f"Features: {X_train.shape[1]}, Classes: {len(torch.unique(y_train))}")
print("Trainable params:", count_params(model))

opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# ===== 5) 训练超参 =====
EPOCHS = 200
PRINT_EVERY = 20  # 每多少轮打印一次

# （可选）使用小批次训练，跟实际更接近；Iris 很小，全量也行
BATCH_SIZE = 32
indices = torch.arange(X_train.shape[0], device=device)

def train_one_epoch():
    model.train()
    # 随机打乱
    perm = indices[torch.randperm(indices.numel())]
    total_loss, total_correct, total_seen = 0.0, 0, 0
    for i in range(0, perm.numel(), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        xb, yb = X_train[idx], y_train[idx]
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

        total_loss += loss.item() * yb.size(0)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_seen += yb.size(0)
    return total_loss / total_seen, total_correct / total_seen

def eval_accuracy(Xt, yt):
    model.eval()
    with torch.no_grad():
        logits = model(Xt)
        acc = (logits.argmax(1) == yt).float().mean().item()
    return acc

# ===== 6) 训练 =====
t0 = time.time()
last_log_time = t0
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch()
    if epoch % PRINT_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
        test_acc = eval_accuracy(X_test, y_test)
        now = time.time()
        print(f"[Epoch {epoch:3d}/{EPOCHS}] "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"test_acc={test_acc:.4f}  (+{now-last_log_time:.2f}s)")
        last_log_time = now
t1 = time.time()

# ===== 7) 最终评估与总结 =====
final_test_acc = eval_accuracy(X_test, y_test)
print("===== Summary =====")
print(f"Used GPU: {use_cuda}  | Device: {device_name}")
print(f"Epochs trained: {EPOCHS}")
print(f"Final Test Accuracy: {final_test_acc:.4f}")
print(f"Total training time: {t1 - t0:.2f}s")

