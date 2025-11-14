import numpy as np

# 逐行输入  直接抄吧浩然  这个不太懂 没怎么验证 光看了看样例能过
n, max_iter, lr, lam, tol = input().split()
n        = int(n)
max_iter = int(max_iter)
lr       = float(lr)
lam      = float(lam)
tol      = float(tol)

X = np.empty((n, 3), dtype=float)
y = np.empty(n, dtype=float)
for i in range(n):
    a, inc, t, lab = map(float, input().split())
    X[i] = [a, inc, t]
    y[i] = lab

m = int(input())
Xte = np.empty((m, 3), dtype=float)
for i in range(m):
    a, inc, t = map(float, input().split())
    Xte[i] = [a, inc, t]

# 逻辑回归（GD + L2）
w = np.zeros(3, dtype=float)
b = 0.0
prev_loss = float("inf")
n_inv = 1.0 / n

for _ in range(max_iter):
    z = X @ w + b
    p = 1.0 / (1.0 + np.exp(-z))  # sigmoid
    # 稳定交叉熵 + L2(w)，不惩罚b
    loss = (np.logaddexp(0.0, z) - y * z).mean() + (lam * 0.5 * n_inv) * (w @ w)
    if abs(prev_loss - loss) < tol:
        break
    prev_loss = loss
    g = (p - y)                       # shape (n,)
    grad_w = (X.T @ g) * n_inv + (lam * n_inv) * w
    grad_b = g.mean()
    w -= lr * grad_w
    b -= lr * grad_b

# 预测
z_te = Xte @ w + b
p_te = 1.0 / (1.0 + np.exp(-z_te))
pred = (p_te >= 0.5).astype(int)

# 输出
for pi, pr in zip(pred, p_te):
    print(f"{pi} {pr:.4f}")
