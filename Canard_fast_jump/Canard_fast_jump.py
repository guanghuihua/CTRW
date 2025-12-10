import numpy as np
import math
import random

# ==========================
# 1. Canard 系统参数与漂移
# ==========================

delta = 0.1
a = 1 - delta/8 - 3*delta**2/32 - 173*delta**3/1024 - 0.01

sigma_x = 0.0       # 噪声只加在慢变量 x2 上
sigma_y = 0.08

def mu_vec(x):
    """Canard 系统漂移向量 mu(x) = (mu1, mu2)."""
    x1, x2 = x
    mu1 = (x1 + x2 - x1**3 / 3.0) / delta
    mu2 = a - x1
    return np.array([mu1, mu2])

def sigma_vec(x):
    """扩散系数向量 (sigma_x, sigma_y)."""
    return np.array([sigma_x, sigma_y])


# 初始点取在左侧慢流形上
x0 = np.array([-1.5, (-1.5**3)/3.0 - (-1.5)])

# 快跳阈值与观察时间窗
x_th = 1.5
T_obs = 3.0   # 对这个缩放，快跳时间大约在 0.5 左右，3 已足够覆盖

# ==================================
# 2. 时间离散：Euler / tamed EM 等
# ==================================

def step_em(x, dt, method="em"):
    """单步 Euler / tamed EM."""
    mu = mu_vec(x)
    sig = sigma_vec(x)
    dW = np.sqrt(dt) * np.random.normal(size=2)
    if method == "tamed":
        # 简单的标量驯服：mu / (1 + |mu| dt)
        mu_eff = mu / (1.0 + np.linalg.norm(mu) * dt)
    else:
        mu_eff = mu
    return x + mu_eff * dt + sig * dW

def first_jump_time_em(dt, T, x_init, method="em", x_threshold=1.5):
    """给定时间步长 dt，模拟一条路径的首次快跳时间。"""
    n_steps = int(T / dt)
    x = np.array(x_init, dtype=float)
    t = 0.0
    for k in range(n_steps):
        x = step_em(x, dt, method=method)
        t += dt
        if x[0] >= x_threshold:
            return t, True
    return T, False

def stats_jump_em(dt, T, x_init, method="em", x_threshold=1.5, n_paths=1000):
    """重复模拟 n_paths 条路径，统计快跳概率与平均快跳时间。"""
    hit_times = []
    hits = 0
    for _ in range(n_paths):
        t_hit, hit = first_jump_time_em(dt, T, x_init, method=method,
                                        x_threshold=x_threshold)
        if hit:
            hits += 1
            hit_times.append(t_hit)
    p_jump = hits / n_paths
    m_jump = np.mean(hit_times) if hits > 0 else T
    return p_jump, m_jump


# =================================
# 3. 空间离散：二维 CTRW (Q_u)
# =================================

# 计算域与网格
x_min, x_max = -2.5, 2.5
y_min, y_max = -1.0, 3.0
hx = 0.05
hy = 0.05

x_grid = np.arange(x_min, x_max + 1e-12, hx)
y_grid = np.arange(y_min, y_max + 1e-12, hy)
Nx, Ny = len(x_grid), len(y_grid)

# 对角扩散矩阵 M = diag(M11, M22)
M11 = sigma_x**2 / 2.0
M22 = sigma_y**2 / 2.0

def simulate_ctrw_one(T, x_init, x_threshold=1.5, max_jumps=10**6):
    """
    基于 Q_u 的 CTRW/SSA，模拟一条路径的首次快跳时间。
    """
    # 初值投影到最近网格点
    i = int(round((x_init[0] - x_min) / hx))
    j = int(round((x_init[1] - y_min) / hy))
    i = max(0, min(Nx - 1, i))
    j = max(0, min(Ny - 1, j))

    t = 0.0
    while t < T and max_jumps > 0:
        x1 = x_grid[i]
        x2 = y_grid[j]
        if x1 >= x_threshold:
            return t, True

        mu = mu_vec((x1, x2))
        mu1, mu2 = mu[0], mu[1]

        # Q_u 格式在 x, y 两个方向的跃迁率
        rates = []
        neighbors = []

        # x+
        if i + 1 < Nx:
            q_x_plus = max(mu1, 0.0) / hx + M11 / (hx * hx)
            if q_x_plus > 0:
                rates.append(q_x_plus)
                neighbors.append((i + 1, j))

        # x-
        if i - 1 >= 0:
            q_x_minus = max(-mu1, 0.0) / hx + M11 / (hx * hx)
            if q_x_minus > 0:
                rates.append(q_x_minus)
                neighbors.append((i - 1, j))

        # y+
        if j + 1 < Ny:
            q_y_plus = max(mu2, 0.0) / hy + M22 / (hy * hy)
            if q_y_plus > 0:
                rates.append(q_y_plus)
                neighbors.append((i, j + 1))

        # y-
        if j - 1 >= 0:
            q_y_minus = max(-mu2, 0.0) / hy + M22 / (hy * hy)
            if q_y_minus > 0:
                rates.append(q_y_minus)
                neighbors.append((i, j - 1))

        if not rates:
            # 没有出度，视为被困住
            return T, False

        lambda_tot = sum(rates)

        # 抽样停留时间：Exp(lambda_tot)
        u = random.random()
        tau = -math.log(u) / lambda_tot
        t += tau
        if t >= T:
            return T, False

        # 抽样跳跃方向
        u2 = random.random() * lambda_tot
        cum = 0.0
        for rate, neigh in zip(rates, neighbors):
            cum += rate
            if u2 <= cum:
                i, j = neigh
                break

        max_jumps -= 1

    return T, False

def stats_jump_ctrw(T, x_init, x_threshold=1.5, n_paths=1000):
    """
    CTRW/SSA 下重复模拟，统计快跳概率与平均快跳时间。
    """
    hit_times = []
    hits = 0
    for _ in range(n_paths):
        t_hit, hit = simulate_ctrw_one(T, x_init,
                                       x_threshold=x_threshold)
        if hit:
            hits += 1
            hit_times.append(t_hit)
    p_jump = hits / n_paths
    m_jump = np.mean(hit_times) if hits > 0 else T
    return p_jump, m_jump


# ============================
# 4. 运行一次对比示例
# ============================

if __name__ == "__main__":
    np.random.seed(1234)
    random.seed(1234)

    # 参考：CTRW (空间离散) 的快跳统计
    p_ctrw, m_ctrw = stats_jump_ctrw(T_obs, x0, x_threshold=x_th,
                                     n_paths=500)
    print("CTRW:  p_jump = {:.3f},  mean tau = {:.3f}".format(p_ctrw, m_ctrw))

    # 时间离散：不同步长的对比
    for dt in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
        p_em, m_em = stats_jump_em(dt, T_obs, x0, method="tamed",
                                   x_threshold=x_th, n_paths=500)
        print("tamed EM, dt = {:g}:  p_jump = {:.3f},  mean tau = {:.3f}".format(
            dt, p_em, m_em
        ))
