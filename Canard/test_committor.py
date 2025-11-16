# Committor 可视化（随机 Canard 系统，空间离散解 L q = 0）
# 说明：
# - 2D 网格上用“可实现跳率”（近似改进的 \tilde Q_u 思想）搭建局部方程，
#   对内点满足：-(sum rates) q_ij + Σ rates * q_neighbor = 0；
#   用 Gauss–Seidel + 松弛迭代（SOR）求解。
# - A 集：x <= xA，B 集：x >= xB；外边界取反射（通过去掉出界率实现）。
# - 画出 q(x,y) 热力图，叠加 q=0.5 等值线、临界流形 S0 以及折叠线 x=1、A/B 边界。

import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 参数与场 ----------------------
eps = 0.02
a = 1.0
sigma_x = 0.25
sigma_y = 0.0  # 可改为 np.sqrt(eps)*0.15 试试独立慢噪声

def S0(x):
    return x - x**3/3.0

def mu(x, y):
    fx = x - x**3/3.0 - y
    fy = eps * (x - a)
    return fx, fy

# ---------------------- 网格与集合 ----------------------
x_min, x_max, nx = -2.2, 2.0, 141
y_min, y_max, ny = -1.5, 1.8, 101
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
hx = (x_max - x_min) / (nx - 1)
hy = (y_max - y_min) / (ny - 1)

# 各向异性建议：hy ~ c * eps * hx（这里只用均匀网格以便演示；若要更精细，可把 ny 提高）

# 目标集合（committor 的 Dirichlet 边界）
xA, xB = 0.0, 1.2
# A: x <= xA, q=0；B: x >= xB, q=1
A_mask = np.zeros((ny, nx), dtype=bool)
B_mask = np.zeros((ny, nx), dtype=bool)
for j in range(ny):
    for i in range(nx):
        if x[i] <= xA: A_mask[j, i] = True
        if x[i] >= xB: B_mask[j, i] = True

# ---------------------- 迭代求解器（基于跳率） ----------------------
def committor_solver(max_iter=20000, tol=1e-5, omega=1.6, verbose=False):
    # 初值：从 0..1 的线性插值
    q = np.zeros((ny, nx), dtype=float)
    for j in range(ny):
        q[j, :] = (x - xA) / max(1e-8, (xB - xA))
    q = np.clip(q, 0.0, 1.0)
    # 强制边界：A=0, B=1
    q[A_mask] = 0.0
    q[B_mask] = 1.0

    # 迭代
    for it in range(max_iter):
        max_res = 0.0
        # Gauss-Seidel 扫描
        for j in range(ny):
            for i in range(nx):
                # Dirichlet 边界点跳过
                if A_mask[j, i] or B_mask[j, i]:
                    continue

                xi, yj = x[i], y[j]
                mux, muy = mu(xi, yj)

                # 扩散强度
                sigx = sigma_x
                sigy = sigma_y

                # 反射边界：出界方向的率置 0；邻点索引夹紧
                iL = i-1 if i-1 >= 0    else i
                iR = i+1 if i+1 < nx    else i
                jD = j-1 if j-1 >= 0    else j
                jU = j+1 if j+1 < ny    else j

                # “改进 Qu~”风格的可实现率（补偿扩散项保持非负）
                Mx_comp = max(0.0, 0.5*(sigx**2 - abs(mux)*hx))
                My_comp = max(0.0, 0.5*(sigy**2 - abs(muy)*hy))

                qx_p = (max(mux, 0.0)/hx) + Mx_comp/(hx*hx) if (iR != i) else 0.0
                qx_m = (max(-mux, 0.0)/hx) + Mx_comp/(hx*hx) if (iL != i) else 0.0
                qy_p = (max(muy, 0.0)/hy) + My_comp/(hy*hy) if (jU != j) else 0.0
                qy_m = (max(-muy, 0.0)/hy) + My_comp/(hy*hy) if (jD != j) else 0.0

                S = qx_p + qx_m + qy_p + qy_m
                if S <= 0.0:
                    # 零速率：保持原值
                    continue

                q_new = (qx_m * q[j, iL] + qx_p * q[j, iR] + qy_m * q[jD, i] + qy_p * q[jU, i]) / S
                # SOR 更新
                q_old = q[j, i]
                q[j, i] = (1.0 - omega) * q_old + omega * q_new

                # 残差（局部）
                res = abs(-(S) * q[j, i] + (qx_m * q[j, iL] + qx_p * q[j, iR] + qy_m * q[jD, i] + qy_p * q[jU, i]))
                if res > max_res:
                    max_res = res

        # 强制 Dirichlet（避免数值漂移）
        q[A_mask] = 0.0
        q[B_mask] = 1.0

        if verbose and it % 200 == 0:
            print(f"iter {it}: max residual ~ {max_res:.3e}")
        if max_res < tol:
            if verbose:
                print(f"converged at iter={it}, residual={max_res:.3e}")
            break

    return q

q = committor_solver(max_iter=12000, tol=5e-5, omega=1.6, verbose=False)

# ---------------------- 可视化 ----------------------
XX, YY = np.meshgrid(x, y)

fig, ax = plt.subplots(figsize=(7.2, 5.4))

# 热力图（默认色图），注意：不手动指定颜色，遵循系统要求
im = ax.imshow(q, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='auto')

# q=0.5 等值线
cs = ax.contour(XX, YY, q, levels=[0.5])

# 临界流形 S0（仅绘可见范围）
xs = np.linspace(x_min, x_max, 800)
ys = S0(xs)
ax.plot(xs, ys)

# 折叠线与 A/B 边界
ax.axvline(1.0)      # fold x=1
ax.axvline(xA)       # A 边界
ax.axvline(xB)       # B 边界

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('随机 Canard：committor q(x,y)（空间离散解）')

plt.tight_layout()
plt.show()