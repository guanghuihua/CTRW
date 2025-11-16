# ---- 放在 import matplotlib.pyplot as plt 之前 ----
import os
import matplotlib
from matplotlib import rcParams, font_manager

def use_chinese_font():
    # 常见中文字体候选（按顺序尝试）
    candidates = [
        (r"C:\Windows\Fonts\msyh.ttc", "Microsoft YaHei"),  # 微软雅黑
        (r"C:\Windows\Fonts\simhei.ttf", "SimHei"),         # 黑体
        (r"C:\Windows\Fonts\simsun.ttc", "SimSun"),         # 宋体
        # 如果你安装了 Noto/思源，也可以加上：
        (r"C:\Windows\Fonts\NotoSansCJK-Regular.ttc", "Noto Sans CJK SC"),
        (r"C:\Windows\Fonts\SourceHanSansCN-Regular.otf", "Source Han Sans CN"),
    ]
    for path, name in candidates:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            rcParams["font.family"] = name
            break
    # 解决负号无法显示为中文字体的问题
    rcParams["axes.unicode_minus"] = False

use_chinese_font()


import numpy as np
import math
import dataclasses
import time
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# =========================
# 1) 模型与参数
# =========================
@dataclasses.dataclass
class Params:
    eps: float = 0.02               # 慢变量尺度
    a: float = 1.0                  # canard 偏置
    sigma_x: float = 0.25           # 只给快变量加噪声
    sigma_y: float = 0.0            # 若要两方向独立噪声，取 sqrt(eps)*sigma_y
    # 计算域（反射边界）
    x_min: float = -2.2
    x_max: float =  2.0
    y_min: float = -1.5
    y_max: float =  1.8
    # committor 目标集合
    xA: float = 0.0                 # A = {x <= xA} 设 q=0
    xB: float = 1.2                 # B = {x >= xB} 设 q=1

def S0(x):  # 临界流形（慢流形骨架）
    return x - x**3/3.0

def drift(x, y, p: Params):
    return (x - x**3/3.0 - y, p.eps*(x - p.a))

def sigmas(p: Params):
    return p.sigma_x, math.sqrt(p.eps)*p.sigma_y

# =========================
# 2) 空间离散：组装 Q 并解 Q q = 0
# =========================
@dataclasses.dataclass
class Grid:
    nx: int
    ny: int
    x: np.ndarray
    y: np.ndarray
    hx: float
    hy: float

def make_grid(p: Params, nx: int, ny: int) -> Grid:
    x = np.linspace(p.x_min, p.x_max, nx)
    y = np.linspace(p.y_min, p.y_max, ny)
    return Grid(nx, ny, x, y, (p.x_max-p.x_min)/(nx-1), (p.y_max-p.y_min)/(ny-1))

def assemble_Q(grid: Grid, p: Params) -> csr_matrix:
    """基于“可实现跳率”的 5 点离散，反射边界=出界方向率置 0"""
    nx, ny, x, y, hx, hy = grid.nx, grid.ny, grid.x, grid.y, grid.hx, grid.hy
    sx, sy = sigmas(p)
    N = nx*ny
    Q = lil_matrix((N, N))
    def idx(i,j): return j*nx + i

    for j in range(ny):
        yj = y[j]
        for i in range(nx):
            xi = x[i]
            mux, muy = drift(xi, yj, p)

            # 扩散补偿项（保非负；对应改进的 \tilde Q_u 思想）
            Mx = max(0.0, 0.5*(sx**2 - abs(mux)*hx))
            My = max(0.0, 0.5*(sy**2 - abs(muy)*hy))

            qx_p = (max(mux,0.0)/hx + Mx/(hx*hx)) if i<nx-1 else 0.0
            qx_m = (max(-mux,0.0)/hx + Mx/(hx*hx)) if i>0    else 0.0
            qy_p = (max(muy,0.0)/hy + My/(hy*hy)) if j<ny-1 else 0.0
            qy_m = (max(-muy,0.0)/hy + My/(hy*hy)) if j>0    else 0.0
            lam  = qx_p + qx_m + qy_p + qy_m
            k = idx(i,j)
            Q[k, k] = -lam
            if i<nx-1: Q[k, idx(i+1,j)] = qx_p
            if i>0:    Q[k, idx(i-1,j)] = qx_m
            if j<ny-1: Q[k, idx(i,j+1)] = qy_p
            if j>0:    Q[k, idx(i,j-1)] = qy_m

    return Q.tocsr()

def solve_committor(grid: Grid, p: Params, Q: csr_matrix) -> np.ndarray:
    """在 A: x<=xA, B: x>=xB 上施加 Dirichlet 边界，解 Q q = 0"""
    nx, ny, x = grid.nx, grid.ny, grid.x
    N = nx*ny
    def idx(i,j): return j*nx + i

    A = np.zeros(N, dtype=bool)
    B = np.zeros(N, dtype=bool)
    for j in range(ny):
        for i in range(nx):
            if x[i] <= p.xA: A[idx(i,j)] = True
            if x[i] >= p.xB: B[idx(i,j)] = True

    fix = A | B
    g = np.zeros(N); g[B] = 1.0
    Qii = Q[~fix][:, ~fix]
    rhs = - Q[~fix][:, fix] @ g[fix]

    q_int = spsolve(Qii, rhs)
    q = g.copy(); q[~fix] = q_int
    return q.reshape(grid.ny, grid.nx)

def residual_inf(Q: csr_matrix, q: np.ndarray, grid: Grid, p: Params) -> float:
    """||Q q||_inf（只统计内点，以观测离散谐解残差）"""
    nx, ny, x = grid.nx, grid.ny, grid.x
    N = nx*ny
    qv = q.reshape(N)
    # 内点掩码（排除 A/B）
    A = np.zeros(N, dtype=bool)
    B = np.zeros(N, dtype=bool)
    def idx(i,j): return j*nx + i
    for j in range(ny):
        for i in range(nx):
            if x[i] <= p.xA: A[idx(i,j)] = True
            if x[i] >= p.xB: B[idx(i,j)] = True
    interior = ~(A | B)
    r = Q @ qv
    return float(np.max(np.abs(r[interior])))

# =========================
# 3) 时间离散（对照）：tamed EM 估计 q(z)
# =========================
def tamed_em_committor(z0, p: Params, dt: float, Tcap: float, rng: np.random.Generator) -> int:
    """从 z0 出发，返回是否先达 B(=1) 或先达 A(=0)；超时按 0.5 也可，但此处直接按未命中视作 0"""
    x, y = z0
    sx, sy = sigmas(p)
    t = 0.0
    while t < Tcap and (x > p.xA) and (x < p.xB):
        mux, muy = drift(x, y, p)
        # tamed Euler
        denom = 1.0 + dt*(abs(mux)+abs(muy))
        x += (dt*mux)/denom + sx*math.sqrt(dt)*rng.standard_normal()
        y += (dt*muy)/denom + sy*math.sqrt(dt)*rng.standard_normal()
        # 反射外边界
        if x < p.x_min: x = p.x_min + (p.x_min - x)
        if x > p.x_max: x = p.x_max - (x - p.x_max)
        if y < p.y_min: y = p.y_min + (p.y_min - y)
        if y > p.y_max: y = p.y_max - (y - p.y_max)
        t += dt
    return 1 if x >= p.xB else 0

def mc_committor_at_points(points, p: Params, dt: float, Tcap: float, M: int, seed=42):
    rng = np.random.default_rng(seed)
    est = []
    for (x0,y0) in points:
        hits = 0
        for _ in range(M):
            hits += tamed_em_committor((x0,y0), p, dt, Tcap, rng)
        est.append(hits/M)
    return np.array(est)

# =========================
# 4) 实验驱动
# =========================
def run_all():
    p = Params()
    # —— 参考解（细网格；各向异性建议：ny ~ O(eps * nx) 的常数倍）
    grid_ref = make_grid(p, nx=321, ny=201)
    Q_ref = assemble_Q(grid_ref, p)
    t0 = time.time()
    q_ref = solve_committor(grid_ref, p, Q_ref)
    t1 = time.time()
    res_ref = residual_inf(Q_ref, q_ref, grid_ref, p)
    print(f"[REF] nx={grid_ref.nx}, ny={grid_ref.ny}, solve {t1-t0:.2f}s, ||Qq||_inf≈{res_ref:.2e}")

    # —— 图1：热力图 + q=0.5 + S0 + 竖线
    XX, YY = np.meshgrid(grid_ref.x, grid_ref.y)
    plt.figure(figsize=(7.2,5.2))
    plt.imshow(q_ref, origin='lower',
               extent=[p.x_min,p.x_max,p.y_min,p.y_max], aspect='auto')
    cs = plt.contour(XX, YY, q_ref, levels=[0.5], linewidths=1.5)
    xs = np.linspace(p.x_min, p.x_max, 800)
    plt.plot(xs, S0(xs))                        # 临界流形
    plt.axvline(1.0)                            # 折叠位置
    plt.axvline(p.xA); plt.axvline(p.xB)        # A/B
    plt.xlabel('x'); plt.ylabel('y')
    plt.title('Committor q(x,y)（空间离散参考解）')
    plt.tight_layout()
    plt.savefig('committor_canard_heatmap.png', dpi=160)

    # —— 图2：空间离散 误差–代价（与参考解对比）
    levels = [(161,105),(241,161),(321,201)]   # 可再加细
    errs, works = [], []
    for nx,ny in levels:
        g = make_grid(p, nx, ny)
        Q = assemble_Q(g, p)
        q = solve_committor(g, p, Q)
        # 误差（将粗网格点“最近邻”映射到参考网格）
        # 简化：只在粗网格节点上，用参考 q_ref 双线性插值
        from scipy.interpolate import RectBivariateSpline
        interp = RectBivariateSpline(grid_ref.y, grid_ref.x, q_ref)
        q_on_ref = interp(g.y, g.x)             # 形状 (ny,nx)
        err = float(np.max(np.abs(q_on_ref - q)))
        errs.append(err)
        # 代价“work”：nnz(Q) 近似（也可乘以迭代步数；此处直接用消元求解）
        works.append(Q.nnz)
        print(f"[SPACE] nx={nx}, ny={ny}, Linf_err≈{err:.3e}, work≈{works[-1]}")

    plt.figure()
    plt.loglog(works, errs, 'o-')
    plt.xlabel('work (≈ nnz(Q))'); plt.ylabel('L∞ error vs fine reference')
    plt.title('空间离散：误差–代价（committor）')
    plt.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.savefig('space_error_vs_work.png', dpi=160)

    # —— 图3：时间离散 误差–代价（选若干初值点对比）
    # 选 9 个采样点（避免在 A/B 上）
    pts = []
    xlist = np.linspace(-1.2, 1.1, 3)
    ylist = np.linspace(-0.8, 0.8, 3)
    for xv in xlist:
        for yv in ylist:
            pts.append((float(xv), float(yv)))
    # 参考值：用 q_ref 双线性插值获得
    from scipy.interpolate import RectBivariateSpline
    interp_ref = RectBivariateSpline(grid_ref.y, grid_ref.x, q_ref)
    q_true = np.array([float(interp_ref(y0,x0)) for (x0,y0) in pts])

    dt_list = [2**-6, 2**-7, 2**-8]
    M_list  = [2_000, 8_000, 32_000]
    errs_t, works_t, labels_t = [], [], []
    Tcap = 80.0
    for dt in dt_list:
        for M in M_list:
            est = mc_committor_at_points(pts, p, dt, Tcap, M, seed=123)
            err = float(np.max(np.abs(est - q_true)))   # Linf over sample points
            work = int(Tcap/dt)*M*len(pts)              # 步数×路径数×点数（近似口径）
            errs_t.append(err); works_t.append(work); labels_t.append((dt, M))
            print(f"[TIME] dt={dt:g}, M={M}, Linf_err≈{err:.3e}, work≈{work:.2e}")

    plt.figure()
    plt.loglog(works_t, errs_t, 's', alpha=0.85)
    for w,e,(dt,M) in zip(works_t, errs_t, labels_t):
        plt.text(w, e*1.08, f"dt={dt:g},M={M}", fontsize=8)
    plt.xlabel('work (≈ steps×paths×points)'); plt.ylabel('L∞ error @ sample points')
    plt.title('时间离散：误差–代价（committor）')
    plt.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.savefig('time_error_vs_work.png', dpi=160)

    print("Done. Saved:",
          "committor_canard_heatmap.png,",
          "space_error_vs_work.png,",
          "time_error_vs_work.png")

if __name__ == "__main__":
    run_all()
