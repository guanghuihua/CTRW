import numpy as np
import matplotlib.pyplot as plt

# 设置参数
T = 3.0  # 总时间
N = 50000  # 时间步数
dt = T / N  # 每步的时间增量
t = np.linspace(-3, T, N+1)  # 时间网格
sqrt_dt = np.sqrt(dt)  # sqrt(Δt)

# 漂移项和扩散项
def f(x):
    return -x**3

def g(x):
    return np.sqrt(2)

# 欧拉-马鲁雅玛方法
def euler_maruyama(x0, f, g, N, dt, sqrt_dt):
    X = np.zeros(N+1)
    X[0] = x0
    for k in range(N):
        dW = np.random.randn() * sqrt_dt  # ΔWk, 标准正态分布增量
        X[k+1] = X[k] + f(X[k]) * dt + g(X[k]) * dW  # EM 迭代公式
    return X

# 欧拉方法（无扩散项）
def euler_ode(x0, f, N, dt):
    X = np.zeros(N+1)
    X[0] = x0
    for k in range(N):
        X[k+1] = X[k] + f(X[k]) * dt  # 仅计算漂移项
    return X

# 初始条件
# x0 = 1.0  # 初始值
x0 = np.random.rand()

# 使用Euler-Maruyama方法求解SDE
X_em = euler_maruyama(x0, f, g, N, dt, sqrt_dt)

# 使用Euler方法求解无扩散项的ODE
X_ode = euler_ode(x0, f, N, dt)

# 绘图
plt.figure(figsize=(10, 6))

# 绘制Euler-Maruyama解 (包含扩散项)
plt.plot(t, X_em, label="Euler-Maruyama Solution (SDE)", color='blue')

# 绘制Euler解 (不包含扩散项)
plt.plot(t, X_ode, label="Euler Solution (ODE)", color='red', linestyle='--')

# 标题与标签
plt.title(r'Comparison of SDE and ODE: $dX_t = -x^3 dt + \sqrt{2} dW_t$')
plt.xlabel('Time')
plt.ylabel('X(t)')
plt.grid(True)
plt.legend()

# 显示图像
plt.show()
