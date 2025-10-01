import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# van der Pol 振荡器模型
def van_der_pol(t, z, mu):
    x, y = z
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]

# 设定参数
mu = 5  # 非线性项的参数
z0 = [1.0, 0.0]  # 初始条件
t_span = (0, 100)  # 时间范围
t_eval = np.linspace(t_span[0], t_span[1], 10000)  # 采样时间点

# 求解微分方程
solution = solve_ivp(van_der_pol, t_span, z0, args=(mu,), t_eval=t_eval)

# 绘制相图
plt.figure(figsize=(8, 6))
plt.plot(solution.y[0], solution.y[1], label=f"mu={mu}")
plt.title("Van der Pol Oscillator - Phase Portrait")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()



# Consider stimulate the Canard phenomenon by adding a small perturbation with Euler-Maruyama method
def euler_maruyama(vdp, z0, t_span, dt, mu, noise_std=0.01):
    num_steps = int((t_span[1] - t_span[0]) / dt)
    z = np.zeros((2, num_steps))
    z[:, 0] = z0
    t = np.linspace(t_span[0], t_span[1], num_steps)

    for i in range(1, num_steps):
        dz = vdp(t[i-1], z[:, i-1], mu) * dt
        noise = np.random.normal(0, noise_std, size=2) * np.sqrt(dt)
        z[:, i] = z[:, i-1] + dz + noise

    return t, z     
