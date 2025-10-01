import numpy as np
import matplotlib.pyplot as plt

# 定义漂移项 mu 和扩散项 sigma
def mu(x):
    return -x**5

def sigma(x):
    return x

# 驯化欧拉方法 (Tamed Euler)
def tamed_euler(X0, T, N, dt):
    """
    X0: 初始条件
    T: 模拟的总时间
    N: 时间步数
    dt: 时间步长
    """
    # 时间步数
    t = np.linspace(0, T, N+1)
    X = np.zeros(N+1)
    X[0] = X0

    # 模拟布朗运动增量
    dW = np.sqrt(dt) * np.random.randn(N)

    # 使用驯化欧拉方法进行数值模拟
    for n in range(N):
        v = mu(X[n]) * dt
        X[n+1] = X[n] + v / (1 + abs(v)) + sigma(X[n]) * dW[n]

    return t, X

# 数值实验参数设置
T = 10.0       # 总时间
N = 2**10      # 时间步数
dt = T / N    # 每步的时间长度
X0 = 1        # 初始条件 X_0

# 使用驯化欧拉方法进行模拟
t, X = tamed_euler(X0, T, N, dt)

# 结果可视化
plt.plot(t, X, label='Tamed Euler Approximation')
plt.xlabel('Time')
plt.ylabel('X_t')
plt.title('Tamed Euler Approximation for SDE')
plt.legend()
plt.grid(True)
plt.show()

if __name__ == '__main__':
    pass