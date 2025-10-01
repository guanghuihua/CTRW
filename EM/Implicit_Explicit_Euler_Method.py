import numpy as np
import matplotlib.pyplot as plt

'''
T = 1
N = 1000
h = T / N
t = np.linspace(0, T, N)
x0 = 1

# 初始化数值解数组
X_explicit = np.zeros(N)
X_explicit[0] = x0

X_implicit = np.zeros(N)
X_implicit[0] = x0

# 定义函数 f(x) = -15x
def f_xt(x):
    return -15 * x

# 显式欧拉方法迭代公式
for i in range(N - 1):
    X_explicit[i + 1] = X_explicit[i] + h * f_xt(X_explicit[i])  # 显式欧拉法
    # 隐式欧拉法的递推公式
    X_implicit[i + 1] = X_implicit[i] / (1 + 15 * h)

# 解析解
def X_analytic(x: np.array) -> np.array:
    return np.exp(-15 * x)

X_exact_value = X_analytic(t)

# 绘图
plt.plot(t, X_exact_value, label='The analytic solution of the ODE', linestyle='--')
plt.plot(t, X_explicit, label="The explicit Euler method to the solution", linestyle='-')
plt.plot(t, X_implicit, label="The implicit Euler method to the solution", linestyle='-.')
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.title("Explicit vs Implicit Euler Method for ODE Solution")
plt.legend()
plt.show()
'''

import numpy as np
from scipy.optimize import fsolve

# Define the drift and diffusion functions
mu = lambda x: -x ** 5
sigma = lambda x: x

# Set parameters
Y = 1
N = 2 ** 17

# Implementing the implicit Euler method
for n in range(N):
    # Define the equation whose root we want to find
    equation = lambda x: x - Y - mu(x) / N - sigma(Y) * np.random.randn() / np.sqrt(N)

    # Solve the equation using fsolve (similar to fzero in MATLAB)
    Y = fsolve(equation, Y)[0]

print(Y)  # Final value after N steps

if __name__ == '__main__':
    pass
