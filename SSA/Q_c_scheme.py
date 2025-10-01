from numba import njit
import numpy as np
import math
import random
import matplotlib.pyplot as plt


@njit
def eqn321Fun(xx):
    """定义漂移项U'(x)"""
    return - pow(xx, 3)


@njit
def M_fun(xx):
    """定义扩散项M(x)"""
    return 0.5  # 可以根据实际问题定义扩散项

@njit
def Q_c_jump_rates(Sample_h, h):
    """根据 Q_c 公式计算跳跃率"""
    # 计算当前位置处的漂移项和扩散项
    U_prime = eqn321Fun(Sample_h)
    M = M_fun(Sample_h)

    # 计算相邻网格点的扩散项
    M_right = M_fun(Sample_h + h)
    M_left = M_fun(Sample_h - h)

    # 右跳跃率
    q_right = (M_right + M) / (2 * h * h) * math.exp(- U_prime * h / 2)

    # 左跳跃率
    q_left = (M_left + M) / (2 * h * h) * math.exp(U_prime * h / 2)

    return q_right, q_left


@njit
def SSA_Q_c():
    """使用 Q_c 跳跃率的 SSA"""
    data = np.zeros(N)
    X0 = low + Sp * random.random()
    Sample_h = low + h / 2.0 + round((X0 - low - h / 2.0) / h) * h
    n = 0
    time = 0.0

    while n < valid_number:
        q = np.zeros(2)

        # 使用 Q_c 计算跳跃率
        q[0], q[1] = Q_c_jump_rates(Sample_h, h)

        Lambda = q[0] + q[1]
        rnd1 = random.random()
        rnd2 = random.random()

        tau = -math.log(1 - rnd1) / Lambda
        time += tau

        # 确定跳跃方向
        mu_number = 0
        amu = q[mu_number]
        while amu < rnd2 * Lambda:
            mu_number += 1
            amu += q[mu_number]

        # 更新 Sample_h
        if mu_number == 0:
            Sample_h += h  # 向右跳跃
        elif mu_number == 1:
            Sample_h -= h  # 向左跳跃
        else:
            print("mu_number is wrong")

        # # 如果 Sample_h 超出范围，将其限制在范围内
        # if Sample_h < low:
        #     Sample_h = low
        # elif Sample_h > (low + Sp):
        #     Sample_h = low + Sp

        # 如果 Sample_h 超出范围，采用反射边界条件
        if Sample_h < low:
            Sample_h = low + (low - Sample_h)  # 反射回到内部
        elif Sample_h > (low + Sp):
            Sample_h = low + Sp - (Sample_h - (low + Sp))  # 反射回到内部

        # 记录跳跃后的位置
        x_n = round((Sample_h - low - h / 2.0) / h)
        if x_n >= 0 and x_n < N:
            data[x_n] += 1
            n += 1
        else:
            print("x_n out of scope")

    data /= n * h
    return data


@njit
def l1_norm(u):
    accum = 0.0
    for ii in range(len(u)):
        accum += abs(u[ii]) * h
    return accum


@njit
def KK():
    local_N = 100000000
    upper_bound = 100.0
    lower_bound = 0.0
    local_h = (upper_bound - lower_bound) / local_N
    local_K = 0.0
    for ii in range(local_N):
        local_K += math.exp(-0.25 * pow((ii + 0.5) * local_h, 4))
    return 2 * local_K * local_h


@njit
def my_fun(data):
    for ii in range(loop):
        data[ii, :] = SSA_Q_c()
        print("Loop is", ii)
    return data


# 参数设置
N = 120
loop = 5
valid_number = 1e+7
low = -3.0
Sp = 6
h = Sp / N

# 归一化常数
Z = KK()

# 真值分布
data1 = np.zeros(N)
for i in range(N):
    x = low + h / 2.0 + i * h
    data1[i] = 1 / Z * math.exp(-0.25 * pow(x, 4))

# 运行SSA
data_matrix = np.zeros((loop, N))
output_data = sum(my_fun(data_matrix))
Final_data = output_data / loop

# 计算误差
data2 = Final_data - data1
print("l1 norm of SSA_Qc is %f " % l1_norm(data2))
print("h is %f " % h)

# 绘图
plt.plot(Final_data, label="Qc distribution")
plt.plot(data1, label="True distribution")
plt.legend()
plt.show()
