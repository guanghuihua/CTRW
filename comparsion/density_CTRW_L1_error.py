from numba import njit, prange
import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt
import numba
numba.config.NUMBA_DEFAULT_NUM_THREADS = 8


@njit(fastmath=True)
def eqn321Fun(xx):
    # 漂移项 mu(x) = -x^3
    return - pow(xx, 3)


@njit(fastmath=True)
def SSA(N, h, valid_number, low, Sp):
    """
    空间离散 CTRW / SSA 方法，模拟一维立方振子在网格上的平稳分布。
    返回的是区间 [low, low+Sp] 上 N 个网格点对应的“概率密度近似”。
    """
    data = np.zeros(N)
    X0 = low + Sp * np.random.random()  # 初始点：区间上均匀分布
    Sample_h = low + h / 2.0 + round((X0 - low - h / 2.0) / h) * h
    n = 0
    time_now = 0.0

    while n < valid_number:
        q = np.zeros(2)
        tmp = eqn321Fun(Sample_h)
        # M 是“补偿项”，保证跳率非负
        M = 0.5 * max(2 - abs(tmp) * h, 0.0)

        q[0] = max(tmp, 0.0) / h + M / (h * h)
        q[1] = -min(tmp, 0.0) / h + M / (h * h)

        Lambda = q[0] + q[1]
        rnd1 = np.random.random()
        rnd2 = np.random.random()

        # 指数等待时间
        tau = -math.log(1 - rnd1) / Lambda
        time_now += tau

        # 选择跳向左还是向右
        mu_number = 0
        amu = q[mu_number]
        while amu < rnd2 * Lambda:
            mu_number += 1
            amu += q[mu_number]

        if mu_number == 0:
            Sample_h = Sample_h + h
        elif mu_number == 1:
            Sample_h = Sample_h - h
        else:
            raise ValueError("mu_number is wrong")

        # 映射回网格索引
        x_n = round((Sample_h - low - h / 2.0) / h)
        if 0 <= x_n < N:
            data[x_n] += 1
            n += 1
        else:
            # 跳出模拟区间的话直接报错（论文中一般会控制 Sp 足够大）
            raise ValueError("x_n out of scope")

    # 归一化：得到区间 [low, low+Sp] 上的“概率密度”近似
    data /= (n * h)
    return data


@njit(fastmath=True)
def l1_norm(u, h):
    """
    计算向量 u 在网格步长 h 下的 L1 范数：
        ||u||_{L1} ≈ sum_i |u_i| * h
    在这里我们会把 u 取成“数值平稳密度 - 精确平稳密度”的差。
    """
    accumulate = 0.0
    for ii in range(len(u)):
        accumulate += abs(u[ii]) * h
    return accumulate


@njit(fastmath=True, parallel=True)
def KK():
    """
    数值近似归一化常数：
        Z = ∫_R exp(-x^4/4) dx
      ≈ 2 * ∫_0^{100} exp(-x^4/4) dx
    """
    local_N = 100000000
    upper_bound = 100.0
    lower_bound = 0.0
    local_h = (upper_bound - lower_bound) / local_N
    local_K = 0.0
    for ii in prange(local_N):
        local_K += math.exp(-0.25 * pow((ii + 0.5) * local_h, 4))
    return 2 * local_K * local_h


@njit(fastmath=True, parallel=True)
def my_fun(loop, N, h, valid_number, low, Sp):
    """
    重复做 loop 次 SSA 模拟，并把每次得到的平稳分布存成一行。
    """
    data = np.zeros((loop, N))
    for ii in prange(loop):
        data[ii, :] = SSA(N, h, valid_number, low, Sp)
    return data


if __name__ == "__main__":
    start = time.time()

    # 网格参数
    N = 120
    # loop = 1000   # 论文中可以设置更大做平均
    loop = 5
    valid_number = int(1e7)

    low = -3.0
    Sp = 6.0
    h = Sp / N

    # 精确平稳密度：nu(x) = Z^{-1} * exp(-x**4 / 4)
    Z = KK()
    data1 = np.zeros(N)
    for i in range(N):
        x = low + h / 2.0 + i * h
        data1[i] = 1.0 / Z * math.exp(-0.25 * pow(x, 4))

    # 数值平稳分布（SSA / CTRW 方法）
    data_matrix = my_fun(loop, N, h, valid_number, low, Sp)
    output_data = np.sum(data_matrix, axis=0)
    Final_data = output_data / loop

    # 计算 L1 误差： ||rho_num - rho_true||_{L1}
    diff = Final_data - data1
    l1_err = l1_norm(diff, h)
    print("l1 norm of SSA_Qu (sim vs exact) = %f" % l1_err)
    print("h = %f" % h)

    end = time.time()
    print("Total time (with compilation) = %s s" % (end - start))

    # 画出数值平稳分布与精确分布
    plt.plot(Final_data, label="SSA/Qu distribution")
    plt.plot(data1, label="True stationary distribution")
    plt.legend()
    plt.show()
