import numpy as np

# 初始分子数量
num_A = 10
num_B = 5
num_C = 0

# 反应速率常数
k1 = 0.1  # A -> 2A
k2 = 0.05  # A + B -> C

# 模拟参数
t_max = 100
t = 0

# 分子数量记录
time_points = [t]
A_points = [num_A]
B_points = [num_B]
C_points = [num_C]

while t < t_max:
    # 计算每种反应的反应速率
    a1 = k1 * num_A
    a2 = k2 * num_A * num_B

    # 计算总反应速率
    a0 = a1 + a2

    # 检查是否还有反应可以发生
    if a0 == 0:
        break

        # 抽取下一个反应的时间间隔
    r1 = np.random.rand()
    tau = (1.0 / a0) * np.log(1.0 / r1)

    # 更新时间
    t += tau

    # 抽取反应类型
    r2 = np.random.rand()
    if r2 * a0 < a1:
        # 反应1发生：A -> 2A
        num_A += 1
    else:
        # 反应2发生：A + B -> C
        num_A -= 1
        num_B -= 1
        num_C += 1

        # 检查分子数量是否非负
        if num_A < 0 or num_B < 0:
            print("Error: Negative molecule count. Simulation stopped.")
            break

            # 记录时间和分子数量
    time_points.append(t)
    A_points.append(num_A)
    B_points.append(num_B)
    C_points.append(num_C)

# 可视化结果（可选）
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(time_points, A_points, label='A', marker='o', linestyle='-')
plt.plot(time_points, B_points, label='B', marker='s', linestyle='-')
plt.plot(time_points, C_points, label='C', marker='^', linestyle='-')
plt.xlabel('Time')
plt.ylabel('Molecule Count')
plt.title('Gillespie Algorithm Simulation with Multiple Reactions and Species')
plt.legend()
plt.grid(True)
plt.show()