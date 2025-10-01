import numpy as np
import matplotlib.pyplot as plt

# 定义反应速率常数
c1 = 20.0  # 出生速率常数
c2 = 0.5  # 死亡速率常数

X0 = 10  # 初始分子数
t_max = 100  # 最大模拟时间
t = 0  # 初始时间
X = X0  # 初始分子数
time_points = [t]
X_points = [X]

while t < t_max:
    a1 = c1  # 出生速率
    a2 = c2 * X  # 死亡速率

    a0 = a1 + a2  # 总反应速率
    if a0 == 0:
        break

    r1 = np.random.rand()  # 用于确定时间间隔的随机数
    r2 = np.random.rand()  # 用于确定反应类型的随机数

    # 计算下一个反应的时间间隔
    tau = (1.0 / a0) * np.log(1.0 / r1)

    # 确定反应类型并更新分子数
    if r2 * a0 < a1:
        X += 1
    else:
        X -= 1

        # 更新时间
    t += tau

    # 记录时间和分子数
    time_points.append(t)
    X_points.append(X)

# 可视化结果
plt.plot(time_points, X_points, marker='o', linestyle='-', drawstyle='steps-post')
plt.xlabel('Time')
plt.ylabel('Molecule Count')
plt.title('Gillespie Algorithm Simulation: Birth-Death Process')
plt.grid(True)
plt.show()

# 如果你还想查看每个时间点的分子数（可选）
for i, (time, count) in enumerate(zip(time_points, X_points)):
    print(f"Time {time:.2f}: Molecule Count = {count}")