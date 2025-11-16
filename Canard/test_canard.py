import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # 添加3D支持

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def van_der_pol_relaxation(t, z, mu):
    """
    范德波尔方程 - 弛豫振荡形式
    x'' - μ(1-x²)x' + x = 0
    """
    x, v = z
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return [dxdt, dvdt]

# 创建图形和子图
fig = plt.figure(figsize=(15, 10))

# 定义子图布局 - 简化布局，移除3D图
ax1 = plt.subplot2grid((2, 2), (0, 0))  # 时间序列
ax2 = plt.subplot2grid((2, 2), (0, 1))  # 相图
ax3 = plt.subplot2grid((2, 2), (1, 0))  # 参数空间
ax4 = plt.subplot2grid((2, 2), (1, 1))  # 能量图

# 设置大μ值用于演示弛豫振荡
mu_large = 8.0
t_span = (0, 30)
t_eval = np.linspace(0, 30, 3000)
z0 = [0.1, 0.1]

print("计算弛豫振荡轨迹...")
sol = solve_ivp(van_der_pol_relaxation, t_span, z0, args=(mu_large,), 
                t_eval=t_eval, method='BDF')

# 提取解
x = sol.y[0]
v = sol.y[1]
t = sol.t

# 初始化动画元素
line_time, = ax1.plot([], [], 'b-', linewidth=2, alpha=0.7)
point_time, = ax1.plot([], [], 'ro', markersize=8)
line_phase, = ax2.plot([], [], 'g-', linewidth=1, alpha=0.7)
point_phase, = ax2.plot([], [], 'ro', markersize=8)
line_energy, = ax4.plot([], [], 'purple', linewidth=2, alpha=0.7)
point_energy, = ax4.plot([], [], 'ro', markersize=8)

# 设置图表
ax1.set_xlim(0, max(t))
ax1.set_ylim(min(x)-0.5, max(x)+0.5)
ax1.set_xlabel('时间')
ax1.set_ylabel('位移 x')
ax1.set_title('范德波尔振荡器 - 时间序列 (μ=8.0)')
ax1.grid(True, alpha=0.3)

ax2.set_xlim(min(x)-0.5, max(x)+0.5)
ax2.set_ylim(min(v)-1, max(v)+1)
ax2.set_xlabel('位移 x')
ax2.set_ylabel('速度 v')
ax2.set_title('相平面')
ax2.grid(True, alpha=0.3)

# 绘制零斜线（nullclines）在相图中
x_range = np.linspace(min(x), max(x), 100)
v_nullcline = x_range / (mu_large * (1 - x_range**2))
v_nullcline[np.abs(v_nullcline) > 10] = np.nan  # 处理奇点
ax2.plot(x_range, v_nullcline, 'r--', alpha=0.5, label='v零斜线')
ax2.axhline(0, color='gray', linestyle='--', alpha=0.5, label='x零斜线')
ax2.legend()

# 参数空间图 - 展示μ值变化的影响
ax3.plot([mu_large], [0], 'ro', markersize=10, label=f'当前μ={mu_large}')
ax3.set_xlabel('μ值')
ax3.set_ylabel('振荡类型')
ax3.set_title('参数空间 - 红色点表示当前μ值')
ax3.set_xlim(0, 10)
ax3.set_ylim(-1, 1)
ax3.grid(True, alpha=0.3)
ax3.legend()

# 能量图
ax4.set_xlim(0, max(t))
ax4.set_ylim(0, max(x**2 + v**2) + 0.5)
ax4.set_xlabel('时间')
ax4.set_ylabel('能量 (x² + v²)')
ax4.set_title('系统能量随时间变化')
ax4.grid(True, alpha=0.3)

# 跃迁检测函数
def detect_transitions(x, v, threshold=3):
    """检测快速跃迁点"""
    transitions = []
    for i in range(1, len(v)-1):
        if abs(v[i]) > threshold and abs(v[i]) > abs(v[i-1]) and abs(v[i]) > abs(v[i+1]):
            transitions.append(i)
    return transitions

# 检测跃迁点
transitions = detect_transitions(x, v)
print(f"检测到 {len(transitions)} 个跃迁点")

# 动画更新函数
def update(frame):
    # 更新当前帧的索引（每10帧跳一次以加快动画）
    idx = min(frame * 10, len(t)-1)
    
    # 更新时间序列图
    line_time.set_data(t[:idx], x[:idx])
    point_time.set_data([t[idx]], [x[idx]])
    
    # 更新相图
    line_phase.set_data(x[:idx], v[:idx])
    point_phase.set_data([x[idx]], [v[idx]])
    
    # 更新能量图
    energy = x[:idx]**2 + v[:idx]**2
    line_energy.set_data(t[:idx], energy)
    point_energy.set_data([t[idx]], [energy[-1] if len(energy) > 0 else 0])
    
    # 在相图中标记跃迁区域
    if abs(v[idx]) > 5:  # 高速区域视为跃迁
        ax2.plot(x[idx], v[idx], 'yo', markersize=6, alpha=0.5)
    
    # 在时间序列图中标记跃迁点
    if idx in transitions:
        ax1.plot(t[idx], x[idx], 'ro', markersize=8, alpha=0.7)
    
    return line_time, point_time, line_phase, point_phase, line_energy, point_energy

# 创建动画
print("创建动画...")
ani = FuncAnimation(fig, update, frames=len(t)//10, interval=50, blit=True)

plt.tight_layout()
plt.show()

# 单独创建一个对比动画：不同μ值的行为
print("\n创建对比动画：不同μ值的行为...")

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

mu_values_compare = [0.5, 1.0, 3.0, 8.0]
colors = ['blue', 'green', 'orange', 'red']
solutions = []

for mu, color in zip(mu_values_compare, colors):
    sol_compare = solve_ivp(van_der_pol_relaxation, (0, 20), [2, 0], args=(mu,), 
                           t_eval=np.linspace(0, 20, 1000), method='BDF')
    solutions.append((sol_compare, color, mu))

lines_time = []
lines_phase = []
points_time = []
points_phase = []

for i, (sol, color, mu) in enumerate(solutions):
    line_t, = ax1.plot([], [], color=color, linewidth=2, label=f'μ={mu}')
    point_t, = ax1.plot([], [], 'o', color=color, markersize=8)
    lines_time.append(line_t)
    points_time.append(point_t)
    
    line_p, = ax2.plot([], [], color=color, linewidth=1, label=f'μ={mu}')
    point_p, = ax2.plot([], [], 'o', color=color, markersize=8)
    lines_phase.append(line_p)
    points_phase.append(point_p)

ax1.set_xlim(0, 20)
ax1.set_ylim(-3, 3)
ax1.set_xlabel('时间')
ax1.set_ylabel('位移 x')
ax1.set_title('不同μ值的时间序列对比')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlim(-3, 3)
ax2.set_ylim(-5, 5)
ax2.set_xlabel('位移 x')
ax2.set_ylabel('速度 v')
ax2.set_title('不同μ值的相图对比')
ax2.legend()
ax2.grid(True, alpha=0.3)

def update_compare(frame):
    idx = min(frame * 10, 999)
    
    for i, (sol, color, mu) in enumerate(solutions):
        t_comp = sol.t
        x_comp = sol.y[0]
        v_comp = sol.y[1]
        
        lines_time[i].set_data(t_comp[:idx], x_comp[:idx])
        points_time[i].set_data([t_comp[idx]], [x_comp[idx]])
        
        lines_phase[i].set_data(x_comp[:idx], v_comp[:idx])
        points_phase[i].set_data([x_comp[idx]], [v_comp[idx]])
    
    return lines_time + points_time + lines_phase + points_phase

ani_compare = FuncAnimation(fig2, update_compare, frames=100, interval=50, blit=True)
plt.tight_layout()
plt.show()

# 专门展示跃迁的动画
print("\n创建跃迁细节动画...")

# 找到一个完整的跃迁周期
if len(transitions) >= 2:
    start_idx = max(0, transitions[0] - 100)
    end_idx = min(len(t)-1, transitions[1] + 100)
    
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 提取跃迁周期的数据
    t_transition = t[start_idx:end_idx]
    x_transition = x[start_idx:end_idx]
    v_transition = v[start_idx:end_idx]
    
    line_transition_time, = ax1.plot([], [], 'b-', linewidth=2)
    point_transition_time, = ax1.plot([], [], 'ro', markersize=8)
    line_transition_phase, = ax2.plot([], [], 'g-', linewidth=2)
    point_transition_phase, = ax2.plot([], [], 'ro', markersize=8)
    
    ax1.set_xlim(min(t_transition), max(t_transition))
    ax1.set_ylim(min(x_transition)-0.5, max(x_transition)+0.5)
    ax1.set_xlabel('时间')
    ax1.set_ylabel('位移 x')
    ax1.set_title('跃迁过程 - 时间序列')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlim(min(x_transition)-0.5, max(x_transition)+0.5)
    ax2.set_ylim(min(v_transition)-1, max(v_transition)+1)
    ax2.set_xlabel('位移 x')
    ax2.set_ylabel('速度 v')
    ax2.set_title('跃迁过程 - 相平面')
    ax2.grid(True, alpha=0.3)
    
    def update_transition(frame):
        idx = min(frame, len(t_transition)-1)
        
        line_transition_time.set_data(t_transition[:idx], x_transition[:idx])
        point_transition_time.set_data([t_transition[idx]], [x_transition[idx]])
        
        line_transition_phase.set_data(x_transition[:idx], v_transition[:idx])
        point_transition_phase.set_data([x_transition[idx]], [v_transition[idx]])
        
        # 标记跃迁区域
        if abs(v_transition[idx]) > 4:
            ax2.plot(x_transition[idx], v_transition[idx], 'yo', markersize=6, alpha=0.7)
        
        return line_transition_time, point_transition_time, line_transition_phase, point_transition_phase
    
    ani_transition = FuncAnimation(fig3, update_transition, frames=len(t_transition), 
                                 interval=30, blit=True)
    plt.tight_layout()
    plt.show()