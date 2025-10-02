import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = np.loadtxt("data.txt")
x, y = data[:,1], data[:,2]

fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
line, = ax.plot([], [], lw=2)

def init():
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(x[:frame], y[:frame])
    return line,

ani = FuncAnimation(fig, update, frames=len(x), init_func=init, interval=20, blit=True)
plt.show()

