import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data.txt")
x = data[:,1]
y = data[:,2]

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Phase Portrait')
plt.grid(True)
plt.show()

