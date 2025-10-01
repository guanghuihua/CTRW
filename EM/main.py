import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

N = 1000
X_start = 0
X_end = 100
t = np.linspace(X_start, X_end, N)
X = np.zeros(N)
def dxdt(x):
    return x**2
for i in range(N-1):
    X[i+1] = X[i] + dxdt(t[i])
    
plt.plot(t, X)
plt.show()