import numpy as np
import matplotlib.pyplot as plt
import time

def Van_der_Pol_oscillator( delta, x, y ):
    dxdt = 1/delta * (y - x**3/3 + x)
    dydt = a - x 
    return [dxdt, dydt]


N = 10**8
t = np.linspace(-10, 10, N)
xi_x = np.random.normal(0, 20/N, N)   
xi_y = np.random.normal(0, 20/N, N)
X = np.zeros(N)
Y = np.zeros(N)
delta = 0.01
a = 1 - delta/8 -3*delta**2/32 -173*delta**3/256 - 0.01

X[0] = 1
Y[0] = 0


start_time = time.time()
for i in range(N-1):
    dxdt, dydt = Van_der_Pol_oscillator( delta, X[i], Y[i] )
    X[i+1] = X[i] + dxdt * (t[1]-t[0]) + xi_x[i] 
    Y[i+1] = Y[i] + dydt * (t[1]-t[0]) + xi_y[i] 
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Stimulating number:{N}, elapsed_time: {elapsed_time:.4f}")

plt.figure(figsize=(8,6))               
plt.plot(X,Y)
plt.show()