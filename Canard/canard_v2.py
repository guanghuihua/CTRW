import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the Canard fast-slow system
def canard_ode(t, z, delta=0.1, a=None):
    x, y = z
    if a is None:
        a = 1 - delta/8 - 3*delta**2/32 - 173*delta**3/1024 - 0.01
    dxdt = (x + y - x**3 / 3) / delta
    dydt = a - x
    return [dxdt, dydt]

# Time span and initial condition
t_span = (0, 10)
z0 = [1.5, -1.5]  # starting near slow manifold

# Solve ODE
sol = solve_ivp(lambda t, z: canard_ode(t, z), t_span, z0, t_eval=np.linspace(*t_span, 10000))

# Plot the trajectory
x_vals, y_vals = sol.y
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label='Canard trajectory', color='darkblue')
plt.title('Canard Trajectory in Phase Space (x vs y)')
plt.xlabel('x (fast variable)')
plt.ylabel('y (slow variable)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
