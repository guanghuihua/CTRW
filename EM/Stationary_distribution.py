import numpy as np
import matplotlib.pyplot as plt

# Define the drift and diffusion functions
def f(x):
    return -x**3  # Example: cubic drift

def g(x):
    return np.sqrt(2)  # Constant diffusion

# Define the truncation function
def mu_inv(h_delta):
    return np.sqrt(h_delta)

def h(delta):
    return delta**(1/4)

# Time parameters
T = 1000  # Total time
dt = 0.01  # Time step size
num_steps = int(T / dt)

# Initialize the process
x = np.zeros(num_steps)
x[0] = 1  # Initial condition

# EM simulation with truncation
for t in range(num_steps - 1):
    delta_b = np.sqrt(dt) * np.random.randn()  # Brownian increment
    trunc = mu_inv(h(dt))
    x_trunc = np.sign(x[t]) * min(abs(x[t]), trunc)
    x[t + 1] = x[t] + f(x_trunc) * dt + g(x_trunc) * delta_b

# Plot the histogram of the final values to estimate the invariant distribution
plt.hist(x, bins=50, density=True)
plt.title("Estimated Invariant Distribution")
plt.show()

if __name__ == '__main__':
    pass
