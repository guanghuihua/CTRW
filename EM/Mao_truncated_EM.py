import numpy as np
import matplotlib.pyplot as plt
import math

# Define the drift and diffusion functions
def f(x):
    return -x**3  # Example: cubic drift

def g(x):
    return np.sqrt(2)  # Constant diffusion

# mu(x) = np.sqrt(2) * np.exp(x)
def mu_inv(h_delta):
    return np.log(h_delta) - 0.5 * np.log(2)
    # return np.sqrt(h_delta)

def h(delta):
    return 1/np.power(delta, 5)
    # return delta**(1/4)

# Time parameters
T = 100 # Total time
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

# Function to calculate the steady-state distribution
def KK() -> object:
    local_N = 100000000
    upper_bound = 100.0
    lower_bound = 0.0
    local_h = (upper_bound - lower_bound) / local_N
    local_K = 0.0
    for ii in range(local_N):
        local_K += math.exp(-0.25 * pow((ii + 0.5) * local_h, 4))
    return 2 * local_K * local_h

# Steady-state distribution function
def steady_state_distribution(x, K):
    return np.exp(-x**4 / 4) / K

# Plotting the results
x_values = np.linspace(-3, 3, 10000)  # Range of x values for steady-state distribution
K_value = KK()  # Normalization constant 归一化常数

# Plot histogram of the EM method simulation results
plt.hist(x, bins=50, density=True, alpha=0.5, label="Truncated EM Simulation")

# Plot the steady-state distribution
steady_state_values = steady_state_distribution(x_values, K_value)
plt.plot(x_values, steady_state_values, label="Steady-State Distribution", color='r')

plt.title("Comparison of Simulated and Steady-State Distributions")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()

if __name__ == '__main__':
    pass
