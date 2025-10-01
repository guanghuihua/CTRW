import numpy as np
import matplotlib.pyplot as plt

# Parameters for the SDE
T = 100.0          # Total time
N = 1000        # Number of time steps
dt = T / N       # Time step
sqrt_dt = np.sqrt(dt)
X0 = 1.0         # Initial condition

# Define the drift and diffusion functions
def f(X):
    return -X**3

def g(X):
    return np.sqrt(2)

# Time array and Wiener process increments
t = np.linspace(0, T, N + 1)
dW = np.random.normal(0, np.sqrt(dt), N)  # Brownian increments

# Initialize the solution array
X_ssbe = np.zeros(N + 1)
X_ssbe[0] = X0

# Apply the Split-Step Backward Euler (SSBE) method
for k in range(N):
    # Compute Y_k^* using implicit Euler step for the drift
    Y_star = X_ssbe[k] + dt * f(X_ssbe[k])
    # Update X_{k+1} with the diffusion term
    X_ssbe[k + 1] = Y_star + g(Y_star) * dW[k]

# Plotting the result
plt.figure(figsize=(10, 6))
plt.plot(t, X_ssbe, label="SSBE Solution")
plt.title("Solution of SDE: dX_t = -X_t^3 dt + sqrt(2) dW_t using SSBE")
plt.xlabel("Time t")
plt.ylabel("X_t")
plt.legend()
plt.grid(True)
plt.show()


# Parameters for the SDE
T = 100.0          # Total time
N = 1000000        # Number of time steps
dt = T / N       # Time step
X0 = 100.0         # Initial condition
X_end = 10
# Define the drift and diffusion functions
def f(X):
    return -X**3

def g(X):
    return np.sqrt(2)

# 隐式方法需要求解代数方程
from scipy.optimize import fsolve
# def equation(x):
#     return x**3 * dt + x - const
def Slove_Y_Star(x):
    const = x
    equation = lambda x:x**3 * dt + x - const
    solution = fsolve(equation, x)
    return solution[0]

def Backward_Euler():
    # Time array and Wiener process increments
    t = np.linspace(0, T, N + 1)
    dW = np.random.normal(0, np.sqrt(dt), N)  # Brownian increments

    # Initialize the solution array
    X_ssbe = np.zeros(N + 1)
    X_ssbe[0] = X0

    # Apply the Split-Step Backward Euler (SSBE) method
    for n in range(N):
        # Compute Y_k^* using implicit Euler step for the drift
#         Y_star = X_ssbe[n] + dt * f(Y_star)
        print(X_ssbe[n])
        Y_star = Slove_Y_Star(X_ssbe[n] )
        print(Y_star)
        # Update X_{k+1} with the diffusion term
        X_ssbe[n + 1] = Y_star + g(Y_star) * dW[n]
        #确定首次退出时间
        if X_ssbe[n] <= X_end:
            print(X_ssbe[n])
            return n,X_ssbe
    return n, X_ssbe

Path_num = 10
Time = np.zeros(Path_num)
for item in range(Path_num):
    n, X_ssbe = Backward_Euler()
    Time[item] = n*dt
#     print(f" The exit time equals {n*dt}.")
print(f"The average time of the stimulation is {Time.sum()/Path_num}")

if __name__ == '__main__':
    pass
