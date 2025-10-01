import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

# --------------------- canard ODE ---------------------
def canard(x):
    delta = 0.1
    a = 1 - delta / 8 - 3 * delta**2 / 32 - 173 * delta**3 / 1024 - 0.01
    dx1 = (x[0] + x[1] - x[0]**3 / 3) / delta
    dx2 = a - x[0]
    return np.array([dx1, dx2])

# --------------------- Monte Carlo method ---------------------
def MC_2D(lowx, lowy, Sp, N, eps, Sample, dt):
    h = Sp / N
    data = np.zeros(N * N)
    P = np.zeros((N, N))
    x_old = np.array([1.0, 1.0])

    for _ in range(Sample):
        noise = eps * np.sqrt(dt) * np.random.randn(2)
        x_new = x_old + dt * canard(x_old) + noise
        xx, yy = x_new

        x_n = int(np.ceil((xx - lowx) / h)) - 1
        y_n = int(np.ceil((yy - lowy) / h)) - 1

        if 0 <= x_n < N and 0 <= y_n < N:
            P[x_n, y_n] += 1
        x_old = x_new

    for i in range(N):
        for j in range(N):
            data[i * N + j] = P[i, j]

    data = data / (h**2 * np.sum(data))
    print(f"Sample Size = {Sample}")

    return data

# --------------------- SSA method ---------------------
def SSA_2D(lowx, lowy, Sp, N, eps, Sample):
    h = Sp / N
    data = np.zeros((N + 1) * (N + 1))
    P = np.zeros((N + 1, N + 1))
    x, y = 1.0, 1.0
    t = 0.0
    t_stop = 50000
    delta = 0.1
    a = 1 - delta / 8 - 3 * delta**2 / 32 - 173 * delta**3 / 1024 - 0.01
    sigma = eps
    M_11 = sigma**2 / 2
    M_22 = sigma**2 / 2
    C1 = M_11 / h**2
    C2 = M_22 / h**2

    n = 0
    while t < t_stop:
        mu_1 = (y - x**3 / 3 + x) / delta
        mu_2 = a - x

        q1 = max(mu_1, 0) / h + C1
        q2 = -min(mu_1, 0) / h + C1
        q3 = max(mu_2, 0) / h + C2
        q4 = -min(mu_2, 0) / h + C2

        lambda_sum = q1 + q2 + q3 + q4
        r = np.random.rand(2)
        tau = -np.log(r[0]) / lambda_sum
        mu_number = np.searchsorted(np.cumsum([q1, q2, q3, q4]), r[1] * lambda_sum) + 1

        t += tau

        if mu_number == 4:
            x += h
        elif mu_number == 3:
            x -= h
        elif mu_number == 2:
            y += h
        elif mu_number == 1:
            y -= h

        x_n = int(round((x - lowx) / h))
        y_n = int(round((y - lowy) / h))

        if 0 <= x_n <= N and 0 <= y_n <= N:
            P[x_n, y_n] += 1

        n += 1
        if n == Sample:
            break

    for i in range(N + 1):
        for j in range(N + 1):
            data[i * (N + 1) + j] = P[i, j]

    data /= h**2 * np.sum(data)
    print(f"Sample Size = {n}")

    return data

# --------------------- Build Matrix ---------------------
def Matrix_2D(lowx, lowy, Sp, N, eps0):
    eps = eps0 ** 2 / 2
    h = Sp / N

    def f1(x, y):
        return canard([x, y])[0]

    def f2(x, y):
        return canard([x, y])[1]

    b = np.zeros((N - 1) * (N - 1) + 1)
    b[-1] = 1 / h**2

    max_entries = 5 * (N - 1)**2 + (N + 1)**2
    index1 = np.zeros(max_entries, dtype=int)
    index2 = np.zeros(max_entries, dtype=int)
    value = np.zeros(max_entries)
    count = 0

    for i in range(2, N + 1):
        for j in range(2, N + 1):
            xx = (i - 1) * h + lowx
            yy = (j - 1) * h + lowy
            row_idx = (i - 2) * (N - 1) + (j - 1)

            center_idx  = (i - 1) * (N + 1) + j
            right_idx   = (i - 1) * (N + 1) + (j + 1)
            left_idx    = (i - 1) * (N + 1) + (j - 1)
            up_idx      = (i - 2) * (N + 1) + j
            down_idx    = i * (N + 1) + j

            index1[count] = row_idx
            index2[count] = down_idx
            value[count] = -f1(xx + h, yy) / (2 * h) + eps / h**2
            count += 1

            index1[count] = row_idx
            index2[count] = up_idx
            value[count] = f1(xx - h, yy) / (2 * h) + eps / h**2
            count += 1

            index1[count] = row_idx
            index2[count] = right_idx
            value[count] = -f2(xx, yy + h) / (2 * h) + eps / h**2
            count += 1

            index1[count] = row_idx
            index2[count] = left_idx
            value[count] = f2(xx, yy - h) / (2 * h) + eps / h**2
            count += 1

            index1[count] = row_idx
            index2[count] = center_idx
            value[count] = -4 * eps / h**2
            count += 1

    for i in range(N + 1):
        for j in range(N + 1):
            index1[count] = (N - 1) * (N - 1)
            index2[count] = i * (N + 1) + j
            value[count] = 1
            count += 1

    A = coo_matrix((value[:count], (index1[:count], index2[:count])),
                   shape=((N - 1) * (N - 1) + 1, (N + 1) * (N + 1)))

    return A, b

# --------------------- Run the simulation and solve the system ---------------------
if __name__ == "__main__":
    N = 600
    eps = 0.3
    lowx = -3.0
    lowy = -3.0
    Sp = 6.0
    Sample = 10_000_000
    dt = 0.001

    import time
    t1 = time.time()
    data2 = SSA_2D(lowx, lowy, Sp, N, eps, Sample)
    t2 = time.time()
    print("SSA_2D data generated")
    print("Elapsed time:", t2 - t1)

    A, b = Matrix_2D(lowx, lowy, Sp, N, eps)
    print("Matrix built")
    b = b - A.dot(data2)
    x = lsqr(A, b)[0]
    y = x + data2

    V = np.zeros((N + 1, N + 1))
    W = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            idx = i * (N + 1) + j
            V[i, j] = y[idx]
            W[i, j] = data2[idx]

    X, Y = np.meshgrid(np.linspace(lowx, lowx + Sp, N + 1),
                       np.linspace(lowy, lowy + Sp, N + 1))

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    ax1.plot_surface(X, Y, W.T, cmap='viridis')
    ax1.set_title('Monte Carlo Solution (W)')

    ax2 = fig.add_subplot(2, 1, 2, projection='3d')
    ax2.plot_surface(X, Y, V.T, cmap='plasma')
    ax2.set_title('Corrected Solution (V)')

    plt.tight_layout()
    plt.show()
