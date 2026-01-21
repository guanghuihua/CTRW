import time

import matplotlib.pyplot as plt
import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def ssa_2d_timeweighted(
    lowx: float,
    lowy: float,
    span: float,
    n: int,
    eps: float,
    sample: int,
    bin_factor: int,
) -> tuple:
    """SSA with time-weighted occupancy for invariant density."""
    h = span / n
    counts = np.zeros((n + 1, n + 1), dtype=float)

    x = 1.0
    y = 1.0
    t = 0.02
    t_stop = 1e6
    n_events = 0
    delta = 0.1
    a = 1 - delta / 8 - 3 * delta**2 / 32 - 173 * delta**3 / 1024 - 0.01
    sigma = eps
    m11 = sigma**2 / 2.0
    m22 = sigma**2 / 2.0
    c1 = m11 / h**2
    c2 = m22 / h**2

    while t < t_stop:
        mu_1 = (y - (x**3) / 3.0 + x) / delta
        mu_2 = a - x

        q1 = max(mu_1, 0.0) / h + c1
        q2 = -min(mu_1, 0.0) / h + c1
        q3 = max(mu_2, 0.0) / h + c2
        q4 = -min(mu_2, 0.0) / h + c2

        lam = q1 + q2 + q3 + q4
        r1 = np.random.random()
        r2 = np.random.random()
        tau = -np.log(r1) / lam

        r = r2 * lam
        if r <= q1:
            mu_number = 4
        elif r <= q1 + q2:
            mu_number = 3
        elif r <= q1 + q2 + q3:
            mu_number = 2
        else:
            mu_number = 1

        # Time-weighted occupancy at current location
        x_n = round((x - lowx) / h)
        y_n = round((y - lowy) / h)
        if 1 <= x_n <= n + 1 and 1 <= y_n <= n + 1:
            counts[x_n - 1, y_n - 1] += tau

        t += tau
        if mu_number == 4:
            x += h
        elif mu_number == 3:
            x -= h
        elif mu_number == 2:
            y += h
        elif mu_number == 1:
            y -= h

        n_events += 1
        if n_events == sample:
            break

    if bin_factor > 1:
        out_n = n // bin_factor
        coarse = np.zeros((out_n, out_n), dtype=float)
        for i in range(out_n):
            i0 = i * bin_factor
            for j in range(out_n):
                j0 = j * bin_factor
                s = 0.0
                for di in range(bin_factor):
                    for dj in range(bin_factor):
                        s += counts[i0 + di, j0 + dj]
                coarse[i, j] = s
        data = coarse.T.ravel()
        h_eff = h * bin_factor
    else:
        data = counts.T.ravel()
        h_eff = h

    total_time = data.sum()
    if total_time > 0:
        data /= (h_eff**2 * total_time)
    return data, n_events


def main() -> None:
    n = 1000
    eps = 0.3
    lowx = -3.0
    lowy = -3.0
    span = 6.0
    sample = 1_000_000
    out_n = 100
    if n % out_n != 0:
        raise ValueError("n must be divisible by out_n")
    bin_factor = n // out_n

    t1 = time.perf_counter()
    data2, n_events = ssa_2d_timeweighted(
        lowx, lowy, span, n, eps, sample, bin_factor
    )
    t2 = time.perf_counter()
    print("SSA_2D time-weighted data generated")
    print(f"Sample Size = {n_events}")
    print(f"Elapsed time: {t2 - t1:.2f} seconds")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    c = ax.pcolormesh(data2.reshape((out_n, out_n)), cmap="inferno")
    fig.colorbar(c)
    ax.set_title("SSA 2D Time-Weighted Density")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(
        *np.meshgrid(np.arange(out_n), np.arange(out_n), indexing="ij"),
        data2.reshape((out_n, out_n)),
        cmap="viridis",
        edgecolor="none",
    )
    ax.set_title("SSA 2D Time-Weighted Density Surface Plot")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()


if __name__ == "__main__":
    main()
