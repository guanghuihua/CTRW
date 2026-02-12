from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


SAVE_DIR = Path(__file__).resolve().parent


def U(x: np.ndarray) -> np.ndarray:
    # Cubic-oscillator potential in the paper benchmark.
    return x**4 / 4.0


def mu(x: np.ndarray) -> np.ndarray:
    # Drift: dX = -U'(X) dt + sqrt(2) dW = -x^3 dt + sqrt(2) dW
    return -x**3


M = 1.0  # sigma^2 / 2 with sigma = sqrt(2)


def build_Q_u(x: np.ndarray, dx: float):
    n = len(x)
    mup = np.maximum(mu(x), 0.0)
    mum = np.maximum(-mu(x), 0.0)
    up = mup / dx + M / dx**2
    um = mum / dx + M / dx**2
    main = -(up + um)

    # no jumps outside the truncated domain [-L, L]
    up[-1] = 0.0
    um[0] = 0.0
    main[0] = -(up[0] + um[0])
    main[-1] = -(up[-1] + um[-1])

    if SCIPY_OK:
        return sp.diags([um[1:], main, up[:-1]], offsets=[-1, 0, 1], format="csc")

    q = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        q[i, i] = main[i]
        if i + 1 < n:
            q[i, i + 1] = up[i]
        if i - 1 >= 0:
            q[i, i - 1] = um[i]
    return q


def build_Q_c(x: np.ndarray, dx: float):
    m = mu(x)
    up = (M / dx**2) * np.exp((m * dx) / (2.0 * M))
    um = (M / dx**2) * np.exp(-(m * dx) / (2.0 * M))
    main = -(up + um)

    up[-1] = 0.0
    um[0] = 0.0
    main[0] = -(up[0] + um[0])
    main[-1] = -(up[-1] + um[-1])

    if SCIPY_OK:
        return sp.diags([um[1:], main, up[:-1]], offsets=[-1, 0, 1], format="csc")

    n = len(x)
    q = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        q[i, i] = main[i]
        if i + 1 < n:
            q[i, i + 1] = up[i]
        if i - 1 >= 0:
            q[i, i - 1] = um[i]
    return q


def stationary_density(q, dx: float) -> np.ndarray:
    n = q.shape[0]
    if SCIPY_OK:
        qt = q.T.tolil()
        b = np.zeros(n, dtype=np.float64)
        qt[0, :] = dx  # normalization row: sum_i pi_i dx = 1
        b[0] = 1.0
        pi = spla.spsolve(qt.tocsc(), b)
    else:
        qt = q.T.copy()
        b = np.zeros(n, dtype=np.float64)
        qt[0, :] = dx
        b[0] = 1.0
        pi = np.linalg.solve(qt, b)
    pi = np.maximum(pi, 0.0)
    pi /= np.sum(pi) * dx
    return pi


def reference_density(x: np.ndarray) -> np.ndarray:
    w = np.exp(-U(x))
    z = np.trapz(w, x)
    return w / z


def make_stationary_density_accuracy_plot() -> Path:
    # Use ascending h so curves rise from left to right on log-log axes.
    dx_list = np.array([0.0125, 0.025, 0.05, 0.1, 0.2], dtype=np.float64)
    l = 8.0
    err_u = []
    err_c = []

    for dx in dx_list:
        x = np.arange(-l, l + 0.5 * dx, dx)
        q_u = build_Q_u(x, dx)
        q_c = build_Q_c(x, dx)
        pi_u = stationary_density(q_u, dx)
        pi_c = stationary_density(q_c, dx)
        pi_ref = reference_density(x)
        err_u.append(np.sum(np.abs(pi_u - pi_ref)) * dx)
        err_c.append(np.sum(np.abs(pi_c - pi_ref)) * dx)

    err_u = np.array(err_u)
    err_c = np.array(err_c)

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.loglog(
        dx_list,
        err_u,
        "o-",
        color="#0072E3",
        ms=8,
        lw=0.8,
        mfc="none",
        mew=0.8,
        label=r"$Q_u$",
    )
    ax.loglog(
        dx_list,
        err_c,
        "o-",
        color="#E69F00",
        ms=8,
        lw=0.8,
        mfc="none",
        mew=0.8,
        label=r"$Q_c$",
    )

    # Shift reference lines away from data to avoid overlap.
    c1 = 0.6 * err_u[2] / dx_list[2]
    c2 = 0.6 * err_c[2] / (dx_list[2] ** 2)
    ax.loglog(
        dx_list,
        c1 * dx_list,
        "--",
        color="#D95319",
        lw=0.8,
        dashes=(7, 5),
        label=r"$O(h)$",
    )
    ax.loglog(
        dx_list,
        c2 * dx_list**2,
        "--",
        color="#7E2F8E",
        lw=0.8,
        dashes=(7, 4),
        label=r"$O(h^2)$",
    )

    ax.set_title("Stationary Density Accuracy")
    ax.set_xlabel("spatial stepsize h")
    ax.set_ylabel(r"$l^1$-error")
    ax.set_xlim(dx_list.min() * 0.8, dx_list.max() * 2.2)
    ax.set_ylim(1e-6, 2e-1)
    ax.grid(False)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black")

    out = SAVE_DIR / "order_stationary_density_accuracy.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.show()
    return out


if __name__ == "__main__":
    out_path = make_stationary_density_accuracy_plot()
    print(f"Saved: {out_path}")
