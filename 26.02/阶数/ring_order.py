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
TWO_PI = 2.0 * np.pi


def drift(theta: np.ndarray) -> np.ndarray:
    # Ring model drift on S^1 (periodic).
    return np.sin(theta) + 0.3 * np.sin(2.0 * theta)


def reference_density(theta: np.ndarray, n_quad: int = 20000) -> np.ndarray:
    """
    Reference stationary density for
        dtheta = f(theta) dt + sigma dW  (mod 2pi),
    with sigma = sqrt(2), i.e. M = sigma^2/2 = 1.
    Uses periodic Fokker-Planck closed form with constant flux.
    """
    m = 1.0
    s = np.linspace(0.0, TWO_PI, n_quad, endpoint=False)
    ds = TWO_PI / n_quad
    f = drift(s)

    phi = np.zeros(n_quad, dtype=np.float64)
    for i in range(1, n_quad):
        phi[i] = phi[i - 1] + (f[i - 1] / m) * ds
    exp_neg_phi = np.exp(-phi)
    exp_phi = np.exp(phi)

    z = np.sum(exp_neg_phi) * ds
    flux = (1.0 - np.exp(phi[-1] + (f[-1] / m) * ds)) / z

    inner = np.cumsum(exp_neg_phi) * ds
    rho = exp_phi * (1.0 - flux * inner)
    rho = np.maximum(rho, 0.0)
    rho /= np.sum(rho) * ds

    out = np.interp(theta, s, rho, period=TWO_PI)
    h = TWO_PI / len(theta)
    out /= np.sum(out) * h
    return out


def build_Q_u(theta: np.ndarray, h: float):
    n = len(theta)
    m = 1.0
    b = drift(theta)
    bp = np.maximum(b, 0.0)
    bm = np.maximum(-b, 0.0)
    up = bp / h + m / (h * h)
    um = bm / h + m / (h * h)
    main = -(up + um)

    if SCIPY_OK:
        rows = np.arange(n)
        cols_p = (rows + 1) % n
        cols_m = (rows - 1) % n
        q = sp.csc_matrix(
            (
                np.concatenate([main, up, um]),
                (
                    np.concatenate([rows, rows, rows]),
                    np.concatenate([rows, cols_p, cols_m]),
                ),
            ),
            shape=(n, n),
        )
        return q

    q = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        q[i, i] = main[i]
        q[i, (i + 1) % n] = up[i]
        q[i, (i - 1) % n] = um[i]
    return q


def build_Q_c(theta: np.ndarray, h: float):
    n = len(theta)
    m = 1.0
    b = drift(theta)
    up = (m / (h * h)) * np.exp((b * h) / (2.0 * m))
    um = (m / (h * h)) * np.exp(-(b * h) / (2.0 * m))
    main = -(up + um)

    if SCIPY_OK:
        rows = np.arange(n)
        cols_p = (rows + 1) % n
        cols_m = (rows - 1) % n
        q = sp.csc_matrix(
            (
                np.concatenate([main, up, um]),
                (
                    np.concatenate([rows, rows, rows]),
                    np.concatenate([rows, cols_p, cols_m]),
                ),
            ),
            shape=(n, n),
        )
        return q

    q = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        q[i, i] = main[i]
        q[i, (i + 1) % n] = up[i]
        q[i, (i - 1) % n] = um[i]
    return q


def stationary_density(q, h: float) -> np.ndarray:
    n = q.shape[0]
    if SCIPY_OK:
        qt = q.T.tolil()
        rhs = np.zeros(n, dtype=np.float64)
        qt[0, :] = h
        rhs[0] = 1.0
        pi = spla.spsolve(qt.tocsc(), rhs)
    else:
        qt = q.T.copy()
        rhs = np.zeros(n, dtype=np.float64)
        qt[0, :] = h
        rhs[0] = 1.0
        pi = np.linalg.solve(qt, rhs)
    pi = np.maximum(pi, 0.0)
    pi /= np.sum(pi) * h
    return pi


def main() -> None:
    ks = np.array([50, 100, 200, 400, 800], dtype=np.int64)
    h_vals = TWO_PI / ks
    err_u = np.zeros_like(h_vals, dtype=np.float64)
    err_c = np.zeros_like(h_vals, dtype=np.float64)

    for j, k in enumerate(ks):
        h = TWO_PI / k
        theta = np.arange(k, dtype=np.float64) * h
        pi_ref = reference_density(theta)

        q_u = build_Q_u(theta, h)
        q_c = build_Q_c(theta, h)
        pi_u = stationary_density(q_u, h)
        pi_c = stationary_density(q_c, h)

        err_u[j] = np.sum(np.abs(pi_u - pi_ref)) * h
        err_c[j] = np.sum(np.abs(pi_c - pi_ref)) * h
        print(f"K={k:4d}, h={h:.6f}, err_u={err_u[j]:.6e}, err_c={err_c[j]:.6e}")

    # Reference slopes aligned near middle to avoid overlap.
    i0 = 2
    c1 = 0.7 * err_u[i0] / h_vals[i0]
    c2 = 0.7 * err_c[i0] / (h_vals[i0] ** 2)

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.loglog(
        h_vals,
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
        h_vals,
        err_c,
        "o-",
        color="#E69F00",
        ms=8,
        lw=0.8,
        mfc="none",
        mew=0.8,
        label=r"$Q_c$",
    )
    ax.loglog(h_vals, c1 * h_vals, "--", color="#D95319", lw=0.8, dashes=(7, 5), label=r"$O(h)$")
    ax.loglog(
        h_vals,
        c2 * h_vals**2,
        "--",
        color="#7E2F8E",
        lw=0.8,
        dashes=(7, 4),
        label=r"$O(h^2)$",
    )

    ax.set_title("Ring Stationary Density Accuracy")
    ax.set_xlabel("spatial stepsize h")
    ax.set_ylabel(r"$l^1$-error")
    ax.grid(False)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black")

    out = SAVE_DIR / "ring_stationary_density_accuracy.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()

