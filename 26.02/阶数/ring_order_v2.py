from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numba as nb
import numpy as np

TWO_PI = 2.0 * np.pi


@nb.njit(cache=True, fastmath=True)
def drift_scalar(theta: float) -> float:
    return np.sin(theta) + 0.3 * np.sin(2.0 * theta)


@nb.njit(cache=True, fastmath=True)
def build_rates_qu(theta: np.ndarray, h: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    n = theta.size
    up = np.zeros(n, dtype=np.float64)
    um = np.zeros(n, dtype=np.float64)
    m = 0.5 * sigma * sigma
    diff = m / (h * h)
    for i in range(n):
        b = drift_scalar(theta[i])
        up[i] = max(b, 0.0) / h + diff
        um[i] = max(-b, 0.0) / h + diff
    return up, um


@nb.njit(cache=True, fastmath=True)
def build_rates_qc(theta: np.ndarray, h: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    n = theta.size
    up = np.zeros(n, dtype=np.float64)
    um = np.zeros(n, dtype=np.float64)
    m = 0.5 * sigma * sigma
    coef = m / (h * h)
    for i in range(n):
        b = drift_scalar(theta[i])
        up[i] = coef * np.exp((b * h) / (2.0 * m))
        um[i] = coef * np.exp(-(b * h) / (2.0 * m))
    return up, um


@nb.njit(cache=True, fastmath=True)
def stationary_density_uniformization(
    up: np.ndarray,
    um: np.ndarray,
    h: float,
    tol: float = 1e-13,
    max_iter: int = 500000,
) -> np.ndarray:
    """
    Solve stationary distribution via uniformization + power iteration.
    CTMC generator Q (periodic nearest-neighbor) and chain P = I + Q/alpha share
    the same stationary distribution.
    """
    n = up.size
    lam = up + um
    alpha = np.max(lam) * 1.05
    if alpha <= 0.0:
        out = np.ones(n, dtype=np.float64)
        out /= np.sum(out) * h
        return out

    pi = np.ones(n, dtype=np.float64) / n
    nxt = np.zeros(n, dtype=np.float64)

    for _ in range(max_iter):
        for j in range(n):
            im = j - 1
            if im < 0:
                im = n - 1
            ip = j + 1
            if ip == n:
                ip = 0
            stay = 1.0 - lam[j] / alpha
            nxt[j] = (
                pi[j] * stay
                + pi[im] * up[im] / alpha
                + pi[ip] * um[ip] / alpha
            )

        s = np.sum(nxt)
        if s > 0.0:
            nxt /= s

        err = 0.0
        for j in range(n):
            d = abs(nxt[j] - pi[j])
            if d > err:
                err = d
            pi[j] = nxt[j]
        if err < tol:
            break

    pi /= np.sum(pi) * h
    return pi


def stationary_density_reference(
    theta_grid: np.ndarray, sigma: float, n_quad: int = 120000
) -> np.ndarray:
    """
    Reference density from periodic Fokker-Planck closed form with flux.
    """
    s = np.linspace(0.0, TWO_PI, n_quad, endpoint=False)
    ds = TWO_PI / n_quad
    f_s = np.sin(s) + 0.3 * np.sin(2.0 * s)
    c = 2.0 / (sigma * sigma)

    phi = np.zeros(n_quad, dtype=np.float64)
    for i in range(1, n_quad):
        phi[i] = phi[i - 1] + c * f_s[i - 1] * ds

    exp_neg_phi = np.exp(-phi)
    exp_phi = np.exp(phi)
    z = np.sum(exp_neg_phi) * ds
    flux = (1.0 - np.exp(phi[-1] + c * f_s[-1] * ds)) / z

    inner = np.cumsum(exp_neg_phi) * ds
    rho = exp_phi * (1.0 - flux * inner)
    rho = np.maximum(rho, 0.0)
    rho /= np.sum(rho) * ds

    ref = np.interp(theta_grid, s, rho, period=TWO_PI)
    h = TWO_PI / theta_grid.size
    ref /= np.sum(ref) * h
    return ref


@nb.njit(cache=True, fastmath=True)
def l1_error_density(pi_num: np.ndarray, pi_ref: np.ndarray, h: float) -> float:
    return np.sum(np.abs(pi_num - pi_ref)) * h


def local_slopes(h: np.ndarray, e: np.ndarray) -> np.ndarray:
    return np.log(e[1:] / e[:-1]) / np.log(h[1:] / h[:-1])


def main() -> None:
    # Improvements for asymptotic regime visibility:
    # 1) finer K list, 2) higher reference quadrature, 3) report fine-grid slopes.
    k_list = np.array([50, 100, 200, 400, 800, 1600, 3200], dtype=np.int64)
    sigma = np.sqrt(2.0)

    h_vals = TWO_PI / k_list
    err_u = np.zeros_like(h_vals, dtype=np.float64)
    err_c = np.zeros_like(h_vals, dtype=np.float64)

    # warmup numba
    theta_w = np.arange(50, dtype=np.float64) * (TWO_PI / 50)
    up_w, um_w = build_rates_qu(theta_w, TWO_PI / 50, sigma)
    _ = stationary_density_uniformization(up_w, um_w, TWO_PI / 50)

    for j, k in enumerate(k_list):
        h = TWO_PI / k
        theta = np.arange(k, dtype=np.float64) * h

        up_u, um_u = build_rates_qu(theta, h, sigma)
        up_c, um_c = build_rates_qc(theta, h, sigma)
        pi_u = stationary_density_uniformization(up_u, um_u, h)
        pi_c = stationary_density_uniformization(up_c, um_c, h)
        pi_ref = stationary_density_reference(theta, sigma, n_quad=120000)

        err_u[j] = l1_error_density(pi_u, pi_ref, h)
        err_c[j] = l1_error_density(pi_c, pi_ref, h)
        print(
            f"K={k:4d}, h={h:.6e}, err_u={err_u[j]:.6e}, err_c={err_c[j]:.6e}, "
            f"mass_u={np.sum(pi_u)*h:.12f}, mass_c={np.sum(pi_c)*h:.12f}"
        )

    slope_u = local_slopes(h_vals, err_u)
    slope_c = local_slopes(h_vals, err_c)
    print("Local slopes Qu:", slope_u)
    print("Local slopes Qc:", slope_c)
    print("Fine-grid mean slope Qu (last 3):", np.mean(slope_u[-3:]))
    print("Fine-grid mean slope Qc (last 3):", np.mean(slope_c[-3:]))

    i0 = 3  # align around middle-fine point to avoid overlap
    c1 = 0.6 * err_u[i0] / h_vals[i0]
    c2 = 0.6 * err_c[i0] / (h_vals[i0] ** 2)

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    ax.loglog(
        h_vals,
        err_u,
        "o-",
        color="#0072E3",
        ms=7,
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
        ms=7,
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

    ax.set_title("Ring Stationary Density Accuracy (v2)")
    ax.set_xlabel("spatial stepsize h")
    ax.set_ylabel(r"$l^1$-error")
    ax.grid(False)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black")

    out_path = Path(__file__).resolve().parent / "ring_stationary_density_accuracy_v2.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

