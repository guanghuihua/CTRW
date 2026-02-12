from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numba as nb
import numpy as np

TWO_PI = 2.0 * math.pi


@nb.njit(cache=True, fastmath=True)
def _drift_scalar(theta: float) -> float:
    return math.sin(theta) + 0.3 * math.sin(2.0 * theta)


@nb.njit(cache=True, fastmath=True)
def _wrap_angle(theta: float) -> float:
    return theta % TWO_PI


def drift(theta: np.ndarray | float) -> np.ndarray | float:
    return np.sin(theta) + 0.3 * np.sin(2.0 * theta)


def stationary_density_reference(
    theta_grid: np.ndarray, sigma: float, n_quad: int = 60000
) -> np.ndarray:
    """
    Periodic Fokker-Planck reference density for
        dtheta = f(theta) dt + sigma dW (mod 2pi).
    """
    s = np.linspace(0.0, TWO_PI, n_quad, endpoint=False)
    ds = TWO_PI / n_quad
    f_s = drift(s)
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
def rates_midpoint(theta_i: float, h: float, sigma: float, tau_mid: float) -> tuple[float, float]:
    theta_mid = _wrap_angle(theta_i + 0.5 * tau_mid * _drift_scalar(theta_i))
    b_mid = _drift_scalar(theta_mid)
    diff = (sigma * sigma) / (2.0 * h * h)
    q_plus = max(b_mid, 0.0) / h + diff
    q_minus = max(-b_mid, 0.0) / h + diff
    return q_plus, q_minus


@nb.njit(cache=True, fastmath=True)
def _precompute_rates_numba(
    k: int, sigma: float, tau_mid: float, mode_flag: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    mode_flag=0 -> Qu midpoint upwind+diffusion
    mode_flag=1 -> Qc midpoint exponential-fitted
    """
    h = TWO_PI / k
    q_plus = np.zeros(k, dtype=np.float64)
    q_minus = np.zeros(k, dtype=np.float64)
    m = (sigma * sigma) / 2.0

    for i in range(k):
        theta_i = i * h
        theta_mid = _wrap_angle(theta_i + 0.5 * tau_mid * _drift_scalar(theta_i))
        b_mid = _drift_scalar(theta_mid)
        if mode_flag == 0:
            diff = m / (h * h)
            qp = max(b_mid, 0.0) / h + diff
            qm = max(-b_mid, 0.0) / h + diff
        else:
            qp = (m / (h * h)) * math.exp((b_mid * h) / (2.0 * m))
            qm = (m / (h * h)) * math.exp(-(b_mid * h) / (2.0 * m))
        q_plus[i] = qp
        q_minus[i] = qm

    lam = q_plus + q_minus
    return q_plus, q_minus, lam


@nb.njit(cache=True, fastmath=True)
def _simulate_ssa_ctmc_numba(
    k: int,
    sigma: float,
    t_burn: float,
    t_sample: float,
    tau_mid: float,
    seed: int,
    mode_flag: int,
) -> np.ndarray:
    np.random.seed(seed)
    h = TWO_PI / k
    q_plus, q_minus, lam = _precompute_rates_numba(k, sigma, tau_mid, mode_flag)

    idx = int(np.random.randint(0, k))
    t = 0.0
    t_end = t_burn + t_sample
    occ = np.zeros(k, dtype=np.float64)

    while t < t_end:
        li = lam[idx]
        if li <= 0.0:
            break
        u = np.random.random()
        if u < 1e-15:
            u = 1e-15
        tau = -math.log(u) / li
        if t >= t_burn:
            occ[idx] += tau

        r = np.random.random() * li
        if r < q_plus[idx]:
            idx += 1
            if idx == k:
                idx = 0
        else:
            idx -= 1
            if idx < 0:
                idx = k - 1
        t += tau

    total = np.sum(occ)
    if total <= 0.0:
        return np.zeros(k, dtype=np.float64)
    return occ / (total * h)


def simulate_ssa_ctmc(
    k: int,
    sigma: float,
    t_burn: float,
    t_sample: float,
    tau_mid: float,
    seed: int,
    mode: str = "Qu",
) -> np.ndarray:
    mode_flag = 0 if mode == "Qu" else 1
    pi_hat = _simulate_ssa_ctmc_numba(k, sigma, t_burn, t_sample, tau_mid, seed, mode_flag)
    if np.sum(pi_hat) <= 0.0:
        raise ValueError("No occupancy recorded. Increase t_sample or reduce t_burn.")
    return pi_hat


@nb.njit(cache=True, fastmath=True)
def l1_error_density(pi_hat: np.ndarray, pi_ref: np.ndarray, h: float) -> float:
    return np.sum(np.abs(pi_hat - pi_ref)) * h


def plot_accuracy(h_vals: np.ndarray, err_u: np.ndarray, err_c: np.ndarray, out_path: Path) -> None:
    i0 = len(h_vals) // 2
    c1 = 0.65 * err_u[i0] / h_vals[i0]
    c2 = 0.65 * err_c[i0] / (h_vals[i0] ** 2)

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
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
    ax.loglog(
        h_vals,
        c1 * h_vals,
        "--",
        color="#D95319",
        lw=0.8,
        dashes=(7, 5),
        label=r"$O(h)$",
    )
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
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.show()


def main() -> None:
    # More robust defaults to reduce Monte Carlo crossing.
    k_list = [50, 100, 200, 400, 800]
    sigma = 0.6
    tau_mid = 0.01
    t_burn = 500.0
    t_sample = 20000.0
    n_rep = 12
    base_seed = 20260212

    ks = np.array(k_list, dtype=np.int64)
    h_vals = TWO_PI / ks
    err_u = np.zeros_like(h_vals, dtype=np.float64)
    err_c = np.zeros_like(h_vals, dtype=np.float64)

    # trigger JIT once
    _ = rates_midpoint(0.1, TWO_PI / 50, sigma, tau_mid)

    for j, k in enumerate(ks):
        h = TWO_PI / k
        theta = np.arange(k, dtype=np.float64) * h
        pi_ref = stationary_density_reference(theta, sigma)
        print(f"K={k:4d} | ref mass={np.sum(pi_ref) * h:.12f}")

        e_u_rep = np.zeros(n_rep, dtype=np.float64)
        e_c_rep = np.zeros(n_rep, dtype=np.float64)
        for r in range(n_rep):
            seed_u = base_seed + 10000 * j + 2 * r
            seed_c = base_seed + 10000 * j + 2 * r + 1
            pi_u = simulate_ssa_ctmc(k, sigma, t_burn, t_sample, tau_mid, seed_u, mode="Qu")
            pi_c = simulate_ssa_ctmc(k, sigma, t_burn, t_sample, tau_mid, seed_c, mode="Qc")
            e_u_rep[r] = l1_error_density(pi_u, pi_ref, h)
            e_c_rep[r] = l1_error_density(pi_c, pi_ref, h)

        err_u[j] = float(np.mean(e_u_rep))
        err_c[j] = float(np.mean(e_c_rep))
        print(
            f"  mean errors: Qu={err_u[j]:.6e}, Qc={err_c[j]:.6e}; "
            f"std Qu={np.std(e_u_rep):.2e}, std Qc={np.std(e_c_rep):.2e}"
        )

    out_path = Path(__file__).resolve().parent / "ring_stationary_density_accuracy_fig3.png"
    plot_accuracy(h_vals, err_u, err_c, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

