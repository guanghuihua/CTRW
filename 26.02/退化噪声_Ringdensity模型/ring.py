from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TWO_PI = 2.0 * math.pi


def wrap_angle(theta: float) -> float:
    return theta % TWO_PI


def drift(theta: np.ndarray | float) -> np.ndarray | float:
    """
    Drift on S^1.
    This choice is periodic and nontrivial:
        f(theta) = sin(theta) + 0.3 * sin(2*theta)
    """
    return np.sin(theta) + 0.3 * np.sin(2.0 * theta)


def stationary_density_reference(
    theta_grid: np.ndarray, sigma: float, n_quad: int = 20000
) -> np.ndarray:
    """
    Reference invariant density on S^1 for:
        dtheta = f(theta) dt + sigma dW  (mod 2pi)
    using periodic Fokker-Planck closed-form with constant flux.
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
    rho_s = exp_phi * (1.0 - flux * inner)
    rho_s = np.maximum(rho_s, 0.0)
    rho_s /= np.sum(rho_s) * ds

    ref = np.interp(theta_grid, s, rho_s, period=TWO_PI)
    ref /= np.sum(ref) * (TWO_PI / theta_grid.size)
    return ref


def rates_midpoint(theta_i: float, h: float, sigma: float, tau_m: float) -> tuple[float, float]:
    """
    Midpoint tau-leaping-inspired rates:
      theta_mid = theta_i + 0.5 * tau_m * f(theta_i)
      b_mid = f(theta_mid)
      q+ = max(b_mid,0)/h + sigma^2/(2 h^2)
      q- = max(-b_mid,0)/h + sigma^2/(2 h^2)
    """
    theta_mid = wrap_angle(theta_i + 0.5 * tau_m * drift(theta_i))
    b_mid = drift(theta_mid)
    diff = (sigma * sigma) / (2.0 * h * h)
    q_plus = max(b_mid, 0.0) / h + diff
    q_minus = max(-b_mid, 0.0) / h + diff
    return q_plus, q_minus


def simulate_ssa_ctmc(
    k: int,
    sigma: float,
    t_burn: float,
    t_sample: float,
    tau_mid: float,
    seed: int,
) -> np.ndarray:
    """
    Exact SSA (Gillespie) simulation for the midpoint-rate CTMC on periodic grid.
    Occupancy estimator is time-weighted histogram, then normalized to density.
    """
    rng = np.random.default_rng(seed)
    h = TWO_PI / k
    theta_nodes = np.arange(k, dtype=np.float64) * h
    q_plus = np.zeros(k, dtype=np.float64)
    q_minus = np.zeros(k, dtype=np.float64)
    lam = np.zeros(k, dtype=np.float64)
    for i in range(k):
        qp, qm = rates_midpoint(theta_nodes[i], h, sigma, tau_mid)
        q_plus[i] = qp
        q_minus[i] = qm
        lam[i] = qp + qm

    idx = rng.integers(0, k)
    t = 0.0
    t_end = t_burn + t_sample
    occ = np.zeros(k, dtype=np.float64)

    while t < t_end:
        li = lam[idx]
        if li <= 0.0:
            break
        u = rng.random()
        tau = -math.log(u) / li
        if t >= t_burn:
            occ[idx] += tau

        r = rng.random() * li
        if r < q_plus[idx]:
            idx = (idx + 1) % k
        else:
            idx = (idx - 1) % k
        t += tau

    total = np.sum(occ)
    if total <= 0.0:
        raise ValueError("No occupancy recorded. Increase t_sample or reduce t_burn.")
    density_hat = occ / (total * h)
    return density_hat


def l1_error_density(pi_hat: np.ndarray, pi_ref: np.ndarray, h: float) -> float:
    return np.sum(np.abs(pi_hat - pi_ref)) * h


def aligned_ref_line(h_vals: np.ndarray, err_vals: np.ndarray, p: float) -> np.ndarray:
    c = err_vals[0] / (h_vals[0] ** p)
    return c * (h_vals ** p)


def main() -> None:
    sigma = 0.6
    ks = np.array([50, 100, 200, 400, 800], dtype=np.int64)
    t_burn = 200.0
    t_sample = 4000.0
    tau_mid = 0.02
    base_seed = 20260212

    h_vals = TWO_PI / ks
    errors = np.zeros_like(h_vals, dtype=np.float64)

    for j, k in enumerate(ks):
        h = TWO_PI / k
        theta = np.arange(k, dtype=np.float64) * h
        pi_ref = stationary_density_reference(theta, sigma)
        pi_hat = simulate_ssa_ctmc(
            k=k,
            sigma=sigma,
            t_burn=t_burn,
            t_sample=t_sample,
            tau_mid=tau_mid,
            seed=base_seed + int(k),
        )
        errors[j] = l1_error_density(pi_hat, pi_ref, h)
        print(f"K={k:4d}, h={h:.6f}, L1 error={errors[j]:.6e}")

    p_fit = np.polyfit(np.log(h_vals), np.log(errors), 1)[0]
    ref_o1 = aligned_ref_line(h_vals, errors, p=1.0)
    ref_o2 = aligned_ref_line(h_vals, errors, p=2.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(h_vals, errors, "o-", lw=1.8, label=r"$\|\hat\pi_h-\pi_{\rm ref}\|_1$")
    ax.loglog(h_vals, ref_o1, "--", lw=1.2, label=r"$O(h)$")
    ax.loglog(h_vals, ref_o2, "--", lw=1.2, label=r"$O(h^2)$")
    ax.set_xlabel(r"$h = 2\pi/K$")
    ax.set_ylabel(r"$L^1$ error")
    ax.set_title(f"Ring Model: SSA of Midpoint-Rate CTMC (fit slope = {p_fit:.3f})")
    ax.grid(True, which="both", ls=":")
    ax.legend()

    k_inset = 800
    h_in = TWO_PI / k_inset
    theta_in = np.arange(k_inset, dtype=np.float64) * h_in
    pi_ref_in = stationary_density_reference(theta_in, sigma)

    axins = ax.inset_axes([0.58, 0.52, 0.36, 0.36])
    axins.plot(theta_in, pi_ref_in, lw=1.2)
    axins.set_title(r"Inset: $\pi_{\rm ref}(\theta)$", fontsize=9)
    axins.set_xlabel(r"$\theta$", fontsize=8)
    axins.set_ylabel(r"$\pi_{\rm ref}$", fontsize=8)
    axins.tick_params(labelsize=8)
    axins.grid(True, ls=":")

    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "ring_convergence_with_inset.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    print("How to read order: slope in log-log (error vs h) ~= p implies O(h^p).")
    plt.show()


if __name__ == "__main__":
    main()

