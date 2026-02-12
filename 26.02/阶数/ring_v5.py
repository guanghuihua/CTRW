from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def ring_drift(x0: float, x1: float) -> tuple[float, float]:
    mu1 = -4.0 * x0 * (x0 * x0 + x1 * x1 - 1.0) + x1
    mu2 = -4.0 * x1 * (x0 * x0 + x1 * x1 - 1.0) - x0
    return mu1, mu2


@nb.njit(cache=True, fastmath=True)
def _project_to_grid(v: float, low: float, h: float) -> float:
    return low + h / 2.0 + round((v - low - h / 2.0) / h) * h


@nb.njit(cache=True, fastmath=True)
def _rates(mu1: float, mu2: float, h: float, eps0: float, scheme_id: int) -> tuple[float, float, float, float]:
    """
    scheme_id:
      0 -> Q_u
      1 -> Q_c
      2 -> Q_tilde_u
    """
    m = 0.5 * eps0 * eps0
    if scheme_id == 0:
        # Baseline upwind + constant diffusion
        q1 = max(mu1, 0.0) / h + m / (h * h)
        q2 = max(-mu1, 0.0) / h + m / (h * h)
        q3 = max(mu2, 0.0) / h + m / (h * h)
        q4 = max(-mu2, 0.0) / h + m / (h * h)
    elif scheme_id == 1:
        # Exponential-fitted (Q_c)
        coef = m / (h * h)
        q1 = coef * np.exp((mu1 * h) / (2.0 * m))
        q2 = coef * np.exp(-(mu1 * h) / (2.0 * m))
        q3 = coef * np.exp((mu2 * h) / (2.0 * m))
        q4 = coef * np.exp(-(mu2 * h) / (2.0 * m))
    else:
        # Improved \tilde{Q}_u from Ringdensity_FiniteTimeError.py
        m1 = 0.5 * max(eps0 * eps0 - abs(mu1) * h, 0.0)
        m2 = 0.5 * max(eps0 * eps0 - abs(mu2) * h, 0.0)
        q1 = max(mu1, 0.0) / h + m1 / (h * h)
        q2 = max(-mu1, 0.0) / h + m1 / (h * h)
        q3 = max(mu2, 0.0) / h + m2 / (h * h)
        q4 = max(-mu2, 0.0) / h + m2 / (h * h)
    return q1, q2, q3, q4


@nb.njit(cache=True, fastmath=True)
def simulate_stationary_histogram(
    n: int,
    scheme_id: int,
    eps0: float,
    lowx: float,
    lowy: float,
    span: float,
    tau: float,
    burn_steps: int,
    sample_steps: int,
    seed: int,
) -> np.ndarray:
    """
    Tau-leaping on a projected lattice, then occupancy-frequency estimator.
    """
    np.random.seed(seed)
    h = span / n
    x = lowx + np.random.random() * span
    y = lowy + np.random.random() * span
    x = _project_to_grid(x, lowx, h)
    y = _project_to_grid(y, lowy, h)

    counts = np.zeros((n, n), dtype=np.float64)

    total_steps = burn_steps + sample_steps
    for step in range(total_steps):
        mu1, mu2 = ring_drift(x, y)
        q1, q2, q3, q4 = _rates(mu1, mu2, h, eps0, scheme_id)

        # Skellam increment via Poisson difference.
        dx = np.random.poisson(q1 * tau) - np.random.poisson(q2 * tau)
        dy = np.random.poisson(q3 * tau) - np.random.poisson(q4 * tau)

        x += dx * h
        y += dy * h

        # hard clipping + projection, consistent with legacy scripts
        x = min(max(x, lowx + h / 2.0), lowx + span - h / 2.0)
        y = min(max(y, lowy + h / 2.0), lowy + span - h / 2.0)
        x = _project_to_grid(x, lowx, h)
        y = _project_to_grid(y, lowy, h)

        if step >= burn_steps:
            ix = int(round((x - lowx - h / 2.0) / h))
            iy = int(round((y - lowy - h / 2.0) / h))
            if 0 <= ix < n and 0 <= iy < n:
                counts[ix, iy] += 1.0

    total = np.sum(counts)
    if total <= 0.0:
        return np.zeros((n, n), dtype=np.float64)
    pi_hat = counts / (total * h * h)
    return pi_hat


def true_invariant_density(n: int, eps0: float, lowx: float, lowy: float, span: float) -> np.ndarray:
    """
    For b = -grad U + rotational part, with U=(r^2-1)^2 and isotropic noise eps0:
      rho(x) ∝ exp(-2 U / eps0^2)
    """
    h = span / n
    x = np.linspace(lowx + h / 2.0, lowx + span - h / 2.0, n)
    y = np.linspace(lowy + h / 2.0, lowy + span - h / 2.0, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    U = (X * X + Y * Y - 1.0) ** 2
    rho = np.exp(-2.0 * U / (eps0 * eps0))
    rho /= np.sum(rho) * h * h
    return rho


def l1_error(pi_hat: np.ndarray, pi_true: np.ndarray, h: float) -> float:
    return float(np.sum(np.abs(pi_hat - pi_true)) * h * h)


def local_slope(h: np.ndarray, e: np.ndarray) -> np.ndarray:
    return np.log(e[1:] / e[:-1]) / np.log(h[1:] / h[:-1])


def main() -> None:
    # Paper-aligned settings from your screenshot (Ring density model section).
    h_list = [0.2, 0.1, 0.05, 0.025, 0.0125]
    eps0 = 0.15
    lowx = -2.0
    lowy = -2.0
    span = 4.0
    tau = 0.1
    # Text in screenshot mentions random-walk time T=2,000,000 and ending time T=5*10^6.
    t_burn = 2_000_000.0
    t_sample = 5_000_000.0
    burn_steps = int(round(t_burn / tau))
    sample_steps = int(round(t_sample / tau))
    n_rep = 1
    scheme_id = 2
    scheme_label = r"$\tilde{Q}_u$"

    print("Running v5 pipeline with paper data:")
    print(f"sigma={eps0}, domain=[{lowx},{lowx+span}]x[{lowy},{lowy+span}]")
    print(f"tau={tau}, burn_steps={burn_steps}, sample_steps={sample_steps}, n_rep={n_rep}")

    # Warm-up JIT
    _ = simulate_stationary_histogram(
        16, scheme_id, eps0, lowx, lowy, span, tau, 10, 20, seed=1
    )

    h_vals = np.array(h_list, dtype=np.float64)
    n_list = [int(round(span / h)) for h in h_vals]
    err = np.zeros_like(h_vals)

    for i, h in enumerate(h_vals):
        n = n_list[i]
        rho_true = true_invariant_density(n, eps0, lowx, lowy, span)
        e_sum = 0.0
        for r in range(n_rep):
            pi_hat = simulate_stationary_histogram(
                n,
                scheme_id,
                eps0,
                lowx,
                lowy,
                span,
                tau,
                burn_steps,
                sample_steps,
                seed=1000 * n + 10 * r + 1,
            )
            e_sum += l1_error(pi_hat, rho_true, h)
        err[i] = e_sum / n_rep
        print(f"h={h:.6f}, N={n:4d}, {scheme_label} L1 error={err[i]:.6e}")

    slopes = local_slope(h_vals, err)
    print("local slopes:", slopes)

    i0 = 1
    c1 = 0.6 * err[i0] / h_vals[i0]
    c2 = 0.6 * err[i0] / (h_vals[i0] ** 2)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.loglog(h_vals, err, "s-", color="#2CA02C", mfc="none", lw=0.9, ms=6, label=scheme_label)
    ax.loglog(h_vals, c1 * h_vals, "--", color="#D95319", dashes=(7, 5), lw=0.9, label=r"$O(h)$")
    ax.loglog(h_vals, c2 * h_vals**2, "--", color="#7E2F8E", dashes=(7, 4), lw=0.9, label=r"$O(h^2)$")

    ax.set_title("Ring Invariant Density L1 Error (Single Scheme, Numba)")
    ax.set_xlabel("spatial stepsize h")
    ax.set_ylabel(r"$L^1$ error")
    ax.grid(False)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black")

    out = Path(__file__).resolve().parent / "ring_v5_l1_error.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
