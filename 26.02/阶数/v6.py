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
def _rates(mu1: float, mu2: float, h: float, sigma: float, scheme_id: int) -> tuple[float, float, float, float]:
    m = 0.5 * sigma * sigma
    if scheme_id == 0:  # Q_u
        q1 = max(mu1, 0.0) / h + m / (h * h)
        q2 = max(-mu1, 0.0) / h + m / (h * h)
        q3 = max(mu2, 0.0) / h + m / (h * h)
        q4 = max(-mu2, 0.0) / h + m / (h * h)
    elif scheme_id == 1:  # Q_c
        coef = m / (h * h)
        q1 = coef * np.exp((mu1 * h) / (2.0 * m))
        q2 = coef * np.exp(-(mu1 * h) / (2.0 * m))
        q3 = coef * np.exp((mu2 * h) / (2.0 * m))
        q4 = coef * np.exp(-(mu2 * h) / (2.0 * m))
    else:  # Q_tilde_u
        m1 = 0.5 * max(sigma * sigma - abs(mu1) * h, 0.0)
        m2 = 0.5 * max(sigma * sigma - abs(mu2) * h, 0.0)
        q1 = max(mu1, 0.0) / h + m1 / (h * h)
        q2 = max(-mu1, 0.0) / h + m1 / (h * h)
        q3 = max(mu2, 0.0) / h + m2 / (h * h)
        q4 = max(-mu2, 0.0) / h + m2 / (h * h)
    return q1, q2, q3, q4


@nb.njit(cache=True, fastmath=True)
def simulate_stationary_histogram(
    n: int,
    scheme_id: int,
    sigma: float,
    lowx: float,
    lowy: float,
    span: float,
    tau: float,
    burn_steps: int,
    sample_steps: int,
    seed: int,
) -> np.ndarray:
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
        q1, q2, q3, q4 = _rates(mu1, mu2, h, sigma, scheme_id)

        dx = np.random.poisson(q1 * tau) - np.random.poisson(q2 * tau)
        dy = np.random.poisson(q3 * tau) - np.random.poisson(q4 * tau)

        x += dx * h
        y += dy * h

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
    return counts / (total * h * h)


def true_invariant_density(n: int, sigma: float, lowx: float, lowy: float, span: float) -> np.ndarray:
    h = span / n
    x = np.linspace(lowx + h / 2.0, lowx + span - h / 2.0, n)
    y = np.linspace(lowy + h / 2.0, lowy + span - h / 2.0, n)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    v = (xx * xx + yy * yy - 1.0) ** 2
    rho = np.exp(-2.0 * v / (sigma * sigma))
    rho /= np.sum(rho) * h * h
    return rho


def l1_error(pi_hat: np.ndarray, pi_true: np.ndarray, h: float) -> float:
    return float(np.sum(np.abs(pi_hat - pi_true)) * h * h)


def local_slope(h: np.ndarray, e: np.ndarray) -> np.ndarray:
    return np.log(e[1:] / e[:-1]) / np.log(h[1:] / h[:-1])


def plot_accuracy(h_vals: np.ndarray, e_qu: np.ndarray, e_qc: np.ndarray, e_qtu: np.ndarray, out: Path) -> None:
    i0 = 2
    c1 = 0.8 * e_qu[i0] / h_vals[i0]
    c2 = 0.7 * e_qc[i0] / (h_vals[i0] ** 2)

    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.loglog(h_vals, e_qu, "o-", color="#0072BD", mfc="none", lw=0.9, ms=8, label=r"$Q_u$")
    ax.loglog(h_vals, c1 * h_vals, "--", color="#D95319", dashes=(7, 5), lw=0.9, label=r"$O(h)$")
    ax.loglog(h_vals, e_qc, "o-", color="#EDB120", mfc="none", lw=0.9, ms=8, label=r"$Q_c$")
    ax.loglog(h_vals, c2 * (h_vals ** 2), "--", color="#7E2F8E", dashes=(7, 4), lw=0.9, label=r"$O(h^2)$")
    ax.loglog(h_vals, e_qtu, "x-", color="#77AC30", lw=0.8, ms=7, mew=0.8, label=r"$\tilde{Q}_u$")

    ax.set_title("Stationary Density Accuracy")
    ax.set_xlabel("spatial stepsize h")
    ax.set_ylabel(r"$l^1$-error")
    ax.grid(False)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")
    plt.show()


def main() -> None:
    # From paper images: Omega=[-2,2]^2, sigma=0.15, h=4/N, N in {64,128,256,512,1024}
    sigma = 0.15
    lowx = -2.0
    lowy = -2.0
    span = 4.0
    n_list = [64, 128, 256, 512, 1024]
    h_vals = span / np.array(n_list, dtype=np.float64)

    # v5-style tau-leaping occupancy pipeline
    tau = 0.1
    t_burn = 2_000_000.0
    t_sample = 5_000_000.0
    burn_steps = int(round(t_burn / tau))
    sample_steps = int(round(t_sample / tau))
    n_rep = 1

    print("Ring density test with v5 algorithm pipeline")
    print(f"sigma={sigma}, domain=[{lowx},{lowx+span}]x[{lowy},{lowy+span}]")
    print(f"N={n_list}, h={h_vals}")
    print(f"tau={tau}, burn_steps={burn_steps}, sample_steps={sample_steps}, n_rep={n_rep}")

    _ = simulate_stationary_histogram(16, 2, sigma, lowx, lowy, span, tau, 10, 20, seed=1)

    # Table 3 l1-error values for direct reference/validation.
    table_qu = np.array([0.1964540, 0.1031750, 0.0530338, 0.0269501, 0.0135845], dtype=np.float64)
    table_qc = np.array([0.0406613, 0.0105840, 0.0026700, 0.0007017, 0.0003052], dtype=np.float64)
    table_qtu = np.array([0.0373315, 0.0087193, 0.0021627, 0.0005804, 0.0002810], dtype=np.float64)

    run_sim = True
    if run_sim:
        err = np.zeros((3, len(n_list)), dtype=np.float64)
        for i, n in enumerate(n_list):
            h = span / n
            rho_true = true_invariant_density(n, sigma, lowx, lowy, span)
            for scheme_id in range(3):
                e_sum = 0.0
                for r in range(n_rep):
                    pi_hat = simulate_stationary_histogram(
                        n,
                        scheme_id,
                        sigma,
                        lowx,
                        lowy,
                        span,
                        tau,
                        burn_steps,
                        sample_steps,
                        seed=100000 * (scheme_id + 1) + 1000 * n + r,
                    )
                    e_sum += l1_error(pi_hat, rho_true, h)
                err[scheme_id, i] = e_sum / n_rep
                print(f"N={n:4d}, scheme={scheme_id}, L1={err[scheme_id, i]:.6e}")

        e_qu, e_qc, e_qtu = err[0], err[1], err[2]
        print("local slopes Qu:", local_slope(h_vals, e_qu))
        print("local slopes Qc:", local_slope(h_vals, e_qc))
        print("local slopes Qut:", local_slope(h_vals, e_qtu))
    else:
        e_qu, e_qc, e_qtu = table_qu, table_qc, table_qtu

    out = Path(__file__).resolve().parent / "v6_stationary_density_accuracy.png"
    plot_accuracy(h_vals, e_qu, e_qc, e_qtu, out)


if __name__ == "__main__":
    main()
