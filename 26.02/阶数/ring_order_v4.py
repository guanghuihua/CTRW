from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def ring_density_xy(x0: float, x1: float) -> tuple[float, float]:
    mu1 = -4.0 * x0 * (x0 * x0 + x1 * x1 - 1.0) + x1
    mu2 = -4.0 * x1 * (x0 * x0 + x1 * x1 - 1.0) - x0
    return mu1, mu2


@nb.njit(cache=True, fastmath=True)
def rates_for_scheme(mu1: float, mu2: float, h: float, tau: float, eps0: float, scheme_id: int) -> tuple[float, float, float, float]:
    """
    Return tau-scaled jump rates (q1,q2,q3,q4) for x+/x-/y+/y-.
    """
    m = 0.5 * eps0 * eps0
    if scheme_id == 0:
        q1 = tau * (max(mu1, 0.0) / h + m / (h * h))
        q2 = tau * (-min(mu1, 0.0) / h + m / (h * h))
        q3 = tau * (max(mu2, 0.0) / h + m / (h * h))
        q4 = tau * (-min(mu2, 0.0) / h + m / (h * h))
    elif scheme_id == 1:
        coef = tau * m / (h * h)
        q1 = coef * np.exp((mu1 * h) / (2.0 * m))
        q2 = coef * np.exp(-(mu1 * h) / (2.0 * m))
        q3 = coef * np.exp((mu2 * h) / (2.0 * m))
        q4 = coef * np.exp(-(mu2 * h) / (2.0 * m))
    else:
        m1 = max(2.0 - abs(mu1) * h, 0.0) / 2.0
        m2 = max(2.0 - abs(mu2) * h, 0.0) / 2.0
        q1 = tau * (max(mu1, 0.0) / h + m1 / (h * h))
        q2 = tau * (-min(mu1, 0.0) / h + m1 / (h * h))
        q3 = tau * (max(mu2, 0.0) / h + m2 / (h * h))
        q4 = tau * (-min(mu2, 0.0) / h + m2 / (h * h))
    # Keep rates positive for Poisson difference (Skellam equivalent).
    tiny = 1e-12
    q1 = max(q1, tiny)
    q2 = max(q2, tiny)
    q3 = max(q3, tiny)
    q4 = max(q4, tiny)
    return q1, q2, q3, q4


@nb.njit(cache=True, fastmath=True)
def _project_scalar_x(v: float, low: float, h: float) -> float:
    return low + h + round((v - low - h) / h) * h


@nb.njit(cache=True, fastmath=True)
def one_path_error_numba(
    h: float,
    tau: float,
    end_time: float,
    eps0: float,
    lowx: float,
    lowy: float,
    span: float,
    scheme_id: int,
    seed: int,
) -> float:
    """
    Coupled pathwise error between step h and 2h dynamics, using shared uniforms.
    """
    # Initialize on 2h grid then copy to h path (same idea as Ringdensity_FiniteTimeError.py).
    np.random.seed(seed)
    x2_0 = lowx + h + round((1.0 - lowx - h) / (2.0 * h)) * 2.0 * h
    x2_1 = lowy + h + round((1.0 - lowy - h) / (2.0 * h)) * 2.0 * h
    xh_0 = x2_0
    xh_1 = x2_1

    t = 0.0
    while t < end_time:
        mu1_h, mu2_h = ring_density_xy(xh_0, xh_1)
        q1, q2, q3, q4 = rates_for_scheme(mu1_h, mu2_h, h, tau, eps0, scheme_id)

        mu1_2h, mu2_2h = ring_density_xy(x2_0, x2_1)
        qq1, qq2, qq3, qq4 = rates_for_scheme(mu1_2h, mu2_2h, 2.0 * h, tau, eps0, scheme_id)

        # Skellam via difference of two Poisson variables.
        dx_h = np.random.poisson(q1) - np.random.poisson(q2)
        dy_h = np.random.poisson(q3) - np.random.poisson(q4)
        dx_2h = np.random.poisson(qq1) - np.random.poisson(qq2)
        dy_2h = np.random.poisson(qq3) - np.random.poisson(qq4)

        xh_0 += dx_h * h
        xh_1 += dy_h * h
        x2_0 += dx_2h * (2.0 * h)
        x2_1 += dy_2h * (2.0 * h)

        # Clip to domain and reproject to corresponding grid.
        xh_0 = min(max(xh_0, lowx + h), lowx + span - h)
        xh_1 = min(max(xh_1, lowy + h), lowy + span - h)
        x2_0 = min(max(x2_0, lowx + h), lowx + span - h)
        x2_1 = min(max(x2_1, lowy + h), lowy + span - h)
        xh_0 = _project_scalar_x(xh_0, lowx, h)
        xh_1 = _project_scalar_x(xh_1, lowy, h)
        x2_0 = lowx + h + round((x2_0 - lowx - h) / (2.0 * h)) * 2.0 * h
        x2_1 = lowy + h + round((x2_1 - lowy - h) / (2.0 * h)) * 2.0 * h
        t += tau

    return abs(xh_0 - x2_0) + abs(xh_1 - x2_1)


@nb.njit(cache=True, fastmath=True)
def estimate_error_curve_numba(
    n_list: list[int],
    count: int,
    tau: float,
    end_time: float,
    eps0: float,
    lowx: float,
    lowy: float,
    span: float,
    scheme_id: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    h_vals = np.array([span / n for n in n_list], dtype=np.float64)
    errs = np.zeros_like(h_vals)
    for i, h in enumerate(h_vals):
        e_sum = 0.0
        for r in range(count):
            e_sum += one_path_error_numba(
                h, tau, end_time, eps0, lowx, lowy, span, scheme_id, seed + i * 100000 + r
            )
        errs[i] = e_sum / count
    return h_vals, errs


def local_slope(h: np.ndarray, e: np.ndarray) -> np.ndarray:
    return np.log(e[1:] / e[:-1]) / np.log(h[1:] / h[:-1])


def main() -> None:
    # Finite-time strong-error experiment (reference: Ringdensity_FiniteTimeError.py).
    n_list = [64, 128, 256, 512]
    count = 300
    eps0 = 0.5
    lowx = -2.0
    lowy = -2.0
    span = 4.0
    end_time = 10.0
    tau = 0.1

    # numba warmup (tiny run)
    _ = one_path_error_numba(span / n_list[0], tau, 0.2, eps0, lowx, lowy, span, 0, 1)
    h, err_qu = estimate_error_curve_numba(
        n_list, count, tau, end_time, eps0, lowx, lowy, span, 0, seed=123
    )
    _, err_qc = estimate_error_curve_numba(
        n_list, count, tau, end_time, eps0, lowx, lowy, span, 1, seed=456
    )
    _, err_qut = estimate_error_curve_numba(
        n_list, count, tau, end_time, eps0, lowx, lowy, span, 2, seed=789
    )
    for i, n in enumerate(n_list):
        print(f"[Qu ] N={n:4d}, h={h[i]:.6e}, mean error={err_qu[i]:.6e}")
    for i, n in enumerate(n_list):
        print(f"[Qc ] N={n:4d}, h={h[i]:.6e}, mean error={err_qc[i]:.6e}")
    for i, n in enumerate(n_list):
        print(f"[Qut] N={n:4d}, h={h[i]:.6e}, mean error={err_qut[i]:.6e}")

    s_qu = local_slope(h, err_qu)
    s_qc = local_slope(h, err_qc)
    s_qut = local_slope(h, err_qut)
    print("local slope Qu :", s_qu)
    print("local slope Qc :", s_qc)
    print("local slope Qut:", s_qut)

    # Reference lines aligned near middle point.
    i0 = 1
    c1 = 0.6 * err_qu[i0] / h[i0]
    c2 = 0.6 * err_qut[i0] / (h[i0] ** 2)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.loglog(h, err_qu, "o-", color="#0072E3", mfc="none", lw=0.9, ms=7, label=r"$Q_u$")
    ax.loglog(h, err_qc, "o-", color="#E69F00", mfc="none", lw=0.9, ms=7, label=r"$Q_c$")
    ax.loglog(h, err_qut, "s-", color="#2CA02C", mfc="none", lw=0.9, ms=6, label=r"$\tilde{Q}_u$")
    ax.loglog(h, c1 * h, "--", color="#D95319", dashes=(7, 5), lw=0.9, label=r"$O(h)$")
    ax.loglog(h, c2 * h**2, "--", color="#7E2F8E", dashes=(7, 4), lw=0.9, label=r"$O(h^2)$")

    ax.set_title("RingDensity Finite-Time Error (Three Schemes)")
    ax.set_xlabel("spatial stepsize h")
    ax.set_ylabel("relative/absolute error")
    ax.grid(False)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black")

    out = Path(__file__).resolve().parent / "ring_order_v4_finite_time_error.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
