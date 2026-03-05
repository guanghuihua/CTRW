from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import numba as nb
except Exception:  # pragma: no cover
    class _NB:
        @staticmethod
        def njit(*_args, **_kwargs):
            def deco(func):
                return func
            return deco

    nb = _NB()


@nb.njit(cache=True)
def U_prime(x: float) -> float:
    # U(x) = x^2 / 2
    return x


@nb.njit(cache=True)
def bernoulli_b(z: float) -> float:
    # B(z) = z / (exp(z) - 1), stable near z=0
    az = abs(z)
    if az < 1e-8:
        return 1.0 - 0.5 * z + (z * z) / 12.0
    return z / np.expm1(z)


@nb.njit(cache=True)
def rates_mixed_qc_qu(
    x: float,
    v: float,
    h: float,
    gamma: float,
    sigma: float,
    kappa_x: float,
) -> tuple[float, float, float, float]:
    """
    Mixed rates for underdamped Langevin:
    - x direction (no noise): Q_c-like flux (with adaptive numerical diffusion)
    - v direction (has noise): Q_u (upwind + physical diffusion)
    """
    # ----- x direction: Q_c-like -----
    a = v
    # No physical diffusion in x; add small adaptive viscosity for realizability.
    d_x = 0.5 * kappa_x * abs(a) * h + 1e-14
    z = a * h / d_x
    r_xp = (d_x / (h * h)) * bernoulli_b(-z)
    r_xm = (d_x / (h * h)) * bernoulli_b(z)

    # ----- v direction: Q_u -----
    b = -(U_prime(x) + gamma * v)
    d_v = 0.5 * sigma * sigma
    r_vp = max(b, 0.0) / h + d_v / (h * h)
    r_vm = max(-b, 0.0) / h + d_v / (h * h)

    return r_xp, r_xm, r_vp, r_vm


@nb.njit(cache=True)
def simulate_ssa_stationary_density(
    n: int,
    l_x: float,
    l_v: float,
    gamma: float,
    sigma: float,
    t_burn: float,
    t_sample: float,
    kappa_x: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    np.random.seed(seed)

    x_grid = np.linspace(-l_x, l_x, n)
    v_grid = np.linspace(-l_v, l_v, n)
    h = x_grid[1] - x_grid[0]

    # Start from center
    ix = n // 2
    iv = n // 2

    occ = np.zeros((n, n), dtype=np.float64)

    t = 0.0
    t_end = t_burn + t_sample

    while t < t_end:
        x = x_grid[ix]
        v = v_grid[iv]

        r_xp, r_xm, r_vp, r_vm = rates_mixed_qc_qu(x, v, h, gamma, sigma, kappa_x)

        # Reflecting boundaries
        if ix == n - 1:
            r_xp = 0.0
        if ix == 0:
            r_xm = 0.0
        if iv == n - 1:
            r_vp = 0.0
        if iv == 0:
            r_vm = 0.0

        r_sum = r_xp + r_xm + r_vp + r_vm
        if r_sum <= 0.0:
            break

        u = np.random.random()
        dt = -np.log(max(u, 1e-15)) / r_sum

        # Time-weighted occupancy estimator
        if t >= t_burn:
            occ[ix, iv] += dt

        t += dt

        u2 = np.random.random() * r_sum
        if u2 < r_xp:
            ix += 1
        elif u2 < r_xp + r_xm:
            ix -= 1
        elif u2 < r_xp + r_xm + r_vp:
            iv += 1
        else:
            iv -= 1

    total = np.sum(occ)
    if total <= 0.0:
        return np.zeros((n, n), dtype=np.float64), h

    rho_hat = occ / (total * h * h)
    return rho_hat, h


def true_invariant_density(
    n: int,
    l_x: float,
    l_v: float,
    gamma: float,
    sigma: float,
) -> tuple[np.ndarray, float]:
    x = np.linspace(-l_x, l_x, n)
    v = np.linspace(-l_v, l_v, n)
    h = x[1] - x[0]

    beta = 2.0 * gamma / (sigma * sigma)
    x_mesh, v_mesh = np.meshgrid(x, v, indexing="ij")
    u = 0.5 * x_mesh * x_mesh
    rho = np.exp(-beta * (u + 0.5 * v_mesh * v_mesh))
    rho /= np.sum(rho) * h * h

    return rho, h


def l1_error(rho_hat: np.ndarray, rho_true: np.ndarray, h: float) -> float:
    return float(np.sum(np.abs(rho_hat - rho_true)) * h * h)


def fit_order_tail(h_vals: np.ndarray, err_vals: np.ndarray, tail_points: int = 3) -> float:
    x = np.log(h_vals[-tail_points:])
    y = np.log(err_vals[-tail_points:])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def main() -> None:
    # Model params
    gamma = 1.0
    sigma = 1.0

    # Truncation domain
    l_x = 4.0
    l_v = 4.0

    # Q_c-like strength in x direction (no-noise direction)
    kappa_x = 1.0

    # Grid refinement
    n_list = [41, 61, 81, 101, 141]

    # SSA horizon
    t_burn = 100.0
    t_sample = 400.0
    n_rep = 4

    # Warm-up (JIT)
    _ = rates_mixed_qc_qu(0.0, 0.0, 0.1, gamma, sigma, kappa_x)

    h_vals = np.zeros(len(n_list), dtype=np.float64)
    err_vals = np.zeros(len(n_list), dtype=np.float64)

    print("Running mixed SSA: x-Q_c-like, v-Q_u")
    print(f"gamma={gamma}, sigma={sigma}, domain=[{-l_x},{l_x}]x[{-l_v},{l_v}]")
    print(f"t_burn={t_burn}, t_sample={t_sample}, n_rep={n_rep}, kappa_x={kappa_x}")

    for i, n in enumerate(n_list):
        rho_true, h = true_invariant_density(n, l_x, l_v, gamma, sigma)

        e_sum = 0.0
        for r in range(n_rep):
            rho_hat, _ = simulate_ssa_stationary_density(
                n=n,
                l_x=l_x,
                l_v=l_v,
                gamma=gamma,
                sigma=sigma,
                t_burn=t_burn,
                t_sample=t_sample,
                kappa_x=kappa_x,
                seed=10000 * n + 37 * r + 1,
            )
            e_sum += l1_error(rho_hat, rho_true, h)

        h_vals[i] = h
        err_vals[i] = e_sum / n_rep
        print(f"N={n:4d}, h={h:.6e}, L1 error={err_vals[i]:.6e}")

    # Reference slopes
    i_anchor = len(h_vals) - 2
    c1 = 0.8 * err_vals[i_anchor] / h_vals[i_anchor]
    c2 = 0.8 * err_vals[i_anchor] / (h_vals[i_anchor] ** 2)
    ref1 = c1 * h_vals
    ref2 = c2 * (h_vals ** 2)

    local_slopes = np.log(err_vals[1:] / err_vals[:-1]) / np.log(h_vals[1:] / h_vals[:-1])
    tail_order = fit_order_tail(h_vals, err_vals, tail_points=3)

    print("Local slopes:", local_slopes)
    print("Tail fitted order (last 3 points):", tail_order)

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.loglog(h_vals, err_vals, "o-", color="#1f77b4", mfc="none", lw=1.0, ms=7,
              label="mixed scheme (x: Q_c-like, v: Q_u)")
    ax.loglog(h_vals, ref1, "--", color="#d95319", dashes=(7, 5), lw=0.9, label=r"$O(h)$")
    ax.loglog(h_vals, ref2, "--", color="#7e2f8e", dashes=(7, 4), lw=0.9, label=r"$O(h^2)$")

    ax.set_title("Underdamped Langevin Invariant Density Accuracy")
    ax.set_xlabel("grid size h")
    ax.set_ylabel(r"$L^1$ error")
    ax.grid(False)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black")

    out_png = Path(__file__).resolve().parent / "mix_v5_accuracy.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    print(f"Saved figure: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
