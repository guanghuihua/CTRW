from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numba as nb
import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


@nb.njit(cache=True, fastmath=True)
def ring_density_vec(x0: float, x1: float) -> tuple[float, float]:
    """
    User-specified ring model drift:
      y[0] = -4*x*(x^2+y^2-1) + y
      y[1] = -4*y*(x^2+y^2-1) - x
    """
    r2 = x0 * x0 + x1 * x1
    mu1 = -4.0 * x0 * (r2 - 1.0) + x1
    mu2 = -4.0 * x1 * (r2 - 1.0) - x0
    return mu1, mu2


@nb.njit(cache=True, fastmath=True)
def build_rates_flat(
    xg: np.ndarray, yg: np.ndarray, sigma: float, scheme: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    scheme:
      0 -> Q_u (upwind + constant diffusion)
      1 -> Q_c (exponential fitted)
      2 -> \\tilde{Q}_u (user requested m1,m2 correction)
    """
    n = xg.size
    h = xg[1] - xg[0]
    m = 0.5 * sigma * sigma
    diff = m / (h * h)

    rxp = np.zeros((n, n), dtype=np.float64)
    rxm = np.zeros((n, n), dtype=np.float64)
    ryp = np.zeros((n, n), dtype=np.float64)
    rym = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        x = xg[i]
        for j in range(n):
            y = yg[j]
            mu1, mu2 = ring_density_vec(x, y)
            if scheme == 0:
                # Q_u
                qxp = max(mu1, 0.0) / h + diff
                qxm = max(-mu1, 0.0) / h + diff
                qyp = max(mu2, 0.0) / h + diff
                qym = max(-mu2, 0.0) / h + diff
            elif scheme == 1:
                # Q_c
                coef = m / (h * h)
                qxp = coef * np.exp((mu1 * h) / (2.0 * m))
                qxm = coef * np.exp(-(mu1 * h) / (2.0 * m))
                qyp = coef * np.exp((mu2 * h) / (2.0 * m))
                qym = coef * np.exp(-(mu2 * h) / (2.0 * m))
            else:
                # \tilde{Q}_u (exactly as requested)
                m1 = max(2.0 - abs(mu1) * h, 0.0) / 2.0
                m2 = max(2.0 - abs(mu2) * h, 0.0) / 2.0
                qxp = max(mu1, 0.0) / h + m1 / (h * h)
                qxm = max(-mu1, 0.0) / h + m1 / (h * h)
                qyp = max(mu2, 0.0) / h + m2 / (h * h)
                qym = max(-mu2, 0.0) / h + m2 / (h * h)

            rxp[i, j] = qxp
            rxm[i, j] = qxm
            ryp[i, j] = qyp
            rym[i, j] = qym

    return rxp, rxm, ryp, rym


def build_generator(
    xg: np.ndarray, yg: np.ndarray, sigma: float, scheme: int
) -> sp.csc_matrix:
    n = xg.size
    rxp, rxm, ryp, rym = build_rates_flat(xg, yg, sigma, scheme)
    nn = n * n
    q = sp.lil_matrix((nn, nn), dtype=np.float64)

    def idx(i: int, j: int) -> int:
        return i * n + j

    for i in range(n):
        for j in range(n):
            k = idx(i, j)
            qxp = rxp[i, j]
            qxm = rxm[i, j]
            qyp = ryp[i, j]
            qym = rym[i, j]
            diag = 0.0

            if i + 1 < n:
                q[k, idx(i + 1, j)] = qxp
                diag += qxp
            if i - 1 >= 0:
                q[k, idx(i - 1, j)] = qxm
                diag += qxm
            if j + 1 < n:
                q[k, idx(i, j + 1)] = qyp
                diag += qyp
            if j - 1 >= 0:
                q[k, idx(i, j - 1)] = qym
                diag += qym

            q[k, k] = -diag

    return q.tocsc()


def stationary_density_from_q(q: sp.csc_matrix, h: float) -> np.ndarray:
    n = q.shape[0]
    qt = q.T.tolil()
    b = np.zeros(n, dtype=np.float64)
    qt[0, :] = h * h
    b[0] = 1.0
    pi = spla.spsolve(qt.tocsc(), b)
    pi = np.maximum(pi, 0.0)
    s = np.sum(pi) * h * h
    if s <= 0.0:
        raise ValueError("Failed to normalize stationary density.")
    pi /= s
    return pi


def restrict_density(pi_ref_fine: np.ndarray, n_ref: int, n: int) -> np.ndarray:
    if n_ref % n != 0:
        raise ValueError("n_ref must be divisible by n for restriction.")
    r = n_ref // n
    fine2d = pi_ref_fine.reshape((n_ref, n_ref))
    coarse = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        i0 = i * r
        for j in range(n):
            j0 = j * r
            coarse[i, j] = np.mean(fine2d[i0 : i0 + r, j0 : j0 + r])
    return coarse.ravel()


def l1_error_2d(pi_num: np.ndarray, pi_ref: np.ndarray, h: float) -> float:
    return np.sum(np.abs(pi_num - pi_ref)) * h * h


def local_slopes(h: np.ndarray, e: np.ndarray) -> np.ndarray:
    return np.log(e[1:] / e[:-1]) / np.log(h[1:] / h[:-1])


def main() -> None:
    if not SCIPY_OK:
        raise RuntimeError("This script requires scipy (sparse linear solver).")

    # Grid list (coarse -> fine); fine reference uses n_ref.
    n_list = np.array([16, 32, 64, 128], dtype=np.int64)
    n_ref = 256
    sigma = 1.0
    low = -2.0
    span = 4.0

    h_vals = span / n_list
    err_qu = np.zeros_like(h_vals, dtype=np.float64)
    err_qc = np.zeros_like(h_vals, dtype=np.float64)
    err_qut = np.zeros_like(h_vals, dtype=np.float64)

    # Build reference once (using Q_c on fine grid as high-accuracy proxy).
    x_ref = np.linspace(low + span / (2 * n_ref), low + span - span / (2 * n_ref), n_ref)
    y_ref = x_ref.copy()
    print("Building fine reference generator (Q_c) ...")
    q_ref = build_generator(x_ref, y_ref, sigma, scheme=1)
    h_ref = span / n_ref
    pi_ref = stationary_density_from_q(q_ref, h_ref)
    print("Fine reference done.")

    for t, n in enumerate(n_list):
        h = span / n
        x = np.linspace(low + h / 2.0, low + span - h / 2.0, n)
        y = x.copy()

        q_u = build_generator(x, y, sigma, scheme=0)
        q_c = build_generator(x, y, sigma, scheme=1)
        q_ut = build_generator(x, y, sigma, scheme=2)

        pi_u = stationary_density_from_q(q_u, h)
        pi_c = stationary_density_from_q(q_c, h)
        pi_ut = stationary_density_from_q(q_ut, h)
        pi_ref_n = restrict_density(pi_ref, n_ref, n)
        pi_ref_n /= np.sum(pi_ref_n) * h * h

        err_qu[t] = l1_error_2d(pi_u, pi_ref_n, h)
        err_qc[t] = l1_error_2d(pi_c, pi_ref_n, h)
        err_qut[t] = l1_error_2d(pi_ut, pi_ref_n, h)

        print(
            f"N={n:4d}, h={h:.6e}, err_Qu={err_qu[t]:.6e}, "
            f"err_Qc={err_qc[t]:.6e}, err_Qut={err_qut[t]:.6e}"
        )

    s_qu = local_slopes(h_vals, err_qu)
    s_qc = local_slopes(h_vals, err_qc)
    s_qut = local_slopes(h_vals, err_qut)
    print("Local slopes Qu:", s_qu)
    print("Local slopes Qc:", s_qc)
    print("Local slopes Qut:", s_qut)

    # Reference lines aligned at middle point.
    i0 = 1
    c1 = 0.7 * err_qu[i0] / h_vals[i0]
    c2 = 0.7 * err_qut[i0] / (h_vals[i0] ** 2)

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    ax.loglog(h_vals, err_qu, "o-", color="#0072E3", mfc="none", lw=0.8, ms=7, label=r"$Q_u$")
    ax.loglog(h_vals, err_qc, "o-", color="#E69F00", mfc="none", lw=0.8, ms=7, label=r"$Q_c$")
    ax.loglog(
        h_vals, err_qut, "s-", color="#2CA02C", mfc="none", lw=0.8, ms=6, label=r"$\tilde{Q}_u$"
    )
    ax.loglog(h_vals, c1 * h_vals, "--", color="#D95319", dashes=(7, 5), lw=0.8, label=r"$O(h)$")
    ax.loglog(
        h_vals, c2 * h_vals**2, "--", color="#7E2F8E", dashes=(7, 4), lw=0.8, label=r"$O(h^2)$"
    )
    ax.set_title("Ring Stationary Density Accuracy (v3, 2D RingDensity)")
    ax.set_xlabel("spatial stepsize h")
    ax.set_ylabel(r"$l^1$-error")
    ax.grid(False)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black")

    out_path = Path(__file__).resolve().parent / "ring_stationary_density_accuracy_v3.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

