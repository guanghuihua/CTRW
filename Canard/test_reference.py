import numpy as np
from numpy.random import default_rng
import math
import time
import dataclasses
import functools
import matplotlib.pyplot as plt

# =========================
# 1) Problem specification
# =========================

@dataclasses.dataclass
class CanardParams:
    eps: float = 0.01     # slow scale ε
    a: float = 1.0        # canard parameter (fold near a≈1)
    sigma: float = 0.2    # noise strength in fast x
    T: float = 4.0        # final time
    x0: float = -1.5      # initial x (on attracting left branch)
    y0: float = 1.0       # initial y

def drift(z, p: CanardParams):
    """ Drift μ(z) for the canard system. z=(x,y). """
    x, y = z
    fx = x - (x**3)/3.0 - y
    fy = p.eps * (x - p.a)
    return np.array([fx, fy], dtype=float)

def diffusion(z, p: CanardParams):
    """ Diffusion matrix G(z): here only fast variable has noise σ. """
    # Shape (2,1): dW is scalar
    return np.array([[p.sigma],[0.0]], dtype=float)

# Norm helpers
def norm2(v): return float(np.sqrt(np.dot(v, v)))

# ===========================================
# 2) Time-discretization: tamed EM & truncated EM
# ===========================================

def tamed_em_step(z, dt, dW, p: CanardParams):
    mu = drift(z, p)
    denom = 1.0 + dt * norm2(mu)
    z_new = z + (dt * mu) / denom + (diffusion(z, p).flatten() * dW)
    return z_new

def truncation_radius(dt, alpha=0.25, c=1.0):
    # R(Δ) = c * Δ^{-α}
    return c * (dt ** (-alpha))

def pi_trunc(z, R):
    nrm = norm2(z)
    if nrm <= R or nrm == 0.0:
        return z
    return z * (R / nrm)

def truncated_em_step(z, dt, dW, p: CanardParams, alpha=0.25, cR=1.0):
    R = truncation_radius(dt, alpha=alpha, c=cR)
    z_t = pi_trunc(z, R)
    # evaluate drift/diffusion at truncated point
    mu  = drift(z_t, p)
    G   = diffusion(z_t, p).flatten()
    return z + dt * mu + G * dW

def simulate_time_method(method, p: CanardParams, dt, n_paths, rng, ref=False, alpha=0.25, cR=1.0):
    """Simulate n_paths trajectories on [0,T] with time step dt.
       Returns times, paths array shape (n_paths, n_steps+1, 2) and total work counter."""
    n_steps = int(np.ceil(p.T / dt))
    t = np.linspace(0.0, n_steps*dt, n_steps+1)
    X = np.empty((n_paths, n_steps+1, 2), dtype=float)
    X[:,0,:] = np.array([p.x0, p.y0], dtype=float)

    work = 0
    for k in range(n_steps):
        dW = math.sqrt(dt) * rng.standard_normal(size=(n_paths,))
        for i in range(n_paths):
            if method == 'tamed':
                X[i,k+1,:] = tamed_em_step(X[i,k,:], dt, dW[i], p); work += 1
            elif method == 'truncated':
                X[i,k+1,:] = truncated_em_step(X[i,k,:], dt, dW[i], p, alpha=alpha, cR=cR); work += 1
            else:
                raise ValueError("Unknown method")
    return t, X, work

# ===========================================
# 3) Space-discretization: CTRW/SSA with improved Q_u tilde
# ===========================================

@dataclasses.dataclass
class Grid:
    x_min: float; x_max: float; nx: int
    y_min: float; y_max: float; ny: int

    def hx(self): return (self.x_max - self.x_min) / (self.nx - 1)
    def hy(self): return (self.y_max - self.y_min) / (self.ny - 1)
    def clamp(self, ix, iy): # reflecting boundary
        ix = min(max(ix, 0), self.nx-1)
        iy = min(max(iy, 0), self.ny-1)
        return ix, iy

def grid_coords(grid: Grid, ix, iy):
    return np.array([grid.x_min + ix*grid.hx(),
                     grid.y_min + iy*grid.hy()], dtype=float)

def qu_tilde_rates(z, grid: Grid, p: CanardParams):
    """Compute 4 neighbor jump rates using Q_u tilde.
       Diagonal diffusion: only x has σ, y has 0."""
    hx = grid.hx(); hy = grid.hy()
    mu = drift(z, p);  # length 2
    sigx = p.sigma; sigy = 0.0

    # forward/backward steps (use uniform grid => h^+=h^-=h)
    # i=0 -> x, i=1 -> y
    # Nonnegative compensated variances:
    Mx = 0.5 * sigx**2
    My = 0.5 * sigy**2

    # Compensate drift variance per Zu (M^{±} = 0.5 (σ^2 - |μ| h^{±})∨0 )
    Mx_p = max(0.0, 0.5*(sigx**2 - abs(mu[0])*hx))
    Mx_m = max(0.0, 0.5*(sigx**2 - abs(mu[0])*hx))
    My_p = max(0.0, 0.5*(sigy**2 - abs(mu[1])*hy))
    My_m = max(0.0, 0.5*(sigy**2 - abs(mu[1])*hy))

    # Rates (Qu-tilde): q^+ = (μ^+)/h + M^+/(h*h), q^- = ((-μ)^+)/h + M^-/(h*h)
    qx_p = max(mu[0],0.0)/hx + Mx_p/(hx*hx)
    qx_m = max(-mu[0],0.0)/hx + Mx_m/(hx*hx)
    qy_p = max(mu[1],0.0)/hy + My_p/(hy*hy)
    qy_m = max(-mu[1],0.0)/hy + My_m/(hy*hy)

    # To keep positivity in the y-direction when sigma=0, ensure hy>0; if μ_y≈0 rates will be ~0.
    rates = np.array([qx_m, qx_p, qy_m, qy_p], dtype=float) # order: x-, x+, y-, y+
    return rates

def simulate_ctrw(grid: Grid, p: CanardParams, h_factor, T, n_paths, rng):
    """SSA on grid using Q_u~. h_factor is a multiplier on base spacing (1.0 = grid spacing)."""
    hx = grid.hx()*h_factor; hy = grid.hy()*h_factor

    # Effective sub-grid traversal: we keep grid fixed but jump only to immediate neighbors, so h_factor affects rates via hx, hy
    # Start index nearest to initial condition
    ix0 = int(round((p.x0 - grid.x_min)/grid.hx()))
    iy0 = int(round((p.y0 - grid.y_min)/grid.hy()))
    ix0, iy0 = grid.clamp(ix0, iy0)

    # We will record on a uniform time mesh for error calc
    dt_record = 1e-2
    n_rec = int(np.ceil(T/dt_record))
    t_rec = np.linspace(0.0, n_rec*dt_record, n_rec+1)

    paths = np.empty((n_paths, n_rec+1, 2), dtype=float)
    for m in range(n_paths):
        t = 0.0
        ix, iy = ix0, iy0
        z = grid_coords(grid, ix, iy)
        rec_idx = 0
        paths[m, rec_idx, :] = z
        while rec_idx < n_rec:
            # Compute rates at current z but with step sizes (hx, hy) used in rates
            # We emulate h_factor by temporarily overriding grid spacing in the rate call
            # (simple closure copy)
            class Gtmp:
                def __init__(self, base): self.base = base
                def hx(self): return hx
                def hy(self): return hy
            rates = qu_tilde_rates(z, Gtmp(grid), p)
            lam = rates.sum()
            if lam <= 0.0:
                # no movement; advance to next record
                t_next = t_rec[rec_idx+1]
                t = t_next
                rec_idx += 1
                paths[m, rec_idx, :] = z
                continue
            # sample next jump time and which jump
            tau = rng.exponential(1.0/lam)
            # fill record times up to next jump
            while rec_idx < n_rec and t + tau >= t_rec[rec_idx+1]:
                rec_idx += 1
                t = t_rec[rec_idx]
                paths[m, rec_idx, :] = z
            if rec_idx >= n_rec: break
            # perform jump
            probs = rates / lam
            u = rng.random()
            # x-, x+, y-, y+
            if u < probs[0]:
                ix = max(ix-1, 0)
            elif u < probs[0]+probs[1]:
                ix = min(ix+1, grid.nx-1)
            elif u < probs[0]+probs[1]+probs[2]:
                iy = max(iy-1, 0)
            else:
                iy = min(iy+1, grid.ny-1)
            z = grid_coords(grid, ix, iy)
            t += tau
        # done one path
    work = n_paths * n_rec  # simple proxy (rate evals per recorded step)
    return t_rec, paths, work

# ===========================================
# 4) Error estimators & experiment runner
# ===========================================

def strong_error_at_T(paths, T, t_grid_ref, ref_paths):
    # Interpolate (nearest left) the method paths at time T and compute RMS error vs reference
    # paths: (n, n_steps, 2) over a uniform grid t from 0..T_method
    # ref_paths: same n, at t_grid_ref
    n = paths.shape[0]
    # time indices
    def pick_last_idx(t_grid, T):
        idx = np.searchsorted(t_grid, T, side='right') - 1
        return max(0, min(idx, len(t_grid)-1))
    # if 'paths' share the same time grid length, assume last index is T
    # but do robustly:
    # We need method times; infer from length and total T
    t_method = np.linspace(0.0, T, paths.shape[1])
    i_m = pick_last_idx(t_method, T)
    i_r = pick_last_idx(t_grid_ref, T)
    diffs = paths[:, i_m, :] - ref_paths[:, i_r, :]
    return math.sqrt(np.mean(np.sum(diffs*diffs, axis=1)))

def run_experiment(seed=12345):
    rng = default_rng(seed)
    p = CanardParams()

    # Reference for time methods: very fine tamed EM
    dt_ref = 2**-12
    n_ref = 256  # reference Monte Carlo size
    t_ref, Xref, work_ref = simulate_time_method('tamed', p, dt_ref, n_ref, rng)

    # Time methods at multiple Δ
    dts = [2**-k for k in [6,7,8,9,10]]
    errs_tamed = []; errs_trunc = []; work_tamed=[]; work_trunc=[]
    n_mc = 256

    for dt in dts:
        # reuse the same Brownian strings per path across methods by using the same RNG sequence;
        # here we simply resimulate independently; for stricter coupling, one can sub-sample the fine increments.
        t1, Xt, w1 = simulate_time_method('tamed', p, dt, n_mc, rng)
        e1 = strong_error_at_T(Xt, p.T, t_ref, Xref)
        errs_tamed.append(e1); work_tamed.append(w1)

        t2, Xtu, w2 = simulate_time_method('truncated', p, dt, n_mc, rng)
        e2 = strong_error_at_T(Xtu, p.T, t_ref, Xref)
        errs_trunc.append(e2); work_trunc.append(w2)

    # Space method: CTRW with improved Q_u tilde
    # Reference: much finer space (but same SSA recording mesh)
    grid = Grid(x_min=-3.0, x_max=3.0, nx=241, y_min=-1.5, y_max=2.5, ny=161)
    h_ref_factor = 0.25  # rates use 1/4 of base spacing
    t_ctrw_ref, Xctrw_ref, w_ctrw_ref = simulate_ctrw(grid, p, h_ref_factor, p.T, n_ref, rng)

    h_factors = [2.0, 1.0, 0.5, 0.35]
    errs_ctrw = []; work_ctrw=[]
    n_mc_ctrw = 256

    for hf in h_factors:
        t_ctrw, Xc, w = simulate_ctrw(grid, p, hf, p.T, n_mc_ctrw, rng)
        # error vs the reference CTRW (same recording mesh)
        e = strong_error_at_T(Xc, p.T, t_ctrw_ref, Xctrw_ref)
        errs_ctrw.append(e); work_ctrw.append(w)

    # Plot error vs "work"
    plt.figure()
    plt.loglog(work_tamed, errs_tamed, 'o-', label='tamed EM (time)')
    plt.loglog(work_trunc, errs_trunc, 's-', label='truncated EM (time)')
    plt.loglog(work_ctrw, errs_ctrw, '^-', label='CTRW $\\tilde Q_u$ (space)')
    plt.xlabel('work (proxy)')
    plt.ylabel('RMS strong error at T')
    plt.legend()
    plt.title('Canard SDE: error vs work')
    plt.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.show()

    # Optional: visualize a few sample paths to show canard behavior
    kshow = 8
    plt.figure()
    for i in range(kshow):
        plt.plot(Xref[i,:,0], Xref[i,:,1], lw=1, alpha=0.6)
    plt.xlabel('x'); plt.ylabel('y'); plt.title('Reference tamed-EM sample paths (canard)')
    plt.grid(True, ls=':')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()
