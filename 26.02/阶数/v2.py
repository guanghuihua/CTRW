from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Try to use scipy if available; fall back to dense solves if not
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

save_dir = Path(__file__).resolve().parent  # 当前.py文件所在文件夹

def U(x):
    return x**4 / 4.0

def mu(x):
    return -x**3

M = 1.0  # sigma^2/2 for sigma = sqrt(2)

def build_Q_u(x, dx):
    n = len(x)
    mup = np.maximum(mu(x), 0.0)
    mum = np.maximum(-mu(x), 0.0)
    up = mup/dx + M/dx**2
    um = mum/dx + M/dx**2
    main = -(up + um)

    # boundary adjustments: no jump outside the domain
    up[-1] = 0.0
    um[0]  = 0.0
    main[0]  = -(up[0] + um[0])
    main[-1] = -(up[-1] + um[-1])

    if SCIPY_OK:
        Q = sp.diags([um[1:], main, up[:-1]], offsets=[-1,0,1], format="csc")
    else:
        Q = np.zeros((n,n))
        for i in range(n):
            Q[i,i] = main[i]
            if i+1 < n: Q[i,i+1] = up[i]
            if i-1 >= 0: Q[i,i-1] = um[i]
    return Q

def build_Q_c(x, dx):
    n = len(x)
    m = mu(x)
    up = (M/dx**2) * np.exp( (m*dx)/(2*M) )
    um = (M/dx**2) * np.exp( -(m*dx)/(2*M) )
    main = -(up + um)

    # boundary: no jump outside
    up[-1] = 0.0
    um[0]  = 0.0
    main[0]  = -(up[0] + um[0])
    main[-1] = -(up[-1] + um[-1])

    if SCIPY_OK:
        Q = sp.diags([um[1:], main, up[:-1]], offsets=[-1,0,1], format="csc")
    else:
        Q = np.zeros((n,n))
        for i in range(n):
            Q[i,i] = main[i]
            if i+1 < n: Q[i,i+1] = up[i]
            if i-1 >= 0: Q[i,i-1] = um[i]
    return Q

def stationary_density(Q, dx):
    n = Q.shape[0]
    if SCIPY_OK:
        QT = Q.T.tolil()
        b = np.zeros(n)
        QT[0,:] = dx  # normalization row: sum v_i dx = 1
        b[0] = 1.0
        v = spla.spsolve(QT.tocsc(), b)
    else:
        QT = Q.T.copy()
        b = np.zeros(n)
        QT[0,:] = dx
        b[0] = 1.0
        v = np.linalg.solve(QT, b)
    v = np.maximum(v, 0.0)
    v /= (v.sum() * dx)
    return v

def mfpt_exit(Q, x, a=0.0, bnd=2.0):
    n = len(x)
    inside = np.where((x > a) & (x < bnd))[0]
    outside = np.where((x <= a) | (x >= bnd))[0]
    tau = np.zeros(n)
    if SCIPY_OK:
        Qii = Q[inside[:,None], inside].tocsc()
        rhs = -np.ones(len(inside))
        tau_inside = spla.spsolve(Qii, rhs)
    else:
        Qii = Q[np.ix_(inside, inside)]
        rhs = -np.ones(len(inside))
        tau_inside = np.linalg.solve(Qii, rhs)
    tau[inside] = tau_inside
    tau[outside] = 0.0
    return tau

def committor(Q, x, a=0.0, bnd=2.0):
    n = len(x)
    q = np.zeros(n)
    iL = np.argmin(np.abs(x-a))
    iR = np.argmin(np.abs(x-bnd))
    q[iL] = 0.0
    q[iR] = 1.0
    inside = np.array([i for i in range(n) if i not in (iL,iR) and (x[i] > a) and (x[i] < bnd)])
    fixed = np.array([iL, iR])
    if SCIPY_OK:
        Qii = Q[inside[:,None], inside].tocsc()
        Qif = Q[inside[:,None], fixed].tocsc()
        rhs = -(Qif @ q[fixed])
        q_inside = spla.spsolve(Qii, rhs)
    else:
        Qii = Q[np.ix_(inside, inside)]
        Qif = Q[np.ix_(inside, fixed)]
        rhs = -(Qif @ q[fixed])
        q_inside = np.linalg.solve(Qii, rhs)
    q[inside] = q_inside
    return q

# Reference solutions on [0,2] using fine-grid integral formulas
def reference_committor_and_mfpt(num=200001):
    y = np.linspace(0.0, 2.0, num)
    dy = y[1]-y[0]
    expU = np.exp(U(y))
    expmU = np.exp(-U(y))

    # S(y) = int_0^y exp(U)
    S = np.cumsum((expU[:-1] + expU[1:]) * 0.5 * dy)
    S = np.concatenate([[0.0], S])
    S2 = S[-1]

    q = S / S2

    # A(y)=int_0^y S(z) exp(-U(z)) dz
    integrandA = S * expmU
    A = np.cumsum((integrandA[:-1] + integrandA[1:]) * 0.5 * dy)
    A = np.concatenate([[0.0], A])

    # B(y)=int_0^y (S2 - S(z)) exp(-U(z)) dz
    integrandB = (S2 - S) * expmU
    B = np.cumsum((integrandB[:-1] + integrandB[1:]) * 0.5 * dy)
    B = np.concatenate([[0.0], B])
    B2 = B[-1]

    T = ((S2 - S)/S2) * A + (S/S2) * (B2 - B)
    return y, q, T

# Reference stationary density on [-L,L] (truncated and renormalized)
def reference_density(L, num=400001):
    y = np.linspace(-L, L, num)
    dy = y[1]-y[0]
    w = np.exp(-U(y))
    Z = np.trapz(w, y)
    nu = w / Z
    return y, nu

# Compute errors for a list of dx values
dx_list = np.array([0.4, 0.2, 0.1, 0.05, 0.025])

# Precompute references
y_ref, q_ref, T_ref = reference_committor_and_mfpt(num=200001)
# density reference uses L that we pick
L = 8.0
yd_ref, nu_ref = reference_density(L, num=200001)

def interp_ref(x, y, f):
    return np.interp(x, y, f)

# Error containers
err_density_u, err_density_c = [], []
err_mfpt_u, err_mfpt_c = [], []
err_comm_u, err_comm_c = [], []

for dx in dx_list:
    # --- Density on [-L, L] ---
    xD = np.arange(-L, L + 0.5*dx, dx)
    QDu = build_Q_u(xD, dx)
    QDc = build_Q_c(xD, dx)
    nu_u = stationary_density(QDu, dx)
    nu_c = stationary_density(QDc, dx)
    nu_t = interp_ref(xD, yd_ref, nu_ref)
    # L1 error
    err_density_u.append(np.sum(np.abs(nu_u - nu_t)) * dx)
    err_density_c.append(np.sum(np.abs(nu_c - nu_t)) * dx)

    # --- MFPT and Committor on [0,2] ---
    n = int(round(2.0/dx)) + 1
    x = np.linspace(0.0, 2.0, n)  # ensures boundary alignment
    Qu = build_Q_u(x, dx)
    Qc = build_Q_c(x, dx)

    # Make boundaries absorbing for exit problems
    if SCIPY_OK:
        Qu = Qu.tolil()
        Qc = Qc.tolil()
        Qu[0,:] = 0.0; Qu[-1,:] = 0.0
        Qc[0,:] = 0.0; Qc[-1,:] = 0.0
        Qu = Qu.tocsc()
        Qc = Qc.tocsc()
    else:
        Qu[0,:] = 0.0; Qu[-1,:] = 0.0
        Qc[0,:] = 0.0; Qc[-1,:] = 0.0

    T_u = mfpt_exit(Qu, x, 0.0, 2.0)
    T_c = mfpt_exit(Qc, x, 0.0, 2.0)
    q_u = committor(Qu, x, 0.0, 2.0)
    q_c = committor(Qc, x, 0.0, 2.0)

    T_t = interp_ref(x, y_ref, T_ref)
    q_t = interp_ref(x, y_ref, q_ref)

    # relative L2 errors on interior (avoid trivial boundary values)
    interior = (x > 0.0) & (x < 2.0)
    def rel_l2(a, b):
        num = np.linalg.norm((a-b)[interior])
        den = np.linalg.norm(b[interior])
        return num/den

    err_mfpt_u.append(rel_l2(T_u, T_t))
    err_mfpt_c.append(rel_l2(T_c, T_t))
    err_comm_u.append(rel_l2(q_u, q_t))
    err_comm_c.append(rel_l2(q_c, q_t))

err_density_u = np.array(err_density_u)
err_density_c = np.array(err_density_c)
err_mfpt_u = np.array(err_mfpt_u)
err_mfpt_c = np.array(err_mfpt_c)
err_comm_u = np.array(err_comm_u)
err_comm_c = np.array(err_comm_c)

# Helper to plot order reference lines
def plot_accuracy(dx, e_u, e_c, title, ylabel, outfile):
    plt.figure()
    plt.loglog(dx, e_u, marker='o', linestyle='-', label=r'$\widetilde Q_u$')
    plt.loglog(dx, e_c, marker='x', linestyle='-', label=r'$\widetilde Q_c$')

    # reference slopes scaled to first point
    c1 = e_u[0] / dx[0]
    c2 = e_c[0] / (dx[0]**2)
    plt.loglog(dx, c1*dx, linestyle='--', label=r'$O(\delta x)$')
    plt.loglog(dx, c2*(dx**2), linestyle='--', label=r'$O(\delta x^2)$')

    plt.gca().invert_xaxis()  # match typical "smaller dx to the right" style? (optional)
    plt.title(title)
    plt.xlabel('spatial stepsize')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.show()

plot_accuracy(dx_list, err_density_u, err_density_c,
              'Stationary Density Accuracy (1D Cubic Oscillator)',
              'L1 error', save_dir/'fig_density_accuracy.png')

plot_accuracy(dx_list, err_mfpt_u, err_mfpt_c,
              'MFPT Accuracy: exit from (0,2)',
              'relative L2 error', save_dir/'fig_mfpt_accuracy.png')
plot_accuracy(dx_list, err_comm_u, err_comm_c,
              'Committor Accuracy: hit 2 before 0',
              'relative L2 error', save_dir/'fig_committor_accuracy.png')

# Also generate a sample path comparison via SSA for one dx
def ssa_path(Q, x, x0, T_end, seed=0):
    rng = np.random.default_rng(seed)
    # map x0 to nearest grid point
    i = int(np.argmin(np.abs(x - x0)))
    t = 0.0
    ts = [t]
    xs = [x[i]]

    # pre-extract neighbor rates for speed (tridiagonal assumption)
    while t < T_end:
        # rates to neighbors
        if SCIPY_OK:
            # read from sparse row
            row = Q.getrow(i).toarray().ravel()
            lam_p = row[i+1] if i+1 < len(x) else 0.0
            lam_m = row[i-1] if i-1 >= 0 else 0.0
        else:
            lam_p = Q[i, i+1] if i+1 < len(x) else 0.0
            lam_m = Q[i, i-1] if i-1 >= 0 else 0.0
        lam = lam_p + lam_m
        if lam <= 0:
            break
        tau = rng.exponential(1.0/lam)
        t_next = t + tau
        if t_next > T_end:
            break
        if rng.random() < lam_p/lam:
            i = min(i+1, len(x)-1)
        else:
            i = max(i-1, 0)
        t = t_next
        ts.append(t)
        xs.append(x[i])
    return np.array(ts), np.array(xs)

dx_path = 0.25
x_path_grid = np.arange(-8.0, 8.0 + 0.5*dx_path, dx_path)
Qu_path = build_Q_u(x_path_grid, dx_path)
Qc_path = build_Q_c(x_path_grid, dx_path)

T_end = 100.0
ts_u, xs_u = ssa_path(Qu_path, x_path_grid, x0=10.0, T_end=T_end, seed=1)
ts_c, xs_c = ssa_path(Qc_path, x_path_grid, x0=10.0, T_end=T_end, seed=2)

plt.figure()
plt.step(ts_u, xs_u, where='post', label=r'$\widetilde Q_u$-SSA')
plt.step(ts_c, xs_c, where='post', label=r'$\widetilde Q_c$-SSA')
plt.title('Sample Paths via SSA (time-continuous)')
plt.xlabel('t')
plt.ylabel('Y(t)')
plt.legend()
plt.tight_layout()
plt.savefig(save_dir/'fig_sample_paths_ssa.png', dpi=200)
plt.show()
