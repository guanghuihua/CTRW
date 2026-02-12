import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def mu(x):  # cubic drift
    return -x**3

M = 1.0  # sigma^2/2 with sigma = sqrt(2)

def build_Q_u(x, dx):
    n = len(x)
    mup = np.maximum(mu(x), 0.0)
    mum = np.maximum(-mu(x), 0.0)
    up = mup/dx + M/dx**2
    um = mum/dx + M/dx**2
    main = -(up + um)
    Q = sp.diags([um[1:], main, up[:-1]], offsets=[-1,0,1], format="csc")
    return Q

def build_Q_c(x, dx):
    n = len(x)
    m = mu(x)
    up = (M/dx**2) * np.exp( (m*dx)/(2*M) )
    um = (M/dx**2) * np.exp( -(m*dx)/(2*M) )
    main = -(up + um)
    Q = sp.diags([um[1:], main, up[:-1]], offsets=[-1,0,1], format="csc")
    return Q

def stationary_density(Q, dx):
    # Solve Q^T v = 0 with normalization sum v dx = 1
    QT = Q.T
    n = Q.shape[0]
    # Replace one equation by normalization
    A = QT.tolil()
    b = np.zeros(n)
    A[0,:] = dx  # normalization row
    b[0] = 1.0
    v = spla.spsolve(A.tocsc(), b)
    v = np.maximum(v, 0.0)
    v /= (v.sum()*dx)
    return v

def mfpt_exit(Q, x, a=0.0, bnd=2.0):
    # Solve (Q tau)_i = -1 on (a,bnd), tau=0 on outside
    inside = np.where((x > a) & (x < bnd))[0]
    outside = np.where((x <= a) | (x >= bnd))[0]

    n = len(x)
    tau = np.zeros(n)

    # Build reduced system for inside nodes with Dirichlet outside
    Qii = Q[inside[:,None], inside]
    rhs = -np.ones(len(inside))

    # Add boundary contributions: sum_{j outside} Q_{i,j} * tau_j = 0 because tau_j=0
    tau_inside = spla.spsolve(Qii, rhs)
    tau[inside] = tau_inside
    tau[outside] = 0.0
    return tau

def committor(Q, x, a=0.0, bnd=2.0):
    # Solve (Q q)=0 on (a,bnd), q(a)=0, q(bnd)=1
    n = len(x)
    q = np.zeros(n)

    iL = np.argmin(np.abs(x-a))
    iR = np.argmin(np.abs(x-bnd))
    q[iL] = 0.0
    q[iR] = 1.0

    inside = np.array([i for i in range(n) if i not in (iL,iR) and (x[i] > a) and (x[i] < bnd)])
    fixed = np.array([iL, iR])

    Qii = Q[inside[:,None], inside]
    Qif = Q[inside[:,None], fixed]

    rhs = -Qif @ q[fixed]
    q_inside = spla.spsolve(Qii, rhs)
    q[inside] = q_inside
    return q
