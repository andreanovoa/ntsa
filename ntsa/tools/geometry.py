"""Attractor geometry from pairwise distances: recurrence, MDS, correlation dimension."""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def recurrence_matrix(Y, eps_frac=0.10, rr=None):
    """Boolean (T, T) recurrence matrix: pairwise distance <= threshold.

    The threshold is ``eps_frac`` times the maximum pairwise distance or, if ``rr``
    is given, the ``rr``-quantile of the pairwise distances — i.e. a fixed
    recurrence rate, the robust choice when the density degenerates (Marwan 2007).
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    d_cond = pdist(Y)
    if rr is not None:
        return squareform(d_cond) <= np.quantile(d_cond, rr)
    D = squareform(d_cond)
    return D <= eps_frac * D.max()


def classical_mds(X, n_coords=3, nmax=2000):
    """Classical multidimensional scaling of a trajectory (Nt, Nphi).

    Double-centres the squared Euclidean distance matrix and maps snapshots via
    gamma = V sqrt(Lambda) from the top eigenpairs of A = -1/2 C D^2 C.

    Returns
    -------
    gamma : ndarray (T, n_coords)
    idx : ndarray
        Indices of the (possibly subsampled) snapshots used.
    """
    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    idx = np.round(np.linspace(0, T - 1, nmax)).astype(int) if T > nmax else np.arange(T)
    T = len(idx)
    D2 = squareform(pdist(X[idx])) ** 2
    C = np.eye(T) - np.ones((T, T)) / T
    A = -0.5 * C @ D2 @ C
    eigvals, eigvecs = np.linalg.eigh(A)  # ascending
    lam = eigvals[-n_coords:][::-1]
    V = eigvecs[:, -n_coords:][:, ::-1]
    gamma = V * np.sqrt(np.maximum(lam, 0.0))
    return gamma, idx


def correlation_dimension(Y, nmax=2000):
    """Grassberger-Procaccia correlation dimension D2 (Kantz & Schreiber ch. 6).

    C(r) = fraction of point pairs closer than r; D2 = d log C / d log r fitted
    over the 2nd-50th percentile of the pairwise distances. On a k-torus D2 ~ k
    (limit cycle 1, quasiperiodic 2-torus 2, 3-torus 3); chaos gives a fractal
    value — together with the number of neutral Lyapunov exponents (`n_neutral`
    in `classify_regime` evidence) this pins down the torus dimension along a
    Ruelle-Takens-Newhouse route. Accepts a delay embedding, the full state
    trajectory, or a cloud of Poincare-section points (section of a k-torus has
    D2 ~ k-1).
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if len(Y) > nmax:
        Y = Y[::max(1, len(Y) // nmax)][:nmax]
    d = np.sort(pdist(Y))
    d = d[d > 0]
    r = np.logspace(np.log10(d[int(0.02 * len(d))]), np.log10(d[int(0.5 * len(d))]), 20)
    C = np.searchsorted(d, r) / len(d)
    D2 = float(np.polyfit(np.log(r), np.log(C), 1)[0])
    return D2, (np.log(r), np.log(C))
