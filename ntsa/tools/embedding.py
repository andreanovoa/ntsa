"""Delay embeddings and lag/dimension selection (Kantz & Schreiber ch. 3 and 9)."""

import numpy as np
from scipy.spatial import cKDTree

from ntsa.tools.statistics import autocorrelation


def delay_embed(x, dim, lag):
    """Delay-coordinate embedding of a scalar series: rows (x[i], x[i+lag], ...)."""
    x = np.asarray(x)
    N = len(x) - (dim - 1) * lag
    if N <= 0:
        raise ValueError(f'series too short for dim={dim}, lag={lag}')
    return np.column_stack([x[i * lag:i * lag + N] for i in range(dim)])


def average_mutual_information(x, max_lag, n_bins=64):
    """AMI (nats) between x(t) and x(t+lag) for lags 1..max_lag; shape (max_lag,)."""
    x = np.asarray(x, dtype=float)
    x = (x - x.min()) / max(np.ptp(x), 1e-30)
    ami = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        h, _, _ = np.histogram2d(x[:-lag], x[lag:], bins=n_bins, range=[[0, 1], [0, 1]])
        p = h / h.sum()
        px, py = p.sum(axis=1), p.sum(axis=0)
        nz = p > 0
        ami[lag - 1] = np.sum(p[nz] * np.log(p[nz] / np.outer(px, py)[nz]))
    return ami


def optimal_lag(x, max_lag=None, n_bins=64):
    """Embedding lag (samples): first local minimum of the AMI curve (Fraser & Swinney 1986).

    When the AMI is nearly lag-independent (min > 0.5*max — e.g. noiseless periodic
    signals, where local minima are pure histogram-quantization jitter), falls back
    to the first zero crossing of the autocorrelation (Kantz & Schreiber sec. 3.3.1).
    """
    if max_lag is None:
        max_lag = min(len(x) // 10, 500)
    a = average_mutual_information(x, max_lag, n_bins=n_bins)
    if a.min() > 0.5 * a.max():
        below = np.flatnonzero(autocorrelation(x, max_lag) <= 0)
        if below.size:
            return max(int(below[0]), 1)
    interior = np.flatnonzero((a[1:-1] < a[:-2]) & (a[1:-1] < a[2:]))
    lag = interior[0] + 2 if interior.size else int(np.argmin(a)) + 1
    return max(int(lag), 1)


def false_nearest_neighbours(x, lag, d_max=10, Rtol=10.0, Atol=2.0, threshold=0.01, n_query=5000, seed=0):
    """Embedding dimension via false nearest neighbours (Kennel et al. 1992).

    Returns
    -------
    d : int
        First dimension with FNN fraction < `threshold` (else argmin of fractions).
    fractions : ndarray (d_max,)
        FNN fraction per dimension 1..d_max; nan where not computed (early exit).
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    sigma = x.std()
    fractions = np.full(d_max, np.nan)
    for d in range(1, d_max + 1):
        n_rows = len(x) - d * lag  # rows for which the (d+1)-th coordinate x[i + d*lag] exists
        if n_rows < 2:
            break
        Y = delay_embed(x, d, lag)[:n_rows]
        tree = cKDTree(Y)
        q = rng.choice(n_rows, size=min(n_query, n_rows), replace=False)
        dist, j = tree.query(Y[q], k=2)
        dist, j = dist[:, 1], j[:, 1]  # k=2 to skip self-match
        extra = np.abs(x[q + d * lag] - x[j + d * lag])
        false = (extra / np.maximum(dist, 1e-12) > Rtol) | (np.sqrt(dist ** 2 + extra ** 2) / sigma > Atol)
        fractions[d - 1] = false.mean()
        if fractions[d - 1] < threshold:
            return d, fractions
    return int(np.nanargmin(fractions)) + 1, fractions
