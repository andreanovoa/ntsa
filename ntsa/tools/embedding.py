"""Delay embeddings and lag/dimension selection (Kantz & Schreiber ch. 3 and 9; Cao 1997)."""

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


def _nearest_valid(tree, Y, q, theiler, r_min, k0=8):
    """Index of the nearest neighbour (max norm) of each query row Y[q] that lies outside
    the Theiler window, |j - i| > theiler, and at nonzero distance (Cao 1997, sec. 2);
    distances up to r_min count as zero, so that round-off duplicates are skipped too."""
    n = len(Y)
    nn = np.full(len(q), -1)
    todo = np.arange(len(q))
    k = min(n, 2 * theiler + 1 + k0)
    while todo.size:
        dist, j = tree.query(Y[q[todo]], k=k, p=np.inf)
        ok = (np.abs(j - q[todo, None]) > theiler) & (dist > r_min) & (j < n)
        has = ok.any(axis=1)
        nn[todo[has]] = j[has, np.argmax(ok[has], axis=1)]
        todo = todo[~has]
        if k == n:
            break
        k = min(n, 2 * k)
    return nn


def cao_embedding_dimension(x, lag, d_max=10, theiler=0, tol=0.05, n_query=5000, seed=0):
    r"""Minimum embedding dimension by Cao's method (Cao 1997).

    For each delay vector $\mathbf{y}_i(d)$ with nearest neighbour $\mathbf{y}_{n(i,d)}(d)$
    in the maximum norm, the method compares the neighbour distance in $d$ and $d+1$
    dimensions,

    $$
    a(i, d) = \frac{\lVert \mathbf{y}_i(d+1) - \mathbf{y}_{n(i,d)}(d+1) \rVert_\infty}
                   {\lVert \mathbf{y}_i(d) - \mathbf{y}_{n(i,d)}(d) \rVert_\infty},
    \qquad
    E(d) = \langle a(i, d) \rangle_i,
    \qquad
    E_1(d) = \frac{E(d+1)}{E(d)}.
    $$

    $E_1(d)$ stops changing once $d$ unfolds the attractor; the minimum embedding
    dimension is the first $d$ with $\lvert E_1(d+1) - E_1(d) \rvert <$ `tol`. Unlike
    false nearest neighbours, no distance threshold is chosen. The second quantity,

    $$
    E^*(d) = \langle \lvert x_{i+d\zeta} - x_{n(i,d)+d\zeta} \rvert \rangle_i,
    \qquad
    E_2(d) = \frac{E^*(d+1)}{E^*(d)},
    $$

    separates deterministic from stochastic data: for random data the future value is
    independent of the past, so $E_2(d) \approx 1$ for every $d$, whereas for deterministic
    data $E_2(d)$ departs from 1 for some $d$. For noise, $E_1(d)$ also saturates, so the
    returned dimension is meaningful only when $E_2$ is not flat at 1. The converse does not
    hold: on high-dimensional chaos with a delay long compared with the predictability time,
    $E_2$ also approaches 1, and a shorter delay restores the contrast.

    Neighbours inside the Theiler window $\lvert i - j \rvert \le$ `theiler` are excluded,
    and, as in Cao (1997), a neighbour at zero distance is replaced by the next nearest
    neighbour at nonzero distance.

    Parameters
    ----------
    x : array_like (N,)
        Scalar time series.
    lag : int
        Embedding delay $\zeta$ in samples (e.g. `optimal_lag`).
    d_max : int
        Largest dimension at which $E_1$ and $E_2$ are evaluated.
    theiler : int
        Theiler window in samples; 0 excludes only the point itself.
    tol : float
        Saturation tolerance on successive changes of $E_1$.
    n_query : int
        Number of reference points sampled at random (all points if fewer).
    seed : int
        Seed of the reference-point sampling.

    Returns
    -------
    d : int
        Estimated minimum embedding dimension (`d_max` if $E_1$ never saturates).
    E1 : ndarray (d_max,)
        $E_1(d)$ for $d = 1, \dots, d_\mathrm{max}$.
    E2 : ndarray (d_max,)
        $E_2(d)$ for $d = 1, \dots, d_\mathrm{max}$.

    References
    ----------
    Cao, L. (1997). Practical method for determining the minimum embedding dimension of a
    scalar time series. *Physica D* 110, 43-50.
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    r_min = 1e-10 * x.std()  # round-off scale: smaller distances count as zero
    E, Es = np.empty(d_max + 1), np.empty(d_max + 1)
    for d in range(1, d_max + 2):
        n_rows = len(x) - d * lag  # rows i for which y_i(d+1) exists
        if n_rows < 2 * theiler + 3:
            raise ValueError(f'series too short for d_max={d_max}, lag={lag}, theiler={theiler}')
        Y = delay_embed(x, d, lag)[:n_rows]
        q = np.sort(rng.choice(n_rows, size=min(n_query, n_rows), replace=False))
        j = _nearest_valid(cKDTree(Y), Y, q, theiler, r_min)
        keep = j >= 0
        q, j = q[keep], j[keep]
        r_d = np.max(np.abs(Y[q] - Y[j]), axis=1)
        extra = np.abs(x[q + d * lag] - x[j + d * lag])
        E[d - 1] = np.mean(np.maximum(r_d, extra) / r_d)
        Es[d - 1] = np.mean(extra)
    E1, E2 = E[1:] / E[:-1], Es[1:] / Es[:-1]
    flat = np.flatnonzero(np.abs(np.diff(E1)) < tol)
    d = int(flat[0]) + 1 if flat.size else d_max
    return d, E1, E2


def takens_bound(D=None, x=None, lag=None, d_embed=None):
    r"""Upper bound on the minimum embedding dimension from the embedding theorems.

    Takens' theorem guarantees that a generic delay map is an embedding for every
    $m \ge 2D + 1$, so the minimum embedding dimension satisfies $m_\mathrm{min} \le 2D + 1$.
    Because the reconstruction must also contain the attractor, $m_\mathrm{min} > D$, and

    $$
    D < m_\mathrm{min} \le \lfloor 2D + 1 \rfloor .
    $$

    The bound reads in two ways, depending on $D$:

    - **Takens (1981).** $D$ is the (integer) dimension of the manifold that contains the
      dynamics. For Lorenz-63, the state dimension $D = 3$ gives the bound 7.
    - **Sauer, Yorke & Casdagli (1991).** $D$ is the box-counting dimension of the
      attractor, generally fractal, and every $m > 2 D_\mathrm{box}$ embeds. For Lorenz-63,
      $D \approx 2.06$ gives the bound 5.

    The condition is sufficient, not necessary: the minimum embedding dimension that
    Cao's $E_1$ (`cao_embedding_dimension`) and false nearest neighbours
    (`false_nearest_neighbours`) estimate from the data (3 for Lorenz-63) typically lies
    below the bound.

    Parameters
    ----------
    D : float, optional
        Dimension of the system (manifold or attractor). If omitted, $D$ is estimated
        from the series `x` as the Grassberger-Procaccia correlation dimension $D_2$
        (`correlation_dimension`) of its delay embedding. Because $D_2 \le D_\mathrm{box}$,
        the estimate reads in the sense of Sauer et al. and can be low, which lowers the bound.
    x : array_like (N,), optional
        Scalar time series, required when `D` is omitted.
    lag : int, optional
        Embedding delay in samples for the estimate of $D$ (default `optimal_lag(x)`).
    d_embed : int, optional
        Dimension of the delay embedding on which $D_2$ is measured (default
        `cao_embedding_dimension(x, lag)`, which unfolds the attractor).

    Returns
    -------
    m_max : int
        Upper bound $\lfloor 2D + 1 \rfloor$ on the minimum embedding dimension.
    D : float
        The dimension used (given or estimated), which is the lower bound.

    References
    ----------
    Takens, F. (1981). Detecting strange attractors in turbulence. *Lecture Notes in
    Math.* 898, 366-381.
    Sauer, T., Yorke, J. A. & Casdagli, M. (1991). Embedology. *J. Stat. Phys.* 65, 579-616.
    """
    if D is None:
        if x is None:
            raise ValueError('pass either the dimension D or a series x to estimate it from')
        from ntsa.tools.geometry import correlation_dimension
        if lag is None:
            lag = optimal_lag(x)
        if d_embed is None:
            d_embed, _, _ = cao_embedding_dimension(x, lag)
        D, _ = correlation_dimension(delay_embed(x, d_embed, lag))
    return int(np.floor(2 * D + 1)), float(D)
