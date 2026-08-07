"""Nonlinear time-series analysis tools for dynamical-system models.

Delay embeddings, mutual-information lag selection, false-nearest-neighbour
dimension estimation, return maps, recurrence matrices, classical MDS, regime
classification, and bifurcation sweeps. Reference: Kantz & Schreiber,
*Nonlinear Time Series Analysis* (2nd ed., CUP 2004).

Run as a script for a bifurcation-diagram demo, e.g.::

    python -m ntsa.tools --model lorenz63 --param rho --pmin 20 --pmax 100 --save-figs
"""

from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kurtosis, skew
from tqdm import tqdm


def fun_PSD(dt, X):
    """Compute the Power Spectral Density of one or more signals.

    Parameters
    ----------
    dt : float
        Sampling time.
    X : np.ndarray
        Signal(s), shape ``(Nq, Nt)`` (1D signals are promoted to ``(1, Nt)``; a 2D
        array is transposed if its first dimension is larger than its second, i.e.
        the longer axis is assumed to be time).

    Returns
    -------
    f : np.ndarray
        Frequencies, shape ``(Nt // 2,)``.
    PSD : list of np.ndarray
        Power Spectral Density of each row of `X`.
    """
    if X.ndim == 2:
        if X.shape[0] > X.shape[1]:
            X = X.T
    elif X.ndim == 1:
        X = np.expand_dims(X, axis=0)
    else:
        raise AssertionError('X must be 2 dimensional')

    len_x = X.shape[-1]
    f = np.linspace(0.0, 1.0 / (2.0 * dt), len_x // 2)
    PSD = []
    for x in X:
        yt = np.fft.fft(x)
        PSD.append(2.0 / len_x * np.abs(yt[0:len_x // 2]))

    return f, PSD


EXTREMA_COLORS = dict(max='k', min='crimson')


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def respawn(model, psi0=None, dt=None, **overrides):
    """Fresh instance of ``type(model)`` with `alpha0` and fixed params carried over.

    Parameters in `overrides` win over the carried-over values. `psi0` defaults
    to ``model.psi0[:, 0]``; `dt` defaults to ``model.dt``.
    """
    if psi0 is None:
        psi0 = model.psi0[:, 0]
    params = {**model.alpha0,
              **{key: getattr(model, key) for key in model.fixed_params}}
    # observation config is neither an alpha nor a fixed param — carry it over too
    for key in ('observe_dims', 'observed_idx'):
        if hasattr(model, key):
            params[key] = list(getattr(model, key))
    params.update(overrides)
    return type(model)(psi0=np.array(psi0, dtype=float), dt=dt or model.dt, **params)


def run_long(model, t_run, t_transient=None, trim_transient=True):
    """Integrate `model` past the transient, then record `t_run` time units.

    Mutates `model` (history is reset to t=0 after the transient) and closes it.
    With ``trim_transient`` (default), residual transient drift that outlives
    ``t_transient`` is detected on the first observable (``stationary_start``)
    and the leading drifting samples are dropped, with a printed note — raise
    ``t_transient`` to keep the full horizon.

    Returns
    -------
    t : ndarray (Nt,)
    y : ndarray (Nt, Nq)
        Observable history of ensemble member 0.
    psi : ndarray (Nt, Nphi)
        State history of ensemble member 0.
    """
    if t_transient is None:
        t_transient = model.t_transient
    Nt_tr = int(round(t_transient / model.dt))
    if Nt_tr > 0:
        psi, _ = model.time_integrate(Nt=Nt_tr)
        model.update_history(psi[-1:], t=np.array([0.]), reset=True)
    psi, tt = model.time_integrate(int(round(t_run / model.dt)))
    model.update_history(psi, tt)  # keep the integrator's stamps (dt..Nt*dt after the reset to t=0)
    model.close()
    t, y, psi = model.hist_t, model.get_observable_hist()[:, :, 0], model.hist[:, :model.Nphi, 0]
    if trim_transient:
        i0 = stationary_start(y[:, 0])
        if i0:
            print(f'[run_long] trimmed {t[i0] - t[0]:.3g} t.u. of residual transient drift '
                  f'({100 * i0 / len(t):.0f}% of the record) — raise t_transient to keep the full horizon')
            t, y, psi = t[i0:], y[i0:], psi[i0:]
    return t, y, psi


# ---------------------------------------------------------------------------
# Embedding diagnostics (Kantz & Schreiber ch. 3 and 9)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Return maps, recurrence, statistics
# ---------------------------------------------------------------------------

def first_return_map(x, prominence=None):
    """Successive local maxima of `x`: returns (xm_i, xm_{i+1}, peak indices)."""
    x = np.asarray(x, dtype=float)
    if prominence is None:
        prominence = 1e-3 * np.ptp(x) or None
    idx, _ = find_peaks(x, prominence=prominence)
    xm = x[idx]
    return xm[:-1], xm[1:], idx


def stationary_start(x, tol=0.05, block=10):
    """First sample index where the oscillation amplitude has stopped drifting.

    Guards return maps and bifurcation diagrams against residual transients
    (e.g. slow Hopf growth that outlives ``t_transient``): the median of each
    ``block`` successive local maxima must fall inside the [5%, 95%] quantile
    range of the last-half maxima, padded by ``tol*ptp(x)``. Quantiles (not a
    median band) so that broad but stationary maxima distributions — chaotic
    lobe-switching, quasiperiodic envelopes — are never trimmed.

    Returns 0 when the signal is already stationary or has too few maxima
    (< 4*block) to judge; never returns more than half the record.
    """
    x = np.asarray(x, dtype=float)
    idx, _ = find_peaks(x, prominence=1e-3 * np.ptp(x) or None)
    if len(idx) < 4 * block:
        return 0
    xm = x[idx]
    lo, hi = np.quantile(xm[len(xm) // 2:], [0.05, 0.95])
    slack = tol * np.ptp(x)
    for b0 in range(0, len(xm) // 2, block):
        if lo - slack <= np.median(xm[b0:b0 + block]) <= hi + slack:
            return int(idx[b0]) if b0 else 0
    return int(idx[len(xm) // 2])


def poincare_section(x, zeta, level=None, direction=1):
    """Plane-crossing Poincare section of the delay embedding (x, x+zeta, x+2*zeta).

    Section plane x(t+2*zeta) = `level` (median by default), crossed in `direction`
    (+1: upward); returns the (x(t), x(t+zeta)) coordinates, linearly interpolated
    at each crossing, shape (n_crossings, 2). Complements the maxima return map:
    a period-k limit cycle gives k points, a 2-torus a closed loop, a 3-torus a
    filled band, chaos a fractal scatter (D2 of the section ~ attractor D2 - 1).
    """
    Y = delay_embed(np.asarray(x, dtype=float), 3, zeta)
    s = Y[:, 2] - (np.median(Y[:, 2]) if level is None else level)
    if direction < 0:
        s = -s
    i = np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0))
    w = (s[i] / (s[i] - s[i + 1]))[:, None]
    return Y[i, :2] + w * (Y[i + 1, :2] - Y[i, :2])


def count_peak_clusters(xm, x_range, tol=0.02):
    """Number of distinct peak values, i.e. the period of a limit cycle.

    A continuum band of maxima (any cluster with internal spread >= tol*x_range,
    as in chaotic or quasiperiodic signals) returns ``len(xm)`` as a sentinel so
    callers do not mistake it for a small-period cycle.

    # ponytail: 1-D gap heuristic (gaps > tol*x_range split clusters); swap for
    # kmeans on (x_i, x_{i+1}) pairs if it misfires on noisy maxima.
    """
    xm = np.sort(np.asarray(xm, dtype=float))
    if xm.size == 0:
        return 0
    if np.ptp(xm) < tol * x_range:
        return 1
    edges = np.flatnonzero(np.diff(xm) > tol * x_range)
    for cluster in np.split(xm, edges + 1):
        if np.ptp(cluster) >= tol * x_range:  # continuum band, not a tight periodic peak
            return len(xm)
    return len(edges) + 1


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


def autocorrelation(x, n_lags):
    """Normalized autocorrelation for lags 0..n_lags; acf[0] = 1."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    c = np.correlate(x, x, mode='full')[len(x) - 1:len(x) + n_lags]
    return c / (c[0] if c[0] > 0 else 1.0)


def signal_stats(x, dt, n_bins=64):
    """Summary statistics of a scalar series.

    Returns
    -------
    dict with keys: mean, std, skew, kurtosis, f, psd, pdf_centers, pdf, acf, acf_lags.
    """
    x = np.asarray(x, dtype=float)
    f, psd = fun_PSD(dt, x)
    counts, edges = np.histogram(x, bins=n_bins, density=True)
    n_lags = len(x) // 4
    return dict(mean=float(x.mean()), std=float(x.std()),
                skew=float(skew(x)), kurtosis=float(kurtosis(x)),
                f=f, psd=psd[0],
                pdf_centers=0.5 * (edges[:-1] + edges[1:]), pdf=counts,
                acf=autocorrelation(x, n_lags), acf_lags=np.arange(n_lags + 1) * dt)


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

    # ponytail: no Theiler window — the even-stride nmax subsample decorrelates
    # pairs on long records; add one if you feed short, densely sampled series.
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


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_regime(x, dt, lam1=None, lam1_std=0.0, t_total=None, k_max=8, cluster_tol=0.02,
                    lam_tol=None, exponents=None, neutral_tol=2e-3):
    """Classify the dynamical regime of a scalar series (Kantz & Schreiber ch. 1, 3, 5).

    Decision tree: flat tail -> fixed point; few *tight* return-map clusters ->
    period-k limit cycle (a long record whose maxima collapse onto k levels cannot
    be chaotic, and this signal evidence outranks a perturbation-growth lam1 that
    can be inflated by non-normal transient amplification — e.g. Rijke limit
    cycles read lam1 ~ 200 otherwise); positive leading Lyapunov exponent (beyond
    `lam_tol`) -> chaotic; else PSD peak analysis separates limit cycle /
    frequency-locked / quasiperiodic (rational vs irrational peak ratio).

    With `exponents` (a full Lyapunov spectrum), two decisive refinements apply:
    all exponents < -neutral_tol -> fixed point (robust to integrator-noise
    wiggle at a stable focus), and >= 2 neutral exponents (|lam| < neutral_tol)
    -> quasiperiodic (a 2-torus has two zero exponents; a locked periodic orbit
    exactly one — more robust than the PSD rational-ratio test).

    Returns
    -------
    label : str
        One of 'fixed_point', 'chaotic', 'limit_cycle_period_<k>', 'limit_cycle',
        'frequency_locked', 'quasiperiodic'.
    evidence : dict
        Diagnostics: lambda1, lambda1_std, lam_tol, n_clusters, psd_peak_freqs,
        f1, f2, rational_match, tail_flat, n_neutral, label (None where not
        computed).
    """
    x = np.asarray(x, dtype=float)
    evidence = dict(lambda1=lam1, lambda1_std=lam1_std, lam_tol=None, n_clusters=None,
                    psd_peak_freqs=None, f1=None, f2=None, rational_match=None,
                    tail_flat=None, n_neutral=None, label=None)

    def _return(label):
        evidence['label'] = label
        return label, evidence

    # (a) fixed point: flat trailing fifth of the series, or an all-negative
    # Lyapunov spectrum (decisive when integrator noise keeps the tail wiggling
    # at a stable focus — e.g. Lorenz63 at rho=10, lam = [-0.6, -0.6, -12.5])
    tail = x[-(len(x) // 5):]
    evidence['tail_flat'] = bool(np.ptp(tail) < 1e-4 * max(np.ptp(x), abs(x.mean()), 1e-30))
    if evidence['tail_flat']:
        return _return('fixed_point')
    if exponents is not None and np.all(np.asarray(exponents) < -neutral_tol):
        return _return('fixed_point')

    # PSD peak analysis — always recorded, so panels can display f1/f2 for every regime
    f, psd = fun_PSD(dt, x)
    psd = psd[0]
    # skip the DC bin: fun_PSD puts 2*|mean| at f=0, which would swamp the prominence scale
    pk, _ = find_peaks(psd[1:], prominence=0.05 * psd[1:].max())
    pk = pk + 1
    evidence['psd_peak_freqs'] = f[pk]
    f1 = f2 = None
    if pk.size:
        f1 = float(f[pk[np.argmax(psd[pk])]])
        evidence['f1'] = f1
        ratios = f[pk] / f1
        harmonic = (np.round(ratios) >= 1) & (np.abs(ratios - np.round(ratios)) < 0.02 * np.maximum(np.round(ratios), 1))
        survivors = pk[~harmonic]
        if survivors.size:
            f2 = float(f[survivors[np.argmax(psd[survivors])]])
            evidence['f2'] = f2
            frac = Fraction(f2 / f1).limit_denominator(10)
            if abs(f2 / f1 - float(frac)) < 0.005:
                evidence['rational_match'] = f'{frac.numerator}/{frac.denominator}'

    # (b) period-k limit cycle: k distinct *tight* clusters of local maxima.
    # Checked before the Lyapunov step: k tight levels over a long record are
    # incompatible with chaos (count_peak_clusters returns a len(xm) sentinel for
    # any continuum band), whereas a perturbation-growth lam1 can read large and
    # positive on a stable orbit through non-normal transient amplification.
    _, _, pk_idx = first_return_map(x)
    k = count_peak_clusters(x[pk_idx], x_range=np.ptp(x), tol=cluster_tol)
    evidence['n_clusters'] = k
    if 0 < k <= k_max:
        return _return(f'limit_cycle_period_{k}')

    # (c) chaotic: leading Lyapunov exponent significantly positive
    if t_total is None:
        t_total = len(x) * dt
    if lam_tol is None:
        lam_tol = max(3 * lam1_std, 10.0 / t_total)
    evidence['lam_tol'] = lam_tol
    if lam1 is not None and np.isfinite(lam1) and lam1 > lam_tol:
        return _return('chaotic')

    # (d) quasiperiodic vs frequency-locked. A full Lyapunov spectrum settles it
    # outright: a 2-torus has TWO neutral exponents, a (locked) periodic orbit has
    # exactly one — more robust than the PSD ratio test, whose limit_denominator(10)
    # rationals are dense enough to "match" incommensurate ratios (e.g. L96 F=4.4:
    # f2/f1 = 0.552 ~ 5/9 but lam = [0.001, 0.000, -0.003] is a torus).
    if exponents is not None:
        n0 = int(np.sum(np.abs(np.asarray(exponents)) < neutral_tol))
        evidence['n_neutral'] = n0
        if n0 >= 2:
            return _return('quasiperiodic')
    if f2 is None:
        return _return('limit_cycle')
    if evidence['rational_match']:
        return _return('frequency_locked')
    return _return('quasiperiodic')


# ---------------------------------------------------------------------------
# Bifurcation diagrams
# ---------------------------------------------------------------------------

def bifurcation_sweep(model, param, values, dt=None, t_transient=None, t_sample=None,
                      extrema=('max',), continuation=False, seed=0, **overrides):
    """Sweep `param` over `values`, collecting local extrema of every observable.

    By default the whole sweep is ONE ensemble forecast: every parameter value
    shares the same initial state, `param` is augmented into the state
    (``init_ensemble(est_alpha=[param])``) and member `k` integrates at
    ``values[k]`` — the integrator parallelizes across parameter values in a
    single run. With ``continuation=True`` it falls back to a serial m=1 loop
    where each run starts from the previous endpoint plus a small perturbation —
    the classic way to follow an attractor branch (e.g. through a hysteresis),
    and inherently sequential.

    Returns
    -------
    values : ndarray
    peaks : dict
        ``peaks[ext][iq][k]`` = extrema (ragged 1-D arrays) of observable `iq`
        at ``values[k]``, for each `ext` in `extrema` ('max' and/or 'min').
    """
    if param not in model.params:
        raise ValueError(f"'{param}' is not in {type(model).__name__}.params = {model.params}")
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    psi0 = model.psi0[:, 0] + 1e-3 * rng.standard_normal(model.Nphi)
    peaks = {ext: [[] for _ in range(model.Nq)] for ext in extrema}

    def collect(y_k, trim=False):
        i0 = stationary_start(y_k[:, 0]) if trim else 0
        for ext in extrema:
            sign = 1.0 if ext == 'max' else -1.0
            for iq in range(y_k.shape[1]):
                idx, _ = find_peaks(sign * y_k[i0:, iq])
                peaks[ext][iq].append(y_k[i0 + idx, iq])

    if continuation:
        for val in tqdm(values, desc=f'{param} sweep'):
            mi = respawn(model, psi0=psi0, dt=dt, **{param: val}, **overrides)
            _, y, psi = run_long(mi, t_run=t_sample or 20 * mi.t_CR, t_transient=t_transient)
            collect(y)   # run_long already trimmed the residual transient
            psi0 = psi[-1] + 1e-6 * rng.standard_normal(model.Nphi)
        return values, peaks

    # identical ICs: one ensemble run, member k carrying values[k] as its parameter
    mi = respawn(model, psi0=psi0, dt=dt, **overrides)
    aug = np.vstack([np.tile(psi0[:, None], (1, values.size)), values[None, :]])
    mi.init_ensemble(m=values.size, est_alpha=[param], ensemble_psi0=aug)
    if t_transient is None:
        t_transient = mi.t_transient
    Nt_tr = int(round(t_transient / mi.dt))
    if Nt_tr > 0:
        psi, _ = mi.time_integrate(Nt=Nt_tr)
        mi.update_history(psi[-1:], t=np.array([0.]), reset=True)
    psi, tt = mi.time_integrate(int(round((t_sample or 20 * mi.t_CR) / mi.dt)))
    mi.update_history(psi, tt)
    mi.close()
    y = mi.get_observable_hist()                     # (Nt, Nq, m)
    for k in range(values.size):
        collect(y[:, :, k], trim=True)               # per-member residual-transient trim
    return values, peaks


def plot_bifurcation(values, peaks, param_label, obs_labels, filename=None):
    """Bifurcation diagram: one axis per observable, extrema vs parameter."""
    Nq = len(obs_labels)
    fig, axs = plt.subplots(Nq, 1, figsize=(8, 2.5 * Nq), sharex=True,
                            layout='constrained', squeeze=False)
    axs = axs.ravel()
    for iq, (ax, label) in enumerate(zip(axs, obs_labels)):
        for ext, per_obs in peaks.items():
            color = EXTREMA_COLORS.get(ext, 'k')
            for p, pk in zip(values, per_obs[iq]):
                ax.plot(np.full_like(pk, p), pk, '.', color=color, ms=1.5, alpha=0.4)
        ax.set_ylabel(f'local extrema of {label}')
    axs[-1].set_xlabel(param_label)
    if len(peaks) > 1:
        handles = [Line2D([], [], ls='', marker='.', color=EXTREMA_COLORS.get(e, 'k'), label=f'{e}ima')
                   for e in peaks]
        axs[0].legend(handles=handles, loc='best')
    if filename:
        fig.savefig(filename, dpi=200)
    return fig, axs


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import os

    import matplotlib
    matplotlib.use('Agg')

    from dynamodels.physical import Lorenz63, Lorenz96, VdP

    DEMOS = {  # model class, default swept param, (pmin, pmax)
        'lorenz63': (Lorenz63, 'rho', (20., 100.)),
        'lorenz96': (Lorenz96, 'F', (2., 18.)),
        'vdp': (VdP, 'beta', (40., 120.)),
    }

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=sorted(DEMOS), default='lorenz63')
    parser.add_argument('--param', default=None)
    parser.add_argument('--pmin', type=float, default=None)
    parser.add_argument('--pmax', type=float, default=None)
    parser.add_argument('--nP', type=int, default=40)
    parser.add_argument('--t-transient', type=float, default=None)
    parser.add_argument('--t-sample', type=float, default=None)
    parser.add_argument('--save-figs', action='store_true')
    args = parser.parse_args()

    cls, param0, (pmin0, pmax0) = DEMOS[args.model]
    model = cls()
    param = args.param or param0
    values = np.linspace(args.pmin if args.pmin is not None else pmin0,
                         args.pmax if args.pmax is not None else pmax0, args.nP)

    values, peaks = bifurcation_sweep(model, param, values, t_transient=args.t_transient,
                                      t_sample=args.t_sample, extrema=('max', 'min'))

    fig_dir = 'figs'  # relative to the caller's cwd, no longer next to the module
    os.makedirs(fig_dir, exist_ok=True)
    filename = os.path.join(fig_dir, f'bifurcation_{args.model}_{param}.pdf') if args.save_figs else None
    plot_bifurcation(values, peaks, model.alpha_labels.get(param, param), model.obs_labels, filename=filename)
    n_pk = sum(len(pk) for pk in peaks['max'][0])
    print(f'{args.model}: swept {param} over [{values[0]:g}, {values[-1]:g}] ({args.nP} points), '
          f'{n_pk} maxima of {model.obs_labels[0]}' + (f'; saved {filename}' if filename else ''))
