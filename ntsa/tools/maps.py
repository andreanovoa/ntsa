"""Return maps, Poincare sections, and peak-based transient/period detection."""

import numpy as np
from scipy.signal import find_peaks

from ntsa.tools.embedding import delay_embed


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
