"""Dynamical-regime classification of scalar time series."""

from fractions import Fraction

import numpy as np
from scipy.signal import find_peaks

from ntsa.tools.maps import count_peak_clusters, first_return_map
from ntsa.tools.statistics import fun_PSD


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
