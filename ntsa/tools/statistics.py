"""Signal statistics: power spectral density, autocorrelation, summary moments."""

import numpy as np
from scipy.stats import kurtosis, skew


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
