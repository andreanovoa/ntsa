"""Spatio-temporal fields: a first look at a gridded record u(t, x) before any model is fitted.

`FieldSeries` wraps a record sampled at a fixed time step on a uniform grid in one, two or three space
dimensions, optionally with several components (e.g. two velocity components). `analyze` computes

- a description of the record: shape, finite values, range, record length in samples and in integral times,
  and whether the domain-averaged energy has stopped drifting (`ntsa.tools.stationary_start`);
- point statistics over time: mean, rms, skewness and flatness fields, and the pdf of the fluctuations;
- the time-averaged energy spectrum of the fluctuations (shell-averaged in 2D and 3D);
- the two-point correlation along each space axis and its integral length;
- the autocorrelation and Welch spectrum of a few probes, the integral time and the dominant frequency;
- the POD (snapshot method) energy and the number of modes for 90 % and 99 % of the fluctuation energy.

`figures` draws one figure per quantity, each with a caption that says what it shows and what to look for;
`report` writes them, an animation (a space-time diagram in 1D), the numbers in ``stats.json``, and one
self-contained HTML page with the captioned figures embedded. Fluctuations are taken about the time mean at each point.
Periodic directions are transformed as they are; non-periodic ones are Hann-windowed before the transform.
"""

import base64
import html
import json
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from ntsa.tools import stationary_start

#: figure style of the user's papers (washout-free ESN-DA): serif Termes/STIX, no top and right spines
STYLE = {'font.family': 'serif', 'font.serif': ['TeX Gyre Termes', 'STIXGeneral'], 'mathtext.fontset': 'stix',
         'pdf.fonttype': 42, 'font.size': 10, 'legend.fontsize': 8, 'axes.labelsize': 10,
         'xtick.labelsize': 9, 'ytick.labelsize': 9, 'lines.linewidth': 1.2, 'savefig.dpi': 200, 'axes.spines.top': False,
         'axes.spines.right': False, 'legend.frameon': False}
#: colour-blind-safe colours (Okabe-Ito, with Paul Tol's red for marks); the reference data are grey and dashed
C = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'vermilion': '#D55E00', 'pink': '#CC79A7',
     'sky': '#56B4E9', 'mark': '#CC3311', 'grey': '#BBBBBB'}
INK = '#333333'
C_REF = '0.45'
WIDTH = 7.2                                         # inches: full text width of the papers
SERIES = [C[k] for k in ('blue', 'orange', 'green', 'vermilion', 'pink', 'sky')]
REF_STYLE = dict(color=C_REF, ls='--', lw=1.2, zorder=1)
#: quantities a reference curve can be laid over, and the panel that holds each
REFERENCE_PANELS = ('energy', 'pdf', 'E', 'corr', 'acf', 'psd', 'pod_energy', 'mean', 'rms')


_GREEK = {'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'kappa', 'lambda', 'mu', 'nu',
          'xi', 'pi', 'rho', 'sigma', 'tau', 'phi', 'chi', 'psi', 'omega', 'Gamma', 'Delta', 'Theta', 'Phi', 'Psi',
          'Omega'}


def _letter(ax, s):
    ax.text(0.0, 1.03, rf'$\mathbf{{({s})}}$', transform=ax.transAxes, ha='left', va='bottom')


def _integral(r, step):
    """Integral of a correlation r(lag) up to its first zero crossing (the whole curve if it never crosses)."""
    r = np.asarray(r)
    below = np.flatnonzero(r <= 0)
    n = below[0] if len(below) else len(r)
    return float(r[:n].sum() * step)


def _signed_norm(a):
    lo, hi = np.nanpercentile(a, [0.5, 99.5])
    if lo < 0 < hi:
        m = max(-lo, hi)
        return TwoSlopeNorm(vcenter=0.0, vmin=-m, vmax=m), 'RdBu_r'
    return plt.Normalize(lo, hi), 'viridis'


@dataclass
class FieldSeries:
    """A gridded spatio-temporal record and its first-look characterisation.

    Parameters
    ----------
    u : np.ndarray
        Record of shape ``(Nt, *space)``, or ``(Nt, *space, Nc)`` when `components` is given; one, two or three
        space dimensions, ordered as stored (for 2D, ``(Ny, Nx)`` plots with x horizontal).
    dt : float
        Time step between snapshots.
    dx : float or sequence of float
        Grid spacing per space axis (a scalar applies to every axis).
    label : str
        Case label used in the report.
    components : sequence of str, optional
        Names of the components on the last axis (e.g. ``('u', 'v')``); a scalar field when omitted.
    periodic : bool or sequence of bool
        Periodicity per space axis.
    max_snapshots : int
        Snapshots used by the POD (evenly subsampled when the record is longer).
    references : list of dict, optional
        Reference data laid over the figures, grey and dashed, e.g. published DNS statistics. Each entry has
        ``quantity`` (one of `REFERENCE_PANELS`), ``x`` and ``y`` in the units of the record, and ``label``
        (the source, cited as author and year). `add_reference` appends one.
    """

    u: np.ndarray
    dt: float
    dx: object = 1.0
    label: str = 'field'
    components: tuple = None
    periodic: object = True
    max_snapshots: int = 2000
    references: list = field(default_factory=list)
    results: dict = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.u = np.asarray(self.u)
        if self.components is None:
            self.u = self.u[..., None]
            self.components = ('u',)
        self.components = tuple(self.components)
        self._sym = tuple(rf'\{c}' if c in _GREEK else c for c in self.components)   # 'omega' -> $\omega$
        if self.u.shape[-1] != len(self.components):
            raise ValueError(f'last axis has {self.u.shape[-1]} entries, components={self.components}')
        nd = self.u.ndim - 2
        if nd not in (1, 2, 3):
            raise ValueError(f'expected 1 to 3 space dimensions, got shape {self.u.shape}')
        self.dx = tuple(np.broadcast_to(np.asarray(self.dx, dtype=float), (nd,)))
        self.periodic = tuple(np.broadcast_to(np.asarray(self.periodic, dtype=bool), (nd,)))

    def add_reference(self, quantity, x, y, label):
        """Lay a reference curve y(x) over the panel of `quantity` (see `REFERENCE_PANELS`)."""
        if quantity not in REFERENCE_PANELS:
            raise ValueError(f'quantity must be one of {REFERENCE_PANELS}, got {quantity!r}')
        self.references.append(dict(quantity=quantity, x=np.asarray(x, float), y=np.asarray(y, float), label=label))

    def _overlay(self, ax, quantity, swap=False):
        for ref in self.references:
            if ref['quantity'] == quantity:
                x, y = (ref['y'], ref['x']) if swap else (ref['x'], ref['y'])
                ax.plot(x, y, label=ref['label'], **REF_STYLE)

    # -- geometry -----------------------------------------------------------------------------------------------
    @property
    def nt(self):
        return self.u.shape[0]

    @property
    def space(self):
        return self.u.shape[1:-1]

    @property
    def t(self):
        return np.arange(self.nt) * self.dt

    @property
    def axes_names(self):
        return {1: ('x',), 2: ('y', 'x'), 3: ('z', 'y', 'x')}[len(self.space)]

    def _window(self):
        """Separable taper over the space axes: ones where periodic, Hann otherwise."""
        w = np.ones(self.space)
        for i, (n, p) in enumerate(zip(self.space, self.periodic)):
            if not p:
                shape = [1] * len(self.space)
                shape[i] = n
                w = w * np.hanning(n).reshape(shape)
        return w

    def _fluct(self, c, idx=slice(None)):
        return self.u[idx, ..., c].astype(np.float64) - self.results['mean'][..., c]

    # -- analysis -----------------------------------------------------------------------------------------------
    def analyze(self, n_probes=3):
        """Compute every statistic of the module docstring and cache them in `results`."""
        u = self.u
        finite = np.isfinite(u)
        r = {'mean': np.nanmean(u, axis=0, dtype=np.float64)}
        self.results = r
        r['rms'] = np.nanstd(u, axis=0, dtype=np.float64)
        flat = u.reshape(self.nt, -1, len(self.components))
        r['skew'] = skew(flat, axis=0, nan_policy='omit').reshape(r['mean'].shape)
        r['flatness'] = (kurtosis(flat, axis=0, nan_policy='omit') + 3.0).reshape(r['mean'].shape)
        sp_axes = tuple(range(1, u.ndim - 1))
        r['energy_t'] = 0.5 * np.nansum((u - r['mean']) ** 2, axis=(*sp_axes, u.ndim - 1)) / np.prod(self.space)
        r['i_stationary'] = stationary_start(r['energy_t'])

        # pdf of the fluctuations, normalised by the global rms of each component
        r['pdf'] = []
        for c in range(len(self.components)):
            q = self._fluct(c).ravel()
            q = q[np.isfinite(q)] / (np.sqrt(np.mean(q[np.isfinite(q)] ** 2)) or 1.0)
            h, e = np.histogram(q, bins=81, range=(-6, 6), density=True)
            r['pdf'].append((0.5 * (e[1:] + e[:-1]), h))

        self._spectra_and_correlations()
        self._probes(n_probes)
        self._pod()
        r['describe'] = self._describe(finite)
        return r

    def _spectra_and_correlations(self):
        """Time-averaged energy spectrum E(k) and two-point correlation R(lag) per space axis."""
        r, nd = self.results, len(self.space)
        sp_axes = tuple(range(1, nd + 1))
        w = self._window()
        stride = max(1, self.nt // self.max_snapshots)
        P = 0.0
        R = [0.0] * nd
        for c in range(len(self.components)):
            for i0 in range(0, self.nt, 256 * stride):
                q = np.nan_to_num(self._fluct(c, slice(i0, i0 + 256 * stride, stride)))
                P = P + (np.abs(np.fft.fftn(q * w, axes=sp_axes)) ** 2).sum(0)
                for a in range(nd):
                    n = self.space[a]
                    F = np.fft.fft(q, n=n if self.periodic[a] else 2 * n, axis=a + 1)
                    R[a] = R[a] + np.moveaxis(np.fft.ifft(np.abs(F) ** 2, axis=a + 1).real, a + 1, 1)[:, :n // 2].sum(0)
        # shell average on the wavenumber magnitude in units of the fundamental of the first axis
        k_axes = [2 * np.pi * np.fft.fftfreq(n, d) for n, d in zip(self.space, self.dx)]
        K = np.sqrt(sum(k ** 2 for k in np.meshgrid(*k_axes, indexing='ij')))
        dk = min(2 * np.pi / (n * d) for n, d in zip(self.space, self.dx))
        shell = np.rint(K / dk).astype(int)
        nk = min(n // 2 for n in self.space)
        E = np.bincount(shell.ravel(), weights=P.ravel(), minlength=nk + 1)[:nk + 1]
        r['k'] = np.arange(1, nk + 1) * dk
        r['E'] = E[1:] / E[1:].sum() * 0.5 * np.mean(r['rms'] ** 2) * len(self.components) / dk
        r['corr'] = []
        for a in range(nd):
            n = self.space[a]
            Ra = R[a].reshape(n // 2, -1).sum(1)
            if not self.periodic[a]:
                Ra = Ra / (n - np.arange(n // 2))
            Ra = Ra / Ra[0]
            lag = np.arange(n // 2) * self.dx[a]
            r['corr'].append((lag, Ra, _integral(Ra, self.dx[a])))

    def _probes(self, n_probes):
        """Time statistics averaged over up to 256 grid points per component: autocorrelation, Welch spectrum,
        integral time; `n_probes` of them are kept for the probe list of the report."""
        r = self.results
        npts = int(np.prod(self.space))
        pts = np.unique(np.linspace(0, npts - 1, min(npts, 256)).astype(int))
        r['probes'] = [tuple(int(i) for i in np.unravel_index(p, self.space)) for p in pts[:: max(1, len(pts) // n_probes)]]
        nlag = min(self.nt // 2, 2000)
        nseg = min(self.nt, 1024)
        r['acf'], r['psd'] = [], []
        for c in range(len(self.components)):
            X = np.nan_to_num(self.u[..., c].reshape(self.nt, -1)[:, pts].astype(np.float64))
            X = X - X.mean(0)
            F = np.fft.rfft(X, n=2 * self.nt, axis=0)
            acf = np.fft.irfft(np.abs(F) ** 2, axis=0)[:nlag + 1].mean(1) / (self.nt - np.arange(nlag + 1))
            r['acf'].append(acf / (acf[0] or 1.0))
            f, s = welch(X, fs=1 / self.dt, nperseg=nseg, axis=0)
            r['psd'].append((f, s.mean(1)))
        r['acf_lag'] = np.arange(nlag + 1) * self.dt
        r['n_points'] = len(pts)
        r['T_int'] = float(np.mean([_integral(a, self.dt) for a in r['acf']]))
        r['acf_lag1'] = float(np.mean([a[1] for a in r['acf']])) if nlag else float('nan')
        s_mean = np.mean([s for _, s in r['psd']], axis=0)
        i = int(np.argmax(s_mean[1:])) + 1          # a peak must stand out of the low-frequency plateau
        r['f_peak'] = float(r['psd'][0][0][i]) if s_mean[i] > 2 * s_mean[1:4].mean() else None

    def _pod(self):
        """Snapshot POD of the fluctuations: energy fractions and the four leading modes."""
        r = self.results
        idx = np.linspace(0, self.nt - 1, min(self.nt, self.max_snapshots)).astype(int)
        Q = np.nan_to_num(np.stack([self._fluct(c, idx) for c in range(len(self.components))], axis=-1))
        Q = Q.reshape(len(idx), -1)
        C = Q @ Q.T
        lam, V = np.linalg.eigh(C)
        lam, V = np.clip(lam[::-1], 0, None), V[:, ::-1]
        r['pod_energy'] = lam / lam.sum()
        cum = np.cumsum(r['pod_energy'])
        r['pod_n90'] = int(np.searchsorted(cum, 0.90) + 1)
        r['pod_n99'] = int(np.searchsorted(cum, 0.99) + 1)
        nm = min(4, len(lam))
        modes = (Q.T @ V[:, :nm]) / np.sqrt(np.maximum(lam[:nm], 1e-300))
        r['pod_modes'] = modes.T.reshape(nm, *self.space, len(self.components))

    def _checks(self, n_bad):
        """Warnings computed from the data: what to fix or keep in mind before fitting a model."""
        r, out = self.results, []
        if n_bad:
            out.append(f'{n_bad} non-finite values: mask or interpolate them first.')
        if r['acf_lag1'] < 0.5:
            out.append(f'Consecutive snapshots are nearly uncorrelated (lag-one autocorrelation {r["acf_lag1"]:.2f}): '
                       'the record is not time-resolved, or it is shuffled. Time statistics (integral time, '
                       'spectra in frequency) are not meaningful until the order and the time step are checked.')
        elif self.nt * self.dt < 100 * r['T_int']:
            out.append(f'The record spans {self.nt * self.dt / r["T_int"]:.0f} integral times (fewer than 100): '
                       'expect noisy statistics.')
        if r['i_stationary']:
            out.append(f'The domain energy drifts until t = {r["i_stationary"] * self.dt:.4g}: discard it as transient.')
        for (lag, R, L), name, n, d in zip(r['corr'], self.axes_names, self.space, self.dx):
            if L > 0.1 * n * d:
                out.append(f'The integral length along {name} is {L / (n * d):.0%} of the domain: the box may be too '
                           'small for the largest structures.')
        tail = r['E'][-max(1, len(r['E']) // 10):].mean() / r['E'].max()
        if tail > 1e-4:
            out.append(f'The energy spectrum holds {tail:.1e} of its peak at the highest wavenumbers: the grid may '
                       'under-resolve the field.')
        return out or ['None: the record passes every check.']

    def _describe(self, finite):
        r = self.results
        T = r['T_int']
        L = {f'integral length along {n}': r['corr'][a][2] for a, n in enumerate(self.axes_names)}
        r['checks'] = self._checks(int((~finite).sum()))
        return {
            'label': self.label,
            'shape (Nt, space, components)': [self.nt, list(self.space), list(self.components)],
            'dt': float(self.dt), 'dx': [float(d) for d in self.dx], 'periodic': [bool(p) for p in self.periodic],
            'dtype': str(self.u.dtype),
            'non-finite values': int((~finite).sum()),
            'min': float(np.nanmin(self.u)), 'max': float(np.nanmax(self.u)),
            'mean (domain average)': [float(np.mean(r['mean'][..., c])) for c in range(len(self.components))],
            'rms (domain average)': [float(np.sqrt(np.mean(r['rms'][..., c] ** 2))) for c in range(len(self.components))],
            'record length': self.nt * self.dt,
            'integral time (probe mean)': T,
            'lag-one autocorrelation (probe mean)': r['acf_lag1'],
            'record length / integral time': self.nt * self.dt / T if T > 0 else None,
            'dominant frequency': r['f_peak'],
            'stationary from t': float(r['i_stationary'] * self.dt),
            **L,
            'checks': r['checks'],
            'POD modes for 90 % / 99 % energy': [r['pod_n90'], r['pod_n99']],
            'POD snapshots used': int(min(self.nt, self.max_snapshots)),
            'references': sorted({f"{ref['label']} ({ref['quantity']})" for ref in self.references}),
        }

    # -- figures ------------------------------------------------------------------------------------------------
    def _extent(self):
        ny, nx = self.space[-2:] if len(self.space) > 1 else (1, self.space[0])
        return [0, nx * self.dx[-1], 0, ny * self.dx[-2] if len(self.space) > 1 else 1]

    def _slice(self, a):
        """2D view of a spatial array (mid-plane of the first axis in 3D)."""
        return a[a.shape[0] // 2] if a.ndim == 3 else a

    def _map(self, fig, ax, a, label, cmap=None):
        """Image of a 2D field with its colour bar; signed fields on RdBu_r centred on zero."""
        norm, cm = (plt.Normalize(np.nanmin(a), np.nanmax(a)), cmap) if cmap else _signed_norm(a)
        m = ax.imshow(a, origin='lower', extent=self._extent(), norm=norm, cmap=cm)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(True)
        fig.colorbar(m, ax=ax, label=label, shrink=0.85, aspect=20)
        return m

    # -- figures: one per quantity ------------------------------------------------------------------------------
    def figures(self):
        """Ordered list of (name, figure, caption): one figure per quantity, each explained by its caption."""
        if self.results is None:
            self.analyze()
        r, nd = self.results, len(self.space)
        out = []
        with plt.rc_context(STYLE):
            out.append(('snapshots', self.plot_snapshots(), self._cap_snapshots()))
            out.append(('mean_rms', self.plot_mean_rms(),
                        ('Time mean (top) and rms of the fluctuations (bottom) along the domain.' if nd == 1 else
                         'Time mean (left) and rms of the fluctuations (right).')
                        + ' Structure in the mean is the forced or statistically steady pattern; a uniform rms '
                        'means statistically homogeneous fluctuations.'
                        + (' Three-dimensional record: mid-plane of the first axis.' if nd == 3 else '')))
            out.append(('energy', self.plot_energy(),
                        r'Fluctuation energy averaged over the domain, \(\tfrac12\langle q^{\prime 2}\rangle\), against '
                        'time. A drift or a jump at the start is a transient to discard'
                        + (f'; the dotted line marks where the energy stops drifting (t = '
                           f'{r["i_stationary"] * self.dt:.3g})' if r['i_stationary'] else '')
                        + ('. The snapshots are not time-resolved, so they are drawn as points.'
                           if r['acf_lag1'] < 0.5 else '.')))
            out.append(('pdf', self.plot_pdf(),
                        'Probability density of the fluctuations, each normalised by its rms, against the Gaussian '
                        '(grey). Tails heavier than the Gaussian point to intermittency; an asymmetric density shows '
                        'up as a non-zero skewness.'))
            out.append(('spectrum', self.plot_spectrum(),
                        'Time-averaged energy spectrum of the fluctuations' + (
                            ', shell-averaged over the wavenumber magnitude' if nd > 1 else '') +
                        ', normalised so that its integral is the fluctuation energy; the dealiased tail is cut. '
                        'The peak gives the energy-containing scale; the decay at high wavenumbers shows how well '
                        'the grid resolves the field, and a pile-up at the end would mean under-resolution.'))
            out.append(('correlation', self.plot_correlation(),
                        'Two-point correlation of the fluctuations along each space axis; the legend gives the '
                        'integral length (integral up to the first zero crossing). A correlation still far from zero '
                        'at half the domain means the box is too small for the largest structures.'))
            if r['acf_lag1'] >= 0.5:
                out.append(('autocorrelation', self.plot_autocorrelation(),
                            f'Autocorrelation in time, averaged over {r["n_points"]} grid points. Its integral up '
                            'to the first zero is the integral time, which sets how many independent samples the '
                            'record holds.'))
                out.append(('frequency', self.plot_frequency(),
                            f'Welch frequency spectrum, averaged over {r["n_points"]} grid points' + (
                                f'; the dotted line marks the dominant frequency, f = {r["f_peak"]:.3g}'
                                if r['f_peak'] else '; there is no dominant frequency (broadband spectrum)') + '.'))
            out.append(('pod_energy', self.plot_pod_energy(),
                        f'Energy fraction of each POD mode (snapshot method, {min(self.nt, self.max_snapshots)} '
                        f'snapshots); {r["pod_n90"]} modes hold 90 % of the fluctuation energy and {r["pod_n99"]} '
                        'hold 99 % (dotted lines). These numbers set the scale of any reduced-order model.'))
            out.append(('pod_modes', self.plot_pod_modes(),
                        f'Leading POD modes of {self._words(0)}, with their energy fraction. Modes in pairs with '
                        'equal energy and shifted shapes are travelling structures.'))
        return out

    def _words(self, c):
        return f'\\({self._sym[c]}\\)'                    # inline maths for the HTML captions (MathJax)

    def _cap_snapshots(self):
        nd = len(self.space)
        if nd == 1:
            return (f'Space-time diagram of the first {min(self.nt, 1500)} snapshots. Tilted streaks are travelling '
                    'structures; their slope is the propagation speed.')
        return ('Three snapshots spanning the record, on one colour scale per component'
                + (', mid-plane of the first axis' if nd == 3 else '') + '. The animation shows the evolution.')

    def plot_snapshots(self):
        nc, nd = len(self.components), len(self.space)
        if nd == 1:
            fig, axes = plt.subplots(nc, 1, figsize=(WIDTH, 2.4 * nc), squeeze=False, layout='constrained')
            x = np.arange(self.space[0]) * self.dx[0]
            n = min(self.nt, 1500)
            for c, ax in enumerate(axes[:, 0]):
                norm, cmap = _signed_norm(self.u[:n, :, c])
                m = ax.pcolormesh(self.t[:n], x, self.u[:n, :, c].T, norm=norm, cmap=cmap, shading='auto',
                                  rasterized=True)
                fig.colorbar(m, ax=ax, label=f'${self._sym[c]}$', pad=0.01, aspect=25)
                ax.set(xlabel='$t$', ylabel='$x$')
            return fig
        times = np.linspace(0, self.nt - 1, 3).astype(int)
        fig, axes = plt.subplots(nc, 3, figsize=(WIDTH, 2.3 * nc), squeeze=False, sharex=True, sharey=True,
                                 layout='constrained')
        for c in range(nc):
            snaps = [self._slice(self.u[i, ..., c]) for i in times]
            norm, cmap = _signed_norm(np.stack(snaps))
            for ax, snap, i in zip(axes[c], snaps, times):
                m = ax.imshow(snap, origin='lower', extent=self._extent(), norm=norm, cmap=cmap)
                for side in ('top', 'right'):
                    ax.spines[side].set_visible(True)
                if c == 0:
                    ax.text(0.5, 1.03, f'$t$ = {i * self.dt:.3g}', transform=ax.transAxes, ha='center',
                            va='bottom', fontsize=9)
            fig.colorbar(m, ax=axes[c], label=f'${self._sym[c]}$', shrink=0.9, aspect=15)
            axes[c, 0].set_ylabel('$y$')
        for ax in axes[-1]:
            ax.set_xlabel('$x$')
        return fig

    def plot_mean_rms(self):
        r, nc = self.results, len(self.components)
        if len(self.space) == 1:
            x = np.arange(self.space[0]) * self.dx[0]
            fig, (a, b) = plt.subplots(2, 1, figsize=(WIDTH, 3.2), sharex=True, layout='constrained')
            for c, col in zip(range(nc), SERIES):
                a.plot(x, r['mean'][:, c], color=col, label=f'${self._sym[c]}$')
                b.plot(x, r['rms'][:, c], color=col)
            self._overlay(a, 'mean')
            self._overlay(b, 'rms')
            a.set(ylabel=r'$\overline{q}$')
            b.set(xlabel='$x$', ylabel=r'$q^\prime_\mathrm{rms}$', xlim=(x[0], x[-1]))
            if nc > 1 or a.get_legend_handles_labels()[0][1:]:
                a.legend(ncol=nc)
            return fig
        fig, axes = plt.subplots(nc, 2, figsize=(WIDTH * 0.75, 2.5 * nc), squeeze=False, sharex=True,
                                 sharey=True, layout='constrained')
        for c in range(nc):
            s = self._sym[c]
            self._map(fig, axes[c, 0], self._slice(r['mean'][..., c]), rf'$\overline{{{s}}}$')
            self._map(fig, axes[c, 1], self._slice(r['rms'][..., c]), rf'${s}^\prime_\mathrm{{rms}}$', 'Blues')
            axes[c, 0].set_ylabel('$y$')
        for ax in axes[-1]:
            ax.set_xlabel('$x$')
        return fig

    def _single(self):
        return plt.subplots(figsize=(WIDTH / 2, 2.5), layout='constrained')

    def plot_energy(self):
        r = self.results
        fig, ax = plt.subplots(figsize=(WIDTH, 2.2), layout='constrained')
        if r['acf_lag1'] >= 0.5:
            ax.plot(self.t, r['energy_t'], color=INK, lw=0.7)
        else:
            ax.plot(self.t, r['energy_t'], '.', color=INK, ms=1.5, mec='none')
        if r['i_stationary']:
            ax.axvline(r['i_stationary'] * self.dt, color=C['mark'], ls=':', lw=1)
        self._overlay(ax, 'energy')
        ax.set(xlabel='$t$', ylabel=r'$\frac{1}{2}\langle q^{\prime 2}\rangle$', xlim=(0, self.t[-1]))
        return fig

    def plot_pdf(self):
        r = self.results
        fig, ax = self._single()
        g = np.linspace(-5, 5, 400)
        ax.semilogy(g, np.exp(-g ** 2 / 2) / np.sqrt(2 * np.pi), color=C['grey'], lw=1, label='Gaussian')
        for (xc, h), s, col in zip(r['pdf'], self._sym, SERIES):
            ax.semilogy(xc, np.where(h > 0, h, np.nan), color=col, label=f'${s}^\\prime$')
        self._overlay(ax, 'pdf')
        ax.set(xlabel=r'$q^\prime / q^\prime_\mathrm{rms}$', ylabel='pdf', ylim=(1e-5, 1), xlim=(-5, 5))
        ax.legend(loc='lower center', handlelength=1.2)
        return fig

    def plot_spectrum(self):
        r = self.results
        fig, ax = self._single()
        last = np.flatnonzero(r['E'] > 1e-10 * r['E'].max())[-1] + 1
        ax.loglog(r['k'][:last], r['E'][:last], color=INK)
        self._overlay(ax, 'E')
        ax.set(xlabel='$k$', ylabel='$E(k)$')
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        return fig

    def plot_correlation(self):
        r = self.results
        fig, ax = self._single()
        for (lag, R, L), name, col in zip(r['corr'], self.axes_names, SERIES):
            ax.plot(lag, R, color=col, label=f'$\\Delta {name}$, $L_{name}$ = {L:.2g}')
        ax.axhline(0, color=C['grey'], lw=0.6, zorder=0)
        self._overlay(ax, 'corr')
        ax.set(xlabel=r'$\Delta$', ylabel=r'$R(\Delta)$')
        ax.legend(handlelength=1.2)
        return fig

    def plot_autocorrelation(self):
        r = self.results
        fig, ax = self._single()
        for acf, s, col in zip(r['acf'], self._sym, SERIES):
            ax.plot(r['acf_lag'], acf, color=col, label=f'${s}$')
        ax.axhline(0, color=C['grey'], lw=0.6, zorder=0)
        self._overlay(ax, 'acf')
        ax.set(xlabel=r'$\tau$', ylabel=r'$\rho(\tau)$', xlim=(0, min(r['acf_lag'][-1], 10 * r['T_int'])))
        if len(ax.get_legend_handles_labels()[0]) > 1:
            ax.legend(handlelength=1.2)
        return fig

    def plot_frequency(self):
        r = self.results
        fig, ax = self._single()
        for (fq, s), name, col in zip(r['psd'], self._sym, SERIES):
            ax.loglog(fq[1:], s[1:], color=col, label=f'${name}$')
        if r['f_peak']:
            ax.axvline(r['f_peak'], color=C['grey'], ls=':', lw=0.8)
        self._overlay(ax, 'psd')
        ax.set(xlabel='$f$', ylabel=r'$\Phi(f)$')
        if len(ax.get_legend_handles_labels()[0]) > 1:
            ax.legend(handlelength=1.2)
        return fig

    def plot_pod_energy(self):
        r = self.results
        fig, ax = self._single()
        n = np.arange(1, len(r['pod_energy']) + 1)
        ok = r['pod_energy'] > 1e-12
        ax.loglog(n[ok], r['pod_energy'][ok], '.', color=INK, ms=2.5)
        for nq in (r['pod_n90'], r['pod_n99']):
            ax.axvline(nq, color=C['mark'], ls=':', lw=0.9)
        self._overlay(ax, 'pod_energy')
        ax.set(xlabel='$n$', ylabel=r'$\lambda_n / \sum_j \lambda_j$')
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        return fig

    def plot_pod_modes(self):
        r = self.results
        nm = len(r['pod_modes'])
        if len(self.space) == 1:
            x = np.arange(self.space[0]) * self.dx[0]
            fig, axes = plt.subplots(nm, 1, figsize=(WIDTH, 0.9 + 0.75 * nm), sharex=True, sharey=True,
                                     layout='constrained')
            for j, ax in enumerate(np.atleast_1d(axes)):
                ax.plot(x, r['pod_modes'][j][..., 0], color=INK, lw=1)
                ax.axhline(0, color=C['grey'], lw=0.6, zorder=0)
                ax.set(ylabel=rf'$\phi_{j + 1}$', xlim=(x[0], x[-1]))
                ax.text(1.0, 1.0, f'{r["pod_energy"][j]:.1%}', transform=ax.transAxes, ha='right', va='top',
                        fontsize=9)
            ax.set_xlabel('$x$')
            return fig
        fig, axes = plt.subplots(1, nm, figsize=(WIDTH, 2.1), sharey=True, layout='constrained')
        for j, ax in enumerate(np.atleast_1d(axes)):
            mode = self._slice(r['pod_modes'][j][..., 0])
            norm, cmap = _signed_norm(mode)
            ax.imshow(mode, origin='lower', extent=self._extent(), norm=norm, cmap=cmap)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(True)
            ax.set(xlabel='$x$', ylabel='$y$' if j == 0 else None)
            ax.text(0.5, 1.03, rf'$\phi_{j + 1}$, {r["pod_energy"][j]:.1%}', transform=ax.transAxes, ha='center',
                    va='bottom', fontsize=9)
        return fig

    def animate(self, path, n_frames=120, fps=12, dpi=100):
        """Gif of the first component (mid-plane in 3D); not written for 1D fields (see `plot_snapshots`)."""
        from matplotlib.animation import FuncAnimation, PillowWriter

        if len(self.space) == 1:
            return None
        with plt.rc_context(STYLE):
            idx = np.linspace(0, self.nt - 1, min(self.nt, n_frames)).astype(int)
            frames = [self._slice(self.u[i, ..., 0]) for i in idx]
            norm, cmap = _signed_norm(np.stack(frames))
            ny, nx = frames[0].shape
            fig, ax = plt.subplots(figsize=(3.2 * min(2.5, max(0.6, nx / ny)) + 0.8, 3.0), layout='constrained')
            m = ax.imshow(frames[0], origin='lower', extent=self._extent(), norm=norm, cmap=cmap)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(True)
            fig.colorbar(m, ax=ax, label=f'${self._sym[0]}$', shrink=0.9)
            ax.set(xlabel='$x$', ylabel='$y$')
            clock = ax.text(1.0, 1.02, '', transform=ax.transAxes, ha='right', va='bottom', fontsize=9)

            def draw(n):
                m.set_data(frames[n])
                clock.set_text(f'$t$ = {idx[n] * self.dt:.3g}')
                return m, clock

            FuncAnimation(fig, draw, frames=len(frames), blit=True).save(path, writer=PillowWriter(fps=fps), dpi=dpi)
            plt.close(fig)
        return path

    # -- report -------------------------------------------------------------------------------------------------
    def report(self, out_dir, gif=True, notes=None):
        """Write one PNG per quantity, an animation, ``stats.json`` and a self-contained ``index.html`` to `out_dir`.

        `notes` is an optional list of HTML snippets (e.g. dataset provenance, how the record was loaded) placed
        under the summary table. Returns the path of the HTML page.
        """
        if self.results is None:
            self.analyze()
        os.makedirs(out_dir, exist_ok=True)
        items = []
        if gif and len(self.space) > 1:
            path = self.animate(os.path.join(out_dir, 'animation.gif'))
            items.append((path, f'Time evolution of {self._words(0)} over the whole record'
                          + (', mid-plane of the first axis' if len(self.space) == 3 else '') + '.'))
        for name, fig, caption in self.figures():
            path = os.path.join(out_dir, f'{name}.png')
            fig.savefig(path)
            plt.close(fig)
            items.append((path, caption))
        desc = self.results['describe']
        with open(os.path.join(out_dir, 'stats.json'), 'w') as fh:
            json.dump(desc, fh, indent=1, default=float)
        page = os.path.join(out_dir, 'index.html')
        with open(page, 'w') as fh:
            fh.write(_html(desc, items, notes or []))
        return page


def _img(path):
    mime = 'image/gif' if path.endswith('.gif') else 'image/png'
    with open(path, 'rb') as fh:
        return f'data:{mime};base64,{base64.b64encode(fh.read()).decode()}'


def _fmt(v):
    if isinstance(v, float):
        return f'{v:.4g}'
    if isinstance(v, list):
        return ', '.join(_fmt(x) for x in v)
    return html.escape(str(v))


def _html(desc, items, notes):
    rows = '\n'.join(f'<tr><th>{html.escape(k)}</th><td>{_fmt(v)}</td></tr>' for k, v in desc.items()
                     if k not in ('checks', 'references'))
    checks = '\n'.join(f'<li>{html.escape(c)}</li>' for c in desc.get('checks', []))
    extra = '\n'.join(f'<p>{n}</p>' for n in notes)
    if desc.get('references'):
        extra += '<p><strong>Reference data</strong> (grey, dashed): ' + '; '.join(
            html.escape(x) for x in desc['references']) + '.</p>'
    figs = '\n'.join(f'<figure><img src="{_img(p)}" alt="Figure {i}"><figcaption><strong>Figure {i}.</strong> '
                     f'{cap}</figcaption></figure>' for i, (p, cap) in enumerate(items, 1))
    title = html.escape(str(desc['label']))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} first look</title>
<script>window.MathJax = {{ tex: {{ inlineMath: [['\\\\(', '\\\\)']] }} }};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"></script>
<style>
:root {{ --bg: #ffffff; --fg: #222222; --muted: #666666; --rule: #dddddd; --accent: #0072B2; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #16181c; --fg: #e6e6e6;
  --muted: #a0a0a0; --rule: #333333; --accent: #56B4E9; }} }}
:root[data-theme="dark"] {{ --bg: #16181c; --fg: #e6e6e6; --muted: #a0a0a0; --rule: #333333; --accent: #56B4E9; }}
body {{ background: var(--bg); color: var(--fg); font: 16px/1.55 Georgia, 'Times New Roman', serif; margin: 0 auto;
  max-width: 860px; padding: 24px 16px; }}
h1 {{ font-size: 1.6rem; margin: 0 0 4px; }} h2 {{ font-size: 1.15rem; margin-top: 36px; color: var(--accent); }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 0.92rem; }}
th, td {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--rule); vertical-align: top; }}
th {{ font-weight: normal; color: var(--muted); width: 45%; }}
figure {{ margin: 28px 0; }} img {{ display: block; max-width: 100%; margin: 0 auto; background: #ffffff;
  border-radius: 3px; }}
figcaption {{ font-size: 0.93rem; margin-top: 8px; }}
.wrap {{ overflow-x: auto; }}
</style></head><body>
<h1>{title}</h1>
<p style="color: var(--muted)">First look at the record, computed by <code>ntsa.fields.FieldSeries</code>.</p>
<h2>Checks</h2>
<ul>{checks}</ul>
<h2>Summary</h2>
<div class="wrap"><table>{rows}</table></div>
{extra}
<h2>Figures</h2>
{figs}
</body></html>
"""
