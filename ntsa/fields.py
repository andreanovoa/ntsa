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

`report` writes the figures, an animation (a space-time diagram in 1D), the numbers in ``stats.json``, and one
self-contained HTML page with the figures embedded. Fluctuations are taken about the time mean at each point.
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
         'pdf.fonttype': 42, 'font.size': 10, 'legend.fontsize': 8, 'axes.spines.top': False,
         'axes.spines.right': False, 'legend.frameon': False}
#: colour-blind-safe colours (Okabe-Ito, with Paul Tol's red for marks); the reference data are grey and dashed
C = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'vermilion': '#D55E00', 'pink': '#CC79A7',
     'sky': '#56B4E9', 'mark': '#CC3311', 'grey': '#BBBBBB'}
INK = '#333333'
C_REF = '0.45'
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
        """Probes evenly spaced along the domain diagonal: autocorrelation, Welch spectrum, integral time."""
        r = self.results
        frac = (np.arange(n_probes) + 0.5) / n_probes
        r['probes'] = [tuple(int(f * n) for n in self.space) for f in frac]
        nlag = min(self.nt // 2, 2000)
        r['acf'], r['psd'] = [], []
        T = []
        for p in r['probes']:
            x = np.nan_to_num(self.u[(slice(None), *p, 0)].astype(np.float64))
            x = x - x.mean()
            c = np.correlate(x, x, mode='full')[len(x) - 1:len(x) + nlag] / (len(x) - np.arange(min(nlag + 1, len(x))))
            c = c / (c[0] or 1.0)
            r['acf'].append(c)
            T.append(_integral(c, self.dt))
            f, s = welch(x, fs=1 / self.dt, nperseg=min(len(x), 1024))
            r['psd'].append((f, s))
        r['acf_lag'] = np.arange(nlag + 1) * self.dt
        r['T_int'] = float(np.mean(T))
        r['acf_lag1'] = float(np.mean([c[1] for c in r['acf']])) if nlag else float('nan')
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

    def plot_fields(self):
        with plt.rc_context(STYLE):
            return self._plot_fields()

    def _plot_fields(self):
        """Snapshots, mean and rms (2D/3D), or space-time diagram with mean and rms profiles (1D)."""
        r = self.results
        nc = len(self.components)
        if len(self.space) == 1:
            fig, axes = plt.subplots(nc, 2, figsize=(10, 3.2 * nc), width_ratios=(3, 1), squeeze=False,
                                     layout='constrained')
            x = np.arange(self.space[0]) * self.dx[0]
            n = min(self.nt, 1500)
            for c, (a, b) in enumerate(axes):
                norm, cmap = _signed_norm(self.u[:n, :, c])
                m = a.pcolormesh(self.t[:n], x, self.u[:n, :, c].T, norm=norm, cmap=cmap, shading='auto',
                                 rasterized=True)
                fig.colorbar(m, ax=a, label=f'${self._sym[c]}$', pad=0.01)
                a.set(xlabel='$t$', ylabel='$x$')
                b.plot(r['mean'][:, c], x, color=C['blue'], label='mean')
                b.plot(r['rms'][:, c], x, color=C['orange'], label='rms')
                self._overlay(b, 'mean', swap=True)
                self._overlay(b, 'rms', swap=True)
                b.set(xlabel=f'${self._sym[c]}$', ylabel='$x$', ylim=(x[0], x[-1]))
                b.legend()
            for ax, s in zip(axes.ravel(), 'abcdefgh'):
                _letter(ax, s)
            return fig
        times = np.linspace(0, self.nt - 1, 3).astype(int)
        fig, axes = plt.subplots(nc, 5, figsize=(14, 2.9 * nc), squeeze=False, layout='constrained')
        ext = self._extent()
        letters = iter('abcdefghijklmnopqrst')
        for c in range(nc):
            snaps = [self._slice(self.u[i, ..., c]) for i in times]
            norm, cmap = _signed_norm(np.stack(snaps))
            for ax, s, i in zip(axes[c, :3], snaps, times):
                m = ax.imshow(s, origin='lower', extent=ext, norm=norm, cmap=cmap)
                ax.text(0.98, 0.02, f't = {i * self.dt:.4g}', transform=ax.transAxes, ha='right', fontsize=7,
                        color=INK, bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))
            fig.colorbar(m, ax=axes[c, :3], label=f'${self._sym[c]}$', shrink=0.9)
            mean = self._slice(r['mean'][..., c])
            norm, cmap = _signed_norm(mean)
            m = axes[c, 3].imshow(mean, origin='lower', extent=ext, norm=norm, cmap=cmap)
            fig.colorbar(m, ax=axes[c, 3], label=rf'$\overline{{{self._sym[c]}}}$', shrink=0.9)
            m = axes[c, 4].imshow(self._slice(r['rms'][..., c]), origin='lower', extent=ext, cmap='Blues')
            fig.colorbar(m, ax=axes[c, 4], label=f"${self._sym[c]}'_{{\\mathrm{{rms}}}}$", shrink=0.9)
            for ax in axes[c]:
                ax.set(xlabel='$x$', ylabel='$y$')
                _letter(ax, next(letters))
        return fig

    def plot_statistics(self):
        with plt.rc_context(STYLE):
            return self._plot_statistics()

    def _plot_statistics(self):
        """Energy history, pdf, energy spectrum, two-point correlation, probe autocorrelation and spectrum."""
        r = self.results
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), layout='constrained')
        (a, b, c), (d, e, f) = axes
        a.plot(self.t, r['energy_t'], color=INK, lw=0.8)
        if r['i_stationary']:
            a.axvline(r['i_stationary'] * self.dt, color=C['mark'], ls=':', lw=1)
            a.text(r['i_stationary'] * self.dt, 1.0, ' stationary from here', transform=a.get_xaxis_transform(),
                   va='top', fontsize=7, color=C['mark'])
        a.set(xlabel='$t$', ylabel='fluctuation energy (domain mean)')
        self._overlay(a, 'energy')
        g = np.linspace(-6, 6, 400)
        for (xc, h), name, col in zip(r['pdf'], self._sym, SERIES):
            b.semilogy(xc, np.where(h > 0, h, np.nan), color=col, label=f"${name}'$")
        b.semilogy(g, np.exp(-g ** 2 / 2) / np.sqrt(2 * np.pi), color=C['grey'], lw=1, label='Gaussian')
        self._overlay(b, 'pdf')
        b.set(xlabel=r"$q'/q'_{\mathrm{rms}}$", ylabel='pdf', ylim=(1e-5, 1))
        b.legend()
        c.loglog(r['k'], r['E'], color=INK)
        self._overlay(c, 'E')
        c.set(xlabel='$k$', ylabel='$E(k)$', ylim=(max(r['E'].max() * 1e-12, r['E'][r['E'] > 0].min()),
                                                   2 * r['E'].max()))
        for (lag, R, L), name, col in zip(r['corr'], self.axes_names, SERIES):
            d.plot(lag, R, color=col, label=f'along ${name}$, $L$ = {L:.3g}')
        d.axhline(0, color=C['grey'], lw=0.8, zorder=0)
        self._overlay(d, 'corr')
        d.set(xlabel=r'separation $\Delta$', ylabel=r'$R(\Delta)$')
        d.legend()
        for acf, col, p in zip(r['acf'], SERIES, r['probes']):
            e.plot(r['acf_lag'], acf, color=col, lw=1, label=f'probe {p}')
            e.axhline(0, color=C['grey'], lw=0.8, zorder=0)
        e.set(xlabel=r'lag $\tau$', ylabel=r'$\rho(\tau)$', xlim=(0, min(r['acf_lag'][-1], 20 * r['T_int'])))
        self._overlay(e, 'acf')
        e.legend(fontsize=7)
        for (fq, s), col in zip(r['psd'], SERIES):
            f.loglog(fq[1:], s[1:], color=col, lw=1)
        if r['f_peak']:
            f.axvline(r['f_peak'], color=C['grey'], ls=':', lw=0.8)
        self._overlay(f, 'psd')
        if f.get_legend_handles_labels()[0]:
            f.legend(fontsize=7)
        f.set(xlabel='frequency', ylabel='PSD (probes)')
        for ax, s in zip(axes.ravel(), 'abcdef'):
            _letter(ax, s)
        return fig

    def plot_pod(self):
        with plt.rc_context(STYLE):
            return self._plot_pod()

    def _plot_pod(self):
        """POD energy spectrum and cumulative energy, and the leading modes of the first component."""
        r = self.results
        nm = len(r['pod_modes'])
        fig, axes = plt.subplots(1, 2 + nm, figsize=(3.0 * (2 + nm), 3.0), layout='constrained')
        n = np.arange(1, len(r['pod_energy']) + 1)
        axes[0].semilogy(n, r['pod_energy'], '.', color=INK, ms=3)
        axes[0].set(xlabel='mode $n$', ylabel=r'$\lambda_n / \sum\lambda$', xscale='log')
        self._overlay(axes[0], 'pod_energy')
        if axes[0].get_legend_handles_labels()[0]:
            axes[0].legend(fontsize=7)
        cum = np.cumsum(r['pod_energy'])
        axes[1].semilogx(n, cum, color=INK)
        for q, nq in ((0.9, r['pod_n90']), (0.99, r['pod_n99'])):
            axes[1].plot(nq, q, 'o', color=C['mark'], ms=4)
            axes[1].annotate(f'{q:.0%}: {nq} modes', (nq, q), xytext=(6, -10), textcoords='offset points',
                             fontsize=7, color=C['mark'])
        axes[1].set(xlabel='mode $n$', ylabel='cumulative energy', ylim=(0, 1.02))
        for j, ax in enumerate(axes[2:]):
            mode = r['pod_modes'][j][..., 0]
            if len(self.space) == 1:
                ax.plot(np.arange(self.space[0]) * self.dx[0], mode, color=INK)
                ax.set(xlabel='$x$', ylabel=rf'$\phi_{j + 1}$')
            else:
                norm, cmap = _signed_norm(self._slice(mode))
                ax.imshow(self._slice(mode), origin='lower', extent=self._extent(), norm=norm, cmap=cmap)
                ax.set(xlabel='$x$', ylabel='$y$')
            ax.text(0.98, 0.98, f'mode {j + 1}, {r["pod_energy"][j]:.1%}', transform=ax.transAxes, ha='right',
                    va='top', fontsize=7, color=INK, bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))
        for ax, s in zip(axes, 'abcdefgh'):
            _letter(ax, s)
        return fig

    def animate(self, path, n_frames=120, fps=12, dpi=72):
        with plt.rc_context(STYLE):
            return self._animate(path, n_frames, fps, dpi)

    def _animate(self, path, n_frames, fps, dpi):
        """Gif of the first component (mid-plane in 3D); not written for 1D fields (see `plot_fields`)."""
        from matplotlib.animation import FuncAnimation, PillowWriter

        if len(self.space) == 1:
            return None
        idx = np.linspace(0, self.nt - 1, min(self.nt, n_frames)).astype(int)
        frames = [self._slice(self.u[i, ..., 0]) for i in idx]
        norm, cmap = _signed_norm(np.stack(frames))
        ny, nx = frames[0].shape
        fig, ax = plt.subplots(figsize=(4.5 * min(2.5, max(0.6, nx / ny)), 4.0), layout='constrained')
        m = ax.imshow(frames[0], origin='lower', extent=self._extent(), norm=norm, cmap=cmap)
        fig.colorbar(m, ax=ax, label=f'${self._sym[0]}$', shrink=0.9)
        ax.set(xlabel='$x$', ylabel='$y$')
        clock = ax.text(0.98, 1.01, '', transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color=INK)

        def draw(n):
            m.set_data(frames[n])
            clock.set_text(f't = {idx[n] * self.dt:.4g}')
            return m, clock

        FuncAnimation(fig, draw, frames=len(frames), blit=True).save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
        return path

    # -- report -------------------------------------------------------------------------------------------------
    def report(self, out_dir, gif=True, notes=None):
        """Write figures, an animation, ``stats.json`` and a self-contained ``index.html`` to `out_dir`.

        `notes` is an optional list of HTML snippets (e.g. dataset provenance, how the record was loaded) placed
        under the summary table. Returns the path of the HTML page.
        """
        if self.results is None:
            self.analyze()
        os.makedirs(out_dir, exist_ok=True)
        figs = {'fields': (self.plot_fields(), 'Fields'), 'statistics': (self.plot_statistics(), 'Statistics'),
                'pod': (self.plot_pod(), 'POD')}
        paths = {}
        for name, (fig, _) in figs.items():
            paths[name] = os.path.join(out_dir, f'{name}.png')
            fig.savefig(paths[name], dpi=150)
            plt.close(fig)
        if gif and len(self.space) > 1:
            paths['animation'] = self.animate(os.path.join(out_dir, 'animation.gif'))
        desc = self.results['describe']
        with open(os.path.join(out_dir, 'stats.json'), 'w') as fh:
            json.dump(desc, fh, indent=1, default=float)
        page = os.path.join(out_dir, 'index.html')
        with open(page, 'w') as fh:
            fh.write(_html(desc, paths, self._captions(), notes or []))
        return page

    def _captions(self):
        r = self.results
        nd = len(self.space)
        fields = (f'(a, c, ...) space-time diagram of each component over the first {min(self.nt, 1500)} snapshots; '
                  '(b, d, ...) its time-mean and rms profiles over the whole record.'
                  if nd == 1 else
                  'Per component: three snapshots spanning the record, then the time mean and the rms.'
                  + (' Three-dimensional record: mid-plane of the first axis.' if nd == 3 else ''))
        stats = ('(a) domain-averaged fluctuation energy; (b) pdf of the fluctuations; (c) time-averaged energy '
                 'spectrum' + (' (shell-averaged)' if nd > 1 else '') + '; (d) two-point correlation per axis with its '
                 'integral length; (e, f) autocorrelation and Welch spectrum at the probes.')
        pod = (f'(a) POD energy per mode and (b) cumulative energy ({len(r["pod_energy"])} snapshots); '
               f'(c) onwards: leading modes of the first component.')
        return {'fields': fields, 'statistics': stats, 'pod': pod,
                'animation': 'Time evolution of the first component.'}


_LOOK_FOR = [
    'Energy history: a drift or a jump at the start is a transient to discard; the dashed line marks where '
    'the domain energy stops drifting.',
    'Record length in integral times: fewer than about a hundred gives noisy statistics and spectra.',
    'pdf: tails heavier than the Gaussian point to intermittency; an asymmetric pdf shows up as a non-zero '
    'skewness.',
    'Energy spectrum: the decay rate at high wavenumbers tells how well the grid resolves the field; a pile-up '
    'at the largest wavenumbers means the grid is too coarse or there is aliasing.',
    'Two-point correlation: a correlation that is still well above zero at half the domain means the box is '
    'too small for the largest structures.',
    'POD: the number of modes for 90 % and 99 % of the energy sets the scale of any reduced-order model.',
]


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


def _html(desc, paths, captions, notes):
    rows = '\n'.join(f'<tr><th>{html.escape(k)}</th><td>{_fmt(v)}</td></tr>' for k, v in desc.items()
                     if k not in ('checks', 'references'))
    checks = '\n'.join(f'<li>{html.escape(c)}</li>' for c in desc.get('checks', []))
    order = [k for k in ('animation', 'fields', 'statistics', 'pod') if paths.get(k)]
    extra = '\n'.join(f'<p>{n}</p>' for n in notes)
    if desc.get('references'):
        extra += '<p><strong>Reference data</strong> (grey, dashed): ' + '; '.join(
            html.escape(x) for x in desc['references']) + '.</p>'
    figs = '\n'.join(f'<figure><img src="{_img(paths[k])}" alt="{k}"><figcaption>{html.escape(captions[k])}'
                     f'</figcaption></figure>' for k in order)
    look = '\n'.join(f'<li>{html.escape(s)}</li>' for s in _LOOK_FOR)
    title = html.escape(str(desc['label']))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} first look</title>
<style>
:root {{ --bg: #ffffff; --fg: #222222; --muted: #666666; --rule: #dddddd; --accent: #4477AA; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #16181c; --fg: #e6e6e6;
  --muted: #a0a0a0; --rule: #333333; --accent: #66CCEE; }} }}
:root[data-theme="dark"] {{ --bg: #16181c; --fg: #e6e6e6; --muted: #a0a0a0; --rule: #333333; --accent: #66CCEE; }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; margin: 0 auto;
  max-width: 1100px; padding: 24px 16px; }}
h1 {{ font-size: 1.5rem; margin: 0 0 4px; }} h2 {{ font-size: 1.1rem; margin-top: 32px; color: var(--accent); }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--rule); vertical-align: top; }}
th {{ font-weight: 500; color: var(--muted); width: 40%; }}
figure {{ margin: 20px 0; }} img {{ max-width: 100%; background: #ffffff; border-radius: 4px; }}
figcaption {{ color: var(--muted); font-size: 0.9rem; margin-top: 6px; }}
.wrap {{ overflow-x: auto; }}
</style></head><body>
<h1>{title}</h1>
<p style="color: var(--muted)">First look at the record: statistics, spectra, correlations and POD energy, computed by
<code>ntsa.fields.FieldSeries</code>.</p>
<h2>Checks</h2>
<ul>{checks}</ul>
<h2>Summary</h2>
<div class="wrap"><table>{rows}</table></div>
{extra}
<h2>Figures</h2>
{figs}
<h2>What to look for</h2>
<ul>{look}</ul>
</body></html>
"""
