"""Data-driven front end: the equation-free half of ntsa on a raw measured series.

`DataSeries` wraps a scalar record x(t) sampled at a fixed dt and runs everything
that does not need model equations: embedding diagnostics, correlation dimension,
regime classification, signal statistics, MDS, and the 8-panel diagnostic row.
Lyapunov exponents and bifurcation sweeps need an integrable model and are
bypassed; an externally estimated `lam1` can be passed through to the classifier.
"""

import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from ntsa import tools as ntsa_tools
from ntsa.characterize import _tight, plot_lyapunov_fit, plot_row, save_figs_pdf_tight
from ntsa.classification import classify_regime
from ntsa.tools import lyapunov as lyap


@dataclass
class DataSeries:
    """A measured scalar time series and its equation-free ntsa characterization.

    Parameters
    ----------
    x : np.ndarray
        Scalar observable, shape (Nt,).
    dt : float
        Sampling time.
    label : str
        Case label used in figure titles.
    Y : np.ndarray, optional
        Simultaneous multivariate record (Nt, N) — e.g. the full measured state —
        used for the MDS panel; the delay embedding of `x` is used when omitted.
    trim : bool
        Trim any residual transient detected by `stationary_start` on construction
        (printed note when nonzero), mirroring `run_long`'s guard for model runs.
    """

    x: np.ndarray
    dt: float
    label: str = 'data'
    Y: np.ndarray = None
    trim: bool = True
    results: dict = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float).ravel()
        if self.Y is not None:
            self.Y = np.asarray(self.Y, dtype=float)
            if len(self.Y) != len(self.x):
                raise ValueError(f'len(Y)={len(self.Y)} != len(x)={len(self.x)}')
        if self.trim:
            i0 = ntsa_tools.stationary_start(self.x)
            if i0:
                print(f'[DataSeries] trimmed {i0 * self.dt:.3g} t.u. of transient '
                      f'({100 * i0 // len(self.x)}% of the record)')
                self.x = self.x[i0:]
                if self.Y is not None:
                    self.Y = self.Y[i0:]

    @property
    def t(self):
        return np.arange(len(self.x)) * self.dt

    def analyze(self, lam1='auto', lam1_std=0.0, mds=True):
        """Run the equation-free pipeline and cache the results dict.

        Same keys as one `ntsa.characterize.characterize` case, plus `D2`/`d2_fit`
        (Grassberger-Procaccia dimension of the delay embedding — the data-only
        substitute for the Kaplan-Yorke dimension). `lam1='auto'` (default)
        estimates the leading Lyapunov exponent from the data itself with
        `lyapunov.rosenstein_lyapunov` (nan when its fit guards reject, so
        non-chaotic data never gets a spurious exponent); pass a float to use an
        external estimate instead, or None to skip. `spectrum` stays None — a
        full spectrum needs model equations.
        """
        x, dt = self.x, self.dt
        zeta = ntsa_tools.optimal_lag(x)
        dim, fnn = ntsa_tools.false_nearest_neighbours(x, zeta)
        E = ntsa_tools.delay_embed(x, dim, zeta)
        D2, d2_fit = ntsa_tools.correlation_dimension(E)
        lyap_fit = None
        if isinstance(lam1, str):  # 'auto'
            try:
                lam1, lam1_std, lyap_fit = lyap.rosenstein_lyapunov(x, dt, zeta=zeta, dim=dim)
            except ValueError as err:
                print(f'[DataSeries] rosenstein_lyapunov skipped: {err}')
                lam1, lam1_std = None, 0.0
        regime, evidence = classify_regime(x, dt, lam1=lam1, lam1_std=lam1_std,
                                           t_total=len(x) * dt)
        gamma = t_gamma = None
        if mds:
            gamma, idx = ntsa_tools.classical_mds(self.Y if self.Y is not None else E)
            t_gamma = self.t[idx]
        self.results = {'label': self.label, 'zeta': zeta, 'dim': dim, 'fnn': fnn,
                        'D2': D2, 'd2_fit': d2_fit,
                        'lambda1': lam1, 'lambda1_std': lam1_std, 'lyap_fit': lyap_fit,
                        'spectrum': None, 'regime': regime, 'evidence': evidence,
                        'stats': ntsa_tools.signal_stats(x, dt),
                        'gamma': gamma, 't_gamma': t_gamma, 't': self.t, 'x': x}
        return self.results

    def characterize(self, pdf_name='figs/ntsa_data.pdf', **analyze_kwargs):
        """Draw the 8-panel diagnostic row (PDF + same-name PNG) via `plot_row`.

        Every panel is computed from the data alone; the Lyapunov panel shows
        the Rosenstein (or external) `lam1` marker, and the Rosenstein
        log-divergence fit is appended as a second PDF page when available.
        """
        res = self.analyze(**analyze_kwargs) if self.results is None or analyze_kwargs else self.results
        f1 = res['evidence'].get('f1')
        t_CR = 1.0 / f1 if f1 else 0.01 * len(self.x) * self.dt  # plot-window fallback only
        lam_ok = res['lambda1'] is not None and np.isfinite(res['lambda1'])
        lam_txt = f', $\\lambda_1$={res["lambda1"]:.3f}' if lam_ok else ''
        title = (f'{res["label"]}\n{res["regime"]}  $\\zeta$={res["zeta"]}, d={res["dim"]}, '
                 f'$D_2$={res["D2"]:.2f}{lam_txt}')
        fig = plt.figure(figsize=(19, 2.7), layout='constrained')
        _tight(fig)
        gs = fig.add_gridspec(1, 1)
        plot_row(fig, gs[0], res['t'], res['x'], self.dt, res['zeta'], res['dim'], t_CR,
                 title=title, evidence=res['evidence'], gamma=res['gamma'],
                 t_gamma=res['t_gamma'], lam1=res['lambda1'], lam1_std=res['lambda1_std'])
        figs = [fig]
        if res['lyap_fit'] is not None:
            figs.append(plot_lyapunov_fit(res['lyap_fit']))
        out_dir = os.path.dirname(pdf_name)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.splitext(pdf_name)[0] + '.png', dpi=150, bbox_inches='tight')
        save_figs_pdf_tight(pdf_name, figs)
        print(f'Saved figures --> {pdf_name}')
        return res
