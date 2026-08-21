"""Bifurcation sweeps and diagrams.

Run as a script for a bifurcation-diagram demo, e.g.::

    python -m ntsa.bifurcation --model lorenz63 --param rho --pmin 20 --pmax 100 --save-figs
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from tqdm import tqdm

from ntsa.tools.maps import stationary_start
from ntsa.tools.runners import respawn, run_long

EXTREMA_COLORS = dict(max='k', min='crimson')


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
