"""Per-case nonlinear time-series characterization figures (Kantz & Schreiber style).

One row of 8 panels per case [time series + zoom inset | semilogy PSD | 3-D delay
portrait | first-return map | plane-crossing Poincaré section | recurrence plot |
3-D MDS | Lyapunov spectrum], followed by Lyapunov-fit, Lyapunov-spectrum and
MDS-embedding pages, all saved to a single PDF.

Run ``python -m ntsa.characterize --help`` for the demo driver.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from ntsa import tools as ntsa_tools
from ntsa.classification import classify_regime
from ntsa.tools import fun_PSD
from ntsa.tools import lyapunov as lyap


def save_figs_pdf_tight(pdf_name, figs):
    """Multi-page PDF with pages cropped to content (a save_figs_to_pdf
    cannot pass bbox_inches, and src/ is off-limits from dev/). Closes the figures."""
    with PdfPages(pdf_name) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)


def _tight(fig):
    """Shrink constrained-layout padding so subplots fill the canvas."""
    eng = fig.get_layout_engine()
    if eng is not None:
        eng.set(w_pad=0.01, h_pad=0.0, wspace=0.02, hspace=0.0)
    return fig

# ---------------------------------------------------------------------------
# Row of 5 diagnostic panels
# ---------------------------------------------------------------------------

def plot_row(fig, gs_row, t, x, dt, zeta, dim, t_CR, title=None, evidence=None,
             gamma=None, t_gamma=None, exponents=None, lam1=None, lam1_std=0.0):
    """Draw one 8-panel diagnostic row into a gridspec row: time series, PSD,
    delay portrait (with D_KY box), maxima return map, plane-crossing Poincare
    section, recurrence plot, 3-D MDS, Lyapunov spectrum.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure.
    gs_row : matplotlib.gridspec.SubplotSpec
        A gridspec slot spanning the row (subdivided into 8 columns here).
    t, x : np.ndarray
        Time vector and scalar observable, shape (Nt,).
    dt : float
        Sampling time of `t`.
    zeta : int
        Delay (samples) for the embedding.
    dim : int
        Embedding dimension (used by the recurrence panel).
    t_CR : float
        Characteristic time of the model; fallback for the zoom/recurrence windows.
    title : str, optional
        Multi-line row title, drawn as the y-label of the first panel.
    evidence : dict, optional
        ``classify_regime`` evidence; its PSD peaks (f1, f2, ratio) are marked and
        listed on the PSD panel so the LC/QP/CH call can be checked by eye.
    gamma, t_gamma : np.ndarray, optional
        ``classical_mds`` coordinates (T, >=2) and their times (colour scale) for
        the MDS panel; blank panel if omitted.
    exponents : np.ndarray, optional
        Lyapunov spectrum for the last panel; when omitted, a finite `lam1` (+/-
        `lam1_std`) is drawn as a single leading-exponent marker instead.
    """
    sub = gs_row.subgridspec(1, 8, wspace=0.05)
    ax_ts = fig.add_subplot(sub[0])
    ax_psd = fig.add_subplot(sub[1])
    ax_3d = fig.add_subplot(sub[2], projection='3d')
    ax_map = fig.add_subplot(sub[3])
    ax_poin = fig.add_subplot(sub[4])
    ax_rec = fig.add_subplot(sub[5])
    ax_mds = fig.add_subplot(sub[6], projection='3d')
    ax_lyap = fig.add_subplot(sub[7])

    # PSD first: the dominant frequency sizes the panel windows
    f, psd = fun_PSD(dt, x)
    p = psd[0]
    f1 = (evidence or {}).get('f1')
    if f1 is None:
        ip = 1 + int(np.argmax(p[1:]))  # skip DC
        f1 = f[ip] if f[ip] > 0 else None

    # oscillation timescale: mean inter-maximum interval (robust for broadband chaos,
    # where the PSD argmax can sit on a low-frequency bump far from the orbit frequency)
    xm_i, xm_ip1, pk_idx = ntsa_tools.first_return_map(x)
    if pk_idx.size > 2:
        T_osc = dt * float(np.mean(np.diff(pk_idx)))
    elif f1:
        T_osc = 1.0 / f1
    else:
        T_osc = t_CR

    # -- time series (red); zoom inset spans ~5 oscillation periods
    ax_ts.plot(t, x, color='tab:red', lw=0.4)
    ax_ts.set_xlabel('$t$')
    ax_ts.margins(x=0)
    if title:
        ax_ts.set_ylabel(title, fontsize=8)
    n_zoom = max(2, min(len(t) - 1, int(round(5 * T_osc / dt))))
    axins = ax_ts.inset_axes([0.08, 0.68, 0.55, 0.28])
    axins.plot(t[-n_zoom:], x[-n_zoom:], color='tab:red', lw=0.5)
    axins.tick_params(labelsize=5, length=2)
    axins.margins(x=0)

    # -- PSD (purple, semilogy)
    ax_psd.semilogy(f[1:], p[1:], color='purple', lw=0.4)
    top = p[1:].max()
    if top > 0:
        ax_psd.set_ylim(top * 1e-8, top * 3)
        # xlim: 5x the dominant frequency (peaks+harmonics), extended to the
        # frequency holding 99% of cumulative power so broadband chaos stays visible
        power = p[1:] ** 2
        f99 = f[1 + int(np.searchsorted(np.cumsum(power), 0.99 * power.sum()))]
        if f1:
            ax_psd.set_xlim(0, min(f[-1], max(5 * f1, f99)))
    ax_psd.set(xlabel='Frequency', ylabel='PSD')
    if evidence is not None:
        _annotate_psd_peaks(ax_psd, f, p, evidence)

    # -- 3-D delay portrait (green), with the Kaplan-Yorke dimension when a spectrum exists
    ax_3d.plot(x[:-2 * zeta], x[zeta:-zeta], x[2 * zeta:], color='green', lw=0.4)
    ax_3d.set_xlabel('$x(t)$', fontsize=8, labelpad=-2)
    ax_3d.set_ylabel(r'$x(t+\zeta)$', fontsize=8, labelpad=-2)
    ax_3d.set_zlabel(r'$x(t+2\zeta)$', fontsize=8, labelpad=-2)
    ax_3d.tick_params(labelsize=6, pad=-1)
    for axis in (ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis):
        axis.pane.fill = False
    if exponents is not None:
        ax_3d.text2D(0.02, 0.92, f'$D_{{KY}}$={lyap.kaplan_yorke(exponents):.2f}',
                     transform=ax_3d.transAxes, fontsize=6,
                     bbox=dict(boxstyle='round', fc='w', ec='0.6', alpha=0.8))

    # -- first-return map of local maxima (blue) with identity line
    if xm_i.size:
        lo = min(xm_i.min(), xm_ip1.min())
        hi = max(xm_i.max(), xm_ip1.max())
        # floor the span at 5% of the signal range: a period-1 cycle then shows as a
        # single dot instead of a magnified cloud of peak-sampling jitter
        min_span = 0.05 * np.ptp(x)
        if hi - lo < min_span:
            mid = 0.5 * (hi + lo)
            lo, hi = mid - 0.5 * min_span, mid + 0.5 * min_span
        pad = 0.05 * ((hi - lo) or max(abs(hi), 1.0))
        lo, hi = lo - pad, hi + pad
    else:
        lo, hi = 0.0, 1.0
    ax_map.plot([lo, hi], [lo, hi], color='grey', lw=0.8, zorder=1)
    if xm_i.size:
        ax_map.scatter(xm_i, xm_ip1, s=8, color='tab:blue', zorder=2)
    ax_map.set(xlim=(lo, hi), ylim=(lo, hi),
               xlabel=r'$x_{\max}(i)$', ylabel=r'$x_{\max}(i+1)$')
    ax_map.set_aspect('equal', adjustable='box')

    # -- plane-crossing Poincare section (orange): x(t+2*zeta) = median, upward
    # crossings — a period-k cycle gives k dots, a 2-torus a closed loop, chaos
    # a fractal scatter (complements the maxima return map)
    P = ntsa_tools.poincare_section(x, zeta)
    if len(P):
        plo = min(P[:, 0].min(), P[:, 1].min())
        phi = max(P[:, 0].max(), P[:, 1].max())
        min_span = 0.05 * np.ptp(x)  # same jitter floor as the return map
        if phi - plo < min_span:
            mid = 0.5 * (phi + plo)
            plo, phi = mid - 0.5 * min_span, mid + 0.5 * min_span
        pad = 0.05 * (phi - plo)
        ax_poin.scatter(P[:, 0], P[:, 1], s=4, color='darkorange')
        ax_poin.set(xlim=(plo - pad, phi + pad), ylim=(plo - pad, phi + pad))
    ax_poin.set(xlabel='$x(t)$', ylabel=r'$x(t+\zeta)$')
    ax_poin.set_aspect('equal', adjustable='box')

    # -- recurrence plot (binary) over a trailing window of ~10 oscillation periods,
    # floored at 500 samples: strongly chaotic signals carry many maxima per orbit, which
    # shrinks T_osc below the recurrence time and leaves only the main diagonal visible
    n_win = int(np.clip(10 * T_osc / dt, 500, 2000))
    n_win = min(n_win, len(x))
    n_min = (dim - 1) * zeta + 2
    if n_win <= n_min:
        n_win = min(len(x), n_min + 50)
    Y = ntsa_tools.delay_embed(x[-n_win:], dim, zeta)
    R = ntsa_tools.recurrence_matrix(Y)
    if not 0.03 <= R.mean() <= 0.4:  # degenerate density -> fixed 10% recurrence rate
        R = ntsa_tools.recurrence_matrix(Y, rr=0.10)
    # embedded vector i sits at t[-n_win + i]: the axis ends (dim-1)*zeta samples before t[-1]
    t_end = t[-n_win] + (R.shape[0] - 1) * dt
    ax_rec.imshow(R, cmap='binary', origin='lower',
                  extent=[t[-n_win], t_end, t[-n_win], t_end], aspect='auto')
    ax_rec.set(xlabel='$t$', ylabel='$t$')

    # -- 3-D MDS portrait (gamma_1, gamma_2, gamma_3) coloured by time
    if gamma is not None:
        c = t_gamma if t_gamma is not None else np.arange(len(gamma))
        g3 = gamma[:, 2] if gamma.shape[1] > 2 else np.zeros(len(gamma))
        ax_mds.scatter(gamma[:, 0], gamma[:, 1], g3, s=2, c=c, cmap='viridis',
                       alpha=0.6, rasterized=True)
        ax_mds.set_xlabel(r'$\gamma_1$', fontsize=8, labelpad=-2)
        ax_mds.set_ylabel(r'$\gamma_2$', fontsize=8, labelpad=-2)
        ax_mds.set_zlabel(r'$\gamma_3$', fontsize=8, labelpad=-2)
        ax_mds.tick_params(labelsize=6, pad=-1)
        for axis in (ax_mds.xaxis, ax_mds.yaxis, ax_mds.zaxis):
            axis.pane.fill = False
    else:
        ax_mds.set_axis_off()

    # -- Lyapunov spectrum (or the leading exponent alone when no spectrum was run)
    if exponents is not None:
        exponents = np.asarray(exponents)
        pos = exponents > 0
        kk = np.arange(1, len(exponents) + 1)
        ax_lyap.axhline(0.0, color='grey', lw=0.6)
        ax_lyap.scatter(kk[pos], exponents[pos], s=14, color='tab:red', zorder=3)
        ax_lyap.scatter(kk[~pos], exponents[~pos], s=14, color='tab:blue', zorder=3)
        for j, lam in zip(kk[pos], exponents[pos]):  # print every positive exponent
            ax_lyap.annotate(f'{lam:.3g}', xy=(j, lam), textcoords='offset points',
                             xytext=(4, 3), fontsize=6)
        linthresh = max(0.05, 2 * exponents[pos].max()) if pos.any() else 0.05
        ax_lyap.set_yscale('symlog', linthresh=linthresh)
        ax_lyap.set(xlabel='$j$', ylabel=r'$\lambda_j$')
    elif lam1 is not None and np.isfinite(lam1):
        ax_lyap.axhline(0.0, color='grey', lw=0.6)
        ax_lyap.errorbar([1], [lam1], yerr=[lam1_std], fmt='o', ms=4,
                         color='tab:red' if lam1 > 0 else 'tab:blue', capsize=3)
        if lam1 > 0:
            ax_lyap.annotate(f'{lam1:.3g}', xy=(1, lam1), textcoords='offset points',
                             xytext=(6, 3), fontsize=6)
        ax_lyap.set(xlim=(0.5, 1.5), xticks=[1], xlabel='$j$', ylabel=r'$\lambda_1$')
    else:
        ax_lyap.axis('off')

    for ax in (ax_ts, ax_psd, ax_map, ax_poin, ax_rec, ax_lyap):
        ax.tick_params(labelsize=7)


def _annotate_psd_peaks(ax, f, p, evidence):
    """Mark classify_regime's PSD peaks and list their values in a text box."""
    peaks = evidence.get('psd_peak_freqs')
    xmax = ax.get_xlim()[1]
    if peaks is not None and len(peaks):
        fp = np.asarray(peaks)
        fp = fp[fp <= xmax]
        idx = np.searchsorted(f, fp).clip(1, len(f) - 1)
        if idx.size > 6:  # only the strongest few, or broadband spectra drown in markers
            keep = np.argsort(p[idx])[-6:]
            fp, idx = fp[keep], idx[keep]
        ax.plot(fp, p[idx] * 2.0, marker='v', ls='none', ms=3, color='k', zorder=3)
    lines = []
    for key in ('f1', 'f2'):
        if evidence.get(key) is not None:
            lines.append(f'$f_{key[1]}$={evidence[key]:.4g}')
    if evidence.get('f1') and evidence.get('f2'):
        ratio = f'$f_2/f_1$={evidence["f2"] / evidence["f1"]:.3f}'
        if evidence.get('rational_match'):
            ratio += f'$\\approx${evidence["rational_match"]}'
        lines.append(ratio)
    if lines:
        ax.text(0.97, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=6,
                ha='right', va='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7', alpha=0.8))


# ---------------------------------------------------------------------------
# Lyapunov and MDS pages
# ---------------------------------------------------------------------------

def plot_lyapunov_fit(res):
    """Plot the leading-Lyapunov log-separation curves and the linear fit.

    Parameters
    ----------
    res : dict
        Result dict from ``lyapunov.leading_lyapunov`` (keys ``t``, ``log_sep``,
        ``mean_log_sep``, ``i1``, ``i2``, ``r2``, ``lam1``, ``lam1_std``).
    """
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    _tight(fig)
    t = np.asarray(res['t'])
    ax.plot(t, res['log_sep'], color='grey', lw=0.5, alpha=0.5)
    ax.plot(t, res['mean_log_sep'], color='k', lw=1.8, label='mean')

    lam1 = res['lam1']
    if lam1 is not None and np.isfinite(lam1):
        i1, i2 = res['i1'], res['i2']
        tf = t[i1:i2]
        line = np.asarray(res['mean_log_sep'])[i1] + lam1 * (tf - tf[0])
        ax.plot(tf, line, color='tab:red', ls='--', lw=1.5,
                label=rf'$\lambda_1 = {lam1:.3f} \pm {res["lam1_std"]:.3f}$  ($R^2={res["r2"]:.2f}$)')
    else:
        ax.set_title(f'no reliable exponential growth ($R^2$ = {res["r2"]:.2f})', fontsize=9)
    ax.set(xlabel='$t$', ylabel='log separation')
    ax.legend(fontsize=8)
    return fig


def plot_lyapunov_spectrum(exponents):
    """Scatter the Lyapunov spectrum vs. index on a symlog axis."""
    exponents = np.asarray(exponents, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    _tight(fig)
    idx = np.arange(1, len(exponents) + 1)
    pos = exponents > 0
    linthresh = max(0.05, 2 * np.abs(exponents[pos]).max()) if pos.any() else 0.05

    ax.plot(idx, exponents, color='grey', lw=0.8, alpha=0.5, zorder=1)
    ax.scatter(idx[pos], exponents[pos], color='tab:red', s=40, zorder=3, label=r'$\lambda_j > 0$')
    ax.scatter(idx[~pos], exponents[~pos], color='tab:blue', s=40, zorder=3, label=r'$\lambda_j \leq 0$')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_yscale('symlog', linthresh=linthresh, linscale=0.5)
    ax.set(xlabel='Index $j$', ylabel=r'$\lambda_j$')
    if pos.any():  # print the value of every positive exponent
        for j, lam in zip(idx[pos], exponents[pos]):
            ax.annotate(f'{lam:.3g}', xy=(j, lam), textcoords='offset points',
                        xytext=(6, 4), fontsize=8)
    else:
        ax.annotate(rf'$\lambda_1 = {exponents[0]:.3g}$', xy=(1, exponents[0]),
                    xytext=(1 + max(len(exponents) / 8, 1.0), exponents[0]),
                    arrowprops=dict(arrowstyle='->', color='k'), fontsize=10)
    ax.legend(fontsize=8, loc='lower left')
    return fig


def plot_mds(gamma, t_sub):
    """Classical-MDS embedding: 2-D (gamma_1, gamma_2) and 3-D (gamma_1..3) coloured by time."""
    fig = plt.figure(figsize=(11, 5), layout='constrained')
    _tight(fig)
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')

    kw = dict(c=t_sub, cmap='viridis', s=4, alpha=0.5, rasterized=True)
    sc = ax2d.scatter(gamma[:, 0], gamma[:, 1], **kw)
    ax3d.scatter(gamma[:, 0], gamma[:, 1], gamma[:, 2], **kw)
    fig.colorbar(sc, ax=ax2d, label='$t$', shrink=0.7)

    ax2d.set(xlabel=r'$\gamma_1$', ylabel=r'$\gamma_2$')
    ax3d.set(xlabel=r'$\gamma_1$', ylabel=r'$\gamma_2$', zlabel=r'$\gamma_3$')
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.fill = False
    ax3d.grid(False)
    ax3d.set_box_aspect([1, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def _lam1_text(res):
    """Displayable leading exponent: prefer the Benettin spectrum; flag transient-growth
    fits that are not significantly positive with '*' (they contradict non-chaotic labels)."""
    if res['spectrum'] is not None:
        return f'{res["spectrum"][0]:.3f}'
    lam = res['lambda1']
    if lam is None or not np.isfinite(lam):
        return ''
    return (f'{lam:.3f}$\\pm${res["lambda1_std"]:.2g}'
            + ('*' if lam - 2 * res['lambda1_std'] <= 0 else ''))


def _default_label(model):
    pars = ', '.join(f'{p}={getattr(model, p):.4g}' for p in model.params)
    return f'{model.name} ({pars})' if pars else model.name


def characterize(models, labels=None, obs_idx=0, t_run=None, t_transient=None,
                 lyapunov='auto', mds=True,
                 spectrum='auto', rows_per_page=4, pdf_name='figs/ntsa_characterization.pdf'):
    """Characterize one or more models: run, embed, classify, and plot to a multi-page PDF.

    Parameters
    ----------
    models : Model or list of Model
        Model instance(s); each is respawned so the caller's copies are untouched.
    labels : list of str, optional
        Case labels; defaults to ``name (param=value, ...)``.
    obs_idx : int
        Index of the observable used for the scalar analysis.
    t_run : float, optional
        Simulation horizon (defaults to ``100 * model.t_CR`` per case).
    t_transient : float, optional
        Discarded transient (defaults to ``model.t_transient``). Raise it for
        cases near a bifurcation (critical slowing down), where residual drift
        pollutes the return map — ``run_long`` trims detectable amplitude drift,
        but structural drift (e.g. slow phase locking) it cannot see.
    lyapunov : {'auto', True, False}
        Run ``leading_lyapunov`` (works for any model); 'auto'/True wraps in try/except.
    mds : bool
        Compute and plot the classical-MDS embedding of the full state.
    spectrum : {'auto', True, False}
        Run ``lyapunov_spectrum``; 'auto' only when the model has ``time_derivative``.
    rows_per_page : int
        Diagnostic rows per PDF page.
    pdf_name : str
        Output PDF path; the first row-grid page is also saved as a same-name PNG.

    Returns
    -------
    list of dict
        Per case: label, zeta, dim, fnn, lambda1, lambda1_std, lyap_fit, spectrum,
        regime, evidence, stats, gamma, t, x.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    if labels is None:
        labels = [_default_label(mi) for mi in models]
    elif isinstance(labels, str):
        labels = [labels]

    results, row_data = [], []
    for mi, case_label in zip(models, labels):
        print(f'-- characterizing: {case_label}')
        run_model = ntsa_tools.respawn(mi)
        t, y, psi = ntsa_tools.run_long(run_model, t_run if t_run is not None else 100 * mi.t_CR,
                                        t_transient=t_transient)
        x = y[:, obs_idx]
        dt = float(t[1] - t[0])

        zeta = ntsa_tools.optimal_lag(x, max_lag=max(2, int(mi.t_CR / mi.dt)))
        dim, fnn = ntsa_tools.false_nearest_neighbours(x, zeta)

        lam1, lam1_std, lyap_res = None, 0.0, None
        if lyapunov:
            try:
                lam1, lam1_std, lyap_res = lyap.leading_lyapunov(ntsa_tools.respawn(mi))
            except Exception as err:
                print(f'   [warning] leading_lyapunov failed: {err}')

        exps = None
        if spectrum and (spectrum != 'auto' or hasattr(mi, 'time_derivative')):
            try:
                exps = lyap.lyapunov_spectrum(ntsa_tools.respawn(mi))
            except Exception as err:
                print(f'   [warning] lyapunov_spectrum failed: {err}')

        # classify with the Benettin lam1 whenever a spectrum was computed: it is
        # convergence-controlled (std ~ 0), whereas the perturbation-growth fit
        # carries a member spread that inflates the 3-sigma trust floor and can be
        # biased by non-normal transients. Without a spectrum, use the growth fit.
        lam1_cls, lam1_std_cls = lam1, lam1_std
        if exps is not None:
            lam1_cls, lam1_std_cls = float(exps[0]), 0.0
        regime, evidence = classify_regime(x, dt, lam1=lam1_cls,
                                           lam1_std=lam1_std_cls,
                                           t_total=t[-1] - t[0],
                                           exponents=exps)

        gamma, t_sub = None, None
        if mds:
            gamma, idx = ntsa_tools.classical_mds(psi)
            t_sub = t[idx]

        stats = ntsa_tools.signal_stats(x, dt)

        results.append({'label': case_label, 'zeta': zeta, 'dim': dim, 'fnn': fnn,
                        'lambda1': lam1, 'lambda1_std': lam1_std, 'lyap_fit': lyap_res,
                        'spectrum': exps, 'regime': regime, 'evidence': evidence,
                        'stats': stats, 'gamma': gamma, 't': t, 'x': x})
        row_data.append({'dt': dt, 't_CR': mi.t_CR, 't_sub': t_sub})

    # -- figure pages: row grids, then per-case Lyapunov fit / spectrum / MDS
    figs = []
    for page0 in range(0, len(results), rows_per_page):
        chunk = results[page0:page0 + rows_per_page]
        fig = plt.figure(figsize=(19, 2.7 * len(chunk)), layout='constrained')
        _tight(fig)
        gs = fig.add_gridspec(len(chunk), 1)
        for r, res in enumerate(chunk):
            rd = row_data[page0 + r]
            lam_txt = _lam1_text(res)
            lam_txt = f', $\\lambda_1$={lam_txt}' if lam_txt else ''
            title = f'{res["label"]}\n{res["regime"]}  $\\zeta$={res["zeta"]}, d={res["dim"]}{lam_txt}'
            plot_row(fig, gs[r], res['t'], res['x'], rd['dt'], res['zeta'], res['dim'],
                     rd['t_CR'], title=title, evidence=res['evidence'],
                     gamma=res['gamma'], t_gamma=rd['t_sub'], exponents=res['spectrum'],
                     lam1=res['lambda1'], lam1_std=res['lambda1_std'])
        figs.append(fig)
    n_rows_pages = len(figs)

    for res in results:
        if res['lyap_fit'] is not None:
            fig = plot_lyapunov_fit(res['lyap_fit'])
            fig.suptitle(res['label'], fontsize=10)
            figs.append(fig)
    for res in results:
        if res['spectrum'] is not None:
            fig = plot_lyapunov_spectrum(res['spectrum'])
            fig.suptitle(res['label'], fontsize=10)
            figs.append(fig)
    for res, rd in zip(results, row_data):
        if res['gamma'] is not None:
            fig = plot_mds(res['gamma'], rd['t_sub'])
            fig.suptitle(res['label'], fontsize=10)
            figs.append(fig)

    out_dir = os.path.dirname(pdf_name)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if n_rows_pages:  # PNG of the first row-grid page (save before the PDF closes the figures)
        figs[0].savefig(os.path.splitext(pdf_name)[0] + '.png', dpi=150, bbox_inches='tight')
    save_figs_pdf_tight(pdf_name, figs)
    print(f'Saved figures --> {pdf_name}')
    return results


# ---------------------------------------------------------------------------
# Demo driver
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import argparse

    from dynamodels.physical import Lorenz63, Lorenz96, VdP

    parser = argparse.ArgumentParser(description='NTSA characterization of dynamodels-style models.')
    parser.add_argument('--model', choices=['lorenz63', 'lorenz96', 'vdp'], default=None,
                        help='single-model sweep (requires --param and --values); default: 4-case demo')
    parser.add_argument('--param', default=None, help='parameter to sweep, e.g. rho')
    parser.add_argument('--values', nargs='+', type=float, default=None, help='parameter values')
    parser.add_argument('--t-run', type=float, default=None, help='run horizon (default 100*t_CR)')
    parser.add_argument('--t-transient', type=float, default=None,
                        help='discarded transient (default model.t_transient); raise near bifurcations')
    parser.add_argument('--no-lyapunov', action='store_true')
    parser.add_argument('--no-spectrum', action='store_true')
    parser.add_argument('--no-mds', action='store_true')
    args = parser.parse_args()

    if args.model:
        if not args.param or not args.values:
            parser.error('--model requires --param and --values')
        cls = {'lorenz63': Lorenz63, 'lorenz96': Lorenz96, 'vdp': VdP}[args.model]
        base = cls()
        cases = [ntsa_tools.respawn(base, **{args.param: v}) for v in args.values]
        tag = f'{args.model}_{args.param}'
    else:
        l63 = Lorenz63()
        cases = [l63,                                       # chaotic
                 ntsa_tools.respawn(l63, rho=350., dt=0.005),  # period-1 window (finer dt: fast orbit)
                 VdP(),                                     # limit cycle
                 Lorenz96(Nx=10)]                           # chaotic, F=8
        tag = 'defaults'

    out = characterize(cases,
                       t_run=args.t_run,
                       t_transient=args.t_transient,
                       lyapunov=False if args.no_lyapunov else 'auto',
                       spectrum=False if args.no_spectrum else 'auto',
                       mds=not args.no_mds,
                       pdf_name=f'figs/ntsa_{tag}.pdf')

    hdr = f'{"case":<42} {"regime":<24} {"zeta":>5} {"d":>3} {"lam1":>16}'
    print('\n' + hdr)
    print('-' * len(hdr))
    for res in out:
        lam_txt = _lam1_text(res) or '--'
        print(f'{res["label"]:<42} {res["regime"]:<24} {res["zeta"]:>5d} {res["dim"]:>3d} {lam_txt:>16}')
    if any(_lam1_text(res).endswith('*') for res in out):
        print('* transient-growth fit not significantly positive (no spectrum available)')
