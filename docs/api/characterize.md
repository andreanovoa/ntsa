# `ntsa.characterize`

## At a glance

| Function | One-liner |
| --- | --- |
| `plot_row(fig, gs_row, t, x, dt, zeta, dim, t_CR, title=None, evidence=None, gamma=None, t_gamma=None, exponents=None, lam1=None, lam1_std=0)` | One 8-panel diagnostic row into a gridspec slot: time series, PSD (peaks + f1/f2 box from `evidence`), delay portrait (+ D_KY box), maxima return map, plane-crossing Poincaré section, recurrence plot, 3-D MDS, Lyapunov spectrum (single-λ1 marker fallback). |
| `plot_lyapunov_fit(res)` | Log-separation curves + fitted λ1 line from `leading_lyapunov`. |
| `plot_lyapunov_spectrum(exponents)` | Spectrum vs. index (symlog, red λ>0 / blue λ≤0). |
| `plot_mds(gamma, t_sub)` | 2-D + 3-D MDS embedding coloured by time. |
| `characterize(models, ...)` | Full pipeline per case → multi-page PDF + list of result dicts. |


## Full reference

::: ntsa.characterize
