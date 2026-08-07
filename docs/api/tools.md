# `ntsa.tools`

## At a glance

| Function | One-liner |
| --- | --- |
| `respawn(model, psi0=None, dt=None, **overrides)` | Fresh model instance with `alpha0` + `fixed_params` carried over; overrides win. |
| `run_long(model, t_run, t_transient=None, trim_transient=True)` | Integrate past transient; returns `(t, y, psi)` for member 0 (serial, m=1). Residual transient drift is detected and trimmed (printed note). |
| `stationary_start(x, tol=0.05, block=10)` | First sample where the maxima envelope stops drifting (residual-transient guard). |
| `delay_embed(x, dim, lag)` | Takens delay-embedding matrix `(N-(dim-1)*lag, dim)`. |
| `average_mutual_information(x, max_lag, n_bins=64)` | AMI for lags `1..max_lag`. |
| `optimal_lag(x, max_lag=None, n_bins=64)` | Delay $\zeta$ = first local minimum of the AMI (samples). |
| `false_nearest_neighbours(x, lag, ...)` | Kennel FNN embedding dimension `d` and per-dimension fractions. |
| `first_return_map(x, prominence=None)` | Successive local maxima `(x_max(i), x_max(i+1))` + peak indices. |
| `poincare_section(x, zeta, level=None, direction=1)` | Plane-crossing section of the delay embedding at $x(t+2\zeta) = \text{level}$: period-k $\to$ k dots, 2-torus $\to$ loop, chaos $\to$ fractal scatter. |
| `correlation_dimension(Y, nmax=2000)` | Grassberger–Procaccia $D_2$ of a trajectory/embedding/section (k-torus $\to$ k; section of a k-torus $\to$ k-1). |
| `count_peak_clusters(xm, x_range, tol=0.02)` | Number of distinct peak levels (period-k detection). |
| `recurrence_matrix(Y, eps_frac=0.10)` | Boolean recurrence matrix, threshold = `eps_frac` $\times$ max distance. |
| `autocorrelation(x, n_lags)` | Normalized ACF, `acf[0]=1`. |
| `signal_stats(x, dt, n_bins=64)` | Dict of moments, PSD, PDF and ACF. |
| `classical_mds(X, n_coords=3, nmax=2000)` | Classical MDS coordinates $\gamma$ of the state trajectory (subsampled). |
| `classify_regime(x, dt, lam1=None, ...)` | `(label, evidence)` — see decision tree below. |
| `bifurcation_sweep(model, param, values, ...)` | Parameter sweep collecting observable extrema: one ensemble forecast — member `k` at `values[k]` — by default, or serial branch-following with `continuation=True`. |
| `plot_bifurcation(values, peaks, ...)` | Bifurcation diagram from `bifurcation_sweep` output. |


## Full reference

::: ntsa.tools
