# ntsa — nonlinear time-series analysis for dynamical-system models

Characterizes the dynamical regime of a model from a single long trajectory:
delay embedding (optimal lag + false nearest neighbours), Lyapunov exponents, regime
classification (fixed point / limit cycle period-k / frequency-locked / quasiperiodic / chaotic),
and a per-case diagnostic figure — one row of 8 panels
[time series (red, with zoom inset) | PSD (purple, semilogy) | 3-D delay portrait (green,
with D_KY box) | first-return map of local maxima (blue) | plane-crossing Poincaré section
(orange) | recurrence plot (black/white) | 3-D MDS | Lyapunov spectrum] — plus Lyapunov-fit,
Lyapunov-spectrum and classical-MDS pages, all in one multi-page PDF.

Reference: Kantz & Schreiber, *Nonlinear Time Series Analysis* (2004).

```bash
pip install ntsa            # once released; until then:
pip install "ntsa @ git+https://github.com/andreanovoa/ntsa"
```

## Quickstart

```python
from dynamodels.physical import Lorenz63
from ntsa import characterize as chz

chz.characterize([Lorenz63()], pdf_name='figs/l63.pdf')
```

```bash
python -m ntsa.characterize                     # 4-case demo -> figs/ntsa_defaults.pdf (+ .png)
python -m ntsa.characterize --model lorenz63 --param rho --values 20 28 100 350
python -m ntsa.tools --model lorenz63 --param rho --nP 40   # bifurcation sweep demo
```

## Model protocol

Every function takes a model by duck type — no base-class requirement.
[`dynamodels`](https://github.com/andreanovoa/dynamodels) is the reference
implementation (and supplies the demo/test models), but any object with this
interface works:

| Requirement | Used for |
| --- | --- |
| constructor `Cls(psi0=..., dt=..., **params)` | `respawn` re-instantiation |
| `dt`, `t_transient`, `t_CR` (optional `t_lyap`) | run horizons and reference times |
| `Nphi`, `Nq`, `psi0 (Nphi, m)`, `alpha0` dict, `fixed_params`, `params` | state/parameter bookkeeping |
| `obs_labels`, `alpha_labels`, `name` | figure labels |
| `time_integrate(Nt) -> (psi, t)`, `update_history(psi, t, reset=)`, `hist`, `hist_t`, `get_observable_hist()`, `close()` | trajectory generation (`run_long`) |
| `time_derivative(t, psi, **params)` + `governing_eqns_params` | QR spectrum only (`lyapunov_spectrum`); discrete maps can still use `leading_lyapunov` |

## API

### `ntsa.tools`

| Function | One-liner |
| --- | --- |
| `respawn(model, psi0=None, dt=None, **overrides)` | Fresh model instance with `alpha0` + `fixed_params` carried over; overrides win. |
| `run_long(model, t_run, t_transient=None, trim_transient=True)` | Integrate past transient; returns `(t, y, psi)` for member 0 (serial, m=1). Residual transient drift is detected and trimmed (printed note). |
| `stationary_start(x, tol=0.05, block=10)` | First sample where the maxima envelope stops drifting (residual-transient guard). |
| `delay_embed(x, dim, lag)` | Takens delay-embedding matrix `(N-(dim-1)*lag, dim)`. |
| `average_mutual_information(x, max_lag, n_bins=64)` | AMI for lags `1..max_lag`. |
| `optimal_lag(x, max_lag=None, n_bins=64)` | Delay ζ = first local minimum of the AMI (samples). |
| `false_nearest_neighbours(x, lag, ...)` | Kennel FNN embedding dimension `d` and per-dimension fractions. |
| `first_return_map(x, prominence=None)` | Successive local maxima `(x_max(i), x_max(i+1))` + peak indices. |
| `poincare_section(x, zeta, level=None, direction=1)` | Plane-crossing section of the delay embedding at `x(t+2ζ)=level`: period-k → k dots, 2-torus → loop, chaos → fractal scatter. |
| `correlation_dimension(Y, nmax=2000)` | Grassberger–Procaccia D2 of a trajectory/embedding/section (k-torus → k; section of a k-torus → k−1). |
| `count_peak_clusters(xm, x_range, tol=0.02)` | Number of distinct peak levels (period-k detection). |
| `recurrence_matrix(Y, eps_frac=0.10)` | Boolean recurrence matrix, threshold = `eps_frac` × max distance. |
| `autocorrelation(x, n_lags)` | Normalized ACF, `acf[0]=1`. |
| `signal_stats(x, dt, n_bins=64)` | Dict of moments, PSD, PDF and ACF. |
| `classical_mds(X, n_coords=3, nmax=2000)` | Classical MDS coordinates γ of the state trajectory (subsampled). |
| `classify_regime(x, dt, lam1=None, ...)` | `(label, evidence)` — see decision tree below. |
| `bifurcation_sweep(model, param, values, ...)` | Serial parameter sweep collecting observable extrema (continuation optional). |
| `plot_bifurcation(values, peaks, ...)` | Bifurcation diagram from `bifurcation_sweep` output. |

### `ntsa.lyapunov`

| Function | One-liner |
| --- | --- |
| `get_rhs(model)` | RHS callable `f(u)` from `model.time_derivative` (continuous models only). |
| `fd_jacobian(f, u, eps=1e-6)` | Finite-difference Jacobian. |
| `get_jacobian(model, f=None)` | Analytic Jacobian for Lorenz63/96, finite-difference otherwise. |
| `lyapunov_spectrum(model, ...)` | Full spectrum via tangent-space QR (Benettin / Gram-Schmidt), sorted descending; runs until the running exponents converge (halving test), `full_output=True` returns the convergence history. |
| `leading_lyapunov(model, n_pert=8, ...)` | λ1 ± std from perturbation-growth fit; works for **any** Model. |
| `fit_log_growth(t, log_seps)` | Linear-region fit of log separation; `nan` if R² < 0.5. |
| `kaplan_yorke(exponents)` | Kaplan–Yorke dimension from a spectrum (shown as a box on the delay-portrait panel). |
| `covariant_lyapunov_vectors(...)` | **TODO stub** (Ginelli et al. 2007) — raises `NotImplementedError`. |

### `ntsa.characterize`

| Function | One-liner |
| --- | --- |
| `plot_row(fig, gs_row, t, x, dt, zeta, dim, t_CR, title=None, evidence=None, gamma=None, t_gamma=None, exponents=None, lam1=None, lam1_std=0)` | One 8-panel diagnostic row into a gridspec slot: time series, PSD (peaks + f1/f2 box from `evidence`), delay portrait (+ D_KY box), maxima return map, plane-crossing Poincaré section, recurrence plot, 3-D MDS, Lyapunov spectrum (single-λ1 marker fallback). |
| `plot_lyapunov_fit(res)` | Log-separation curves + fitted λ1 line from `leading_lyapunov`. |
| `plot_lyapunov_spectrum(exponents)` | Spectrum vs. index (symlog, red λ>0 / blue λ≤0). |
| `plot_mds(gamma, t_sub)` | 2-D + 3-D MDS embedding coloured by time. |
| `characterize(models, ...)` | Full pipeline per case → multi-page PDF + list of result dicts. |

## Defaults

| Quantity | Default |
| --- | --- |
| Delay ζ | First local minimum of the average mutual information |
| Embedding dimension d | Kennel FNN with `Rtol=10`, `Atol=2`, threshold 1% |
| Recurrence threshold ε | 10% of the max pairwise distance; the panel retargets to a fixed 10% recurrence rate (`rr=0.10` quantile) when the density degenerates |
| Zoom inset / recurrence window | ~5 / ~10 mean inter-maximum intervals (recurrence window clipped to [500, 2000] samples) |
| MDS / recurrence subsample | 2000 points max |
| λ1 trust tolerance | `max(3σ, 10/T)` (λ1 must exceed this to call chaos) |
| Spectrum integrator | RK4, `dt = min(model.dt, 0.01)` |
| Spectrum horizon | floor `200 * t_ref`, extended (up to `t_max = 20 * floor`) until `\|λ(T) − λ(T/2)\| < max(atol=2e-3, rtol=5% · \|λ\|)` at two consecutive checks; warns if capped unconverged |
| Spectrum / leading-λ1 warmup | `_settled_state`: transient doubled (up to 4×) while `stationary_start` still sees amplitude drift in a `20*t_ref` probe |
| Run horizon (`characterize`) | `100 * model.t_CR`, after discarding `model.t_transient`; residual drift trimmed by `stationary_start` |

## Regime classification decision tree

1. **Fixed point** — signal variance collapses (no sustained oscillation), or the full
   Lyapunov spectrum is entirely negative (decisive when integrator noise keeps the tail
   wiggling at a stable focus, e.g. Lorenz-63 at ρ=10: λ = [−0.6, −0.6, −12.5]).
2. **Limit cycle, period-k** — return-map local maxima form k ≤ `k_max` distinct *tight*
   clusters. Checked **before** the Lyapunov step: k tight maxima levels over a long record
   are incompatible with chaos, whereas a perturbation-growth λ1 can read large and positive
   on a stable orbit through non-normal transient amplification (Rijke limit cycles read
   λ1 ~ 200 otherwise; `leading_lyapunov` additionally rejects fits whose saturation level
   sits far below the attractor diameter).
3. **Chaotic** — λ1 positive beyond the trust tolerance. `characterize()` classifies with
   the Benettin spectrum's λ1 whenever a spectrum was computed (it is convergence-controlled,
   std ≈ 0), and with the perturbation-growth λ1 otherwise (whose member spread sets the
   3σ part of the tolerance).
4. Otherwise: if a full spectrum is available, ≥ 2 neutral exponents (|λ| < 2e-3) →
   **quasiperiodic** (a 2-torus has two zero exponents; a locked periodic orbit exactly
   one — more robust than the PSD ratio test, whose `limit_denominator(10)` rationals are
   dense enough to "match" incommensurate ratios, e.g. L96 F=4.4: f2/f1 = 0.552 ≈ 5/9).
   Else dominant PSD frequencies rationally related → **frequency-locked**;
   incommensurate → **quasiperiodic**.

Caveat: the period-k count is the number of distinct *maxima levels per cycle*. A period-1
orbit whose waveform has several humps (e.g. the Rijke tube near its Hopf point: one large
and one small maximum per acoustic period) is reported as period-k of the hump count.
Disambiguate with the PSD in the panel text box: a genuine period-doubling shows a *new*
subharmonic at f1/2; a multi-humped period-1 orbit shows only the harmonic comb of f1
(see the Rijke verification scripts in the romda repo, beta=0.4).

## Measured λ1 tables in the model files

The dominant Lyapunov exponents measured here are stored in the `dynamodels`
physical-model files (`dynamodels/physical/{lorenz63,lorenz96,rijke}.py`): a per-model
`_LAM1_MEASURED` dict of chaotic parameter points, from which `__init__` sets the
instance's `t_lyap = 1/λ1` via `Model.t_lyap_from_table` (log-interpolated at the
constructed sweep-parameter value). Class attributes keep the historical constants, so
`scripts/mains` are unaffected; off-table configurations (limit cycles, tori, L96 with
`Nx != 10`) keep them too. Values set once at construction — parameters change by
re-instantiation. Extend the tables with `{param: λ1}` pairs from converged runs;
covered by `test_t_lyap_tables` in the dynamodels test suite.

## Torus dimension along a Ruelle–Takens–Newhouse route

The torus dimension T^k is read off three independent witnesses: the number of
**neutral Lyapunov exponents** (k zeros: limit cycle 1, 2-torus 2, 3-torus 3 — chaos has
a positive one instead), the **correlation dimension** D2 of the attractor (≈ k, fractal
for chaos) and of the plane-crossing **Poincaré section** (≈ k−1: dots → loop → band).
Measured on Lorenz-96 (Nx=10), the route is textbook RTN, with locking windows
interleaving near the breakdown and no stable 3-torus (as RTN predicts — T³ is
generically unstable and the attractor turns strange directly):

| F | λ signature (n₀ = neutrals) | D2 state / section | regime |
| --- | --- | --- | --- |
| 2–3.95 | one zero | 1.05 / — | limit cycle (T¹) |
| 4.0–4.06 | one zero | — | locked windows (periodic on the torus) |
| 4.2–4.4 | **two zeros** | 1.8 / 0.9 | quasiperiodic (T²) |
| 4.45 | λ1=+0.016, n₀=2 | 2.0 / 1.0 | first chaos, interleaved with… |
| 4.5–4.55 | two zeros | 2.1–2.2 / 1.0 | …re-locked/QP windows (Arnold tongues) |
| 4.6 | λ1=+0.039 (converged) | 2.15 / 1.02 | chaotic **wrinkled torus** — section still loop-like |
| 5 | λ1=+0.07 (plateau, λ1·T grows) | 2.9 / 1.1 | chaos on a thickened torus remnant |
| 8 | 3 positive exponents | 4.8 / 1.8 | developed chaos (D_KY ≈ 6.5) |

This is why F=4.6 and F=5 "look QP": the strange attractor inherits the torus geometry
(D2 barely above 2, section barely above a loop) while the dynamics on it are already
exponentially divergent — the spectrum, not the geometry, makes the call.

## Notes

- The spectrum's neutral (along-flow) exponent converges to 0 only as O(1/T), so
  `lyapunov_spectrum` extends its horizon until the halving test passes: on a limit cycle
  the reported λ1 lands within ~`atol`–2`atol` of zero (VdP: +0.005 at the auto-chosen
  T≈31 with the default `atol=2e-3`; +8e-4 with `atol=5e-4`, which runs to the `t_max`
  cap). Tighten `atol` (and raise `t_max`) for a cleaner zero; classification is
  unaffected (the λ1 trust floor `max(3σ, 10/T)` sits far above it).
- The same O(1/T) bias is why a *still-settling initial condition* used to inflate λ1 on
  limit cycles: both `lyapunov_spectrum` and `leading_lyapunov` now warm up with
  `_settled_state`, which doubles the discarded transient until a probe run shows no
  amplitude drift (critical near Hopf points, where the decay rate vanishes and no fixed
  transient multiple is safe).
- `covariant_lyapunov_vectors` is a Ginelli (2007) TODO stub.
- `lyapunov_spectrum` needs `time_derivative` (continuous models); `leading_lyapunov`
  works for any Model (including discrete maps such as ESN or KS).
- `characterize(..., spectrum=False)` still estimates the **dominant** exponent
  automatically: the independent `lyapunov` switch (default `'auto'`) runs
  `leading_lyapunov`, which is based on perturbation growth alone — no Jacobian or
  tangent operator required. `spectrum` only controls the full Benettin QR pass;
  the spectrum's λ1 is used by the classification only as a fallback when the
  growth fit is rejected.
- This package's
  `bifurcation_sweep` is generic but serial (never uses ensemble `m>1` — Py3.14
  forkserver pools can hang the IVP integrator).
- Tutorial notebook: [`tutorial_ntsa.ipynb`](tutorial_ntsa.ipynb) (full walkthrough on Lorenz63 + Lorenz96, executed).
- All randomness is seeded (`np.random.default_rng(seed)`).

## References

- Kantz & Schreiber (2004). *Nonlinear Time Series Analysis*, 2nd ed., Cambridge Univ. Press.
- Kennel, Brown & Abarbanel (1992). Determining embedding dimension for phase-space
  reconstruction using a geometrical construction. *Phys. Rev. A* 45, 3403.
- Benettin, Galgani, Giorgilli & Strelcyn (1980). Lyapunov characteristic exponents for smooth
  dynamical systems and for Hamiltonian systems. *Meccanica* 15, 9–30.
- Ginelli, Poggi, Turchi, Chaté, Livi & Politi (2007). Characterizing dynamics with covariant
  Lyapunov vectors. *Phys. Rev. Lett.* 99, 130601.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/
ruff check ntsa/ tests/
```

## Releasing (maintainer note)

Releases publish to PyPI via GitHub Actions trusted publishing on version tags:
configure a trusted publisher for `andreanovoa/ntsa` (workflow `release.yml`,
environment `pypi`) at pypi.org, then `git tag v0.1.0 && git push --tags`.

## License

MIT
