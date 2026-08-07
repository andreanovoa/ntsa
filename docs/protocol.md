# Model protocol

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
