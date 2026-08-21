# `ntsa.tools.runners`

## At a glance

| Function | One-liner |
| --- | --- |
| `respawn(model, psi0=None, dt=None, **overrides)` | Fresh model instance with `alpha0` + `fixed_params` carried over; overrides win. |
| `run_long(model, t_run, t_transient=None, trim_transient=True)` | Integrate past transient; returns `(t, y, psi)` for member 0 (serial, m=1). Residual transient drift is detected and trimmed (printed note). |

## Full reference

::: ntsa.tools.runners
