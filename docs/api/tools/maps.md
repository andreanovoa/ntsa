# `ntsa.tools.maps`

## At a glance

| Function | One-liner |
| --- | --- |
| `first_return_map(x, prominence=None)` | Successive local maxima `(x_max(i), x_max(i+1))` + peak indices. |
| `stationary_start(x, tol=0.05, block=10)` | First sample where the maxima envelope stops drifting (residual-transient guard). |
| `poincare_section(x, zeta, level=None, direction=1)` | Plane-crossing section of the delay embedding at $x(t+2\zeta) = \text{level}$: period-k $\to$ k dots, 2-torus $\to$ loop, chaos $\to$ fractal scatter. |
| `count_peak_clusters(xm, x_range, tol=0.02)` | Number of distinct peak levels (period-k detection). |

## Full reference

::: ntsa.tools.maps
