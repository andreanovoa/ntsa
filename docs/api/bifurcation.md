# `ntsa.bifurcation`

## At a glance

| Function | One-liner |
| --- | --- |
| `bifurcation_sweep(model, param, values, ...)` | Parameter sweep collecting observable extrema: one ensemble forecast — member `k` at `values[k]` — by default, or serial branch-following with `continuation=True`. |
| `plot_bifurcation(values, peaks, ...)` | Bifurcation diagram from `bifurcation_sweep` output. |

![Bifurcation diagram of Lorenz-96 (Nx=10): local extrema of x0, x5, x9 against F](../assets/bifurcation_lorenz96_F.png)

Produced by the module demo: `python -m ntsa.bifurcation --model lorenz96 --save-figs`.

## Full reference

::: ntsa.bifurcation
