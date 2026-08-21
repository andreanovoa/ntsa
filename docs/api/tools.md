# `ntsa.tools`

The primitives — one module per job; every function is re-exported at the package
top level, so `from ntsa.tools import delay_embed` always works. The higher-level
analyses built on these live one level up: [`ntsa.classification`](classification.md)
and [`ntsa.bifurcation`](bifurcation.md).

## At a glance

| Module | What it does |
| --- | --- |
| [`runners`](tools/runners.md) | Fresh model instances (`respawn`) and long integrations past the transient (`run_long`). |
| [`embedding`](tools/embedding.md) | Delay embeddings and lag/dimension selection (AMI, FNN). |
| [`maps`](tools/maps.md) | Return maps, Poincaré sections, peak-based transient/period detection. |
| [`geometry`](tools/geometry.md) | Attractor geometry from pairwise distances: recurrence matrices, classical MDS, correlation dimension. |
| [`statistics`](tools/statistics.md) | Power spectral density, autocorrelation, summary moments. |
| [`lyapunov`](tools/lyapunov.md) | Lyapunov exponents: Benettin QR spectrum, leading exponent, Kaplan–Yorke, data-only Rosenstein. |

## Data-driven vs model-based

| Needs | Functions |
| --- | --- |
| Data only (raw arrays) | Everything in `embedding`, `maps`, `geometry`, and `statistics`, plus `rosenstein_lyapunov` — they take a scalar series `x` or a trajectory `Y`, so they work on measurements with no equations. This is what `ntsa.data.DataSeries` builds on, and [`ntsa.classification`](classification.md) is data-driven too. |
| Model equations | `respawn` and `run_long` (`runners`) and the rest of `lyapunov` — they instantiate and integrate a model implementing the protocol. So does [`ntsa.bifurcation`](bifurcation.md) (`plot_bifurcation` excepted, which just plots sweep output). |

`classify_regime` accepts optional `lam1`/`exponents` evidence, which can come from
the model-based `lyapunov` estimators or the data-only Rosenstein one.
