# Data-driven ntsa

[Model-based ntsa](protocol.md) asks for equations; this route only asks for
a signal — a scalar record $x(t)$ sampled at a fixed `dt`, no equations, no
`dynamodels.Model`, not even a state dimension. That is enough to embed the
attractor, estimate its correlation dimension, obtain a Lyapunov exponent
from the data itself, and classify the regime — everything
`ntsa.characterize` computes except the two things that genuinely need
equations: the full QR Lyapunov spectrum and bifurcation sweeps.

## Walkthrough

Take a single measured channel — here Lorenz63's $x(t)$, generated once and
then treated as a plain array with no memory of the equations that produced
it:

```python
from dynamodels.physical import Lorenz63
from ntsa.tools import run_long

model = Lorenz63(rho=28.)
t, y, psi = run_long(model, t_run=40.)
x = y[:, 0]   # pretend this is all you have: no equations, no psi
```

Wrap it in a `DataSeries` and run the equation-free pipeline:

```python
from ntsa.data import DataSeries

ds = DataSeries(x, dt=model.dt, label='Lorenz63 (measured)')
res = ds.analyze()
```

`res` has the same keys as one row of `ntsa.characterize`, filled in from the
signal alone:

| Key | Comes from | Meaning |
| --- | --- | --- |
| `zeta`, `dim` | `optimal_lag`, `false_nearest_neighbours` | the delay-embedding parameters |
| `D2` | `correlation_dimension` | Grassberger-Procaccia dimension — the data-only substitute for $D_{KY}$ |
| `lambda1`, `lambda1_std` | `lyapunov.rosenstein_lyapunov` | leading Lyapunov exponent, estimated from the embedded trajectory itself |
| `regime`, `evidence` | `classify_regime` | fixed point / limit cycle / quasiperiodic / chaotic, from the signal's own statistics |
| `gamma` | `classical_mds` | a 3-D embedding of the attractor, for plotting |

For this exact record, `res['lambda1']` comes out at $1.22 \pm 0.16$ against
the true $0.906$ — Rosenstein's estimate from a single 40-time-unit record,
not the QR-converged value model-based ntsa gets from the equations — and
`res['regime']` is `'chaotic'`, recovered with no access to $\sigma$,
$\rho$, $\beta$, or the equations, only the measured $x(t)$. `res['spectrum']`
stays `None`: a full spectrum needs equations, which this route does not
have.

Draw the same 8-panel row `ntsa.characterize` draws for a model:

```python
ds.characterize('figs/lorenz63_from_data.pdf')
```

![8-panel characterization of Lorenz63 from measurements only](assets/ntsa_data_l63.png)

*The same Lorenz63 case as [Model-based ntsa](protocol.md)'s baselines,
characterized from its measurements alone.*

## When only some of the state is measured

Pass the full record as `Y` (any simultaneous multivariate measurement, not
necessarily the physical state) to use it for the MDS panel instead of the
scalar delay embedding — e.g. several probes, or the physical state itself
if it happens to be available for that panel only:

```python
ds = DataSeries(x, dt=model.dt, Y=psi, label='Lorenz63 (measured, full state for MDS)')
```

## Reference

See the [`ntsa.data` API reference](api/data.md) for the full parameter list
and every method.
