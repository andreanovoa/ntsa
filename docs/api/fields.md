# `ntsa.fields`

## At a glance

| Member | One-liner |
| --- | --- |
| `FieldSeries(u, dt, dx, ...)` | A gridded record `u(t, x)` in one to three space dimensions, with optional components. |
| `FieldSeries.analyze()` | Point statistics, energy spectrum, two-point correlations, probe autocorrelations and spectra, POD energy and data checks. |
| `FieldSeries.add_reference(quantity, x, y, label)` | Lay published reference data over a panel, grey and dashed. |
| `FieldSeries.report(out_dir)` | Figures, an animation, `stats.json` and one self-contained HTML page. |

```python
from ntsa.fields import FieldSeries

fs = FieldSeries(omega, dt=0.1, dx=2 * np.pi / 48, components=('omega',), label='Kolmogorov flow, Re = 34')
fs.report('explore/')          # explore/index.html, stats.json, fields.png, statistics.png, pod.png, animation.gif
```

The checks flag a record that is not time-resolved (or is shuffled), one shorter than 100 integral times, a
drifting transient, a box smaller than ten integral lengths, and an under-resolved energy spectrum.

## Full reference

::: ntsa.fields
