# ntsa — nonlinear time-series analysis for dynamical-system models

Characterizes the dynamical regime of a model from a single long trajectory:
delay embedding (optimal lag + false nearest neighbours), Lyapunov exponents, regime
classification (fixed point / limit cycle period-k / frequency-locked / quasiperiodic / chaotic),
and a per-case diagnostic figure — one row of 8 panels
[time series | PSD | 3-D delay portrait | first-return map of local maxima | plane-crossing Poincaré section
| recurrence plot | 3-D MDS | Lyapunov spectrum] — Key Reference: Kantz & Schreiber,
*Nonlinear Time Series Analysis* (2004).

**Documentation: [andreanovoa.github.io/ntsa](https://andreanovoa.github.io/ntsa/)**
· Tutorial: [`tutorial_ntsa.ipynb`](tutorial_ntsa.ipynb)

### Example:

<img width="2877" height="1659" alt="image" src="https://github.com/user-attachments/assets/d366a1f1-6980-4bee-ae1c-a00c514c33b4" />

## Install

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
python -m ntsa.characterize          # 4-case demo -> figs/ntsa_defaults.pdf (+ .png)
```

Works with any model implementing the
[model protocol](https://andreanovoa.github.io/ntsa/protocol/) —
[`dynamodels`](https://github.com/andreanovoa/dynamodels) is the reference
implementation. Part of the same ecosystem as
[romda](https://github.com/andreanovoa/real-time-bias-aware-DA) (real-time
bias-aware data assimilation).

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/
ruff check ntsa/ tests/
```

Releases publish to PyPI via trusted publishing on version tags; docs deploy to
GitHub Pages on push to `main` (see `.github/workflows/`).

## License

MIT
