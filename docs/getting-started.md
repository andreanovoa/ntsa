# Getting started

## Install

```bash
pip install ntsa
```

## Quickstart

### Model-based

```python
from dynamodels.physical import Lorenz63
from ntsa import characterize as chz

chz.characterize([Lorenz63()], pdf_name='figs/l63.pdf')
```

Works with any model implementing the [model protocol](protocol.md), not
just `dynamodels` — see [Model-based ntsa](protocol.md).

### Data-driven

From data alone — no model equations (see [Data-driven ntsa](data_driven.md)):

```python
import numpy as np
from ntsa.data import DataSeries

x = np.load('probe.npy')  # measured scalar record
DataSeries(x, dt=1e-3, label='probe').characterize(pdf_name='figs/probe.pdf')
```

```bash
python -m ntsa.characterize                     # 4-case demo -> figs/ntsa_defaults.pdf (+ .png)
python -m ntsa.characterize --model lorenz63 --param rho --values 20 28 100 350
python -m ntsa.bifurcation --model lorenz63 --param rho --nP 40   # bifurcation sweep demo
```

Figures are written relative to the working directory (`figs/`), so `cd` to where
you want them first.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/
ruff check ntsa/ tests/
```
